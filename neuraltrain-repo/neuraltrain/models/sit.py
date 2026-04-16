# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SiT (Surface vision Transformer) model."""

import logging
import typing as tp
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.ops import MLP

from .base import BaseModelConfig
from .common import Mean, Mlp
from .transformer import TransformerEncoder

logger = logging.getLogger(__name__)


# from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Density(Enum):
    FS2 = 2
    FS3 = 3
    FS4 = 4
    FS5 = 5
    FS6 = 6


# (n_patches_per_hemi, n_vertices_per_patch) for each (mesh, grid) pair
DENSITY_TO_PATCH_VERTEX_SIZE = {
    Density.FS6: {
        Density.FS2: (320, 153),
        Density.FS3: (1280, 45),
        Density.FS4: (5120, 15),
        Density.FS5: (20480, 6),
    },
    Density.FS5: {
        Density.FS2: (320, 45),
        Density.FS3: (1280, 15),
        Density.FS4: (5120, 6),
    },
}


class PositionalEmbeddingConfig(BaseModelConfig):
    """Positional embedding configuration.

    Parameters
    ----------
    pe_type :
        Type of positional encoding (``'trainable'``, ``'sin-cos'``,
        or ``None`` to disable).
    exclude_cls_token :
        Whether to exclude the class token from positional embedding.
    """

    pe_type: tp.Literal["trainable", "sin-cos"] | None = "sin-cos"
    exclude_cls_token: bool = True

    def build(
        self,
        n_patches: int,
        embed_dim: int,
        has_cls_token: bool = False,
    ) -> "PositionalEmbedding":
        return PositionalEmbedding(
            self,
            n_patches,
            embed_dim,
            has_cls_token,
        )


class PositionalEmbedding(nn.Module):
    """Positional embedding supporting learnable and fixed sinusoidal encodings.

    Parameters
    ----------
    config :
        Positional embedding configuration.
    n_patches :
        Number of patches (excluding class token).
    embed_dim :
        Dimension of embeddings.
    """

    def __init__(
        self,
        config: PositionalEmbeddingConfig,
        n_patches: int,
        embed_dim: int,
        has_cls_token: bool = False,
    ):
        super().__init__()
        self.config = config
        self.n_patches = n_patches
        self.embed_dim = embed_dim

        self.pe_len = n_patches
        if not config.exclude_cls_token and has_cls_token:
            self.pe_len += 1

        if config.pe_type is None:
            self.pos_embedding = None
        elif config.pe_type == "trainable":
            self.pos_embedding = nn.Parameter(
                torch.randn(1, self.pe_len, embed_dim) * 0.02,
                requires_grad=True,
            )
        elif config.pe_type == "sin-cos":
            self.pos_embedding = nn.Parameter(
                torch.zeros(1, self.pe_len, embed_dim),
                requires_grad=False,
            )
            pos_embed = _get_1d_sincos_pos_embed_from_grid(
                embed_dim, np.arange(self.pe_len, dtype=np.float32)
            )
            self.pos_embedding.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0)
            )
        else:
            raise ValueError(f"Unknown pe_type: {config.pe_type}")

    def add(self, x: torch.Tensor, has_cls_token: bool = False) -> torch.Tensor:
        """Add positional embeddings to input of shape ``(B, N, D)``."""
        if self.pos_embedding is None:
            return x
        if self.config.exclude_cls_token and has_cls_token:
            x[:, 1:, :] = x[:, 1:, :] + self.pos_embedding[:, :, :]
        else:
            x = x + self.pos_embedding[:, :, :]
        return x


class TokenizerConfig(BaseModelConfig):
    """Surface-based tokenization configuration.

    Parameters
    ----------
    mesh_resolution :
        Resolution of input fsaverage mesh (e.g. ``FS6`` = fsaverage6).
    grid_resolution :
        Resolution of patch sampling grid (e.g. ``FS2`` = 320 patches/hemi).
    hemi :
        Which hemisphere(s) to tokenize (``'both'``, ``'left'``, or ``'right'``).
    triangle_indices_csv_path :
        Path to directory containing CSV files that map patches to vertices.
        These files are generated by the fMRI preprocessing pipeline and
        stored on internal shared storage.
    """

    mesh_resolution: Density = Density.FS6
    grid_resolution: Density = Density.FS2
    hemi: tp.Literal["both", "left", "right"] = "both"
    triangle_indices_csv_path: str = ""

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["triangle_indices_csv_path"]

    def build(self, n_frames: int) -> "Tokenizer":
        return Tokenizer(self, n_frames)


class Tokenizer(nn.Module):
    """Tokenize full-resolution fsaverage surface data into patches.

    Each patch contains vertices from the high-resolution mesh that fall
    within a triangle of the coarser sampling grid.

    Parameters
    ----------
    config :
        Tokenizer configuration.
    n_frames :
        Number of temporal frames.
    """

    patch_indices: torch.Tensor

    def __init__(self, config: TokenizerConfig, n_frames: int) -> None:
        super().__init__()
        self.config = config
        self.n_frames = n_frames

        if config.mesh_resolution not in DENSITY_TO_PATCH_VERTEX_SIZE:
            raise ValueError(
                f"mesh_resolution {config.mesh_resolution} is not supported. "
                f"Supported values: {list(DENSITY_TO_PATCH_VERTEX_SIZE.keys())}"
            )
        if (
            config.grid_resolution
            not in DENSITY_TO_PATCH_VERTEX_SIZE[config.mesh_resolution]
        ):
            valid_grids = list(
                DENSITY_TO_PATCH_VERTEX_SIZE[config.mesh_resolution].keys()
            )
            raise ValueError(
                f"grid_resolution {config.grid_resolution} is not compatible with "
                f"mesh_resolution {config.mesh_resolution}. "
                f"Valid: {valid_grids}"
            )

        self.n_patches_per_hemi, self.n_vertices = DENSITY_TO_PATCH_VERTEX_SIZE[
            config.mesh_resolution
        ][config.grid_resolution]

        if config.hemi == "both":
            self.n_patches_sampling_grid = self.n_patches_per_hemi * 2
        else:
            self.n_patches_sampling_grid = self.n_patches_per_hemi

        csv_path = (
            Path(config.triangle_indices_csv_path)
            / f"fs{config.mesh_resolution.value}_vertices_from_fs{config.grid_resolution.value}_triangles.csv"
        )
        df = pd.read_csv(csv_path)
        indices = np.stack(
            [df[str(j)].to_numpy() for j in range(self.n_patches_per_hemi)]
        )
        self.register_buffer("patch_indices", torch.from_numpy(indices).long())

    def _gather_hemi(self, x: torch.Tensor) -> torch.Tensor:
        """Gather patches from one hemisphere using precomputed indices.

        Parameters
        ----------
        x :
            Hemisphere data of shape ``(B, T, n_vertices_per_hemi)``.

        Returns
        -------
        torch.Tensor
            Patches of shape ``(B, T, n_patches_per_hemi, n_vertices_per_patch)``.
        """
        idx = self.patch_indices.view(1, 1, self.n_patches_per_hemi, -1).expand(
            x.shape[0], x.shape[1], -1, -1
        )
        return torch.gather(
            x.unsqueeze(2).expand(-1, -1, self.n_patches_per_hemi, -1), dim=3, index=idx
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize surface data into patches.

        Parameters
        ----------
        x :
            Input of shape ``(B, n_vertices_total, T)``.

        Returns
        -------
        torch.Tensor
            Patches of shape ``(B, T, n_patches, n_vertices_per_patch)``.
        """
        x = x.permute(0, 2, 1)  # (B, T, V_total)
        n_verts_hemi = x.shape[2] // 2

        if self.config.hemi == "left":
            return self._gather_hemi(x[:, :, :n_verts_hemi])
        elif self.config.hemi == "right":
            return self._gather_hemi(x[:, :, n_verts_hemi:])
        elif self.config.hemi == "both":
            left = self._gather_hemi(x[:, :, :n_verts_hemi])
            right = self._gather_hemi(x[:, :, n_verts_hemi:])
            return torch.cat([left, right], dim=2)
        else:
            raise ValueError(f"hemi value {self.config.hemi} not recognized")


class SiTConfig(BaseModelConfig):
    """SiT (Surface vision Transformer) encoder configuration.

    Parameters
    ----------
    dim :
        Dimension of token embeddings.
    transformer_config :
        Configuration for the transformer encoder.
    tokenizer_config :
        Configuration for surface tokenization.
    pos_emb_config :
        Configuration for positional embeddings.
    pool :
        Pooling strategy (``'mean'``, ``'cls'``, or ``None``).
    emb_dropout :
        Dropout rate for embeddings.
    use_cls_token :
        Whether to prepend a learnable class token.
    in_time_agg :
        Temporal aggregation strategy before patch embedding.

        Space-only (sequence length = n_patches):
            ``in_channels`` — concatenate vertices and time into each token.
            ``in_mean`` — average across time dimension.

        Time-only (sequence length = n_frames):
            ``in_patch_mean`` — mean over vertices, one token per time point.

        Space-time (sequence length = n_patches × n_frames):
            ``in_concat`` — one token per (patch, frame) pair.

    patch_embed_activation :
        Optional activation after patch embedding projection.
    output_head_config :
        Configuration for output head(s).
    """

    dim: int = 192
    transformer_config: TransformerEncoder = TransformerEncoder()
    tokenizer_config: TokenizerConfig = TokenizerConfig()
    pos_emb_config: PositionalEmbeddingConfig = PositionalEmbeddingConfig()
    pool: tp.Literal["mean", "cls"] | None = None
    emb_dropout: float = 0.0
    use_cls_token: bool = False
    in_time_agg: tp.Literal[
        "in_channels",
        "in_mean",
        "in_concat",
        "in_patch_mean",
    ] = "in_channels"
    patch_embed_activation: tp.Literal["relu", "gelu", "elu", "tanh", "silu"] | None = (
        None
    )
    output_head_config: Mlp | dict[str, Mlp] | None = None

    def build(self, n_frames: int = 3) -> nn.Module:
        return SiT(self, n_frames)


class SiT(nn.Module):
    """SiT (Surface vision Transformer) encoder [1].

    Encodes surface-based fMRI data using patch-based tokenization and
    a transformer architecture.

    References
    ----------
    [1] Dahan, Simon, et al. "Surface Vision Transformers: Attention-Based
        Modelling applied to Cortical Analysis." MIDL 2022.
        See https://arxiv.org/abs/2203.16414
    """

    def __init__(self, config: SiTConfig, n_frames: int = 3):
        from einops.layers.torch import Rearrange

        super().__init__()
        self.config = config

        self.tokenizer = config.tokenizer_config.build(n_frames=n_frames)
        self.mesh_resolution = self.tokenizer.config.mesh_resolution
        self.n_frames = self.tokenizer.n_frames
        self.n_patches_sampling_grid = self.tokenizer.n_patches_sampling_grid
        self.n_vertices = self.tokenizer.n_vertices

        # Sequence length after temporal aggregation
        if config.in_time_agg == "in_concat":
            self.n_patches = self.n_patches_sampling_grid * n_frames
        elif config.in_time_agg == "in_patch_mean":
            self.n_patches = n_frames
        else:  # in_channels, in_mean
            self.n_patches = self.n_patches_sampling_grid

        self.transformer = config.transformer_config.build(dim=config.dim)
        self.pool = config.pool

        # Temporal aggregation + patch embedding
        layers: list[nn.Module] = []
        patch_dim: int
        if config.in_time_agg == "in_channels":
            # (B, T, N, V) -> (B, N, V*T)
            patch_dim = n_frames * self.n_vertices
            layers = [Rearrange("b t n v -> b n (v t)")]
        elif config.in_time_agg == "in_mean":
            # (B, T, N, V) -> (B, N, V)
            patch_dim = self.n_vertices
            layers = [Rearrange("b t n v -> b n v t"), Mean(dim=-1, keepdim=False)]
        elif config.in_time_agg == "in_concat":
            # (B, T, N, V) -> (B, N*T, V)
            patch_dim = self.n_vertices
            layers = [Rearrange("b t n v -> b (n t) v")]
        elif config.in_time_agg == "in_patch_mean":
            # (B, T, N, V) -> (B, T, N) via mean over V
            patch_dim = self.n_patches_sampling_grid
            layers = [Mean(dim=-1, keepdim=False)]

        layers.append(nn.Linear(patch_dim, config.dim))

        if config.patch_embed_activation is not None:
            activation_cls = {
                "relu": nn.ReLU,
                "gelu": nn.GELU,
                "elu": nn.ELU,
                "tanh": nn.Tanh,
                "silu": nn.SiLU,
            }[config.patch_embed_activation]
            layers.append(activation_cls())

        self.to_patch_embedding = nn.Sequential(*layers)
        self.dropout = nn.Dropout(config.emb_dropout)
        self.num_prefix_tokens = int(config.use_cls_token)

        self.pos_embedding = config.pos_emb_config.build(
            n_patches=self.n_patches,
            embed_dim=config.dim,
            has_cls_token=config.use_cls_token,
        )

        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, config.dim)) if config.use_cls_token else None
        )

        logger.info(
            "SiT: n_patches=%d  n_patches_grid=%d  cls=%s  agg=%s",
            self.n_patches,
            self.n_patches_sampling_grid,
            config.use_cls_token,
            config.in_time_agg,
        )

        # Output head(s)
        self.output_head: None | MLP | dict[str, Mlp] = None
        if config.output_head_config is not None:
            if isinstance(config.output_head_config, Mlp):
                self.output_head = config.output_head_config.build(input_size=config.dim)
            elif isinstance(config.output_head_config, dict):
                self.output_head = nn.ModuleDict()
                for name, head_config in config.output_head_config.items():
                    self.output_head[name] = head_config.build(input_size=config.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        # (B, V_total, T) -> (B, T, N, V_patch) -> (B, N_seq, D)
        x = self.tokenizer(x)
        x = self.to_patch_embedding(x)

        if self.config.use_cls_token and self.cls_token is not None:
            from einops import repeat

            b = x.shape[0]
            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_embedding.add(x, has_cls_token=self.config.use_cls_token)
        x = self.dropout(x)
        x = self.transformer(x)

        if self.pool == "cls":
            if not self.config.use_cls_token:
                raise ValueError("Cannot use 'cls' pooling without cls_token.")
            x = x[:, 0, :]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        else:
            return x

        if isinstance(self.output_head, MLP):
            return self.output_head(x)
        elif isinstance(self.output_head, nn.ModuleDict):
            return {name: head(x) for name, head in self.output_head.items()}
        return x
