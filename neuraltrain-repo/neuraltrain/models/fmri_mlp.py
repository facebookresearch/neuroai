# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from functools import partial

import torch
from torch import nn
from torchvision.ops import MLP

from .base import BaseModelConfig
from .common import Mean, Mlp, SubjectLayers

logger = logging.getLogger(__name__)


class FmriMlp(BaseModelConfig):
    """Residual MLP for fMRI decoding, adapted from MindEye [1]_.

    Parameters
    ----------
    hidden : int
        Hidden dimension of the residual MLP blocks.
    n_blocks : int
        Number of residual MLP blocks.
    norm_type : {"ln", "bn"}
        Normalization layer type: ``"ln"`` for LayerNorm, ``"bn"`` for
        BatchNorm.
    act_first : bool
        If True, apply activation before normalization in each block.
    n_repetition_times : int
        Number of fMRI repetition times (TRs) expected in the input.
    time_agg : {"in_mean", "in_linear", "out_mean", "out_linear"}
        Temporal aggregation strategy.  ``"in_*"`` aggregates before the MLP;
        ``"out_*"`` aggregates after.
    use_tr_embeds : bool
        Add learned per-TR embeddings to the input features.
    tr_embed_dim : int
        Dimensionality of each TR embedding.
    use_tr_layer : bool
        Use a grouped 1-D convolution (one group per TR) as the first
        projection instead of a shared ``nn.Linear``.
    out_dim : int or None
        Explicit output dimension.  Overridden by *n_outputs* at build time.
    subject_layers_config : SubjectLayers or None
        If set, prepend a per-subject linear projection.
    subject_layers_dim : {"input", "hidden"}
        Dimension used for the subject-layer projection.
    output_head_config : Mlp, dict[str, Mlp], or None
        Optional output head(s) appended after the MLP backbone.

    References
    ----------
    .. [1] Scotti, Paul, et al. "Reconstructing the mind's eye: fMRI-to-image
       with contrastive learning and diffusion priors." NeurIPS 2024.
    """

    # From original MindEye model
    hidden: int = 4096
    n_blocks: int = 4
    norm_type: str = "ln"
    act_first: bool = False

    # Handling multiple TRs
    n_repetition_times: int = 1
    time_agg: tp.Literal["in_mean", "in_linear", "out_mean", "out_linear"] = "out_linear"

    # TR embeddings
    use_tr_embeds: bool = False
    tr_embed_dim: int = 16
    use_tr_layer: bool = False

    # Control output size explicitly
    out_dim: int | None = None

    # Subject-specific settings
    subject_layers_config: SubjectLayers | None = None
    subject_layers_dim: tp.Literal["input", "hidden"] = "hidden"

    # Output heads
    output_head_config: Mlp | dict[str, Mlp] | None = None
    # E.g., to replicate MindEye's projector config:
    # {
    #     "clip": {
    #         "hidden_sizes": [2048, 2048, 768],
    #         "norm_layer": "layer",
    #         "activation_layer": "gelu",
    #     },
    #     "mse": {"hidden_sizes": []}
    # }

    def build(self, n_in_channels: int, n_outputs: int | None) -> nn.Module:
        out_dim = self.out_dim if n_outputs is None else n_outputs
        if out_dim is None:
            raise ValueError("One of n_outputs or config.out_dim must be set.")

        return FmriMlpModel(
            in_dim=n_in_channels,
            out_dim=out_dim,
            config=self,
        )


class FmriMlpModel(nn.Module):
    """Residual MLP adapted from [1].

    See https://github.com/MedARC-AI/fMRI-reconstruction-NSD/blob/main/src/models.py#L171

    References
    ----------
    [1] Scotti, Paul, et al. "Reconstructing the mind's eye: fMRI-to-image with contrastive
        learning and diffusion priors." Advances in Neural Information Processing Systems 36
        (2024).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: FmriMlp | None = None,
    ):
        super().__init__()
        config = config if config is not None else FmriMlp()

        # Temporal aggregation
        self.in_time_agg: None | nn.Module = None
        self.out_time_agg: None | nn.Module = None
        self.n_repetition_times = config.n_repetition_times
        if config.time_agg == "in_mean":
            self.in_time_agg = Mean(dim=2, keepdim=True)
            self.n_repetition_times = 1
        elif config.time_agg == "in_linear":
            self.in_time_agg = nn.LazyLinear(1)
            self.n_repetition_times = 1
        elif config.time_agg == "out_mean":
            self.out_time_agg = Mean(dim=2)
        elif config.time_agg == "out_linear":
            self.out_time_agg = nn.LazyLinear(1)

        norm_func = (
            partial(nn.BatchNorm1d, num_features=config.hidden)
            if config.norm_type == "bn"
            else partial(nn.LayerNorm, normalized_shape=config.hidden)  # type: ignore
        )
        act_fn = partial(nn.ReLU, inplace=True) if config.norm_type == "bn" else nn.GELU
        act_and_norm = (act_fn, norm_func) if config.act_first else (norm_func, act_fn)

        # Subject-specific linear layer
        self.subject_layers = None
        if config.subject_layers_config is not None:
            dim = {"hidden": config.hidden, "input": in_dim}[config.subject_layers_dim]
            self.subject_layers = config.subject_layers_config.build(in_dim, dim)
            in_dim = dim

        # TR embeddings
        self.tr_embeddings = None
        if config.use_tr_embeds:
            self.tr_embeddings = nn.Embedding(
                self.n_repetition_times, config.tr_embed_dim
            )
            in_dim += config.tr_embed_dim

        self.lin0: nn.Conv1d | nn.Linear
        if config.use_tr_layer:
            self.lin0 = nn.Conv1d(
                in_channels=self.n_repetition_times,
                out_channels=self.n_repetition_times * config.hidden,
                kernel_size=in_dim,
                groups=self.n_repetition_times,
                bias=True,
            )
        else:
            self.lin0 = nn.Linear(in_dim, config.hidden)
        self.post_lin0 = nn.Sequential(
            *[item() for item in act_and_norm],
            nn.Dropout(0.5),  # type: ignore
        )

        self.mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden, config.hidden),
                    *[item() for item in act_and_norm],  # type: ignore
                    nn.Dropout(0.15),
                )
                for _ in range(config.n_blocks)
            ]
        )
        self.lin1 = nn.Linear(config.hidden, out_dim, bias=True)
        self.n_blocks = config.n_blocks

        # Separate output head(s)
        self.output_head: None | MLP | dict[str, MLP] = None
        if config.output_head_config is not None:
            if isinstance(config.output_head_config, Mlp):
                self.output_head = config.output_head_config.build(input_size=out_dim)
            elif isinstance(config.output_head_config, dict):
                self.output_head = nn.ModuleDict()
                for name, head_config in config.output_head_config.items():
                    self.output_head[name] = head_config.build(
                        input_size=out_dim,
                    )

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass through the residual MLP.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, F, T)`` or ``(B, ..., T)``.
        subject_ids : Tensor or None
            Per-example subject indices, shape ``(B,)``.
        """
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, F, T)
        if self.in_time_agg is not None:
            x = self.in_time_agg(x)  # (B, F, 1)

        B, _, T = x.shape
        if T != self.n_repetition_times:
            raise ValueError(f"Expected {self.n_repetition_times} TRs, got {T}")

        if self.subject_layers is not None:
            x = self.subject_layers(x, subject_ids)
        x = x.permute(0, 2, 1)  # (B, F, T) -> (B, T, F)

        if self.tr_embeddings is not None:
            embeds = self.tr_embeddings(torch.arange(T, device=x.device))
            embeds = torch.tile(embeds, dims=(B, 1, 1))
            x = torch.cat([x, embeds], dim=2)

        x = self.lin0(x).reshape(B, T, -1)  # (B, T, F) -> (B, T * F, 1) -> (B, T, F)
        x = self.post_lin0(x)

        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x

        x = x.permute(0, 2, 1)  # (B, T, F) -> (B, F, T)
        if self.out_time_agg is not None:
            x = self.out_time_agg(x)  # (B, F, 1)
        x = x.flatten(1)  # Ensure 2D
        x = self.lin1(x)

        # Apply output heads (e.g. for separate CLIP and MSE losses)
        if isinstance(self.output_head, MLP):
            out = self.output_head(x)
        elif isinstance(self.output_head, nn.ModuleDict):
            out = {name: head(x) for name, head in self.output_head.items()}
        else:
            out = x

        return out


class FmriLinear(BaseModelConfig):
    """Single linear layer for fMRI decoding with temporal aggregation.

    Parameters
    ----------
    out_dim : int or None
        Explicit output dimension.  Overridden by *n_outputs* at build time.
    time_agg : {"in_mean", "in_linear", "out_mean", "out_linear"}
        Temporal aggregation strategy.  ``"in_*"`` aggregates before the
        linear layer; ``"out_*"`` aggregates after.
    output_head_config : Mlp, dict[str, Mlp], or None
        Optional output head(s) appended after the linear layer.
    """

    out_dim: int | None = None
    time_agg: tp.Literal["in_mean", "in_linear", "out_mean", "out_linear"] = "out_linear"
    output_head_config: Mlp | dict[str, Mlp] | None = None

    def build(self, n_in_channels: int, n_outputs: int | None) -> nn.Module:
        out_dim = self.out_dim if n_outputs is None else n_outputs
        if out_dim is None:
            raise ValueError("One of n_outputs or config.out_dim must be set.")

        return FmriLinearModel(
            in_dim=n_in_channels,
            out_dim=out_dim,
            config=self,
        )


class FmriLinearModel(nn.Module):
    """``nn.Module`` implementation of :class:`FmriLinear`."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        config: FmriLinear | None = None,
    ):
        super().__init__()
        config = config if config is not None else FmriLinear()

        # Temporal aggregation
        self.in_time_agg: None | nn.Module = None
        self.out_time_agg: None | nn.Module = None
        if config.time_agg == "in_mean":
            self.in_time_agg = Mean(dim=2, keepdim=True)
        elif config.time_agg == "in_linear":
            self.in_time_agg = nn.LazyLinear(1)
        elif config.time_agg == "out_mean":
            self.out_time_agg = Mean(dim=2)
        elif config.time_agg == "out_linear":
            self.out_time_agg = nn.LazyLinear(1)

        self.linear = nn.Linear(in_dim, out_dim, bias=True)

        # Separate output head(s)
        self.output_head: None | MLP | dict[str, MLP] = None
        if config.output_head_config is not None:
            if isinstance(config.output_head_config, Mlp):
                self.output_head = config.output_head_config.build(input_size=out_dim)
            elif isinstance(config.output_head_config, dict):
                self.output_head = nn.ModuleDict()
                for name, head_config in config.output_head_config.items():
                    self.output_head[name] = head_config.build(
                        input_size=out_dim,
                    )

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass through the linear model.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, F, T)``.
        subject_ids : Tensor or None
            Per-example subject indices (unused, kept for API consistency).
        """
        x = x.reshape(x.shape[0], -1, x.shape[-1])  # (B, F, T)
        if self.in_time_agg is not None:
            x = self.in_time_agg(x)  # (B, F, 1)
        x = x.permute(0, 2, 1)  # (B, F, T) -> (B, T, F)
        x = self.linear(x)
        x = x.permute(0, 2, 1)  # (B, T, F) -> (B, F, T)
        if self.out_time_agg is not None:
            x = self.out_time_agg(x)  # (B, F, 1)
        x = x.flatten(1)  # Ensure 2D

        # Apply output heads (e.g. for separate CLIP and MSE losses)
        if isinstance(self.output_head, MLP):
            out = self.output_head(x)
        elif isinstance(self.output_head, nn.ModuleDict):
            out = {name: head(x) for name, head in self.output_head.items()}
        else:
            out = x

        return out
