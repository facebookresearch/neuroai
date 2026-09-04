# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Masked autoencoder (MAE) encoder for time-series neuro data."""

import numpy as np
import torch
from torch import nn

from .base import BaseBrainModelConfig
from .sit import _get_1d_sincos_pos_embed_from_grid
from .transformer import TransformerEncoder


class MaeEncoder(BaseBrainModelConfig):
    """Encoder half of a masked autoencoder over time patches [1]_.

    Self-supervised pretraining is driven by
    :class:`~neuraltrain.mae_module.MaeModule`, which owns the reconstruction
    decoder so that a pretraining checkpoint reloads into this encoder alone.

    Parameters
    ----------
    dim :
        Token embedding dimension.
    patch_size :
        Number of consecutive time samples per token.
    transformer_config :
        Transformer applied to the token sequence.

    References
    ----------
    .. [1] He, Kaiming, et al. "Masked autoencoders are scalable vision
        learners." CVPR 2022.
    """

    dim: int = 256
    patch_size: int = 32
    # Rotary embeddings encode positions *relative to the sequence fed to the
    # transformer*, which MAE masking makes meaningless; positions come from the
    # absolute sin-cos embedding added before masking instead.
    transformer_config: TransformerEncoder = TransformerEncoder(
        heads=8, depth=4, rotary_pos_emb=False
    )

    def build(
        self, n_spatial_locations: int, n_outputs: int | None = None
    ) -> "MaeEncoderModel":
        """Build the MAE encoder.

        Parameters
        ----------
        n_spatial_locations :
            Number of input channels.
        n_outputs :
            Width of a mean-pooled linear output head.  ``None`` builds the
            encoder alone, which is what pretraining and downstream probing
            (where the probe owns the head) both use.
        """
        return MaeEncoderModel(self, n_spatial_locations, n_outputs)


class MaeEncoderModel(nn.Module):
    """``nn.Module`` implementation of :class:`MaeEncoder`."""

    def __init__(
        self, config: MaeEncoder, n_spatial_locations: int, n_outputs: int | None = None
    ) -> None:
        super().__init__()
        self.dim = config.dim
        self.patch_size = config.patch_size
        self.patch_dim = n_spatial_locations * config.patch_size
        self.patch_embed = nn.Linear(self.patch_dim, self.dim)
        self.encoder = config.transformer_config.build(dim=self.dim)
        self.head = None if n_outputs is None else nn.Linear(self.dim, n_outputs)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Cut ``(B, C, T)`` into flattened time patches ``(B, T // patch_size, C * patch_size)``.

        Trailing samples that do not fill a whole patch are dropped.  This is
        also the reconstruction target of the masked autoencoder.
        """
        batch_size, _, n_times = x.shape
        n_patches = n_times // self.patch_size
        if n_patches == 0:
            raise ValueError(
                f"input has {n_times} samples, which is less than "
                f"patch_size={self.patch_size}: no patch can be formed."
            )
        x = x[:, :, : n_patches * self.patch_size]
        x = x.reshape(batch_size, -1, n_patches, self.patch_size)
        return x.permute(0, 2, 1, 3).reshape(batch_size, n_patches, self.patch_dim)

    def positional_embedding(self, n_patches: int) -> torch.Tensor:
        """Fixed sin-cos embedding of shape ``(1, n_patches, dim)``.

        Computed per call rather than stored so that the encoder, and hence
        every checkpoint it produces, is independent of the window length.
        """
        embedding = _get_1d_sincos_pos_embed_from_grid(
            self.dim, np.arange(n_patches, dtype=np.float32)
        )
        return torch.from_numpy(embedding).unsqueeze(0)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Turn ``(B, C, T)`` into positioned tokens ``(B, T // patch_size, dim)``."""
        tokens = self.patch_embed(self.patchify(x))
        return tokens + self.positional_embedding(tokens.shape[1]).to(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, C, T)`` into tokens ``(B, T // patch_size, dim)``.

        With an output head, the tokens are mean-pooled and projected to
        ``(B, n_outputs)`` instead.
        """
        tokens = self.encoder(self.embed(x))
        if self.head is None:
            return tokens
        return self.head(tokens.mean(dim=1))
