# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Masked-prediction encoder for time-series neuro data."""

import numpy as np
import torch
from torch import nn

from .base import BaseBrainModelConfig
from .common import INVALID_POS_VALUE, FourierEmb
from .sit import _get_1d_sincos_pos_embed_from_grid
from .transformer import TransformerEncoder


class MaeEncoder(BaseBrainModelConfig):
    """Encoder trained by masked prediction over channel-time patches.

    The objective follows the masked autoencoder of He et al. [1]_, developed
    for images; patching and channel handling are specific to neural
    time series.

    One token is one channel over one time patch, carrying a sin-cos embedding
    of its time patch plus a Fourier embedding of its channel's 3D position.
    Channels are therefore identified by where they sit on the head rather than
    by their index, which lets one encoder pretrain on datasets that share no
    montage and then score on a task with its own.  Channels a recording does
    not have arrive with invalid positions and are dropped from the attention.
    The sequence is ``n_channels * n_patches`` long, so channel count costs
    attention rather than weights.

    Parameters
    ----------
    dim :
        Token embedding dimension.
    patch_size :
        Number of consecutive time samples per token.
    channel_emb_config :
        Fourier embedding of channel positions.  ``n_dims`` must match the
        ``n_spatial_dims`` of the ``ChannelPositions`` extractor feeding it.
    transformer_config :
        Transformer applied to the token sequence.

    References
    ----------
    .. [1] He, Kaiming, et al. "Masked autoencoders are scalable vision
        learners." CVPR 2022.
    """

    dim: int = 256
    patch_size: int = 32
    channel_emb_config: FourierEmb = FourierEmb(n_freqs=5, n_dims=3)
    # sequence interleaves channel and time: positions absolute, attention flash
    transformer_config: TransformerEncoder = TransformerEncoder(
        heads=8, depth=4, rotary_pos_emb=False, attn_flash=True
    )

    def build(self, n_outputs: int | None = None) -> "MaeEncoderModel":
        """Build the encoder.

        Parameters
        ----------
        n_outputs :
            Width of a linear output head over the pooled tokens.  ``None``
            builds the encoder alone, which is what pretraining and downstream
            probing (where the probe owns the head) both use.
        """
        return MaeEncoderModel(self, n_outputs)


class MaeEncoderModel(nn.Module):
    """``nn.Module`` implementation of :class:`MaeEncoder`."""

    def __init__(self, config: MaeEncoder, n_outputs: int | None = None) -> None:
        super().__init__()
        self.dim = config.dim
        self.patch_size = config.patch_size
        self.patch_embed = nn.Linear(config.patch_size, self.dim)
        self.channel_emb = config.channel_emb_config.build()
        self.channel_embed = nn.Linear(self.channel_emb.total_dim, self.dim)
        self.encoder = config.transformer_config.build(dim=self.dim)
        self.head = None if n_outputs is None else nn.Linear(self.dim, n_outputs)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Cut ``(B, C, T)`` into per-channel time patches.

        Returns ``(B, C, T // patch_size, patch_size)``; trailing samples that
        do not fill a whole patch are dropped.
        """
        batch_size, n_channels, n_times = x.shape
        n_patches = n_times // self.patch_size
        if n_patches == 0:
            raise ValueError(
                f"input has {n_times} samples, which is less than "
                f"patch_size={self.patch_size}: no patch can be formed."
            )
        x = x[:, :, : n_patches * self.patch_size]
        return x.reshape(batch_size, n_channels, n_patches, self.patch_size)

    @staticmethod
    def valid_tokens(channel_positions: torch.Tensor, n_patches: int) -> torch.Tensor:
        """Flag the tokens worth attending to, as ``(B, C * n_patches)``.

        Channels a recording lacks are zero-padded by the extractor and marked
        with :data:`INVALID_POS_VALUE` positions; all their tokens are dropped.
        """
        valid = (channel_positions != INVALID_POS_VALUE).any(dim=-1)
        return valid[:, :, None].expand(-1, -1, n_patches).flatten(1)

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Embed patch contents as ``(B, C * n_patches, dim)``, without positions.

        Separate from :meth:`add_positions` so that pretraining can substitute
        its mask token before positions are added.
        """
        return self.patch_embed(self.patchify(x)).flatten(1, 2)

    def add_positions(
        self, tokens: torch.Tensor, channel_positions: torch.Tensor
    ) -> torch.Tensor:
        """Add time and channel embeddings to a ``(B, C * n_patches, dim)`` sequence."""
        n_channels = channel_positions.shape[1]
        batch_size, n_tokens, _ = tokens.shape
        n_patches = n_tokens // n_channels

        time = self._time_embedding(n_patches).to(tokens)
        channel = self.channel_embed(self.channel_emb(channel_positions))
        positions = time[None, None] + channel[:, :, None]
        return tokens + positions.reshape(batch_size, n_tokens, self.dim)

    def _time_embedding(self, n_patches: int) -> torch.Tensor:
        """Fixed sin-cos embedding of shape ``(n_patches, dim)``.

        Computed per call rather than stored, which keeps the encoder and its
        checkpoints independent of the window length.
        """
        embedding = _get_1d_sincos_pos_embed_from_grid(
            self.dim, np.arange(n_patches, dtype=np.float32)
        )
        return torch.from_numpy(embedding)

    def forward(self, x: torch.Tensor, channel_positions: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, C, T)`` into tokens ``(B, C * T // patch_size, dim)``.

        Tokens of absent channels are zeroed on the way out, so that a caller
        pooling over the sequence -- as ``neuralbench``'s probe does -- averages
        in nothing rather than noise.  With an output head, tokens are pooled
        over the present channels only and projected to ``(B, n_outputs)``.
        """
        n_patches = self.patchify(x).shape[2]
        valid = self.valid_tokens(channel_positions, n_patches)
        tokens = self.add_positions(self.patch_tokens(x), channel_positions)
        encoded = self.encoder(tokens, mask=valid) * valid[..., None]
        if self.head is None:
            return encoded
        pooled = encoded.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1)
        return self.head(pooled)
