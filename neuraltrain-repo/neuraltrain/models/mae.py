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
from .common import ChannelMerger, FourierEmb
from .sit import _get_1d_sincos_pos_embed_from_grid
from .transformer import TransformerEncoder


class MaeEncoder(BaseBrainModelConfig):
    """Encoder trained by masked prediction over time patches [1]_.

    The encoder is the whole model: :class:`~neuraltrain.mae_module.MaeModule`
    hides patches and reconstructs them with a single linear layer, so nothing
    outside this class holds weights worth keeping.  The asymmetric
    encoder/decoder of the original MAE is left as an exercise.

    A :class:`~neuraltrain.models.common.ChannelMerger` first maps whatever
    montage a recording has onto ``n_virtual_channels``, using each channel's
    3D position, so one encoder can pretrain on datasets that share no montage
    and then score on a task with its own.  Channels a recording does not have
    arrive with invalid positions and are masked out of the merge.

    Parameters
    ----------
    dim :
        Token embedding dimension.
    patch_size :
        Number of consecutive time samples per token.
    merger_config :
        Spatial attention mapping the input montage onto a fixed number of
        virtual channels.  ``per_subject`` is not supported, since it would tie
        the encoder to the subjects it pretrained on.
    transformer_config :
        Transformer applied to the token sequence.

    References
    ----------
    .. [1] He, Kaiming, et al. "Masked autoencoders are scalable vision
        learners." CVPR 2022.
    """

    dim: int = 256
    patch_size: int = 32
    merger_config: ChannelMerger = ChannelMerger(
        n_virtual_channels=64,
        fourier_emb_config=FourierEmb(n_freqs=None, total_dim=2000, n_dims=3),
        dropout=0.2,
    )
    # Rotary embeddings encode positions *relative to the sequence fed to the
    # transformer*; the absolute sin-cos embedding below is added after masking
    # instead, so a hidden token still knows where it sits.
    transformer_config: TransformerEncoder = TransformerEncoder(
        heads=8, depth=4, rotary_pos_emb=False
    )

    def build(self, n_outputs: int | None = None) -> "MaeEncoderModel":
        """Build the encoder.

        Parameters
        ----------
        n_outputs :
            Width of a mean-pooled linear output head.  ``None`` builds the
            encoder alone, which is what pretraining and downstream probing
            (where the probe owns the head) both use.

        Notes
        -----
        Takes no channel count: the merger fixes the width of everything
        downstream of it, which is what lets one checkpoint span montages.
        """
        if self.merger_config.per_subject:
            raise ValueError(
                "merger_config.per_subject=True gives each pretraining subject "
                "its own merge weights, which a downstream task cannot index "
                "into. Use per_subject=False."
            )
        return MaeEncoderModel(self, n_outputs)


class MaeEncoderModel(nn.Module):
    """``nn.Module`` implementation of :class:`MaeEncoder`."""

    def __init__(self, config: MaeEncoder, n_outputs: int | None = None) -> None:
        super().__init__()
        self.dim = config.dim
        self.patch_size = config.patch_size
        self.merger = config.merger_config.build()
        self.patch_dim = config.merger_config.n_virtual_channels * config.patch_size
        self.patch_embed = nn.Linear(self.patch_dim, self.dim)
        self.encoder = config.transformer_config.build(dim=self.dim)
        self.head = None if n_outputs is None else nn.Linear(self.dim, n_outputs)

    def merge(self, x: torch.Tensor, channel_positions: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, T)`` on any montage onto ``(B, n_virtual_channels, T)``.

        The merge is a softmax attention over channels, so its output is a
        convex combination of real ones: it cannot collapse to a constant, which
        is what makes it safe to reconstruct.
        """
        subject_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.merger(x, subject_ids, channel_positions)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Cut ``(B, C, T)`` into flattened time patches ``(B, T // patch_size, C * patch_size)``.

        Trailing samples that do not fill a whole patch are dropped.  Applied to
        the merged signal, this is also the reconstruction target.
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

    def patch_tokens(self, merged: torch.Tensor) -> torch.Tensor:
        """Embed merged input as unpositioned tokens ``(B, T // patch_size, dim)``.

        Kept separate from :meth:`add_positions` so that pretraining can swap in
        its mask token before positions are added, leaving a hidden token its
        position and taking only its content.
        """
        return self.patch_embed(self.patchify(merged))

    def positional_embedding(self, n_patches: int) -> torch.Tensor:
        """Fixed sin-cos embedding of shape ``(1, n_patches, dim)``.

        Computed per call rather than stored so that the encoder, and hence
        every checkpoint it produces, is independent of the window length.
        """
        embedding = _get_1d_sincos_pos_embed_from_grid(
            self.dim, np.arange(n_patches, dtype=np.float32)
        )
        return torch.from_numpy(embedding).unsqueeze(0)

    def add_positions(self, tokens: torch.Tensor) -> torch.Tensor:
        """Add the sin-cos embedding to a token sequence."""
        return tokens + self.positional_embedding(tokens.shape[1]).to(tokens)

    def forward(self, x: torch.Tensor, channel_positions: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, C, T)`` into tokens ``(B, T // patch_size, dim)``.

        With an output head, the tokens are mean-pooled and projected to
        ``(B, n_outputs)`` instead.
        """
        merged = self.merge(x, channel_positions)
        tokens = self.encoder(self.add_positions(self.patch_tokens(merged)))
        if self.head is None:
            return tokens
        return self.head(tokens.mean(dim=1))
