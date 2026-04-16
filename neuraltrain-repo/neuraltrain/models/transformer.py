# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrappers around Transformer models from x_transformers package.
"""

import logging

import torch.nn as nn

from .base import BaseModelConfig

logger = logging.getLogger(__name__)


class TransformerEncoder(BaseModelConfig):
    """Transformer encoder/decoder built on top of ``x_transformers``.

    Parameters
    ----------
    heads : int
        Number of attention heads.
    depth : int
        Number of Transformer layers.
    cross_attend : bool
        Enable cross-attention (decoder mode).
    causal : bool
        If True, build a causal ``Decoder`` instead of an ``Encoder``.
    attn_flash : bool
        Use Flash Attention.  Not compatible with ALiBi.
    attn_dropout : float
        Dropout probability inside the attention layers.
    ff_mult : int
        Feed-forward expansion factor (``ff_dim = dim * ff_mult``).
    ff_dropout : float
        Dropout probability in the feed-forward layers.
    use_scalenorm : bool
        Use ScaleNorm instead of LayerNorm.
    use_rmsnorm : bool
        Use RMSNorm instead of LayerNorm.
    rel_pos_bias : bool
        Use relative positional bias.
    alibi_pos_bias : bool
        Use ALiBi positional bias.
    rotary_pos_emb : bool
        Use rotary positional embeddings.
    rotary_xpos : bool
        Use xPos extension for rotary embeddings.
    residual_attn : bool
        Add residual connections around the attention output.
    scale_residual : bool
        Scale residual connections.
    layer_dropout : float
        Probability of dropping an entire Transformer layer during training.
    """

    heads: int = 8
    depth: int = 12

    # Attention blocks
    cross_attend: bool = False
    causal: bool = False
    # Use Flash Attention; not compatible with ALiBi and probably other extractors
    attn_flash: bool = False
    attn_dropout: float = 0.1

    # Feedforward blocks
    ff_mult: int = 4  # Feedforward expansion factor
    ff_dropout: float = 0.0

    # Normalization
    use_scalenorm: bool = True
    use_rmsnorm: bool = False

    # Positional embedding
    rel_pos_bias: bool = False
    alibi_pos_bias: bool = False
    rotary_pos_emb: bool = True
    rotary_xpos: bool = False

    # Others
    residual_attn: bool = False
    scale_residual: bool = True
    layer_dropout: float = 0.0

    def build(self, dim: int) -> nn.Module:
        from x_transformers import Decoder, Encoder  # type: ignore

        if dim % self.heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by the number of heads ({self.heads})"
            )
        if self.rotary_pos_emb and dim // self.heads < 32:
            raise ValueError(
                f"dim_head ({dim // self.heads}) < 32: x-transformers clamps the rotary "
                f"embedding dimension to min 32, causing a shape mismatch. "
                f"Increase dim or reduce heads so that dim // heads >= 32, "
                f"or disable rotary_pos_emb."
            )
        kwargs = self.model_dump()
        kwargs["attn_dim_head"] = dim // self.heads
        del kwargs["name"]
        del kwargs["causal"]
        if self.causal:
            return Decoder(dim=dim, **kwargs)
        else:
            return Encoder(dim=dim, **kwargs)
