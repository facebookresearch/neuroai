# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Conformer configuration wrapper for torchaudio.models.Conformer.
"""

import torch
from torch import nn

from .base import BaseModelConfig


class Conformer(BaseModelConfig):
    """
    Reference: Gulati et al., "Conformer: Convolution-augmented Transformer
    for Speech Recognition", Interspeech 2020.
    See https://arxiv.org/abs/2005.08100.

    The Conformer combines self-attention (Transformer) and local convolution
    blocks to capture both global and short-range temporal dependencies.

    Parameters
    ----------
    num_heads : int, optional
        Number of attention heads used in the multi-head self-attention layers.
        Each head learns a different temporal relationship.

    ffn_dim : int, optional
        Dimension of the feed-forward layer inside each Conformer block.
        Acts as the hidden expansion size for each token.

    num_layers : int, optional
        Number of Conformer layers to stack.
        Controls the model's depth and temporal abstraction capacity.

    depthwise_conv_kernel_size : int, optional
        Kernel size of the depthwise convolution in each convolution module.
        Controls the temporal receptive field of local processing.

    dropout : float, optional
        Dropout probability applied within the Conformer layers.
        Helps regularize the model and prevent overfitting.

    use_group_norm : bool, optional
        Whether to use GroupNorm instead of BatchNorm inside the convolutional
        modules. GroupNorm can be more stable for small batch sizes.

    convolution_first : bool, optional
        If True, applies the convolutional module before the self-attention
        module inside each block. In practice this may slightly alter inductive
        bias but rarely changes performance significantly.

    The values shown below are the default settings used in the original paper.
    """

    num_heads: int = 4
    ffn_dim: int = 144
    num_layers: int = 2
    depthwise_conv_kernel_size: int = 31
    dropout: float = 0.1
    use_group_norm: bool = False
    convolution_first: bool = False

    def build(self, dim: int) -> nn.Module:
        from torchaudio.models import Conformer

        # Subclass with forward method that infers `lengths` and returns only the output tensor
        class ConformerSimpleOutput(Conformer):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                lengths = torch.full(
                    (x.shape[0],),
                    fill_value=x.shape[1],
                    dtype=torch.long,
                    device=x.device,
                )
                out, _ = super().forward(input=x, lengths=lengths)
                return out

        kwargs = self.model_dump()
        del kwargs["name"]
        return ConformerSimpleOutput(input_dim=dim, **kwargs)
