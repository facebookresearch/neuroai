# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from torch import nn

from .base import BaseModelConfig
from .common import SubjectLayers


class Linear(BaseModelConfig):
    """Simple linear projection, with optional per-subject weights.

    Parameters
    ----------
    reduction : {"mean", "concat"}
        How to reduce the time dimension before the linear layer.
        ``"mean"`` averages over time; ``"concat"`` flattens channels and time.
    subject_layers_config : SubjectLayers or None
        If set, use a :class:`SubjectLayersModel` instead of a shared
        ``nn.Linear``.
    """

    reduction: tp.Literal["mean", "concat"] = "mean"
    subject_layers_config: SubjectLayers | None = None

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:
        return LinearModel(
            n_in_channels,
            n_outputs,
            reduction=self.reduction,
            subject_layers_config=self.subject_layers_config,
        )


class LinearModel(nn.Module):
    """``nn.Module`` implementation of :class:`Linear`."""

    def __init__(
        self,
        n_in_channels: int,
        n_outputs: int,
        reduction: str = "mean",
        subject_layers_config: SubjectLayers | None = None,
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.n_outputs = n_outputs
        self.linear: tp.Any
        if subject_layers_config is not None:
            self.linear = subject_layers_config.build(n_in_channels, n_outputs)
        else:
            self.linear = nn.Linear(n_in_channels, n_outputs)
        self.reduction = reduction

    def forward(self, x, subject_id=None):
        """Forward pass: reduce time, then project.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, C, T)`` or ``(B, C)``.
        subject_id : Tensor or None
            Per-example subject indices, shape ``(B,)``.
        """
        if len(x.shape) > 2:
            if self.reduction == "concat":
                x = x.view(x.size(0), -1)
            elif self.reduction == "mean":
                x = x.mean(dim=-1)
        if isinstance(self.linear, nn.Linear):
            x = self.linear(x)
        else:
            x = self.linear(x.unsqueeze(-1), subject_id).squeeze(-1)
        return x
