# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import warnings

import pydantic
import torch
from exca import helpers

from neuralset.base import Frequency


class PaddingStrategy(helpers.DiscriminatedModel, discriminator_key="name"):
    """Base class for padding strategies.

    Subclasses implement ``__call__`` which receives a list of tensors
    (one per segment) and returns a list of tensors padded to uniform
    size.
    """

    def __call__(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        raise NotImplementedError


def _pad_dim(tensor: torch.Tensor, dim: int, target: int) -> torch.Tensor:
    """Pad or crop *tensor* along *dim* to *target* length."""
    current = tensor.shape[dim]
    if current == target:
        return tensor
    if current > target:
        warnings.warn(
            f"Pad target {target} is shorter than tensor {current}, cropping.",
            UserWarning,
            stacklevel=3,
        )
        return tensor.narrow(dim, 0, target)
    ndim = tensor.ndim
    d = dim % ndim
    pad = [0] * (2 * ndim)
    pad[2 * (ndim - 1 - d) + 1] = target - current
    return torch.nn.functional.pad(tensor, pad)


class PadToLength(PaddingStrategy):
    """Pad all tensors to a target length along ``dim``.

    Parameters
    ----------
    length
        Target size (in samples) along the padded dimension, or ``"max"``
        to use the longest tensor in the batch.
    """

    dim: int = -1
    length: int | tp.Literal["max"] = "max"

    def __call__(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        target = (
            max(t.shape[self.dim] for t in tensors)
            if self.length == "max"
            else self.length
        )
        return [_pad_dim(t, self.dim, target) for t in tensors]


class PadToDuration(PaddingStrategy):
    """Pad all tensors to a fixed duration along ``dim``.

    Parameters
    ----------
    duration
        Target duration in seconds.
    The sampling rate used to convert *duration* to samples is auto-filled
    from the owning :class:`BaseExtractor`.
    """

    dim: int = -1
    duration: float
    _frequency: float | None = pydantic.PrivateAttr(None)

    def __call__(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        if self._frequency is None:
            raise RuntimeError(
                "PadToDuration frequency is not set. Set this strategy on a "
                "BaseExtractor with a numeric frequency."
            )
        target = max(1, Frequency(self._frequency).to_ind(self.duration))
        return [_pad_dim(t, self.dim, target) for t in tensors]


__all__: tp.Sequence[str] = [
    "PaddingStrategy",
    "PadToLength",
    "PadToDuration",
]
