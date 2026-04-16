# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic configurations for loss functions."""

import typing as tp
from inspect import isclass

import exca
import pydantic
from torch import nn
from torch.nn.modules.loss import _Loss

from neuraltrain.utils import all_subclasses, convert_to_pydantic

from . import losses

custom_losses = [
    obj for obj in losses.__dict__.values() if isclass(obj) and issubclass(obj, nn.Module)
]

TORCHLOSS_NAMES = {
    loss_class.__name__: loss_class for loss_class in all_subclasses(_Loss)
}


class BaseLoss(exca.helpers.DiscriminatedModel, discriminator_key="name"):
    """Base class for loss configurations."""

    def build(self, **kwargs: tp.Any) -> nn.Module:
        raise NotImplementedError


# Generate config classes using convert_to_pydantic for custom losses
for loss_class in custom_losses:
    if loss_class.__name__ == "MultiLoss":
        continue
    config_cls = convert_to_pydantic(
        loss_class, loss_class.__name__, parent_class=BaseLoss
    )
    globals()[loss_class.__name__] = config_cls


# Base class for torch loss configs (using kwargs pattern)
class BaseTorchLoss(BaseLoss):
    """Base class for torch loss configurations."""

    _LOSS_CLASS: tp.ClassVar[type[nn.Module]]
    kwargs: dict[str, tp.Any] = {}

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # validation of mandatory/extra args + basic types (str/int/float)
        exca.helpers.validate_kwargs(self._LOSS_CLASS, self.kwargs)

    def build(self, **kwargs: tp.Any) -> nn.Module:
        if overlap := set(self.kwargs) & set(kwargs):
            raise ValueError(
                f"Build kwargs overlap with config kwargs for keys: {overlap}."
            )
        kwargs = self.kwargs | kwargs
        return self._LOSS_CLASS(**kwargs)


# Generate config classes for all torch losses
for loss_name, loss_class in TORCHLOSS_NAMES.items():
    torch_loss_cls: type[BaseTorchLoss] = pydantic.create_model(  # type: ignore[assignment]
        loss_name,
        __base__=BaseTorchLoss,
    )
    torch_loss_cls._LOSS_CLASS = loss_class  # type: ignore[attr-defined]
    globals()[loss_name] = torch_loss_cls


class MultiLoss(BaseLoss):
    """Pydantic configuration for multi-loss."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    losses: BaseLoss | dict[str, BaseLoss]
    weights: dict[str, float] | None = None

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if isinstance(self.losses, dict) and isinstance(self.weights, dict):
            diff = set(self.losses).symmetric_difference(self.weights)
            if diff:
                raise ValueError(f"weights and losses key differ: {diff}")

    def build(self, **kwargs: tp.Any) -> nn.Module:
        if not isinstance(self.losses, dict):
            return self.losses.build()
        built_losses = {name: loss.build() for name, loss in self.losses.items()}
        return losses.MultiLoss(built_losses, weights=self.weights)
