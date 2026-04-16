# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic configurations for optimizers."""

import typing as tp

import exca
import pydantic
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from neuraltrain.utils import all_subclasses

TORCH_OPTIMIZER_NAMES = {
    cls.__name__: cls for cls in all_subclasses(Optimizer) if cls.__name__ != "NewCls"
}
TORCH_LR_SCHEDULER_NAMES = {cls.__name__: cls for cls in all_subclasses(LRScheduler)}


class BaseOptimizer(exca.helpers.DiscriminatedModel, discriminator_key="name"):
    """Base class for optimizer configurations."""

    def build(self, params: tp.Iterable[torch.Tensor]) -> Optimizer:
        raise NotImplementedError


# Base class for torch optimizers configs (using kwargs pattern)
class BaseTorchOptimizer(BaseOptimizer):
    """Base class for torch optimizer configurations."""

    _OPTIMIZER_CLASS: tp.ClassVar[type[Optimizer]]
    lr: float
    kwargs: dict[str, tp.Any] = {}

    def build(self, params: tp.Iterable[torch.Tensor]) -> Optimizer:
        if "lr" in self.kwargs:
            raise ValueError(
                "lr should be defined as a base parameter instead of within kwargs."
            )
        exca.helpers.validate_kwargs(
            self._OPTIMIZER_CLASS, self.kwargs | {"params": None}
        )
        return self._OPTIMIZER_CLASS(params, lr=self.lr, **self.kwargs)  # type: ignore[call-arg]


# Generate config classes for all torch optimizers
for optimizer_name, optimizer_class in TORCH_OPTIMIZER_NAMES.items():
    optimizer_config_cls: type[BaseTorchOptimizer] = pydantic.create_model(  # type: ignore[assignment]
        optimizer_name,
        __base__=BaseTorchOptimizer,
    )
    optimizer_config_cls._OPTIMIZER_CLASS = optimizer_class  # type: ignore[attr-defined]
    globals()[optimizer_name] = optimizer_config_cls


class BaseLRScheduler(exca.helpers.DiscriminatedModel, discriminator_key="name"):
    """Base class for learning rate scheduler configurations."""

    def build(self, optimizer: Optimizer, **build_kwargs: tp.Any) -> LRScheduler:
        raise NotImplementedError


# Base class for torch LR scheduler configs (using kwargs pattern)
class BaseTorchLRScheduler(BaseLRScheduler):
    """Base class for torch LR scheduler configurations."""

    _SCHEDULER_CLASS: tp.ClassVar[type[LRScheduler]]
    kwargs: dict[str, tp.Any] = {}

    def build(self, optimizer: Optimizer, **build_kwargs: tp.Any) -> LRScheduler:
        exca.helpers.validate_kwargs(
            self._SCHEDULER_CLASS, self.kwargs | {"optimizer": None}
        )
        return self._SCHEDULER_CLASS(optimizer, **(self.kwargs | build_kwargs))


# Generate config classes for all torch LR schedulers
for scheduler_name, scheduler_class in TORCH_LR_SCHEDULER_NAMES.items():
    scheduler_config_cls: type[BaseTorchLRScheduler] = pydantic.create_model(  # type: ignore[assignment]
        scheduler_name,
        __base__=BaseTorchLRScheduler,
    )
    scheduler_config_cls._SCHEDULER_CLASS = scheduler_class  # type: ignore[attr-defined]
    globals()[scheduler_name] = scheduler_config_cls


class LightningOptimizer(BaseOptimizer):
    """Pydantic configuration for Lightning optimizer."""

    optimizer: BaseOptimizer
    scheduler: BaseLRScheduler | None = None
    interval: tp.Literal["step", "epoch"] = "step"

    def build(  # type: ignore[override]
        self,
        params: tp.Iterable[torch.Tensor],
        **scheduler_build_kwargs: tp.Any,
    ) -> dict[str, tp.Any]:
        out: dict[str, tp.Any] = {"optimizer": self.optimizer.build(params)}
        if self.scheduler is not None:
            scheduler = self.scheduler.build(out["optimizer"], **scheduler_build_kwargs)
            out["lr_scheduler"] = {"scheduler": scheduler, "interval": self.interval}
        return out
