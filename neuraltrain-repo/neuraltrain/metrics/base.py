# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic configurations for metrics."""

import typing as tp
from inspect import isclass

import exca
import pydantic
from torchmetrics import Metric

from neuraltrain.metrics import metrics
from neuraltrain.utils import all_subclasses, convert_to_pydantic

custom_metrics = [
    obj for obj in metrics.__dict__.values() if isclass(obj) and issubclass(obj, Metric)
]


TORCHMETRICS_NAMES = {
    metric_class.__name__: metric_class
    for metric_class in all_subclasses(Metric)
    if metric_class not in custom_metrics
}


class BaseMetric(exca.helpers.DiscriminatedModel, discriminator_key="name"):
    """Base class for metric configurations."""

    log_name: str

    def build(self) -> Metric:
        raise NotImplementedError


# Generate config classes using convert_to_pydantic for custom metrics
for metric_class in custom_metrics:
    config_cls = convert_to_pydantic(
        metric_class,
        metric_class.__name__,
        parent_class=BaseMetric,
        exclude_from_build=["log_name"],
    )
    globals()[metric_class.__name__] = config_cls


# Base class for torchmetrics configs (using kwargs pattern)
class BaseTorchMetric(BaseMetric):
    """Base class for torchmetrics configurations."""

    _METRIC_CLASS: tp.ClassVar[type[Metric]]
    kwargs: dict[str, tp.Any] = {}

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # validation of mandatory/extra args + basic types (str/int/float)
        exca.helpers.validate_kwargs(self._METRIC_CLASS, self.kwargs)

    def build(self) -> Metric:
        return self._METRIC_CLASS(**self.kwargs)


# Generate config classes for all torchmetrics
for metric_name, metric_class in TORCHMETRICS_NAMES.items():
    torch_config_cls: type[BaseTorchMetric] = pydantic.create_model(  # type: ignore[assignment]
        metric_name,
        __base__=BaseTorchMetric,
    )
    torch_config_cls._METRIC_CLASS = metric_class  # type: ignore[attr-defined]
    globals()[metric_name] = torch_config_cls
