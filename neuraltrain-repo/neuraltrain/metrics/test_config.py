# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pydantic
import pytest
import torch
import torchmetrics

from . import BaseMetric
from .base import MeanSquaredError  # type: ignore[attr-defined]
from .metrics import Rank, TopkAcc


@pytest.fixture
def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(8, 4), torch.randn(8, 4)


class Trainer(pydantic.BaseModel):
    metrics: list[BaseMetric]

    def compute_metrics(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tp.List[float]:
        metrics = {metric.log_name: metric.build() for metric in self.metrics}
        return [metrics[k](*inputs) for k in metrics]


def test_multi_metric_config(inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    config = {
        "metrics": [
            {"log_name": "median_rank", "name": "Rank", "reduction": "median"},
            {"log_name": "mean_rank", "name": "Rank", "reduction": "mean"},
            {"log_name": "top1_acc", "name": "TopkAcc", "topk": 1},
            {"log_name": "top5_acc", "name": "TopkAcc", "topk": 5},
        ]
    }
    trainer = Trainer(**config)  # type: ignore

    out = trainer.compute_metrics(inputs)

    metrics = {
        "median_rank": Rank(reduction="median"),
        "mean_rank": Rank(reduction="mean"),
        "top1_acc": TopkAcc(topk=1),
        "top5_acc": TopkAcc(topk=5),
    }

    out2 = [metric(*inputs) for metric in metrics.values()]

    assert out == out2


@pytest.mark.parametrize("kwargs", [{"squared": True}, {"squared": False}])
def test_torchmetrics_config(kwargs: dict[str, tp.Any]) -> None:
    x, y = torch.randn(8, 4), torch.randn(8, 4)
    metric = MeanSquaredError(log_name="mse", kwargs=kwargs).build()
    out = metric(x, y)
    out2 = torchmetrics.MeanSquaredError(**kwargs)(x, y)
    assert out == out2


def test_torchmetrics_config_validation() -> None:
    with pytest.raises(TypeError):
        MeanSquaredError(log_name="blublu", kwargs={"squared": 12}).build()
