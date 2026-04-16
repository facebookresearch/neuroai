# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pydantic
import torch
import torch.nn as nn

from . import BaseLoss, BaseMetric, BaseModelConfig, BaseOptimizer


class Experiment(pydantic.BaseModel):
    brain_model: BaseModelConfig
    loss: BaseLoss
    optimizer: BaseOptimizer
    metrics: list[BaseMetric]

    def run(self) -> tp.Any:
        model = self.brain_model.build(input_size=4, output_size=4)  # type: ignore
        loss = self.loss.build()
        optimizer = self.optimizer.build(model.parameters())
        metrics = {metric.log_name: metric.build() for metric in self.metrics}
        return model, loss, optimizer, metrics


def test_config() -> None:
    config = {
        "brain_model": {
            "name": "Mlp",
            "hidden_sizes": [20, 30],
        },
        "loss": {
            "name": "MSELoss",
        },
        "optimizer": {
            "name": "Adam",
            "lr": 0.001,
        },
        "metrics": [
            {
                "log_name": "top1",
                "name": "Accuracy",
                "kwargs": {
                    "task": "multiclass",
                    "top_k": 1,
                    "num_classes": 4,
                },
            },
            {
                "log_name": "top3",
                "name": "Accuracy",
                "kwargs": {
                    "task": "multiclass",
                    "top_k": 3,
                    "num_classes": 4,
                },
            },
        ],
    }
    experiment = Experiment(**config)  # type: ignore
    model, loss, optimizer, metrics = experiment.run()
    assert isinstance(model, nn.Module)
    assert isinstance(loss, nn.Module)
    assert all(isinstance(m, nn.Module) for m in metrics.values())
    assert isinstance(optimizer, torch.optim.Optimizer)
