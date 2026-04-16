# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torch import nn, optim

from .base import Adam, LightningOptimizer, OneCycleLR  # type: ignore[attr-defined]


def test_torch_optimizer_config() -> None:
    model = nn.Linear(1, 1)

    cfg = Adam(
        lr=1e-5,
        kwargs={
            "betas": (0.1, 0.111),
            "weight_decay": 0.1,
        },
    )
    optimizer = cfg.build(model.parameters())  # type: ignore

    assert isinstance(optimizer, optim.Adam)


def test_torch_lr_scheduler_config() -> None:
    model = nn.Linear(1, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    lr_scheduler = OneCycleLR(
        kwargs={
            "max_lr": 0.1,
            "pct_start": 0.1,
            "anneal_strategy": "linear",
            "total_steps": 10,
        },
    ).build(
        optimizer,
        total_steps=100,
    )  # type: ignore

    assert isinstance(lr_scheduler, optim.lr_scheduler.OneCycleLR)
    assert lr_scheduler.total_steps == 100


@pytest.mark.parametrize("scheduler_name", ["OneCycleLR", "CosineAnnealingLR", None])
def test_lightning_optimizer_config(scheduler_name) -> None:
    model = nn.Linear(1, 1)

    optimizer_config = dict(
        name="Adam",
        lr=1e-5,
    )
    if scheduler_name == "OneCycleLR":
        scheduler_config = dict(
            name="OneCycleLR",
            kwargs={
                "max_lr": 5e-4,
                "pct_start": 0.05,
                "anneal_strategy": "linear",
            },
        )
        scheduler_build_kwargs = {"total_steps": 10}
    elif scheduler_name == "CosineAnnealingLR":
        scheduler_config = dict(
            name="CosineAnnealingLR",
            kwargs={
                "eta_min": 1e-5,
                "T_max": 10,  # Overwritten by scheduler_build_kwargs
                "last_epoch": -1,
            },
        )
        scheduler_build_kwargs = {"T_max": 11}
    else:
        scheduler_config = None
        scheduler_build_kwargs = {}

    cfg = LightningOptimizer(
        optimizer=optimizer_config,  # type: ignore
        scheduler=scheduler_config,  # type: ignore
        interval="step",
    )
    optimizer = cfg.build(model.parameters(), **scheduler_build_kwargs)  # type: ignore

    assert isinstance(optimizer["optimizer"], optim.Adam)
    if scheduler_name is not None:
        assert isinstance(
            optimizer["lr_scheduler"]["scheduler"], optim.lr_scheduler.LRScheduler
        )
