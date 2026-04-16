# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn

from .base import (  # type: ignore[attr-defined]
    ClipLoss,
    CrossEntropyLoss,
    MSELoss,
    MultiLoss,
    SigLipLoss,
)
from .losses import ClipLoss as ClipLossImpl
from .losses import MultiLoss as MultiLossImpl
from .losses import SigLipLoss as SigLipLossImpl


@pytest.fixture
def inputs():
    return torch.randn(8, 4), torch.randn(8, 4)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_mse_loss_config(inputs, reduction: str) -> None:
    loss = MSELoss(kwargs={"reduction": reduction}).build()
    out = loss(*inputs)
    out2 = nn.MSELoss(reduction=reduction)(*inputs)
    assert torch.allclose(out, out2, atol=1e-5)


@pytest.mark.parametrize("weight", [None, [0.1, 0.2, 0.3, 0.4]])
def test_cross_entropy_loss_config(weight) -> None:
    n_examples, n_classes = 8, 4
    y_pred = torch.randn(n_examples, n_classes)
    y_true = torch.arange(n_examples) % n_classes

    kwargs = {
        "weight": weight,
        "ignore_index": 0,
        "reduction": "sum",
        "label_smoothing": 0.1,
    }
    if weight is not None:
        kwargs["weight"] = torch.Tensor(weight)
    loss = CrossEntropyLoss(kwargs=kwargs).build()
    out = loss(y_pred, y_true)
    out2 = nn.CrossEntropyLoss(**kwargs)(y_pred, y_true)
    assert torch.allclose(out, out2, atol=1e-5)


def test_torch_loss_config_with_build_kwargs() -> None:
    n_examples, n_classes = 8, 4
    y_pred = torch.randn(n_examples, n_classes)
    y_true = torch.arange(n_examples) % n_classes

    kwargs = {
        "ignore_index": 0,
        "reduction": "sum",
        "label_smoothing": 0.1,
    }
    weight = torch.Tensor([0.1, 0.2, 0.3, 0.4])

    loss = CrossEntropyLoss(kwargs=kwargs).build(weight=weight)
    out = loss(y_pred, y_true)
    out2 = nn.CrossEntropyLoss(**kwargs, weight=weight)(y_pred, y_true)  # type: ignore
    assert torch.allclose(out, out2, atol=1e-5)


def test_torch_loss_config_with_build_kwargs_overlap() -> None:
    kwargs = {
        "ignore_index": 0,
        "reduction": "sum",
        "label_smoothing": 0.1,
    }
    with pytest.raises(ValueError):
        CrossEntropyLoss(kwargs=kwargs).build(**kwargs)


def test_clip_loss_config(inputs) -> None:
    kwargs = {"norm_kind": "x", "symmetric": True, "temperature": True}
    loss = ClipLoss(**kwargs).build()  # type: ignore
    out = loss(*inputs)
    out2 = ClipLossImpl(**kwargs)(*inputs)  # type: ignore
    assert out == out2


def test_siglip_loss_config(inputs) -> None:
    kwargs = {"norm_kind": "x", "temperature": True, "bias": True}
    loss = SigLipLoss(**kwargs).build()  # type: ignore
    out = loss(*inputs)
    out2 = SigLipLossImpl(**kwargs)(*inputs)  # type: ignore
    assert out == out2


def test_multi_loss_config(inputs) -> None:
    losses = {
        "mse": {"name": "MSELoss", "kwargs": {"reduction": "sum"}},
        "clip": {
            "name": "ClipLoss",
            "norm_kind": "xy",
            "temperature": False,
            "reduction": "sum",
        },
    }
    weights = {"mse": 0.1, "clip": 0.9}

    loss = MultiLoss(losses=losses, weights=weights).build()  # type: ignore
    out = loss(*inputs)

    losses2 = {
        "mse": nn.MSELoss(reduction="sum"),
        "clip": ClipLossImpl(norm_kind="xy", temperature=False, reduction="sum"),
    }
    out2 = MultiLossImpl(losses2, weights=weights)(*inputs)  # type: ignore

    assert torch.allclose(out["mse"], out2["mse"], atol=1e-3)
    assert torch.allclose(out["clip"], out2["clip"], atol=1e-3)


@pytest.mark.parametrize("name", ["MSELoss", "CrossEntropyLoss"])
@pytest.mark.parametrize("kwargs", [{"reduction": "mean"}, {"reduction": "sum"}])
def test_torchloss_config_parametrized(name, kwargs) -> None:
    from . import base

    x, y = torch.randn(8, 4), torch.randn(8, 4)
    loss = getattr(base, name)(kwargs=kwargs).build()
    out = loss(x, y)
    out2 = getattr(nn, name)(**kwargs)(x, y)
    assert out == out2


def test_torchloss_config_validation() -> None:
    with pytest.raises(TypeError):
        MSELoss(kwargs={"reduction": 12})
