# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .common import SubjectLayersModel
from .linear import Linear


@pytest.fixture
def fake_meg() -> torch.Tensor:
    batch_size = 4
    n_channels = 8
    n_times = 120
    meg = torch.randn(batch_size, n_channels, n_times)
    return meg


@pytest.mark.parametrize("subject_layers_config", [{}, None])
@pytest.mark.parametrize("reduction", ["mean", "concat"])
def test_linear_model(
    fake_meg: torch.Tensor, subject_layers_config: dict | None, reduction: str
) -> None:
    if reduction == "concat":
        n_in_channels = fake_meg.shape[1] * fake_meg.shape[2]
    else:
        n_in_channels = fake_meg.shape[1]
    model_config = Linear(
        subject_layers_config=subject_layers_config,  # type: ignore[arg-type]
        reduction=reduction,  # type: ignore[arg-type]
    )
    model = model_config.build(n_in_channels, 10)
    subject_id = torch.randint(0, 2, (4, 1))
    assert model(fake_meg, subject_id=subject_id).shape == (4, 10)
    if subject_layers_config is not None:
        assert isinstance(model.linear, SubjectLayersModel)
    else:
        assert isinstance(model.linear, torch.nn.Linear)
