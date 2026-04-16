# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from . import conformer as _conformer


@pytest.fixture
def fake_input():
    batch_size = 2
    n_channels = 4
    n_times = 1200
    x = torch.randn(batch_size, n_channels, n_times)
    return x


@pytest.mark.parametrize("use_default_config", [True, False])
def test_conformer(fake_input, use_default_config):
    from torchaudio.models import Conformer

    batch_size, n_in_channels, n_times = fake_input.shape

    if use_default_config:
        config_kwargs = dict()
    else:
        config_kwargs = dict(
            num_heads=2,
            num_layers=2,
            ffn_dim=32,
            dropout=0.1,
            depthwise_conv_kernel_size=9,
        )

    model = _conformer.Conformer(**config_kwargs).build(dim=n_in_channels)

    assert isinstance(model, Conformer)

    x = fake_input.transpose(1, 2)  # (B, C, T) → (B, T, C)
    out = model(x)  # Only one output now

    # Check shape of the output
    assert out.shape == (batch_size, n_times, n_in_channels)
