# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch


@pytest.fixture
def fake_meg():
    batch_size = 2
    n_channels = 4
    n_times = 1200
    meg = torch.randn(batch_size, n_channels, n_times)
    return meg


@pytest.mark.parametrize("use_default_config", [True, False])
def test_green(fake_meg, use_default_config):
    pytest.importorskip("green")
    from green.research_code.pl_utils import Green as GreenImpl  # type: ignore

    from .green import Green

    batch_size, n_in_channels, _ = fake_meg.shape
    n_outputs = 3

    if use_default_config:
        config_kwargs = dict()
    else:
        config_kwargs = dict(
            n_freqs=5,  # Learning 5 wavelets
            dropout=0.5,  # Dropout rate of 0.5 in FC layers
            hidden_dim=[100],  # Use 100 units in the hidden layer
            bi_out=[20],  # Use a BiMap layer outputting a 20x20 matrix
        )

    sfreq = 100
    model = Green(**config_kwargs).build(
        n_in_channels=n_in_channels, n_outputs=n_outputs, sfreq=sfreq
    )
    assert isinstance(model, GreenImpl)

    out = model(fake_meg)
    assert out.shape == (batch_size, n_outputs)
