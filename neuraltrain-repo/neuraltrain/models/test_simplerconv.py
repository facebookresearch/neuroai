# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .common import ChannelMergerModel
from .simplerconv import SimplerConv, SimplerConvModel


@pytest.fixture
def fake_eeg():
    batch_size = 16
    n_channels = 8
    n_times = 800
    meg = torch.randn(batch_size, n_channels, n_times)
    return meg


@pytest.mark.parametrize("use_default_config", [True, False])
def test_simpler_conv(fake_eeg, use_default_config):
    batch_size, n_in_channels, n_times = fake_eeg.shape
    channel_positions = torch.rand((batch_size, n_in_channels, 2))

    config_kwargs = {}
    if not use_default_config:
        config_kwargs = {
            # Channel merger
            "merger_config": dict(
                n_virtual_channels=270,
                fourier_emb_config=dict(
                    n_freqs=None,
                    total_dim=2048,
                ),
                dropout=0.2,
                usage_penalty=0.0,
                per_subject=False,
                embed_ref=False,
            ),
            # Encoder
            "kernel_sizes": (5, 2),
            "strides": (5, 1),
            "output_nonlin": False,
            "dropout": 0.5,
        }

    n_outputs = 512
    model = SimplerConv(**config_kwargs).build(n_in_channels, n_outputs=n_outputs)
    assert isinstance(model, SimplerConvModel)
    if not use_default_config:
        assert isinstance(model.merger, ChannelMergerModel)

    out = model(fake_eeg, channel_positions=channel_positions)
    assert out.ndim == 3
    assert out.shape[0] == batch_size
    assert out.shape[1] == n_outputs

    out_n_times = n_times // model.downsampling_factor
    if not use_default_config:
        out_n_times -= 1  # account for no padding
    assert out.shape[2] == out_n_times

    assert model.receptive_field >= model.downsampling_factor
