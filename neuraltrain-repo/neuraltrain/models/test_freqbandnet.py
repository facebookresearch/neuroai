# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .freqbandnet import EEG_FREQ_LIMS_HZ, FreqBandNet, FreqBandNetModel


@pytest.fixture
def fake_meg():
    sfreq = 120.0
    B, C, T = 2, 4, int(sfreq * 20)
    meg = torch.randn(B, C, T)
    return meg, sfreq


@pytest.mark.parametrize("use_default_freq_lims", [True, False])
def test_freqbandnet(fake_meg, use_default_freq_lims):
    meg, sfreq = fake_meg
    batch_size, n_in_channels, _ = meg.shape

    config_kwargs = {}
    if not use_default_freq_lims:
        config_kwargs = {
            "n_filters_conv": 2,
            "freq_lims_hz": None,
            "conv_kernel_len": 121,
            "pool_kernel_len": 60,
            "flat_out": "channels",
            "n_outputs": 10,
        }

    model = FreqBandNet(**config_kwargs).build(sfreq=sfreq)
    assert isinstance(model, FreqBandNetModel)

    out = model(meg)
    assert out.shape[0] == batch_size
    if use_default_freq_lims:
        assert out.shape[1] == len(EEG_FREQ_LIMS_HZ) - 1
        assert out.shape[2] == n_in_channels
        assert out.shape[3] < meg.shape[2]
    else:
        assert out.shape[2] == config_kwargs["n_outputs"]


@pytest.mark.parametrize("flat_out", [None, "channels", "channels_and_time"])
def test_freqbandnet_flat_out(fake_meg, flat_out):
    meg, sfreq = fake_meg
    batch_size, n_in_channels, n_times = meg.shape

    conv_kernel_len = 65
    pool_kernel_len = 10

    model = FreqBandNet(
        sfreq=sfreq,
        conv_kernel_len=conv_kernel_len,
        pool_kernel_len=pool_kernel_len,
        flat_out=flat_out,
    ).build()
    assert isinstance(model, FreqBandNetModel)

    out = model(meg)

    n_features = len(EEG_FREQ_LIMS_HZ) - 1
    n_out_times = (n_times - conv_kernel_len + 1) // pool_kernel_len
    if flat_out == "channels":
        assert out.shape == (batch_size, n_features * n_in_channels, n_out_times)
    elif flat_out == "channels_and_time":
        assert out.shape == (batch_size, n_features * n_in_channels * n_out_times)
    else:
        assert out.shape == (batch_size, n_features, n_in_channels, n_out_times)
