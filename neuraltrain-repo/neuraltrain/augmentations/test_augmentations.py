# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .augmentations import (
    BandstopFilterFFT,
    TrivialBrainAugment,
    TrivialBrainAugmentConfig,
)


@pytest.mark.parametrize("sfreq", [100, 250])
@pytest.mark.parametrize("bandwidth", [2, 4])
def test_bandstop_filter_fft(sfreq, bandwidth):
    transform = BandstopFilterFFT(sfreq=sfreq, bandwidth=bandwidth)
    x = torch.randn(10, 2, 512)
    z = transform(x)
    # Only check shape for now
    assert z.shape == x.shape


@pytest.mark.parametrize("sfreq", [100, 250])
def test_trivial_brain_augment(sfreq):
    transform = TrivialBrainAugment(TrivialBrainAugmentConfig(sfreq=sfreq))
    x = torch.randn(10, 2, 512)
    z = transform(x)
    # Only check shape for now
    assert z.shape == x.shape
