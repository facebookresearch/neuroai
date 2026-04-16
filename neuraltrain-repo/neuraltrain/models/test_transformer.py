# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from x_transformers import Encoder

from .transformer import TransformerEncoder


@pytest.fixture
def fake_sequence():
    batch_size = 2
    dim = 256
    n_times = 10
    seq = torch.randn(batch_size, n_times, dim)
    return seq


def test_transformer(fake_sequence):
    batch_size, n_times, dim = fake_sequence.shape

    model = TransformerEncoder().build(dim)
    assert isinstance(model, Encoder)

    out = model(fake_sequence)
    assert out.shape == (batch_size, n_times, dim)


def test_transformer_dim_not_divisible_by_heads(fake_sequence):
    with pytest.raises(ValueError):
        TransformerEncoder(heads=7).build(fake_sequence.shape[2])
