# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .common import INVALID_POS_VALUE, ChannelMerger, FourierEmb
from .mae import MaeEncoder
from .transformer import TransformerEncoder

N_VIRTUAL, PATCH_SIZE, DIM = 6, 20, 64


@pytest.fixture
def config() -> MaeEncoder:
    return MaeEncoder(
        dim=DIM,
        patch_size=PATCH_SIZE,
        merger_config=ChannelMerger(
            n_virtual_channels=N_VIRTUAL,
            fourier_emb_config=FourierEmb(n_freqs=2, n_dims=3),
            dropout=0.0,
        ),
        transformer_config=TransformerEncoder(heads=2, depth=1, rotary_pos_emb=False),
    )


def _positions(n_channels: int) -> torch.Tensor:
    """Distinct normalized 3D positions, as ``ns.extractors.ChannelPositions`` emits."""
    return torch.rand(2, n_channels, 3)


# One model instance must handle any window length, so that a checkpoint stays
# usable downstream; 205 also exercises the incomplete trailing patch.
@pytest.mark.parametrize("n_times", [200, 205, 2000])
@pytest.mark.parametrize("n_outputs", [None, 3])
def test_build_and_forward(config, n_times, n_outputs) -> None:
    model = config.build(n_outputs=n_outputs)
    out = model(torch.randn(2, 8, n_times), _positions(8))

    expected = (2, 3) if n_outputs is not None else (2, n_times // PATCH_SIZE, DIM)
    assert out.shape == expected, f"unexpected output shape {tuple(out.shape)}"
    assert (n_outputs is not None) == any(
        name.startswith("head.") for name in model.state_dict()
    ), "output head must exist if and only if n_outputs is set"


# The point of the merger: pretraining pools datasets that share no montage, so
# channel count must not reach any weight.
@pytest.mark.parametrize("n_channels", [2, 19, 63])
def test_one_instance_spans_montages(config, n_channels) -> None:
    model = config.build()
    out = model(torch.randn(2, n_channels, 200), _positions(n_channels))

    assert out.shape == (2, 200 // PATCH_SIZE, DIM)


def test_invalid_channels_are_masked_out(config) -> None:
    """A channel whose positions are all ``INVALID_POS_VALUE`` must not be read.

    That is how a recording missing a channel reaches the model: ``MneRaw``
    zero-pads the data to the union montage and ``ChannelPositions`` marks the
    padding invalid.
    """
    model = config.build()
    positions = _positions(4)
    positions[:, 3] = INVALID_POS_VALUE

    x = torch.randn(2, 4, 200)
    other = x.clone()
    other[:, 3] = torch.randn(2, 200)

    assert torch.allclose(model.merge(x, positions), model.merge(other, positions)), (
        "merged signal changed when an invalid channel did"
    )


def test_patchify_rejects_too_short_input(config) -> None:
    model = config.build()
    with pytest.raises(ValueError, match="less than patch_size"):
        model.patchify(torch.randn(2, N_VIRTUAL, PATCH_SIZE - 1))


def test_rejects_per_subject_merger(config) -> None:
    config.merger_config.per_subject = True
    with pytest.raises(ValueError, match="per_subject"):
        config.build()
