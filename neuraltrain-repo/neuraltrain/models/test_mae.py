# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .common import INVALID_POS_VALUE, FourierEmb
from .mae import MaeEncoder
from .transformer import TransformerEncoder

PATCH_SIZE, DIM = 20, 64


@pytest.fixture
def config() -> MaeEncoder:
    return MaeEncoder(
        dim=DIM,
        patch_size=PATCH_SIZE,
        channel_emb_config=FourierEmb(n_freqs=2, n_dims=3),
        transformer_config=TransformerEncoder(heads=2, depth=1, rotary_pos_emb=False),
    )


def _positions(n_channels: int, batch_size: int = 2) -> torch.Tensor:
    """Distinct normalized 3D positions, as ``ns.extractors.ChannelPositions`` emits."""
    return torch.rand(batch_size, n_channels, 3)


# one instance must span window lengths; 205 has an incomplete trailing patch
@pytest.mark.parametrize("n_times", [200, 205, 400])
@pytest.mark.parametrize("n_outputs", [None, 3])
def test_build_and_forward(config, n_times, n_outputs) -> None:
    model = config.build(n_outputs=n_outputs)
    out = model(torch.randn(2, 8, n_times), _positions(8))

    n_patches = n_times // PATCH_SIZE
    expected = (2, 3) if n_outputs is not None else (2, 8 * n_patches, DIM)
    assert out.shape == expected, f"unexpected output shape {tuple(out.shape)}"
    assert (n_outputs is not None) == any(
        name.startswith("head.") for name in model.state_dict()
    ), "output head must exist if and only if n_outputs is set"


@pytest.mark.parametrize("n_channels", [2, 19, 63])
def test_one_instance_spans_montages(config, n_channels) -> None:
    model = config.build()
    out = model(torch.randn(2, n_channels, 200), _positions(n_channels))

    assert out.shape == (2, n_channels * (200 // PATCH_SIZE), DIM)


def test_absent_channels_are_dropped_from_attention(config) -> None:
    model = config.build().eval()
    positions = _positions(4)
    positions[:, 3] = INVALID_POS_VALUE  # how MneRaw zero-padding arrives

    x = torch.randn(2, 4, 200)
    other = x.clone()
    other[:, 3] = torch.randn(2, 200)

    n_kept = 3 * (200 // PATCH_SIZE)
    with torch.no_grad():
        out, out_other = model(x, positions), model(other, positions)

    assert torch.allclose(out[:, :n_kept], out_other[:, :n_kept], atol=1e-5), (
        "tokens of present channels changed when an absent channel's padding did"
    )
    assert not out[:, n_kept:].any(), "absent channels must leave zeroed tokens"


def test_channel_identity_comes_from_position(config) -> None:
    model = config.build().eval()
    x = torch.randn(2, 5, 200)
    positions = _positions(5)
    order = [3, 0, 4, 1, 2]

    n_patches = 200 // PATCH_SIZE
    with torch.no_grad():
        out = model(x, positions).unflatten(1, (5, n_patches))
        shuffled = model(x[:, order], positions[:, order]).unflatten(1, (5, n_patches))

    assert torch.allclose(out[:, order], shuffled, atol=1e-5), (
        "reordering channels with their positions changed their tokens"
    )


def test_patchify_rejects_too_short_input(config) -> None:
    model = config.build()
    with pytest.raises(ValueError, match="less than patch_size"):
        model.patchify(torch.randn(2, 4, PATCH_SIZE - 1))
