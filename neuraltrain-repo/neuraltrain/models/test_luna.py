# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import braindecode.models
import pytest
import torch

from .luna import NtLuna, _LunaEncoderWrapper

_N_CHANS = 4
_N_TIMES = 120
_PATCH_SIZE = 40
_N_OUTPUTS = 3

_TINY_LUNA_KWARGS = dict(
    patch_size=_PATCH_SIZE,
    num_queries=2,
    embed_dim=16,
    depth=1,
    num_heads=2,
    mlp_ratio=2.0,
)


def _make_luna(n_outputs: int = _N_OUTPUTS):
    """Build a tiny from-scratch LUNA model for testing."""
    return braindecode.models.LUNA(
        n_outputs=n_outputs,
        n_chans=_N_CHANS,
        n_times=_N_TIMES,
        **_TINY_LUNA_KWARGS,
    )


# ---------------------------------------------------------------------------
# _LunaEncoderWrapper
# ---------------------------------------------------------------------------


def test_wrapper_forward():
    """Wrapper forwards with and without channel_positions."""
    luna = _make_luna()
    wrapper = _LunaEncoderWrapper(luna)

    X = torch.randn(2, _N_CHANS, _N_TIMES)
    pos = torch.randn(2, _N_CHANS, 3)
    with torch.no_grad():
        out_with = wrapper(X, channel_positions=pos)
        out_without = wrapper(X[:1])

    assert out_with.shape == (2, _N_OUTPUTS)
    assert out_without.shape == (1, _N_OUTPUTS)


def test_wrapper_time_padding():
    """_pad_to_patch_size pads unaligned inputs and is a no-op when aligned."""
    X_aligned = torch.randn(1, _N_CHANS, _PATCH_SIZE * 3)
    assert (
        _LunaEncoderWrapper._pad_to_patch_size(X_aligned, _PATCH_SIZE).shape
        == X_aligned.shape
    )

    X_unaligned = torch.randn(1, _N_CHANS, _PATCH_SIZE + 7)
    padded = _LunaEncoderWrapper._pad_to_patch_size(X_unaligned, _PATCH_SIZE)
    assert padded.shape[-1] == _PATCH_SIZE * 2
    assert padded.shape[-1] % _PATCH_SIZE == 0


def test_wrapper_mapping():
    """mapping prefixes inner keys with 'model.' and includes temperature typo fix."""
    wrapper = _LunaEncoderWrapper(_make_luna())
    mapping = wrapper.mapping

    for v in mapping.values():
        assert v.startswith("model."), f"Expected 'model.' prefix, got {v!r}"
    # Misspelled source key is intentional: tests the typo-fix mapping
    assert mapping["cross_attn.temparature"] == "model.cross_attn.temperature"  # codespell:ignore temparature


# ---------------------------------------------------------------------------
# NtLuna.build
# ---------------------------------------------------------------------------


def test_build_classification():
    """With n_outputs, build returns a wrapper with classification output."""
    cfg = NtLuna(kwargs=_TINY_LUNA_KWARGS)
    model = cfg.build(n_chans=_N_CHANS, n_times=_N_TIMES, n_outputs=_N_OUTPUTS)

    assert isinstance(model, _LunaEncoderWrapper)

    X = torch.randn(2, _N_CHANS, _N_TIMES)
    pos = torch.randn(2, _N_CHANS, 3)
    with torch.no_grad():
        out = model(X, channel_positions=pos)
    assert out.shape == (2, _N_OUTPUTS)


@pytest.mark.parametrize("n_times", [120, 90])
def test_build_encoder_only(n_times):
    """Without n_outputs, final_layer is Identity and output is 3-D.
    Works for both aligned and unaligned n_times."""
    num_queries = _TINY_LUNA_KWARGS["num_queries"]
    embed_dim = _TINY_LUNA_KWARGS["embed_dim"]

    cfg = NtLuna(kwargs=_TINY_LUNA_KWARGS)
    model = cfg.build(n_chans=_N_CHANS, n_times=n_times)

    assert isinstance(model, _LunaEncoderWrapper)
    assert isinstance(model.model.final_layer, torch.nn.Identity)

    X = torch.randn(1, _N_CHANS, n_times)
    pos = torch.randn(1, _N_CHANS, 3)
    with torch.no_grad():
        out = model(X, channel_positions=pos)

    padded_times = (
        n_times
        if n_times % _PATCH_SIZE == 0
        else (_PATCH_SIZE * ((n_times // _PATCH_SIZE) + 1))
    )
    n_patches = padded_times // _PATCH_SIZE
    assert out.shape == (1, n_patches, num_queries * embed_dim)
