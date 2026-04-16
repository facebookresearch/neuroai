# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .bendr import (
    _DEFAULT_AMPLITUDE_RANGE,
    BENDR_CHANNEL_ORDER,
    NtBendr,
    _BendrAllTokensWrapper,
    _build_channel_indices,
)

try:
    import braindecode.models

    braindecode.models.BENDR  # noqa: B018
    _HAS_BENDR = True
except AttributeError:
    _HAS_BENDR = False

requires_bendr = pytest.mark.skipif(
    not _HAS_BENDR,
    reason="braindecode.models.BENDR not available in this environment",
)

pytestmark = requires_bendr

_N_CHANS = 20
_N_OUTPUTS = 3
_ENCODER_H = 64

_TINY_BENDR_KWARGS = dict(
    encoder_h=_ENCODER_H,
    contextualizer_hidden=128,
    transformer_layers=1,
    transformer_heads=2,
    enc_width=(3, 2, 2),
    enc_downsample=(3, 2, 2),
    final_layer=False,
)


@pytest.fixture()
def tiny_model():
    """A small braindecode BENDR for testing (eval mode, no grad)."""
    model = braindecode.models.BENDR(
        n_chans=_N_CHANS, n_times=256, n_outputs=1, **_TINY_BENDR_KWARGS
    )
    model.eval()
    return model


def _encoder_output_length(n_chans: int, n_times: int) -> int:
    """Run the encoder once to determine its output temporal dimension."""
    model = braindecode.models.BENDR(
        n_chans=n_chans, n_times=n_times, n_outputs=1, **_TINY_BENDR_KWARGS
    )
    model.eval()
    with torch.no_grad():
        encoded = model.encoder(torch.randn(1, n_chans, n_times))
    return encoded.shape[2]


def _make_wrapper(model, ch_names, amplitude_range=_DEFAULT_AMPLITUDE_RANGE):
    """Build a _BendrAllTokensWrapper with channel mapping from *ch_names*."""
    pairs = _build_channel_indices(ch_names, channel_mapping=None)
    return _BendrAllTokensWrapper(
        model, channel_pairs=pairs, amplitude_range=amplitude_range
    )


# ---------------------------------------------------------------------------
# Channel mapping tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "names",
    [BENDR_CHANNEL_ORDER, [ch.upper() for ch in BENDR_CHANNEL_ORDER]],
    ids=["exact", "upper"],
)
def test_build_channel_indices_all_match(names):
    """All 19 standard channels resolve regardless of case."""
    pairs = _build_channel_indices(names, channel_mapping=None)
    assert len(pairs) == 19


def test_build_channel_indices_subset():
    """Only present channels appear in pairs; missing ones are absent."""
    subset = ["Fp1", "Cz", "O2"]
    pairs = _build_channel_indices(subset, channel_mapping=None)
    resolved_bendr = {p[1] for p in pairs}
    assert len(pairs) == 3
    for name in subset:
        assert BENDR_CHANNEL_ORDER.index(name) in resolved_bendr


def test_build_channel_indices_explicit_mapping():
    names = ["EEG 007", "EEG 017"]
    mapping = {"EEG 007": "F4", "EEG 017": "T7"}
    pairs = _build_channel_indices(names, channel_mapping=mapping)
    resolved = {BENDR_CHANNEL_ORDER[p[1]] for p in pairs}
    assert resolved == {"F4", "T7"}


def test_build_channel_indices_bipolar_fallback():
    """Bipolar channels like 'Pz-Oz' resolve via anode 'Pz'."""
    pairs = _build_channel_indices(["Pz-Oz"], channel_mapping=None)
    assert len(pairs) == 1
    assert BENDR_CHANNEL_ORDER[pairs[0][1]] == "Pz"


# ---------------------------------------------------------------------------
# Wrapper _prepare_input tests
# ---------------------------------------------------------------------------


def test_prepare_input_shape_and_range(tiny_model):
    """_prepare_input produces (B, 20, T) with EEG channels in [-1, 1]."""
    ch_names = BENDR_CHANNEL_ORDER[:10]
    wrapper = _make_wrapper(tiny_model, ch_names)

    x = torch.randn(2, len(ch_names), 256) * 100
    out = wrapper._prepare_input(x)

    assert out.shape == (2, 20, 256)
    eeg_part = out[:, :19, :]
    assert eeg_part.min() >= -1.0 - 1e-5
    assert eeg_part.max() <= 1.0 + 1e-5


def test_prepare_input_amplitude_channel(tiny_model):
    """The 20th channel encodes relative amplitude = window_range / R_ds."""
    R_ds = 1000.0
    ch_names = BENDR_CHANNEL_ORDER[:5]
    wrapper = _make_wrapper(tiny_model, ch_names, amplitude_range=R_ds)

    x = torch.zeros(1, len(ch_names), 128)
    x[0, 0, 0] = -50.0
    x[0, 0, 1] = 150.0
    out = wrapper._prepare_input(x)
    torch.testing.assert_close(
        out[0, 19, :],
        torch.full((128,), 200.0 / R_ds),
        atol=1e-5,
        rtol=1e-5,
    )


def test_prepare_input_zero_fill_missing(tiny_model):
    """Missing BENDR channels are zero-filled after MinMaxScaler."""
    wrapper = _make_wrapper(tiny_model, ["Fp1"])

    out = wrapper._prepare_input(torch.randn(1, 1, 128))
    fp1_idx = BENDR_CHANNEL_ORDER.index("Fp1")
    fp2_idx = BENDR_CHANNEL_ORDER.index("Fp2")
    assert out[0, fp1_idx].abs().sum() > 0
    assert out[0, fp2_idx].abs().sum() == 0.0


# ---------------------------------------------------------------------------
# Wrapper forward / build tests
# ---------------------------------------------------------------------------


def test_wrapper_cls_matches_original(tiny_model):
    """CLS token (position 0) from the wrapper matches BENDR.forward()."""
    x = torch.randn(1, _N_CHANS, 512)
    with torch.no_grad():
        cls_original = tiny_model(x)

    wrapper = _BendrAllTokensWrapper(tiny_model, channel_pairs=None)
    wrapper.eval()
    with torch.no_grad():
        all_tokens = wrapper(x)

    torch.testing.assert_close(all_tokens[:, 0, :], cls_original)


@pytest.mark.parametrize("n_times", [256, 512])
def test_build_from_scratch(n_times):
    """NtBendr.build() wraps and returns all tokens with correct shape."""
    cfg = NtBendr(kwargs=_TINY_BENDR_KWARGS)
    model = cfg.build(n_chans=_N_CHANS, n_times=n_times, n_outputs=_N_OUTPUTS)

    assert isinstance(model, _BendrAllTokensWrapper)
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(2, _N_CHANS, n_times))

    n_encoded = _encoder_output_length(_N_CHANS, n_times)
    assert out.shape == (2, n_encoded + 1, _ENCODER_H)


def test_build_with_chs_info():
    """NtBendr.build() with chs_info creates a wrapper with channel mapping."""
    ch_names = ["Fp1", "Cz", "Pz", "O1", "O2"]
    chs_info = [{"ch_name": ch} for ch in ch_names]
    cfg = NtBendr(kwargs=_TINY_BENDR_KWARGS)
    model = cfg.build(n_chans=5, n_times=256, n_outputs=_N_OUTPUTS, chs_info=chs_info)

    assert isinstance(model, _BendrAllTokensWrapper)
    assert model._ds_indices is not None

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, len(ch_names), 256))

    n_encoded = _encoder_output_length(20, 256)
    assert out.shape == (1, n_encoded + 1, _ENCODER_H)


PRETRAINED_NAME = "braindecode/braindecode-bendr"


def _pretrained_weights_available() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(PRETRAINED_NAME, "model.safetensors")
        return isinstance(result, str)
    except (ImportError, Exception):
        return False


requires_pretrained = pytest.mark.skipif(
    not _pretrained_weights_available(),
    reason=f"Pretrained weights for {PRETRAINED_NAME!r} not cached locally",
)


@requires_pretrained
def test_build_pretrained():
    """Pretrained build returns _BendrAllTokensWrapper with 3-D output."""
    chs_info = [{"ch_name": ch} for ch in BENDR_CHANNEL_ORDER]
    cfg = NtBendr(
        from_pretrained_name=PRETRAINED_NAME,
        kwargs={"final_layer": False},
    )
    model = cfg.build(chs_info=chs_info)

    assert isinstance(model, _BendrAllTokensWrapper)

    x = torch.randn(1, 19, 512)
    with torch.no_grad():
        out = model(x)

    assert out.ndim == 3
    assert out.shape[0] == 1
    assert out.shape[2] == 512  # default encoder_h
