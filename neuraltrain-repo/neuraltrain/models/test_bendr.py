# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .bendr import (
    BENDR_N_EEG_CHANNELS,
    NtBendr,
    _BendrAllTokensWrapper,
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

_N_CHANS = BENDR_N_EEG_CHANNELS + 1  # 19 EEG + 1 amplitude = 20
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


# ---------------------------------------------------------------------------
# Wrapper _prepare_input tests
# ---------------------------------------------------------------------------


def test_prepare_input_shape_and_range(tiny_model):
    """_prepare_input produces (B, 20, T) with EEG channels in [-1, 1]."""
    wrapper = _BendrAllTokensWrapper(tiny_model)

    x = torch.randn(2, BENDR_N_EEG_CHANNELS, 256) * 100.0
    out = wrapper._prepare_input(x)

    assert out.shape == (2, _N_CHANS, 256)
    eeg_part = out[:, :BENDR_N_EEG_CHANNELS, :]
    assert eeg_part.min() >= -1.0 - 1e-5
    assert eeg_part.max() <= 1.0 + 1e-5


def test_prepare_input_global_minmax(tiny_model):
    """MinMax is global across channels+time, preserving inter-channel ratios."""
    wrapper = _BendrAllTokensWrapper(tiny_model)

    x = torch.zeros(1, BENDR_N_EEG_CHANNELS, 128)
    x[0, 0, 0] = -1.0  # global min
    x[0, 1, 0] = 1.0  # global max
    x[0, 2, 0] = 0.5  # halfway between -> should scale to 0.5 too

    out = wrapper._prepare_input(x)
    torch.testing.assert_close(out[0, 0, 0], torch.tensor(-1.0))
    torch.testing.assert_close(out[0, 1, 0], torch.tensor(1.0))
    torch.testing.assert_close(out[0, 2, 0], torch.tensor(0.5))
    # Unset entries (original 0.0) scale to 0 as well (since midpoint of [-1, 1]).
    torch.testing.assert_close(out[0, 3, 0], torch.tensor(0.0))


def test_prepare_input_amplitude_channel_unclamped(tiny_model):
    """Amplitude channel: 2*(clamp(window_range/R, 1.0) - 0.5) when range<R."""
    R_ds = 1000.0
    wrapper = _BendrAllTokensWrapper(tiny_model, amplitude_range=R_ds)

    x = torch.zeros(1, BENDR_N_EEG_CHANNELS, 128)
    x[0, 0, 0] = -50.0
    x[0, 0, 1] = 150.0  # window_range = 200, rel_amp = 0.2
    out = wrapper._prepare_input(x)

    expected = 2.0 * (0.2 - 0.5)  # -0.6
    torch.testing.assert_close(
        out[0, BENDR_N_EEG_CHANNELS, :],
        torch.full((128,), expected),
        atol=1e-5,
        rtol=1e-5,
    )


def test_prepare_input_amplitude_channel_clamped(tiny_model):
    """Amplitude channel clamps to 1.0 when window_range exceeds R_ds."""
    R_ds = 100.0
    wrapper = _BendrAllTokensWrapper(tiny_model, amplitude_range=R_ds)

    x = torch.zeros(1, BENDR_N_EEG_CHANNELS, 128)
    x[0, 0, 0] = -500.0  # window_range = 1000 >> R_ds
    x[0, 0, 1] = 500.0
    out = wrapper._prepare_input(x)

    expected = 2.0 * (1.0 - 0.5)  # 1.0 after clamp
    torch.testing.assert_close(
        out[0, BENDR_N_EEG_CHANNELS, :],
        torch.full((128,), expected),
        atol=1e-5,
        rtol=1e-5,
    )


def test_prepare_input_constant_input_no_nan(tiny_model):
    """A constant window (range=0) must not produce NaNs."""
    wrapper = _BendrAllTokensWrapper(tiny_model)
    x = torch.zeros(1, BENDR_N_EEG_CHANNELS, 64)
    out = wrapper._prepare_input(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Wrapper forward / build tests
# ---------------------------------------------------------------------------


def test_wrapper_cls_matches_original(tiny_model):
    """CLS token (position 0) from the wrapper matches BENDR.forward() on
    identical preprocessed input."""
    x = torch.randn(1, BENDR_N_EEG_CHANNELS, 512)

    wrapper = _BendrAllTokensWrapper(tiny_model)
    wrapper.eval()
    with torch.no_grad():
        prepared = wrapper._prepare_input(x)
        cls_original = tiny_model(prepared)
        all_tokens = wrapper(x)

    torch.testing.assert_close(all_tokens[:, 0, :], cls_original)


@pytest.mark.parametrize("n_times", [256, 512])
def test_build_from_scratch(n_times):
    """NtBendr.build() wraps and returns all tokens with correct shape."""
    cfg = NtBendr(kwargs=_TINY_BENDR_KWARGS)
    # n_chans passed here is for the *adapter output*, not the inner BENDR,
    # which is fixed at 20.  The test bypasses the adapter so we feed 19
    # channels directly.
    model = cfg.build(n_chans=BENDR_N_EEG_CHANNELS, n_times=n_times, n_outputs=_N_OUTPUTS)

    assert isinstance(model, _BendrAllTokensWrapper)
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(2, BENDR_N_EEG_CHANNELS, n_times))

    n_encoded = _encoder_output_length(_N_CHANS, n_times)
    assert out.shape == (2, n_encoded + 1, _ENCODER_H)


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
    """Pretrained build returns _BendrAllTokensWrapper with correct output shape."""
    cfg = NtBendr(
        from_pretrained_name=PRETRAINED_NAME,
        kwargs={"final_layer": False},
    )
    model = cfg.build()

    assert isinstance(model, _BendrAllTokensWrapper)

    x = torch.randn(1, BENDR_N_EEG_CHANNELS, 512)
    with torch.no_grad():
        out = model(x)

    assert out.ndim == 3
    assert out.shape[0] == 1
    assert out.shape[2] == 512  # default encoder_h
