# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types

import braindecode.models
import pytest
import torch

from .reve import NtReve, _ReveWrapper

BANK_CH_NAMES = ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "Fz", "Cz"]
PRETRAINED_NAME = "brain-bzh/reve-base"

_EMBED_DIM = 16
_N_TIMES = 100
_PATCH_SIZE = 50
_PATCH_OVERLAP = 10
_N_OUTPUTS = 2
_N_PATCHES = 2  # unfold(size=50, step=40) on 100 samples gives 2 patches

_TINY_REVE_KWARGS = dict(
    embed_dim=_EMBED_DIM,
    depth=1,
    heads=2,
    head_dim=8,
    patch_size=_PATCH_SIZE,
    patch_overlap=_PATCH_OVERLAP,
)


def _make_reve(ch_names: list[str], n_outputs: int = _N_OUTPUTS):
    """Build a tiny from-scratch REVE model for the given channel names."""
    return braindecode.models.REVE(
        n_outputs=n_outputs,
        n_chans=len(ch_names),
        n_times=_N_TIMES,
        chs_info=[{"ch_name": n} for n in ch_names],
        **_TINY_REVE_KWARGS,
    )


# ---------------------------------------------------------------------------
# _ReveWrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoder_only", [False, True])
def test_wrapper_forward(encoder_only):
    """Forward with and without channel subsetting, in both modes."""
    ch_all = ["Fp1", "Fp2", "C3", "C4"]
    ch_sub = ["Fp1", "C3"] if not encoder_only else ["Fp1", "C4"]

    wrapper_plain = _ReveWrapper(_make_reve(ch_all), encoder_only=encoder_only)
    wrapper_subset = _ReveWrapper(
        _make_reve(ch_sub),
        channel_indices=[0, 2] if not encoder_only else [0, 3],
        encoder_only=encoder_only,
    )

    with torch.no_grad():
        out = wrapper_plain(torch.randn(2, 4, _N_TIMES))
        out_sub = wrapper_subset(torch.randn(1, 4, _N_TIMES))

    if encoder_only:
        assert out.shape == (2, len(ch_all) * _N_PATCHES, _EMBED_DIM)
        assert out_sub.shape == (1, len(ch_sub) * _N_PATCHES, _EMBED_DIM)
    else:
        assert out.shape == (2, _N_OUTPUTS)
        assert out_sub.shape == (1, _N_OUTPUTS)


# ---------------------------------------------------------------------------
# NtReve._remap_chs_info
# ---------------------------------------------------------------------------


def test_remap_chs_info():
    """No-op without mapping; applies mapping when provided."""
    cfg_plain = NtReve()
    chs = [{"ch_name": "C3"}, {"ch_name": "C4"}]
    assert cfg_plain._remap_chs_info(chs) is chs

    cfg_mapped = NtReve(channel_mapping={"EEG 001": "Fp1", "EEG 002": "Fp2"})
    result = cfg_mapped._remap_chs_info(
        [{"ch_name": "EEG 001"}, {"ch_name": "EEG 002"}, {"ch_name": "Cz"}]
    )
    assert [ch["ch_name"] for ch in result] == ["Fp1", "Fp2", "Cz"]


# ---------------------------------------------------------------------------
# NtReve._derive_bipolar_position (anode fallback)
# ---------------------------------------------------------------------------


def _make_mock_bank(names: list[str]):
    """Create a minimal mock position bank with known embeddings."""
    bank = types.SimpleNamespace()
    bank.mapping = {n: i for i, n in enumerate(names)}
    bank.embedding = torch.randn(len(names), 3)
    return bank


@pytest.mark.parametrize(
    "ch_name, bank_names, expect_result",
    [
        ("Fp1-F3", ["Fp1", "F3", "Cz"], True),
        ("Fp1-F3", ["F3", "Cz"], False),
        ("Fp1", ["Fp1", "Cz"], False),
        ("BOGUS", ["Fp1", "Cz"], False),
    ],
)
def test_derive_bipolar_position(ch_name, bank_names, expect_result):
    """Bipolar lookup returns anode position when available, None otherwise."""
    bank = _make_mock_bank(bank_names)
    result = NtReve._derive_bipolar_position(ch_name, bank)
    if expect_result:
        anode = ch_name.split("-")[0]
        assert result is not None
        assert torch.equal(result, bank.embedding[bank.mapping[anode]])
    else:
        assert result is None


# ---------------------------------------------------------------------------
# NtReve.build (with pretrained weights)
# ---------------------------------------------------------------------------


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
    """Build returns a wrapper in both classification and encoder-only modes."""
    chs_info = [{"ch_name": n} for n in BANK_CH_NAMES]
    cfg = NtReve(from_pretrained_name=PRETRAINED_NAME)

    model_cls = cfg.build(
        n_chans=len(BANK_CH_NAMES), n_times=400, n_outputs=2, chs_info=chs_info
    )
    assert isinstance(model_cls, _ReveWrapper)
    assert model_cls.encoder_only is False
    assert model_cls.channel_indices is None

    model_enc = cfg.build(n_chans=len(BANK_CH_NAMES), n_times=400, chs_info=chs_info)
    assert isinstance(model_enc, _ReveWrapper)
    assert model_enc.encoder_only is True

    eeg = torch.randn(1, len(BANK_CH_NAMES), 400)
    with torch.no_grad():
        out = model_enc(eeg)
    assert out.ndim == 3
    assert out.shape[0] == 1


@requires_pretrained
def test_build_drops_unknown_channels():
    """Channels not in the position bank are dropped and channel_indices is set."""
    known = ["Fp1", "C3"]
    unknown = ["BOGUS_CH"]
    chs_info = [{"ch_name": n} for n in known + unknown]
    cfg = NtReve(from_pretrained_name=PRETRAINED_NAME)
    model = cfg.build(n_chans=len(chs_info), n_times=400, chs_info=chs_info)

    assert isinstance(model, _ReveWrapper)
    assert model.channel_indices is not None
    assert model.channel_indices.tolist() == [0, 1]
