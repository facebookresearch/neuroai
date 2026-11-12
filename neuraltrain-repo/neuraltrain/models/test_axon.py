# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import braindecode.models
import numpy as np
import pytest
import torch

from .axon import NtAxon, _AxonWrapper, _position_bank

if not hasattr(braindecode.models, "AXON"):
    pytest.skip("AXON requires braindecode >= 2.0", allow_module_level=True)

PRETRAINED_NAME = "MannasAI/axon-eeg"
BANK_CH_NAMES = ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "Fz", "Cz"]

_EMBED_DIM = 16
_N_TIMES = 400
_N_OUTPUTS = 3
_N_PATCHES = 2  # patch 200, stride 180 on 400 samples: floor((400 - 200) / 180) + 1

_TINY_AXON_KWARGS = dict(
    embed_dim=_EMBED_DIM,
    depth=1,
    num_heads=2,
    head_hidden_dim=8,
)


def _chs(names: list[str]) -> list[dict]:
    return [{"ch_name": n} for n in names]


def _build_tiny(names: list[str], n_outputs: int | None, **cfg_kwargs):
    cfg = NtAxon(kwargs=_TINY_AXON_KWARGS, **cfg_kwargs)
    return cfg.build(
        n_spatial_locations=len(names),
        n_temporal_samples=_N_TIMES,
        n_outputs=n_outputs,
        chs_info=_chs(names),
    )


# ---------------------------------------------------------------------------
# Position bank
# ---------------------------------------------------------------------------


def test_position_bank():
    """10-05 and EGI HydroCel names are present, in the head frame (metres)."""
    bank = _position_bank()
    for name in ("cz", "fp1", "tp9", "af7", "e1", "e128"):
        assert name in bank
    cz = bank["cz"]
    # Head frame (origin between the preauricular points): Cz is near the vertex.
    assert abs(cz[0]) < 0.01 and abs(cz[1]) < 0.04 and 0.10 < cz[2] < 0.17
    norms = np.linalg.norm(np.stack(list(bank.values())), axis=1)
    assert np.all(norms < 0.2)


# ---------------------------------------------------------------------------
# NtAxon._remap_chs_info
# ---------------------------------------------------------------------------


def test_remap_chs_info():
    """No-op without mapping; applies mapping when provided."""
    chs = _chs(["C3", "C4"])
    assert NtAxon()._remap_chs_info(chs) is chs

    cfg = NtAxon(channel_mapping={"EEG 001": "Fp1", "EEG 002": "Fp2"})
    result = cfg._remap_chs_info(_chs(["EEG 001", "EEG 002", "Cz"]))
    assert [ch["ch_name"] for ch in result] == ["Fp1", "Fp2", "Cz"]


# ---------------------------------------------------------------------------
# NtAxon.build (tiny, from scratch)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoder_only", [False, True])
def test_build_forward(encoder_only):
    """Classification head and encoder-only token grid have the expected shapes."""
    model = _build_tiny(BANK_CH_NAMES, None if encoder_only else _N_OUTPUTS)
    assert isinstance(model, _AxonWrapper)
    assert model.encoder_only is encoder_only
    assert model.channel_indices is None

    with torch.no_grad():
        out = model(torch.randn(2, len(BANK_CH_NAMES), _N_TIMES))
    if encoder_only:
        assert out.shape == (2, len(BANK_CH_NAMES), _N_PATCHES, _EMBED_DIM)
        assert isinstance(model.model.final_layer, torch.nn.Identity)
    else:
        assert out.shape == (2, _N_OUTPUTS)


def test_build_positions_follow_names():
    """Electrode positions passed to AXON are the bank's, case-insensitively."""
    names = ["cz", "EEG Fp1", "T3", "C3-C4"]
    model = _build_tiny(names, _N_OUTPUTS)
    bank = _position_bank()
    expected = np.stack(
        [bank["cz"], bank["fp1"], bank["t7"], (bank["c3"] + bank["c4"]) / 2]
    )
    locs = np.stack([ch["loc"][:3] for ch in model.model.chs_info])
    np.testing.assert_allclose(locs, expected, atol=1e-6)


def test_build_drops_unknown_channels():
    """Channels without a position are dropped and the input is sliced."""
    names = ["Fp1", "BOGUS_CH", "C3", "STI 014"]
    model = _build_tiny(names, None)
    assert model.channel_indices is not None
    assert model.channel_indices.tolist() == [0, 2]
    with torch.no_grad():
        out = model(torch.randn(1, len(names), _N_TIMES))
    assert out.shape == (1, 2, _N_PATCHES, _EMBED_DIM)


def test_build_channel_mapping():
    """``channel_mapping`` lets otherwise unknown names resolve."""
    names = ["EEG 001", "EEG 002"]
    model = _build_tiny(names, None, channel_mapping={"EEG 001": "Fp1", "EEG 002": "Fp2"})
    assert model.channel_indices is None
    assert [ch["ch_name"] for ch in model.model.chs_info] == ["Fp1", "Fp2"]


def test_build_hydrocel_names():
    """EGI HydroCel names (e.g. HBN) resolve without a mapping."""
    names = ["E1", "E2", "E55", "Cz"]
    model = _build_tiny(names, None)
    assert model.channel_indices is None


def test_build_errors():
    """Missing channel info or no resolvable channel raises."""
    cfg = NtAxon(kwargs=_TINY_AXON_KWARGS)
    with pytest.raises(ValueError, match="channel names"):
        cfg.build(n_spatial_locations=2, n_temporal_samples=_N_TIMES)
    with pytest.raises(ValueError, match="channel_mapping"):
        cfg.build(
            n_spatial_locations=2,
            n_temporal_samples=_N_TIMES,
            chs_info=_chs(["FOO", "BAR"]),
        )


# ---------------------------------------------------------------------------
# NtAxon.build (pretrained weights)
# ---------------------------------------------------------------------------


def _pretrained_weights_available() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(PRETRAINED_NAME, "model.safetensors")
        return isinstance(result, str)
    except Exception:
        return False


requires_pretrained = pytest.mark.skipif(
    not _pretrained_weights_available(),
    reason=f"Pretrained weights for {PRETRAINED_NAME!r} not cached locally",
)


@requires_pretrained
def test_build_pretrained():
    """Pretrained encoder-only build matches braindecode's own AXON.encode."""
    cfg = NtAxon(from_pretrained_name=PRETRAINED_NAME)
    model = cfg.build(
        n_spatial_locations=len(BANK_CH_NAMES),
        n_temporal_samples=_N_TIMES,
        n_outputs=None,
        chs_info=_chs(BANK_CH_NAMES),
    ).eval()
    reference = braindecode.models.AXON.from_pretrained(
        PRETRAINED_NAME,
        n_chans=len(BANK_CH_NAMES),
        n_times=_N_TIMES,
        n_outputs=2,
        chs_info=model.model.chs_info,
    ).eval()

    eeg = torch.randn(1, len(BANK_CH_NAMES), _N_TIMES)
    with torch.no_grad():
        out = model(eeg)
        ref = reference.encode(eeg)
    assert out.shape == (1, len(BANK_CH_NAMES), _N_PATCHES, 512)
    torch.testing.assert_close(out, ref)
