# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .common import INVALID_POS_VALUE
from .labram import NtLabram, _LabramChannelWrapper

CH_NAMES = ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "Fz", "Cz"]

try:
    from braindecode.models.labram import (  # noqa: F401  # pylint: disable=unused-import
        LABRAM_CHANNEL_ORDER,
    )

    _has_labram_channel_order = True
except ImportError:
    _has_labram_channel_order = False

requires_labram_channel_order = pytest.mark.skipif(
    not _has_labram_channel_order,
    reason="braindecode too old: LABRAM_CHANNEL_ORDER not available",
)


class _RecordingModel(torch.nn.Module):
    """Thin nn.Module that records the ``ch_names`` and input shape it receives."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.last_ch_names: list[str] | None = None
        self.last_x_shape: tuple[int, ...] | None = None

    def forward(self, x, ch_names=None, **kwargs):
        self.last_ch_names = ch_names
        self.last_x_shape = tuple(x.shape)
        return x.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)


def _make_wrapper(names, remap=None):
    if remap is None:
        remap = {n: n for n in names}
    inner = _RecordingModel()
    return _LabramChannelWrapper(inner, names, remap), inner


# ---------------------------------------------------------------------------
# _LabramChannelWrapper
# ---------------------------------------------------------------------------


def test_wrapper_filters_to_valid_channels():
    wrapper, inner = _make_wrapper(CH_NAMES)

    B, C, T, D = 2, len(CH_NAMES), 100, 2
    x = torch.randn(B, C, T)
    pos = torch.full((B, C, D), INVALID_POS_VALUE)
    pos[:, 0, :] = 0.5  # Fp1
    pos[:, 2, :] = 0.5  # C3
    pos[:, 4, :] = 0.5  # O1

    wrapper(x, pos)

    assert inner.last_ch_names == ["Fp1", "C3", "O1"]
    assert inner.last_x_shape == (B, 3, T)


def test_wrapper_remapping_and_passthrough():
    """Mapped names are remapped; unmapped names pass through as-is."""
    remap = {"Fp1": "FP1", "C3": "C3"}
    wrapper, inner = _make_wrapper(["Fp1", "C3", "WEIRD_CH"], remap)

    wrapper(torch.randn(1, 3, 50), torch.rand(1, 3, 2))

    assert inner.last_ch_names == ["FP1", "C3", "WEIRD_CH"]


def test_wrapper_heterogeneous_batch_uses_intersection():
    wrapper, inner = _make_wrapper(CH_NAMES)

    B, C, T, D = 2, len(CH_NAMES), 50, 2
    pos = torch.full((B, C, D), INVALID_POS_VALUE)
    pos[0, 0, :] = 0.5  # sample 0: Fp1
    pos[0, 1, :] = 0.5  # sample 0: Fp2
    pos[1, 0, :] = 0.5  # sample 1: Fp1
    pos[1, 2, :] = 0.5  # sample 1: C3 (differs!)

    wrapper(torch.randn(B, C, T), pos)

    assert inner.last_ch_names == ["Fp1"]
    assert inner.last_x_shape == (B, 1, T)


# ---------------------------------------------------------------------------
# NtLabram._build_channel_remapping
# ---------------------------------------------------------------------------


@requires_labram_channel_order
def test_build_channel_remapping():
    """Identity, case-insensitive, unknown-excluded, and explicit mapping.

    Case-2 (direct case-insensitive) matches map to LABRAM_CHANNEL_ORDER's
    canonical (uppercase) form, so the wrapper hands braindecode names that
    already match its internal table.
    """
    # Direct case-2 match: dataset-cased names map to LABRAM-cased canonical.
    remap, positionless = NtLabram()._build_channel_remapping(["Fp1", "Cz", "O2"])
    assert remap == {"Fp1": "FP1", "Cz": "CZ", "O2": "O2"}
    assert positionless == set()

    # Case-insensitive match still resolves, output is canonical.
    remap, positionless = NtLabram()._build_channel_remapping(["fp1", "cZ"])
    assert remap == {"fp1": "FP1", "cZ": "CZ"}
    assert positionless == set()

    # Unknown channels excluded
    remap, positionless = NtLabram()._build_channel_remapping(["Fp1", "NONEXISTENT"])
    assert remap == {"Fp1": "FP1"}
    assert positionless == set()

    # Explicit mapping overrides name matching; explicit entries are
    # treated as positionless because they typically denote channel
    # systems whose montage positions cannot be resolved.
    cfg = NtLabram(channel_mapping={"E1": "FP1", "Fp1": "FP2"})
    remap, positionless = cfg._build_channel_remapping(["E1", "Fp1", "Cz"])
    assert remap == {"E1": "FP1", "Fp1": "FP2", "Cz": "CZ"}
    assert positionless == {"E1", "Fp1"}

    # Bipolar fallback channels are positionless.
    remap, positionless = NtLabram()._build_channel_remapping(["Fp1-F3", "Cz"])
    assert remap == {"Fp1-F3": "FP1", "Cz": "CZ"}
    assert positionless == {"Fp1-F3"}


# ---------------------------------------------------------------------------
# NtLabram.build
# ---------------------------------------------------------------------------


@requires_labram_channel_order
def test_build_with_chs_info_returns_wrapper():
    cfg = NtLabram()
    chs_info = [{"ch_name": n} for n in CH_NAMES]
    model = cfg.build(n_chans=len(CH_NAMES), n_times=200, n_outputs=2, chs_info=chs_info)

    assert isinstance(model, _LabramChannelWrapper)
    # Each CH_NAME is in LABRAM_CHANNEL_ORDER; the wrapper stores the
    # LABRAM-cased canonical for every union channel.
    assert model._labram_names == [n.upper() for n in CH_NAMES]


def test_build_without_chs_info_returns_raw_model():
    cfg = NtLabram()
    model = cfg.build(n_chans=8, n_times=200, n_outputs=2)

    assert not isinstance(model, _LabramChannelWrapper)


# ---------------------------------------------------------------------------
# Pretrained model integration tests
# ---------------------------------------------------------------------------

PRETRAINED_NAME = "braindecode/labram-pretrained"


def _pretrained_weights_available() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.utils import EntryNotFoundError

        result = try_to_load_from_cache(PRETRAINED_NAME, "model.safetensors")
        return isinstance(result, str)
    except (ImportError, EntryNotFoundError):
        return False


requires_pretrained = pytest.mark.skipif(
    not _pretrained_weights_available(),
    reason=f"Pretrained weights for {PRETRAINED_NAME!r} not cached locally",
)


@requires_pretrained
@pytest.mark.parametrize("n_times", [3000, 800])
def test_pretrained_forward(n_times: int):
    """Load pretrained weights, forward with a channel subset."""
    subset = ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "Fz", "Cz"]
    chs_info = [{"ch_name": n} for n in subset]

    cfg = NtLabram(from_pretrained_name=PRETRAINED_NAME)
    model = cfg.build(n_times=n_times, chs_info=chs_info)

    assert isinstance(model, _LabramChannelWrapper)

    x = torch.randn(1, len(subset), n_times)
    pos = torch.rand(1, len(subset), 3)

    with torch.no_grad():
        out = model(x, pos)

    assert out.shape[0] == 1
    assert out.ndim == 3  # (B, n_tokens, embed_dim) with return_all_tokens=True


@requires_pretrained
@pytest.mark.parametrize(
    "n_times, expected_patches, pad_right, truncate_right",
    [
        (100, 1, 100, 0),  # shorter than patch_size -> pad
        (500, 2, 0, 100),  # not divisible -> truncate
        (800, 4, 0, 0),  # exact multiple -> no adjustment
        (3000, 15, 0, 0),  # matches pretrained -> no-op
        (6000, 30, 0, 0),  # longer -> interpolation
    ],
)
def test_adapt_pretrained_dimensions(
    n_times: int,
    expected_patches: int,
    pad_right: int,
    truncate_right: int,
):
    """Temporal adaptation: padding, truncation, and embedding interpolation."""
    cfg = NtLabram(from_pretrained_name=PRETRAINED_NAME)
    chs_info = [{"ch_name": n} for n in CH_NAMES]

    model = cfg.build(n_times=n_times, chs_info=chs_info)
    assert isinstance(model, _LabramChannelWrapper)
    assert model._pad_right == pad_right
    assert model._truncate_right == truncate_right
    assert model.model.patch_embed[0].n_patchs == expected_patches  # type: ignore[index,union-attr]
    # cls token + patch tokens
    assert model.model.temporal_embedding.shape[1] == expected_patches + 1  # type: ignore[index]

    x = torch.randn(1, len(CH_NAMES), n_times)
    pos = torch.rand(1, len(CH_NAMES), 3)
    with torch.no_grad():
        out = model(x, pos)
    assert out.shape[0] == 1
    assert out.ndim == 3
