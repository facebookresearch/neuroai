# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .common import INVALID_POS_VALUE
from .labram import (
    NtLabram,
    _build_channel_remapping,
    _LabramChannelWrapper,
)

CH_NAMES = ["Fp1", "Fp2", "C3", "C4", "O1", "O2", "Fz", "Cz"]
PATCH_SIZE = 50

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
    """Stands in for a braindecode ``Labram``, recording what it receives."""

    def __init__(self, n_patchs: int = 2):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.patch_size = PATCH_SIZE
        patch_embed = torch.nn.Module()
        patch_embed.n_patchs = n_patchs  # type: ignore[assignment]
        self.patch_embed = torch.nn.Sequential(patch_embed)
        self.temporal_embedding = torch.nn.Parameter(torch.zeros(1, 16, 4))
        self.last_ch_names: list[str] | None = None
        self.last_x_shape: tuple[int, ...] | None = None

    def forward(self, x, ch_names=None, **kwargs):
        self.last_ch_names = ch_names
        self.last_x_shape = tuple(x.shape)
        return x.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)


def _make_wrapper(names, channel_mapping=None):
    inner = _RecordingModel()
    return _LabramChannelWrapper(inner, names, channel_mapping), inner


# ---------------------------------------------------------------------------
# _LabramChannelWrapper
# ---------------------------------------------------------------------------


@requires_labram_channel_order
def test_wrapper_filters_to_valid_channels():
    wrapper, inner = _make_wrapper(CH_NAMES)

    B, C, T, D = 2, len(CH_NAMES), 100, 2
    x = torch.randn(B, C, T)
    pos = torch.full((B, C, D), INVALID_POS_VALUE)
    pos[:, 0, :] = 0.5  # Fp1
    pos[:, 2, :] = 0.5  # C3
    pos[:, 4, :] = 0.5  # O1

    wrapper(x, pos)

    assert inner.last_ch_names == ["FP1", "C3", "O1"]
    assert inner.last_x_shape == (B, 3, T)


@requires_labram_channel_order
def test_wrapper_remapping_and_unknown_dropped():
    """Mapped names are remapped; names not in LABRAM_CHANNEL_ORDER are dropped.

    Filtering happens inside the wrapper rather than relying on
    braindecode, because braindecode>=1.5 hard-raises on unknown names
    instead of silently dropping them.
    """
    wrapper, inner = _make_wrapper(["Fp1", "C3", "WEIRD_CH"])

    wrapper(torch.randn(1, 3, 50), torch.rand(1, 3, 2))

    assert inner.last_ch_names == ["FP1", "C3"]
    assert inner.last_x_shape == (1, 2, 50)


@requires_labram_channel_order
def test_wrapper_drops_unmapped_egi_channels():
    """Regression test for HBN-style EGI E-channels mixed with mapped names.

    Mirrors the production failure mode on the HBN ``reaction_time`` task:
    most ``E*`` channels are absent from ``channel_mappings/labram.json``
    but still carry valid montage positions, so without explicit filtering
    they would reach braindecode and trigger
    ``ValueError: ch_names contains a name not in LABRAM_CHANNEL_ORDER``.
    """
    union = ["Fp1", "E5", "Cz", "E7", "O2"]
    wrapper, inner = _make_wrapper(union, {"Fp1": "FP1"})

    B, C, T, D = 1, len(union), 50, 2
    pos = torch.rand(B, C, D)
    wrapper(torch.randn(B, C, T), pos)

    assert inner.last_ch_names == ["FP1", "CZ", "O2"]
    assert inner.last_x_shape == (B, 3, T)


@requires_labram_channel_order
def test_wrapper_heterogeneous_batch_uses_intersection():
    wrapper, inner = _make_wrapper(CH_NAMES)

    B, C, T, D = 2, len(CH_NAMES), 50, 2
    pos = torch.full((B, C, D), INVALID_POS_VALUE)
    pos[0, 0, :] = 0.5  # sample 0: Fp1
    pos[0, 1, :] = 0.5  # sample 0: Fp2
    pos[1, 0, :] = 0.5  # sample 1: Fp1
    pos[1, 2, :] = 0.5  # sample 1: C3 (differs!)

    wrapper(torch.randn(B, C, T), pos)

    assert inner.last_ch_names == ["FP1"]
    assert inner.last_x_shape == (B, 1, T)


@requires_labram_channel_order
def test_wrapper_names_channels_per_call():
    """A montage other than the one built with is named at forward time."""
    wrapper, inner = _make_wrapper(CH_NAMES)
    other = ["T7", "T8", "Pz"]

    wrapper(torch.randn(1, 3, 50), torch.rand(1, 3, 2), ch_names=other)

    assert inner.last_ch_names == ["T7", "T8", "PZ"]
    assert inner.last_x_shape == (1, 3, 50)


@requires_labram_channel_order
def test_wrapper_rejects_width_it_cannot_name():
    wrapper, _ = _make_wrapper(CH_NAMES)

    with pytest.raises(ValueError, match="selects its channel embedding by name"):
        wrapper(torch.randn(1, 3, 50), torch.rand(1, 3, 2))


@requires_labram_channel_order
@pytest.mark.parametrize(
    "n_times, n_patches",
    [
        (25, 1),  # shorter than patch_size -> padded
        (125, 2),  # not divisible -> truncated
        (150, 3),  # exact multiple
        (1000, 20),  # more patches than the embedding holds -> interpolated
    ],
)
def test_wrapper_serves_any_window_length(n_times: int, n_patches: int):
    wrapper, inner = _make_wrapper(CH_NAMES)

    wrapper(
        torch.randn(1, len(CH_NAMES), n_times),
        torch.rand(1, len(CH_NAMES), 2),
    )

    assert inner.last_x_shape == (1, len(CH_NAMES), n_patches * PATCH_SIZE)
    assert inner.patch_embed[0].n_patchs == n_patches
    assert inner.temporal_embedding.shape[1] == n_patches + 1


# ---------------------------------------------------------------------------
# _build_channel_remapping
# ---------------------------------------------------------------------------


@requires_labram_channel_order
def test_build_channel_remapping():
    """Identity, case-insensitive, unknown-excluded, and explicit mapping.

    Case-2 (direct case-insensitive) matches map to LABRAM_CHANNEL_ORDER's
    canonical (uppercase) form, so the wrapper hands braindecode names that
    already match its internal table.
    """
    # Direct case-2 match: dataset-cased names map to LABRAM-cased canonical.
    remap, positionless = _build_channel_remapping(["Fp1", "Cz", "O2"])
    assert remap == {"Fp1": "FP1", "Cz": "CZ", "O2": "O2"}
    assert positionless == set()

    # Case-insensitive match still resolves, output is canonical.
    remap, positionless = _build_channel_remapping(["fp1", "cZ"])
    assert remap == {"fp1": "FP1", "cZ": "CZ"}
    assert positionless == set()

    # Unknown channels excluded
    remap, positionless = _build_channel_remapping(["Fp1", "NONEXISTENT"])
    assert remap == {"Fp1": "FP1"}
    assert positionless == set()

    # Explicit mapping overrides name matching; explicit entries are
    # treated as positionless because they typically denote channel
    # systems whose montage positions cannot be resolved.
    remap, positionless = _build_channel_remapping(
        ["E1", "Fp1", "Cz"], {"E1": "FP1", "Fp1": "FP2"}
    )
    assert remap == {"E1": "FP1", "Fp1": "FP2", "Cz": "CZ"}
    assert positionless == {"E1", "Fp1"}

    # Bipolar fallback channels are positionless.
    remap, positionless = _build_channel_remapping(["Fp1-F3", "Cz"])
    assert remap == {"Fp1-F3": "FP1", "Cz": "CZ"}
    assert positionless == {"Fp1-F3"}


# ---------------------------------------------------------------------------
# NtLabram.build
# ---------------------------------------------------------------------------


@requires_labram_channel_order
def test_build_with_chs_info_returns_wrapper():
    cfg = NtLabram()
    model = cfg.build(
        n_spatial_locations=len(CH_NAMES),
        n_temporal_samples=200,
        n_outputs=2,
        chs_info=[{"ch_name": n} for n in CH_NAMES],
    )

    assert isinstance(model, _LabramChannelWrapper)
    # Each CH_NAME is in LABRAM_CHANNEL_ORDER; the wrapper resolves the
    # LABRAM-cased canonical for every union channel.
    assert model._resolve(CH_NAMES)[0] == [n.upper() for n in CH_NAMES]


def test_build_without_chs_info_returns_raw_model():
    cfg = NtLabram()
    model = cfg.build(n_spatial_locations=8, n_temporal_samples=200, n_outputs=2)

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


@pytest.fixture(scope="module")
def pretrained_wrapper() -> _LabramChannelWrapper:
    model = NtLabram(from_pretrained_name=PRETRAINED_NAME).build(
        n_spatial_locations=len(CH_NAMES),
        n_temporal_samples=800,
        n_outputs=None,
        chs_info=[{"ch_name": n} for n in CH_NAMES],
    )
    assert isinstance(model, _LabramChannelWrapper)
    return model


@requires_pretrained
@pytest.mark.parametrize(
    "n_times, n_patches",
    [
        (100, 1),  # shorter than patch_size -> padded
        (500, 2),  # not divisible -> truncated
        (800, 4),  # exact multiple
        (3000, 15),  # matches pretraining
        (6000, 30),  # longer than pretraining -> interpolated embedding
    ],
)
def test_pretrained_forward(
    pretrained_wrapper: _LabramChannelWrapper, n_times: int, n_patches: int
):
    """One instance covers every window length, at a channel subset."""
    x = torch.randn(1, len(CH_NAMES), n_times)
    pos = torch.rand(1, len(CH_NAMES), 3)

    with torch.no_grad():
        out = pretrained_wrapper(x, pos)

    # (B, n_channels * n_patches + [CLS], embed_dim), return_all_tokens=True
    assert out.shape[:2] == (1, len(CH_NAMES) * n_patches + 1)


@requires_pretrained
def test_pretrained_forward_other_montage(pretrained_wrapper: _LabramChannelWrapper):
    """The same instance also serves a montage it was not built with."""
    other = ["T7", "T8", "Pz", "NOT_AN_ELECTRODE"]

    with torch.no_grad():
        out = pretrained_wrapper(
            torch.randn(1, len(other), 400),
            torch.rand(1, len(other), 3),
            ch_names=other,
        )

    assert out.shape[:2] == (1, 3 * 2 + 1)  # the unknown name is dropped
