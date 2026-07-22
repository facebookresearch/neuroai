# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import random
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from neuralbench.utils import SequenceLabelEncoder
from neuralset import utils as ns_utils
from neuralset.events import etypes
from neuralset.extractors.meta import CroppedExtractor

from .data import Data
from .utils import (
    _compute_regression_bin_weights,
    detect_batch_dim,
    make_regression_bin_sampler,
    make_weighted_sampler,
    run_probe_hook,
    seed_worker,
)

# ---------------------------------------------------------------------------
# Regression-bin sampler
# ---------------------------------------------------------------------------

_BMAE_EDGES = (0.0, 40.0, 90.0, 300.0, 600.0)


def test_compute_regression_bin_weights_inverse_frequency():
    """Each populated bin contributes equal mass; inside a bin, weights are equal."""
    targets = torch.tensor(
        [
            5.0,
            10.0,  # bin 0: [0, 40), count = 2
            50.0,  # bin 1: [40, 90), count = 1
            100.0,
            200.0,  # bin 2: [90, 300), count = 2
            400.0,
            580.0,
            590.0,  # bin 3: [300, 600], count = 3
        ]
    )
    weights = _compute_regression_bin_weights(targets, _BMAE_EDGES)

    assert weights.shape == targets.shape
    expected = torch.tensor([1 / 2, 1 / 2, 1 / 1, 1 / 2, 1 / 2, 1 / 3, 1 / 3, 1 / 3])
    assert torch.allclose(weights, expected)
    # Total mass equals the number of populated bins.
    assert torch.isclose(weights.sum(), torch.tensor(4.0))


def test_compute_regression_bin_weights_includes_upper_boundary_in_last_bin():
    """A target equal to the last upper edge belongs to the last bin (matches BinnedMAE)."""
    targets = torch.tensor([100.0, 600.0])
    weights = _compute_regression_bin_weights(targets, _BMAE_EDGES)
    # 100 alone in bin 2, 600 alone in bin 3 -> each weight is 1.0
    assert torch.allclose(weights, torch.tensor([1.0, 1.0]))


def test_compute_regression_bin_weights_handles_empty_bins():
    """Empty bins cause no crash; populated bins still get inverse-frequency weights."""
    targets = torch.tensor([10.0, 15.0, 400.0, 500.0])  # only bins 0 and 3 populated
    weights = _compute_regression_bin_weights(targets, _BMAE_EDGES)
    assert torch.allclose(weights, torch.tensor([0.5, 0.5, 0.5, 0.5]))
    assert torch.isclose(weights.sum(), torch.tensor(2.0))


def test_compute_regression_bin_weights_zeros_out_of_range():
    """Targets outside [bin_edges[0], bin_edges[-1]] get zero weight (matches BinnedMAE)."""
    # 100 in bin 2; -5 below first edge; 700 / 1000 above last edge.
    targets = torch.tensor([-5.0, 100.0, 700.0, 1_000.0])
    weights = _compute_regression_bin_weights(targets, _BMAE_EDGES)
    assert torch.allclose(weights, torch.tensor([0.0, 1.0, 0.0, 0.0]))
    # Out-of-range targets do not contribute to any bin's count.
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_compute_regression_bin_weights_rejects_non_1d_targets():
    """Trailing singleton dims must be squeezed upstream; the helper enforces 1-D input."""
    with pytest.raises(ValueError, match="1-D targets"):
        _compute_regression_bin_weights(torch.zeros(4, 1), _BMAE_EDGES)


def test_compute_regression_bin_weights_rejects_short_edges():
    """A degenerate single edge cannot define any bin."""
    with pytest.raises(ValueError, match=">= 2"):
        _compute_regression_bin_weights(torch.zeros(4), [0.0])


def test_make_regression_bin_sampler_returns_weighted_sampler(mocker):
    """Factory wires `get_targets_from_dataset` -> weights -> WeightedRandomSampler."""
    targets = torch.tensor([5.0, 50.0, 100.0, 400.0, 580.0])
    mocker.patch("neuralbench.utils.get_targets_from_dataset", return_value=targets)
    sampler = make_regression_bin_sampler(
        mocker.MagicMock(), bin_edges=_BMAE_EDGES, logger=logging.getLogger("test")
    )

    assert isinstance(sampler, torch.utils.data.WeightedRandomSampler)
    assert sampler.replacement is True
    assert sampler.num_samples == len(targets)
    # Pin the numerical wiring: passing the wrong tensor downstream would
    # produce a shape-correct sampler with the wrong weights.  ``sampler.weights``
    # is a Python list (float64 when re-tensored), so cast for the comparison.
    expected = _compute_regression_bin_weights(targets, _BMAE_EDGES)
    assert torch.allclose(torch.as_tensor(sampler.weights, dtype=torch.float32), expected)


def test_make_regression_bin_sampler_squeezes_trailing_singleton(mocker):
    """A ``(N, 1)`` target tensor is squeezed to ``(N,)`` before binning."""
    targets_2d = torch.tensor([[5.0], [50.0], [100.0], [400.0], [580.0]])
    mocker.patch("neuralbench.utils.get_targets_from_dataset", return_value=targets_2d)
    sampler = make_regression_bin_sampler(
        mocker.MagicMock(), bin_edges=_BMAE_EDGES, logger=logging.getLogger("test")
    )

    expected = _compute_regression_bin_weights(targets_2d.squeeze(-1), _BMAE_EDGES)
    assert torch.allclose(torch.as_tensor(sampler.weights, dtype=torch.float32), expected)


def test_make_regression_bin_sampler_balances_bins(mocker):
    """Drawing from the sampler yields ~equal counts across populated bins."""
    targets = torch.cat(
        [
            torch.full((10,), 10.0),  # bin 0
            torch.full((50,), 50.0),  # bin 1
            torch.full((200,), 100.0),  # bin 2
            torch.full((1_000,), 400.0),  # bin 3
        ]
    )
    mocker.patch("neuralbench.utils.get_targets_from_dataset", return_value=targets)
    sampler = make_regression_bin_sampler(
        mocker.MagicMock(), bin_edges=_BMAE_EDGES, logger=logging.getLogger("test")
    )

    weights_t = torch.as_tensor(sampler.weights)
    generator = torch.Generator().manual_seed(0)
    drawn_idx = torch.multinomial(
        weights_t, num_samples=100_000, replacement=True, generator=generator
    )

    inner_edges = torch.tensor(_BMAE_EDGES[1:-1])
    drawn_bins = torch.bucketize(targets[drawn_idx], inner_edges, right=False).clamp_(
        0, 3
    )
    counts = torch.bincount(drawn_bins, minlength=4).float()
    proportions = counts / counts.sum()
    assert torch.allclose(proportions, torch.full((4,), 0.25), atol=0.01)


# --- SequenceLabelEncoder -------------------------------------------------
#
# The CTC head's target stream comes from a fixed-length integer-label
# extractor that lives in ``neuralbench`` (not ``neuralset``). We keep
# this CTC-specific shape out of the base ``LabelEncoder`` and read the
# pre-computed ``label`` field that the emg/typing study writes onto
# each Keystroke event.

_KS_PAD = 27  # blank index for the toy 27-class vocab below


@pytest.fixture
def _fresh_warn_registry():
    """warn_once dedupes per-process; reset so per-test assertions are stable."""
    ns_utils.ISSUED_WARNINGS.clear()
    yield
    ns_utils.ISSUED_WARNINGS.clear()


def _ks_events(labels: list[int], starts: list[float] | None = None):
    """Build Keystroke events with a pre-computed integer ``label`` in extras."""
    starts = starts or [0.1 * i for i in range(len(labels))]
    return [
        etypes.Keystroke(
            start=s,
            duration=0.05,
            text=f"k{lbl}",
            timeline="t",
            extra={"label": lbl},
        )
        for s, lbl in zip(starts, labels, strict=False)
    ]


@pytest.mark.parametrize(
    ("labels", "starts", "win_start", "win_dur", "expected"),
    [
        # Every event fits the window.
        ([7, 8, 26], None, 0.0, 1.0, [7, 8, 26]),
        # Window [10.9, 14.9) keeps c@11, d@14.5; the rest fall outside.
        (list(range(5)), [10.0, 10.5, 11.0, 14.5, 14.95], 10.9, 4.0, [2, 3]),
    ],
)
def test_sequence_label_encoder_padded_layout(
    labels, starts, win_start, win_dur, expected
):
    """``SequenceLabelEncoder`` produces a fixed-shape tensor of concatenated
    integer labels, right-padded with ``pad_value`` (the CTC blank)."""
    ext = SequenceLabelEncoder(
        event_types="Keystroke",
        event_field="label",
        allow_missing=True,
        max_length=8,
        pad_value=_KS_PAD,
    )
    events = _ks_events(labels, starts)
    out = ext(events, start=win_start, duration=win_dur)

    n = len(expected)
    assert out.shape == (8,)
    assert out[:n].tolist() == expected
    assert (out[n:] == _KS_PAD).all()


def test_cropped_sequence_label_encoder_composition():
    """``CroppedExtractor`` wrapping ``SequenceLabelEncoder`` restricts label
    collection to the cropped sub-window; ``model_factory`` unwraps the
    ``.extractor`` chain to read ``n_classes`` off the inner encoder and
    size the CTC head through the composition."""
    inner = SequenceLabelEncoder(
        event_types="Keystroke",
        event_field="label",
        allow_missing=True,
        max_length=8,
        pad_value=_KS_PAD,
    )
    # Outer crop [start+0.9, start+0.9+4.0) — for start=10.0 keeps c@11, d@14.5.
    cropped = CroppedExtractor(extractor=inner, offset=0.9, duration=4.0)
    events = _ks_events(list(range(5)), [10.0, 10.5, 11.0, 14.5, 14.95])
    out = cropped(events, start=10.0, duration=5.0)

    assert out.shape == (8,)
    assert out[:2].tolist() == [2, 3]
    assert (out[2:] == _KS_PAD).all()
    # ``CroppedExtractor`` itself doesn't carry ``n_classes``; the head
    # width lives on the wrapped encoder, which ``model_factory`` reaches
    # by unwrapping ``.extractor``.
    assert not hasattr(cropped, "n_classes")
    assert cropped.extractor.n_classes == _KS_PAD + 1


def test_sequence_label_encoder_truncation_warns_once(_fresh_warn_registry):
    """Truncation of overlong segments warns once per process."""
    ext = SequenceLabelEncoder(
        event_types="Keystroke",
        event_field="label",
        allow_missing=True,
        max_length=2,
        pad_value=_KS_PAD,
    )
    events = _ks_events([7, 4, 11, 11, 14])
    with pytest.warns(UserWarning, match="truncating") as records:
        ext(events, start=0.0, duration=1.0)
        ext(events, start=0.0, duration=1.0)
    assert sum("truncating" in str(r.message) for r in records) == 1


def test_sequence_label_encoder_empty_segment():
    """Segments without matching events return all-blank padding."""
    ext = SequenceLabelEncoder(
        event_types="Keystroke",
        event_field="label",
        allow_missing=True,
        max_length=4,
        pad_value=_KS_PAD,
    )
    events = _ks_events([7, 4])
    # Target window 10..11 excludes both events at t=0.0, t=0.1.
    out = ext(events, start=10.0, duration=1.0)
    assert out.shape == (4,)
    assert (out == _KS_PAD).all()


def test_sequence_label_encoder_n_classes_property():
    """``n_classes`` is the CTC head width: ``pad_value + 1``."""
    ext = SequenceLabelEncoder(
        event_types="Keystroke",
        event_field="label",
        allow_missing=True,
        max_length=4,
        pad_value=98,
    )
    assert ext.n_classes == 99


# ---------------------------------------------------------------------------
# seed_worker
# ---------------------------------------------------------------------------


def test_seed_worker_seeds_numpy_and_random(mocker) -> None:
    """seed_worker must reseed numpy and Python random from torch's per-worker seed."""
    mocker.patch.object(
        torch.utils.data, "get_worker_info", return_value=SimpleNamespace(seed=42)
    )

    np.random.seed(0)
    random.seed(0)
    seed_worker(worker_id=0)
    after_np = np.random.rand(3)
    after_py = [random.random() for _ in range(3)]

    np.random.seed(0)
    random.seed(0)
    seed_worker(worker_id=0)
    again_np = np.random.rand(3)
    again_py = [random.random() for _ in range(3)]

    assert np.allclose(after_np, again_np)
    assert after_py == again_py

    np.random.seed(0)
    random.seed(0)
    mocker.patch.object(
        torch.utils.data, "get_worker_info", return_value=SimpleNamespace(seed=999)
    )
    seed_worker(worker_id=0)
    different_np = np.random.rand(3)
    assert not np.allclose(after_np, different_np)


def test_seed_worker_raises_when_called_outside_worker(mocker) -> None:
    """seed_worker should fail loudly when called outside a DataLoader worker."""
    mocker.patch.object(torch.utils.data, "get_worker_info", return_value=None)
    with pytest.raises(AssertionError):
        seed_worker(worker_id=0)


# ---------------------------------------------------------------------------
# make_weighted_sampler
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def train_segment_dataset(build_data: Callable[..., Data]):
    """Real ``SegmentDataset`` over the ``Test2024Eeg`` synthetic study.

    Built once per module so each sampler test reuses the same dataset and
    avoids re-running the (already memoised) ``Study.run`` pipeline.  Using
    a real dataset means ``compute_class_weights_from_dataset`` runs end-to-
    end -- no stubs, no mocks -- and the tests exercise the same code path
    as production.
    """
    return build_data(seed=0).prepare()["train"].dataset


def test_make_weighted_sampler_with_generator_is_deterministic(
    train_segment_dataset,
) -> None:
    """Two samplers built with the same generator-seed must draw the same indices."""
    sampler_a = make_weighted_sampler(
        train_segment_dataset,
        logger=logging.getLogger("t"),
        generator=torch.Generator().manual_seed(7),
    )
    sampler_b = make_weighted_sampler(
        train_segment_dataset,
        logger=logging.getLogger("t"),
        generator=torch.Generator().manual_seed(7),
    )

    assert list(iter(sampler_a)) == list(iter(sampler_b))


def test_make_weighted_sampler_with_different_generators_diverges(
    train_segment_dataset,
) -> None:
    """Different generator seeds must produce different index sequences."""
    indices_7 = list(
        iter(
            make_weighted_sampler(
                train_segment_dataset,
                logger=logging.getLogger("t"),
                generator=torch.Generator().manual_seed(7),
            )
        )
    )
    indices_8 = list(
        iter(
            make_weighted_sampler(
                train_segment_dataset,
                logger=logging.getLogger("t"),
                generator=torch.Generator().manual_seed(8),
            )
        )
    )

    assert indices_7 != indices_8


def test_make_weighted_sampler_without_generator_follows_global_rng(
    train_segment_dataset,
) -> None:
    """Backward compat: with ``generator=None`` the sampler follows the global RNG."""
    torch.manual_seed(123)
    sampler_a = make_weighted_sampler(
        train_segment_dataset, logger=logging.getLogger("t")
    )
    indices_a = list(iter(sampler_a))

    torch.manual_seed(123)
    sampler_b = make_weighted_sampler(
        train_segment_dataset, logger=logging.getLogger("t")
    )
    indices_b = list(iter(sampler_b))

    assert indices_a == indices_b


# ---------------------------------------------------------------------------
# run_probe_hook / detect_batch_dim (probe mechanics, exercised in isolation)
# ---------------------------------------------------------------------------


class _SeqFirstEnc(nn.Module):
    """Emits sequence-first (T, B, D), like a ``batch_first=False`` transformer."""

    def __init__(self, n_in: int, emb: int):
        super().__init__()
        self.lin = nn.Linear(n_in, emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).transpose(0, 1)  # (B, T, F) -> (T, B, D)


class _SeqFirstProbeNet(nn.Module):
    """Probed submodule ``enc`` emits sequence-first (T, B, D)."""

    def __init__(self, n_in: int, emb: int):
        super().__init__()
        self.enc = _SeqFirstEnc(n_in, emb)
        self.head = nn.Linear(emb, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x).mean(0))


class _Const(nn.Module):
    """Emits a fixed-size tensor, independent of the input batch size."""

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(3, 6)


class _BatchInvariantNet(nn.Module):
    """Probed submodule ``const`` emits a fixed-size tensor, ignoring batch size."""

    def __init__(self, n_in: int):
        super().__init__()
        self.const = _Const()
        self.lin = nn.Linear(n_in, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.const(x)  # fire the hook with a batch-independent output
        return self.lin(x)


@pytest.mark.parametrize(
    "model, probe_layer, batch, expected_shape, expected_dim",
    [
        # batch-first (B, D) capture from a Sequential -> batch axis 0
        (
            nn.Sequential(nn.Linear(10, 16), nn.Linear(16, 4)),
            "0",
            {"input": torch.randn(4, 10)},
            (4, 16),
            0,
        ),
        # sequence-first (T, B, D) capture -> batch axis 1 (auto-detected)
        (
            _SeqFirstProbeNet(10, 6),
            "enc",
            {"x": torch.randn(4, 5, 10)},
            (5, 4, 6),
            1,
        ),
    ],
)
def test_run_probe_hook_and_detect_batch_dim(
    model, probe_layer, batch, expected_shape, expected_dim
):
    """run_probe_hook returns the submodule output; detect_batch_dim finds its batch axis."""
    submodule = model.get_submodule(probe_layer)
    assert run_probe_hook(model, submodule, batch, probe_layer).shape == expected_shape
    assert detect_batch_dim(model, submodule, batch, probe_layer) == expected_dim


def test_run_probe_hook_rejects_unreachable_and_non_tensor():
    # A submodule never reached during forward -> RuntimeError.
    model = nn.Sequential(nn.Linear(10, 8), nn.Linear(8, 4))
    detached = nn.Linear(4, 4)  # not part of model's forward graph
    with pytest.raises(RuntimeError, match="did not fire"):
        run_probe_hook(model, detached, {"input": torch.randn(2, 10)}, "detached")
    # A tuple-returning submodule (nn.RNN -> (output, h_n)) -> TypeError.
    rnn = nn.RNN(input_size=10, hidden_size=8, batch_first=True)
    with pytest.raises(TypeError, match="tensor-returning"):
        run_probe_hook(rnn, rnn, {"input": torch.randn(2, 5, 10)}, "")


def test_detect_batch_dim_raises_when_no_axis_scales():
    # A capture independent of batch size yields zero candidate axes.
    model = _BatchInvariantNet(10)
    with pytest.raises(ValueError, match="batch axis is ambiguous"):
        detect_batch_dim(model, model.const, {"x": torch.randn(4, 10)}, "const")
