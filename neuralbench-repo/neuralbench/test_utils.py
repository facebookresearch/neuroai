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

from .data import Data
from .utils import (
    _compute_regression_bin_weights,
    make_regression_bin_sampler,
    make_weighted_sampler,
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
