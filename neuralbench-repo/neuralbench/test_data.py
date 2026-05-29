# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``neuralbench.data.Data`` RNG plumbing.

Uses the synthetic ``Test2024Eeg`` study from neuralset (3 timelines, ~12 train
Word events) so we exercise the real ``SegmentDataset`` + ``DataLoader``
pipeline instead of mocking it.  Each test compares the first-epoch index
sequence from ``train_loader.sampler``, which is the ground-truth signal that
``Data.seed`` actually controls shuffling.

The ``build_data`` factory fixture (see ``conftest.py``) owns the Data
    construction config, so tests here only have to vary ``seed`` /
    ``sampler``.
"""

import random
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Data


def _train_indices(loaders: dict[str, DataLoader]) -> list[int]:
    """Return the first-epoch train-loader index sequence.

    For ``shuffle=True``, ``loader.sampler`` is a ``RandomSampler`` whose
    generator was set in ``Data.prepare``; for ``sampler=ClassificationSampler()``
    it's the ``WeightedRandomSampler`` we constructed.  In both cases iterating
    the sampler yields the per-epoch index permutation, which is the
    seed-controlled signal we care about.
    """
    return list(iter(loaders["train"].sampler))  # type: ignore[arg-type]


def test_train_indices_deterministic_for_given_seed(
    build_data: Callable[..., Data],
) -> None:
    """Two ``Data`` instances with the same ``seed`` must shuffle identically."""
    loaders_a = build_data(seed=7).prepare()
    loaders_b = build_data(seed=7).prepare()
    assert _train_indices(loaders_a) == _train_indices(loaders_b)


def test_train_indices_diverge_for_different_seeds(
    build_data: Callable[..., Data],
) -> None:
    """Changing ``seed`` must change the train-loader shuffle sequence."""
    loaders_a = build_data(seed=7).prepare()
    loaders_b = build_data(seed=8).prepare()
    assert _train_indices(loaders_a) != _train_indices(loaders_b)


def test_train_indices_independent_of_global_rng(
    build_data: Callable[..., Data],
) -> None:
    """The headline property: mutating the global torch / numpy / random state
    between two ``prepare()`` calls with the same ``seed`` must not shift the
    shuffle.  This is the test that earns the explicit-generator plumbing."""
    loaders_a = build_data(seed=7).prepare()

    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)

    loaders_b = build_data(seed=7).prepare()
    assert _train_indices(loaders_a) == _train_indices(loaders_b)


def test_seed_none_falls_back_to_global_rng(
    build_data: Callable[..., Data],
) -> None:
    """``seed=None`` -> shuffle is reproducible only via the global torch RNG."""
    torch.manual_seed(123)
    indices_a = _train_indices(build_data(seed=None).prepare())

    torch.manual_seed(123)
    indices_b = _train_indices(build_data(seed=None).prepare())

    assert indices_a == indices_b

    torch.manual_seed(999)
    indices_c = _train_indices(build_data(seed=None).prepare())
    assert indices_c != indices_a


def test_weighted_sampler_is_seeded_independently_of_global_rng(
    build_data: Callable[..., Data],
) -> None:
    """``WeightedRandomSampler`` draws are determined by ``Data.seed`` and
    immune to global-RNG changes between calls."""
    loaders_a = build_data(seed=7, sampler={"name": "ClassificationSampler"}).prepare()

    torch.manual_seed(999)

    loaders_b = build_data(seed=7, sampler={"name": "ClassificationSampler"}).prepare()
    assert _train_indices(loaders_a) == _train_indices(loaders_b)

    loaders_c = build_data(seed=8, sampler={"name": "ClassificationSampler"}).prepare()
    assert _train_indices(loaders_a) != _train_indices(loaders_c)


def test_train_shuffle_decoupled_from_val_test_loader_generators(
    build_data: Callable[..., Data],
) -> None:
    """Per-split generators: consuming the val/test loader generators (as
    multi-process workers do for base-seed derivation) must not shift the
    train shuffle.  Simulates the production ``num_workers > 0`` regime
    in-process, where every ``iter(val_loader)`` / ``iter(test_loader)``
    draws one int64 from the loader's generator.  Under the previous shared-
    generator design this test would fail because all three loaders pulled
    from the same stream.
    """
    loaders = build_data(seed=7).prepare()

    # Mimic what ``_MultiProcessingDataLoaderIter.__init__`` does each
    # ``iter(loader)``: draw one int64 from ``loader.generator`` to derive
    # the worker base seed.  Repeated draws stand in for Lightning's sanity-
    # check + per-epoch validation + final test cadence.
    for _ in range(20):
        torch.empty((), dtype=torch.int64).random_(
            generator=loaders["val"].generator  # type: ignore[arg-type]
        )
    for _ in range(5):
        torch.empty((), dtype=torch.int64).random_(
            generator=loaders["test"].generator  # type: ignore[arg-type]
        )

    train_indices_after_val_test = _train_indices(loaders)

    # Fresh ``Data`` with the same seed, no val/test consumption.
    fresh = build_data(seed=7).prepare()
    train_indices_fresh = _train_indices(fresh)

    assert train_indices_after_val_test == train_indices_fresh


def test_per_split_loader_generators_have_distinct_seeds(
    build_data: Callable[..., Data],
) -> None:
    """The three split DataLoaders must own independent generators with
    distinct base seeds, derived from the same ``Data.seed`` via
    ``SeedSequence``."""
    loaders = build_data(seed=7).prepare()
    train_seed = loaders["train"].generator.initial_seed()  # type: ignore[union-attr]
    val_seed = loaders["val"].generator.initial_seed()  # type: ignore[union-attr]
    test_seed = loaders["test"].generator.initial_seed()  # type: ignore[union-attr]

    assert len({train_seed, val_seed, test_seed}) == 3, (
        f"Expected three distinct sub-seeds, got "
        f"train={train_seed}, val={val_seed}, test={test_seed}"
    )
