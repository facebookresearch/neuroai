# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``neuralbench.data``: ``Data`` RNG plumbing and the config merge
behind ``get_default_dataloaders``.

The RNG tests use the synthetic ``Test2024Eeg`` study from neuralset (3
timelines, ~12 train Word events) so we exercise the real ``SegmentDataset`` +
``DataLoader`` pipeline instead of mocking it.  Each test compares the
first-epoch index sequence from ``train_loader.sampler``, which is the
ground-truth signal that ``Data.seed`` actually controls shuffling.

The ``build_data`` factory fixture (see ``conftest.py``) owns the Data
    construction config, so tests here only have to vary ``seed`` /
    ``sampler``.

The ``get_default_dataloaders`` tests stub out ``Data.prepare`` only, so the
real task YAMLs still go through ``Data`` validation without reading any data.
"""

import random
import typing as tp
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import neuralset as ns

from .data import Data, get_default_dataloaders


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


def _segments(loaders: dict[str, DataLoader]) -> list[tp.Any]:
    """Every segment across the three split loaders."""
    return [s for loader in loaders.values() for s in loader.dataset.segments]  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "nan_fraction,min_finite,keeps_poisoned",
    [(1.0, 1.0, False), (0.5, 1.0, False), (0.5, 0.75, False), (0.5, 0.5, True)],
)
def test_segments_dropped_below_min_finite_target_fraction(
    build_data: Callable[..., Data],
    monkeypatch: pytest.MonkeyPatch,
    nan_fraction: float,
    min_finite: float,
    keeps_poisoned: bool,
) -> None:
    target: tp.Any = {"name": "MneRaw", "event_types": "Eeg"}
    segments = _segments(build_data(seed=7, target=target).prepare())
    poisoned = set(sorted({s.start for s in segments})[::2])

    extract = ns.extractors.MneRaw._get_timed_array

    def nan_poisoned(
        self: ns.extractors.MneRaw, event: tp.Any, start: float, duration: float
    ) -> tp.Any:
        out = extract(self, event, start, duration)
        if start in poisoned:
            data = np.array(out.data, copy=True)
            data[..., : round(nan_fraction * data.shape[-1])] = np.nan
            out.data = data
        return out

    monkeypatch.setattr(ns.extractors.MneRaw, "_get_timed_array", nan_poisoned)
    filtered = build_data(
        seed=7, target=target, min_finite_target_fraction=min_finite
    ).prepare()

    kept = {s.start for s in _segments(filtered)}
    assert bool(kept & poisoned) == keeps_poisoned, (
        f"a target labelled over {1 - nan_fraction:.0%} of its frames should "
        f"{'survive' if keeps_poisoned else 'not survive'} min_finite={min_finite}"
    )
    assert {s.start for s in segments} - poisoned <= kept, (
        "dropped segments whose target is finite"
    )


def test_get_default_dataloaders_merges_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg: Data | None = None
    expected_loaders: dict[str, DataLoader] = {}

    def fake_prepare(self: Data) -> dict[str, DataLoader]:
        nonlocal cfg
        cfg = self
        return expected_loaders

    monkeypatch.setattr(Data, "prepare", fake_prepare)
    overrides: dict[str, tp.Any] = {"neuro.frequency": 60.0}
    loaders = get_default_dataloaders(
        "eeg",
        "audiovisual_stimulus",
        batch_size=8,
        **overrides,
    )
    assert loaders is expected_loaders
    assert cfg is not None
    target: tp.Any = cfg.target
    assert cfg.trigger_event_type == "Stimulus"  # task config wins over base
    assert target.event_field == "description"  # task =replace= target
    assert cfg.batch_size == 8  # base default -> kwarg override
    assert cfg.neuro.frequency == 60.0  # dotted override (base default 120.0)


def test_get_default_dataloaders_selects_dataset_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Data] = []
    monkeypatch.setattr(Data, "prepare", lambda self: captured.append(self))
    get_default_dataloaders("eeg", "motor_imagery", dataset="schalk2004bci2000")
    study: tp.Any = captured[0].study
    assert type(study.steps["source"]).__name__ == "Schalk2004Bci2000"


@pytest.mark.parametrize(
    "task,dataset",
    [
        ("motor_imgery", None),
        ("all", None),  # aggregate: only the CLI expands it
        ("motor_imagery", "schalk2004"),
    ],
)
def test_get_default_dataloaders_rejects_unknown_inputs(
    task: str, dataset: str | None
) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        get_default_dataloaders("eeg", task, dataset=dataset)
