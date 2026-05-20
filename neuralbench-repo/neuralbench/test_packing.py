# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import pytest
from pydantic import ValidationError

from ._test_helpers import FailingExperiment, ValueExperiment
from .packing import (
    PackedExperiment,
    _should_submit_experiment,
    pack_experiments_for_submission,
)


# ---- fixtures & helpers ----------------------------------------------------


@pytest.fixture
def infra(tmp_path: Path) -> dict[str, tp.Any]:
    """A force-mode TaskInfra config stored under ``tmp_path``."""
    return {"cluster": "auto", "folder": tmp_path, "mode": "force"}


def _run_locally(packed: PackedExperiment, n_jobs: int = 1) -> list[tp.Any]:
    """Re-run a packed job in-process, exercising the chosen ``n_jobs`` path."""
    cfg = packed.infra.model_dump(mode="python")
    cfg["cluster"] = None
    return type(packed)(
        experiments=packed.experiments, infra=cfg, n_jobs=n_jobs
    ).run()


def _sorted_values(experiments: list[ValueExperiment]) -> list[float]:
    """Values in the order the packer sorts pending experiments (by uid)."""
    return [
        float(e.value) for e in sorted(experiments, key=lambda e: e.infra.uid())
    ]


# ---- _should_submit_experiment truth table ---------------------------------


@pytest.mark.parametrize(
    ("mode", "status", "expected"),
    [
        # read-only never submits, whatever the status
        ("read-only", "not submitted", False),
        ("read-only", "completed",     False),
        ("read-only", "failed",        False),
        # force always submits
        ("force",     "not submitted", True),
        ("force",     "completed",     True),
        ("force",     "running",       True),
        # cached: only never-run experiments
        ("cached",    "not submitted", True),
        ("cached",    "completed",     False),
        ("cached",    "running",       False),
        ("cached",    "failed",        False),
        # retry: not-submitted OR failed
        ("retry",     "not submitted", True),
        ("retry",     "failed",        True),
        ("retry",     "completed",     False),
        ("retry",     "running",       False),
    ],
)
def test_should_submit_experiment(mode: str, status: str, expected: bool) -> None:
    class _Infra:
        def __init__(self) -> None:
            self.mode = mode

        def status(self) -> str:
            return status

    class _Exp:
        infra = _Infra()

    assert _should_submit_experiment(_Exp()) is expected  # type: ignore[arg-type]


# ---- packing groups + executes them, serial and parallel -------------------


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize(
    ("experiments_per_job", "expected_sizes"),
    [
        (1,     [1, 1, 1, 1, 1]),
        (2,     [2, 2, 1]),
        (3,     [3, 2]),
        (5,     [5]),
        (10,    [5]),   # caps at len(pending)
        ("all", [5]),
    ],
)
def test_pack_groups_and_runs(
    infra: dict[str, tp.Any],
    experiments_per_job: int | tp.Literal["all"],
    expected_sizes: list[int],
    n_jobs: int,
) -> None:
    experiments = [ValueExperiment(value=i, infra=infra) for i in range(5)]

    packed = pack_experiments_for_submission(
        experiments, experiments_per_job=experiments_per_job, n_jobs=n_jobs
    )

    assert [len(job.experiments) for job in packed] == expected_sizes
    assert len({job.infra.uid() for job in packed}) == len(packed)
    assert all(
        exp.infra.cluster is None for job in packed for exp in job.experiments
    )

    expected_values = _sorted_values(experiments)
    cursor = 0
    for job, size in zip(packed, expected_sizes):
        assert _run_locally(job, n_jobs=n_jobs) == expected_values[
            cursor : cursor + size
        ]
        cursor += size


# ---- pack_experiments_for_submission validation ---------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"experiments_per_job": 0},     r"experiments_per_job must be >= 1 or 'all'\."),
        ({"experiments_per_job": -2},    r"experiments_per_job must be >= 1 or 'all'\."),
        ({"experiments_per_job": "ALL"}, r"got 'ALL'"),
        ({"experiments_per_job": "every"}, r"got 'every'"),
        ({"experiments_per_job": True},  r"got True"),
        ({"experiments_per_job": 1.5},   r"got 1\.5"),
        ({"experiments_per_job": 1, "n_jobs": 0},    r"n_jobs must be >= 1\."),
        ({"experiments_per_job": 1, "n_jobs": -3},   r"n_jobs must be >= 1\."),
        ({"experiments_per_job": 1, "n_jobs": True}, r"n_jobs must be >= 1\."),
    ],
)
def test_pack_validates_arguments(
    infra: dict[str, tp.Any],
    kwargs: dict[str, tp.Any],
    match: str,
) -> None:
    experiment = ValueExperiment(value=1, infra=infra)
    with pytest.raises(ValueError, match=match):
        pack_experiments_for_submission([experiment], **kwargs)


# ---- skip / empty handling -------------------------------------------------


def test_pack_returns_empty_for_empty_input() -> None:
    assert pack_experiments_for_submission([]) == []


def test_pack_skips_all_read_only(infra: dict[str, tp.Any]) -> None:
    read_only = {**infra, "mode": "read-only"}
    experiments = [ValueExperiment(value=i, infra=read_only) for i in range(3)]
    assert pack_experiments_for_submission(experiments) == []


# ---- UID order-independence at multiple group sizes -----------------------


@pytest.mark.parametrize("experiments_per_job", [1, 2, 3, "all"])
def test_pack_uid_order_independent(
    infra: dict[str, tp.Any],
    experiments_per_job: int | tp.Literal["all"],
) -> None:
    experiments = [ValueExperiment(value=i, infra=infra) for i in range(4)]
    a = pack_experiments_for_submission(
        experiments, experiments_per_job=experiments_per_job
    )
    b = pack_experiments_for_submission(
        experiments[::-1], experiments_per_job=experiments_per_job
    )
    assert [j.infra.uid() for j in a] == [j.infra.uid() for j in b]


# ---- error propagation through serial + spawn pool ------------------------


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_packed_experiment_propagates_errors(
    infra: dict[str, tp.Any], n_jobs: int
) -> None:
    failing = [FailingExperiment(value=i, infra=infra) for i in range(2)]
    packed = pack_experiments_for_submission(failing, experiments_per_job="all")
    assert len(packed) == 1
    with pytest.raises(RuntimeError, match="intentional failure"):
        _run_locally(packed[0], n_jobs=n_jobs)


# ---- PackedExperiment constructor validation ------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"n_jobs": 0},  id="n_jobs=0"),
        pytest.param({"n_jobs": -1}, id="n_jobs=-1"),
    ],
)
def test_packed_experiment_rejects_bad_n_jobs(
    infra: dict[str, tp.Any], bad: dict[str, tp.Any]
) -> None:
    exp = ValueExperiment(value=0, infra=infra)
    with pytest.raises(ValidationError):
        PackedExperiment(experiments=[exp], **bad)


def test_packed_experiment_rejects_empty_experiments() -> None:
    with pytest.raises(ValidationError):
        PackedExperiment(experiments=[], n_jobs=1)
