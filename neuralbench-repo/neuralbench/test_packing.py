# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import exca
import pytest
from pydantic import ValidationError

from neuraltrain.utils import BaseExperiment

from .packing import (
    PackedExperiment,
    _should_submit_experiment,
    pack_experiments_for_submission,
)

# ---- module-level helpers (must be module-level so spawn workers can pickle) -


class ValueExperiment(BaseExperiment):
    value: int
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def run(self):
        return self.value


class FailingExperiment(BaseExperiment):
    """Raises ``RuntimeError`` from :meth:`run` for error-propagation tests."""

    value: int
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def run(self):
        raise RuntimeError(f"intentional failure for value={self.value}")


@pytest.fixture
def infra(tmp_path: Path) -> tp.Any:
    # ``tp.Any`` so the dict can be passed to pydantic models without
    # tripping the pydantic-mypy strict-init check (the dict is converted
    # to a ``TaskInfra`` at runtime).
    return {"cluster": "auto", "folder": tmp_path, "mode": "force"}


def _run_locally(packed: PackedExperiment, n_jobs: int = 1) -> list[tp.Any]:
    cfg: tp.Any = packed.infra.model_dump(mode="python")
    cfg["cluster"] = None
    return type(packed)(experiments=packed.experiments, infra=cfg, n_jobs=n_jobs).run()


# ---- _should_submit_experiment truth table ---------------------------------


@pytest.mark.parametrize(
    ("mode", "status", "expected"),
    [
        ("read-only", "not submitted", False),
        ("read-only", "completed", False),
        ("read-only", "failed", False),
        ("force", "not submitted", True),
        ("force", "completed", True),
        ("force", "running", True),
        ("cached", "not submitted", True),
        ("cached", "completed", False),
        ("cached", "running", False),
        ("cached", "failed", False),
        ("retry", "not submitted", True),
        ("retry", "failed", True),
        ("retry", "completed", False),
        ("retry", "running", False),
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


# ---- grouping + running, serial and parallel -------------------------------


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize(
    ("experiments_per_job", "expected_sizes"),
    [
        (1, [1, 1, 1, 1, 1]),
        (2, [2, 2, 1]),
        (3, [3, 2]),
        (5, [5]),
        (10, [5]),
        ("all", [5]),
    ],
)
def test_pack_groups_and_runs(
    infra: tp.Any,
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
    assert all(exp.infra.cluster is None for job in packed for exp in job.experiments)

    expected_values = [
        exp.value for exp in sorted(experiments, key=lambda e: e.infra.uid())
    ]
    cursor = 0
    for job, size in zip(packed, expected_sizes):
        assert _run_locally(job, n_jobs=n_jobs) == expected_values[cursor : cursor + size]
        cursor += size


# ---- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"experiments_per_job": 0}, r"got 0"),
        ({"experiments_per_job": -2}, r"got -2"),
        ({"experiments_per_job": "ALL"}, r"got 'ALL'"),
        ({"experiments_per_job": "every"}, r"got 'every'"),
        ({"experiments_per_job": 1, "n_jobs": 0}, r"n_jobs must be >= 1"),
        ({"experiments_per_job": 1, "n_jobs": -3}, r"n_jobs must be >= 1"),
    ],
)
def test_pack_validates_arguments(
    infra: tp.Any,
    kwargs: dict[str, tp.Any],
    match: str,
) -> None:
    experiment = ValueExperiment(value=1, infra=infra)
    with pytest.raises(ValueError, match=match):
        pack_experiments_for_submission([experiment], **kwargs)


# ---- skip / empty ----------------------------------------------------------


def test_pack_returns_empty_for_empty_input() -> None:
    assert pack_experiments_for_submission([]) == []


def test_pack_skips_all_read_only(infra: tp.Any) -> None:
    read_only: tp.Any = {**infra, "mode": "read-only"}
    experiments = [ValueExperiment(value=i, infra=read_only) for i in range(3)]
    assert pack_experiments_for_submission(experiments) == []


# ---- UID order independence ------------------------------------------------


@pytest.mark.parametrize("experiments_per_job", [1, 2, 3, "all"])
def test_pack_uid_order_independent(
    infra: tp.Any,
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
def test_packed_experiment_propagates_errors(infra: tp.Any, n_jobs: int) -> None:
    failing = [FailingExperiment(value=i, infra=infra) for i in range(2)]
    packed = pack_experiments_for_submission(failing, experiments_per_job="all")
    assert len(packed) == 1
    with pytest.raises(RuntimeError, match="intentional failure"):
        _run_locally(packed[0], n_jobs=n_jobs)


# ---- PackedExperiment constructor validation ------------------------------


@pytest.mark.parametrize("n_jobs", [0, -1])
def test_packed_experiment_rejects_bad_n_jobs(infra: tp.Any, n_jobs: int) -> None:
    exp = ValueExperiment(value=0, infra=infra)
    with pytest.raises(ValidationError):
        PackedExperiment(experiments=[exp], n_jobs=n_jobs)


def test_packed_experiment_rejects_empty_experiments() -> None:
    with pytest.raises(ValidationError):
        PackedExperiment(experiments=[], n_jobs=1)
