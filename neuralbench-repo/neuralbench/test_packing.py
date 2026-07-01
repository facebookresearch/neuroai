# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import typing as tp
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import exca
import pytest

from neuraltrain.utils import BaseExperiment

from . import packing
from .packing import (
    _ExperimentStep,
    _backend_config,
    _pending_experiments,
    _should_submit_experiment,
    submit_packed,
)

# ---- module-level helpers (importable so workers can pickle them) ----------


class ValueExperiment(BaseExperiment):
    value: int
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def run(self) -> int:
        return self.value


class FailingExperiment(BaseExperiment):
    """Raises ``RuntimeError`` from :meth:`run` for fault-isolation tests."""

    value: int
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def run(self) -> int:
        raise RuntimeError(f"intentional failure for value={self.value}")


@pytest.fixture
def infra(tmp_path: Path) -> tp.Any:
    # ``tp.Any`` so the dict can be passed to pydantic models without tripping
    # the pydantic-mypy strict-init check (converted to a ``TaskInfra`` at runtime).
    return {"cluster": None, "folder": tmp_path, "mode": "cached"}


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
    exp = Mock()
    exp.infra.mode = mode
    exp.infra.status.return_value = status
    assert _should_submit_experiment(exp) is expected


# ---- _pending_experiments: filter, sort, order-independence ----------------


def test_pending_experiments_empty() -> None:
    assert _pending_experiments([]) == []


def test_pending_skips_all_read_only(infra: tp.Any) -> None:
    read_only: tp.Any = {**infra, "mode": "read-only"}
    exps = [ValueExperiment(value=i, infra=read_only) for i in range(3)]
    assert _pending_experiments(exps) == []


def test_pending_sorted_by_uid_and_order_independent(infra: tp.Any) -> None:
    exps = [ValueExperiment(value=i, infra=infra) for i in range(4)]
    forward = _pending_experiments(exps)
    reverse = _pending_experiments(exps[::-1])
    uids = [e.infra.uid() for e in forward]
    assert uids == sorted(uids)  # sorted by uid
    assert uids == [e.infra.uid() for e in reverse]  # order-independent


# ---- _backend_config: TaskInfra -> steps backend translation ---------------


def test_backend_config_local_is_cached_without_resource_knobs(tmp_path: Path) -> None:
    ti = exca.TaskInfra(cluster=None, folder=tmp_path, gpus_per_node=2)
    cfg = _backend_config(ti, ti.folder, max_jobs=5)
    assert cfg == {"backend": "Cached", "folder": tmp_path, "mode": "force"}


@pytest.mark.parametrize(("cluster", "backend"), [("auto", "Auto"), ("slurm", "Slurm")])
def test_backend_config_translates_resources(
    tmp_path: Path, cluster: str, backend: str
) -> None:
    ti = exca.TaskInfra(
        cluster=cluster,
        folder=tmp_path,
        gpus_per_node=1,
        cpus_per_task=4,
        slurm_partition="gpu",
        slurm_use_srun=False,
    )
    cfg = _backend_config(ti, ti.folder, max_jobs=7)
    assert cfg["backend"] == backend
    assert cfg["mode"] == "force"
    assert cfg["max_jobs"] == 7
    assert cfg["gpus_per_node"] == 1
    assert cfg["cpus_per_task"] == 4
    assert cfg["partition"] == "gpu"  # slurm_partition -> partition
    assert "use_srun" not in cfg  # default False is dropped
    # the produced dict must build a real backend
    from exca.steps import backends

    backends.Backend.model_validate(cfg)


def test_backend_config_rejects_unknown_cluster(tmp_path: Path) -> None:
    # defensive guard: every real TaskInfra.cluster literal is mapped, so use a
    # stub carrying an out-of-range value.
    stub = SimpleNamespace(cluster="kubernetes")
    with pytest.raises(ValueError, match="unsupported infra.cluster"):
        _backend_config(stub, tmp_path, max_jobs=1)


@pytest.mark.parametrize(
    ("cluster", "backend"),
    [(None, "Cached"), ("debug", "Cached"), ("local", "LocalProcess")],
)
def test_backend_config_maps_all_local_clusters(
    tmp_path: Path, cluster: str | None, backend: str
) -> None:
    ti = exca.TaskInfra(cluster=cluster, folder=tmp_path)
    cfg = _backend_config(ti, ti.folder, max_jobs=3)
    assert cfg["backend"] == backend
    from exca.steps import backends

    backends.Backend.model_validate(cfg)  # must build a real backend


# ---- _ExperimentStep identity + pickle -------------------------------------


def test_experiment_step_identity_excludes_experiment(infra: tp.Any) -> None:
    exps = [ValueExperiment(value=i, infra=infra) for i in range(3)]
    steps = [_ExperimentStep(exp_uid=e.infra.uid()).bind(e) for e in exps]
    # identity is the exp_uid alone; distinct experiments -> distinct steps
    assert len({s.exp_uid for s in steps}) == 3


def test_experiment_step_pickles_with_bound_experiment(infra: tp.Any) -> None:
    exp = ValueExperiment(value=42, infra=infra)
    step = _ExperimentStep(exp_uid=exp.infra.uid()).bind(exp)
    restored = pickle.loads(pickle.dumps(step))  # workers pickle the step
    assert restored._experiment.value == 42


# ---- submit_packed: end-to-end via the local (Cached) backend --------------


@pytest.mark.parametrize("experiments_per_job", [2, 3, "all"])
def test_submit_packed_runs_and_preserves_child_caches(
    infra: tp.Any, experiments_per_job: int | tp.Literal["all"]
) -> None:
    exps = [ValueExperiment(value=i, infra=infra) for i in range(5)]
    dispatched = submit_packed(exps, experiments_per_job=experiments_per_job)
    assert dispatched == 5
    assert all(e.infra.status() == "completed" for e in exps)
    # re-running each experiment hits its own TaskInfra cache
    assert [e.run() for e in exps] == [0, 1, 2, 3, 4]
    # nothing left to submit
    assert submit_packed(exps, experiments_per_job=experiments_per_job) == 0


def test_submit_packed_empty_when_all_cached(infra: tp.Any) -> None:
    exps = [ValueExperiment(value=i, infra=infra) for i in range(3)]
    for e in exps:
        e.run()  # pre-populate caches
    assert submit_packed(exps, experiments_per_job="all") == 0


def test_submit_packed_isolates_failures(infra: tp.Any) -> None:
    exps: list[BaseExperiment] = [
        ValueExperiment(value=10, infra=infra),
        FailingExperiment(value=0, infra=infra),
        ValueExperiment(value=20, infra=infra),
    ]
    # a single failure must not abort the sweep
    dispatched = submit_packed(exps, experiments_per_job="all")
    assert dispatched == 3
    assert exps[0].infra.status() == "completed"
    assert exps[2].infra.status() == "completed"
    assert exps[1].infra.status() == "failed"


def test_submit_packed_experiments_per_job_caps_array_elements(
    infra: tp.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # cluster=slurm so max_jobs lands in the backend; stub Parallel to capture it
    slurm: tp.Any = {**infra, "cluster": "slurm"}
    exps = [ValueExperiment(value=i, infra=slurm) for i in range(10)]

    captured: dict[str, tp.Any] = {}

    class _StubParallel:
        def __init__(self, steps: tp.Any, infra: tp.Any) -> None:
            captured["n_steps"] = len(steps)
            captured["max_jobs"] = infra["max_jobs"]

        def run(self) -> None:
            pass

    monkeypatch.setattr(packing, "Parallel", _StubParallel)
    submit_packed(exps, experiments_per_job=4)
    assert captured["n_steps"] == 10
    assert captured["max_jobs"] == 3  # ceil(10 / 4)
