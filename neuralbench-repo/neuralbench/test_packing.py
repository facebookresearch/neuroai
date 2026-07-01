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
    _backend_config,
    _ExperimentStep,
    _pending_experiments,
    _should_submit_experiment,
    submit_packed,
)

# module-level so spawned workers can pickle them


class ValueExperiment(BaseExperiment):
    value: int
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def run(self) -> int:
        return self.value


class FailingExperiment(BaseExperiment):
    value: int
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def run(self) -> int:
        raise RuntimeError(f"intentional failure for value={self.value}")


@pytest.fixture
def infra(tmp_path: Path) -> tp.Any:
    # tp.Any so the dict passes pydantic strict-init (validated to TaskInfra).
    return {"cluster": None, "folder": tmp_path, "mode": "cached"}


@pytest.mark.parametrize(
    ("mode", "status", "expected"),
    [
        ("read-only", "completed", False),
        ("force", "completed", True),
        ("cached", "not submitted", True),
        ("cached", "completed", False),
        ("cached", "failed", False),
        ("retry", "failed", True),
        ("retry", "completed", False),
    ],
)
def test_should_submit_experiment(mode: str, status: str, expected: bool) -> None:
    exp = Mock()
    exp.infra.mode = mode
    exp.infra.status.return_value = status
    assert _should_submit_experiment(exp) is expected


def test_pending_skips_read_only_sorts_and_is_order_independent(infra: tp.Any) -> None:
    assert _pending_experiments([]) == []
    read_only = [
        ValueExperiment(value=i, infra={**infra, "mode": "read-only"}) for i in range(3)
    ]
    assert _pending_experiments(read_only) == []
    exps = [ValueExperiment(value=i, infra=infra) for i in range(4)]
    uids = [e.infra.uid() for e in _pending_experiments(exps)]
    assert uids == sorted(uids)  # sorted by uid
    assert uids == [e.infra.uid() for e in _pending_experiments(exps[::-1])]


def test_backend_config_local_is_cached_without_resource_knobs(tmp_path: Path) -> None:
    ti = exca.TaskInfra(cluster=None, folder=tmp_path, gpus_per_node=2)
    assert _backend_config(ti, ti.folder, max_jobs=5) == {
        "backend": "Cached",
        "folder": tmp_path,
        "mode": "force",
    }


def test_backend_config_translates_slurm_resources(tmp_path: Path) -> None:
    ti = exca.TaskInfra(
        cluster="slurm", folder=tmp_path, gpus_per_node=1, slurm_partition="gpu"
    )
    cfg = _backend_config(ti, ti.folder, max_jobs=7)
    assert cfg["backend"] == "Slurm"
    assert cfg["mode"] == "force"
    assert cfg["max_jobs"] == 7
    assert cfg["gpus_per_node"] == 1
    assert cfg["partition"] == "gpu"  # slurm_partition -> partition
    assert "use_srun" not in cfg  # default False dropped
    from exca.steps import backends

    backends.Backend.model_validate(cfg)  # must build a real backend


def test_backend_config_rejects_unknown_cluster(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported infra.cluster"):
        _backend_config(SimpleNamespace(cluster="k8s"), tmp_path, max_jobs=1)


def test_experiment_step_pickles_with_bound_experiment(infra: tp.Any) -> None:
    exp = ValueExperiment(value=42, infra=infra)
    step = _ExperimentStep(exp_uid=exp.infra.uid()).bind(exp)
    restored = pickle.loads(pickle.dumps(step))  # workers pickle the step
    assert restored._experiment.value == 42


@pytest.mark.parametrize("experiments_per_job", [2, 3, "all"])
def test_submit_packed_runs_and_preserves_child_caches(
    infra: tp.Any, experiments_per_job: int | tp.Literal["all"]
) -> None:
    exps = [ValueExperiment(value=i, infra=infra) for i in range(5)]
    assert submit_packed(exps, experiments_per_job=experiments_per_job) == 5
    assert all(e.infra.status() == "completed" for e in exps)
    assert [e.run() for e in exps] == [0, 1, 2, 3, 4]  # each hit its own cache
    assert (
        submit_packed(exps, experiments_per_job=experiments_per_job) == 0
    )  # nothing left


def test_submit_packed_isolates_failures(infra: tp.Any) -> None:
    exps: list[BaseExperiment] = [
        ValueExperiment(value=10, infra=infra),
        FailingExperiment(value=0, infra=infra),
        ValueExperiment(value=20, infra=infra),
    ]
    assert submit_packed(exps, experiments_per_job="all") == 3  # sweep not aborted
    assert exps[0].infra.status() == "completed"
    assert exps[1].infra.status() == "failed"  # failure surfaces via TaskInfra
    assert exps[2].infra.status() == "completed"


def test_submit_packed_experiments_per_job_caps_array_elements(
    infra: tp.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    slurm: tp.Any = {**infra, "cluster": "slurm"}
    exps = [ValueExperiment(value=i, infra=slurm) for i in range(10)]
    captured: dict[str, tp.Any] = {}

    class _StubParallel:
        def __init__(self, steps: tp.Any, infra: tp.Any) -> None:
            captured["max_jobs"] = infra["max_jobs"]

        def run(self) -> None:
            pass

    monkeypatch.setattr(packing, "Parallel", _StubParallel)
    submit_packed(exps, experiments_per_job=4)
    assert captured["max_jobs"] == 3  # ceil(10 / 4)
