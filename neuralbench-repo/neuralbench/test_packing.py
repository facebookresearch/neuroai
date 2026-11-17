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
import pydantic
import pytest
from exca.steps import backends

from neuraltrain.utils import BaseExperiment

from . import packing
from .packing import (
    _backend_config,
    _ExperimentStep,
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


def test_backend_config(tmp_path: Path) -> None:
    local = exca.TaskInfra(cluster=None, folder=tmp_path, gpus_per_node=2)
    assert _backend_config(local, tmp_path, max_jobs=5) == {
        "backend": "Cached",  # in-process: no resource knobs
        "folder": tmp_path,
        "mode": "force",
    }
    slurm = exca.TaskInfra(
        cluster="slurm", folder=tmp_path, gpus_per_node=1, slurm_partition="gpu"
    )
    cfg = _backend_config(slurm, tmp_path, max_jobs=7)
    keys = ("backend", "mode", "max_jobs", "gpus_per_node", "partition")
    assert {k: cfg[k] for k in keys} == {
        "backend": "Slurm",
        "mode": "force",
        "max_jobs": 7,
        "gpus_per_node": 1,
        "partition": "gpu",  # slurm_partition -> partition
    }
    assert "use_srun" not in cfg  # defaults are dropped
    backends.Backend.model_validate(cfg)  # builds a real backend
    for cluster in ("auto", "local", "debug"):  # every TaskInfra.cluster value maps
        ti = exca.TaskInfra(cluster=cluster, folder=tmp_path)
        backends.Backend.model_validate(_backend_config(ti, tmp_path, max_jobs=2))
    with pytest.raises(ValueError, match="unsupported infra.cluster"):
        _backend_config(SimpleNamespace(cluster="k8s"), tmp_path, max_jobs=1)


# "local" runs the packed jobs in subprocesses, through exca's LocalProcess backend
@pytest.mark.parametrize(
    "cluster, experiments_per_job", [(None, 2), (None, "all"), ("local", 2)]
)
def test_submit_packed_runs_caches_and_isolates_failures(
    infra: tp.Any, cluster: str | None, experiments_per_job: int | tp.Literal["all"]
) -> None:
    infra = {**infra, "cluster": cluster}
    read_only: tp.Any = {**infra, "mode": "read-only"}
    assert submit_packed([ValueExperiment(value=0, infra=read_only)]) == 0
    exps: list[BaseExperiment] = [ValueExperiment(value=i, infra=infra) for i in range(4)]
    exps.insert(1, FailingExperiment(value=0, infra=infra))
    step = _ExperimentStep(exp_uid=exps[0].infra.uid()).bind(exps[0])
    assert pickle.loads(pickle.dumps(step))._experiment.value == 0  # workers pickle it
    assert submit_packed(exps, experiments_per_job) == 5  # the failure aborts nothing
    assert [e.infra.status() for e in exps] == ["completed", "failed"] + ["completed"] * 3
    assert [e.run() for e in exps if isinstance(e, ValueExperiment)] == [0, 1, 2, 3]
    assert submit_packed(exps, experiments_per_job) == 0  # all cached (failed stays)


def test_submit_packed_caps_jobs_per_resource_request(
    infra: tp.Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweeps: list[tuple[int, int, int]] = []

    class _StubParallel:
        def __init__(self, steps: tp.Any, infra: tp.Any) -> None:
            sweeps.append((len(steps), infra["max_jobs"], infra["gpus_per_node"]))

        def run(self) -> None:
            pass

    monkeypatch.setattr(packing, "Parallel", _StubParallel)
    infras: list[tp.Any] = [
        {**infra, "cluster": "slurm", "gpus_per_node": g} for g in [1] * 10 + [4]
    ]
    exps = [ValueExperiment(value=i, infra=inf) for i, inf in enumerate(infras)]
    assert submit_packed(exps, experiments_per_job=4) == 11
    # one sweep per resource request, ceil(n / 4) jobs each
    assert sorted(sweeps) == [(1, 1, 4), (10, 3, 1)]


def test_aggregator_validates_experiments_per_job() -> None:
    from .main import BenchmarkAggregator

    def per_job(value: tp.Any) -> tp.Any:
        agg = BenchmarkAggregator.model_validate(
            {"experiments": [], "experiments_per_job": value}
        )
        return agg.experiments_per_job

    # the CLI passes strings: pydantic coerces them, and rejects what is not >= 1
    assert (per_job("4"), per_job("all")) == (4, "all")
    for bad in ("0", "-1", "some"):
        with pytest.raises(pydantic.ValidationError):
            per_job(bad)
