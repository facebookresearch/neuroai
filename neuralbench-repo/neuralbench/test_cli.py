# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import typing as tp
import warnings
from collections.abc import Callable

import pytest
from exca import ConfDict

import neuralset as ns

from . import transforms as _transforms  # noqa: F401  — registers Step subclasses
from .cli import run_benchmark
from .experiment_config import (
    _apply_prepare_overlay,
    _warn_slurm_partition,
    prepare_task_configs,
)
from .main import Data
from .registry import ALL_DATASETS, ALL_TASKS, DEFAULTS_DIR, load_yaml_config


def test_build_all_datasets() -> None:
    """Import-time _build_all_datasets parses every config with 'source' key."""
    assert len(ALL_TASKS) > 30
    total_studies = sum(
        len(studies) for tasks in ALL_DATASETS.values() for studies in tasks.values()
    )
    assert total_studies > 50


@pytest.mark.parametrize("dataset", [None, "schalk2004bci2000"])
def test_prepare_task_configs(dataset: str | None) -> None:
    """Merged config produces a valid Data with a Chain study.

    schalk2004bci2000 uses =replace= which wipes the study dict; _restore_default_source
    must re-inject path and infra from the defaults.
    """
    config = ConfDict(load_yaml_config(DEFAULTS_DIR / "config.yaml"))
    grid = ConfDict(load_yaml_config(DEFAULTS_DIR / "grid.yaml"))
    datasets: list[str | None] | None = [dataset] if dataset is not None else None
    configs = prepare_task_configs(
        config,
        grid,
        "eeg",
        "motor_imagery",
        use_task_grid=False,
        debug=False,
        force=False,
        prepare=False,
        download=False,
        models=[None],
        datasets=datasets,
    )
    data = Data(**configs[0]["data"])
    assert isinstance(data.study, ns.Chain)
    steps: tp.Any = data.study.steps
    assert isinstance(steps, dict)
    source: tp.Any = steps["source"]
    assert source.path is not None
    assert source.infra is not None


@pytest.mark.parametrize("cluster", [None, "auto", "slurm"])
def test_cluster_config_wires_all_infra_clusters(
    patch_config: Callable[..., None], cluster: str | None
) -> None:
    """The CLUSTER config value drives training + both cache infra.cluster fields."""
    patch_config(CLUSTER=cluster)
    raw = load_yaml_config(DEFAULTS_DIR / "config.yaml")
    assert raw is not None
    assert raw["infra"]["cluster"] == cluster
    assert raw["data"]["neuro"]["infra"]["cluster"] == cluster
    assert raw["data"]["target"]["infra"]["cluster"] == cluster


@pytest.mark.parametrize("cluster", [None, "auto", "slurm"])
def test_prepare_overlay_respects_cluster(
    patch_config: Callable[..., None], cluster: str | None
) -> None:
    """``--prepare`` uses the configured CLUSTER for the run and both caches.

    "auto" fans out to SLURM when available and runs locally otherwise, so the
    cache infra is never hard-coded to "slurm" (which would fail without SLURM).
    """
    patch_config(CLUSTER=cluster)
    config = ConfDict(load_yaml_config(DEFAULTS_DIR / "config.yaml"))
    _apply_prepare_overlay(config)
    flat = config.flat()
    assert flat["infra.cluster"] == cluster
    assert flat["data.neuro.infra.cluster"] == cluster
    assert flat["data.target.infra.cluster"] == cluster


def _capture_assembled_experiments(
    monkeypatch: pytest.MonkeyPatch,
) -> list[ConfDict]:
    """Run ``run_benchmark`` with a stubbed aggregator, returning assembled configs."""
    captured: dict[str, tp.Any] = {}

    class _FakeAggregator:
        def __init__(self, experiments: list[ConfDict], debug: bool) -> None:
            captured["experiments"] = experiments

        def prepare(self) -> None:
            pass

    monkeypatch.setattr("neuralbench.main.BenchmarkAggregator", _FakeAggregator)
    run_benchmark("eeg", "motor_imagery")
    return captured["experiments"]


def test_run_benchmark_disables_wandb_when_host_blank(
    monkeypatch: pytest.MonkeyPatch, patch_config: Callable[..., None]
) -> None:
    """A blank WANDB_HOST nulls wandb_config across all assembled experiments."""
    patch_config(WANDB_HOST="", SLURM_PARTITION="dummy")
    experiments = _capture_assembled_experiments(monkeypatch)
    assert experiments
    assert all(cfg.get("wandb_config") is None for cfg in experiments)


def test_run_benchmark_keeps_wandb_when_host_set(
    monkeypatch: pytest.MonkeyPatch, patch_config: Callable[..., None]
) -> None:
    """A configured WANDB_HOST is preserved in assembled experiments."""
    host = "https://wandb.example.com"
    patch_config(WANDB_HOST=host, SLURM_PARTITION="dummy")
    experiments = _capture_assembled_experiments(monkeypatch)
    wandb_cfg = experiments[0].get("wandb_config")
    assert wandb_cfg is not None
    assert wandb_cfg["host"] == host


def test_warn_slurm_partition_suppressed_when_cluster_local(
    monkeypatch: pytest.MonkeyPatch, patch_config: Callable[..., None]
) -> None:
    """No SLURM warning when CLUSTER forces local execution, even with srun present."""
    patch_config(CLUSTER=None, SLURM_PARTITION="")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/srun")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_slurm_partition(debug=False)


def test_warn_slurm_partition_fires_without_partition(
    monkeypatch: pytest.MonkeyPatch, patch_config: Callable[..., None]
) -> None:
    """SLURM warning still fires under CLUSTER='auto' with srun but no partition."""
    patch_config(CLUSTER="auto", SLURM_PARTITION="")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/srun")
    with pytest.warns(UserWarning, match="SLURM is available"):
        _warn_slurm_partition(debug=False)


def test_run_benchmark_cli_help_smoke(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``neuralbench --help`` exits 0 and lists devices/tasks via the epilog.

    Smoke-tests that the CLI parser builds, the registry loads, and
    ``_format_datasets_epilog`` renders without crashing.
    """
    from .cli import run_benchmark_cli

    monkeypatch.setattr("sys.argv", ["neuralbench", "--help"])
    with pytest.raises(SystemExit) as exc:
        run_benchmark_cli()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "available datasets per task:" in out
    assert "eeg" in out
