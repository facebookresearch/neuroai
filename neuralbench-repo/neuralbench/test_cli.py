# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pytest
from exca import ConfDict

import neuralset as ns

from . import transforms as _transforms  # noqa: F401  — registers Step subclasses
from .experiment_config import prepare_task_configs
from .main import Data
from .registry import ALL_DATASETS, ALL_TASKS, DEFAULTS_DIR, load_yaml_config


def test_build_all_datasets() -> None:
    """Import-time _build_all_datasets parses every config with 'source' key."""
    assert len(ALL_TASKS) > 30
    total_studies = sum(
        len(studies) for tasks in ALL_DATASETS.values() for studies in tasks.values()
    )
    assert total_studies > 50


@pytest.mark.parametrize("dataset", [None, "schalk2004bci"])
def test_prepare_task_configs(dataset: str | None) -> None:
    """Merged config produces a valid Data with a Chain study.

    schalk2004bci uses =replace= which wipes the study dict; _restore_default_source
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
