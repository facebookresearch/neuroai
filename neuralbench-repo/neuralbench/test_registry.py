# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke coverage for device/task-aware baseline dispatch.

The headline MEG bar chart leans on :func:`_task_aware_baseline` and the
plot-time ``_collapse_feature_based_baselines`` helper to surface a
``Handcrafted`` bar alongside ``Chance`` and ``Dummy`` on MEG tasks, so
these tests guard the end-to-end wiring:

1. ``DEVICE_BASELINE_MODELS["meg"]`` still exposes the Riemannian TS
   pipelines we rely on.
2. ``_task_aware_baseline`` / ``_expand_models`` pick up the canonical
   per-task pipeline for MEG ``image`` and MEG ``typing``.
3. ``_collapse_feature_based_baselines`` relabels the resulting rows to
   the shared ``feature_based`` alias that resolves to ``Handcrafted``.
"""

from __future__ import annotations

import pandas as pd

from neuralbench.plots.tables import _collapse_feature_based_baselines
from neuralbench.registry import (
    DEVICE_BASELINE_MODELS,
    FEATURE_BASED_BY_TASK,
    SKLEARN_BASELINE_MODELS,
    _expand_models,
    _task_aware_baseline,
)


def test_device_baseline_models_meg_has_riemannian_pipelines() -> None:
    meg = DEVICE_BASELINE_MODELS["meg"]
    assert "chance" in meg and "dummy" in meg
    assert "cov_ts_lr" in meg
    assert "cov_ts_ridge" in meg
    # Xdawn / CoSpectra are intentionally EEG-only.
    assert "xdawn_ts_lr" not in meg
    assert "cospectra_log_lr" not in meg


def test_task_aware_baseline_meg_image_picks_canonical_pipeline() -> None:
    assert FEATURE_BASED_BY_TASK["image"] == "cov_ts_ridge"
    assert _task_aware_baseline("meg", "image") == ["chance", "dummy", "cov_ts_ridge"]


def test_task_aware_baseline_meg_typing_uses_multi_output_pipeline() -> None:
    # ``typing`` is a 29-way one-hot retrieval target; the registry maps all
    # multi-output retrieval tasks to ``cov_ts_ridge`` (see the registry
    # comment block above FEATURE_BASED_BY_TASK).
    assert FEATURE_BASED_BY_TASK["typing"] == "cov_ts_ridge"
    assert _task_aware_baseline("meg", "typing") == ["chance", "dummy", "cov_ts_ridge"]


def test_task_aware_baseline_fmri_still_excludes_sklearn_pipelines() -> None:
    # Regression guard: fMRI is still chance+dummy only.
    assert _task_aware_baseline("fmri", "image") == ["chance", "dummy"]


def test_task_aware_baseline_unknown_task_drops_sklearn_pipelines() -> None:
    # For tasks not in FEATURE_BASED_BY_TASK, no sklearn pipeline is appended.
    assert _task_aware_baseline("meg", "__nonexistent_task__") == ["chance", "dummy"]


def test_expand_all_baseline_meg_includes_handcrafted_pipeline() -> None:
    models = [
        m
        for m in _expand_models("all_baseline", device="meg", task_name="image")
        if m is not None
    ]
    assert set(models) == {"chance", "dummy", "cov_ts_ridge"}


def test_expand_all_meg_image_includes_handcrafted_pipeline() -> None:
    # The ``all`` alias expands to classic + FM + baseline; the baseline
    # portion must still pull in the canonical handcrafted pipeline.
    models = [
        m for m in _expand_models("all", device="meg", task_name="image") if m is not None
    ]
    assert "cov_ts_ridge" in models
    assert "chance" in models and "dummy" in models


def test_prepare_task_configs_meg_image_cov_ts_ridge_resolves() -> None:
    """End-to-end CLI-path smoke: MEG image + cov_ts_ridge merges + instantiates.

    Exercises the real import chain (``prepare_task_configs`` + ``Data`` + the
    pydantic ``BaseModelConfig`` discriminator) that the CLI uses to turn
    ``neuralbench meg image -m cov_ts_ridge`` into a runnable config, without
    actually fitting the pipeline.  Guards against silently breaking the MEG
    handcrafted path via YAML / extractor / discriminator regressions.
    """
    from exca import ConfDict

    from neuralbench.experiment_config import prepare_task_configs
    from neuralbench.main import Data
    from neuralbench.registry import DEFAULTS_DIR, load_yaml_config
    from neuralbench.sklearn_baseline import CovTsRidge

    config = ConfDict(load_yaml_config(DEFAULTS_DIR / "config.yaml"))
    grid = ConfDict(load_yaml_config(DEFAULTS_DIR / "grid.yaml"))
    configs = prepare_task_configs(
        config,
        grid,
        "meg",
        "image",
        use_task_grid=False,
        debug=False,
        force=False,
        prepare=False,
        download=False,
        models=["cov_ts_ridge"],
        datasets=None,
    )
    assert len(configs) >= 1
    merged = configs[0]
    assert merged["brain_model_name"] == "cov_ts_ridge"
    # The baseline YAML disables clamp (covariances need full dynamic range);
    # this is load-bearing for MEG so check the merge preserved it.
    assert merged["data"]["neuro"]["clamp"] is None
    # The Data config still builds a valid ns.Chain study.
    data = Data(**merged["data"])
    assert data is not None
    # The brain_model_config round-trips through the pydantic discriminator
    # to the concrete CovTsRidge class.
    model_config = merged["brain_model_config"]
    assert model_config["name"] == "CovTsRidge"
    instance = CovTsRidge(**{k: v for k, v in model_config.items() if k != "name"})
    assert instance.cov_estimator == "oas"


def test_collapse_relabels_meg_sklearn_row_as_handcrafted() -> None:
    df = pd.DataFrame(
        [
            {"brain_model_name": "cov_ts_ridge", "task_name": "image"},
            {"brain_model_name": "chance", "task_name": "image"},
            {"brain_model_name": "EEGNet", "task_name": "image"},
        ]
    )
    out = _collapse_feature_based_baselines(df)
    # The sklearn row is relabeled; other rows pass through untouched.
    assert (out["brain_model_name"] == "feature_based").sum() == 1
    assert {"chance", "EEGNet"}.issubset(set(out["brain_model_name"]))
    # And the alias we collapse to is never a member of SKLEARN_BASELINE_MODELS
    # (otherwise the plot-time display map would look it up as a pipeline name).
    assert "feature_based" not in SKLEARN_BASELINE_MODELS
