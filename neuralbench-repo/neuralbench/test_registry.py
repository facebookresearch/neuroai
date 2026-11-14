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

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from neuralbench import registry
from neuralbench.data import Data
from neuralbench.experiment_config import merge_task_config
from neuralbench.plots.tables import _collapse_feature_based_baselines
from neuralbench.registry import (
    ALL_DATASETS,
    DEVICE_BASELINE_MODELS,
    DEVICE_CLASSIC_MODELS,
    DEVICE_FM_MODELS,
    FEATURE_BASED_BY_TASK,
    SKLEARN_BASELINE_MODELS,
    _expand_models,
    _resolve_dataset_stem,
    _task_aware_baseline,
)
from neuralset.extractors.neuro import FnirsExtractor


def test_include_resolves_within_a_package(tmp_path: Path) -> None:
    rel_path = "models/channel_mappings/standard_1020.json"
    config_path = tmp_path / "model.yaml"
    config_path.write_text(f"mapping: !include neuralbench:{rel_path}\n")
    config = registry.load_yaml_config(config_path)
    assert config is not None
    assert config["mapping"] == json.loads(
        (Path(registry.__file__).parent / rel_path).read_text("utf8")
    )


def test_space_is_rejected_on_a_task_without_spaces() -> None:
    with pytest.raises(ValueError, match=r"Unknown space\(s\) \['mni'\]"):
        registry._resolve_spaces("eeg", "motor_imagery", "mni")


def test_every_listed_dataset_name_resolves() -> None:
    for device, tasks in ALL_DATASETS.items():
        for task, names in tasks.items():
            stems = [_resolve_dataset_stem(device, task, name) for name in names]
            assert stems[0] is None, f"{device}/{task}: default must use config.yaml"
            assert None not in stems[1:], f"{device}/{task}: {names} -> {stems}"
            lower = [_resolve_dataset_stem(device, task, n.lower()) for n in names]
            assert lower == stems, f"{device}/{task}: lowercase names resolve differently"


def test_device_baseline_models_meg_has_riemannian_pipelines() -> None:
    meg = DEVICE_BASELINE_MODELS["meg"]
    assert "chance" in meg and "dummy" in meg
    assert "cov_ts_lr" in meg
    assert "cov_ts_ridge" in meg
    # Xdawn covers the event-locked MEG tasks; CoSpectra stays EEG-only.
    assert "xdawn_ts_lr" in meg
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


_FNIRS_NETS = {
    "fnirsnet": "FNIRSNet",
    "fnirst": "FNIRST",
    "fnirspret": "FNIRSPreT",
    "fnirs4d": "FNIRS4D",
    "fnirscnn1d": "FNIRSCNN1D",
    "fnirsstatnet": "FNIRSStatNet",
    "fnirslstm": "FNIRSLSTM",
    "fnirsmlp": "FNIRSMLP",
    "fnirsgcn": "FNIRSGCN",
    "geonirs": "GeoNIRS",
    "mtrenirs": "MtReNIRS",
    "jungmae": "JungMAE",
}
_FNIRS_BASELINES = {
    "mean_slope_lda": "MeanSlopeLDA",
    "statistical_rbf_svm": "StatisticalRBFSVM",
    "hrf_glm_lr": "HRFGLMLR",
    "hemodynamic_knn": "HemodynamicKNN",
    "hemodynamic_rf": "HemodynamicRF",
    "hemodynamic_gbdt": "HemodynamicGBDT",
    "hemodynamic_nb": "HemodynamicNB",
}


def test_fnirs_device_lists() -> None:
    assert DEVICE_CLASSIC_MODELS["fnirs"] + DEVICE_FM_MODELS["fnirs"] == list(_FNIRS_NETS)
    assert DEVICE_FM_MODELS["fnirs"] == ["geonirs", "mtrenirs", "jungmae"]
    # No fNIRS baseline is task-dispatched: every task runs all of them.
    assert _task_aware_baseline("fnirs", "motor_execution") == [
        "chance",
        "dummy",
        *_FNIRS_BASELINES,
    ]


@pytest.mark.parametrize("model,cls_name", {**_FNIRS_NETS, **_FNIRS_BASELINES}.items())
def test_fnirs_model_config_names_its_class(model: str, cls_name: str) -> None:
    config = registry.load_yaml_config(registry._resolve_model_config_path(model))
    assert config is not None
    assert config["brain_model_config"]["name"] == cls_name


def test_fnirs_defaults_build_the_fnirs_extractor() -> None:
    neuro = dict(registry.load_default_config("fnirs")["data"]["neuro"])
    assert neuro.pop("name") == "FnirsExtractor"
    FnirsExtractor(**neuro)


def _fnirs_chs_info(n_pairs: int) -> list[dict]:
    """HbO then HbR channels of ``n_pairs`` source-detector pairs, 3 cm apart on an
    8 cm ring, with midpoint, source and detector in ``loc[:9]`` as MNE stores them."""
    chs = []
    for kind in ("hbo", "hbr"):
        for i in range(n_pairs):
            src = np.array([0.08 * np.cos(i), 0.08 * np.sin(i), 0.05])
            det = src + np.array([0.03, 0.0, 0.0])
            loc = np.concatenate([(src + det) / 2, src, det, np.zeros(3)])
            chs.append({"ch_name": f"S{i + 1}_D{i + 1} {kind}", "kind": kind, "loc": loc})
    return chs


@pytest.mark.parametrize("cls_name", _FNIRS_NETS.values())
def test_fnirs_braindecode_model_forward(cls_name: str) -> None:
    bdm = pytest.importorskip("braindecode.models")
    if not hasattr(bdm, cls_name):
        pytest.skip(f"installed braindecode has no {cls_name}")
    torch.manual_seed(0)
    kwargs: dict = {"n_chans": 40}
    if cls_name in ("GeoNIRS", "MtReNIRS"):  # both need optode geometry in chs_info
        kwargs = {"chs_info": _fnirs_chs_info(20)}
    model = getattr(bdm, cls_name)(n_outputs=2, n_times=150, sfreq=5.0, **kwargs)
    with torch.no_grad():
        out = model.eval()(torch.randn(2, 40, 150))
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


def test_task_aware_baseline_unknown_task_drops_sklearn_pipelines() -> None:
    # For tasks not in FEATURE_BASED_BY_TASK, no sklearn pipeline is appended.
    assert _task_aware_baseline("meg", "__nonexistent_task__") == ["chance", "dummy"]


def test_unvalidated_task_keeps_its_handcrafted_pipeline() -> None:
    baselines = _task_aware_baseline("meg", "_emotion")
    assert baselines == ["chance", "dummy", "cov_ts_lr"]
    df = pd.DataFrame(
        {"brain_model_name": ["cov_ts_lr", "cov_ts_ridge"], "task_name": "_emotion"}
    )
    assert _collapse_feature_based_baselines(df)["brain_model_name"].tolist() == [
        "feature_based"
    ]


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

    from neuralbench.baselines import CovTsRidge
    from neuralbench.experiment_config import prepare_task_configs
    from neuralbench.main import Data
    from neuralbench.registry import DEFAULTS_DIR, load_default_config, load_yaml_config

    config = ConfDict(load_default_config("meg"))
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


def test_load_default_config_overlays_plugin_device_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    device_dir = tmp_path / "ieeg"
    device_dir.mkdir()
    (device_dir / "config.yaml").write_text("data:\n  neuro:\n    name: IeegExtractor\n")
    monkeypatch.setattr(
        registry, "_all_defaults_roots", lambda: [registry.DEFAULTS_DIR, tmp_path]
    )
    overlaid = registry.load_default_config("ieeg")
    base = registry.load_default_config()
    assert overlaid["data"]["neuro"]["name"] == "IeegExtractor"
    assert "name" not in base["data"]["neuro"]
    # A device with no overlay falls back to the base.
    assert registry.load_default_config("unknown") == base
    # The overlay is partial: base keys it does not mention survive.
    assert overlaid["data"]["neuro"]["infra"] == base["data"]["neuro"]["infra"]


@pytest.mark.parametrize(
    "device", sorted(p.parent.name for p in registry.DEFAULTS_DIR.glob("*/config.yaml"))
)
def test_device_defaults_set_a_default_model(device: str) -> None:
    assert "brain_model_config" in registry.load_default_config(device), (
        f"defaults/{device}/config.yaml needs a brain_model_config: exca keeps "
        "'=replace=' literally when models/*.yaml replace a missing key"
    )


def test_fmri_image_data_builds_without_channel_positions() -> None:
    data = Data(**merge_task_config("fmri", "image")["data"])
    assert data.neuro.name == "FmriExtractor"
    assert data.channel_positions is None


def test_debug_study_queries_merge_plugin_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "debug_study.yaml").write_text("Plugin2026Study: subject_index < 3\n")
    monkeypatch.setattr(
        registry, "_all_defaults_roots", lambda: [registry.DEFAULTS_DIR, tmp_path]
    )
    queries = registry._build_debug_study_queries()
    assert queries["Plugin2026Study"] == "subject_index < 3"
    assert "Lopez2017Tuab" in queries
