# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
from pathlib import Path

import pytest
import torch
from exca import ConfDict

from neuralbench import registry
from neuralbench.data import Data
from neuralbench.defaults.metrics import (
    get_classification_metric_configs,
    get_sleep_onset_metric_configs,
)
from neuralbench.experiment_config import (
    _adapts_a_backbone,
    _expand_grid,
    _warn_unsupported_gpu,
    merge_task_config,
)
from neuralbench.registry import (
    ALL_DOWNSTREAM_WRAPPERS,
    DEFAULTS_DIR,
    FM_MODELS,
    _resolve_model_config_path,
    _resolve_task_dir,
    load_yaml_config,
)
from neuraltrain.optimizers.base import LightningOptimizer


def _expand(config: dict, grid: dict, debug: bool = False):
    return _expand_grid(
        ConfDict(config),
        ConfDict(grid),
        device="eeg",
        task_name="dummy",
        use_task_grid=False,
        prepare=False,
        debug=debug,
        quiet=True,
    )


def test_expand_grid_adaptation_overlay_merges_atomically():
    base = {"seed": 0, "downstream_model_wrapper": {"aggregation": "mean"}}
    overlays = [
        {
            "downstream_model_wrapper": {
                "aggregation": "flatten",
                "probe_config": "linear",
            },
            "lightning_optimizer_config": {"optimizer": {"lr": 5.0e-4}},
        },
        {"downstream_model_wrapper": {"aggregation": "attention"}},
    ]
    configs = _expand(base, {"seed": [0, 1], "_adaptation_overlay": overlays})

    assert len(configs) == 4, "2 seeds x 2 overlays; the overlay list stays one axis"
    for cfg in configs:
        assert "_adaptation_overlay" not in cfg.flat()

    flatten_cfgs = [
        c for c in configs if c["downstream_model_wrapper.aggregation"] == "flatten"
    ]
    assert len(flatten_cfgs) == 2
    for c in flatten_cfgs:
        # all keys of the overlay land together
        assert c["downstream_model_wrapper.probe_config"] == "linear"
        assert c["lightning_optimizer_config.optimizer.lr"] == 5.0e-4


def test_expand_grid_debug_renulls_scheduler_after_overlay():
    base = {"seed": 0}
    overlays = [
        {"lightning_optimizer_config": {"scheduler": {"kwargs": {"max_lr": 1.0}}}}
    ]
    configs = _expand(base, {"_adaptation_overlay": overlays}, debug=True)

    assert len(configs) == 1
    assert configs[0]["lightning_optimizer_config.scheduler"] is None


@functools.cache
def _base_and_model_config(model_name: str) -> ConfDict:
    config = ConfDict(load_yaml_config(DEFAULTS_DIR / "config.yaml"))
    config.update(load_yaml_config(_resolve_model_config_path(model_name)))
    return config


@pytest.mark.parametrize(
    ("model_name", "expected"),
    # mae is a backbone without published weights, so it is absent from FM_MODELS
    [
        *((name, True) for name in [*FM_MODELS, "mae"]),
        ("eegnet", False),
        ("chance", False),
    ],
)
def test_adaptation_applies_to_backbones_only(model_name: str, expected: bool):
    assert _adapts_a_backbone(model_name) is expected


def test_unsupported_gpu_check_survives_a_driver_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _too_old(index: int) -> tuple[int, int]:
        raise RuntimeError("driver too old")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90"])
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", _too_old)
    with pytest.warns(UserWarning, match="driver too old"):
        _warn_unsupported_gpu()


def test_plugin_root_adds_dataset_to_shipped_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "eeg" / "seizure" / "datasets" / "plugin2026.yaml"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("data:\n  study:\n    source:\n      name: Plugin2026Eeg\n")
    monkeypatch.setattr(
        registry, "_all_task_roots", lambda: [registry.BASE_DIR / "tasks", tmp_path]
    )
    config = merge_task_config("eeg", "seizure", "plugin2026")
    assert config["data.study.source.name"] == "Plugin2026Eeg"
    assert config["data.study.split.split_by"] == "subject", "task split not inherited"


def test_replaced_study_drops_default_study_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "eeg" / "word" / "datasets" / "plugin2026.yaml"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        "data:\n  study:\n    =replace=: true\n    source:\n      name: Plugin2026Eeg\n"
    )
    monkeypatch.setattr(
        registry, "_all_task_roots", lambda: [registry.BASE_DIR / "tasks", tmp_path]
    )
    source = merge_task_config("eeg", "word", "plugin2026")["data.study.source"]
    assert "query" not in source, f"default study's query leaked: {source['query']!r}"
    assert {"path", "infra"} <= set(source)


@pytest.mark.parametrize("preset", list(ALL_DOWNSTREAM_WRAPPERS))
@pytest.mark.parametrize("model_name", FM_MODELS)
def test_adaptation_overlay_leaves_a_valid_optimizer(model_name: str, preset: str):
    config = _base_and_model_config(model_name).copy()
    config.update(ALL_DOWNSTREAM_WRAPPERS[preset])
    # a model YAML with scheduler=null would come back nameless: overlays set only max_lr
    LightningOptimizer(**dict(config["lightning_optimizer_config"]))


_UNSCALED = {
    "data.neuro.scaler": None,
    "data.neuro.clamp": None,
    "data.neuro.scale_factor": 1e6,
}
_STREAM_ONLY_DIFF = {
    "sleep_onset": {
        **_UNSCALED,
        "data.neuro.frequency": "native",
        "data.neuro.filter": None,
        "data.neuro.notch_filter": None,
        "data.stream_by": ["timeline"],
        "trainer_config.monitor": "val/wbmae_stream_mean",
    },
    "motor_imagery": {
        **_UNSCALED,
        "data.stream_by": ["subject", "session"],
        "trainer_config.monitor": "val/bal_acc_stream_mean",
    },
}


def _bal_acc_stream_mean(n: int) -> dict:
    return {
        "log_name": "bal_acc_stream_mean",
        "name": "GroupedMetric",
        "metric_name": "Accuracy",
        "kwargs": {"task": "multiclass", "num_classes": n, "average": "macro"},
        "reduction": "mean",
    }


_WBMAE_STREAM_MEAN = {
    "log_name": "wbmae_stream_mean",
    "name": "GroupedMetric",
    "metric_name": "BinnedMAE",
    "kwargs": {
        "bin_boundaries": [0.0, 40.0, 90.0, 300.0, 600.0],
        "bin_weights": [10.0, 5.0, 3.0, 1.0],
    },
    "reduction": "mean",
}
# task -> stream metrics, from the number of outputs; mirrors the
# operator.add blocks in the stream task configs
_STREAM_METRICS = {
    "sleep_onset": lambda n: get_sleep_onset_metric_configs() + [_WBMAE_STREAM_MEAN],
    "motor_imagery": lambda n: (
        get_classification_metric_configs(n) + [_bal_acc_stream_mean(n)]
    ),
}
# (task, stream dataset) -> extra diff; Muse is curated, so no random start
_STREAM_DATASET_DIFF: dict[tuple[str, str | None], dict[str, list[str]]] = {
    ("sleep_onset", dataset): {
        "data.study.annotate_sleep_onset.random_start_splits": ["val", "test"]
    }
    for dataset in ["kemp2000analysis", "alvarez2022haaglanden", "ghassemi2018you"]
}
# stream dataset -> core dataset, where they differ (None: task default)
_STREAM_TO_CORE_DATASET: dict[str, dict[str | None, str | None]] = {
    "sleep_onset": {None: "interaxon2026muse", "kemp2000analysis": None},
    "motor_imagery": {None: "dreyer2026proteus"},
}


@pytest.mark.parametrize(
    "task,dataset",
    [
        (task, dataset)
        for task in _STREAM_ONLY_DIFF
        for dataset in [
            None,
            *(
                p.stem
                for p in (_resolve_task_dir("eeg", f"_{task}_stream") / "datasets").glob(
                    "*.yaml"
                )
            ),
        ]
    ],
)
def test_stream_task_diff(task: str, dataset: str | None):
    core_dataset = _STREAM_TO_CORE_DATASET[task].get(dataset, dataset)
    core_config = merge_task_config("eeg", task, core_dataset)
    stream_config = merge_task_config("eeg", f"_{task}_stream", dataset)
    for config in (core_config, stream_config):
        Data(**config["data"])
    core, stream = core_config.flat(), stream_config.flat()
    diff = {k: stream.get(k) for k in core | stream if core.get(k) != stream.get(k)}
    metrics = _STREAM_METRICS[task](stream["brain_model_output_size"])
    assert diff == {
        **_STREAM_ONLY_DIFF[task],
        **_STREAM_DATASET_DIFF.get((task, dataset), {}),
        **ConfDict(metrics=metrics).flat(),
        "data.test_batch_size": 1,
    }
