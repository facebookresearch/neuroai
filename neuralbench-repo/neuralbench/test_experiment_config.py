# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools

import pytest
import torch
from exca import ConfDict

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
    get_available_datasets,
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


@pytest.mark.parametrize("preset", list(ALL_DOWNSTREAM_WRAPPERS))
@pytest.mark.parametrize("model_name", FM_MODELS)
def test_adaptation_overlay_leaves_a_valid_optimizer(model_name: str, preset: str):
    config = _base_and_model_config(model_name).copy()
    config.update(ALL_DOWNSTREAM_WRAPPERS[preset])
    # a model YAML with scheduler=null would come back nameless: overlays set only max_lr
    LightningOptimizer(**dict(config["lightning_optimizer_config"]))


@pytest.mark.parametrize(
    "dataset", [None, *get_available_datasets("eeg", "_sleep_onset_stream")]
)
def test_sleep_onset_stream_diff(dataset: str | None):
    """The stream task is the single-window cut formulation: one fixed-length
    crop per night ending at a random cut around onset, with a signed target
    and no window grid (so no window-count or window-length leak)."""
    cfg = merge_task_config("eeg", "_sleep_onset_stream", dataset).flat()
    assert cfg["data.study.annotate_sleep_onset.name"] == "AddSleepOnsetCutTargets"
    # Single-window segmentation: one segment per marker, so its fixed length
    # carries no information about the target.
    assert cfg.get("data.duration") is None
    assert cfg.get("data.stride") is None
    # Signed target: a cut past onset reports a negative time-to-onset.
    assert cfg["data.target.floor_s"] < 0
    # One crop per night on eval (no counting cue), more on train.
    assert cfg["data.study.annotate_sleep_onset.n_eval_windows"] == 1
    assert cfg["data.study.annotate_sleep_onset.n_train_windows"] >= 1
    # No streaming machinery left.
    assert not cfg.get("reset_per_timeline", False)
    assert cfg.get("data.test_batch_size") is None
