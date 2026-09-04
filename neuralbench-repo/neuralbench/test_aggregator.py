# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`neuralbench.aggregator`."""

import typing as tp
from types import SimpleNamespace

from exca import ConfDict

from .aggregator import BenchmarkAggregator, _infer_eval_mode
from .modules import DownstreamWrapper
from .registry import (
    ALL_DATASETS,
    ALL_DOWNSTREAM_WRAPPERS,
    DEFAULTS_DIR,
    _resolve_task_dir,
    load_yaml_config,
)


def _eval_mode(wrapper: DownstreamWrapper | None) -> str:
    return _infer_eval_mode(
        tp.cast(tp.Any, SimpleNamespace(downstream_model_wrapper=wrapper))
    )


def test_infer_eval_mode_tags_each_shipped_preset_distinctly():
    modes = {
        name: _eval_mode(DownstreamWrapper(**preset["downstream_model_wrapper"]))
        for name, preset in ALL_DOWNSTREAM_WRAPPERS.items()
    }
    # presets sharing a tag would merge everywhere downstream: plots, tables, folders
    assert modes == {name: name for name in ALL_DOWNSTREAM_WRAPPERS}


def test_infer_eval_mode_without_wrapper_is_finetune():
    assert _eval_mode(None) == "finetune"


def test_every_shipped_task_loss_has_a_headline_metric():
    mapping = BenchmarkAggregator.model_fields["loss_to_metric_mapping"].default
    defaults = load_yaml_config(DEFAULTS_DIR / "config.yaml")
    assert defaults is not None
    base = ConfDict(defaults)
    for device, tasks in ALL_DATASETS.items():
        for task_name in tasks:
            task_cfg = load_yaml_config(
                _resolve_task_dir(device, task_name) / "config.yaml"
            )
            if task_cfg is None:
                continue
            merged = base.copy()
            merged.update(task_cfg)
            loss_name = merged.flat().get("loss.name")
            assert loss_name in mapping, (
                f"{device}/{task_name}: loss {loss_name!r} has no headline metric, "
                "so --plot-cached raises instead of aggregating the task"
            )
