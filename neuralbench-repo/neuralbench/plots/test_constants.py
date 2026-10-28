# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`neuralbench.plots._constants`."""

from __future__ import annotations

import pytest

from neuralbench.aggregator import BenchmarkAggregator
from neuralbench.plots._constants import (
    FM_DISPLAY,
    METRIC_HIGHER_IS_BETTER,
    METRIC_PERFECT_SCORE,
    AdaptationMode,
    eval_mode_label,
    model_group,
    strip_eval_mode_suffix,
)
from neuralbench.plots.tables import eval_mode_suffix


@pytest.mark.parametrize(
    "metric",
    BenchmarkAggregator.model_fields["loss_to_metric_mapping"].default.values(),
)
def test_headline_metrics_have_a_direction_and_perfect_score(metric: str) -> None:
    assert metric in METRIC_HIGHER_IS_BETTER, "normalization would assume higher=better"
    assert metric in METRIC_PERFECT_SCORE, "the perfect ceiling would default to 100"


@pytest.mark.parametrize(
    "tag",
    [
        "finetune",
        "finetune_mean",
        "linear_probe_flatten",
        "attentive_probe",
        "lora_r4",
        "lora_r32_mean",
        "lpft_attentive",
        "some_future_strategy",
    ],
)
def test_adaptation_mode_round_trips(tag: str) -> None:
    assert AdaptationMode.parse(tag).tag == tag


@pytest.mark.parametrize(
    "tag, label",
    [
        ("finetune_mean", "Full FT (mean)"),
        ("linear_probe_flatten", "Linear Probe (flatten)"),
        ("lora_r8", "LoRA r8"),
        ("lpft_attentive", "LP-FT (attentive)"),
        ("some_future_strategy", "some_future_strategy"),
    ],
)
def test_eval_mode_label(tag: str, label: str) -> None:
    assert eval_mode_label(tag) == label


@pytest.mark.parametrize(
    "tag",
    [
        "finetune",
        "finetune_mean",
        "linear_probe_flatten",
        "attentive_probe",
        "lora_r32",
        "lpft",
        "lpft_attentive",
    ],
)
def test_strategy_suffix_strips_back_to_the_base_name(tag: str) -> None:
    name = FM_DISPLAY[0] + eval_mode_suffix(tag)
    assert name != FM_DISPLAY[0]
    assert strip_eval_mode_suffix(name) == FM_DISPLAY[0]
    assert model_group(name) == "foundation"
