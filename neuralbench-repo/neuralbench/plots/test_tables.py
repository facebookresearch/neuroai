# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`neuralbench.plots.tables`."""

from __future__ import annotations

import pytest

from neuralbench.aggregator import BenchmarkAggregator
from neuralbench.plots.tables import (
    build_results_df,
    filter_results_for_eval_mode,
    foundation_eval_modes,
)

_DEFAULT_MAPPING: dict[str, str] = BenchmarkAggregator.model_fields[
    "loss_to_metric_mapping"
].default


def _row(
    *,
    loss_name: str,
    brain_model_name: str = "EEGNet",
    eval_mode: str | None = None,
    seed: int = 0,
    **metric_values: float,
) -> dict:
    """Build one synthetic result row with the columns ``build_results_df`` reads.

    ``seed`` distinguishes replicate rows so they aren't flagged as colliding
    configs by ``build_results_df``'s collision guard.
    """
    row: dict = {
        "loss": {"name": loss_name},
        "brain_model_name": brain_model_name,
        "task_name": "sleep_onset",
        "seed": seed,
        **metric_values,
    }
    if eval_mode is not None:
        row["eval_mode"] = eval_mode
    return row


def test_build_results_df_resolves_multi_loss_to_bmae():
    """Sleep-onset rows logged with ``MultiLoss`` must select ``test/bmae``."""
    results = [_row(loss_name="MultiLoss", **{"test/bmae": 42.0})]
    df = build_results_df(results, _DEFAULT_MAPPING)
    assert df["metric_name"].tolist() == ["test/bmae"]
    assert df["metric_value"].tolist() == [42.0]


def test_build_results_df_raises_for_unmapped_loss():
    """An unknown loss name surfaces a clear error, not ``KeyError: nan``."""
    results = [_row(loss_name="NewlyAddedLoss", **{"test/something": 1.0})]
    with pytest.raises(KeyError, match="NewlyAddedLoss"):
        build_results_df(results, _DEFAULT_MAPPING)


def test_build_results_df_single_eval_mode_keeps_bare_model_name():
    results = [
        _row(loss_name="MultiLoss", eval_mode="finetune", seed=0, **{"test/bmae": 1.0}),
        _row(loss_name="MultiLoss", eval_mode="finetune", seed=1, **{"test/bmae": 2.0}),
    ]
    df = build_results_df(results, _DEFAULT_MAPPING)
    assert df["model_name"].unique().tolist() == ["EEGNet"]


def test_build_results_df_multi_eval_mode_suffixes_model_name():
    results = [
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="finetune_mean",
            **{"test/bmae": 1.0},
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="linear_probe_flatten",
            **{"test/bmae": 2.0},
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="linear_probe_mean",
            **{"test/bmae": 3.0},
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="lora_r4_flatten",
            **{"test/bmae": 4.0},
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="attentive_probe",
            **{"test/bmae": 5.0},
        ),
    ]
    df = build_results_df(results, _DEFAULT_MAPPING)
    # REVE is the display name for NtReve
    expected = {
        "REVE (FT mean)",
        "REVE (LP flatten)",
        "REVE (LP mean)",
        "REVE (AP)",
        "REVE (LoRA r4 flatten)",
    }
    assert set(df["model_name"]) == expected
    assert df["base_model_name"].unique().tolist() == ["REVE"]


def test_build_results_df_non_fm_eval_mode_does_not_suffix_foundation_models():
    results = [
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="finetune_mean",
            seed=0,
            **{"test/bmae": 1.0},
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="finetune_mean",
            seed=1,
            **{"test/bmae": 2.0},
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="EEGNet",
            eval_mode="linear_probe_mean",
            **{"test/bmae": 3.0},
        ),
    ]
    df = build_results_df(results, _DEFAULT_MAPPING)
    assert set(df["model_name"]) == {"REVE", "EEGNet"}, (
        "only foundation-model strategies should decide suffixing"
    )


def test_build_results_df_defaults_missing_eval_mode_to_finetune():
    results = [
        _row(loss_name="MultiLoss", brain_model_name="NtReve", **{"test/bmae": 1.0}),
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="linear_probe",
            **{"test/bmae": 2.0},
        ),
    ]
    df = build_results_df(results, _DEFAULT_MAPPING)
    assert set(df["model_name"]) == {"REVE (FT)", "REVE (LP)"}, (
        "a legacy row must be rebranded finetune, not suffixed '(nan)'"
    )


def test_build_results_df_suffixes_only_foundation_models():
    results = [
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="finetune",
            **{"test/bmae": 1.0},
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="linear_probe",
            **{"test/bmae": 2.0},
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="EEGNet",
            eval_mode="finetune",
            **{"test/bmae": 3.0},
        ),
    ]
    df = build_results_df(results, _DEFAULT_MAPPING)
    assert set(df["model_name"]) == {"REVE (FT)", "REVE (LP)", "EEGNet"}


def test_foundation_eval_modes_orders_and_ignores_non_fm():
    results = [
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="lora_r32_flatten",
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="linear_probe_mean",
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="lora_r4_flatten",
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="attentive_probe",
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="linear_probe_flatten",
        ),
        # non-FM: must not contribute "finetune"
        _row(loss_name="MultiLoss", brain_model_name="EEGNet", eval_mode="finetune"),
    ]
    assert foundation_eval_modes(results) == [
        "linear_probe_flatten",
        "linear_probe_mean",
        "attentive_probe",
        "lora_r4_flatten",
        "lora_r32_flatten",
    ]


def test_filter_results_for_eval_mode_keeps_non_fm_reference():
    results = [
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="linear_probe",
            **{"test/bmae": 1.0},
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="NtReve",
            eval_mode="finetune",
            **{"test/bmae": 2.0},
        ),
        _row(
            loss_name="MultiLoss",
            brain_model_name="EEGNet",
            eval_mode="finetune",
            **{"test/bmae": 3.0},
        ),
    ]
    subset = filter_results_for_eval_mode(results, "linear_probe")
    df = build_results_df(subset, _DEFAULT_MAPPING, suffix_eval_mode=False)
    # NtReve@finetune dropped; NtReve@linear_probe + EEGNet kept, both bare.
    assert set(df["model_name"]) == {"REVE", "EEGNet"}
