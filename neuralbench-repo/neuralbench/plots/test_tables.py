# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`neuralbench.plots.tables`."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralbench.aggregator import BenchmarkAggregator
from neuralbench.plots.tables import (
    build_results_df,
    filter_results_for_eval_mode,
    foundation_eval_modes,
    make_rank_table,
    make_results_table,
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
    task_name: str = "sleep_onset",
    **metric_values: float,
) -> dict:
    """Build one synthetic result row with the columns ``build_results_df`` reads.

    ``seed`` distinguishes replicate rows so they aren't flagged as colliding
    configs by ``build_results_df``'s collision guard.
    """
    row: dict = {
        "loss": {"name": loss_name},
        "brain_model_name": brain_model_name,
        "task_name": task_name,
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


def test_build_results_df_tags_spaces_only_where_a_dataset_has_several():
    spaces = {
        "Bold5000": ["mni", "mni2fsaverage5"],
        "Allen2022MassiveRaw": ["fsaverage5"],
    }
    results = [
        {
            **_row(loss_name="MultiLoss", **{"test/bmae": 1.0}),
            "dataset_name": d,
            "space_name": s,
        }
        for d, names in spaces.items()
        for s in names
    ]
    df = build_results_df(results, _DEFAULT_MAPPING)
    assert dict(zip(df["space_name"], df["model_name"])) == {
        "mni": "EEGNet [space=mni]",
        "mni2fsaverage5": "EEGNet [space=mni2fsaverage5]",
        "fsaverage5": "EEGNet",
    }


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


@pytest.mark.parametrize(
    ("values", "expected"),
    [((42.0,), "42.000"), ((42.0, 44.0), "43.000 \u00b1 1.414")],
)
def test_results_table_reports_spread_only_when_seeds_define_one(
    tmp_path: Path, values: tuple[float, ...], expected: str
):
    results = [
        _row(loss_name="MultiLoss", seed=seed) | {"test/bmae": value}
        for seed, value in enumerate(values)
    ]
    table = make_results_table(build_results_df(results, _DEFAULT_MAPPING), tmp_path)
    assert table.iloc[0, 0] == expected


def test_rank_table_average_covers_only_tasks_every_model_ranked(tmp_path: Path):
    # Chance alone scores "age", so averaging it in would rank models on
    # different task sets.
    results = [
        _row(loss_name="MultiLoss", brain_model_name=model, task_name=task)
        | {"test/bmae": value}
        for model, task, value in [
            ("EEGNet", "sleep_onset", 1.0),
            ("chance", "sleep_onset", 2.0),
            ("chance", "age", 3.0),
        ]
    ]
    df = build_results_df(results, _DEFAULT_MAPPING)
    with pytest.warns(UserWarning, match="age"):
        table = make_rank_table(df, tmp_path)
    # averaging "age" in would lift chance to 1.5 on a task EEGNet never ran
    assert table.loc["average"].to_dict() == {"EEGNet": 1.0, "chance": 2.0}
