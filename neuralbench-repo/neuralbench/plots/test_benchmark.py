# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the output layout of :func:`neuralbench.plots.benchmark.plot_all_results`."""

from __future__ import annotations

import typing as tp
from pathlib import Path

import pytest

from neuralbench.aggregator import BenchmarkAggregator
from neuralbench.plots import benchmark

_MAPPING: dict[str, str] = BenchmarkAggregator.model_fields[
    "loss_to_metric_mapping"
].default


def _row(model: str, eval_mode: str) -> dict[str, tp.Any]:
    return {
        "loss": {"name": "MultiLoss"},
        "brain_model_name": model,
        "task_name": "sleep_onset",
        "seed": 0,
        "eval_mode": eval_mode,
        "test/bmae": 1.0,
    }


@pytest.mark.parametrize(
    ("fm_modes", "expected"),
    [
        ((), {"core/default", "full/default"}),
        (
            ("finetune_mean", "lora_r32_flatten"),
            {
                "core/finetune_mean",
                "full/finetune_mean",
                "core/lora_r32_flatten",
                "full/lora_r32_flatten",
                "core/all_strategies",
            },
        ),
    ],
)
def test_every_run_writes_one_folder_per_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fm_modes: tuple[str, ...],
    expected: set[str],
) -> None:
    written: set[str] = set()

    def record(*dirs: Path) -> None:
        written.update(d.relative_to(tmp_path).as_posix() for d in dirs)

    def fake_steps(core_df, full_df, core_dir, full_dir):
        record(core_dir, full_dir)
        return []

    monkeypatch.setattr(benchmark, "_core_full_steps", fake_steps)
    monkeypatch.setattr(benchmark, "plot_core_rank_boxplot", lambda df, d, **_: record(d))
    monkeypatch.setattr(benchmark, "plot_adaptation_comparison", lambda df, d: record(d))
    monkeypatch.setattr(benchmark, "_load_optional_scaling", lambda: (None, None))

    results = [_row("EEGNet", "default")] + [_row("NtReve", m) for m in fm_modes]
    benchmark.plot_all_results(results, _MAPPING, tmp_path)
    assert written == expected
