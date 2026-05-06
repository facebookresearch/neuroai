# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the dataset-level variability analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neuralbench.plots.variability import (
    compute_model_task_rank_variability,
    compute_task_dataset_ranks,
    plot_full_variability_panel,
)

_MODEL_A = "EEGNet"  # has to be a known display name for the plot path
_MODEL_B = "Dummy"


def _make_toy_df() -> pd.DataFrame:
    """Two paradigms: 'p_stable' (identical rankings) and 'p_unstable' (flipped)."""
    rows: list[dict] = []
    # p_stable: A always outperforms B across 3 datasets.
    for ds in ["d1", "d2", "d3"]:
        for model, val in ((_MODEL_A, 0.9), (_MODEL_B, 0.6)):
            rows.append(
                dict(
                    task_name="p_stable",
                    dataset_name=ds,
                    model_name=model,
                    metric_value=val,
                    metric_name="test/bal_acc",
                )
            )
    # p_unstable: rankings flip on 1/3 of the datasets.
    pairs = [("d1", 0.9, 0.6), ("d2", 0.6, 0.9), ("d3", 0.9, 0.6)]
    for ds, a, b in pairs:
        rows.append(
            dict(
                task_name="p_unstable",
                dataset_name=ds,
                model_name=_MODEL_A,
                metric_value=a,
                metric_name="test/bal_acc",
            )
        )
        rows.append(
            dict(
                task_name="p_unstable",
                dataset_name=ds,
                model_name=_MODEL_B,
                metric_value=b,
                metric_name="test/bal_acc",
            )
        )
    return pd.DataFrame(rows)


def test_compute_task_dataset_ranks_shape() -> None:
    df = _make_toy_df()
    ranks = compute_task_dataset_ranks(df)
    # MultiIndex (task, dataset) x (model).
    assert list(ranks.index.names) == ["task_name", "dataset_name"]
    assert ranks.shape == (6, 2)
    assert ranks.loc[("p_stable", "d1"), _MODEL_A] == 1.0
    assert ranks.loc[("p_stable", "d1"), _MODEL_B] == 2.0


def test_compute_model_task_rank_variability_std() -> None:
    df = _make_toy_df()
    matrix = compute_model_task_rank_variability(df)
    # Toy data: 2 models x 2 multi-dataset paradigms.
    assert matrix.shape == (2, 2)
    assert set(matrix.index) == {_MODEL_A, _MODEL_B}
    assert set(matrix.columns) == {"p_stable", "p_unstable"}
    # p_stable: ranks are constant -> std = 0 for every model.
    assert matrix.loc[_MODEL_A, "p_stable"] == 0.0
    assert matrix.loc[_MODEL_B, "p_stable"] == 0.0
    # p_unstable: per-model ranks across 3 datasets are [1, 2, 1] (or
    # [2, 1, 2]) -> sample std (ddof=1) = sqrt(1/3) ~= 0.5774 for both.
    expected_std = (1.0 / 3.0) ** 0.5
    assert matrix.loc[_MODEL_A, "p_unstable"] == pytest.approx(expected_std)
    assert matrix.loc[_MODEL_B, "p_unstable"] == pytest.approx(expected_std)


def test_plot_full_variability_panel_returns_none_without_multi_dataset(
    tmp_path: Path,
) -> None:
    """No multi-dataset paradigm -> heatmap can't render -> panel returns None."""
    df = pd.DataFrame(
        [
            dict(
                task_name="solo",
                dataset_name="d1",
                model_name=_MODEL_A,
                metric_value=0.5,
                metric_name="test/bal_acc",
            )
        ]
    )
    assert plot_full_variability_panel(df, df, tmp_path, min_datasets=1) is None


def test_plot_full_variability_panel_endtoend(tmp_path: Path) -> None:
    df = _make_toy_df()
    png_path = plot_full_variability_panel(df, df, tmp_path, min_datasets=1)
    assert png_path is not None and png_path.exists()
    assert png_path.name == "full_variability_panel.png"
    assert (tmp_path / "full_variability_panel.pdf").exists()
