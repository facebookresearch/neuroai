# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-task / per-(task, dataset) rank computations.

Centralising the rank helpers in a leaf module breaks the circular
import that previously forced ``tables.py`` and ``variability.py`` to
fetch :func:`compute_task_ranks` from :mod:`neuralbench.plots.benchmark`
lazily inside their function bodies.
"""

from __future__ import annotations

import pandas as pd

from neuralbench.plots._constants import (
    METRIC_HIGHER_IS_BETTER,
    TASK_DISPLAY_NAMES,
)
from neuralbench.plots._style import with_year


def compute_task_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Rank models per task (1 = best) based on mean metric value.

    The ranking direction is determined per task from
    ``METRIC_HIGHER_IS_BETTER``.  Ties receive the same (lowest) rank
    (``method="min"``).
    """
    agg = df.groupby(["task_name", "model_name"]).metric_value.mean().reset_index()
    metric_per_task = df.groupby("task_name")["metric_name"].first()
    wide = agg.pivot(
        index="task_name",
        columns="model_name",
        values="metric_value",
    )
    ranks = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for task in wide.index:
        higher = METRIC_HIGHER_IS_BETTER.get(metric_per_task.get(task, ""), True)
        ranks.loc[task] = wide.loc[task].rank(method="min", ascending=not higher)
    return ranks


def compute_task_dataset_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Rank models per ``(task, dataset)`` -- mirrors :func:`compute_task_ranks`.

    Parameters
    ----------
    df
        Long-form DataFrame with at least ``task_name``, ``dataset_name``,
        ``model_name``, ``metric_value`` and ``metric_name`` columns.

    Returns
    -------
    ranks
        Wide DataFrame with MultiIndex ``(task_name, dataset_name)`` on
        the rows and ``model_name`` on the columns.  Ranks are computed
        per row using ``method="min"``; direction is set by
        ``METRIC_HIGHER_IS_BETTER`` (default: higher is better).
    """
    if "dataset_name" not in df.columns:
        raise ValueError(
            "DataFrame is missing the 'dataset_name' column required for "
            "dataset-level variability analysis."
        )

    agg = (
        df.groupby(["task_name", "dataset_name", "model_name"])
        .metric_value.mean()
        .reset_index()
    )
    metric_per_task = df.groupby("task_name")["metric_name"].first()

    wide = agg.pivot(
        index=["task_name", "dataset_name"],
        columns="model_name",
        values="metric_value",
    )
    ranks = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    for idx in wide.index:
        task = idx[0]
        higher = METRIC_HIGHER_IS_BETTER.get(metric_per_task.get(task, ""), True)
        ranks.loc[idx] = wide.loc[idx].rank(method="min", ascending=not higher)
    return ranks


def compute_core_mean_ranks(core_df: pd.DataFrame) -> pd.Series:
    """Return per-model mean rank on the Core benchmark.

    One rank per ``(task, model)`` row averaged across tasks.
    """
    rank_wide = compute_task_ranks(core_df)
    return rank_wide.mean(axis=0).sort_values()


def compute_full_mean_ranks(full_df: pd.DataFrame) -> pd.Series:
    """Return per-model mean rank on the Full benchmark.

    Computes per-``(task, dataset)`` ranks, averages within each task,
    then averages across tasks.
    """
    rank_wide = compute_task_dataset_ranks(full_df)
    per_task = rank_wide.groupby(level="task_name").mean()
    return per_task.mean(axis=0).sort_values()


def _prepare_rank_data(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-task ranks and return a long-form DataFrame with display names."""
    rank_wide = compute_task_ranks(df)
    long_df = rank_wide.reset_index().melt(
        id_vars="task_name",
        var_name="model_name",
        value_name="rank",
    )
    long_df["task_name"] = long_df["task_name"].map(
        lambda x: TASK_DISPLAY_NAMES.get(x, x.replace("_", " ").title())
    )
    long_df["model_name"] = long_df["model_name"].map(with_year)
    return long_df


def compute_row_rank_stats(
    df: pd.DataFrame,
    row_tasks: list[str],
    model_order: list[str],
    *,
    by_dataset: bool = False,
    normalize: bool = True,
) -> pd.DataFrame:
    """Per-model mean + SEM of rank across the cells of a figure row.

    Used by the per-row marginal "mean rank" bar plot in the core and
    full bar-chart grids.  Ranks are computed only over the supplied
    *row_tasks* so each marginal exactly summarises the bars shown on
    its row.  Set *by_dataset* to ``True`` for the full-benchmark chart,
    where each subplot is one ``(task, dataset)`` cell -- the per-cell
    ranks are then averaged across all cells in the row's task(s).

    When *normalize* is true (the default) each per-cell rank is divided
    by the number of models actually compared in that cell, giving a
    value in ``(0, 1]`` -- ``1/N`` for the best model in a cell and
    ``1.0`` for the worst.  This makes the marginal directly comparable
    across rows whose subplots involve different numbers of models
    (e.g. some baselines may not apply to every metric).

    Returns a DataFrame indexed by *model_order* with two columns:

    ``mean``
        Per-model mean of the per-cell ranks (normalised when
        ``normalize=True``).
    ``sem``
        Per-model standard error of the same.  ``NaN`` when fewer than
        two non-NaN ranks contribute (a single-cell row, or a model
        with only one valid rank).

    Models absent from the supplied *df* yield ``NaN`` and are silently
    skipped by the renderer.
    """
    sub = df[df["task_name"].isin(row_tasks)]
    out = pd.DataFrame(index=list(model_order), columns=["mean", "sem"], dtype=float)
    if sub.empty:
        return out
    if by_dataset:
        ranks = compute_task_dataset_ranks(sub)
    else:
        ranks = compute_task_ranks(sub)
    if normalize:
        # Per-cell N = number of models that received a rank in that cell.
        # Dividing the cell's ranks by its own N means each cell contributes
        # to ``mean`` on the same [1/N, 1] scale regardless of how many
        # models competed in it.
        n_per_cell = ranks.notna().sum(axis=1).replace(0, pd.NA)
        ranks = ranks.div(n_per_cell, axis=0)
    out["mean"] = ranks.mean(axis=0).reindex(out.index)
    out["sem"] = ranks.sem(axis=0).reindex(out.index)
    return out


def compute_row_mean_ranks(
    df: pd.DataFrame,
    row_tasks: list[str],
    model_order: list[str],
    *,
    by_dataset: bool = False,
) -> pd.Series:
    """Backward-compatible thin wrapper returning only the per-model mean rank."""
    stats = compute_row_rank_stats(
        df, row_tasks=row_tasks, model_order=model_order, by_dataset=by_dataset
    )
    return stats["mean"]
