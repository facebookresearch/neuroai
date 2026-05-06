# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Table generation and data aggregation helpers for benchmark results."""

from __future__ import annotations

import typing as tp
from pathlib import Path

import pandas as pd

from neuralbench.plots._constants import MODEL_DISPLAY_NAMES
from neuralbench.plots._filters import multi_dataset_tasks
from neuralbench.plots.ranking import compute_task_ranks
from neuralbench.registry import FEATURE_BASED_BY_TASK, SKLEARN_BASELINE_MODELS

# Synthetic ``brain_model_name`` assigned to the task-appropriate sklearn row
# after collapsing; resolves to ``"Handcrafted"`` via ``MODEL_DISPLAY_NAMES``.
_FEATURE_BASED_LABEL = "feature_based"

# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


def _collapse_feature_based_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-pipeline sklearn baselines into a single "Handcrafted" row.

    For each task, keeps only the sklearn-baseline rows whose
    ``brain_model_name`` matches :data:`FEATURE_BASED_BY_TASK[task_name]` and
    relabels the kept rows to :data:`_FEATURE_BASED_LABEL` (so they resolve to
    ``"Handcrafted"`` via ``MODEL_DISPLAY_NAMES``).  Non-matching sklearn
    rows are dropped; deep-learning and constant-predictor rows pass through
    unchanged.

    This gives the headline bar / box charts a clean three-bar baseline story
    (Chance, Dummy, Handcrafted) while still leaving the per-pipeline
    results available in the cache for deeper diagnostic plots.
    """
    required = {"brain_model_name", "task_name"}
    if not required.issubset(df.columns):
        return df
    is_sklearn = df["brain_model_name"].isin(SKLEARN_BASELINE_MODELS)
    if not is_sklearn.any():
        return df
    preferred = df["task_name"].map(FEATURE_BASED_BY_TASK)
    # Keep rows that are either non-sklearn OR the task-preferred sklearn pipeline.
    keep = (~is_sklearn) | (df["brain_model_name"] == preferred)
    out = df.loc[keep].copy()
    out.loc[out["brain_model_name"].isin(SKLEARN_BASELINE_MODELS), "brain_model_name"] = (
        _FEATURE_BASED_LABEL
    )
    return out


def build_results_df(
    results: list[dict[str, tp.Any]],
    loss_to_metric_mapping: dict[str, str],
) -> pd.DataFrame:
    """Transform raw experiment dicts into a plot-ready DataFrame."""
    df = pd.DataFrame(results)
    df = _collapse_feature_based_baselines(df)
    df["model_name"] = df["brain_model_name"].map(
        lambda name: MODEL_DISPLAY_NAMES.get(name, name)
    )
    df["loss_name"] = df.loss.apply(pd.Series).name
    df["metric_name"] = df.loss_name.map(loss_to_metric_mapping)
    df["metric_value"] = df.apply(lambda x: x[x["metric_name"]], axis=1)
    _SCALE_TO_PERCENT = {"test/bal_acc", "test/full_retrieval/top5_acc_subject-agg"}
    df.loc[df.metric_name.isin(_SCALE_TO_PERCENT), "metric_value"] *= 100.0
    return df


# ---------------------------------------------------------------------------
# Results tables
# ---------------------------------------------------------------------------


def make_results_table(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    facet_column: str = "task_name",
    color_column: str = "model_name",
) -> pd.DataFrame:
    """Aggregate mean +/- std per task/model and save as CSV."""
    agg_df = df.groupby([facet_column, "metric_name", color_column]).metric_value.agg(
        ["mean", "std"]
    )
    agg_df["perf"] = agg_df.apply(
        lambda x: f"{x['mean']:0.3f} \u00b1 {x['std']:0.3f}", axis=1
    )
    wide_df = agg_df.reset_index().pivot(
        index=[facet_column, "metric_name"],
        columns=color_column,
        values="perf",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(output_dir / "core_results_table.csv")
    return wide_df


def make_rank_table(
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Build a per-task rank table with an average row and save as CSV."""
    rank_df = compute_task_ranks(df)
    rank_df.loc["average"] = rank_df.mean(axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_df.to_csv(output_dir / "core_rank_table.csv")
    return rank_df


def make_full_table(
    df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame | None:
    """Per-dataset breakdown table for the Full variant: mean +/- std (over seeds) for each combination."""
    if "dataset_name" not in df.columns:
        return None

    multi_tasks = multi_dataset_tasks(df)
    if not multi_tasks:
        return None

    sub = df[df.task_name.isin(multi_tasks)]
    agg = sub.groupby(["task_name", "dataset_name", "metric_name", "model_name"])[
        "metric_value"
    ].agg(["mean", "std"])
    agg["perf"] = agg.apply(lambda x: f"{x['mean']:0.3f} \u00b1 {x['std']:0.3f}", axis=1)
    wide = agg.reset_index().pivot(
        index=["task_name", "dataset_name", "metric_name"],
        columns="model_name",
        values="perf",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "full_table.csv"
    wide.to_csv(out_path)
    print(f"  Full-variant table saved to {out_path}")
    return wide


# ---------------------------------------------------------------------------
# Skip table
# ---------------------------------------------------------------------------


def print_skip_table(
    total: dict[tuple[str, str], int],
    skipped: dict[tuple[str, str], int],
) -> None:
    """Print a tasks x models table of included / total experiment counts."""
    tasks = sorted({t for t, _ in total})
    models = sorted({m for _, m in total})

    cells: dict[tuple[str, str], str] = {}
    model_totals: dict[str, tuple[int, int]] = {m: (0, 0) for m in models}
    task_totals: dict[str, tuple[int, int]] = {t: (0, 0) for t in tasks}
    for task in tasks:
        for model in models:
            key = (task, model)
            n_total = total.get(key, 0)
            n_included = n_total - skipped.get(key, 0)
            cells[key] = f"{n_included}/{n_total}" if n_total else "-"
            inc, tot = model_totals[model]
            model_totals[model] = (inc + n_included, tot + n_total)
            tinc, ttot = task_totals[task]
            task_totals[task] = (tinc + n_included, ttot + n_total)

    total_row: dict[str, str] = {}
    grand_inc, grand_tot = 0, 0
    for model in models:
        inc, tot = model_totals[model]
        total_row[model] = f"{inc}/{tot}"
        grand_inc += inc
        grand_tot += tot

    table_df = pd.DataFrame(cells, index=["done"]).T
    table_df.index = pd.MultiIndex.from_tuples(table_df.index, names=["task", "model"])
    table_df = table_df.reset_index().pivot(index="task", columns="model", values="done")

    table_df["ALL"] = [f"{task_totals[t][0]}/{task_totals[t][1]}" for t in table_df.index]
    total_row["ALL"] = f"{grand_inc}/{grand_tot}"
    total_series = pd.DataFrame(total_row, index=["TOTAL"])
    total_series.index.name = "task"
    table_df = pd.concat([table_df, total_series])
    table_df.index.name = None
    table_df.columns.name = None

    n_skip = sum(skipped.values())
    n_total = sum(total.values())
    print(
        f"\nExperiments with cached results ({n_total - n_skip}/{n_total} included):\n"
        f"{table_df.to_string()}"
    )
