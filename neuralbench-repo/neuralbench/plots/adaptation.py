# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Adaptation-strategy comparison plot for the ``adaptation/`` output group.

:func:`plot_adaptation_comparison` answers "for each foundation model and task,
which adaptation strategy wins, and by how much?".  It is a no-op unless the
frame holds at least two distinct ``eval_mode`` values.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from neuralbench.plots._constants import (
    FEATURE_BASED_COLOR,
    MEEG_FM_DISPLAY,
    TASK_DISPLAY_NAMES,
    AdaptationMode,
)
from neuralbench.plots._style import save_figure

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Labels and colours for eval-mode tags (tag parsing lives in ``_constants``)
# ---------------------------------------------------------------------------

_STRATEGY_LABEL: dict[str, str] = {
    "linear_probe": "Linear Probe",
    "attentive_probe": "Attentive Probe",
    "finetune": "Full FT",
}


def eval_mode_label(tag: str) -> str:
    """Human-readable label for an ``eval_mode`` tag (``"lora_r8"`` -> ``"LoRA r8"``)."""
    mode = AdaptationMode.parse(tag)
    if mode.is_lora:
        label = f"LoRA r{mode.lora_rank}"
    elif mode.strategy in _STRATEGY_LABEL:
        label = _STRATEGY_LABEL[mode.strategy]
    else:
        return tag
    return f"{label} ({mode.aggregation})" if mode.aggregation else label


def order_modes(modes: list[str]) -> list[str]:
    """Order eval-mode tags LP -> LoRA(ascending rank) -> FT."""
    return sorted(set(modes), key=lambda m: AdaptationMode.parse(m).sort_key)


def mode_palette(modes: list[str]) -> dict[str, str]:
    """Sequential hex colour per mode following the ordered (param-budget) progression."""
    ordered = order_modes(modes)
    n = max(len(ordered), 1)
    cmap = plt.get_cmap("viridis")
    positions = np.linspace(0.12, 0.88, n)
    return {m: mcolors.to_hex(cmap(pos)) for m, pos in zip(ordered, positions)}


_BASELINE_COLORS: dict[str, str] = {
    "Chance": "#888888",
    "Handcrafted": FEATURE_BASED_COLOR,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_adaptation_comparison(
    df: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    """Write ``adaptation_per_task_bars.{png,pdf}`` to ``output_dir``.

    Parameters
    ----------
    df
        Output of :func:`neuralbench.plots.tables.build_results_df`.  Must
        contain ``eval_mode``, ``model_name``, ``task_name``,
        ``metric_name`` and ``metric_value`` columns.
    output_dir
        Destination directory (created if needed).

    Returns
    -------
    Path or None
        Path to the saved PNG, or ``None`` when there is nothing to plot (no
        foundation-model rows, or fewer than two of their ``eval_mode`` values).
    """
    if "eval_mode" not in df.columns:
        LOGGER.info("plot_adaptation_comparison: no 'eval_mode' column -- skipping.")
        return None

    # strip any " (LoRA r8)" suffix build_results_df added, to group by model
    work = df.copy()
    work["display_name"] = (
        work["model_name"].astype(str).str.split(" (", n=1, regex=False).str[0]
    )

    fm_names = set(MEEG_FM_DISPLAY)
    fm_work = work[work["display_name"].isin(fm_names)].copy()
    if fm_work.empty:
        LOGGER.info("plot_adaptation_comparison: no foundation-model rows -- skipping.")
        return None

    modes_present = order_modes(fm_work["eval_mode"].dropna().unique().tolist())
    if len(modes_present) < 2:
        LOGGER.info(
            "plot_adaptation_comparison: only %d FM eval_mode(s) present -- skipping.",
            len(modes_present),
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    palette = mode_palette(modes_present)
    out_path = _plot_per_task_bars(fm_work, work, modes_present, palette, output_dir)
    LOGGER.info("Wrote adaptation comparison to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Per-task grouped bars (mean +/- SEM, house style)
# ---------------------------------------------------------------------------


def _plot_per_task_bars(
    fm_df: pd.DataFrame,
    all_df: pd.DataFrame,
    modes_present: list[str],
    palette: dict[str, str],
    output_dir: Path,
) -> Path | None:
    """One subplot per task; per-FM clusters with one bar per strategy.

    Bars are mean +/- SEM across seeds; available baselines are dashed h-lines.
    """
    tasks = sorted(fm_df["task_name"].unique())
    if not tasks:
        return None

    n_tasks = len(tasks)
    n_cols = min(3, n_tasks)
    n_rows = int(np.ceil(n_tasks / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.0 * n_cols, 4.2 * n_rows),
        squeeze=False,
    )

    fm_order = [
        name for name in MEEG_FM_DISPLAY if name in fm_df["display_name"].unique()
    ]
    mode_colors = [palette[m] for m in modes_present]
    mode_labels = [eval_mode_label(m) for m in modes_present]

    for ax, task in zip(axes.flat, tasks):
        task_fm = fm_df[fm_df["task_name"] == task]
        sns.barplot(
            data=task_fm,
            x="display_name",
            y="metric_value",
            hue="eval_mode",
            order=fm_order,
            hue_order=modes_present,
            palette=mode_colors,
            errorbar="se",
            err_kws={"linewidth": 1.0},
            ax=ax,
        )
        if ax.get_legend() is not None:
            for txt, label in zip(ax.get_legend().get_texts(), mode_labels):
                txt.set_text(label)
            ax.get_legend().set_title("Strategy")
            ax.get_legend().set_visible(ax is axes.flat[0])

        task_metric = str(task_fm["metric_name"].iloc[0])
        ax.set_xlabel("")
        ax.set_ylabel(task_metric)
        ax.set_title(TASK_DISPLAY_NAMES.get(task, task))
        ax.tick_params(axis="x", rotation=30)

        task_all = all_df[all_df["task_name"] == task]
        for baseline, color in _BASELINE_COLORS.items():
            base_rows = task_all[task_all["display_name"] == baseline]
            if base_rows.empty:
                continue
            value = float(base_rows["metric_value"].mean())
            ax.axhline(
                value,
                color=color,
                linestyle="--",
                linewidth=1.0,
                label=f"{baseline} ({value:.1f})",
            )

    for ax in axes.flat[n_tasks:]:
        ax.set_visible(False)

    fig.suptitle("Per-task adaptation comparison (mean +/- SEM)", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_figure(fig, output_dir, "adaptation_per_task_bars")
