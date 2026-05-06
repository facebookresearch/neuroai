# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Score normalisation and the per-task summary line plot.

The dummy-to-perfect normalisation in this module makes scores from
wildly different metrics (balanced accuracy %, Pearson R, top-5
retrieval %, ...) directly comparable across tasks.
"""

from __future__ import annotations

import typing as tp
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from neuralbench.plots._constants import (
    CLASSIC_DISPLAY,
    FM_DISPLAY,
    METRIC_HIGHER_IS_BETTER,
    METRIC_PERFECT_SCORE,
    TASK_DISPLAY_NAMES,
)
from neuralbench.plots._style import add_watermark

Ceiling = tp.Literal["perfect", "task_max"]

# ---------------------------------------------------------------------------
# Dummy-to-ceiling normalisation
# ---------------------------------------------------------------------------


def _compute_dummy_normalized_scores(
    df: pd.DataFrame | str | Path,
    *,
    ceiling: Ceiling = "perfect",
) -> pd.DataFrame:
    """Normalise scores as ``(x - dummy) / (ceiling - dummy)``.

    The floor is the Dummy model's score (falls back to the Chance model
    if Dummy is unavailable).  The ceiling is selected by *ceiling*:

    * ``"perfect"`` -- the metric's theoretical perfect score (e.g.
      100 % for balanced accuracy, 1.0 for Pearson R).  0 = dummy level,
      1 = perfect score.
    * ``"task_max"`` -- the best score observed for the task across
      modeled entries (``FM_DISPLAY`` + ``CLASSIC_DISPLAY``; for
      lower-is-better metrics the per-task min over the same set).
      Dummy/Chance and feature-based baselines are excluded so they
      cannot be both floor and ceiling.  0 = dummy level, 1 = best
      modeled score on that task.

    Returns a wide-form DataFrame (tasks x models).
    """
    if isinstance(df, (str, Path)):
        raw = pd.read_csv(df, index_col=[0, 1])
        metrics = raw.index.get_level_values("metric_name").to_series(
            index=raw.index.get_level_values("task_name")
        )
        metrics = metrics[~metrics.index.duplicated()]

        def _extract_mean(s: str) -> float:
            if pd.isna(s) or str(s).strip() == "":
                return np.nan
            return float(str(s).split("\u00b1")[0].strip())

        values = raw.droplevel("metric_name").apply(lambda col: col.map(_extract_mean))
    else:
        agg = df.groupby(["task_name", "model_name"]).metric_value.mean()
        values = agg.unstack("model_name")
        metrics = df.groupby("task_name")["metric_name"].first()

    values = values.dropna(how="all")

    modeled_cols = [m for m in (FM_DISPLAY + CLASSIC_DISPLAY) if m in values.columns]

    norm = pd.DataFrame(index=values.index, columns=values.columns, dtype=float)
    for task in values.index:
        metric = metrics.get(task, "test/bal_acc")
        higher = METRIC_HIGHER_IS_BETTER.get(metric, True)

        if ceiling == "task_max":
            if not modeled_cols:
                norm.loc[task] = np.nan
                continue
            row = values.loc[task, modeled_cols]
            ceiling_val = float(row.max()) if higher else float(row.min())
            if np.isnan(ceiling_val):
                norm.loc[task] = np.nan
                continue
        else:
            ceiling_val = METRIC_PERFECT_SCORE.get(metric, 100.0 if higher else 0.0)

        dummy_val = np.nan
        for baseline in ("Dummy", "Chance"):
            if baseline in values.columns:
                v = values.loc[task, baseline]
                if pd.notna(v):
                    dummy_val = float(v)
                    break

        if np.isnan(dummy_val):
            norm.loc[task] = np.nan
            continue

        if higher:
            denom = ceiling_val - dummy_val
        else:
            denom = dummy_val - ceiling_val

        if abs(denom) < 1e-12:
            norm.loc[task] = np.nan
            continue

        if higher:
            norm.loc[task] = (values.loc[task] - dummy_val) / denom
        else:
            norm.loc[task] = (dummy_val - values.loc[task]) / denom

    return norm.dropna(how="all")


# ---------------------------------------------------------------------------
# FM vs Classic summary lines plot
# ---------------------------------------------------------------------------

_COLOR_CLASSIC = "#2166ac"
_COLOR_FM = "#d95f02"


def _draw_summary_lines(
    ax: plt.Axes,  # type: ignore[name-defined]
    x: np.ndarray,
    best_fm: np.ndarray,
    avg_fm: np.ndarray,
    best_classic: np.ndarray,
    avg_classic: np.ndarray,
    *,
    add_legend_labels: bool = True,
) -> None:
    """Draw the four FM/classic summary lines, best-of-best gap fill, and
    per-task winner markers.

    Used by :func:`plot_normalized_lines_summary`.
    """
    # Best-of-best gap, tinted toward the winning family. Sits *below* the
    # within-family bands so the latter (alpha 0.12) remain legible on top.
    fm_ahead = best_fm >= best_classic
    ax.fill_between(
        x,
        best_classic,
        best_fm,
        where=fm_ahead.tolist(),
        color=_COLOR_FM,
        alpha=0.18,
        zorder=0.5,
        interpolate=True,
        linewidth=0,
    )
    ax.fill_between(
        x,
        best_fm,
        best_classic,
        where=(~fm_ahead).tolist(),
        color=_COLOR_CLASSIC,
        alpha=0.18,
        zorder=0.5,
        interpolate=True,
        linewidth=0,
    )

    ax.plot(
        x,
        best_fm,
        color=_COLOR_FM,
        linewidth=1.0,
        marker="o",
        markersize=2,
        label="Best foundation model" if add_legend_labels else None,
        zorder=5,
    )
    ax.plot(
        x,
        avg_fm,
        color=_COLOR_FM,
        linewidth=0.9,
        linestyle="--",
        marker="o",
        markersize=1.5,
        alpha=0.85,
        label="Mean over foundation models" if add_legend_labels else None,
        zorder=4,
    )
    ax.fill_between(x, avg_fm, best_fm, color=_COLOR_FM, alpha=0.12, zorder=1)

    ax.plot(
        x,
        best_classic,
        color=_COLOR_CLASSIC,
        linewidth=1.0,
        marker="o",
        markersize=2,
        label="Best task-specific model" if add_legend_labels else None,
        zorder=5,
    )
    ax.plot(
        x,
        avg_classic,
        color=_COLOR_CLASSIC,
        linewidth=0.9,
        linestyle="--",
        marker="o",
        markersize=1.5,
        alpha=0.85,
        label="Mean over task-specific models" if add_legend_labels else None,
        zorder=4,
    )
    ax.fill_between(
        x, avg_classic, best_classic, color=_COLOR_CLASSIC, alpha=0.12, zorder=1
    )

    # Per-task winner markers: small triangles just inside the bottom of
    # the axes, so they do not collide with rotated x-tick labels in the
    # gutter between rows.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i in range(len(x)):
        if np.isnan(best_fm[i]) or np.isnan(best_classic[i]):
            continue
        win_color = _COLOR_FM if best_fm[i] >= best_classic[i] else _COLOR_CLASSIC
        ax.scatter(
            [x[i]],
            [0.02],
            transform=trans,
            marker="^",
            s=10,
            color=win_color,
            edgecolor="none",
            clip_on=False,
            zorder=10,
        )

    ax.axhline(0, color="#999999", linewidth=0.5, linestyle=":", zorder=0)


def plot_normalized_lines_summary(
    df: pd.DataFrame | str | Path,
    output_dir: Path | None = None,
    *,
    group_by_category: bool = False,
    watermark: str | None = None,
    ylim: tuple[float, float] | None = None,
    ceiling: Ceiling = "perfect",
) -> plt.Figure:  # type: ignore[name-defined]
    """Summary line plot comparing foundation models vs classic models.

    Instead of one line per model, four aggregated lines are drawn:

    * **Best foundation model** -- per-task maximum score among FMs
    * **Mean over foundation models** -- per-task mean across FMs
    * **Best classic model** -- per-task maximum among classic models
    * **Mean over classic models** -- per-task mean across classic models

    Scores are normalised as ``(x - dummy) / (ceiling - dummy)``.  When
    *ceiling* is ``"perfect"`` (default) the ceiling is the metric's
    theoretical perfect score, so 1 corresponds to a perfect prediction.
    When *ceiling* is ``"task_max"`` the ceiling is the best score
    observed for the task across modeled entries (FMs + classics), so 1
    corresponds to the best task-specific model and the relative gap to
    the best is what gets visualised.

    When *group_by_category* is True the plot uses one subplot per task
    category (sharing the y-axis), visually separating each group.

    Parameters
    ----------
    df
        Long-form DataFrame with ``task_name``, ``model_name``,
        ``metric_value`` and ``metric_name`` columns, or a path to a
        cached results-table CSV.
    output_dir
        If given, save PNG + PDF here.
    group_by_category
        One subplot per category with shared y-axis.
    watermark
        Optional watermark text.
    ylim
        Y-axis limits as ``(ymin, ymax)``.  ``None`` for auto.
    ceiling
        Which ceiling to normalise against: ``"perfect"`` (theoretical
        max) or ``"task_max"`` (best modeled score on the task).
    """
    norm = _compute_dummy_normalized_scores(df, ceiling=ceiling)

    if norm.empty:
        warnings.warn(
            "Skipping normalized-lines-summary plot: no tasks with baseline data"
        )
        fig, ax = plt.subplots()
        plt.close(fig)
        return fig

    fm_cols = [m for m in FM_DISPLAY if m in norm.columns]
    classic_cols = [m for m in CLASSIC_DISPLAY if m in norm.columns]

    best_fm = norm[fm_cols].max(axis=1)
    avg_fm = norm[fm_cols].mean(axis=1)
    best_classic = norm[classic_cols].max(axis=1)
    avg_classic = norm[classic_cols].mean(axis=1)

    # Overall best per task (used for intra-category sorting)
    task_best = pd.concat([best_fm, best_classic], axis=1).max(axis=1)

    # -- build per-category task lists (sorted by performance) --------------
    if group_by_category:
        from neuralbench.plots._constants import (
            CATEGORY_COLORS,
            CATEGORY_ORDER,
            TASK_CATEGORIES,
        )

        available = set(norm.index)
        cat_task_lists: list[tuple[str, list[str]]] = []
        for cat in CATEGORY_ORDER:
            tasks_in_cat = [t for t in TASK_CATEGORIES.get(cat, []) if t in available]
            if tasks_in_cat:
                tasks_in_cat = sorted(
                    tasks_in_cat,
                    key=lambda t: task_best.get(t, 0.0),
                )
                cat_task_lists.append((cat, tasks_in_cat))
        leftover = [
            t for t in norm.index if t not in {t for _, ts in cat_task_lists for t in ts}
        ]
        if leftover:
            leftover = sorted(
                leftover,
                key=lambda t: task_best.get(t, 0.0),
            )
            cat_task_lists.append(("Other", leftover))
    else:
        all_tasks = sorted(
            norm.index.tolist(),
            key=lambda t: task_best.get(t, 0.0),
        )
        cat_task_lists = [("", all_tasks)]

    n_panels = len(cat_task_lists)
    if n_panels == 0:
        warnings.warn("Skipping normalized-lines-summary plot: no plottable categories")
        fig, ax = plt.subplots()
        plt.close(fig)
        return fig

    raw_widths = [len(tasks) for _, tasks in cat_task_lists]
    # Inflate panels with only a couple of tasks so that their rotated
    # x-tick labels have enough horizontal room and do not spill into the
    # neighbouring panel.
    _MIN_PANEL_W = 3.0
    widths = [max(w, _MIN_PANEL_W) for w in raw_widths]

    _CAT_TITLE: dict[str, str] = {
        "Evoked Responses": "Evoked\nResponses",
        "Internal State": "Internal\nState",
    }

    # -- figure ---------------------------------------------------------------
    # Target the full text width of a letter-format paper with ~0.75"
    # margins (≈7"), so that `\includegraphics[width=\linewidth]{...}`
    # fills the text area without scaling.
    _MAX_FIG_W = 7.0
    _YLABEL_GUTTER = 0.5
    _subplot_w = (_MAX_FIG_W - _YLABEL_GUTTER) / max(sum(widths), 1)
    fig_w = sum(widths) * _subplot_w + _YLABEL_GUTTER
    with sns.axes_style("whitegrid"):
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=(fig_w, 3.2),
            sharey=True,
            gridspec_kw={"width_ratios": widths, "wspace": 0.25},
        )
    if n_panels == 1:
        axes = [axes]

    for panel_idx, (cat, tasks) in enumerate(cat_task_lists):
        ax = axes[panel_idx]
        bfm = np.asarray(best_fm.loc[tasks].values, dtype=float)
        afm = np.asarray(avg_fm.loc[tasks].values, dtype=float)
        bcl = np.asarray(best_classic.loc[tasks].values, dtype=float)
        acl = np.asarray(avg_classic.loc[tasks].values, dtype=float)
        x = np.arange(len(tasks))
        labels_list = [
            TASK_DISPLAY_NAMES.get(t, t.replace("_", " ").title()) for t in tasks
        ]

        _draw_summary_lines(
            ax,
            x,
            bfm,
            afm,
            bcl,
            acl,
            add_legend_labels=(panel_idx == 0),
        )
        ax.axhline(1.0, color="#999999", linewidth=0.5, linestyle=":", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=6)
        ax.set_xlim(-0.5, len(tasks) - 0.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if panel_idx > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False)

        if group_by_category and cat:
            color = CATEGORY_COLORS.get(cat, "#555555")
            title = _CAT_TITLE.get(cat, cat)
            ax.set_title(
                title,
                fontsize=7,
                fontweight="bold",
                color=color,
                pad=4,
            )
            for label in ax.get_xticklabels():
                label.set_color("black")

        if panel_idx == 0:
            ylabel = (
                r"Normalized score $\tilde{s}^{\star}$"
                if ceiling == "task_max"
                else r"Normalized score $\tilde{s}$"
            )
            ax.set_ylabel(ylabel, fontsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)

    top_anchor_label = "Task best" if ceiling == "task_max" else "Perfect"
    anchor_tr = blended_transform_factory(axes[-1].transAxes, axes[-1].transData)
    axes[-1].text(
        0.98,
        1.0,
        top_anchor_label,
        transform=anchor_tr,
        fontsize=6,
        alpha=0.5,
        va="bottom",
        ha="right",
        zorder=3,
    )
    axes[-1].text(
        0.98,
        0.0,
        "Dummy",
        transform=anchor_tr,
        fontsize=6,
        alpha=0.5,
        va="bottom",
        ha="right",
        zorder=3,
    )

    # -- legend (two rows of 3, anchored below the axes) ---------------------
    fm_tri = Line2D(
        [0],
        [0],
        marker="^",
        color="none",
        markerfacecolor=_COLOR_FM,
        markeredgecolor=_COLOR_FM,
        markersize=5,
        linestyle="None",
    )
    cl_tri = Line2D(
        [0],
        [0],
        marker="^",
        color="none",
        markerfacecolor=_COLOR_CLASSIC,
        markeredgecolor=_COLOR_CLASSIC,
        markersize=5,
        linestyle="None",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [fm_tri, cl_tri]
    labels += ["Foundation model wins", "Task-specific model wins"]
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.34)
    fig.legend(
        handles,
        labels,
        fontsize=7,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.08),
        bbox_transform=fig.transFigure,
        handlelength=2.0,
        columnspacing=1.6,
        handletextpad=0.4,
    )

    if watermark:
        add_watermark(fig, watermark)

    stem = "core_normalized_summary"
    if ceiling == "task_max":
        stem = f"{stem}_task_max"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"{stem}.png",
            bbox_inches="tight",
            dpi=300,
        )
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    return fig
