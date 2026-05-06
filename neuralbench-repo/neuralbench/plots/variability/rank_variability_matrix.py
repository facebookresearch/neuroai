# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-(model, task) rank-variability matrix and heatmap.

This module resolves variability at the cell level: how unstable is a
*single model* on a *single task*, across that task's datasets?  This
makes it possible to spot the few ``(model, task)`` pairs that drive
most of the dataset-choice sensitivity.

The variability statistic used here is the per-paradigm *standard
deviation* of the model's per-dataset ranks (sample std, ``ddof=1``).
Lower values mean a more stable model on that task; ``0`` means the
model achieves the same rank on every dataset of the task.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from neuralbench.plots._constants import (
    CATEGORY_ORDER,
    CLASSIC_DISPLAY,
    DUMMY_DISPLAY,
    FM_DISPLAY,
    TASK_CATEGORIES,
    TASK_DISPLAY_NAMES,
    TASK_TO_CATEGORY,
)
from neuralbench.plots._filters import multi_dataset_tasks
from neuralbench.plots._models import lookup_by_name
from neuralbench.plots._style import group_color
from neuralbench.plots.ranking import compute_task_dataset_ranks

_FONT_OVERRIDES = {
    "font.family": ["sans-serif"],
    "font.sans-serif": ["DejaVu Sans"]
    + [f for f in plt.rcParams["font.sans-serif"] if f != "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def compute_model_task_rank_variability(
    df: pd.DataFrame,
    *,
    min_datasets: int = 1,
) -> pd.DataFrame:
    """Per-(model, task) rank standard deviation across the task's datasets.

    For each NeuralBench-Full multi-dataset paradigm, every model gets a
    sequence of per-dataset ranks (via :func:`compute_task_dataset_ranks`).
    This helper collapses that sequence to the *standard deviation*
    (sample std, ``ddof=1``) per ``(model, task)`` cell -- a smooth
    summary of how much the model's rank moves around across the
    paradigm's datasets.  ``0`` means the model holds the same rank on
    every dataset of the task; larger values flag (model, task) pairs
    whose ranking is most sensitive to dataset choice.

    Single-dataset paradigms are always skipped (their variability is
    undefined / zero by construction and would dilute the matrix).

    Parameters
    ----------
    df
        Long-form benchmark DataFrame with at least ``task_name``,
        ``dataset_name``, ``model_name``, ``metric_value`` and
        ``metric_name`` columns.
    min_datasets
        Minimum number of datasets a task must have to be included.
        Defaults to 1 (i.e. only single-dataset tasks are excluded).
        Use ``min_datasets=5`` to restrict to tasks with at least 5
        datasets, which gives more reliable variability estimates.

    Returns
    -------
    matrix
        Wide DataFrame indexed by ``model_name`` (rows) and
        ``task_name`` (columns).  Cells are NaN when a model never
        produced a finite rank on the corresponding paradigm (e.g. a
        foundation model that wasn't evaluated there).  Returns an
        empty DataFrame when *df* contains no qualifying paradigm.
    """
    if "dataset_name" not in df.columns:
        return pd.DataFrame()

    threshold = max(2, min_datasets)
    multi_tasks = multi_dataset_tasks(df, threshold=threshold)
    if not multi_tasks:
        return pd.DataFrame()

    sub = df[df["task_name"].isin(multi_tasks)]
    ranks = compute_task_dataset_ranks(sub)  # (task, dataset) x model

    # Sample std (ddof=1, pandas default) of each model's per-dataset
    # ranks within each task.  Tasks with a single dataset have already
    # been filtered out above.
    per_task = ranks.groupby(level="task_name").std()

    # Result so far: rows = task, cols = model.  Transpose to the
    # caller-friendly shape (rows = model, cols = task).
    matrix = per_task.T

    # Drop columns that are entirely NaN (paradigm with no usable
    # ranks) -- defensive; should not normally happen given the
    # multi-dataset gating above.
    matrix = matrix.dropna(axis=1, how="all")
    return matrix


def _order_model_task_matrix(
    matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Order matrix rows (models) and columns (tasks) for the heatmap.

    **Columns** follow the canonical category order from the bar chart
    (``CATEGORY_ORDER`` / ``TASK_CATEGORIES``), so the column layout is
    easy to cross-reference with other figures.  Within each category,
    tasks are sorted by ascending mean variability across models.

    **Rows** are split into two blocks separated by a visual rule at
    render time:

    1. *Real models* (task-specific + foundation), sorted by ascending mean
       variability across tasks -- the most stable model sits on top.
    2. *Baselines* (Dummy, Chance, Handcrafted), pinned at the bottom
       in ``DUMMY_DISPLAY`` order.

    This keeps the reader's eye on the interesting models first and
    avoids the trivially-stable constant-predictor baselines from
    dominating the top rows.
    """
    if matrix.empty:
        return matrix

    # ---- column order: category, then ascending mean within category
    task_set = set(matrix.columns)
    col_order: list[str] = []
    for cat in CATEGORY_ORDER:
        tasks_in_cat = [t for t in TASK_CATEGORIES.get(cat, []) if t in task_set]
        cat_means = matrix[tasks_in_cat].mean(axis=0, skipna=True)
        col_order.extend(cat_means.sort_values(kind="stable").index.tolist())
    for t in matrix.columns:
        if t not in col_order:
            col_order.append(t)

    # ---- row order: task-specific (by mean) | FM (by mean) | baselines (fixed)
    classic_set = set(CLASSIC_DISPLAY)
    fm_set = set(FM_DISPLAY)
    baseline_set = set(DUMMY_DISPLAY)
    row_means = matrix.mean(axis=1, skipna=True)

    classics = [m for m in row_means.sort_values(kind="stable").index if m in classic_set]
    fms = [m for m in row_means.sort_values(kind="stable").index if m in fm_set]
    baselines = [m for m in DUMMY_DISPLAY if m in matrix.index]
    others = [
        m
        for m in row_means.sort_values(kind="stable").index
        if m not in classic_set and m not in fm_set and m not in baseline_set
    ]
    row_order = classics + fms + others + baselines

    return matrix.loc[row_order, col_order]


def _append_marginal_means(matrix: pd.DataFrame) -> pd.DataFrame:
    """Append a ``Mean`` column (axis=1) and ``Mean`` row (axis=0).

    The bottom-right corner is the grand mean over the *data* cells
    (i.e. the mean of the row means, equal to the mean of the column
    means in absence of NaNs).  NaN cells are skipped throughout.
    """
    if matrix.empty:
        return matrix
    out = matrix.copy()
    out["Mean"] = matrix.mean(axis=1, skipna=True)
    col_means = matrix.mean(axis=0, skipna=True)
    grand = (
        float(matrix.values[~np.isnan(matrix.values)].mean())
        if matrix.size
        else float("nan")
    )
    out.loc["Mean"] = list(col_means.values) + [grand]
    return out


def _model_group(name: str) -> str:
    """Return ``"classic"``, ``"fm"`` or ``"baseline"`` for *name*."""
    entry = lookup_by_name(name)
    if entry is None:
        return "baseline"
    if entry.family == "foundation":
        return "fm"
    return entry.family  # "classic"


_GROUP_BRACKET_COLORS: dict[str, tuple[str, str]] = {
    "classic": ("#3575b5", "Task-specific"),
    "fm": ("#e8833a", "Foundation"),
    "baseline": ("#888888", "Baseline"),
}
"""(colour, display label) for each model group bracket."""


def _heatmap_figsize(
    df: pd.DataFrame,
    *,
    min_datasets: int = 5,
    paper_width: float = 6.5,
    font_scale: float = 1.0,
) -> tuple[float, float] | None:
    """Return the standalone ``(width, height)`` used by the heatmap, or ``None``.

    *font_scale* must match the value passed to
    :func:`_draw_model_task_rank_variability_heatmap` so the figure is
    tall enough to accommodate the (possibly enlarged) tick labels,
    bracket strip and colour bar.
    """
    matrix = compute_model_task_rank_variability(df, min_datasets=min_datasets)
    if matrix.empty:
        return None
    n_tasks = matrix.shape[1]
    fig_w = max(paper_width, 7.5)
    cell_h = 0.22
    # Buffers above the heatmap (rotated x-tick labels), the bracket
    # strip, and the colour bar all scale with the font size.
    top_buffer = 1.2 * font_scale
    bracket_h = 0.42 * font_scale
    cbar_h = 0.55 * font_scale
    fig_h = cell_h * (n_tasks + 1) + top_buffer + bracket_h + cbar_h
    return (fig_w, fig_h)


def _draw_model_task_rank_variability_heatmap(
    parent: Figure | SubFigure,
    df: pd.DataFrame,
    *,
    min_datasets: int = 5,
    font_scale: float = 1.0,
) -> float | None:
    """Draw the per-(model, task) rank-variability heatmap onto *parent*.

    *parent* may be a top-level :class:`~matplotlib.figure.Figure` or a
    nested :class:`~matplotlib.figure.SubFigure`.  *font_scale* multiplies
    every text element (cell annotations, tick labels, bracket labels,
    colour-bar label/ticks) and the colour-bar strip height; pass
    ``font_scale > 1`` when rendering into a larger combined figure
    where the standalone sizes are too small to read.

    Returns the y position (in ``parent.transSubfigure`` coords) of the
    bracket-label band, so a calling combiner can align an external
    panel label with it.  Returns ``None`` when *df* contains no
    qualifying paradigm.
    """
    matrix = compute_model_task_rank_variability(df, min_datasets=min_datasets)
    if matrix.empty:
        return None

    ordered = _order_model_task_matrix(matrix)
    full = _append_marginal_means(ordered)

    # Transpose: rows = tasks, columns = models.
    full_t = full.T

    # ---- block geometry ----
    model_names: list[str] = list(ordered.index)
    task_keys: list[str] = list(ordered.columns)
    n_models = len(model_names)
    n_tasks = len(task_keys)

    # Group boundaries for columns (task-specific | fm | baseline).
    group_spans: list[tuple[int, int, str]] = []
    prev_grp = _model_group(model_names[0]) if model_names else ""
    start = 0
    for idx, m in enumerate(model_names):
        grp = _model_group(m)
        if grp != prev_grp:
            group_spans.append((start, idx, prev_grp))
            start = idx
            prev_grp = grp
    if model_names:
        group_spans.append((start, n_models, prev_grp))

    # Task category boundaries for row separators.
    task_cat_boundaries: list[int] = []
    prev_cat = TASK_TO_CATEGORY.get(task_keys[0], "") if task_keys else ""
    for idx, t in enumerate(task_keys):
        cat = TASK_TO_CATEGORY.get(t, "")
        if cat != prev_cat and idx > 0:
            task_cat_boundaries.append(idx)
        prev_cat = cat

    # ---- per-task dataset counts ----
    ds_counts = (
        df[df["task_name"].isin(ordered.columns)]
        .groupby("task_name")["dataset_name"]
        .nunique()
        .to_dict()
    )
    task_display: list[str] = []
    for t in task_keys:
        nice = TASK_DISPLAY_NAMES.get(t, t.replace("_", " ").title())
        n = ds_counts.get(t)
        task_display.append(f"{nice} (N={int(n)})" if n else nice)
    task_display.append("Mean")

    model_display = model_names + ["Mean"]

    # ---- colour scale: single-hue black -> red, linearly normalised ----
    # Lower is better: the most stable cell in the data maps to pure
    # black, the saturated red end corresponds to the largest std
    # observed in the data, so the limited dynamic range of std values
    # still uses the full ramp.  A pure-black anchor at the bottom
    # makes it obvious which cell(s) define the minimum.
    data_vals = ordered.values
    finite = data_vals[np.isfinite(data_vals)]
    if finite.size:
        vmin = float(finite.min())
        vmax = float(np.ceil(float(finite.max())))
    else:
        vmin = 0.0
        vmax = 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "black_to_red", ["#000000", "#d62728"]
    )

    # ---- annotation format ----
    # Std values are floats with a small dynamic range (~0-9); one
    # decimal is enough for both data and marginal-mean cells.
    data_annot_fmt = "{:.1f}"
    mean_annot_fmt = "{:.1f}"

    # ---- vertical anatomy of the panel ----
    # Height of the colour-bar strip in inches; we reserve a slice of
    # the parent's (or sub-figure's) bottom for it.  Scales with
    # ``font_scale`` so the strip stays tall enough to fit a larger
    # label / tick text.
    cbar_h = 0.55 * font_scale

    annot_fs = 5.5 * font_scale
    tick_fs = 7.0 * font_scale
    bracket_fs = 6.5 * font_scale
    cbar_label_fs = 7.0 * font_scale
    cbar_tick_fs = 6.0 * font_scale

    cbar_label = "Rank std (\u2193)"

    # In a SubFigure, our slice of vertical space is
    # ``parent_fig_h * subfig_h_frac``; ``bbox.height`` ratios give the
    # fractional height directly.
    if isinstance(parent, SubFigure):
        parent_fig = parent.figure
        parent_h = float(parent_fig.get_size_inches()[1])
        slice_frac = float(parent.bbox.height / parent_fig.bbox.height)
        slice_h_in = parent_h * slice_frac
    else:
        slice_h_in = float(parent.get_size_inches()[1])
    bottom_reserve = (cbar_h + 0.05) / slice_h_in

    with plt.rc_context(_FONT_OVERRIDES), sns.axes_style("white"):
        ax = parent.subplots()

        sns.heatmap(
            full_t,
            ax=ax,
            cmap=cmap,
            norm=norm,
            annot=False,
            cbar=False,
            linewidths=0.3,
            linecolor="white",
            xticklabels=model_display,
            yticklabels=task_display,
            square=False,
        )

        # ---- subtle zebra striping on alternating data rows ----
        for row_idx in range(n_tasks):
            if row_idx % 2 == 1:
                ax.add_patch(
                    FancyBboxPatch(
                        (0, row_idx),
                        n_models + 1,
                        1,
                        boxstyle="square,pad=0",
                        facecolor="#000000",
                        edgecolor="none",
                        alpha=0.04,
                        zorder=0,
                    )
                )

        # ---- light grey tint for marginal Mean row/column ----
        mean_bg = "#d8d8d8"
        # Mean row (last row)
        ax.add_patch(
            FancyBboxPatch(
                (0, n_tasks),
                n_models + 1,
                1,
                boxstyle="square,pad=0",
                facecolor=mean_bg,
                edgecolor="none",
                alpha=0.55,
                zorder=0,
            )
        )
        # Mean column (last column)
        ax.add_patch(
            FancyBboxPatch(
                (n_models, 0),
                1,
                n_tasks + 1,
                boxstyle="square,pad=0",
                facecolor=mean_bg,
                edgecolor="none",
                alpha=0.55,
                zorder=0,
            )
        )

        # ---- cell annotations ----
        annot_src = full_t
        n_total_rows = full_t.shape[0]
        n_total_cols = full_t.shape[1]
        for i in range(n_total_rows):
            for j in range(n_total_cols):
                color_val = float(full_t.iat[i, j])  # type: ignore[arg-type]
                annot_val = annot_src.iat[i, j]
                if not np.isfinite(color_val):
                    continue
                is_marginal = (i == n_total_rows - 1) or (j == n_total_cols - 1)
                normed = norm(color_val)
                normed = max(0.0, min(1.0, float(normed)))
                rgba = cmap(normed)
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if lum < 0.50 else "#222222"
                fmt = mean_annot_fmt if is_marginal else data_annot_fmt
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    fmt.format(annot_val),
                    ha="center",
                    va="center",
                    fontsize=annot_fs,
                    color=text_color,
                    fontweight="bold" if is_marginal else "normal",
                )

        # ---- separator lines ----
        # Marginal-mean separators (thick solid).
        ax.axhline(n_tasks, color="#222222", linewidth=1.0)
        ax.axvline(n_models, color="#222222", linewidth=1.0)
        # Task-category separators (thin dotted horizontal).
        for boundary in task_cat_boundaries:
            ax.axhline(boundary, color="#bbbbbb", linewidth=0.5, linestyle=":")

        # ---- model tick labels on x-axis (top) ----
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", which="major", pad=2)
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="left",
            rotation_mode="anchor",
            fontsize=tick_fs,
        )
        for label, mname in zip(ax.get_xticklabels(), model_names + ["Mean"]):
            if mname == "Mean":
                label.set_color("#222222")
                label.set_fontweight("bold")
            else:
                label.set_color(group_color(mname))

        # ---- group bracket labels above the x-tick labels ----
        # Finalize the layout *once* so axes positions are stable, then
        # freeze it so savefig won't re-run tight_layout and shift the
        # axes out from under the brackets we're about to place.  The
        # ``rect`` reserves a strip at the bottom for the horizontal
        # colour-bar we add below -- without this, tight_layout grows
        # the heatmap into that strip and the bar overlaps the last
        # row of cells.
        if isinstance(parent, SubFigure):
            # SubFigure doesn't support tight_layout; we approximate it
            # with explicit margins.  Values were tuned visually so the
            # x-tick labels and bracket band fit at the top, and the
            # colour-bar fits in the bottom strip we reserve for it.
            parent.subplots_adjust(
                left=0.10,
                right=0.98,
                top=0.80,
                bottom=bottom_reserve + 0.04,
            )
        else:
            parent.tight_layout(rect=(0.0, bottom_reserve, 1.0, 1.0))
            parent.set_layout_engine("none")
        parent.canvas.draw()
        renderer = parent.canvas.get_renderer()  # type: ignore[attr-defined]
        top_y_fig = 0.0
        for lbl in ax.get_xticklabels():
            bb = lbl.get_window_extent(renderer)
            bb_fig = bb.transformed(parent.transSubfigure.inverted())
            top_y_fig = max(top_y_fig, bb_fig.y1)

        bracket_y0 = top_y_fig + 0.015  # line just above tallest label
        bracket_y1 = bracket_y0 + 0.025  # label sits above the line

        for s, e, grp in group_spans:
            color_hex, disp_label = _GROUP_BRACKET_COLORS.get(
                grp, ("#888888", grp.title())
            )
            # Map column indices to display-x via the data transform.
            x_left = ax.transData.transform((s + 0.05, 0))[0]
            x_right = ax.transData.transform((e - 0.05, 0))[0]
            x_mid = ax.transData.transform(((s + e) / 2.0, 0))[0]
            # Convert to subfigure-relative coords.
            inv = parent.transSubfigure.inverted()
            fx_left = inv.transform((x_left, 0))[0]
            fx_right = inv.transform((x_right, 0))[0]
            fx_mid = inv.transform((x_mid, 0))[0]

            # Horizontal bracket line
            line = Line2D(
                [fx_left, fx_right],
                [bracket_y0, bracket_y0],
                transform=parent.transSubfigure,
                color=color_hex,
                linewidth=1.5,
                clip_on=False,
            )
            parent.add_artist(line)
            # End-ticks
            for fx in (fx_left, fx_right):
                tick_line = Line2D(
                    [fx, fx],
                    [bracket_y0 - 0.012, bracket_y0],
                    transform=parent.transSubfigure,
                    color=color_hex,
                    linewidth=1.5,
                    clip_on=False,
                )
                parent.add_artist(tick_line)
            # Label
            parent.text(
                fx_mid,
                bracket_y1,
                disp_label,
                ha="center",
                va="bottom",
                fontsize=bracket_fs,
                fontweight="bold",
                color=color_hex,
            )

        # ---- task tick labels on y-axis (uniform black) ----
        plt.setp(ax.get_yticklabels(), fontsize=tick_fs, rotation=0, color="#222222")
        for label, task in zip(ax.get_yticklabels(), task_keys + ["Mean"]):
            if task == "Mean":
                label.set_fontweight("bold")

        # ---- colorbar: horizontal, centred below the heatmap ----
        # Place the bar inside the bottom strip we reserved via the
        # layout (tight_layout(rect=...) for a Figure, subplots_adjust
        # for a SubFigure).  It is centred under the data columns
        # (excluding the trailing Mean column) and uses 45% of that
        # span so it reads as a scale rather than a full-width axis.
        # The label sits below the bar inside the same strip.
        data_left = ax.transData.transform((0.0, 0.0))[0]
        data_right = ax.transData.transform((float(n_models), 0.0))[0]
        inv = parent.transSubfigure.inverted()
        fx_data_left = inv.transform((data_left, 0.0))[0]
        fx_data_right = inv.transform((data_right, 0.0))[0]
        cbar_width_fig = 0.45 * (fx_data_right - fx_data_left)
        cbar_x0 = (fx_data_left + fx_data_right) / 2.0 - cbar_width_fig / 2.0
        cbar_height_fig = 0.022
        # Position the bar in the upper half of the reserved strip so
        # there's room for its label underneath.
        cbar_y0 = bottom_reserve * 0.55
        cbar_ax = parent.add_axes((cbar_x0, cbar_y0, cbar_width_fig, cbar_height_fig))
        cbar = parent.colorbar(ax.collections[0], cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=cbar_tick_fs)
        cbar.set_label(cbar_label, fontsize=cbar_label_fs, labelpad=4)
        # Three ticks at the data minimum, midpoint, and rounded max.
        # Format as integers when both ends are integral, otherwise
        # keep one decimal so the min-end tick reflects the actual
        # data minimum (which need not be an integer).
        ticks = [vmin, (vmin + vmax) / 2.0, vmax]
        cbar.set_ticks(ticks)
        all_int = all(abs(t - round(t)) < 1e-9 for t in ticks)
        if all_int:
            cbar.set_ticklabels([f"{int(round(t))}" for t in ticks])
        else:
            cbar.set_ticklabels([f"{t:.1f}" for t in ticks])

        ax.set_xlabel("")
        ax.set_ylabel("")

    return float(bracket_y1)
