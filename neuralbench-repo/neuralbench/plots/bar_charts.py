# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Bar-chart plots for NeuralBench-Core and NeuralBench-Full.

Two public entry points:

* :func:`plot_bar_chart` -- one subplot per task on the core benchmark,
  arranged in a category-aware grid.
* :func:`plot_full_bar_chart` -- one row per multi-dataset task on the
  full benchmark, with one column per dataset.  Driven by a
  :class:`_FullGridStyle` dataclass; the default ``LETTER_STYLE``
  reproduces the dense letter-format chart that ships in the paper.
"""

from __future__ import annotations

import functools
import re
import typing as tp
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from neuralbench.plots._constants import (
    CATEGORY_COLORS,
    CATEGORY_ROW_GROUPS,
    TASK_CATEGORIES,
    TASK_DISPLAY_NAMES,
    TASK_TO_CATEGORY,
)
from neuralbench.plots._filters import multi_dataset_tasks
from neuralbench.plots._style import (
    MarginalRankSpec,
    SubplotSpec,
    add_watermark,
    bold_legend_titles,
    build_color_dict,
    build_grouped_legend,
    build_grouped_legend_two_col,
    draw_left_margin_label,
    finalize_bar_grid,
    order_models_with_baselines_first,
    render_bars_grid,
)
from neuralbench.plots._task_metadata import (
    build_task_n_examples_map,
    build_task_split_map,
    default_study_names,
)
from neuralbench.plots.ranking import compute_row_rank_stats

# ---------------------------------------------------------------------------
# Font overrides
#
# Some transitive imports (notably ``moabb``) change ``font.family`` to
# ``"serif"`` at import time, which then leaks into every subsequent
# ``ax.text`` / ``ax.set_title`` call that doesn't explicitly pin a font.
# We re-pin matplotlib's defaults locally here so the bar charts always
# render in DejaVu Sans -- even when the plotting code is invoked from a
# Python process that has already imported ``moabb``.  ``pdf.fonttype = 42``
# embeds real TrueType fonts in the PDF instead of Type-3 outlines, which
# avoids font substitution in viewers and downstream LaTeX inclusions.
# ---------------------------------------------------------------------------
_FONT_OVERRIDES = {
    "font.family": ["sans-serif"],
    "font.sans-serif": ["DejaVu Sans"]
    + [f for f in plt.rcParams["font.sans-serif"] if f != "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


_F = tp.TypeVar("_F", bound=tp.Callable[..., tp.Any])


def _with_font_overrides(func: _F) -> _F:
    """Run *func* under a ``plt.rc_context`` pinning matplotlib's defaults.

    Applied to ``plot_bar_chart`` and ``plot_full_bar_chart`` so that font
    rcParams modified by transitive imports (notably ``moabb`` setting
    ``font.family = "serif"``) cannot leak into the rendered chart.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(_FONT_OVERRIDES):
            return func(*args, **kwargs)

    return tp.cast(_F, wrapper)


# ---------------------------------------------------------------------------
# Core bar chart
# ---------------------------------------------------------------------------


@_with_font_overrides
def plot_bar_chart(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    facet_column: str = "task_name",
    color_column: str = "model_name",
    y_column: str = "metric_value",
    watermark: str | None = None,
) -> Path:
    """Faceted bar chart comparing models across tasks.

    Tasks are arranged in a category-aware grid: each row corresponds to
    one or more task categories (as defined by ``CATEGORY_ROW_GROUPS``),
    with at most 7 columns per row.
    """
    task_to_metric = df.groupby(facet_column)["metric_name"].first().to_dict()
    available = set(df[facet_column].unique())

    display_models = df[color_column].unique().tolist()
    color_dict = build_color_dict(display_models)
    model_order = order_models_with_baselines_first(display_models)

    split_map = build_task_split_map()
    n_examples_map = build_task_n_examples_map()

    # ``_max_tasks`` is the maximum number of *task* subplots per row
    # (Evoked Responses has the most: 7).  ``_max_cols`` adds 1 leftmost
    # column for the per-row marginal mean-rank bar plot.  The marginal
    # column is slightly wider than a task column to host the
    # multi-word "Mean norm.\nrank \u2191" title plus the
    # "Best (0)" / "Worst (1)" tick labels without crowding.
    _max_tasks = 7
    _max_cols = _max_tasks + 1
    _marginal_width_ratio = 0.95
    row_layouts: list[list[str]] = []
    row_cat_spans: list[list[tuple[str, int, int]]] = []
    for row_group in CATEGORY_ROW_GROUPS:
        tasks_in_row: list[str] = []
        spans: list[tuple[str, int, int]] = []
        for cat in row_group:
            cat_tasks = [t for t in TASK_CATEGORIES.get(cat, []) if t in available]
            if cat_tasks:
                col_start = len(tasks_in_row)
                tasks_in_row.extend(cat_tasks)
                spans.append((cat, col_start, col_start + len(cat_tasks) - 1))
        if tasks_in_row:
            row_layouts.append(tasks_in_row)
            row_cat_spans.append(spans)

    # Standalone landscape sizing: 3" x 3" per cell -- the same
    # generous proportions as the original ``core_bar_chart`` from
    # commit 5a8aa7d8a -- so the per-task panels read at the same
    # visual density as the standalone reference (e.g.
    # ``core_non_eeg_bar_chart.png``).  ``hspace=0.55`` was tuned in
    # that PR to keep the per-row info badge from colliding with the
    # next row's title.
    n_rows = len(row_layouts)
    _subplot_w, _subplot_h = 3.0, 3.0
    figsize = (
        (_marginal_width_ratio + _max_tasks) * _subplot_w,
        n_rows * _subplot_h,
    )

    with sns.axes_style("white"):
        fig, axes = plt.subplots(
            n_rows,
            _max_cols,
            figsize=figsize,
            squeeze=False,
            gridspec_kw={
                "wspace": 0.3,
                "hspace": 0.55,
                "width_ratios": [_marginal_width_ratio] + [1.0] * _max_tasks,
            },
        )

    grid: list[list[SubplotSpec | MarginalRankSpec | None]] = []
    for row_idx, row_tasks in enumerate(row_layouts):
        # Marginal mean-rank panel for this row's tasks.
        rank_stats = compute_row_rank_stats(
            df,
            row_tasks=row_tasks,
            model_order=model_order,
            by_dataset=False,
        )
        row_specs: list[SubplotSpec | MarginalRankSpec | None] = [
            MarginalRankSpec(rank_stats=rank_stats)
        ]
        for task_name in row_tasks:
            task_df = df[df[facet_column] == task_name]
            agg = (
                task_df.groupby(color_column)[y_column]
                .agg(["mean", "sem"])
                .reindex(model_order)
            )
            nice_task = TASK_DISPLAY_NAMES.get(
                task_name, task_name.replace("_", " ").title()
            )
            # ``task_df`` has been reduced to the core dataset per task
            # upstream via ``filter_core_dataset`` so pulling the first
            # ``dataset_name`` is sufficient for the pretraining-overlap check.
            dataset_name = (
                str(task_df["dataset_name"].iloc[0])
                if "dataset_name" in task_df.columns and not task_df.empty
                else ""
            )
            row_specs.append(
                SubplotSpec(
                    title=nice_task.replace("\n", " "),
                    agg=agg,
                    dataset_name=dataset_name,
                    metric_key=task_to_metric.get(task_name, ""),
                    split_kind=split_map.get((task_name, dataset_name)),
                    n_examples=n_examples_map.get((task_name, dataset_name)),
                )
            )
        row_specs.extend([None] * (_max_cols - len(row_specs)))
        grid.append(row_specs)

    # Use the standalone-chart defaults from the original PR (visible
    # y-tick labels, single-line metric subtitle below each title,
    # and single-line "split-kind  \u00b7  n=..." badge under the
    # x-axis) -- the 3" x 3" cells have plenty of room.  Slightly
    # bigger error-bar caps / whiskers stay on so the SEM markers
    # remain legible across all density levels.
    render_bars_grid(
        axes,
        grid,
        model_order,
        color_dict,
        error_capsize=2.5,
        error_linewidth=1.0,
    )

    # Category labels -- placed after tight_layout so figure coords are
    # final.  Anchored on the *marginal* column (column 0) so labels
    # sit in the figure's left margin, before the very first subplot of
    # the row.  Multi-category rows are rendered as a single combined
    # ``"Cat1 / Cat2"`` label at the same fontsize as single-category
    # rows so the left margin reads uniformly across rows.
    def _draw_category_labels(
        fig: plt.Figure,  # type: ignore[name-defined]
        axes: tp.Any,
    ) -> None:
        _base_x_offset = 0.025
        for row_idx, spans in enumerate(row_cat_spans):
            if not spans:
                continue
            pos0 = axes[row_idx, 0].get_position()
            y_center = (pos0.y0 + pos0.y1) / 2
            label = " / ".join(cat for cat, _, _ in spans)
            fig.text(
                pos0.x0 - _base_x_offset,
                y_center,
                label,
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="black",
                rotation=90,
            )

    # Faint background tile behind the marginal column.  Spans the full
    # height of the axes stack and extends slightly left / right of the
    # first subplot so the "Best\n(0)" / "Worst\n(1)" tick labels and
    # category labels sit on top of the tile too.  The patch is added
    # via ``fig.add_artist`` at low zorder so axes content (bars, error
    # bars, titles) renders on top.  Each marginal axis' face colour is
    # set to "none" so its default white background no longer punches a
    # hole through the tile -- the strip therefore reads as one
    # continuous band.  Colour is intentionally near-white so the tile
    # whispers rather than shouts.
    _bg_color = "#ededed"

    def _draw_marginal_background(
        fig: plt.Figure,  # type: ignore[name-defined]
        axes: tp.Any,
    ) -> None:
        for row_idx in range(axes.shape[0]):
            axes[row_idx, 0].set_facecolor("none")
        first_pos = axes[0, 0].get_position()
        last_pos = axes[-1, 0].get_position()
        pad_left = 0.02
        pad_right = 0.005
        pad_y_bot = 0.015
        # Top padding extends just past row 0's marginal title.  The
        # title is rendered via ``ax.annotate`` with ``va="top"``, so
        # its top edge sits ~30 pt above the axes top edge
        # (``title_pad + title_fontsize = 16 + 14``, see
        # render_marginal_rank_bars).  Convert to a figure fraction
        # plus a small buffer so the rectangle wraps the title cleanly
        # regardless of figure height -- the bold "Mean norm.\nrank
        # \u2191" extends *downward* from that anchor, so we don't need
        # a second fontsize of headroom.
        fig_h_in = fig.get_figheight()
        pad_y_top = (16 + 14 + 4) / 72.0 / fig_h_in
        rect_x = first_pos.x0 - pad_left
        rect_w = (first_pos.x1 - first_pos.x0) + pad_left + pad_right
        rect_y = last_pos.y0 - pad_y_bot
        rect_h = (first_pos.y1 - last_pos.y0) + pad_y_bot + pad_y_top
        rect = mpatches.Rectangle(
            (rect_x, rect_y),
            rect_w,
            rect_h,
            facecolor=_bg_color,
            edgecolor="none",
            zorder=-10,
            transform=fig.transFigure,
        )
        fig.add_artist(rect)

    def _pre_legend_hook(
        fig: plt.Figure,  # type: ignore[name-defined]
        axes: tp.Any,
    ) -> None:
        _draw_marginal_background(fig, axes)
        _draw_category_labels(fig, axes)

    # Park the legend in the empty top-right cells of the grid (last 3
    # columns of the last 3 rows: Internal State / Phenotyping / Sleep
    # + Misc rows leave columns 5-7 unused once their 3-4 task panels
    # are placed).  This recovers ~9 sq in of whitespace that would
    # otherwise sit blank, and lets ``bbox_inches="tight"`` crop the
    # output snugly to the chart instead of expanding it to fit a
    # legend hung beneath the bottom row.  The 2-col layout (Baselines
    # + Foundation stacked in one column, Task-specific in the other)
    # is taller than wide, which fits the near-square empty region
    # better than the default 3-col layout.
    #
    # The top-right placement requires the grid to have at least 3
    # rows (``legend_box=(-3, -3, -1, -1)`` dereferences rows -3..-1).
    # Smaller grids -- e.g. MEG-only runs whose only tasks (image,
    # typing) both land in a single "Cognitive" row -- don't have those
    # empty cells, so fall back to the default below-the-grid anchor.
    _legend_box: tuple[int, int, int, int] | None = (
        (-3, -3, -1, -1) if axes.shape[0] >= 3 else None
    )
    return finalize_bar_grid(
        fig,
        axes,
        row_widths=[len(row) + 1 for row in row_layouts],
        color_dict=color_dict,
        display_models=model_order,
        output_dir=output_dir,
        stem="core_bar_chart",
        watermark=watermark,
        pre_legend_hook=_pre_legend_hook,
        legend_box=_legend_box,
        legend_ncol=2,
        legend_fontsize=16,
        legend_builder=lambda c, m: build_grouped_legend_two_col(
            c, m, include_overlap=True
        ),
    )


# ---------------------------------------------------------------------------
# Full bar chart (multi-dataset paradigms)
# ---------------------------------------------------------------------------


def _wrap_dataset_name(name: str, max_len: int = 13) -> str:
    """Insert a newline after the 4-digit year in long dataset names.

    ``"Castillos2023BurstCvep100"`` becomes ``"Castillos2023\\nBurstCvep100"``.
    Names shorter than *max_len* are returned unchanged.
    """
    if len(name) <= max_len:
        return name
    m = re.search(r"(\d{4})", name)
    if m:
        pos = m.end()
        return name[:pos] + "\n" + name[pos:]
    return name


@dataclass(frozen=True)
class _FullGridStyle:
    """Visual parameters that drive :func:`plot_full_bar_chart`.

    The ``LETTER_STYLE`` instance below produces the dense letter-format
    chart that ships in the paper; all other knobs are exposed so that
    future styles (e.g. one for slide-deck use) can be added without
    forking the rendering code.
    """

    max_cols: int
    page_size: tuple[float, float]
    # Target height (inches) per subplot row.  The actual figure height is
    # ``min(page_size[1], n_rows * row_height)`` so partial runs (debug
    # subsets with only a few tasks) keep approximately the same per-row
    # density as the full benchmark instead of stretching each row to fill
    # ``page_size[1]``, which produced absurdly tall, narrow cells.
    row_height: float
    title_fs: float
    tick_fs: float
    title_pad: float
    wspace: float
    hspace: float
    show_badge: bool
    show_metric_label: bool
    show_ytick_labels: bool
    show_error_bars: bool
    wrap_long_titles: bool
    task_label_fs: float
    task_label_x_offset: float
    legend_fs: float
    legend_ncol: int
    tight_layout_pad: float
    tight_layout_rect: tuple[float, float, float, float]
    watermark_fontsize: int
    pdf_stem: str
    png_stem: str
    # Where to place the inline legend, expressed as ``axes[row, col]``
    # corner indices (negative indices are supported).  The legend is
    # centered between ``axes[top_row, left_col]`` and
    # ``axes[bot_row, right_col]``.
    legend_box: tuple[int, int, int, int]


LETTER_STYLE = _FullGridStyle(
    # 1 marginal-rank column + up to 9 dataset columns per row.
    max_cols=10,
    page_size=(8.0, 10.5),
    # ~10.5" / 8 rows: matches the per-row density of the original full
    # benchmark layout so cells render as approximately square (column
    # width ~= 8 / (0.85 + 9) ~= 0.81").
    row_height=1.3,
    # ``title_fs=4`` (down from 5) keeps the dense c-VEP / Motor Imagery /
    # P300 rows from overlapping their neighbours when dataset names
    # have very long prefixes (e.g. ``CabreraCastillos2023`` / ``MartinezCagigal2024``)
    # that ``_wrap_dataset_name`` cannot shorten further.
    title_fs=4,
    tick_fs=5,
    title_pad=2,
    wspace=0.20,
    hspace=0.55,
    show_badge=False,
    show_metric_label=False,
    show_ytick_labels=False,
    show_error_bars=False,
    wrap_long_titles=True,
    task_label_fs=5,
    task_label_x_offset=0.018,
    legend_fs=7,
    legend_ncol=3,
    tight_layout_pad=0.2,
    tight_layout_rect=(0.04, 0.0, 1.0, 1.0),
    watermark_fontsize=36,
    pdf_stem="full_bar_chart_letter",
    png_stem="full_bar_chart_letter",
    # Inline legend in the empty trailing cells (cols 5-9 of last 4 rows).
    legend_box=(-4, -5, -1, -1),
)


def _legend_region_is_empty(
    grid: list[list[tp.Any]],
    top_row: int,
    bot_row: int,
    left_col: int,
    right_col: int,
) -> bool:
    """True iff every grid cell in ``[top_row..bot_row, left_col..right_col]`` is ``None``.

    Used to decide between inline legend placement (in trailing empty
    cells, the LETTER_STYLE default) and placement beneath the chart
    (when the cells the legend would land on are populated by real
    subplots, e.g. for a partial run with one task that fills both
    rows densely).  Indices may be negative; they are normalised
    against ``len(grid)`` / ``len(grid[0])``.
    """
    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        return False
    tr = top_row if top_row >= 0 else top_row + n_rows
    br = bot_row if bot_row >= 0 else bot_row + n_rows
    lc = left_col if left_col >= 0 else left_col + n_cols
    rc = right_col if right_col >= 0 else right_col + n_cols
    for r in range(tr, br + 1):
        for c in range(lc, rc + 1):
            if grid[r][c] is not None:
                return False
    return True


def _ordered_multi_tasks(multi_tasks: list[str]) -> list[str]:
    """Order tasks following CATEGORY_ROW_GROUPS / TASK_CATEGORIES."""
    ordered: list[str] = []
    for cat_group in [
        "Cognitive",
        "BCI",
        "Evoked Responses",
        "Clinical",
        "Internal State",
        "Sleep",
        "Phenotyping",
        "Misc",
    ]:
        for t in TASK_CATEGORIES.get(cat_group, []):
            if t in multi_tasks:
                ordered.append(t)
    for t in multi_tasks:
        if t not in ordered:
            ordered.append(t)
    return ordered


def _build_full_row_entries(
    df: pd.DataFrame,
    ordered_tasks: list[str],
    default_studies: dict[str, str],
    max_cols: int,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, int, int]]]:
    """Group ``(task, dataset)`` pairs into chunked rows of width ``max_cols``."""
    row_entries: list[tuple[str, list[str]]] = []
    task_row_spans: list[tuple[str, int, int]] = []

    for task in ordered_tasks:
        task_df = df[df["task_name"] == task]
        datasets = sorted(task_df["dataset_name"].unique())
        core = default_studies.get(task, "")
        if core in datasets:
            datasets.remove(core)
            datasets.insert(0, core)

        first_row = len(row_entries)
        for chunk_start in range(0, len(datasets), max_cols):
            chunk = datasets[chunk_start : chunk_start + max_cols]
            row_entries.append((task, chunk))
        last_row = len(row_entries) - 1
        task_row_spans.append((task, first_row, last_row))
    return row_entries, task_row_spans


@_with_font_overrides
def plot_full_bar_chart(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    style: _FullGridStyle = LETTER_STYLE,
    watermark: str | None = None,
) -> Path | None:
    """Per-dataset bar chart for the Full benchmark variant.

    Each row belongs to one task, each column/subplot to a dataset
    within that task.  Tasks with more datasets than ``style.max_cols``
    wrap onto extra rows.  Long dataset titles can be auto-wrapped via
    ``style.wrap_long_titles``.

    The default :data:`LETTER_STYLE` lays the full chart out on a single
    US-Letter page with an inline legend; the output is a two-page PDF
    (chart + PNG of the chart) where the legend sits in the chart's empty
    trailing cells.  Figure height scales with ``n_rows`` (capped at
    ``style.page_size[1]``) so partial runs keep approximately square
    cells instead of stretching to fill the full page.
    """
    if "dataset_name" not in df.columns:
        return None

    multi_tasks = multi_dataset_tasks(df)
    if not multi_tasks:
        return None

    ordered_tasks = _ordered_multi_tasks(multi_tasks)
    default_studies = default_study_names()
    task_to_metric = df.groupby("task_name")["metric_name"].first().to_dict()

    display_models = (
        df[df["task_name"].isin(ordered_tasks)]["model_name"].unique().tolist()
    )
    color_dict = build_color_dict(display_models)
    model_order = order_models_with_baselines_first(display_models)

    # Reserve column 0 of every row for the marginal mean-rank bar plot;
    # the dataset columns are therefore chunked to ``max_cols - 1`` per row.
    _max_datasets_per_row = style.max_cols - 1
    _marginal_width_ratio = 0.85
    row_entries, task_row_spans = _build_full_row_entries(
        df, ordered_tasks, default_studies, _max_datasets_per_row
    )

    n_rows = len(row_entries)
    if n_rows == 0:
        return None

    # Map each row index to "is this the first wrap-row of its task?" so the
    # marginal panel is rendered only once per task.
    first_row_to_task: dict[int, str] = {first: task for task, first, _ in task_row_spans}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{style.pdf_stem}.pdf"
    png_path = output_dir / f"{style.png_stem}.png"

    handles, labels = build_grouped_legend(
        color_dict, display_models, include_overlap=True
    )

    # Scale the figure height with the number of rows, capped at the
    # letter-page height so the full benchmark still fits on one page.
    # Without this, partial runs (e.g. ``-d`` debug subsets with a
    # single multi-dataset task) stretched a single row to fill the
    # full ``page_size[1]``, producing absurdly tall, narrow cells.
    _fig_h = min(style.page_size[1], n_rows * style.row_height)
    _figsize = (style.page_size[0], _fig_h)

    with PdfPages(pdf_path) as pdf:
        with sns.axes_style("white"):
            fig, axes = plt.subplots(
                n_rows,
                style.max_cols,
                figsize=_figsize,
                squeeze=False,
                gridspec_kw={
                    "wspace": style.wspace,
                    "hspace": style.hspace,
                    "width_ratios": (
                        [_marginal_width_ratio] + [1.0] * _max_datasets_per_row
                    ),
                },
            )

        grid: list[list[SubplotSpec | MarginalRankSpec | None]] = []
        for row_idx, (task, datasets) in enumerate(row_entries):
            task_df = df[df["task_name"] == task]
            core = default_studies.get(task, "")
            metric_key = task_to_metric.get(task, "")

            # Column 0: marginal mean-rank for the *whole* task, but only on
            # the first wrap-row of that task; continuation rows leave the
            # marginal blank to avoid duplicating the same bars.
            row_specs: list[SubplotSpec | MarginalRankSpec | None]
            if first_row_to_task.get(row_idx) == task:
                rank_stats = compute_row_rank_stats(
                    df,
                    row_tasks=[task],
                    model_order=model_order,
                    by_dataset=True,
                )
                # Generic subtitle -- the task name is already in the
                # left-margin row label, so the marginal only needs to
                # say what's being aggregated.
                row_specs = [
                    MarginalRankSpec(rank_stats=rank_stats, subtitle="over datasets")
                ]
            else:
                row_specs = [None]

            for ds_name in datasets:
                sub_df = task_df[task_df["dataset_name"] == ds_name]
                agg = (
                    sub_df.groupby("model_name")["metric_value"]
                    .agg(["mean", "sem"])
                    .reindex(model_order)
                )
                row_specs.append(
                    SubplotSpec(
                        title=(
                            _wrap_dataset_name(ds_name)
                            if style.wrap_long_titles
                            else ds_name
                        ),
                        agg=agg,
                        dataset_name=ds_name,
                        metric_key=metric_key,
                        title_weight="bold" if ds_name == core else "normal",
                    )
                )
            row_specs.extend([None] * (style.max_cols - len(row_specs)))
            grid.append(row_specs)

        render_bars_grid(
            axes,
            grid,
            model_order,
            color_dict,
            title_fontsize=style.title_fs,
            tick_labelsize=style.tick_fs,
            title_pad=style.title_pad,
            show_badge=style.show_badge,
            show_metric_label=style.show_metric_label,
            show_ytick_labels=style.show_ytick_labels,
            show_error_bars=style.show_error_bars,
        )

        # Hide the marginal panel's "Best (0)" / "Worst (1)" y-tick
        # labels on every row but the first.  The labels are identical
        # across rows (the marginal y-axis is the same normalised-rank
        # scale everywhere), so repeating them on every task row is
        # visual noise; keeping them only on row 0 leaves the reader a
        # single legend-like reference at the top of the column.  Tick
        # marks themselves stay visible so the bars still read against
        # a defined scale.
        for _row_idx in range(1, axes.shape[0]):
            _ax_marginal = axes[_row_idx, 0]
            if _ax_marginal.get_visible():
                _ax_marginal.set_yticklabels([])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout(pad=style.tight_layout_pad, rect=style.tight_layout_rect)

        # Task labels in the left margin (after tight_layout so positions are
        # final).  Anchored on column 0 (the marginal panel) so the labels
        # sit in the figure's left margin, *before* the very first subplot
        # of the row -- not between the marginal and the bar block.
        for task, first_row, last_row in task_row_spans:
            nice_task = TASK_DISPLAY_NAMES.get(task, task.replace("_", " ").title())
            cat = TASK_TO_CATEGORY.get(task, "")
            draw_left_margin_label(
                fig,
                axes,
                label=nice_task.replace("\n", " "),
                first_row=first_row,
                last_row=last_row,
                color=CATEGORY_COLORS.get(cat, "#555555"),
                fontsize=style.task_label_fs,
                x_offset=style.task_label_x_offset,
                anchor_col=0,
            )

        # Inline legend in the empty region defined by ``style.legend_box``.
        top_row, left_col, bot_row, right_col = style.legend_box
        # ``legend_box`` is tuned for the full benchmark grid; for smaller
        # task subsets (e.g. ``-d`` debug runs) ``n_rows`` may be too small
        # for the hardcoded negative offsets, which would raise IndexError
        # below.  Clamp each index into ``axes.shape`` while preserving the
        # negative-indexing semantics so the legend collapses gracefully.
        _n_rows_ax, _n_cols_ax = axes.shape
        top_row = max(-_n_rows_ax, min(top_row, _n_rows_ax - 1))
        bot_row = max(-_n_rows_ax, min(bot_row, _n_rows_ax - 1))
        left_col = max(-_n_cols_ax, min(left_col, _n_cols_ax - 1))
        right_col = max(-_n_cols_ax, min(right_col, _n_cols_ax - 1))

        _legend_kwargs: dict[str, tp.Any] = dict(
            ncol=style.legend_ncol,
            fontsize=style.legend_fs,
            frameon=True,
            edgecolor="#cccccc",
            fancybox=False,
            borderpad=0.8,
            columnspacing=1.2,
            handlelength=1.2,
            handletextpad=0.4,
        )
        if _legend_region_is_empty(grid, top_row, bot_row, left_col, right_col):
            # Inline placement: centre the legend on the empty trailing
            # cells, as designed for the full benchmark layout.
            _leg_top = axes[top_row, left_col].get_position()
            _leg_bot = axes[bot_row, right_col].get_position()
            _leg_cx = (_leg_top.x0 + _leg_bot.x1) / 2
            _leg_cy = (_leg_top.y1 + _leg_bot.y0) / 2
            legend = fig.legend(
                handles,
                labels,
                loc="center",
                bbox_to_anchor=(_leg_cx, _leg_cy),
                **_legend_kwargs,
            )
        else:
            # The cells the inline legend would land on are populated
            # by real subplots (e.g. a partial run with one task that
            # fills both rows densely).  Place the legend just beneath
            # the bottom-most axes instead; ``bbox_inches="tight"`` on
            # save will expand the saved bounding box to include it.
            # Anchoring on the actual axes position (rather than on the
            # figure's bottom edge) keeps the legend snug against the
            # chart even when ``tight_layout`` reserves padding below
            # the axes for tick labels.
            _bottom_y = axes[-1, 0].get_position().y0
            legend = fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, _bottom_y - 0.01),
                bbox_transform=fig.transFigure,
                **_legend_kwargs,
            )
        bold_legend_titles(legend)

        if watermark:
            add_watermark(fig, watermark, fontsize=style.watermark_fontsize)

        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2)
        fig.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)

    return png_path
