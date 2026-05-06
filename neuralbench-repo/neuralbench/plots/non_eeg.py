# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Non-EEG (MEG, fMRI, ...) bar-chart plot.

The single public entry point :func:`plot_non_eeg_bar_chart` arranges
all non-EEG ``(task, core-dataset)`` combinations in a single row of
subplots and draws a coloured device header bracket above each group of
contiguous same-device subplots.  Layout is finalised through the
shared :func:`finalize_bar_grid` helper with a custom legend builder
and a pre-legend hook that places the device headers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from neuralbench.plots._constants import (
    CATEGORY_ROW_GROUPS,
    DEVICE_DISPLAY_NAMES,
    DUMMY_DISPLAY,
    FMRI_CLASSIC_DISPLAY,
    FMRI_FM_DISPLAY,
    MEEG_CLASSIC_DISPLAY,
    MEEG_FM_DISPLAY,
    TASK_CATEGORIES,
    TASK_DISPLAY_NAMES,
)
from neuralbench.plots._style import (
    _LEGEND_TITLE_MARK,
    ColorGroup,
    LegendGroup,
    base_name,
    build_color_dict_from_groups,
    build_grouped_legend_from_groups,
    finalize_bar_grid,
    ordered_models,
    render_bars_on_axis,
)
from neuralbench.plots._task_metadata import (
    build_task_device_map,
    build_task_n_examples_map,
    build_task_split_map,
    default_studies_per_task,
)

# Custom purple colormap anchored on the matplotlib tab:purple #9467bd.
# Used for fMRI classic bars; green is intentionally avoided here since
# ``FEATURE_BASED_COLOR`` reserves a sage-green hue for the Handcrafted
# baseline in the main bar charts.
_FMRI_PURPLES = LinearSegmentedColormap.from_list(
    "fmri_purples",
    ["#ece3f2", "#9467bd", "#4a3470"],
)


def _build_non_eeg_color_dict(display_models: list[str]) -> dict[str, str]:
    """Color mapping with distinct palettes per device/group combination.

    M/EEG classic → Blues, M/EEG foundation → Oranges (matching the main
    bar chart), fMRI classic → Purples (anchored on ``#9467bd``),
    fMRI foundation → Reds, baselines → Greys.  Purples -- rather than
    the previous Greens -- are used for fMRI so its bars don't clash
    with the sage-green ``FEATURE_BASED_COLOR`` used for the Handcrafted
    baseline elsewhere.
    """
    base_to_full = {base_name(m): m for m in display_models}

    def _pick(ordered: list[str]) -> list[str]:
        return [base_to_full[m] for m in ordered if m in base_to_full]

    cmap_blues = cm.Blues  # type: ignore[attr-defined]
    cmap_oranges = cm.Oranges  # type: ignore[attr-defined]
    cmap_reds = cm.Reds  # type: ignore[attr-defined]
    cmap_greys = cm.Greys  # type: ignore[attr-defined]
    return build_color_dict_from_groups(
        [
            ColorGroup(
                members=_pick(MEEG_CLASSIC_DISPLAY), cmap=cmap_blues, lo=0.35, span=0.55
            ),
            ColorGroup(
                members=_pick(MEEG_FM_DISPLAY), cmap=cmap_oranges, lo=0.35, span=0.55
            ),
            ColorGroup(
                members=_pick(FMRI_CLASSIC_DISPLAY),
                cmap=_FMRI_PURPLES,
                lo=0.40,
                span=0.45,
            ),
            ColorGroup(
                members=_pick(FMRI_FM_DISPLAY), cmap=cmap_reds, lo=0.40, span=0.45
            ),
            ColorGroup(members=_pick(DUMMY_DISPLAY), cmap=cmap_greys, lo=0.30, span=0.25),
        ]
    )


def _build_non_eeg_legend(
    color_dict: dict[str, str],
    display_models: list[str],
) -> tuple[list, list[str]]:
    """Legend columns: EEG classic, fMRI classic, EEG FM, fMRI FM, Baselines.

    Empty groups (e.g. fMRI foundation models when no such models exist
    in the collected results) are dropped from the legend entirely.
    """
    base_to_full = {base_name(m): m for m in display_models}

    def _pick(ordered: list[str]) -> list[str]:
        return [base_to_full[m] for m in ordered if m in base_to_full]

    return build_grouped_legend_from_groups(
        color_dict,
        [
            LegendGroup(title="Baselines", members=_pick(DUMMY_DISPLAY)),
            LegendGroup(
                title="EEG task-specific models", members=_pick(MEEG_CLASSIC_DISPLAY)
            ),
            LegendGroup(title="EEG foundation models", members=_pick(MEEG_FM_DISPLAY)),
            LegendGroup(
                title="fMRI task-specific models", members=_pick(FMRI_CLASSIC_DISPLAY)
            ),
            LegendGroup(title="fMRI foundation models", members=_pick(FMRI_FM_DISPLAY)),
        ],
        drop_empty=True,
    )


def _order_with_baselines_first(
    models: list[str], baselines_order: list[str]
) -> list[str]:
    """``ordered_models`` with the dummy baselines pushed to the front."""
    ordered = ordered_models(models)
    baseline_rank = {name: idx for idx, name in enumerate(baselines_order)}
    baselines = sorted(
        (m for m in ordered if base_name(m) in baseline_rank),
        key=lambda m: baseline_rank[base_name(m)],
    )
    rest = [m for m in ordered if base_name(m) not in baseline_rank]
    return baselines + rest


def _build_subplot_entries(
    df: pd.DataFrame, devices: tuple[str, ...]
) -> list[tuple[str, str, str]]:
    """Resolve the flat ``(device, task, core-dataset)`` tuple for each subplot."""
    studies_per_task = default_studies_per_task()

    ordered_task_order: list[str] = []
    for row_group in CATEGORY_ROW_GROUPS:
        for cat in row_group:
            ordered_task_order.extend(TASK_CATEGORIES.get(cat, []))

    entries: list[tuple[str, str, str]] = []
    for device in devices:
        sub = df[df["_device"] == device]
        if sub.empty:
            continue
        available_tasks = set(sub["task_name"].unique())
        ordered_tasks = [t for t in ordered_task_order if t in available_tasks] + sorted(
            t for t in available_tasks if t not in ordered_task_order
        )
        for task in ordered_tasks:
            task_sub = sub[sub["task_name"] == task]
            datasets = sorted(task_sub["dataset_name"].unique())
            # The same task name can be registered under multiple devices
            # (e.g., MEG ``image`` and fMRI ``image``), so any of the
            # per-task defaults that is present in this device's datasets
            # is a valid core dataset -- otherwise fall back to the first one.
            candidates = studies_per_task.get(task, [])
            dataset = next((c for c in candidates if c in datasets), datasets[0])
            entries.append((device, task, dataset))
    return entries


def _device_spans(entries: list[tuple[str, str, str]]) -> list[tuple[str, int, int]]:
    """Group consecutive entries that share the same device.

    Returns ``[(device, col_start, col_end_inclusive), ...]``.
    """
    spans: list[tuple[str, int, int]] = []
    if not entries:
        return spans
    n_cols = len(entries)
    start = 0
    for i in range(1, n_cols + 1):
        if i == n_cols or entries[i][0] != entries[start][0]:
            spans.append((entries[start][0], start, i - 1))
            start = i
    return spans


def _make_device_header_hook(
    entries: list[tuple[str, str, str]],
    n_cols: int,
) -> Callable[[plt.Figure, np.ndarray], None]:
    """Return a ``finalize_bar_grid`` ``pre_legend_hook`` that draws device brackets."""
    spans = _device_spans(entries)
    # Keep the device headers neutral so they don't compete with the bar
    # palettes (MEG/fMRI subplots already use distinct Blues/Oranges vs
    # Greens palettes to distinguish device, so the bracket need not be
    # coloured as well).
    header_color = "#333333"

    def hook(fig: plt.Figure, axes: np.ndarray) -> None:  # type: ignore[name-defined]
        # Place device headers just above the tallest subplot title.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
        title_top_fig = max(
            fig.transFigure.inverted().transform(
                (0, axes[0, c].title.get_window_extent(renderer).y1)
            )[1]
            for c in range(n_cols)
        )
        line_y = title_top_fig + 0.04
        text_y = line_y + 0.015
        for device, col_start, col_end in spans:
            left = axes[0, col_start].get_position().x0
            right = axes[0, col_end].get_position().x1
            fig.text(
                (left + right) / 2,
                text_y,
                DEVICE_DISPLAY_NAMES.get(device, device.upper()),
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold",
                color=header_color,
            )
            fig.add_artist(
                plt.Line2D(
                    [left, right],
                    [line_y, line_y],
                    transform=fig.transFigure,
                    color=header_color,
                    linewidth=2.0,
                    solid_capstyle="butt",
                )
            )

    return hook


def plot_non_eeg_bar_chart(
    df: pd.DataFrame,
    output_dir: Path,  # type: ignore[name-defined]
    *,
    devices: tuple[str, ...] = ("meg", "fmri"),
    watermark: str | None = None,
) -> Path | None:
    """Small bar chart highlighting neuralbench's non-EEG tasks.

    Arranges all non-EEG (task, core-dataset) combinations in a single
    row of subplots.  Contiguous subplots sharing the same recording
    device are grouped under a coloured device header (e.g. "MEG",
    "fMRI"), mirroring the category-header style used by
    :func:`plot_bar_chart`.  Tasks within a device are ordered following
    ``CATEGORY_ROW_GROUPS`` / ``TASK_CATEGORIES``.

    Returns ``None`` (without writing any file) when no rows in *df*
    match any of the requested devices.
    """
    from pathlib import Path

    if "dataset_name" not in df.columns:
        return None

    device_map = build_task_device_map()
    task_device = df.set_index(["task_name", "dataset_name"]).index.map(
        lambda key: device_map.get(tuple(key), "eeg")
    )
    df = df.assign(_device=task_device.values)

    entries = _build_subplot_entries(df, devices)
    if not entries:
        return None

    task_to_metric = df.groupby("task_name")["metric_name"].first().to_dict()

    display_models = df[df["_device"].isin(devices)]["model_name"].unique().tolist()
    color_dict = _build_non_eeg_color_dict(display_models)

    # Per-device model order: keep each subplot compact (fMRI column
    # shouldn't reserve empty slots for EEG/MEG classic models).
    device_model_orders: dict[str, list[str]] = {}
    for device, _, _ in entries:
        if device not in device_model_orders:
            device_models = df[df["_device"] == device]["model_name"].unique().tolist()
            device_model_orders[device] = _order_with_baselines_first(
                device_models, DUMMY_DISPLAY
            )

    n_cols = len(entries)
    _subplot_w, _subplot_h = 3, 3
    # Extra vertical padding at the top leaves room for the device headers.
    figsize = (n_cols * _subplot_w, _subplot_h + 0.6)

    with sns.axes_style("white"):
        fig, axes = plt.subplots(
            1,
            n_cols,
            figsize=figsize,
            squeeze=False,
            gridspec_kw={"wspace": 0.3},
        )

    split_map = build_task_split_map()
    n_examples_map = build_task_n_examples_map()

    for col_idx, (device, task_name, dataset_name) in enumerate(entries):
        ax = axes[0, col_idx]
        row_models = device_model_orders[device]
        n_row_models = len(row_models)
        sub_df = df[
            (df["task_name"] == task_name)
            & (df["dataset_name"] == dataset_name)
            & (df["_device"] == device)
        ]
        agg = (
            sub_df.groupby("model_name")["metric_value"]
            .agg(["mean", "sem"])
            .reindex(row_models)
        )
        nice_task = TASK_DISPLAY_NAMES.get(task_name, task_name.replace("_", " ").title())
        metric_key = task_to_metric.get(task_name, "")
        render_bars_on_axis(
            ax,
            agg,
            row_models,
            color_dict,
            dataset_name,
            metric_key,
            n_row_models,
            title=nice_task.replace("\n", " "),
            split_kind=split_map.get((task_name, dataset_name)),
            n_examples=n_examples_map.get((task_name, dataset_name)),
        )
        ax.set_box_aspect(1)

    # ``finalize_bar_grid`` runs ``tight_layout`` then the pre-legend hook
    # then anchors the legend.  We pre-compute the column count from the
    # legend payload so we know the ``ncol`` to forward.
    handles_for_count, labels_for_count = _build_non_eeg_legend(
        color_dict, display_models
    )
    # Each non-empty group contributes exactly one bold-title label
    # (marked with ``_LEGEND_TITLE_MARK``); that count is also the number
    # of legend columns we want matplotlib to lay out.
    n_legend_cols = sum(
        1 for lbl in labels_for_count if lbl.startswith(_LEGEND_TITLE_MARK)
    )

    return finalize_bar_grid(
        fig,
        axes,
        row_widths=[n_cols],
        color_dict=color_dict,
        display_models=display_models,
        output_dir=Path(output_dir),
        stem="core_non_eeg_bar_chart",
        watermark=watermark,
        legend_ncol=n_legend_cols,
        legend_fontsize=11,
        tight_layout_kwargs={"rect": (0.0, 0.0, 1.0, 0.85)},
        pre_legend_hook=_make_device_header_hook(entries, n_cols),
        legend_builder=_build_non_eeg_legend,
        legend_anchor_offset=0.05,
    )
