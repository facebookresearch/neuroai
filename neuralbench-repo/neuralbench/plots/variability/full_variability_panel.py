# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Combined two-panel figure: core-vs-full bump chart + rank-variability heatmap.

This is the paper-friendly artefact built on top of the two
NeuralBench-Full variability panels.  It stacks them vertically with
``A`` / ``B`` labels so they can travel as a single figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from neuralbench.plots._style import (
    add_watermark,
    save_figure,
)
from neuralbench.plots.variability.core_vs_full import (
    _FONT_OVERRIDES as _BUMP_FONT_OVERRIDES,
)
from neuralbench.plots.variability.core_vs_full import (
    _bump_chart_figsize,
    _draw_core_vs_full_bump_chart,
)
from neuralbench.plots.variability.rank_variability_matrix import (
    _draw_model_task_rank_variability_heatmap,
    _heatmap_figsize,
)

_PANEL_LABEL_FS = 14
_PANEL_LABEL_X = 0.005

# Multiplier on the heatmap's text/colour-bar sizes when rendered as
# Panel B.  The standalone heatmap fonts are tuned for a compact
# stand-alone figure; in the combined figure they read too small next
# to Panel A, so we boost them.
_HEATMAP_FONT_SCALE = 1.4


def plot_full_variability_panel(
    core_df: pd.DataFrame,
    full_df: pd.DataFrame,
    output_dir: Path,
    *,
    min_datasets: int = 5,
    watermark: str | None = None,
) -> Path | None:
    """Build the 2-panel (bump chart + variability heatmap) figure.

    Panel **A** is the Core-vs-Full bump chart and panel **B** is the
    per-(model, task) rank-variability heatmap.  Both are rendered in
    nested :class:`~matplotlib.figure.SubFigure` instances so the
    combined PDF stays vector-quality.

    Returns the saved PNG path, or ``None`` when either panel cannot be
    built (e.g. ``full_df`` lacks a ``dataset_name`` column or no
    paradigm passes the ``min_datasets`` threshold).
    """
    bump_size = _bump_chart_figsize(core_df, full_df)
    heatmap_size = _heatmap_figsize(
        full_df, min_datasets=min_datasets, font_scale=_HEATMAP_FONT_SCALE
    )
    if bump_size is None or heatmap_size is None:
        return None

    bump_w, bump_h = bump_size
    heat_w, heat_h = heatmap_size

    # Visible gap (inches) between Panel A's rank axis labels and the
    # Panel B bracket strip ("Task-specific" / "Foundation" / "Baseline").
    gap_h = 1.5

    # Floor on the figure width so both panels have plenty of
    # horizontal room.  The bump chart's standalone width caps at 8.0"
    # and the heatmap at 7.5"; this floor lets the combined figure go
    # wider so each panel breathes.
    fig_w = max(bump_w, heat_w, 10.0)
    fig_h = bump_h + heat_h + gap_h

    with plt.rc_context(_BUMP_FONT_OVERRIDES), sns.axes_style("white"):
        fig = plt.figure(figsize=(fig_w, fig_h))
        sub_top, sub_bot = fig.subfigures(
            2,
            1,
            height_ratios=[bump_h + gap_h, heat_h],
            hspace=0.0,
        )

        # Move the bump chart's legend to the upper-right so the
        # upper-left corner is reserved for the "A" panel label.  In
        # the standalone chart the legend stays at "upper left".
        if not _draw_core_vs_full_bump_chart(
            sub_top, core_df, full_df, legend_loc="upper right"
        ):
            plt.close(fig)
            return None
        # Stretch the bump chart to use the full panel width and push
        # the axes upward so the rank-axis labels don't run into the
        # heatmap's bracket strip below.  ``left`` / ``right`` override
        # matplotlib's default 0.125 / 0.9 margins, which would
        # otherwise leave Panel A noticeably narrower than Panel B.
        sub_top.subplots_adjust(
            left=0.005,
            right=0.998,
            bottom=gap_h / (bump_h + gap_h),
            top=0.98,
        )

        bracket_label_y = _draw_model_task_rank_variability_heatmap(
            sub_bot,
            full_df,
            min_datasets=min_datasets,
            font_scale=_HEATMAP_FONT_SCALE,
        )
        if bracket_label_y is None:
            plt.close(fig)
            return None

        # Panel labels.  Render them on the parent figure (rather than
        # each sub-figure) so we can express both Ys in a single
        # coordinate system and anchor each letter to its panel's *top
        # visible content*:
        #
        #   * "A" -> top edge of Panel A (top of sub_top).
        #   * "B" -> bracket-label band of Panel B (the brackets sit
        #     just above the heatmap's x-tick labels and form the top
        #     visible row of Panel B).
        #
        # ``bracket_label_y`` is in sub_bot's transSubfigure coords, so
        # we map it back to figure-y by composing with sub_bot's bbox.
        sub_top_y1_fig = float(sub_top.bbox.y1 / fig.bbox.height)
        sub_bot_y0_fig = float(sub_bot.bbox.y0 / fig.bbox.height)
        sub_bot_h_fig = float(sub_bot.bbox.height / fig.bbox.height)

        a_y = sub_top_y1_fig - 0.005
        b_y = sub_bot_y0_fig + bracket_label_y * sub_bot_h_fig

        fig.text(
            _PANEL_LABEL_X,
            a_y,
            "A",
            fontsize=_PANEL_LABEL_FS,
            fontweight="bold",
            ha="left",
            va="top",
        )
        fig.text(
            _PANEL_LABEL_X,
            b_y,
            "B",
            fontsize=_PANEL_LABEL_FS,
            fontweight="bold",
            ha="left",
            va="center",
        )

        if watermark:
            add_watermark(fig, watermark)

        return save_figure(fig, Path(output_dir), "full_variability_panel")
