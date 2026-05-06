# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Rank box-plot for the NeuralBench-Core summary view."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from neuralbench.plots._style import (
    add_watermark,
    apply_two_tone_labels,
    bold_legend_titles,
    build_color_dict,
    build_grouped_legend_stacked,
    darken,
    save_figure,
)
from neuralbench.plots.ranking import _prepare_rank_data


def _render_rank_boxplot(
    long_df: pd.DataFrame,
    model_order: list[str],
    palette: dict[str, str],
    ax: plt.Axes,  # type: ignore[name-defined]
    n_models: int,
) -> None:
    """Draw a horizontal boxplot chart of per-task normalized ranks.

    Ranks are min-max scaled to ``[0, 1]`` via
    ``(rank - 1) / (n_models - 1)`` so that the best model lands at 0
    and the worst at 1, regardless of the number of models.
    """
    box_data = [
        long_df.loc[long_df["model_name"] == m, "norm_rank"].dropna().values
        for m in model_order
    ]
    bp = ax.boxplot(
        box_data,  # type: ignore[arg-type]
        positions=range(n_models),
        vert=False,
        widths=0.5,
        showfliers=True,
        patch_artist=True,
        boxprops=dict(linewidth=0.8),
        medianprops=dict(color="k", linewidth=1.2),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker=".", markersize=4, alpha=0.45),
        zorder=2,
    )
    for i, (patch, name) in enumerate(zip(bp["boxes"], model_order)):
        fill = palette.get(name, "#888888")
        edge = darken(fill)
        patch.set_facecolor(fill)
        patch.set_edgecolor(edge)
        bp["medians"][i].set_color(edge)
        for artist_key in ("whiskers", "caps"):
            for j in (2 * i, 2 * i + 1):
                bp[artist_key][j].set_color(edge)
        bp["fliers"][i].set_markerfacecolor(edge)
        bp["fliers"][i].set_markeredgecolor("none")

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_order)

    mean_norm_ranks = long_df.groupby("model_name")["norm_rank"].mean()
    annot_x = 1.13
    for i, name in enumerate(model_order):
        ax.text(
            annot_x,
            i,
            f"{mean_norm_ranks[name]:.2f}",
            va="center",
            ha="center",
            fontsize=7,
            color="#555555",
        )

    ax.invert_yaxis()
    ax.text(
        annot_x,
        -0.8,
        "Mean norm.\nrank",
        va="bottom",
        ha="center",
        fontsize=6.5,
        color="#555555",
        fontweight="bold",
        linespacing=0.9,
    )
    ax.set_xlim(-0.03, 1.22)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    for x in (0.25, 0.5, 0.75, 1.0):
        ax.axvline(
            x,
            color="#cccccc",
            linewidth=0.6,
            linestyle=":",
            zorder=1,
        )
    ax.set_ylabel("")
    ax.set_xlabel("Normalized rank (lower is better)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_core_rank_boxplot(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    watermark: str | None = None,
) -> Path:
    """Box chart of per-task normalized ranks, sorted by mean, colored by group.

    Ranks are min-max normalized to ``[0, 1]`` (best model = 0, worst
    model = 1) so the x-axis scale is independent of the number of
    models being compared.
    """
    long_df = _prepare_rank_data(df)

    mean_ranks = long_df.groupby("model_name")["rank"].mean().sort_values()
    model_order = mean_ranks.index.tolist()
    n_models = len(model_order)

    if n_models < 2:
        raise ValueError(
            f"Normalized-rank boxplot requires at least 2 models; got {n_models}."
        )
    long_df["norm_rank"] = (long_df["rank"] - 1.0) / (n_models - 1)

    display_models = long_df["model_name"].unique().tolist()
    color_dict = build_color_dict(display_models)
    palette = {m: color_dict.get(m, "#888888") for m in model_order}

    # Sized for a portrait US-letter document with ~1" side margins:
    # full usable width (~6.5") and at most ~33% of the 11" page height.
    # The legend sits to the right of the axes and stacks the three
    # model groups in a single column.
    fig_width = 6.5
    fig_height = 3.5
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    fig.subplots_adjust(left=0.25, right=0.71, top=0.93, bottom=0.18)

    _render_rank_boxplot(long_df, model_order, palette, ax, n_models)
    apply_two_tone_labels(fig, ax, fontsize=7)

    handles, labels = build_grouped_legend_stacked(color_dict, display_models)

    ax_pos = ax.get_position()
    legend = fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(ax_pos.x1 + 0.02, ax_pos.y1),
        bbox_transform=fig.transFigure,
        ncol=1,
        fontsize=7,
        frameon=True,
        edgecolor="#cccccc",
        handlelength=1.0,
        handletextpad=0.5,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    bold_legend_titles(legend)

    if watermark:
        add_watermark(fig, watermark)

    return save_figure(fig, output_dir, "core_rank_boxplot")
