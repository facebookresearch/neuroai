# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Core-vs-Full bump (slopegraph) chart."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy import stats as sps

from neuralbench.plots._style import group_color
from neuralbench.plots.ranking import (
    compute_core_mean_ranks as _compute_core_mean_ranks,
)
from neuralbench.plots.ranking import (
    compute_full_mean_ranks as _compute_full_mean_ranks,
)

_BUMP_COLOR_IMPROVED = "tab:green"
_BUMP_COLOR_WORSENED = "black"
_BUMP_COLOR_STABLE = "#bbbbbb"

# Pin matplotlib's default font so seaborn / moabb global theme tweaks
# (which can set ``font.family`` to ``"serif"`` or ``"Arial"`` on import)
# don't bleed into this figure.
_FONT_OVERRIDES = {
    "font.family": ["sans-serif"],
    "font.sans-serif": ["DejaVu Sans"]
    + [f for f in plt.rcParams["font.sans-serif"] if f != "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _bump_line_color(pos_diff: int) -> tuple[str, int]:
    """Return ``(color, zorder)`` for an ordinal rank-position change.

    ``pos_diff`` is ``core_position - full_position`` where the position
    is the integer ordinal rank in the global ordering (1 = best).  A
    positive diff means the model moved *up* the leaderboard (e.g. 6th
    on Core -> 4th on Full, ``pos_diff = +2``).
    """
    if pos_diff >= 1:
        return _BUMP_COLOR_IMPROVED, 3
    if pos_diff <= -1:
        return _BUMP_COLOR_WORSENED, 3
    return _BUMP_COLOR_STABLE, 2


def _prepare_bump_chart_df(
    core_df: pd.DataFrame, full_df: pd.DataFrame
) -> pd.DataFrame | None:
    """Return the per-model bump-chart data, or ``None`` when not buildable."""
    if "dataset_name" not in full_df.columns:
        return None

    core_ranks = _compute_core_mean_ranks(core_df)
    full_ranks = _compute_full_mean_ranks(full_df)

    df = pd.DataFrame({"core_rank": core_ranks, "full_rank": full_ranks}).dropna()
    if df.empty:
        return None

    df["base_name"] = df.index.to_series()
    df["display_name"] = df["base_name"]
    df["core_pos"] = df["core_rank"].rank(method="min", ascending=True).astype(int)
    df["full_pos"] = df["full_rank"].rank(method="min", ascending=True).astype(int)
    df["pos_diff"] = df["core_pos"] - df["full_pos"]
    df = df.sort_values("core_rank").reset_index(drop=True)
    return df


def _bump_chart_figsize(
    core_df: pd.DataFrame, full_df: pd.DataFrame
) -> tuple[float, float] | None:
    """Return the standalone ``(width, height)`` used by the bump chart, or ``None``."""
    df = _prepare_bump_chart_df(core_df, full_df)
    if df is None:
        return None
    n_models = len(df)
    width = min(8.0, max(6.5, 0.48 * n_models))
    return (width, 3.2)


def _draw_core_vs_full_bump_chart(
    parent: Figure | SubFigure,
    core_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    legend_loc: str = "upper left",
) -> bool:
    """Draw the core-vs-full bump chart onto *parent*.

    Parameters
    ----------
    legend_loc:
        Where to place the in-axes legend.  Defaults to ``"upper left"``
        (the standalone chart's home), but can be overridden, e.g. to
        ``"upper right"`` when the upper-left corner is needed for an
        external panel label.

    Returns ``True`` on success, or ``False`` when the input data is
    missing the columns / rows required to build the chart.
    """
    df = _prepare_bump_chart_df(core_df, full_df)
    if df is None:
        return False

    tau, tau_p = sps.kendalltau(df["core_rank"], df["full_rank"])
    rho, rho_p = sps.spearmanr(df["core_rank"], df["full_rank"])

    def _fmt_p(p: float) -> str:
        return "p < 0.001" if p < 1e-3 else f"p = {p:.3g}"

    print(
        "[plot_core_vs_full_bump_chart] "
        f"Kendall tau = {tau:.3f} ({_fmt_p(tau_p)})    "
        f"Spearman rho = {rho:.3f} ({_fmt_p(rho_p)})"
    )

    n_models = len(df)

    with plt.rc_context(_FONT_OVERRIDES), sns.axes_style("white"):
        ax = parent.subplots()

        y_core, y_full = 1.0, 0.0
        label_offset = 0.25

        rank_min = float(min(df["core_rank"].min(), df["full_rank"].min()))
        rank_max = float(max(df["core_rank"].max(), df["full_rank"].max()))
        rank_top = max(1, int(np.floor(rank_min)))
        rank_bottom = min(n_models, int(np.ceil(rank_max)))

        label_packing_min_sep = 0.55
        x_max_allowed = float(rank_bottom) + 0.4

        def _pack_label_x(values: np.ndarray) -> np.ndarray:
            order = np.argsort(values)
            sorted_vals = values[order].astype(float).copy()
            for i in range(1, len(sorted_vals)):
                if sorted_vals[i] - sorted_vals[i - 1] < label_packing_min_sep:
                    sorted_vals[i] = sorted_vals[i - 1] + label_packing_min_sep
            if sorted_vals[-1] > x_max_allowed:
                sorted_vals[-1] = x_max_allowed
                for i in range(len(sorted_vals) - 2, -1, -1):
                    if sorted_vals[i + 1] - sorted_vals[i] < label_packing_min_sep:
                        sorted_vals[i] = sorted_vals[i + 1] - label_packing_min_sep
            out = np.empty_like(sorted_vals)
            out[order] = sorted_vals
            return out

        core_label_x = _pack_label_x(df["core_rank"].to_numpy())

        for i, (_, row) in enumerate(df.iterrows()):
            c_rank = row["core_rank"]
            f_rank = row["full_rank"]
            color, zorder = _bump_line_color(int(row["pos_diff"]))

            arrow = FancyArrowPatch(
                (c_rank, y_core),
                (f_rank, y_full),
                arrowstyle="-|>",
                mutation_scale=10,
                color=color,
                linewidth=1.5,
                shrinkA=0,
                shrinkB=0,
                zorder=zorder,
            )
            ax.add_patch(arrow)
            ax.plot(
                [c_rank],
                [y_core],
                marker="o",
                markersize=4,
                color=color,
                zorder=zorder,
            )

            label_color = group_color(row["base_name"])
            c_label_x = float(core_label_x[i])

            if not np.isclose(c_label_x, c_rank, atol=1e-3):
                ax.plot(
                    [c_rank, c_label_x],
                    [y_core, y_core + label_offset],
                    color="#aaaaaa",
                    linewidth=0.5,
                    zorder=1,
                )
            ax.text(
                c_label_x,
                y_core + label_offset,
                row["display_name"],
                ha="left",
                va="bottom",
                rotation=30,
                rotation_mode="anchor",
                fontsize=8,
                color=label_color,
            )
        title_x = rank_top - 0.6
        ax.set_xlim(title_x - 3.8, rank_bottom + 0.4)
        ax.set_ylim(y_full - 0.85, y_core + 1.40)

        _title_bbox = dict(
            facecolor="white",
            edgecolor="none",
            boxstyle="round,pad=0.3",
        )
        ax.text(
            title_x,
            y_core,
            "NeuralBench-EEG-Core v1.0\n(1 dataset/task)",
            ha="right",
            va="center",
            fontsize=10,
            bbox=_title_bbox,
            zorder=5,
        )
        ax.text(
            title_x,
            y_full,
            "NeuralBench-EEG-Full v1.0\n(All datasets)",
            ha="right",
            va="center",
            fontsize=10,
            bbox=_title_bbox,
            zorder=5,
        )

        xticks = list(range(rank_top, rank_bottom + 1))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks], fontsize=7)
        rank_axis_y = y_full - 0.40
        ax.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=True,
            labeltop=False,
            length=2,
            direction="out",
            color="#888888",
            labelcolor="#666666",
        )
        ax.set_xlabel("Mean rank", fontsize=8, color="#666666", labelpad=4)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color("#cccccc")
        ax.spines["bottom"].set_position(("data", rank_axis_y))
        ax.spines["bottom"].set_bounds(3, rank_bottom)
        for y in (y_core, y_full):
            ax.axhline(
                y=y,
                color="#dddddd",
                linewidth=0.6,
                zorder=0,
                xmin=0,
                xmax=1,
            )

        def _arrow_handle(color: str, label: str) -> Line2D:
            handle = Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.5,
                marker=">",
                markersize=6,
                markeredgecolor=color,
                markerfacecolor=color,
                label=label,
            )
            return handle

        legend_handles = [
            _arrow_handle(_BUMP_COLOR_IMPROVED, "Climbed"),
            _arrow_handle(_BUMP_COLOR_WORSENED, "Dropped"),
            _arrow_handle(_BUMP_COLOR_STABLE, "Same"),
        ]
        ax.legend(
            handles=legend_handles,
            loc=legend_loc,
            fontsize=7,
            frameon=True,
            edgecolor="#cccccc",
            framealpha=0.9,
            handlelength=1.5,
            borderpad=0.3,
            labelspacing=0.3,
            ncol=1,
            columnspacing=1.0,
        )

    return True
