# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visual / palette helpers and figure utilities for benchmark plots.

Colour mapping, model-name formatting, legend construction,
watermark overlay, and figure-saving utilities.
"""

from __future__ import annotations

import typing as tp
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

if tp.TYPE_CHECKING:
    import pandas as pd

from neuralbench.plots._constants import (
    CLASSIC_DISPLAY,
    DUMMY_DISPLAY,
    FEATURE_BASED_COLOR,
    FM_DISPLAY,
    METRIC_DISPLAY_NAMES,
    METRIC_HIGHER_IS_BETTER,
    METRIC_PERFECT_SCORE,
    MODEL_PARAMS,
    MODEL_YEAR,
    PRETRAINING_OVERLAP,
)
from neuralbench.plots._models import lookup_by_name

# ---------------------------------------------------------------------------
# Model-name formatting
# ---------------------------------------------------------------------------


def format_params(n: int, precise_below: int = 1_000) -> str:
    """Format a parameter count as a compact human-readable string.

    Parameters
    ----------
    n
        The integer parameter count to format.
    precise_below
        Threshold (in raw integer units, not thousands) below which K-scale
        values render with one decimal of precision (``"N.1K"``) instead of
        rounding to the nearest thousand. The default ``precise_below=1_000``
        reproduces the original plot-legend formatting (every K-scale value
        rounds, so ``1456 -> "1K"``); pass ``precise_below=10_000`` for the
        LaTeX-table formatting which keeps one decimal between 1K and 10K
        (so ``1456 -> "1.5K"``, ``36320 -> "36K"``). Values ``>= 1_000_000``
        always render as ``"N.1M"``.
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= precise_below:
        return f"{n / 1_000:.0f}K"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def with_year(name: str) -> str:
    """Append param count and publication year, e.g. ``'REVE (69.2M; 2025)'``."""
    year = MODEL_YEAR.get(name)
    params = MODEL_PARAMS.get(name)
    parts: list[str] = []
    if params:
        parts.append(format_params(params))
    if year:
        parts.append(str(year))
    return f"{name} ({'; '.join(parts)})" if parts else name


def base_name(name: str) -> str:
    """Strip a trailing parenthesised suffix added by :func:`with_year`."""
    if name.endswith(")") and " (" in name:
        base, _, _ = name.rpartition(" (")
        return base
    return name


# ---------------------------------------------------------------------------
# Color mapping
# ---------------------------------------------------------------------------


def _hex(rgba: tuple) -> str:
    r, g, b = (int(c * 255) for c in rgba[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass(frozen=True)
class ColorGroup:
    """One band of a multi-group categorical palette.

    *members* are the display names that should be assigned colours from
    *cmap* spread linearly between ``lo`` and ``lo + span``.  Empty
    groups produce no entries.
    """

    members: list[str]
    cmap: tp.Any
    lo: float
    span: float


def build_color_dict_from_groups(groups: list[ColorGroup]) -> dict[str, str]:
    """Assign a hex color to every member of every :class:`ColorGroup`.

    Each group spreads its members linearly across its colormap from
    ``lo`` to ``lo + span``; single-member groups land at ``lo``.
    """
    colors: dict[str, str] = {}
    for grp in groups:
        n = len(grp.members)
        if n == 0:
            continue
        for i, name in enumerate(grp.members):
            t = grp.lo + grp.span * i / max(n - 1, 1)
            colors[name] = _hex(grp.cmap(t))
    return colors


def build_color_dict(models: list[str]) -> dict[str, str]:
    """Build a color mapping with distinct palettes per model group.

    Three-band variant used by the EEG benchmark plots: classic / FM /
    dummy.  Thin wrapper over :func:`build_color_dict_from_groups`.
    The Handcrafted baseline is post-overridden to a light green so it
    visually pops out of the otherwise-grey baseline block.
    """
    base_to_full = {base_name(m): m for m in models}
    classic = [base_to_full[m] for m in CLASSIC_DISPLAY if m in base_to_full]
    fm = [base_to_full[m] for m in FM_DISPLAY if m in base_to_full]
    dummy = [base_to_full[m] for m in DUMMY_DISPLAY if m in base_to_full]
    colors = build_color_dict_from_groups(
        [
            ColorGroup(members=classic, cmap=cm.Blues, lo=0.35, span=0.55),  # type: ignore[attr-defined]
            ColorGroup(members=fm, cmap=cm.Oranges, lo=0.35, span=0.55),  # type: ignore[attr-defined]
            ColorGroup(members=dummy, cmap=cm.Greys, lo=0.30, span=0.25),  # type: ignore[attr-defined]
        ]
    )
    for alias in ("Handcrafted", "Feature-based"):
        if alias in base_to_full:
            colors[base_to_full[alias]] = FEATURE_BASED_COLOR
    return colors


def ordered_models(display_models: list[str]) -> list[str]:
    """Return *display_models* sorted into classic / FM / dummy order.

    Models present in *display_models* but absent from all three display
    lists are silently dropped.  A warning is emitted so new models are
    not accidentally invisible in plots.
    """
    base_to_full = {base_name(m): m for m in display_models}
    known = set(CLASSIC_DISPLAY) | set(FM_DISPLAY) | set(DUMMY_DISPLAY)
    unknown = sorted(set(base_to_full) - known)
    if unknown:
        warnings.warn(
            f"Models not in any display list (will be excluded from plots): "
            f"{unknown}. Add them to CLASSIC_DISPLAY, FM_DISPLAY, or "
            f"DUMMY_DISPLAY in _constants.py.",
            stacklevel=2,
        )
    return (
        [base_to_full[m] for m in CLASSIC_DISPLAY if m in base_to_full]
        + [base_to_full[m] for m in FM_DISPLAY if m in base_to_full]
        + [base_to_full[m] for m in DUMMY_DISPLAY if m in base_to_full]
    )


def order_models_with_baselines_first(display_models: list[str]) -> list[str]:
    """Same as :func:`ordered_models`, but pushes baselines to the front.

    The bar charts read left-to-right from cheapest to most expensive, so
    Chance / Dummy / Handcrafted should appear before the classic DL and
    foundation-model bars. Order within the baseline block follows
    :data:`DUMMY_DISPLAY`.
    """
    model_order = ordered_models(display_models)
    baseline_rank = {name: idx for idx, name in enumerate(DUMMY_DISPLAY)}
    baselines = sorted(
        (m for m in model_order if base_name(m) in baseline_rank),
        key=lambda m: baseline_rank[base_name(m)],
    )
    rest = [m for m in model_order if base_name(m) not in baseline_rank]
    return baselines + rest


def darken(hex_color: str, factor: float = 0.65) -> str:
    """Return a darker shade of *hex_color* for use as an edge color."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"


# ---------------------------------------------------------------------------
# Group-label colours (for tick labels on rank plots)
# ---------------------------------------------------------------------------

GROUP_LABEL_COLORS = {
    "classic": cm.Blues(0.8),  # type: ignore[attr-defined]
    "fm": cm.Oranges(0.8),  # type: ignore[attr-defined]
    "dummy": cm.Greys(0.55),  # type: ignore[attr-defined]
}


def group_color(name: str) -> tuple:
    """Return the group-palette color for a display model *name*."""
    entry = lookup_by_name(name)
    if entry is not None:
        if entry.family == "classic":
            return GROUP_LABEL_COLORS["classic"]
        if entry.family == "foundation":
            return GROUP_LABEL_COLORS["fm"]
    return GROUP_LABEL_COLORS["dummy"]


# ---------------------------------------------------------------------------
# Two-tone y-tick labels (rank plots)
# ---------------------------------------------------------------------------


def apply_two_tone_labels(
    fig: plt.Figure,  # type: ignore[name-defined]
    ax: plt.Axes,  # type: ignore[name-defined]
    *,
    meta_color: str = "#bbbbbb",
    fontsize: int = 9,
) -> None:
    """Re-render y-tick labels with model name in group colour and metadata greyed out."""
    for label in ax.get_yticklabels():
        base = base_name(label.get_text())
        if base == label.get_text():
            label.set_color(group_color(base))
        else:
            label.set_color(meta_color)
        label.set_fontsize(fontsize)

    fig.canvas.draw()
    fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]

    for label in ax.get_yticklabels():
        full = label.get_text()
        base = base_name(full)
        if base == full:
            continue

        bb = label.get_window_extent(renderer)
        x_fig, y_fig = fig.transFigure.inverted().transform((bb.x0, (bb.y0 + bb.y1) / 2))
        fig.text(
            x_fig,
            y_fig,
            base,
            fontsize=fontsize,
            color=group_color(base),
            ha="left",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=0),
        )


# ---------------------------------------------------------------------------
# Shared legend builder
# ---------------------------------------------------------------------------


_OVERLAP_SENTINEL = "__overlap__"
_OVERLAP_LABEL = "Seen during\npretraining"

# Sentinel marker prepended to legend group titles so :func:`bold_legend_titles`
# can identify and re-style them after the legend is created.  The marker is
# stripped from the visible label inside the helper.
_LEGEND_TITLE_MARK = "\u200b"  # zero-width space


@dataclass(frozen=True)
class LegendGroup:
    """One column of a grouped legend (bold title + entries).

    *members* must already be in display order.  Empty groups are
    silently dropped by :func:`build_grouped_legend_from_groups`.
    """

    title: str
    members: list[str]


def build_grouped_legend_from_groups(
    color_dict: dict[str, str],
    groups: list[LegendGroup],
    *,
    overlap_group: int | None = None,
    drop_empty: bool = True,
    min_per_col: int = 1,
) -> tuple[list, list[str]]:
    """Generic grouped-legend builder.

    Each group becomes a column that starts with its bold title row,
    then one row per member, blank-padded so all columns share the same
    height.

    When *overlap_group* indexes one of the input groups, a
    ``Seen during pretraining`` entry is appended to that group before
    padding.  Set *drop_empty* to ``False`` to keep empty groups (a bold
    title followed by blanks) in the layout -- the original
    ``build_grouped_legend`` did this so the EEG legend always shows
    three columns.  *min_per_col* ensures the column height stays at
    least this tall when every group is empty (matches the
    ``max(..., 1)`` floor in the legacy implementation).
    """
    blank = Patch(facecolor="none", edgecolor="none")
    overlap_patch = Patch(facecolor="#dddddd", edgecolor="#666666", hatch="///")

    expanded: list[LegendGroup] = []
    for idx, grp in enumerate(groups):
        members = list(grp.members)
        if overlap_group is not None and idx == overlap_group:
            members.append(_OVERLAP_SENTINEL)
        expanded.append(LegendGroup(title=grp.title, members=members))

    laid_out = [g for g in expanded if g.members] if drop_empty else expanded
    if not laid_out:
        return [], []

    n_per_col = max((len(g.members) for g in laid_out), default=0)
    n_per_col = max(n_per_col, min_per_col) + 1

    handles: list[Patch] = []
    labels: list[str] = []
    for grp in laid_out:
        handles.append(blank)
        # Title rendered as plain text (no mathtext) so it shares the
        # same font family as the rest of the legend.  The leading
        # zero-width sentinel lets :func:`bold_legend_titles` find these
        # rows and bold them after the legend is created.
        labels.append(_LEGEND_TITLE_MARK + grp.title)
        for name in grp.members:
            if name == _OVERLAP_SENTINEL:
                handles.append(overlap_patch)
                labels.append(_OVERLAP_LABEL)
            else:
                handles.append(Patch(facecolor=color_dict[name], edgecolor="none"))
                labels.append(base_name(name))
        for _ in range(n_per_col - len(grp.members) - 1):
            handles.append(blank)
            labels.append("")

    return handles, labels


def bold_legend_titles(legend: tp.Any) -> None:
    """Bold every text row in *legend* whose label is a group title.

    Group titles are produced by :func:`build_grouped_legend_from_groups`
    with a leading zero-width sentinel so we can identify them here
    without forcing callers to track indices.  The sentinel is stripped
    from the visible label as a side effect.
    """
    for text in legend.get_texts():
        s = text.get_text()
        if s.startswith(_LEGEND_TITLE_MARK):
            text.set_text(s[len(_LEGEND_TITLE_MARK) :])
            text.set_fontweight("bold")


def build_grouped_legend(
    color_dict: dict[str, str],
    display_models: list[str],
    *,
    include_overlap: bool = False,
) -> tuple[list, list[str]]:
    """Build legend handles/labels grouped into classic / FM / dummy columns.

    Thin wrapper over :func:`build_grouped_legend_from_groups` that
    keeps even empty groups so the legend always renders three columns
    (matches pre-refactor behaviour).  When *include_overlap* is True
    an extra "Seen during pretraining" entry is appended to the dummy
    column.
    """
    base_to_full = {base_name(m): m for m in display_models}
    groups = [
        LegendGroup(
            title="Baselines",
            members=[base_to_full[m] for m in DUMMY_DISPLAY if m in base_to_full],
        ),
        LegendGroup(
            title="Task-specific models",
            members=[base_to_full[m] for m in CLASSIC_DISPLAY if m in base_to_full],
        ),
        LegendGroup(
            title="Foundation models",
            members=[base_to_full[m] for m in FM_DISPLAY if m in base_to_full],
        ),
    ]
    return build_grouped_legend_from_groups(
        color_dict,
        groups,
        overlap_group=0 if include_overlap else None,
        drop_empty=False,
    )


def build_grouped_legend_two_col(
    color_dict: dict[str, str],
    display_models: list[str],
    *,
    include_overlap: bool = False,
) -> tuple[list, list[str]]:
    """Two-column legend: Baselines + Task-specific share one column, Foundation the other.

    Designed for inline placement in a near-square empty region of the
    chart (e.g. the trailing cells of the core bar chart).  Column 1
    stacks the trained-from-scratch groups (Baselines on top,
    Task-specific below, separated by a blank row); column 2 carries
    the pretrained Foundation models.  Both columns are padded to the
    same height so matplotlib ``ncol=2`` distributes them cleanly, one
    column per group bundle.
    """
    base_to_full = {base_name(m): m for m in display_models}
    baselines = [base_to_full[m] for m in DUMMY_DISPLAY if m in base_to_full]
    classic = [base_to_full[m] for m in CLASSIC_DISPLAY if m in base_to_full]
    fm = [base_to_full[m] for m in FM_DISPLAY if m in base_to_full]

    blank = Patch(facecolor="none", edgecolor="none")
    overlap_patch = Patch(facecolor="#dddddd", edgecolor="#666666", hatch="///")

    def _patch(name: str) -> Patch:
        return Patch(facecolor=color_dict[name], edgecolor="none")

    # Column 1: Baselines block + blank separator + Task-specific block.
    col1_handles: list[Patch] = [blank]
    col1_labels: list[str] = [_LEGEND_TITLE_MARK + "Baselines"]
    for name in baselines:
        col1_handles.append(_patch(name))
        col1_labels.append(base_name(name))
    if include_overlap:
        col1_handles.append(overlap_patch)
        col1_labels.append(_OVERLAP_LABEL)
    col1_handles.append(blank)
    col1_labels.append("")
    col1_handles.append(blank)
    col1_labels.append(_LEGEND_TITLE_MARK + "Task-specific models")
    for name in classic:
        col1_handles.append(_patch(name))
        col1_labels.append(base_name(name))

    # Column 2: Foundation block.
    col2_handles: list[Patch] = [blank]
    col2_labels: list[str] = [_LEGEND_TITLE_MARK + "Foundation models"]
    for name in fm:
        col2_handles.append(_patch(name))
        col2_labels.append(base_name(name))

    n_per_col = max(len(col1_handles), len(col2_handles))
    while len(col1_handles) < n_per_col:
        col1_handles.append(blank)
        col1_labels.append("")
    while len(col2_handles) < n_per_col:
        col2_handles.append(blank)
        col2_labels.append("")

    return col1_handles + col2_handles, col1_labels + col2_labels


def build_grouped_legend_split_classic(
    color_dict: dict[str, str],
    display_models: list[str],
    *,
    include_overlap: bool = False,
) -> tuple[list, list[str]]:
    """Variant of :func:`build_grouped_legend` that splits ``Task-specific models``
    in half across two columns so the per-column height shrinks (handy
    for inline placement in tight cells).  The second classic column
    reuses the bold ``Task-specific models`` title with a leading
    ``(cont.)`` so the grouping is obvious.
    """
    base_to_full = {base_name(m): m for m in display_models}
    classic = [base_to_full[m] for m in CLASSIC_DISPLAY if m in base_to_full]
    half = (len(classic) + 1) // 2
    classic_a, classic_b = classic[:half], classic[half:]
    groups = [
        LegendGroup(title="Task-specific models", members=classic_a),
        LegendGroup(title="Task-specific models (cont.)", members=classic_b),
        LegendGroup(
            title="Foundation models",
            members=[base_to_full[m] for m in FM_DISPLAY if m in base_to_full],
        ),
        LegendGroup(
            title="Baselines",
            members=[base_to_full[m] for m in DUMMY_DISPLAY if m in base_to_full],
        ),
    ]
    return build_grouped_legend_from_groups(
        color_dict,
        groups,
        overlap_group=3 if include_overlap else None,
        drop_empty=False,
    )


def build_grouped_legend_stacked(
    color_dict: dict[str, str],
    display_models: list[str],
    *,
    include_overlap: bool = False,
) -> tuple[list, list[str]]:
    """Build legend handles/labels with all groups stacked in one column.

    Same group set as :func:`build_grouped_legend` (Foundation /
    Task-specific / Baselines) but laid out vertically: each group
    renders as a bold title followed by its members, with a single
    blank row separating consecutive groups.  Intended for ``ncol=1``
    legends placed alongside (rather than below) the axes.
    """
    base_to_full = {base_name(m): m for m in display_models}
    groups = [
        LegendGroup(
            title="Foundation",
            members=[base_to_full[m] for m in FM_DISPLAY if m in base_to_full],
        ),
        LegendGroup(
            title="Task-specific",
            members=[base_to_full[m] for m in CLASSIC_DISPLAY if m in base_to_full],
        ),
        LegendGroup(
            title="Baselines",
            members=[base_to_full[m] for m in DUMMY_DISPLAY if m in base_to_full],
        ),
    ]
    if include_overlap:
        groups[-1] = LegendGroup(
            title=groups[-1].title,
            members=[*groups[-1].members, _OVERLAP_SENTINEL],
        )

    blank = Patch(facecolor="none", edgecolor="none")
    overlap_patch = Patch(facecolor="#dddddd", edgecolor="#666666", hatch="///")

    handles: list[Patch] = []
    labels: list[str] = []
    for idx, grp in enumerate(groups):
        if not grp.members:
            continue
        if handles:
            handles.append(blank)
            labels.append("")
        handles.append(blank)
        labels.append(_LEGEND_TITLE_MARK + grp.title)
        for name in grp.members:
            if name == _OVERLAP_SENTINEL:
                handles.append(overlap_patch)
                labels.append(_OVERLAP_LABEL)
            else:
                handles.append(Patch(facecolor=color_dict[name], edgecolor="none"))
                labels.append(base_name(name))

    return handles, labels


# ---------------------------------------------------------------------------
# Rendering a single bar-chart subplot
# ---------------------------------------------------------------------------


SplitKind = tp.Literal["cross_subject", "within_subject", "random"]

_SPLIT_KIND_LABELS: dict[str, str] = {
    "cross_subject": "cross-subject",
    "within_subject": "within-subject",
    "random": "random",
}


def _humanize_count(n: int) -> str:
    """Compact human-readable integer (``6.7K``, ``171K``, ``1.3M``)."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        # Drop the decimal at >=10K to avoid "6.7K" vs "167.0K" asymmetry.
        if n >= 10_000:
            return f"{n / 1_000:.0f}K"
        return f"{n / 1_000:.1f}K"
    return str(n)


def _format_info_badge(
    split_kind: SplitKind | None,
    n_examples: int | None,
    *,
    stack: bool = False,
) -> str | None:
    """Return the subplot info-badge string, or ``None`` if nothing to show.

    With *stack=True* the parts are stacked on two lines (split-kind on
    top, sample count below).  This keeps the badge narrow enough to
    fit under a sub-1" panel; the default joins them on one line for
    wider layouts.
    """
    parts: list[str] = []
    if split_kind is not None:
        parts.append(_SPLIT_KIND_LABELS.get(split_kind, split_kind))
    if n_examples is not None:
        parts.append(f"n={_humanize_count(int(n_examples))}")
    if not parts:
        return None
    if stack:
        return "\n".join(parts)
    return "  \u00b7  ".join(parts)


def _draw_bars(
    ax: plt.Axes,  # type: ignore[name-defined]
    agg: "pd.DataFrame",
    model_order: list[str],
    color_dict: dict[str, str],
    dataset_name: str,
    *,
    bar_width: float = 1.0,
    show_error_bars: bool = True,
    clip_on: bool = False,
    error_capsize: float = 1.5,
    error_linewidth: float = 0.8,
) -> None:
    """Draw bar + error-bar artists only.  No styling.

    Shared between :func:`render_bars_on_axis` (one ax, ``clip_on=False``
    so error bars can punch through the spine) and the broken-axis path
    (two stacked axes, ``clip_on=True`` so each axis only shows its own
    portion of the bars).

    *error_capsize* / *error_linewidth* control the SEM whisker
    geometry; bump them up for dense letter-format charts where the
    default thin caps become unreadable.
    """
    for i, model in enumerate(model_order):
        if model not in agg.index or np.isnan(agg.loc[model, "mean"]):  # type: ignore[arg-type]
            continue
        se = agg.loc[model, "sem"]
        bname = base_name(model)
        is_overlap = dataset_name in PRETRAINING_OVERLAP.get(bname, set())
        bar_color = color_dict.get(model, "#888888")
        yerr = (se if not np.isnan(se) else 0) if show_error_bars else 0  # type: ignore[arg-type]
        container = ax.bar(
            i,
            agg.loc[model, "mean"],  # type: ignore[arg-type]
            bar_width,
            yerr=yerr,
            color=bar_color,
            edgecolor=darken(bar_color) if is_overlap else bar_color,
            hatch="///" if is_overlap else "",
            linewidth=0.6 if is_overlap else 0.4,
            capsize=error_capsize if show_error_bars else 0,
            error_kw={"color": "k", "linewidth": error_linewidth},
        )
        for patch in container.patches:
            patch.set_clip_on(clip_on)
        if container.errorbar is not None:
            _, caplines, barlinecols = container.errorbar.lines
            for line in caplines:
                line.set_clip_on(clip_on)
            for col in barlinecols:
                col.set_clip_on(clip_on)


def should_break_axis(
    agg: "pd.DataFrame",
    *,
    threshold: float = 0.5,
) -> bool:
    """Heuristic: break the y-axis when the bar range sits in the upper half.

    Returns ``True`` when the smallest non-NaN bar mean is at least
    *threshold* of the largest (default ``0.5``).  Both extrema must be
    strictly positive -- the heuristic is undefined on negative bars or
    a single bar.
    """
    if "mean" not in agg.columns:
        return False
    means = agg["mean"].dropna()
    if len(means) < 2:
        return False
    mn = float(means.min())
    mx = float(means.max())
    if mx <= 0 or mn <= 0:
        return False
    return (mn / mx) >= threshold


def _clamp_axis_top_to_metric_ceiling(ymax: float, metric_key: str) -> float:
    """Clamp ``ymax`` to the metric's natural perfect-score ceiling.

    For metrics where higher-is-better and a finite perfect score exists
    (e.g. ``test/bal_acc`` -> 100, ``test/pearsonr`` -> 1.0), bars cannot
    legitimately exceed that ceiling; matplotlib's auto-padding can
    nevertheless push ``ymax`` slightly above it (e.g. 101.55 for a bar
    at 99) which then bleeds into the in-panel "top value" annotation
    -- e.g. SSVEP's ``Wang2017Benchmark`` showing ``101`` despite
    balanced accuracy capping at 100.  Clamping ``ymax`` here keeps both
    the y-axis upper bound and the displayed annotation honest.

    Returns ``ymax`` unchanged for lower-is-better metrics (e.g. RMSE)
    or metrics without a registered perfect score.
    """
    perfect = METRIC_PERFECT_SCORE.get(metric_key)
    if perfect is None:
        return ymax
    if not METRIC_HIGHER_IS_BETTER.get(metric_key, True):
        return ymax
    return min(ymax, float(perfect))


def _nice_break_low(min_bar: float) -> float:
    """Pick a 'nice' lower bound for the upper panel of a broken axis.

    Snaps to a step that scales with magnitude (10 for >=50, 5 for >=10,
    1 for >=1, else 0.1) and keeps at least one full step of headroom
    below the smallest bar so it does not visually touch the break.
    """
    if min_bar >= 50:
        step = 10.0
    elif min_bar >= 10:
        step = 5.0
    elif min_bar >= 1:
        step = 1.0
    else:
        step = 0.1
    low = (int(min_bar / step)) * step
    if low > min_bar - step:
        low -= step
    return max(0.0, low)


def render_bars_on_axis(
    ax: plt.Axes,  # type: ignore[name-defined]
    agg: "pd.DataFrame",
    model_order: list[str],
    color_dict: dict[str, str],
    dataset_name: str,
    metric_key: str,
    n_models: int,
    *,
    title: str,
    bar_width: float = 1.0,
    title_fontsize: float = 14,
    title_pad: float = 16,
    title_weight: str = "normal",
    tick_labelsize: float = 12,
    split_kind: SplitKind | None = None,
    n_examples: int | None = None,
    show_metric_label: bool = True,
    show_ytick_labels: bool = True,
    show_error_bars: bool = True,
    error_capsize: float = 1.5,
    error_linewidth: float = 0.8,
    stack_badge: bool = False,
) -> None:
    """Draw bars + error bars for a single subplot and style the axis.

    The ``dataset_name`` argument must be the ``neuralfetch`` study class name
    used as the core dataset for the subplot (e.g. ``"Lopez2017Tuab"``).
    It is matched against :data:`PRETRAINING_OVERLAP` to decide whether each
    bar should be hatched as "seen during pretraining".

    When *split_kind* and/or *n_examples* are provided, a small info badge
    like ``cross-subject  ·  n=700K`` is drawn between the metric subtitle
    and the title.  Passing ``None`` for both preserves the pre-badge
    layout exactly.

    Set *show_metric_label* to ``False`` to suppress the metric subtitle
    below the title (useful for compact/letter-format figures).

    Set *show_ytick_labels* to ``False`` to hide y-axis tick labels while
    keeping the spine and adding a subtle reference gridline at y=50.

    Set *show_error_bars* to ``False`` to draw bars without error bars
    (useful at very small subplot sizes where error bars add clutter).
    """
    _draw_bars(
        ax,
        agg,
        model_order,
        color_dict,
        dataset_name,
        bar_width=bar_width,
        show_error_bars=show_error_bars,
        clip_on=False,
        error_capsize=error_capsize,
        error_linewidth=error_linewidth,
    )

    metric_label = METRIC_DISPLAY_NAMES.get(metric_key, metric_key)

    badge_text = _format_info_badge(split_kind, n_examples, stack=stack_badge)
    metric_fontsize = min(title_fontsize - 3, 11)
    badge_fontsize = max(metric_fontsize - 2, 8)

    ax.set_title(title, fontsize=title_fontsize, pad=title_pad, fontweight=title_weight)
    if show_metric_label:
        ax.text(
            0.5,
            1.005,
            metric_label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=metric_fontsize,
            color="#888888",
        )
    ax.set_xlim(-0.5, n_models - 0.5)
    ax.set_xticks([])
    if metric_key == "test/pearsonr":
        ax.set_ylim(bottom=0)
    ax.set_ylabel("")
    # Cap the y-axis at the metric's perfect-score ceiling (e.g. 100 for
    # balanced accuracy, 1.0 for Pearson R) so neither the spine nor the
    # in-panel "top value" annotation drifts into a value that bars
    # cannot legitimately reach -- matplotlib's auto-padding can push
    # the spine to e.g. 101.55 even when the tallest bar is 99.
    _ymin0, _ymax0 = ax.get_ylim()
    _ymax_clamped = _clamp_axis_top_to_metric_ceiling(_ymax0, metric_key)
    if _ymax_clamped < _ymax0:
        ax.set_ylim(top=_ymax_clamped)
    if show_ytick_labels:
        ax.tick_params(axis="y", labelsize=tick_labelsize)
    else:
        # Hide tick *labels* entirely (they collide with the metric
        # subtitle at the top and the badge at the bottom in dense
        # letter-format layouts) but keep small marks at the bottom
        # / top so the spine still reads as an actual axis.
        ymin, ymax = ax.get_ylim()
        top = int(ymax)
        ax.set_yticks([0, top])
        ax.set_yticklabels(["", ""], fontsize=tick_labelsize)
        ax.tick_params(axis="y", length=2, pad=1)
        # Annotate the top value as a small grey number tucked inside
        # the upper-LEFT corner of the panel so the reader still has
        # an absolute scale reference without crowding the subtitle.
        ax.text(
            0.08,
            0.98,
            str(top),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=max(tick_labelsize - 1, 4),
            color="#888888",
            zorder=5,
            bbox={
                "boxstyle": "square,pad=0.1",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.7,
            },
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    # Disable the implicit grid lines that would otherwise draw a faint
    # horizontal stroke at every y-tick (``axes.grid`` is flipped to True
    # transitively by ``moabb``).
    ax.grid(False)
    # Split-type + example count as a small, unobtrusive line below the
    # x-axis.  Using ``set_xlabel`` (instead of a free ``ax.text``) plays
    # well with ``tight_layout`` so the badge never collides with the
    # legend or the next row.
    if badge_text is not None:
        # Drawn as ``ax.text`` (rather than ``ax.set_xlabel``) so we can
        # reliably force the badge above bars / error bars that render
        # outside the axes box (clip_on=False) on tasks like
        # psychopathology where Pearson R error bars cross zero.  A
        # semi-transparent white pad keeps the text legible on top of
        # anything that happens to punch through the baseline.
        ax.annotate(
            badge_text,
            xy=(0.5, 0),
            xycoords=ax.transAxes,
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=badge_fontsize,
            color="#888888",
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )


def render_bars_on_broken_axis(
    fig: plt.Figure,  # type: ignore[name-defined]
    parent_ax: plt.Axes,  # type: ignore[name-defined]
    agg: "pd.DataFrame",
    model_order: list[str],
    color_dict: dict[str, str],
    dataset_name: str,
    metric_key: str,
    n_models: int,
    *,
    title: str,
    bar_width: float = 1.0,
    title_fontsize: float = 14,
    title_pad: float = 16,
    title_weight: str = "normal",
    tick_labelsize: float = 12,
    split_kind: SplitKind | None = None,
    n_examples: int | None = None,
    show_metric_label: bool = True,
    show_ytick_labels: bool = True,
    show_error_bars: bool = True,
    error_capsize: float = 1.5,
    error_linewidth: float = 0.8,
    stack_badge: bool = False,
    # Bottom panel and inter-panel gap are intentionally compact: the
    # bottom panel only exists to anchor the bars at zero, so it gets
    # ~10% of the parent slot's height (down ~30% from the original
    # ``(6, 1)`` ratio) and an equally tightened ``hspace``.
    height_ratios: tuple[float, float] = (6.0, 0.7),
    hspace: float = 0.056,
) -> tuple[plt.Axes, plt.Axes]:  # type: ignore[name-defined]
    """Render a broken-y-axis subplot inside *parent_ax*'s gridspec slot.

    The parent axis is hidden and replaced by two stacked axes (top and
    bottom) sharing the same horizontal extent.  The top axis covers the
    range ``[break_low, ymax]`` (where the bars actually are); the bottom
    axis covers a small sliver near zero so the reader still sees that
    bars originate from y=0.  Diagonal slashes mark the break.

    Returns ``(ax_top, ax_bot)`` so callers can attach extra annotations
    (e.g. left-margin labels) if needed.
    """
    parent_ax.set_visible(False)
    parent_subplotspec = parent_ax.get_subplotspec()
    assert parent_subplotspec is not None
    gs = parent_subplotspec.subgridspec(
        2, 1, height_ratios=list(height_ratios), hspace=hspace
    )
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1])

    _draw_bars(
        ax_top,
        agg,
        model_order,
        color_dict,
        dataset_name,
        bar_width=bar_width,
        show_error_bars=show_error_bars,
        clip_on=True,
        error_capsize=error_capsize,
        error_linewidth=error_linewidth,
    )
    _draw_bars(
        ax_bot,
        agg,
        model_order,
        color_dict,
        dataset_name,
        bar_width=bar_width,
        show_error_bars=show_error_bars,
        clip_on=True,
        error_capsize=error_capsize,
        error_linewidth=error_linewidth,
    )

    means = agg["mean"].dropna()
    min_bar = float(means.min())
    break_low = _nice_break_low(min_bar)

    # Use the auto-computed top (tallest bar + error-bar headroom).
    top_ymax = float(ax_top.get_ylim()[1])
    if top_ymax <= break_low:
        top_ymax = float(means.max()) * 1.05
    # Clamp to the metric's perfect-score ceiling so the broken-axis
    # spine and the in-panel top-value annotation never claim a value
    # bars cannot legitimately reach (e.g. ``101`` for a balanced
    # accuracy panel).
    top_ymax = _clamp_axis_top_to_metric_ceiling(top_ymax, metric_key)

    # Keep visual y-units consistent across the break by scaling the
    # bottom panel's range to match its height ratio.
    h_top, h_bot = float(height_ratios[0]), float(height_ratios[1])
    top_range = top_ymax - break_low
    bot_range = top_range * (h_bot / h_top) if h_top > 0 else top_range * 0.15

    ax_top.set_ylim(break_low, top_ymax)
    ax_bot.set_ylim(0, bot_range)

    ax_top.set_xlim(-0.5, n_models - 0.5)
    ax_bot.set_xlim(-0.5, n_models - 0.5)
    ax_top.set_xticks([])
    ax_bot.set_xticks([])

    # Spines: hide the touching pair so the break looks intentional.
    # Explicitly re-enable the left + bottom spines because some transitive
    # imports (e.g. ``moabb``) flip ``axes.spines.{left,bottom}`` rcParams
    # to False, which would otherwise leave the broken-axis subplots with
    # no visible y-axis line at all.
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["left"].set_visible(True)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.spines["left"].set_visible(True)
    ax_bot.spines["bottom"].set_visible(True)
    # Disable the implicit grid lines that would otherwise draw a faint
    # horizontal stroke at every y-tick (``axes.grid`` is flipped to True
    # transitively by ``moabb``); the broken-axis chart relies on the
    # spines + diagonal slashes to convey the axis structure.
    ax_top.grid(False)
    ax_bot.grid(False)

    ax_top.set_ylabel("")
    ax_bot.set_ylabel("")
    if show_ytick_labels:
        ax_top.tick_params(axis="y", labelsize=tick_labelsize)
        ax_bot.tick_params(axis="y", labelsize=tick_labelsize)
        ax_bot.set_yticks([0])
    else:
        # Hide the labels but keep tiny tick marks; show the upper
        # bound as a small grey number tucked inside the upper-LEFT
        # corner of the top panel so the reader still has an absolute
        # scale reference without colliding with the metric subtitle.
        top_int = int(top_ymax)
        ax_top.set_yticks([top_int])
        ax_top.set_yticklabels([""], fontsize=tick_labelsize)
        ax_top.tick_params(axis="y", length=2, pad=1)
        ax_bot.set_yticks([0])
        ax_bot.set_yticklabels([""], fontsize=tick_labelsize)
        ax_bot.tick_params(axis="y", length=2, pad=1)
        _value_bbox = {
            "boxstyle": "square,pad=0.1",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.7,
        }
        ax_top.text(
            0.08,
            0.98,
            str(top_int),
            transform=ax_top.transAxes,
            ha="left",
            va="top",
            fontsize=max(tick_labelsize - 1, 4),
            color="#888888",
            zorder=5,
            bbox=_value_bbox,
        )
        # Mirror the top-value annotation in the lower-LEFT corner of the
        # top panel to indicate the value at which bars restart after the
        # axis break -- without this, readers can see the panel ceiling
        # but not the floor, which makes broken-axis bar heights hard to
        # interpret.
        ax_top.text(
            0.08,
            0.02,
            str(int(break_low)),
            transform=ax_top.transAxes,
            ha="left",
            va="bottom",
            fontsize=max(tick_labelsize - 1, 4),
            color="#888888",
            zorder=5,
            bbox=_value_bbox,
        )

    # Diagonal slashes at the break.  Sized to scale with tick fontsize
    # so they remain visible on the dense letter-format chart.
    d = 0.015 * max(tick_labelsize / 12.0, 0.5)
    kw: dict[str, tp.Any] = dict(color="k", clip_on=False, lw=0.8)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kw)
    # Slashes on ax_bot scaled by the height ratio so they appear the
    # same physical size as on ax_top.
    d_bot = d * (h_top / h_bot) if h_bot > 0 else d
    ax_bot.plot((-d, +d), (1 - d_bot, 1 + d_bot), transform=ax_bot.transAxes, **kw)
    ax_bot.plot((1 - d, 1 + d), (1 - d_bot, 1 + d_bot), transform=ax_bot.transAxes, **kw)

    # Title + metric label go on the top axis.
    metric_label = METRIC_DISPLAY_NAMES.get(metric_key, metric_key)
    metric_fontsize = min(title_fontsize - 3, 11)
    badge_fontsize = max(metric_fontsize - 2, 8)
    ax_top.set_title(
        title, fontsize=title_fontsize, pad=title_pad, fontweight=title_weight
    )
    if show_metric_label:
        ax_top.text(
            0.5,
            1.005,
            metric_label,
            transform=ax_top.transAxes,
            ha="center",
            va="bottom",
            fontsize=metric_fontsize,
            color="#888888",
        )

    # Badge goes under the bottom axis.
    badge_text = _format_info_badge(split_kind, n_examples, stack=stack_badge)
    if badge_text is not None:
        ax_bot.annotate(
            badge_text,
            xy=(0.5, 0),
            xycoords=ax_bot.transAxes,
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=badge_fontsize,
            color="#888888",
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )

    return ax_top, ax_bot


# ---------------------------------------------------------------------------
# Marginal mean-rank subplot (per-row summary)
# ---------------------------------------------------------------------------


def render_marginal_rank_bars(
    ax: plt.Axes,  # type: ignore[name-defined]
    rank_stats: "pd.DataFrame",
    model_order: list[str],
    color_dict: dict[str, str],
    n_models: int,
    *,
    title: str = "Mean norm.\nrank \u2191",
    subtitle: str | None = None,
    bar_width: float = 1.0,
    title_fontsize: float = 14,
    title_pad: float = 16,
    title_weight: str = "bold",
    tick_labelsize: float = 12,
    show_ytick_labels: bool = True,
    show_error_bars: bool = True,
) -> None:
    """Draw a marginal "mean rank" bar chart for one row of the grid.

    *rank_stats* carries already-normalised per-model ranks in ``[0, 1]``
    (best = ``1/N`` near the top, worst = ``1`` at the bottom).  The
    y-axis is inverted so the *best* models reach the highest into the
    panel; visually a *taller* bar still means a *better* model -- the
    upward arrow in the default title makes that direction explicit.
    """
    ymax = 1.0
    ymin = 0.0
    has_sem = "sem" in rank_stats.columns
    for i, model in enumerate(model_order):
        if model not in rank_stats.index:
            continue
        r = rank_stats["mean"].get(model)
        if r is None or np.isnan(r):  # type: ignore[arg-type]
            continue
        bar_color = color_dict.get(model, "#888888")
        # Bar grows from the worst-rank baseline at the bottom (y=ymax)
        # up to the model's mean normalised rank.  ``bottom=ymax`` plus a
        # negative height encodes that on the standard (non-inverted)
        # data axis; ``set_ylim(ymax, ymin)`` then flips it so the bar
        # visually extends *upward* on the rendered figure.
        ax.bar(
            i,
            float(r) - ymax,
            bar_width,
            bottom=ymax,
            color=bar_color,
            edgecolor=bar_color,
            linewidth=0.4,
        )
        if show_error_bars and has_sem:
            se = rank_stats["sem"].get(model)
            if se is not None and not np.isnan(se):  # type: ignore[arg-type]
                # ``ax.errorbar`` with no marker draws just the cap +
                # whiskers centred on the bar tip.
                ax.errorbar(
                    i,
                    float(r),
                    yerr=float(se),
                    fmt="none",
                    ecolor="k",
                    elinewidth=0.8,
                    capsize=1.5,
                    capthick=0.8,
                    zorder=3,
                )

    ax.set_xlim(-0.5, n_models - 0.5)
    ax.set_xticks([])
    # Inverted so 0 (best, theoretical) is at the top and 1 (worst) at
    # the bottom -- matches the visual "taller is better" intuition.
    ax.set_ylim(ymax, ymin)
    ax.set_ylabel("")

    # Best/Worst tick labels: stacked on two lines ("Best\n(0)" /
    # "Worst\n(1)") so the descriptor and its rank value read as a
    # tight unit and fit comfortably outside the narrow marginal
    # column.
    ytick_fs = max(tick_labelsize - 2, 4)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["Best\n(0)", "Worst\n(1)"], fontsize=ytick_fs)
    if show_ytick_labels:
        ax.tick_params(axis="y", labelsize=ytick_fs)
    else:
        ax.tick_params(axis="y", length=2, pad=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Top-align the marginal title with the per-task titles on the
    # same row.  ``ax.set_title(title, pad=title_pad)`` would place
    # the title's *baseline* at ``pad`` points above the axis spine;
    # because this title is two lines tall ("Mean norm.\nrank \u2191"),
    # set_title-with-pad puts its visual TOP much higher than the
    # single-line task titles.  We instead anchor the title's TOP at
    # the same height as a single-line task title's TOP by computing
    # ``pad + title_fontsize`` and using ``va="top"``.
    n_title_lines = title.count("\n") + 1
    title_height_pt = n_title_lines * title_fontsize * 1.2
    title_top_offset_pt = title_pad + title_fontsize
    ax.annotate(
        title,
        xy=(0.5, 1.0),
        xycoords="axes fraction",
        xytext=(0, title_top_offset_pt),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=title_fontsize,
        fontweight=title_weight,
        annotation_clip=False,
    )
    if subtitle:
        # Subtitle sits JUST ABOVE the title as a small italic grey
        # caption -- e.g. "over category" / "over datasets" -- naming
        # what the marginal aggregates over.
        subtitle_fs = max(min(title_fontsize - 2, 11), 5)
        ax.annotate(
            subtitle,
            xy=(0.5, 1.0),
            xycoords="axes fraction",
            xytext=(0, title_top_offset_pt + 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=subtitle_fs,
            color="#888888",
            fontstyle="italic",
            annotation_clip=False,
        )
    # Suppress placeholder for the unused ``title_height_pt`` -- kept
    # in scope so future tweaks (e.g., adding a third caption line)
    # have the multi-line height measurement readily available.
    _ = title_height_pt


# ---------------------------------------------------------------------------
# Subplot-grid helpers (shared by bar-chart functions)
# ---------------------------------------------------------------------------


@dataclass
class SubplotSpec:
    """Per-subplot payload consumed by :func:`render_bars_grid`.

    ``dataset_name`` is the ``neuralfetc`` study class name for the subplot's
    core dataset (e.g. ``"Lopez2017Tuab"``); it drives the pretraining-
    overlap hatching via :data:`PRETRAINING_OVERLAP`.

    ``split_kind`` and ``n_examples`` feed the optional info badge drawn
    above the axes (``cross-subject  ·  n=700K``).  hBoth default to
    ``None`` so callers that don't supply this metadata get the exact
    pre-badge layout.
    """

    title: str
    agg: "pd.DataFrame"
    dataset_name: str
    metric_key: str
    title_weight: str = "normal"
    split_kind: SplitKind | None = None
    n_examples: int | None = None


@dataclass
class MarginalRankSpec:
    """Marginal mean-rank bar subplot consumed by :func:`render_bars_grid`.

    ``rank_stats`` is a DataFrame indexed by the *display* model name
    with at least a ``mean`` column (the per-row mean rank, with 1 =
    best).  An optional ``sem`` column drives the error caps on the
    bars.

    ``subtitle`` adds a small italic grey line below the bold title --
    use it to spell out which tasks/datasets the marginal aggregates
    over (e.g. ``"over Cognitive decoding"``).
    """

    rank_stats: "pd.DataFrame"
    # Two-line title keeps the heading inside the narrow marginal
    # column.  ``render_marginal_rank_bars`` uses ``annotate`` (not
    # ``set_title``) to place title + subtitle so the per-line offsets
    # are consistent across rows.
    title: str = "Mean norm.\nrank \u2191"
    title_weight: str = "bold"
    subtitle: str | None = None


GridSpec = SubplotSpec | MarginalRankSpec | None


def render_bars_grid(
    axes: np.ndarray,
    grid: list[list[GridSpec]],
    model_order: list[str],
    color_dict: dict[str, str],
    *,
    title_fontsize: float = 14,
    tick_labelsize: float = 12,
    title_pad: float = 16,
    show_badge: bool = True,
    show_metric_label: bool = True,
    show_ytick_labels: bool = True,
    show_error_bars: bool = True,
    error_capsize: float = 1.5,
    error_linewidth: float = 0.8,
    stack_badge: bool = False,
    enable_axis_break: bool = True,
    break_threshold: float = 0.5,
) -> None:
    """Render a 2D grid of bar-chart subplots.

    *grid* is a rectangular ``list[list[GridSpec]]`` matching the shape of
    *axes*.  ``None`` cells are made invisible; :class:`SubplotSpec` cells
    are rendered via :func:`render_bars_on_axis` (or
    :func:`render_bars_on_broken_axis` when *enable_axis_break* is set
    and the bar range sits in the upper *break_threshold* fraction);
    :class:`MarginalRankSpec` cells are rendered via
    :func:`render_marginal_rank_bars`.

    *title_pad*, *show_badge*, *show_metric_label*, *show_ytick_labels*,
    and *show_error_bars* are forwarded to :func:`render_bars_on_axis`.
    """
    n_models = len(model_order)
    fig = axes.flat[0].figure
    for row_idx, row in enumerate(grid):
        for col_idx, spec in enumerate(row):
            ax = axes[row_idx, col_idx]
            if spec is None:
                ax.set_visible(False)
                continue
            if isinstance(spec, MarginalRankSpec):
                render_marginal_rank_bars(
                    ax,
                    spec.rank_stats,
                    model_order,
                    color_dict,
                    n_models,
                    title=spec.title,
                    subtitle=spec.subtitle,
                    title_fontsize=title_fontsize,
                    title_pad=title_pad,
                    title_weight=spec.title_weight,
                    tick_labelsize=tick_labelsize,
                    show_ytick_labels=show_ytick_labels,
                    show_error_bars=show_error_bars,
                )
                continue
            if enable_axis_break and should_break_axis(
                spec.agg, threshold=break_threshold
            ):
                render_bars_on_broken_axis(
                    fig,
                    ax,
                    spec.agg,
                    model_order,
                    color_dict,
                    spec.dataset_name,
                    spec.metric_key,
                    n_models,
                    title=spec.title,
                    title_fontsize=title_fontsize,
                    title_pad=title_pad,
                    title_weight=spec.title_weight,
                    tick_labelsize=tick_labelsize,
                    split_kind=spec.split_kind if show_badge else None,
                    n_examples=spec.n_examples if show_badge else None,
                    show_metric_label=show_metric_label,
                    show_ytick_labels=show_ytick_labels,
                    show_error_bars=show_error_bars,
                    error_capsize=error_capsize,
                    error_linewidth=error_linewidth,
                    stack_badge=stack_badge,
                )
                continue
            render_bars_on_axis(
                ax,
                spec.agg,
                model_order,
                color_dict,
                spec.dataset_name,
                spec.metric_key,
                n_models,
                title=spec.title,
                title_fontsize=title_fontsize,
                title_pad=title_pad,
                title_weight=spec.title_weight,
                tick_labelsize=tick_labelsize,
                split_kind=spec.split_kind if show_badge else None,
                n_examples=spec.n_examples if show_badge else None,
                show_metric_label=show_metric_label,
                show_ytick_labels=show_ytick_labels,
                show_error_bars=show_error_bars,
                error_capsize=error_capsize,
                error_linewidth=error_linewidth,
                stack_badge=stack_badge,
            )


def finalize_bar_grid(
    fig: plt.Figure,  # type: ignore[name-defined]
    axes: np.ndarray,
    row_widths: list[int],
    color_dict: dict[str, str],
    display_models: list[str],
    output_dir: Path,
    stem: str,
    *,
    watermark: str | None = None,
    legend_ncol: int = 3,
    legend_fontsize: int = 14,
    tight_layout_kwargs: dict[str, tp.Any] | None = None,
    pre_legend_hook: tp.Callable[[plt.Figure, np.ndarray], None] | None = None,  # type: ignore[name-defined]
    watermark_fontsize: int = 72,
    legend_builder: tp.Callable[[dict[str, str], list[str]], tuple[list, list[str]]]
    | None = None,
    legend_anchor_offset: float = 0.02,
    legend_box: tuple[int, int, int, int] | None = None,
    legend_box_loc: str = "center",
) -> Path:
    """Apply ``tight_layout``, draw the grouped legend, watermark, and save.

    *row_widths* gives the number of visible (non-``None``) subplots per row,
    used to anchor the legend beneath the last row.  *pre_legend_hook* lets
    the caller place additional figure text (e.g. row labels) after
    ``tight_layout`` but before the legend is anchored.

    Pass *legend_builder* to override the default
    ``build_grouped_legend(..., include_overlap=True)`` -- useful for the
    non-EEG bar chart whose legend lays out five (device, group) columns.

    *legend_box* (``(top_row, left_col, bot_row, right_col)``) overrides
    the default below-the-grid placement: when supplied, the legend is
    centred inside the rectangle spanned by ``axes[top_row, left_col]``
    and ``axes[bot_row, right_col]``.  Negative indices are supported.
    Use this to recover empty cells in rows that don't fill all columns.
    """
    if legend_builder is None:
        handles, labels = build_grouped_legend(
            color_dict, display_models, include_overlap=True
        )
    else:
        handles, labels = legend_builder(color_dict, display_models)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(**(tight_layout_kwargs or {}))

    if pre_legend_hook is not None:
        pre_legend_hook(fig, axes)

    if legend_box is not None:
        top_row, left_col, bot_row, right_col = legend_box
        _leg_top = axes[top_row, left_col].get_position()
        _leg_bot = axes[bot_row, right_col].get_position()
        _leg_cy = (_leg_top.y1 + _leg_bot.y0) / 2
        # Pick the anchor x-coord based on ``legend_box_loc``.
        # ``"center left"`` glues the legend's left edge to the LEFT of
        # the box (so it can't overlap the subplots just to its left,
        # while overflow harmlessly extends into the right margin --
        # which ``bbox_inches="tight"`` preserves).  ``"center"`` keeps
        # the legacy behaviour of centering inside the box.
        if legend_box_loc == "center left":
            _leg_anchor_x = _leg_top.x0
        else:
            _leg_anchor_x = (_leg_top.x0 + _leg_bot.x1) / 2
        legend = fig.legend(
            handles,
            labels,
            loc=legend_box_loc,
            bbox_to_anchor=(_leg_anchor_x, _leg_cy),
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            frameon=True,
            edgecolor="#cccccc",
            fancybox=False,
            borderpad=0.6,
            columnspacing=1.2,
            handlelength=1.4,
            handletextpad=0.4,
        )
    else:
        n_rows = axes.shape[0]
        last_row_width = row_widths[-1]
        last_row_bottom = min(
            axes[n_rows - 1, c].get_position().y0 for c in range(last_row_width)
        )
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, last_row_bottom - legend_anchor_offset),
            ncol=legend_ncol,
            fontsize=legend_fontsize,
            frameon=True,
            edgecolor="#cccccc",
        )
    bold_legend_titles(legend)

    if watermark:
        add_watermark(fig, watermark, fontsize=watermark_fontsize)

    return save_figure(fig, output_dir, stem)


# ---------------------------------------------------------------------------
# Left-margin labels (task / category headers in bar-chart grids)
# ---------------------------------------------------------------------------


def draw_left_margin_label(
    fig: plt.Figure,  # type: ignore[name-defined]
    axes: np.ndarray,
    *,
    label: str,
    first_row: int,
    last_row: int,
    color: str,
    fontsize: float,
    x_offset: float = 0.025,
    rotation: int = 90,
    fontweight: str = "bold",
    anchor_col: int = 0,
) -> None:
    """Draw a vertical, bold label centred to the left of a span of rows.

    Used for task labels in ``plot_full_bar_chart`` and category labels
    in ``plot_bar_chart``.  ``first_row`` and ``last_row`` are inclusive
    indices into ``axes`` (the rectangular grid returned by
    ``plt.subplots``); the label is positioned vertically halfway
    between the top of the first row and the bottom of the last row.

    *anchor_col* selects which axes column to read positions from -- use
    ``1`` (instead of the default ``0``) when column ``0`` is occupied
    by an unrelated subplot (e.g. a marginal mean-rank panel) so the
    label still hugs the actual content block.
    """
    top_pos = axes[first_row, anchor_col].get_position()
    bot_pos = axes[last_row, anchor_col].get_position()
    y_center = (top_pos.y1 + bot_pos.y0) / 2
    fig.text(
        top_pos.x0 - x_offset,
        y_center,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        rotation=rotation,
    )


# ---------------------------------------------------------------------------
# Figure save / watermark utilities
# ---------------------------------------------------------------------------


def add_watermark(
    fig: plt.Figure,  # type: ignore[name-defined]
    watermark: str,
    *,
    fontsize: int = 48,
) -> None:
    """Overlay a translucent diagonal watermark on the figure."""
    fig.text(
        0.5,
        0.5,
        watermark,
        fontsize=fontsize,
        color="red",
        alpha=0.25,
        ha="center",
        va="center",
        rotation=30,
        transform=fig.transFigure,
        fontweight="bold",
    )


def save_figure(
    fig: plt.Figure,  # type: ignore[name-defined]
    output_dir: Path,
    stem: str,
) -> Path:
    """Save *fig* as PNG + PDF under *output_dir* and close it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname_png = output_dir / f"{stem}.png"
    fname_pdf = output_dir / f"{stem}.pdf"
    fig.savefig(fname_png, bbox_inches="tight", dpi=150)
    fig.savefig(fname_pdf, bbox_inches="tight")
    plt.close(fig)
    return fname_png
