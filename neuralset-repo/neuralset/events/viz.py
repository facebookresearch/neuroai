# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualization functions for neuralset events.

- :func:`plot_events` — timeline view of a single recording
- :func:`plot_study` — treemap of study > subject > timeline
- :func:`plot_overlap` — UpSet-style plot of stimulus sharing across groups

All functions return a :class:`plotly.graph_objects.Figure`.
"""

import typing as tp
from collections import Counter
from pathlib import Path

import pandas as pd

from . import etypes


def _format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.0f}min"
    return f"{seconds:.0f}s"


# ═══════════════════════════════════════════════════════════════════
# plot_events
# ═══════════════════════════════════════════════════════════════════


def plot_events(events: pd.DataFrame) -> tp.Any:
    """Plot events as horizontal bars with event types on the y-axis.

    Parameters
    ----------
    events : pd.DataFrame
        Events from a **single** timeline. Must have ``timeline``, ``type``,
        ``start`` and ``duration`` columns.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If *events* contains more than one timeline.
    """
    import plotly.graph_objects as go

    timelines = events["timeline"].unique()
    if len(timelines) != 1:
        raise ValueError(
            f"plot_events expects a single timeline, got {len(timelines)}: "
            f"{list(timelines)}"
        )

    event_types = sorted(events["type"].unique())
    text_types = {
        n for n, c in etypes.Event._CLASSES.items() if issubclass(c, etypes.BaseText)
    }
    data_types = {
        n for n, c in etypes.Event._CLASSES.items() if issubclass(c, etypes.BaseDataEvent)
    }

    colors = [
        f"hsl({i * 360 // max(len(event_types), 1)}, 65%, 55%)"
        for i in range(len(event_types))
    ]

    fig = go.Figure()

    for i, etype in enumerate(event_types):
        subset = events[events["type"] == etype]
        durations = subset["duration"].fillna(0)
        spans = subset[durations > 0]
        points = subset[durations <= 0]

        if len(spans):
            has_text = etype in text_types and "text" in events.columns
            has_fp = etype in data_types and "filepath" in events.columns
            labels = []
            hovers = []
            for _, row in spans.iterrows():
                if has_text:
                    lbl = str(row.get("text", ""))
                elif has_fp:
                    fp = str(row.get("filepath", ""))
                    lbl = Path(fp).name if fp else ""
                else:
                    lbl = ""
                labels.append(lbl)
                hovers.append(
                    f"<b>{etype}</b><br>"
                    f"start: {row['start']:.2f}s<br>"
                    f"duration: {row['duration']:.2f}s" + (f"<br>{lbl}" if lbl else "")
                )

            for start, dur, label, hover in zip(
                spans["start"], spans["duration"], labels, hovers
            ):
                fig.add_trace(
                    go.Bar(
                        x=[dur],
                        y=[etype],
                        base=[start],
                        orientation="h",
                        marker_color=colors[i],
                        text=label if label else None,
                        textposition="inside",
                        insidetextanchor="middle",
                        textfont_size=9,
                        hovertext=hover,
                        hoverinfo="text",
                        showlegend=False,
                        width=0.6,
                    )
                )

        if len(points):
            fig.add_trace(
                go.Scatter(
                    x=points["start"],
                    y=[etype] * len(points),
                    mode="markers",
                    marker=dict(symbol="line-ns", size=12, color=colors[i], line_width=2),
                    hovertext=[
                        f"<b>{etype}</b><br>start: {s:.2f}s" for s in points["start"]
                    ],
                    hoverinfo="text",
                    showlegend=False,
                )
            )

    parts = []
    for col in ("study", "subject"):
        if col in events.columns:
            vals = events[col].dropna().unique()
            if len(vals):
                parts.append(str(vals[0]))
    parts.append(timelines[0])

    t_min = events["start"].min()
    t_max = (events["start"] + events["duration"].fillna(0)).max()

    fig.update_layout(
        title=":".join(parts),
        xaxis_title="Time (s)",
        xaxis_range=[t_min, t_max],
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(reversed(event_types)),
        ),
        barmode="overlay",
        height=max(250, len(event_types) * 50 + 100),
        margin=dict(l=100, r=30, t=50, b=50),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=False)
    return fig


# ═══════════════════════════════════════════════════════════════════
# plot_study
# ═══════════════════════════════════════════════════════════════════


def plot_study(events: pd.DataFrame) -> tp.Any:
    """Treemap of events grouped by **study > subject > timeline**.

    Rectangle areas are proportional to each timeline's duration.

    Parameters
    ----------
    events : pd.DataFrame
        Events with ``timeline``, ``start``, ``duration`` columns and
        optionally ``study`` and ``subject``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.express as px

    df = events.copy()
    for col in ("study", "subject"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    stops = df["start"] + df["duration"].fillna(0)
    tl_info = (
        df.assign(_stop=stops)
        .groupby(["study", "subject", "timeline"])
        .agg(start=("start", "min"), stop=("_stop", "max"))
        .reset_index()
    )
    tl_info["duration"] = tl_info["stop"] - tl_info["start"]
    tl_info["duration_label"] = tl_info["duration"].apply(_format_duration)

    fig = px.treemap(
        tl_info,
        path=["study", "subject", "timeline"],
        values="duration",
        custom_data=["duration_label"],
    )
    fig.update_traces(
        texttemplate="%{label}<br>%{customdata[0]}",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "duration: %{customdata[0]}<br>"
            "share: %{percentRoot:.1%}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        height=600,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# plot_overlap
# ═══════════════════════════════════════════════════════════════════


def plot_overlap(
    events: pd.DataFrame,
    *,
    event_type: str | tp.Type,
    x: str = "subject",
    y: str | None = None,
    n_top: int = 25,
    log: str = "",
) -> tp.Any:
    """UpSet-style plot showing overlap of one event dimension across another.

    For each unique *y*-value, computes which *x*-groups contain it,
    then displays the intersection patterns.

    When there are more than 20 groups, automatically switches to a
    sharing-degree histogram for readability.

    Parameters
    ----------
    events : pd.DataFrame
        Events DataFrame (may span multiple subjects / timelines).
    event_type : str or Event subclass
        Which event type to analyze (e.g. ``"Image"``, ``"Word"``).
    x : str
        The "sets" dimension (default ``"subject"``).
    y : str or None
        The "elements" dimension — counted in intersection bars.
        Auto-detected from *event_type* if *None*: ``"text"`` for text
        events, ``"filepath"`` for data events.
    n_top : int
        Maximum number of intersection patterns to display (UpSet mode).
    log : str
        Axes to show on log scale. ``"x"``, ``"y"``, or ``"xy"`` / ``"both"``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    helper = etypes.EventTypesHelper(event_type)
    if y is None:
        cls = helper.classes[0]
        if issubclass(cls, etypes.BaseText):
            y = "text"
        elif issubclass(cls, etypes.BaseDataEvent):
            y = "filepath"
        else:
            raise ValueError(
                f"Cannot auto-detect y for {event_type!r} — pass y= explicitly."
            )

    df = events[events["type"].isin(helper.names)]
    if df.empty:
        raise ValueError(f"No events of type {event_type!r} found")
    for col in (x, y):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in events")

    membership: dict[str, set[str]] = {}
    for elem, grp in zip(df[y].astype(str), df[x].astype(str)):
        membership.setdefault(elem, set()).add(grp)

    all_groups = sorted(df[x].astype(str).unique())
    n_groups = len(all_groups)

    set_sizes: dict[str, int] = {g: 0 for g in all_groups}
    for groups in membership.values():
        for g in groups:
            set_sizes[g] += 1

    type_label = event_type if isinstance(event_type, str) else event_type.__name__
    total_unique = len(membership)

    log_x = "x" in log.lower() or log.lower() == "both"
    log_y = "y" in log.lower() or log.lower() == "both"

    if n_groups <= 20:
        pattern_counts = Counter(frozenset(gs) for gs in membership.values())
        top = pattern_counts.most_common(n_top)
        return _plot_overlap_upset(
            top,
            all_groups,
            set_sizes,
            n_groups,
            type_label,
            total_unique,
            x,
            y,
            log_x=log_x,
            log_y=log_y,
        )
    return _plot_overlap_histogram(
        membership,
        all_groups,
        set_sizes,
        n_groups,
        type_label,
        total_unique,
        x,
        y,
        log_x=log_x,
        log_y=log_y,
    )


def _plot_overlap_upset(
    top: list,
    all_groups: list,
    set_sizes: dict,
    n_groups: int,
    type_label: str,
    total_unique: int,
    x: str,
    y: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
) -> tp.Any:
    """UpSet dot-matrix layout — used when n_groups <= 20."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_patterns = len(top)
    group_to_idx = {g: i for i, g in enumerate(all_groups)}
    pattern_labels = [
        " & ".join(sorted(p)) if len(p) <= 3 else f"{len(p)} {x}s" for p, _ in top
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        row_heights=[0.3, 0.7],
        column_widths=[0.35, 0.65],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    # Top-right: per-group set sizes
    sizes = [set_sizes[g] for g in all_groups]
    fig.add_trace(
        go.Bar(
            x=all_groups,
            y=sizes,
            marker_color="steelblue",
            text=[f"{s:,}" for s in sizes],
            textposition="outside",
            textfont_size=9,
            hovertemplate="%{x}: %{y:,} unique " + y + "<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Bottom-left: intersection bars (horizontal)
    counts = [c for _, c in top]
    bar_colors = ["steelblue" if len(p) > 1 else "lightsteelblue" for p, _ in top]
    fig.add_trace(
        go.Bar(
            y=list(range(n_patterns)),
            x=counts,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{c:,}" for c in counts],
            textposition="outside",
            textfont_size=9,
            hovertext=[
                f"{pattern_labels[i]}: {c:,} unique {type_label}"
                for i, (_, c) in enumerate(top)
            ],
            hoverinfo="text",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Bottom-right: dot matrix
    for i, (pattern, _count) in enumerate(top):
        xs_in = sorted(group_to_idx[g] for g in pattern if g in group_to_idx)
        xs_out = [j for j in range(n_groups) if j not in xs_in]
        in_names = [all_groups[j] for j in xs_in]
        out_names = [all_groups[j] for j in xs_out]

        fig.add_trace(
            go.Scatter(
                x=in_names,
                y=[i] * len(xs_in),
                mode="markers",
                marker=dict(size=10, color="black"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=2,
        )
        if out_names:
            fig.add_trace(
                go.Scatter(
                    x=out_names,
                    y=[i] * len(xs_out),
                    mode="markers",
                    marker=dict(
                        size=6, color="lightgray", line=dict(color="lightgray", width=1)
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

        if len(xs_in) > 1:
            fig.add_trace(
                go.Scatter(
                    x=[in_names[0], in_names[-1]],
                    y=[i, i],
                    mode="lines",
                    line=dict(color="black", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

    # Layout
    fig.update_xaxes(
        showticklabels=False,
        row=1,
        col=2,
    )
    fig.update_xaxes(
        autorange="reversed",
        title_text=f"Unique {type_label}",
        type="log" if log_x else "linear",
        row=2,
        col=1,
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=all_groups,
        row=2,
        col=2,
    )
    fig.update_yaxes(
        title_text=f"Total unique {y}",
        type="log" if log_y else "linear",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        showticklabels=False,
        autorange="reversed",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        showticklabels=False,
        autorange="reversed",
        row=2,
        col=2,
    )
    # Hide top-left cell
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=1)

    fig.update_layout(
        title=dict(
            text=(f"{type_label} overlap by {x}  ({total_unique:,} unique {y} total)"),
            font_size=15,
        ),
        height=max(400, n_patterns * 35 + 200),
        width=max(500, n_groups * 50 + 300),
        plot_bgcolor="white",
        margin=dict(l=50, r=30, t=60, b=50),
    )
    return fig


def _plot_overlap_histogram(
    membership: dict,
    all_groups: list,
    set_sizes: dict,
    n_groups: int,
    type_label: str,
    total_unique: int,
    x: str,
    y: str,
    *,
    log_x: bool = False,
    log_y: bool = False,
) -> tp.Any:
    """Sharing-degree histogram — used when there are too many groups for UpSet."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    degrees = [len(gs) for gs in membership.values()]
    degree_counts: dict[int, int] = Counter(degrees)

    bins = sorted(degree_counts)
    counts = [degree_counts[b] for b in bins]
    colors = []
    for b in bins:
        if b == n_groups:
            colors.append("steelblue")
        elif b == 1:
            colors.append("lightsteelblue")
        else:
            colors.append("#6baed6")

    sorted_groups = sorted(all_groups, key=lambda g: -set_sizes[g])
    show_groups = sorted_groups[:40]
    show_sizes = [set_sizes[g] for g in show_groups]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.08,
        subplot_titles=[
            "Sharing degree",
            f"Per-{x} totals" + (f" (top 40 / {n_groups})" if n_groups > 40 else ""),
        ],
    )

    fig.add_trace(
        go.Bar(
            x=bins,
            y=counts,
            marker_color=colors,
            text=[f"{c:,}" for c in counts],
            textposition="outside",
            textfont_size=9,
            hovertemplate=(
                "Shared by %{x} "
                + x
                + "s: %{y:,} unique "
                + type_label
                + "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Annotations for key values
    annotations = []
    if n_groups in degree_counts:
        annotations.append(
            dict(
                x=n_groups,
                y=degree_counts[n_groups],
                text=f"shared by all {n_groups}",
                showarrow=True,
                arrowhead=2,
                font=dict(color="steelblue", size=11),
                arrowcolor="steelblue",
                xref="x1",
                yref="y1",
            )
        )
    if 1 in degree_counts:
        annotations.append(
            dict(
                x=1,
                y=degree_counts[1],
                text=f"unique to 1 {x}",
                showarrow=True,
                arrowhead=2,
                font=dict(color="gray", size=11),
                arrowcolor="gray",
                ax=40,
                ay=-40,
                xref="x1",
                yref="y1",
            )
        )

    fig.add_trace(
        go.Bar(
            y=show_groups,
            x=show_sizes,
            orientation="h",
            marker_color="steelblue",
            text=[f"{s:,}" for s in show_sizes],
            textposition="outside",
            textfont_size=8,
            hovertemplate="%{y}: %{x:,} unique " + y + "<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text=f"Shared by N {x}s",
        type="log" if log_x else "linear",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=f"Unique {type_label}",
        type="log" if log_y else "linear",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text=f"Unique {y}",
        type="log" if log_x else "linear",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont_size=8,
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{type_label} overlap by {x}  "
                f"({total_unique:,} unique {y}, {n_groups} {x}s)"
            ),
            font_size=15,
        ),
        annotations=annotations,
        height=max(400, len(show_groups) * 18 + 150),
        width=900,
        plot_bgcolor="white",
        margin=dict(l=50, r=30, t=80, b=50),
    )
    return fig
