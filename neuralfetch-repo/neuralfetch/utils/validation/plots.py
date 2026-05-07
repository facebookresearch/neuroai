# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Plotting helpers for validation reports: decoding, TRF, QC, and interactive HTML."""

from __future__ import annotations

import typing as tp

import numpy as np
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib import pyplot as plt

from .config import StudyValidation

if tp.TYPE_CHECKING:
    import mne
    import xarray


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _subject_sort_key(subject_id: str) -> tuple[int, int, str]:
    """Natural sort key for subject IDs of the form ``"StudyName/N"``.

    Numeric suffixes (e.g. ``"Grootswagers2022Human/10"``) are ordered as
    integers so the sequence is 1, 2, 3 … rather than 1, 10, 11, 2 ….
    Non-numeric suffixes fall back to plain string comparison.
    """
    suffix = subject_id.split("/", 1)[-1] if "/" in subject_id else subject_id
    # The tuple is compared element-by-element by sorted():
    #   [0] group: 0 for numeric suffixes, 1 for non-numeric → all numeric IDs
    #              sort before all non-numeric ones as a group.
    #   [1] within numeric group: the integer value (2 < 10 < 11), not the
    #              string ("10" < "2" lexicographically).  Placeholder 0 for
    #              non-numeric (position [2] handles ordering there instead).
    #   [2] within non-numeric group: plain alphabetical string order.
    #              Placeholder "" for numeric (position [1] handles it).
    # ValueError is raised by int() when suffix is not a valid integer string.
    try:
        return (0, int(suffix), "")
    except ValueError:
        return (1, 0, suffix)


def _short_subject_labels(subject_ids: list[str]) -> list[str]:
    """Shorten subject identifiers for the per-study clickable legend.

    The study name prefix (e.g. ``"Grootswagers2022Human/"``) is stripped
    since the section is already scoped to one study, and purely numeric
    suffixes are zero-padded to a common width so sorted labels align
    visually (``"0" -> "00"``, ``"49" -> "49"``).  Non-numeric suffixes
    are left untouched beyond prefix stripping.
    """
    stripped = [s.split("/", 1)[-1] if "/" in s else s for s in subject_ids]
    if all(part.isdigit() for part in stripped) and stripped:
        width = max(2, max(len(p) for p in stripped))
        return [p.zfill(width) for p in stripped]
    return stripped


# ---------------------------------------------------------------------------
# Decoding plots
# ---------------------------------------------------------------------------


def _plot_subject_scores(
    scores: "xarray.DataArray",
    subject_id: str,
    validation: StudyValidation,
) -> plt.Figure:
    """Create a time-resolved decoding figure for a single subject."""
    df = scores.sel(subject=subject_id).to_dataframe("score").reset_index()

    has_test_query = "test_query" in df.columns and df["test_query"].nunique() > 1
    fig, ax = plt.subplots(figsize=(8, 4))
    hue = "test_query" if has_test_query else None
    sns.lineplot(x="train_time", y="score", data=df, hue=hue, ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(validation.reference_metric_name or "Score", rotation=90)
    short_id = subject_id.split("/", 1)[-1] if "/" in subject_id else subject_id
    ax.set_title(f"Participant {short_id}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    if validation.reference_metric is not None:
        ax.axhline(
            validation.reference_metric,
            color="red",
            linestyle=":",
            linewidth=1,
            label=f"reference ({validation.reference_metric:.3f})",
        )
    if has_test_query or validation.reference_metric is not None:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=3,
            fancybox=True,
            shadow=True,
        )
    fig.tight_layout()
    return fig


def _plot_grand_average(
    scores: "xarray.DataArray",
    validation: StudyValidation,
) -> plt.Figure:
    """Create a grand-average decoding figure across all participants."""
    df = scores.to_dataframe("score").reset_index()

    has_test_query = "test_query" in df.columns and df["test_query"].nunique() > 1
    fig, ax = plt.subplots(figsize=(8, 4))
    hue = "test_query" if has_test_query else None
    sns.lineplot(x="train_time", y="score", data=df, hue=hue, ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(validation.reference_metric_name or "Score", rotation=90)
    ax.set_title("Grand Average (all participants)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    if validation.reference_metric is not None:
        ax.axhline(
            validation.reference_metric,
            color="red",
            linestyle=":",
            linewidth=1,
            label=f"reference ({validation.reference_metric:.3f})",
        )
    if has_test_query or validation.reference_metric is not None:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=3,
            fancybox=True,
            shadow=True,
        )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Interactive group grand average (Plotly)
# ---------------------------------------------------------------------------


def _plot_group_grand_average_html(
    scores: "xarray.DataArray",
    validation: StudyValidation,
) -> str:
    """Build an HTML-embedded interactive grand-average plot via Plotly.

    Renders one translucent line per subject plus a bold grand-average
    line.  Hovering over any participant line highlights it and dims all
    others; moving the cursor away restores the default state.  The
    legend is placed horizontally below the figure.  A compact "Toggle
    participants" button above the chart shows or hides all per-subject
    traces in one click.
    """
    import math
    import uuid

    import plotly.graph_objects as go

    df = scores.to_dataframe("score").reset_index()
    if "train_time" not in df.columns or "subject" not in df.columns:
        raise ValueError("scores must have 'train_time' and 'subject' coords")
    per_subj = df.groupby(["subject", "train_time"])["score"].mean().reset_index()

    subject_order: list[str] = sorted(
        [str(s) for s in per_subj["subject"].unique()], key=_subject_sort_key
    )
    short_labels = _short_subject_labels(subject_order)

    # Cycle through a soft qualitative palette for participant lines.
    _PALETTE = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
        "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    ]

    _PARTICIPANT_OPACITY = 0.35
    _PARTICIPANT_WIDTH = 1.2

    fig = go.Figure()

    # --- per-subject traces (initially all visible at low opacity) ---
    for full_id, short in zip(subject_order, short_labels):
        sub_df = (
            per_subj[per_subj["subject"] == full_id]
            .sort_values("train_time")
        )
        color = _PALETTE[subject_order.index(full_id) % len(_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=sub_df["train_time"].to_numpy(),
                y=sub_df["score"].to_numpy(),
                mode="lines",
                name=short,
                line=dict(color=color, width=_PARTICIPANT_WIDTH),
                opacity=_PARTICIPANT_OPACITY,
                hovertemplate=f"<b>Participant {short}</b><br>"
                              "Time: %{x:.3f} s<br>"
                              "Score: %{y:.4f}<extra></extra>",
                legendgroup="participants",
            )
        )

    # --- grand average ---
    grand = per_subj.groupby("train_time")["score"].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=grand["train_time"].to_numpy(),
            y=grand["score"].to_numpy(),
            mode="lines",
            name="Grand Average",
            line=dict(color="black", width=3),
            opacity=1.0,
            hovertemplate="<b>Grand Average</b><br>"
                          "Time: %{x:.3f} s<br>"
                          "Score: %{y:.4f}<extra></extra>",
            legendgroup="summary",
        )
    )

    # --- zero baseline ---
    x_range = grand["train_time"].to_numpy()
    fig.add_trace(
        go.Scatter(
            x=[x_range[0], x_range[-1]],
            y=[0, 0],
            mode="lines",
            line=dict(color="gray", width=1, dash="dash"),
            opacity=0.6,
            name="Baseline (0)",
            hoverinfo="skip",
            showlegend=False,
            legendgroup="summary",
        )
    )

    # --- optional reference metric ---
    if validation.reference_metric is not None:
        ref_label = (
            f"Reference ({validation.reference_metric_name or 'ref'}"
            f" = {validation.reference_metric:.3f})"
        )
        fig.add_trace(
            go.Scatter(
                x=[x_range[0], x_range[-1]],
                y=[validation.reference_metric, validation.reference_metric],
                mode="lines",
                line=dict(color="red", width=1.5, dash="dot"),
                opacity=0.8,
                name=ref_label,
                hoverinfo="skip",
                legendgroup="summary",
            )
        )

    # Lock y-axis range from the full data so it doesn't rescale when
    # participants are toggled on/off.
    all_y = per_subj["score"].values.tolist()
    if validation.reference_metric is not None:
        all_y.append(validation.reference_metric)
    y_min, y_max = min(all_y), max(all_y)
    y_pad = max((y_max - y_min) * 0.08, 0.01)
    y_range = [y_min - y_pad, y_max + y_pad]

    # Estimate legend height: ~18 short subject IDs fit per row at font=11.
    # Add 2 extra rows for Grand Average / Reference lines.
    n_legend_rows = math.ceil(len(subject_order) / 18) + 2
    legend_height_px = n_legend_rows * 22
    bottom_margin = legend_height_px + 20

    score_label = validation.reference_metric_name or "Score"
    fig.update_layout(
        title="Group Grand Average",
        xaxis_title="Time (s)",
        yaxis_title=score_label,
        height=540,
        margin=dict(l=60, r=20, t=50, b=bottom_margin),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="left",
            x=0,
            font=dict(size=11),
            tracegroupgap=0,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(showgrid=True, gridcolor="#ebebeb", range=y_range),
    )

    uid = uuid.uuid4().hex[:8]
    div_id = f"ga-plotly-{uid}"
    button_id = f"ga-toggle-{uid}"

    fig_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        div_id=div_id,
        config={"displayModeBar": False},
    )

    # Number of participant traces (all traces except grand avg, baseline,
    # and optional reference).  These are always the first N traces.
    n_participants = len(subject_order)

    # Indices of non-participant traces — these must keep their opacity
    # untouched during hover so grand avg and reference stay prominent.
    toggle_js = f"""
(function() {{
  var divId = '{div_id}';
  var btnId = '{button_id}';
  var nPart = {n_participants};
  var defaultOpacity = {_PARTICIPANT_OPACITY};
  var allShown = true;

  function getDiv() {{ return document.getElementById(divId); }}

  // --- hover highlight ---
  document.getElementById(divId).on('plotly_hover', function(eventData) {{
    var el = getDiv();
    if (!el || !el.data) return;
    var hovered = eventData.points[0].curveNumber;
    var opacities = el.data.map(function(trace, i) {{
      if (i >= nPart) return trace.opacity;   // grand avg / baseline / ref: unchanged
      return i === hovered ? 1.0 : 0.05;
    }});
    Plotly.restyle(el, {{ opacity: opacities }});
  }});

  document.getElementById(divId).on('plotly_unhover', function() {{
    var el = getDiv();
    if (!el || !el.data) return;
    var opacities = el.data.map(function(trace, i) {{
      return i < nPart ? defaultOpacity : trace.opacity;
    }});
    Plotly.restyle(el, {{ opacity: opacities }});
  }});

  // --- toggle button ---
  document.getElementById(btnId).addEventListener('click', function() {{
    var el = getDiv();
    if (!el || !el.data) return;
    allShown = !allShown;
    var indices = Array.from({{length: nPart}}, function(_, i) {{ return i; }});
    var visibility = allShown ? 'true' : 'legendonly';
    Plotly.restyle(el, {{ visible: visibility }}, indices);
    this.textContent = allShown ? 'Hide participants' : 'Show participants';
  }});
}})();
"""

    button_html = (
        f'<button id="{button_id}" type="button" '
        'style="font-size:0.8em; padding:2px 8px; margin-bottom:4px; cursor:pointer;">'
        "Hide participants</button>"
    )

    return f"{button_html}{fig_html}<script>{toggle_js}</script>"


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _scores_summary_table(
    scores: "xarray.DataArray",
) -> list[dict[str, tp.Any]]:
    """Compute per-subject peak score and grand-average peak."""
    df = scores.to_dataframe("score").reset_index()
    rows: list[dict[str, tp.Any]] = []
    for subject_id, sub_df in df.groupby("subject"):
        peak_score = sub_df["score"].max()
        peak_time = sub_df.loc[sub_df["score"].idxmax(), "train_time"]
        sid = str(subject_id)
        short_id = sid.split("/", 1)[-1] if "/" in sid else sid
        rows.append(
            {"subject": short_id, "peak_score": peak_score, "peak_time": peak_time}
        )
    rows.sort(key=lambda r: _subject_sort_key(r["subject"]))
    grand_mean_score = float(np.mean([r["peak_score"] for r in rows]))
    grand_mean_time = float(np.mean([r["peak_time"] for r in rows]))
    rows.append({"subject": "Grand Average", "peak_score": grand_mean_score, "peak_time": grand_mean_time})
    return rows


def _summary_table_html(rows: list[dict[str, tp.Any]]) -> str:
    """Render the scores summary as a Bootstrap-styled HTML table."""
    lines = [
        '<div style="overflow-x:auto;">',
        '<table class="table table-sm table-striped table-hover table-bordered">',
        '<thead class="table-dark"><tr><th>Participant</th><th>Peak Score</th><th>Peak Time (s)</th></tr></thead>',
        "<tbody>",
    ]
    for row in rows:
        peak = (
            f"{row['peak_score']:.4f}"
            if isinstance(row["peak_score"], float)
            else str(row["peak_score"])
        )
        time = (
            f"{row['peak_time']:.3f}"
            if isinstance(row["peak_time"], float)
            else str(row["peak_time"])
        )
        row_class = ' class="table-primary fw-bold"' if row["subject"] == "Grand Average" else ""
        lines.append(
            f"<tr{row_class}><td>{row['subject']}</td><td>{peak}</td><td>{time}</td></tr>"
        )
    lines.append("</tbody></table></div>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# QC drop grid
# ---------------------------------------------------------------------------


def _plot_drop_grid(
    subject_channels: dict[str, list[str]],
    subject_bads: dict[str, list[str]],
    excluded_subjects: tp.Iterable[str],
    all_subjects: tp.Iterable[str],
) -> plt.Figure:
    """Render a Subjects x Channels drop grid.

    ``1`` (red) = channel is bad for that subject, OR the subject is
    fully excluded (whole row marked bad).
    """
    subjects_sorted = sorted({str(s) for s in all_subjects})
    excluded = {str(s) for s in excluded_subjects}
    channel_union: set[str] = set()
    for chs in subject_channels.values():
        channel_union.update(chs)
    channels_sorted = sorted(channel_union)

    n_rows = len(subjects_sorted)
    n_cols = max(len(channels_sorted), 1)
    matrix = np.zeros((n_rows, n_cols), dtype=float)
    for i, subj in enumerate(subjects_sorted):
        if subj in excluded:
            matrix[i, :] = 1.0
            continue
        bads = set(subject_bads.get(subj, ()))
        for j, ch in enumerate(channels_sorted):
            if ch in bads:
                matrix[i, j] = 1.0

    # Strip study-name prefix for compact y-axis labels.
    short_labels = [s.split("/", 1)[-1] if "/" in s else s for s in subjects_sorted]

    height = max(2.0, 0.3 * n_rows + 1.5)
    width = max(6.0, 0.18 * n_cols + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(matrix, aspect="auto", cmap="Reds", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(channels_sorted, rotation=90, fontsize=6)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Participant")
    ax.set_title("Participants x Channels: drops (red = bad / excluded)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# TRF plots
# ---------------------------------------------------------------------------


def _plot_trf_grand_average(
    trf_scores: "xarray.DataArray",
) -> plt.Figure:
    """Render a grand-average TRF encoding score bar chart across all subjects.

    Averages *trf_scores* over the ``subject`` dimension (and ``split`` if
    present) and plots per-channel mean Pearson r with ±1 SEM error bars.

    Parameters
    ----------
    trf_scores : xarray.DataArray
        Scores from :meth:`neuralyze.trf.TRFScoring.get_scores` with at least
        dimensions ``subject`` and ``dim`` (channels).

    Returns
    -------
    matplotlib.figure.Figure
    """
    reduced = trf_scores
    if "split" in reduced.dims:
        reduced = reduced.mean(dim="split")

    n_subjects = len(reduced.coords["subject"])
    mean_scores = np.atleast_1d(np.asarray(reduced.mean(dim="subject").values).squeeze())
    sem_scores = np.atleast_1d(
        np.asarray(reduced.std(dim="subject").values).squeeze()
    ) / np.sqrt(n_subjects)

    fig, ax = plt.subplots(figsize=(max(6.0, 0.15 * len(mean_scores)), 4))
    ax.bar(
        range(len(mean_scores)),
        mean_scores,
        yerr=sem_scores,
        color="steelblue",
        width=0.8,
        error_kw={"linewidth": 0.7, "capsize": 2},
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Pearson r")
    ax.set_title(
        f"Grand Average — TRF per-channel scores"
        f" (mean ± SEM, n={n_subjects} participants)"
    )
    fig.tight_layout()
    return fig


def _plot_trf_scores(
    trf_scores: "xarray.DataArray",
    subject_id: str,
    info: "mne.Info | None" = None,
) -> plt.Figure:
    """Render per-channel TRF encoding scores for a single subject.

    Creates a figure with one or two subplots:

    * **Bar chart** (always): channel index on the x-axis, mean Pearson r on
      the y-axis.  Each bar represents one MEG/EEG channel.
    * **Topomap** (when *info* is provided): spatially arranged channel scores
      via :func:`mne.viz.plot_topomap`.  Skipped gracefully when no sensor
      positions are found in *info*.

    Parameters
    ----------
    trf_scores : xarray.DataArray
        Scores from :meth:`neuralyze.trf.TRFScoring.get_scores` with at least
        dimensions ``subject`` and ``dim`` (channels).  A ``split`` dimension
        is averaged over if present.
    subject_id : str
        Full subject identifier (e.g. ``"ThingsOpm2025Expanded/1"``).
    info : mne.Info or None
        MNE info object containing sensor positions for the topomap.  When
        ``None`` or when the info has no digitisation / channel locations,
        only the bar chart is shown.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import logging

    import mne

    _logger = logging.getLogger(__name__)

    subj_scores = trf_scores.sel(subject=subject_id)
    if "split" in subj_scores.dims:
        subj_scores = subj_scores.mean(dim="split")
    scores_np = np.atleast_1d(np.asarray(subj_scores.values).squeeze())

    short_id = subject_id.split("/", 1)[-1] if "/" in subject_id else subject_id

    # Attempt topomap: create both subplots unconditionally, then hide the
    # topomap axis if mne.viz.plot_topomap raises (e.g. no channel positions,
    # or channel-count mismatch after non-PCA scoring).
    fig, (ax_bar, ax_topo) = plt.subplots(1, 2, figsize=(12, 4))

    ax_bar.bar(range(len(scores_np)), scores_np, color="steelblue", width=0.8)
    ax_bar.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax_bar.set_xlabel("Channel index")
    ax_bar.set_ylabel("Pearson r")
    ax_bar.set_title(f"Participant {short_id} — TRF per-channel scores")

    if info is not None:
        try:
            mne.viz.plot_topomap(
                scores_np,
                info,
                axes=ax_topo,
                show=False,
                contours=4,
            )
            ax_topo.set_title(f"Participant {short_id} — TRF topomap")
        except Exception as exc:
            _logger.warning("TRF topomap failed for %s: %s", subject_id, exc)
            ax_topo.set_visible(False)
    else:
        ax_topo.set_visible(False)

    fig.tight_layout()
    return fig
