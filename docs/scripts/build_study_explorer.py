# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate the NeuralFetch dataset explorer from StudyInfo metadata."""

import math
import typing as tp
from html import escape
from pathlib import Path

import pandas as pd

import neuralset as ns
from neuralset.events import study

_SUMMARY_COLUMNS = [
    "name",
    "module",
    "aliases",
    "description",
    "url",
    "neuro_event_type",
    "event_types",
    "other_event_types",
    "n_subjects",
    "n_timelines",
    "n_query_events",
    "n_hours",
    "data_shape",
    "frequency",
    "query",
]

_REPORT_ROOT_ID = "neuralfetch-study-explorer"
_FRAGMENT_HEADER = (
    "<!-- Auto-generated from docs/scripts/build_study_explorer.py using neuralfetch StudyInfo metadata; "
    "hand-edits will be clobbered on regeneration. -->"
)


def _estimate_hours(info: study.StudyInfo) -> float:
    """Estimate total recording duration from StudyInfo without loading events."""
    if not info.data_shape or info.frequency <= 0 or info.num_timelines <= 0:
        return math.nan
    return info.num_timelines * info.data_shape[-1] / info.frequency / 3600


def _format_number(value: float) -> str:
    if pd.isna(value):
        return ""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _scale_log(
    value: float, min_value: float, max_value: float, out_min: float, out_max: float
) -> float:
    log_value = math.log10(value)
    log_min = math.log10(min_value)
    log_max = math.log10(max_value)
    if log_min == log_max:
        return (out_min + out_max) / 2
    return out_min + (log_value - log_min) / (log_max - log_min) * (out_max - out_min)


class StudyInfoSummaries(ns.BaseModel):
    """Summaries and HTML reports for discovered studies using only ``Study._info``."""

    neuro_types: str | list[str] | tp.Literal["all"] = "all"

    _NEURO_TYPES = {"Eeg", "Meg", "Emg", "Fmri", "Fnirs", "Ieeg"}

    def model_post_init(self, __context: tp.Any) -> None:
        if self.neuro_types == "all":
            return
        neuro_types = (
            self.neuro_types if isinstance(self.neuro_types, list) else [self.neuro_types]
        )
        for neuro_type in neuro_types:
            if neuro_type not in self._NEURO_TYPES:
                raise ValueError(
                    f"Not a valid neuro type: {neuro_type} "
                    f"(valid types: {self._NEURO_TYPES})"
                )

    def get_summaries(self) -> pd.DataFrame:
        rows: list[dict[str, tp.Any]] = []
        neuro_filter = self._neuro_filter()
        for name, cls in sorted(ns.Study.catalog().items()):
            if cls._info is None:
                continue
            neuro_event_types = sorted(cls.neuro_types())
            if not neuro_event_types:
                continue
            if neuro_filter is not None and not (set(neuro_event_types) & neuro_filter):
                continue

            info = cls._info
            n_hours = _estimate_hours(info)
            if pd.isna(n_hours):
                continue
            event_types = sorted(info.event_types_in_query)
            other_event_types = sorted(info.event_types_in_query - self._NEURO_TYPES)
            rows.append(
                {
                    "name": name,
                    "module": cls.__module__,
                    "aliases": ", ".join(cls.aliases),
                    "description": cls.description,
                    "url": cls.url,
                    "neuro_event_type": ", ".join(neuro_event_types),
                    "event_types": event_types,
                    "other_event_types": other_event_types,
                    "n_subjects": info.num_subjects,
                    "n_timelines": info.num_timelines,
                    "n_query_events": info.num_events_in_query,
                    "n_hours": n_hours,
                    "data_shape": info.data_shape,
                    "frequency": info.frequency,
                    "query": info.query,
                }
            )
        return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)

    def render_html_fragment(self) -> str:
        """Render a Sphinx-friendly HTML fragment for the study explorer."""
        summaries = self.get_summaries()
        if summaries.empty:
            raise RuntimeError("No studies with StudyInfo were found.")
        return "\n".join(
            [
                _FRAGMENT_HEADER,
                "<style>",
                _CSS,
                "</style>",
                self._render_body(summaries, include_title=False),
                "<script>",
                _JS,
                "</script>",
            ]
        )

    def to_html_fragment(self, path: Path) -> Path:
        """Write the Sphinx-friendly HTML fragment used by the docs."""
        path.write_text(self.render_html_fragment() + "\n", encoding="utf8")
        return path

    def to_html_report(self, path: Path) -> Path:
        """Write a self-contained HTML report with a scatterplot and event table."""
        summaries = self.get_summaries()
        if summaries.empty:
            raise RuntimeError("No studies with StudyInfo were found.")
        html = "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f"<title>{escape(self._report_title())}</title>",
                "<style>",
                "body { margin: 24px; }",
                _CSS,
                "</style>",
                "</head>",
                "<body>",
                self._render_body(summaries, include_title=True),
                "<script>",
                _JS,
                "</script>",
                "</body>",
                "</html>",
            ]
        )
        path.write_text(html, encoding="utf8")
        return path

    def _render_body(self, summaries: pd.DataFrame, *, include_title: bool) -> str:
        pieces = [f'<div id="{_REPORT_ROOT_ID}">']
        if include_title:
            pieces.append(f"<h1>{escape(self._report_title())}</h1>")
        pieces.extend(
            [
                '<div id="study-tooltip" hidden></div>',
                self._render_controls(summaries),
                self._render_summary(summaries),
                "<section>",
                "<h2>Study Volume</h2>",
                '<p class="note">Hover over studies for descriptions, and click to access their webpage.</p>',
                self._render_scatter(summaries),
                "</section>",
                "<section>",
                "<h2>Event Types</h2>",
                '<p class="note">Click on a column header to filter for studies containing a particular event type.</p>',
                self._render_event_table(summaries),
                "</section>",
                "</div>",
            ]
        )
        return "\n".join(pieces)

    def _render_controls(self, summaries: pd.DataFrame) -> str:
        event_options = "".join(
            f'<option value="{escape(event_type)}">{escape(event_type)}</option>'
            for event_type in self._event_filter_options(summaries)
        )
        return (
            '<section class="controls">'
            '<label for="event-filter-select">Add event filter</label>'
            '<select id="event-filter-select">'
            '<option value="">Choose an event type...</option>'
            f"{event_options}"
            "</select>"
            '<span id="event-filter-status">All event types</span>'
            '<button id="clear-event-filter" type="button">Clear event filter</button>'
            "</section>"
        )

    def _event_filter_options(self, summaries: pd.DataFrame) -> list[str]:
        event_types = {
            event_type
            for event_types_ in summaries.event_types
            for event_type in event_types_
        }
        return sorted(
            event_types,
            key=lambda event_type: (event_type not in self._NEURO_TYPES, event_type),
        )

    def _render_summary(self, summaries: pd.DataFrame) -> str:
        n_subjects = int(summaries["n_subjects"].sum())
        n_timelines = int(summaries["n_timelines"].sum())
        n_hours = summaries["n_hours"].dropna().sum()
        return (
            '<section class="summary">'
            f'<div><strong id="summary-studies">{len(summaries)}</strong><span>studies</span></div>'
            f'<div><strong id="summary-subjects">{n_subjects:,}</strong><span>subjects</span></div>'
            f'<div><strong id="summary-timelines">{n_timelines:,}</strong><span>timelines</span></div>'
            f'<div><strong id="summary-hours">{_format_number(n_hours)}</strong><span>estimated hours</span></div>'
            "</section>"
        )

    def _render_scatter(self, summaries: pd.DataFrame) -> str:
        rows = summaries.dropna(subset=["n_hours"]).copy()
        rows = rows[(rows["n_hours"] > 0) & (rows["n_subjects"] > 0)]
        if rows.empty:
            return "<p>No studies have enough StudyInfo metadata to estimate volume.</p>"

        rows["hours_per_subject"] = rows["n_hours"] / rows["n_subjects"]
        rows = rows[rows["hours_per_subject"] > 0]
        if rows.empty:
            return "<p>No studies have positive hours per subject.</p>"

        width = 1800
        height = 1100
        pad_left = 90
        pad_right = 40
        pad_top = 60
        pad_bottom = 90
        plot_width = width - pad_left - pad_right
        plot_height = height - pad_top - pad_bottom
        x_min = float(rows["hours_per_subject"].min())
        x_max = float(rows["hours_per_subject"].max())
        y_min = float(rows["n_subjects"].min())
        y_max = float(rows["n_subjects"].max())
        h_min = float(rows["n_hours"].min())
        h_max = float(rows["n_hours"].max())

        def x_pos(value: float) -> float:
            return _scale_log(value, x_min, x_max, pad_left, pad_left + plot_width)

        def y_pos(value: float) -> float:
            return _scale_log(value, y_min, y_max, pad_top + plot_height, pad_top)

        def radius(value: float) -> float:
            if h_min == h_max:
                return 7
            return (
                4
                + (math.log10(value) - math.log10(h_min))
                / (math.log10(h_max) - math.log10(h_min))
                * 12
            )

        x_ticks = _log_ticks(x_min, x_max)
        y_ticks = _log_ticks(y_min, y_max)
        pieces = [
            '<div class="scatter-wrap">',
            f'<svg viewBox="0 0 {width} {height}" role="img" '
            'aria-label="Study volume scatterplot">',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        ]
        for tick in x_ticks:
            x = x_pos(tick)
            pieces.append(
                f'<line class="grid" x1="{x:.1f}" y1="{pad_top}" '
                f'x2="{x:.1f}" y2="{pad_top + plot_height}"/>'
            )
            pieces.append(
                f'<text class="axis-tick" x="{x:.1f}" y="{height - 45}" '
                f'text-anchor="middle">{escape(_format_number(tick))}</text>'
            )
        for tick in y_ticks:
            y = y_pos(tick)
            pieces.append(
                f'<line class="grid" x1="{pad_left}" y1="{y:.1f}" '
                f'x2="{pad_left + plot_width}" y2="{y:.1f}"/>'
            )
            pieces.append(
                f'<text class="axis-tick" x="{pad_left - 15}" y="{y + 4:.1f}" '
                f'text-anchor="end">{escape(_format_number(tick))}</text>'
            )
        pieces.extend(
            [
                f'<line class="axis" x1="{pad_left}" y1="{pad_top + plot_height}" '
                f'x2="{pad_left + plot_width}" y2="{pad_top + plot_height}"/>',
                f'<line class="axis" x1="{pad_left}" y1="{pad_top}" '
                f'x2="{pad_left}" y2="{pad_top + plot_height}"/>',
                f'<text class="axis-label" x="{pad_left + plot_width / 2:.1f}" '
                f'y="{height - 12}" text-anchor="middle">Estimated hours per subject</text>',
                f'<text class="axis-label" x="18" y="{pad_top + plot_height / 2:.1f}" '
                'text-anchor="middle" transform="rotate(-90 18 '
                f'{pad_top + plot_height / 2:.1f})">Number of subjects</text>',
            ]
        )

        for row in rows.sort_values("name").itertuples():
            x = x_pos(float(row.hours_per_subject))
            y = y_pos(float(row.n_subjects))
            r = radius(float(row.n_hours))
            color = _color_for_neuro_type(str(row.neuro_event_type))
            device = _primary_neuro_type(str(row.neuro_event_type))
            event_data = _data_list(row.event_types)  # type: ignore[arg-type]
            description = _description_attr(str(row.description))
            url = _url_attr(str(row.url))
            link_open = (
                f'<a href="{escape(url)}" target="_blank" rel="noopener noreferrer">'
                if url
                else ""
            )
            link_close = "</a>" if url else ""
            pieces.append(
                f'<g class="study-point" data-device="{escape(device)}" '
                f'data-events="{escape(event_data)}" data-url="{escape(url)}" '
                f'data-name="{escape(row.name)}" data-aliases="{escape(str(row.aliases))}" '
                f'data-description="{escape(description)}">'
                f"{link_open}"
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
                f'fill="{color}" fill-opacity="0.68" stroke="#1f2937" stroke-width="1"/>'
                f"{link_close}</g>"
            )
        pieces.extend(["</svg>", _render_legend(), "</div>"])
        return "\n".join(pieces)

    def _render_event_table(self, summaries: pd.DataFrame) -> str:
        if summaries.empty:
            return "<p>No studies with StudyInfo were found.</p>"
        neuro_event_types = sorted(
            {
                event_type
                for event_types_ in summaries.event_types
                for event_type in event_types_
                if event_type in self._NEURO_TYPES
            }
        )
        other_event_counts: dict[str, int] = {}
        for event_types_ in summaries.event_types:
            for event_type in event_types_:
                if event_type in self._NEURO_TYPES:
                    continue
                other_event_counts[event_type] = other_event_counts.get(event_type, 0) + 1
        other_event_types = sorted(
            event_type for event_type, count in other_event_counts.items() if count >= 3
        )
        event_types = neuro_event_types + other_event_types
        if not event_types:
            return "<p>No event types were declared in StudyInfo.</p>"

        def _event_header(event_type: str, class_name: str) -> str:
            return (
                f'<th class="event-col {class_name}" '
                f'data-event="{escape(event_type)}">'
                f'<button type="button">{escape(event_type)}</button></th>'
            )

        header = (
            "<thead>"
            '<tr class="event-super-header">'
            '<th class="study-col" colspan="6"></th>'
            f'<th colspan="{len(neuro_event_types)}">Neuro events</th>'
            + (
                f'<th colspan="{len(other_event_types)}">Other events</th>'
                if other_event_types
                else ""
            )
            + "</tr><tr>"
            '<th class="study-col">Study</th>'
            "<th>Alias</th>"
            "<th>Neuro</th>"
            "<th>Subjects</th>"
            "<th>Timelines</th>"
            "<th>Hours</th>"
            + "".join(
                _event_header(event_type, "neuro-event-col")
                for event_type in neuro_event_types
            )
            + "".join(
                _event_header(event_type, "other-event-col")
                for event_type in other_event_types
            )
            + "</tr></thead>"
        )
        body_rows = []
        for row in summaries.sort_values("name").itertuples():
            row_event_types: list[str] = row.event_types  # type: ignore[assignment]
            device = _primary_neuro_type(str(row.neuro_event_type))
            event_data = _data_list(row_event_types)
            description = _description_attr(str(row.description))
            url = _url_attr(str(row.url))
            study_name = (
                f'<a href="{escape(url)}" target="_blank" rel="noopener noreferrer">'
                f"{escape(row.name)}</a>"
                if url
                else escape(row.name)
            )
            cells = [
                f'<th class="study-col study-name" data-module="{escape(row.module)}" '
                f'data-url="{escape(url)}" data-events="{escape(event_data)}" '
                f'data-aliases="{escape(str(row.aliases))}" '
                f'data-description="{escape(description)}">'
                f"{study_name}</th>",
                f"<td>{escape(str(row.aliases))}</td>",
                f"<td>{escape(str(row.neuro_event_type))}</td>",
                f'<td class="num">{int(row.n_subjects):,}</td>',
                f'<td class="num">{int(row.n_timelines):,}</td>',
                f'<td class="num">{escape(_format_number(float(row.n_hours)))}</td>',
            ]
            cells.extend(
                '<td class="tick">&#10003;</td>'
                if event_type in row_event_types
                else "<td></td>"
                for event_type in neuro_event_types
            )
            cells.extend(
                '<td class="tick">&#10003;</td>'
                if event_type in row_event_types
                else "<td></td>"
                for event_type in other_event_types
            )
            body_rows.append(
                f'<tr class="study-row device-{device.lower()}" '
                f'data-device="{escape(device)}" data-events="{escape(event_data)}" '
                f'data-subjects="{int(row.n_subjects)}" '
                f'data-timelines="{int(row.n_timelines)}" '
                f'data-hours="{float(row.n_hours)}">' + "".join(cells) + "</tr>"
            )
        return (
            '<div class="table-wrap"><table>'
            + header
            + "<tbody>"
            + "\n".join(body_rows)
            + "</tbody></table></div>"
        )

    def _report_title(self) -> str:
        return "Neuralfetch study explorer"

    def _neuro_filter(self) -> set[str] | None:
        if self.neuro_types == "all":
            return None
        if isinstance(self.neuro_types, list):
            return set(self.neuro_types)
        return {self.neuro_types}


NeuralfetchInfoSummaries = StudyInfoSummaries


def build_docs_study_explorer(path: Path) -> Path:
    """Regenerate the NeuralFetch docs study explorer fragment."""
    return StudyInfoSummaries().to_html_fragment(path)


_COLORS = {
    "Eeg": "#2563eb",
    "Meg": "#c026d3",
    "Fmri": "#dc2626",
    "Ieeg": "#f97316",
    "Emg": "#16a34a",
    "Fnirs": "#eab308",
}


def _color_for_neuro_type(neuro_type: str) -> str:
    return _COLORS.get(_primary_neuro_type(neuro_type), "#6b7280")


def _primary_neuro_type(neuro_type: str) -> str:
    if not neuro_type:
        return "Other"
    return neuro_type.split(", ")[0]


def _data_list(values: list[str] | tuple[str, ...]) -> str:
    return "|".join(values)


def _description_attr(description: str) -> str:
    text = " ".join(description.split())
    return text or "No description available."


def _url_attr(url: str) -> str:
    text = url.strip()
    if text.startswith(("http://", "https://")):
        return text
    return ""


def _log_ticks(min_value: float, max_value: float) -> list[float]:
    start = math.floor(math.log10(min_value))
    stop = math.ceil(math.log10(max_value))
    ticks = []
    for exponent in range(start, stop + 1):
        for multiplier in (1, 2, 5):
            value = multiplier * 10**exponent
            if min_value <= value <= max_value:
                ticks.append(float(value))
    return ticks


def _render_legend() -> str:
    items = "".join(
        f'<span><i style="background:{color}"></i>{escape(neuro_type)}</span>'
        for neuro_type, color in _COLORS.items()
    )
    return f'<div class="legend">{items}</div>'


_CSS = """
#neuralfetch-study-explorer {
  color: #111827;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
#neuralfetch-study-explorer h1,
#neuralfetch-study-explorer h2 {
  margin-bottom: 8px;
}
#neuralfetch-study-explorer section {
  margin-top: 28px;
}
#neuralfetch-study-explorer .note {
  color: #4b5563;
}
#neuralfetch-study-explorer .controls {
  align-items: center;
  background: rgba(255, 255, 255, 0.96);
  border: 1px solid #d1d5db;
  border-radius: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  padding: 12px;
  position: sticky;
  top: 0;
  z-index: 10;
}
#neuralfetch-study-explorer .controls label {
  color: #374151;
  font-weight: 600;
}
#neuralfetch-study-explorer .controls select,
#neuralfetch-study-explorer .controls button {
  border: 1px solid #9ca3af;
  border-radius: 6px;
  font: inherit;
  padding: 5px 8px;
}
#neuralfetch-study-explorer .controls button:disabled {
  color: #9ca3af;
}
#neuralfetch-study-explorer #event-filter-status {
  background: #f3f4f6;
  border-radius: 999px;
  color: #374151;
  padding: 6px 10px;
}
#neuralfetch-study-explorer .summary {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  margin-top: 18px;
}
#neuralfetch-study-explorer .summary div {
  background: #f3f4f6;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 14px 16px;
}
#neuralfetch-study-explorer .summary strong {
  display: block;
  font-size: 24px;
}
#neuralfetch-study-explorer .summary span {
  color: #4b5563;
}
#neuralfetch-study-explorer .scatter-wrap,
#neuralfetch-study-explorer .table-wrap {
  border: 1px solid #d1d5db;
  border-radius: 10px;
  overflow: auto;
}
#neuralfetch-study-explorer svg {
  display: block;
  min-width: 1400px;
}
#neuralfetch-study-explorer .grid {
  stroke: #e5e7eb;
  stroke-width: 1;
}
#neuralfetch-study-explorer .axis {
  stroke: #111827;
  stroke-width: 2;
}
#neuralfetch-study-explorer .axis-label {
  fill: #111827;
  font-size: 18px;
  font-weight: 600;
}
#neuralfetch-study-explorer .axis-tick {
  fill: #4b5563;
  font-size: 14px;
}
#neuralfetch-study-explorer .study-point,
#neuralfetch-study-explorer .study-name {
  cursor: help;
}
#neuralfetch-study-explorer .study-name a {
  color: inherit;
  text-decoration: none;
}
#neuralfetch-study-explorer .study-name a:hover {
  text-decoration: underline;
}
#neuralfetch-study-explorer #study-tooltip {
  background: #111827;
  border-radius: 8px;
  color: white;
  font-size: 13px;
  line-height: 1.35;
  max-width: 420px;
  padding: 10px 12px;
  pointer-events: auto;
  position: fixed;
  z-index: 100;
}
#neuralfetch-study-explorer #study-tooltip a {
  color: #93c5fd;
  display: inline-block;
  margin-top: 8px;
}
#neuralfetch-study-explorer .tooltip-title {
  font-size: 16px;
  font-weight: 700;
  margin-bottom: 6px;
}
#neuralfetch-study-explorer .legend {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  padding: 12px;
}
#neuralfetch-study-explorer .legend span {
  align-items: center;
  display: inline-flex;
  gap: 6px;
}
#neuralfetch-study-explorer .legend i {
  border: 1px solid #1f2937;
  border-radius: 999px;
  display: inline-block;
  height: 12px;
  width: 12px;
}
#neuralfetch-study-explorer table {
  border-collapse: separate;
  border-spacing: 0;
  font-size: 13px;
  min-width: 100%;
}
#neuralfetch-study-explorer th,
#neuralfetch-study-explorer td {
  border-bottom: 1px solid #e5e7eb;
  border-right: 1px solid #e5e7eb;
  padding: 7px 9px;
  text-align: center;
  white-space: nowrap;
}
#neuralfetch-study-explorer thead th {
  background: #f9fafb;
  position: sticky;
  top: 0;
  z-index: 2;
}
#neuralfetch-study-explorer .study-col {
  background: #fff;
  left: 0;
  position: sticky;
  text-align: left;
  z-index: 3;
}
#neuralfetch-study-explorer thead .study-col {
  background: #f9fafb;
}
#neuralfetch-study-explorer .event-col {
  writing-mode: vertical-rl;
}
#neuralfetch-study-explorer .event-col button {
  background: transparent;
  border: 0;
  cursor: pointer;
  font: inherit;
  padding: 0;
  writing-mode: vertical-rl;
}
#neuralfetch-study-explorer .event-col.active,
#neuralfetch-study-explorer .event-col.active button {
  background: #fef3c7;
  color: #92400e;
}
#neuralfetch-study-explorer .event-super-header th {
  background: #e5e7eb;
  color: #374151;
  font-weight: 700;
  text-align: center;
}
#neuralfetch-study-explorer .neuro-event-col,
#neuralfetch-study-explorer .neuro-event-col button {
  background: #eef2ff;
}
#neuralfetch-study-explorer .other-event-col,
#neuralfetch-study-explorer .other-event-col button {
  background: #f9fafb;
}
#neuralfetch-study-explorer .num {
  font-variant-numeric: tabular-nums;
  text-align: right;
}
#neuralfetch-study-explorer .tick {
  color: #166534;
  font-weight: 700;
}
#neuralfetch-study-explorer [hidden] {
  display: none !important;
}
#neuralfetch-study-explorer .study-row.device-eeg td,
#neuralfetch-study-explorer .study-row.device-eeg .study-col {
  background: #eff6ff;
}
#neuralfetch-study-explorer .study-row.device-meg td,
#neuralfetch-study-explorer .study-row.device-meg .study-col {
  background: #f5f3ff;
}
#neuralfetch-study-explorer .study-row.device-fmri td,
#neuralfetch-study-explorer .study-row.device-fmri .study-col {
  background: #fef2f2;
}
#neuralfetch-study-explorer .study-row.device-ieeg td,
#neuralfetch-study-explorer .study-row.device-ieeg .study-col {
  background: #fff7ed;
}
#neuralfetch-study-explorer .study-row.device-emg td,
#neuralfetch-study-explorer .study-row.device-emg .study-col {
  background: #f0fdf4;
}
#neuralfetch-study-explorer .study-row.device-fnirs td,
#neuralfetch-study-explorer .study-row.device-fnirs .study-col {
  background: #ecfeff;
}
#neuralfetch-study-explorer .study-row.device-other td,
#neuralfetch-study-explorer .study-row.device-other .study-col {
  background: #f9fafb;
}
""".strip()


_JS = """
(() => {
  const root = document.getElementById("neuralfetch-study-explorer");
  if (!root) {
    return;
  }
  const clearEventFilter = root.querySelector("#clear-event-filter");
  const eventSelect = root.querySelector("#event-filter-select");
  const eventStatus = root.querySelector("#event-filter-status");
  const eventHeaders = Array.from(root.querySelectorAll("th.event-col"));
  const tooltip = root.querySelector("#study-tooltip");
  if (!clearEventFilter || !eventSelect || !eventStatus || !tooltip) {
    return;
  }
  const activeEvents = new Set();
  let tooltipHideTimer = null;

  function hasEvent(element, eventName) {
    return (element.dataset.events || "").split("|").includes(eventName);
  }

  function matchesFilters(element) {
    const eventMatches = Array.from(activeEvents).every((eventName) => hasEvent(element, eventName));
    return eventMatches;
  }

  function formatNumber(value) {
    if (!Number.isFinite(value)) {
      return "";
    }
    if (value >= 1000000) {
      return `${(value / 1000000).toFixed(1)}M`;
    }
    if (value >= 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    if (value >= 100) {
      return value.toFixed(0);
    }
    if (value >= 10) {
      return value.toFixed(1);
    }
    return value.toFixed(2);
  }

  function updateSummary() {
    let studies = 0;
    let subjects = 0;
    let timelines = 0;
    let hours = 0;
    root.querySelectorAll(".study-row").forEach((row) => {
      if (row.hasAttribute("hidden")) {
        return;
      }
      studies += 1;
      subjects += Number(row.dataset.subjects || 0);
      timelines += Number(row.dataset.timelines || 0);
      const rowHours = Number(row.dataset.hours || 0);
      if (Number.isFinite(rowHours)) {
        hours += rowHours;
      }
    });
    root.querySelector("#summary-studies").textContent = studies.toLocaleString();
    root.querySelector("#summary-subjects").textContent = subjects.toLocaleString();
    root.querySelector("#summary-timelines").textContent = timelines.toLocaleString();
    root.querySelector("#summary-hours").textContent = formatNumber(hours);
  }

  function updateEventSelect() {
    Array.from(eventSelect.options).forEach((option) => {
      if (option.value !== "") {
        option.disabled = activeEvents.has(option.value);
      }
    });
    eventSelect.value = "";
  }

  function applyFilters() {
    root.querySelectorAll(".study-row, .study-point").forEach((element) => {
      if (matchesFilters(element)) {
        element.removeAttribute("hidden");
      } else {
        element.setAttribute("hidden", "");
      }
    });
    eventHeaders.forEach((header) => {
      header.classList.toggle("active", activeEvents.has(header.dataset.event || ""));
    });
    eventStatus.textContent = activeEvents.size
      ? `Events: ${Array.from(activeEvents).join(", ")}`
      : "All event types";
    clearEventFilter.disabled = activeEvents.size === 0;
    updateEventSelect();
    updateSummary();
  }

  function moveTooltip(event) {
    const margin = 14;
    const rect = tooltip.getBoundingClientRect();
    let left = event.clientX + margin;
    let top = event.clientY + margin;
    if (left + rect.width > window.innerWidth) {
      left = event.clientX - rect.width - margin;
    }
    if (top + rect.height > window.innerHeight) {
      top = event.clientY - rect.height - margin;
    }
    tooltip.style.left = `${Math.max(margin, left)}px`;
    tooltip.style.top = `${Math.max(margin, top)}px`;
  }

  function showTooltip(event) {
    const name = event.currentTarget.dataset.name || "";
    const aliases = event.currentTarget.dataset.aliases || "";
    const description = event.currentTarget.dataset.description || "No description available.";
    const eventTypes = (event.currentTarget.dataset.events || "").split("|").filter(Boolean);
    const url = event.currentTarget.dataset.url || "";
    clearTimeout(tooltipHideTimer);
    tooltip.innerHTML = "";
    if (name !== "") {
      const title = document.createElement("div");
      title.className = "tooltip-title";
      title.textContent = name;
      tooltip.appendChild(title);
    }
    const text = document.createElement("div");
    text.textContent = description;
    tooltip.appendChild(text);
    if (aliases !== "") {
      const aliasText = document.createElement("div");
      aliasText.textContent = `Aliases: ${aliases}`;
      tooltip.appendChild(aliasText);
    }
    if (eventTypes.length > 0) {
      const events = document.createElement("div");
      events.textContent = `Event types: ${eventTypes.join(", ")}`;
      tooltip.appendChild(events);
    }
    if (url !== "") {
      const link = document.createElement("a");
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = url;
      tooltip.appendChild(link);
    }
    tooltip.removeAttribute("hidden");
    moveTooltip(event);
  }

  function hideTooltip() {
    tooltipHideTimer = setTimeout(() => {
      tooltip.setAttribute("hidden", "");
    }, 120);
  }

  clearEventFilter.addEventListener("click", () => {
    activeEvents.clear();
    applyFilters();
  });
  eventSelect.addEventListener("change", () => {
    const eventName = eventSelect.value;
    if (eventName !== "") {
      activeEvents.add(eventName);
      applyFilters();
    }
  });
  eventHeaders.forEach((header) => {
    header.addEventListener("click", () => {
      const eventName = header.dataset.event || "";
      if (activeEvents.has(eventName)) {
        activeEvents.delete(eventName);
      } else {
        activeEvents.add(eventName);
      }
      applyFilters();
    });
  });
  root.querySelectorAll("[data-description]").forEach((element) => {
    element.addEventListener("mouseenter", showTooltip);
    element.addEventListener("mousemove", moveTooltip);
    element.addEventListener("mouseleave", hideTooltip);
  });
  tooltip.addEventListener("mouseenter", () => {
    clearTimeout(tooltipHideTimer);
  });
  tooltip.addEventListener("mouseleave", hideTooltip);
  applyFilters();
})();
""".strip()


if __name__ == "__main__":
    docs_root = Path(__file__).resolve().parents[1]
    report_path = build_docs_study_explorer(
        docs_root / "neuralfetch" / "_explore_studies.html"
    )
    print(f"Saved HTML fragment to {report_path}")
