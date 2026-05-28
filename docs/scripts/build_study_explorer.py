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
    "requirements",
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
                    "requirements": tuple(getattr(cls, "requirements", ()) or ()),
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
                '<section class="explorer-tabs-section">',
                '<div class="explorer-tabs" role="tablist" aria-label="Explorer views">',
                '<button type="button" class="explorer-tab active" role="tab" '
                'aria-selected="true" data-tab="plot" id="tab-plot" '
                'aria-controls="panel-plot">'
                '<i class="fas fa-chart-area" aria-hidden="true"></i>'
                "&nbsp;Plot</button>",
                '<button type="button" class="explorer-tab" role="tab" '
                'aria-selected="false" data-tab="table" id="tab-table" '
                'aria-controls="panel-table" tabindex="-1">'
                '<i class="fas fa-table" aria-hidden="true"></i>'
                "&nbsp;Table</button>",
                "</div>",
                '<div class="explorer-tab-panel" role="tabpanel" id="panel-plot" '
                'aria-labelledby="tab-plot" data-panel="plot">',
                self._render_scatter(summaries),
                "</div>",
                '<div class="explorer-tab-panel" role="tabpanel" id="panel-table" '
                'aria-labelledby="tab-table" data-panel="table" hidden>',
                '<p class="note">'
                '<i class="fas fa-arrows-left-right" aria-hidden="true"></i>'
                "&nbsp;Scroll horizontally to see every event column &middot;"
                '&nbsp;<i class="fas fa-mouse-pointer" aria-hidden="true"></i>'
                "&nbsp;<strong>click</strong> a column header to toggle a filter "
                "(<strong>double-click</strong> to solo) &middot; "
                "click a study name for details."
                "</p>",
                self._render_event_table(summaries),
                "</div>",
                self._render_details_panel(),
                "</section>",
                "</div>",
            ]
        )
        return "\n".join(pieces)

    def _render_details_panel(self) -> str:
        return (
            '<div id="study-details" class="study-details" hidden '
            'aria-live="polite" aria-labelledby="study-details-name">'
            '<header class="study-details-header">'
            "<div>"
            '<strong id="study-details-name" class="study-details-name"></strong>'
            '<span class="study-details-device-badge" hidden></span>'
            "</div>"
            '<button type="button" class="study-details-close" aria-label="Close details" '
            'data-details-close="1">&times;</button>'
            "</header>"
            '<p class="study-details-aliases" hidden></p>'
            '<p class="study-details-description"></p>'
            '<dl class="study-details-stats" aria-label="Study statistics"></dl>'
            '<div class="study-details-events" hidden>'
            '<span class="study-details-events-label">Event types:</span>'
            '<span class="study-details-events-list"></span>'
            "</div>"
            '<div class="study-details-snippets">'
            '<div class="study-details-snippet-wrap">'
            '<div class="study-details-snippet-label">'
            '<i class="fas fa-terminal" aria-hidden="true"></i>'
            "&nbsp;Install dependencies</div>"
            '<pre class="study-details-snippet study-details-snippet--bash">'
            "<code></code></pre>"
            "</div>"
            '<div class="study-details-snippet-wrap">'
            '<div class="study-details-snippet-label">'
            '<i class="fab fa-python" aria-hidden="true"></i>'
            "&nbsp;Python</div>"
            '<pre class="study-details-snippet study-details-snippet--python">'
            "<code></code></pre>"
            "</div>"
            "</div>"
            '<div class="study-details-actions">'
            '<a class="study-details-btn study-details-open" '
            'target="_blank" rel="noopener noreferrer" hidden>'
            '<i class="fas fa-external-link-alt" aria-hidden="true"></i>'
            "&nbsp;Original study</a>"
            '<a class="study-details-btn study-details-github" '
            'target="_blank" rel="noopener noreferrer" hidden>'
            '<i class="fab fa-github" aria-hidden="true"></i>'
            "&nbsp;Curation code</a>"
            "</div>"
            "</div>"
        )

    def _render_controls(self, summaries: pd.DataFrame) -> str:
        device_chips: list[str] = []
        event_chips: list[str] = []
        for event_type in self._event_filter_options(summaries):
            if event_type in self._NEURO_TYPES:
                color = _COLORS.get(event_type, "#6b7280")
                device_chips.append(
                    f'<button type="button" class="event-chip event-chip--device" '
                    f'data-event="{escape(event_type)}" data-category="device" '
                    f'style="--chip-color: {color}">{escape(event_type)}</button>'
                )
            else:
                event_chips.append(
                    f'<button type="button" class="event-chip event-chip--event" '
                    f'data-event="{escape(event_type)}" data-category="event">'
                    f"{escape(event_type)}</button>"
                )
        # Append the synthetic Multimodal chip — studies whose neuro_event_type
        # lists more than one modality are tagged with this pseudo-event in
        # `data-events`, so the chip filters them just like a real event chip.
        if any("," in str(row.neuro_event_type) for row in summaries.itertuples()):
            device_chips.append(
                '<button type="button" class="event-chip event-chip--device" '
                'data-event="Multimodal" data-category="device" '
                f'style="--chip-color: {_COLORS["Multimodal"]}">Multimodal</button>'
            )

        rows: list[str] = []
        if device_chips:
            rows.append(
                '<div class="controls-row">'
                '<span class="controls-label">Devices</span>'
                '<div class="event-chips" role="group" aria-label="Filter by recording device">'
                + "".join(device_chips)
                + "</div>"
                "</div>"
            )
        if event_chips:
            rows.append(
                '<div class="controls-row">'
                '<span class="controls-label">Events</span>'
                '<div class="event-chips" role="group" aria-label="Filter by event type">'
                + "".join(event_chips)
                + "</div>"
                "</div>"
            )
        return '<section class="controls">' + "".join(rows) + "</section>"

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
        rows = rows[rows["n_subjects"] > 0]
        if rows.empty:
            return "<p>No studies have enough StudyInfo metadata to estimate volume.</p>"

        # Clip 0-hour studies (synthetic / placeholder examples like
        # ExampleMultiModal) to a small positive floor so they still show up
        # on the log axis at the left edge instead of being silently dropped.
        _HOURS_FLOOR = 0.01
        rows["n_hours_plot"] = rows["n_hours"].clip(lower=_HOURS_FLOOR)
        rows["hours_per_subject"] = rows["n_hours_plot"] / rows["n_subjects"]
        rows = rows[rows["hours_per_subject"] > 0]
        if rows.empty:
            return "<p>No studies have positive hours per subject.</p>"

        width = 900
        height = 520
        pad_left = 60
        pad_right = 24
        pad_top = 32
        pad_bottom = 60
        plot_width = width - pad_left - pad_right
        plot_height = height - pad_top - pad_bottom
        x_min = float(rows["hours_per_subject"].min())
        x_max = float(rows["hours_per_subject"].max())
        y_min = float(rows["n_subjects"].min())
        y_max = float(rows["n_subjects"].max())
        h_min = float(rows["n_hours_plot"].min())
        h_max = float(rows["n_hours_plot"].max())

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
        ]
        for tick in x_ticks:
            x = x_pos(tick)
            pieces.append(
                f'<line class="grid" x1="{x:.1f}" y1="{pad_top}" '
                f'x2="{x:.1f}" y2="{pad_top + plot_height}"/>'
            )
            pieces.append(
                f'<text class="axis-tick" x="{x:.1f}" y="{height - 32}" '
                f'text-anchor="middle">{escape(_format_number(tick))}</text>'
            )
        for tick in y_ticks:
            y = y_pos(tick)
            pieces.append(
                f'<line class="grid" x1="{pad_left}" y1="{y:.1f}" '
                f'x2="{pad_left + plot_width}" y2="{y:.1f}"/>'
            )
            pieces.append(
                f'<text class="axis-tick" x="{pad_left - 8}" y="{y + 3:.1f}" '
                f'text-anchor="end">{escape(_format_number(tick))}</text>'
            )
        pieces.extend(
            [
                f'<line class="axis" x1="{pad_left}" y1="{pad_top + plot_height}" '
                f'x2="{pad_left + plot_width}" y2="{pad_top + plot_height}"/>',
                f'<line class="axis" x1="{pad_left}" y1="{pad_top}" '
                f'x2="{pad_left}" y2="{pad_top + plot_height}"/>',
                f'<text class="axis-label" x="{pad_left + plot_width / 2:.1f}" '
                f'y="{height - 10}" text-anchor="middle">Estimated hours per subject</text>',
                f'<text class="axis-label" x="14" y="{pad_top + plot_height / 2:.1f}" '
                'text-anchor="middle" transform="rotate(-90 14 '
                f'{pad_top + plot_height / 2:.1f})">Number of subjects</text>',
            ]
        )

        for row in rows.sort_values("name").itertuples():
            x = x_pos(float(row.hours_per_subject))
            y = y_pos(float(row.n_subjects))
            r = radius(float(row.n_hours_plot))
            color = _color_for_neuro_type(str(row.neuro_event_type))
            device = _primary_neuro_type(str(row.neuro_event_type))
            event_data = _data_list(_events_for_filter(row))  # type: ignore[arg-type]
            description = _description_attr(str(row.description))
            url = _url_attr(str(row.url))
            pieces.append(
                f'<g class="study-point" data-action="study" '
                f'tabindex="0" role="button" '
                f'data-device="{escape(device)}" '
                f'data-neuro="{escape(str(row.neuro_event_type))}" '
                f'data-module="{escape(str(row.module))}" '
                f'data-events="{escape(event_data)}" data-url="{escape(url)}" '
                f'data-name="{escape(row.name)}" data-aliases="{escape(str(row.aliases))}" '
                f'data-description="{escape(description)}" '
                f'data-subjects="{int(row.n_subjects)}" '
                f'data-timelines="{int(row.n_timelines)}" '
                f'data-hours="{float(row.n_hours)}" '
                f'data-frequency="{float(row.frequency) if pd.notna(row.frequency) else 0}" '
                f'data-channels="{_channel_count(row.data_shape)}" '
                f'data-requirements="{escape(_data_list(row.requirements))}">'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
                f'fill="{color}" fill-opacity="0.68" stroke-width="1"/>'
                "</g>"
            )
        pieces.extend(["</svg>", "</div>"])
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

        def _event_header(event_type: str, class_name: str, category: str) -> str:
            style = ""
            if category == "device":
                color = _COLORS.get(event_type, "#6b7280")
                style = f' style="--chip-color: {color}"'
            title = f"Click to toggle {event_type} filter (double-click to solo)"
            return (
                f'<th class="event-col {class_name}" '
                f'data-event="{escape(event_type)}" data-category="{category}"{style} '
                f'title="{escape(title)}">'
                f'<button type="button" tabindex="-1">{escape(event_type)}</button></th>'
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
                _event_header(event_type, "neuro-event-col", "device")
                for event_type in neuro_event_types
            )
            + "".join(
                _event_header(event_type, "other-event-col", "event")
                for event_type in other_event_types
            )
            + "</tr></thead>"
        )
        body_rows = []
        for row in summaries.sort_values("name").itertuples():
            row_event_types: list[str] = row.event_types  # type: ignore[assignment]
            device = _primary_neuro_type(str(row.neuro_event_type))
            event_data = _data_list(_events_for_filter(row))
            description = _description_attr(str(row.description))
            url = _url_attr(str(row.url))
            cells = [
                f'<th class="study-col study-name" data-action="study" '
                f'data-name="{escape(row.name)}" '
                f'data-module="{escape(row.module)}" '
                f'data-device="{escape(device)}" '
                f'data-neuro="{escape(str(row.neuro_event_type))}" '
                f'data-url="{escape(url)}" data-events="{escape(event_data)}" '
                f'data-aliases="{escape(str(row.aliases))}" '
                f'data-description="{escape(description)}" '
                f'data-subjects="{int(row.n_subjects)}" '
                f'data-timelines="{int(row.n_timelines)}" '
                f'data-hours="{float(row.n_hours)}" '
                f'data-frequency="{float(row.frequency) if pd.notna(row.frequency) else 0}" '
                f'data-channels="{_channel_count(row.data_shape)}" '
                f'data-requirements="{escape(_data_list(row.requirements))}">'
                f'<button type="button" class="study-link">{escape(row.name)}</button>'
                "</th>",
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
    # Studies that declare more than one neuro modality (e.g. MOABB2025) get
    # their own black bubble so they don't get silently coloured as their
    # first-listed modality and blend in with the single-modality cohort.
    "Multimodal": "#111827",
}


def _color_for_neuro_type(neuro_type: str) -> str:
    return _COLORS.get(_primary_neuro_type(neuro_type), "#6b7280")


def _primary_neuro_type(neuro_type: str) -> str:
    if not neuro_type:
        return "Other"
    types = [t for t in (s.strip() for s in neuro_type.split(",")) if t]
    if len(types) > 1:
        return "Multimodal"
    return types[0] if types else "Other"


def _data_list(values: list[str] | tuple[str, ...]) -> str:
    return "|".join(values)


def _events_for_filter(row: tp.Any) -> list[str]:
    """Event-type list used for chip / column-header filtering.

    Mirrors ``row.event_types`` but appends the pseudo-event ``"Multimodal"``
    for studies that declare more than one neuro modality, so the
    Multimodal device chip naturally filters them via the same
    ``some(event in active)`` check as every other chip.
    """
    events = list(row.event_types)
    if "," in str(row.neuro_event_type):
        events.append("Multimodal")
    return events


def _channel_count(data_shape: tp.Any) -> int:
    """Best-effort channel count from a ``StudyInfo.data_shape`` tuple.

    ``data_shape`` is typically ``(n_channels, n_samples)`` for neuro
    extractors. Return 0 when it's missing or malformed so the popover
    can hide the row.
    """
    if not data_shape:
        return 0
    try:
        if len(data_shape) >= 2:
            return int(data_shape[0])
        if len(data_shape) == 1:
            return int(data_shape[0])
    except (TypeError, ValueError):
        return 0
    return 0


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


_CSS = """
#neuralfetch-study-explorer {
  --nse-eeg: #2563eb;
  --nse-meg: #c026d3;
  --nse-fmri: #dc2626;
  --nse-ieeg: #f97316;
  --nse-emg: #16a34a;
  --nse-fnirs: #eab308;
  --nse-other: #6b7280;
  --nse-border: var(--color-background-border, #d1d5db);
  --nse-panel: var(--color-background-secondary, #f8fafc);
  --nse-text: var(--color-foreground-primary, #111827);
  --nse-muted: var(--color-foreground-muted, #4b5563);
  --nse-brand: var(--color-brand-primary, #448aff);
  --nse-tile: color-mix(in srgb, currentColor 5%, transparent);
  color: var(--nse-text);
}
#neuralfetch-study-explorer h1,
#neuralfetch-study-explorer h2 {
  margin-bottom: 8px;
}
#neuralfetch-study-explorer section {
  margin-top: 28px;
}
#neuralfetch-study-explorer .note {
  color: var(--nse-muted);
}
#neuralfetch-study-explorer .controls {
  background: var(--nse-panel);
  border: 1px solid var(--nse-border);
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 12px;
}
#neuralfetch-study-explorer .controls-row {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}
#neuralfetch-study-explorer .controls-label {
  color: var(--nse-text);
  flex: 0 0 64px;
  font-size: 0.78em;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
#neuralfetch-study-explorer .event-chips {
  display: flex;
  flex: 1 1 auto;
  flex-wrap: wrap;
  gap: 6px;
}
#neuralfetch-study-explorer .event-chip {
  --chip-color: var(--nse-brand);
  background: transparent;
  border: 1px solid var(--nse-border);
  border-radius: 999px;
  color: var(--nse-text);
  cursor: pointer;
  font: inherit;
  font-size: 0.85em;
  padding: 3px 10px;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
}
#neuralfetch-study-explorer .event-chip--device {
  border-color: color-mix(in srgb, var(--chip-color) 40%, var(--nse-border));
}
#neuralfetch-study-explorer .event-chip:hover {
  border-color: var(--chip-color);
  color: var(--chip-color);
}
#neuralfetch-study-explorer .event-chip.active {
  background: var(--chip-color);
  border-color: var(--chip-color);
  color: #fff;
}
#neuralfetch-study-explorer .event-chip:not(.active) {
  background: transparent;
  color: var(--nse-muted);
  opacity: 0.5;
}
#neuralfetch-study-explorer .event-chip:not(.active):hover {
  opacity: 0.95;
}
/* Tabbed explorer: shared section for the scatter and the table. */
#neuralfetch-study-explorer .explorer-tabs-section {
  margin-top: 22px;
}
#neuralfetch-study-explorer .explorer-tabs {
  border-bottom: 1px solid var(--nse-border);
  display: flex;
  gap: 4px;
  margin-bottom: 14px;
}
#neuralfetch-study-explorer .explorer-tab {
  background: transparent;
  border: 0;
  border-bottom: 2px solid transparent;
  color: var(--nse-muted);
  cursor: pointer;
  font: inherit;
  font-size: 0.95em;
  font-weight: 600;
  margin-bottom: -1px;
  padding: 8px 14px;
  transition: color 0.12s, border-color 0.12s;
}
#neuralfetch-study-explorer .explorer-tab:hover:not(.active) {
  color: var(--nse-text);
}
#neuralfetch-study-explorer .explorer-tab.active {
  border-bottom-color: var(--nse-brand);
  color: var(--nse-brand);
}
#neuralfetch-study-explorer .explorer-tab:focus-visible {
  outline: 2px solid var(--nse-brand);
  outline-offset: 2px;
}
#neuralfetch-study-explorer .explorer-tab-panel[hidden] {
  display: none !important;
}
#neuralfetch-study-explorer .summary {
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  margin-top: 14px;
}
#neuralfetch-study-explorer .summary div {
  background: var(--nse-panel);
  border: 1px solid var(--nse-border);
  border-radius: 10px;
  padding: 10px 14px;
}
#neuralfetch-study-explorer .summary strong {
  display: block;
  font-size: 20px;
}
#neuralfetch-study-explorer .summary span {
  color: var(--nse-muted);
  font-size: 0.85em;
}
#neuralfetch-study-explorer .scatter-wrap,
#neuralfetch-study-explorer .table-wrap {
  border: 1px solid var(--nse-border);
  border-radius: 10px;
  overflow: auto;
}
/* Horizontal scroll shadows on the events table: the two outer
   radial-gradients fade in/out with `background-attachment: scroll` to
   signal that the table extends past the visible edge. */
#neuralfetch-study-explorer .table-wrap {
  background:
    radial-gradient(
      farthest-side at 0 50%,
      color-mix(in srgb, var(--nse-text) 14%, transparent),
      transparent
    ),
    radial-gradient(
      farthest-side at 100% 50%,
      color-mix(in srgb, var(--nse-text) 14%, transparent),
      transparent
    ) right;
  background-attachment: scroll, scroll;
  background-color: var(--color-background-primary, #fff);
  background-repeat: no-repeat;
  background-size: 16px 100%, 16px 100%;
}
#neuralfetch-study-explorer .scatter-wrap {
  background: var(--nse-panel);
}
#neuralfetch-study-explorer .scatter-wrap svg {
  display: block;
  height: auto;
  margin: 0 auto;
  max-width: 900px;
  width: 100%;
}
#neuralfetch-study-explorer .grid {
  stroke: color-mix(in srgb, var(--nse-text) 12%, transparent);
  stroke-width: 1;
}
#neuralfetch-study-explorer .axis {
  stroke: color-mix(in srgb, var(--nse-text) 55%, transparent);
  stroke-width: 1.5;
}
#neuralfetch-study-explorer .axis-label {
  fill: var(--nse-text);
  font-size: 13px;
  font-weight: 600;
}
#neuralfetch-study-explorer .axis-tick {
  fill: var(--nse-muted);
  font-size: 11px;
}
#neuralfetch-study-explorer .study-point {
  cursor: pointer;
}
#neuralfetch-study-explorer .study-point:focus {
  outline: none;
}
#neuralfetch-study-explorer .study-point circle {
  stroke: color-mix(in srgb, var(--nse-text) 65%, transparent);
  transition: stroke-width 0.12s, fill-opacity 0.12s;
}
#neuralfetch-study-explorer .study-point:hover circle,
#neuralfetch-study-explorer .study-point:focus circle {
  fill-opacity: 0.9;
  stroke-width: 2;
}
#neuralfetch-study-explorer .study-link {
  background: transparent;
  border: 0;
  color: inherit;
  cursor: pointer;
  font: inherit;
  padding: 0;
  text-align: left;
}
#neuralfetch-study-explorer .study-link:hover,
#neuralfetch-study-explorer .study-link:focus-visible {
  color: var(--nse-brand);
  text-decoration: underline;
}
#neuralfetch-study-explorer #study-tooltip {
  background: var(--color-background-primary, #fff);
  border: 1px solid var(--nse-border);
  border-radius: 8px;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.18);
  color: var(--nse-text);
  font-size: 12px;
  line-height: 1.4;
  max-width: 360px;
  padding: 10px 12px;
  pointer-events: none;
  position: fixed;
  z-index: 100;
}
#neuralfetch-study-explorer .tooltip-title {
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 4px;
}
#neuralfetch-study-explorer table {
  border-collapse: separate;
  border-spacing: 0;
  font-size: 13px;
  min-width: 100%;
}
#neuralfetch-study-explorer th,
#neuralfetch-study-explorer td {
  border-bottom: 1px solid var(--nse-border);
  border-right: 1px solid var(--nse-border);
  padding: 7px 9px;
  text-align: center;
  white-space: nowrap;
}
#neuralfetch-study-explorer thead th {
  background: var(--nse-panel);
  color: var(--nse-text);
  position: sticky;
  top: 0;
  z-index: 2;
}
#neuralfetch-study-explorer .study-col {
  background: var(--color-background-primary, #fff);
  left: 0;
  position: sticky;
  text-align: left;
  z-index: 3;
}
#neuralfetch-study-explorer thead .study-col {
  background: var(--nse-panel);
}
#neuralfetch-study-explorer .event-col {
  cursor: pointer;
  user-select: none;
  writing-mode: vertical-rl;
}
#neuralfetch-study-explorer .event-col:hover {
  background: color-mix(in srgb, var(--chip-color, var(--nse-brand)) 30%, transparent);
}
#neuralfetch-study-explorer .event-col button {
  background: transparent;
  border: 0;
  color: inherit;
  cursor: pointer;
  font: inherit;
  padding: 0;
  pointer-events: none;
  writing-mode: vertical-rl;
}
#neuralfetch-study-explorer .event-col:not(.active) button {
  opacity: 0.5;
}
#neuralfetch-study-explorer .event-col:not(.active):hover button {
  opacity: 0.95;
}
#neuralfetch-study-explorer .event-col {
  --chip-color: var(--nse-brand);
}
#neuralfetch-study-explorer .event-col.active,
#neuralfetch-study-explorer .event-col.active button {
  background: color-mix(in srgb, var(--chip-color) 25%, transparent);
  color: var(--nse-text);
}
#neuralfetch-study-explorer .event-super-header th {
  background: color-mix(in srgb, var(--nse-text) 8%, transparent);
  color: var(--nse-muted);
  font-weight: 700;
  text-align: center;
}
#neuralfetch-study-explorer .neuro-event-col,
#neuralfetch-study-explorer .other-event-col {
  background: var(--nse-panel);
}
#neuralfetch-study-explorer .num {
  font-variant-numeric: tabular-nums;
  text-align: right;
}
#neuralfetch-study-explorer .tick {
  color: color-mix(in srgb, var(--nse-emg) 80%, var(--nse-text));
  font-weight: 700;
}
#neuralfetch-study-explorer [hidden] {
  display: none !important;
}
/* Row backgrounds are opaque mixes into the page background so the
   left-sticky study-name column doesn't bleed through when the table
   scrolls horizontally. */
#neuralfetch-study-explorer .study-row.device-eeg td,
#neuralfetch-study-explorer .study-row.device-eeg .study-col {
  background: color-mix(in srgb, var(--nse-eeg) 8%, var(--color-background-primary, #fff));
}
#neuralfetch-study-explorer .study-row.device-meg td,
#neuralfetch-study-explorer .study-row.device-meg .study-col {
  background: color-mix(in srgb, var(--nse-meg) 8%, var(--color-background-primary, #fff));
}
#neuralfetch-study-explorer .study-row.device-fmri td,
#neuralfetch-study-explorer .study-row.device-fmri .study-col {
  background: color-mix(in srgb, var(--nse-fmri) 8%, var(--color-background-primary, #fff));
}
#neuralfetch-study-explorer .study-row.device-ieeg td,
#neuralfetch-study-explorer .study-row.device-ieeg .study-col {
  background: color-mix(in srgb, var(--nse-ieeg) 8%, var(--color-background-primary, #fff));
}
#neuralfetch-study-explorer .study-row.device-emg td,
#neuralfetch-study-explorer .study-row.device-emg .study-col {
  background: color-mix(in srgb, var(--nse-emg) 8%, var(--color-background-primary, #fff));
}
#neuralfetch-study-explorer .study-row.device-fnirs td,
#neuralfetch-study-explorer .study-row.device-fnirs .study-col {
  background: color-mix(in srgb, var(--nse-fnirs) 10%, var(--color-background-primary, #fff));
}
#neuralfetch-study-explorer .study-row.device-multimodal td,
#neuralfetch-study-explorer .study-row.device-multimodal .study-col {
  background: color-mix(in srgb, #111827 8%, var(--color-background-primary, #fff));
}
#neuralfetch-study-explorer .study-row.device-other td,
#neuralfetch-study-explorer .study-row.device-other .study-col {
  background: var(--color-background-primary, #fff);
}
/* Inline study-details panel — shown below the active tab when a study
   is clicked. Same panel is used for plot circles and table rows. */
#neuralfetch-study-explorer .study-details {
  background: var(--color-background-primary, #fff);
  border: 1px solid var(--nse-border);
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
  color: var(--nse-text);
  margin-top: 18px;
  padding: 18px 22px 18px;
}
#neuralfetch-study-explorer .study-details-header {
  align-items: flex-start;
  display: flex;
  gap: 12px;
  justify-content: space-between;
  margin-bottom: 6px;
}
#neuralfetch-study-explorer .study-details-header > div {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
#neuralfetch-study-explorer .study-details-name {
  font-size: 1.15em;
  font-weight: 700;
  word-break: break-word;
}
#neuralfetch-study-explorer .study-details-device-badge {
  border-radius: 999px;
  color: #fff;
  font-size: 0.72em;
  font-weight: 700;
  letter-spacing: 0.04em;
  padding: 2px 8px;
  text-transform: uppercase;
}
#neuralfetch-study-explorer .study-details-close {
  background: transparent;
  border: 0;
  color: var(--nse-muted);
  cursor: pointer;
  font-size: 22px;
  line-height: 1;
  margin-top: -2px;
  padding: 0 4px;
}
#neuralfetch-study-explorer .study-details-close:hover {
  color: var(--nse-text);
}
#neuralfetch-study-explorer .study-details-aliases {
  color: var(--nse-muted);
  font-size: 0.85em;
  font-style: italic;
  margin: 0 0 8px;
}
#neuralfetch-study-explorer .study-details-description {
  color: var(--nse-text);
  font-size: 0.92em;
  line-height: 1.45;
  margin: 0 0 14px;
}
#neuralfetch-study-explorer .study-details-stats {
  background: var(--nse-panel);
  border: 1px solid var(--nse-border);
  border-radius: 8px;
  display: grid;
  gap: 4px 16px;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  margin: 0 0 12px;
  padding: 10px 14px;
}
#neuralfetch-study-explorer .study-details-stats > div {
  align-items: baseline;
  display: flex;
  gap: 8px;
  justify-content: space-between;
  padding: 4px 0;
}
#neuralfetch-study-explorer .study-details-stats dt {
  color: var(--nse-muted);
  font-size: 0.78em;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}
#neuralfetch-study-explorer .study-details-stats dd {
  color: var(--nse-text);
  font-size: 0.95em;
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  margin: 0;
  text-align: right;
}
#neuralfetch-study-explorer .study-details-events {
  font-size: 0.85em;
  margin-bottom: 14px;
}
#neuralfetch-study-explorer .study-details-events-label {
  color: var(--nse-muted);
  font-weight: 600;
  margin-right: 6px;
}
#neuralfetch-study-explorer .study-details-events-list .event-pill {
  background: color-mix(in srgb, var(--nse-brand) 10%, transparent);
  border-radius: 999px;
  color: var(--nse-text);
  display: inline-block;
  font-size: 0.95em;
  margin: 2px 4px 2px 0;
  padding: 1px 8px;
}
#neuralfetch-study-explorer .study-details-snippets {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 14px;
}
#neuralfetch-study-explorer .study-details-snippet-wrap {
  background: var(--nse-panel);
  border: 1px solid var(--nse-border);
  border-radius: 8px;
  padding: 8px 12px 10px;
}
#neuralfetch-study-explorer .study-details-snippet-label {
  color: var(--nse-muted);
  font-size: 0.72em;
  font-weight: 700;
  letter-spacing: 0.05em;
  margin-bottom: 4px;
  text-transform: uppercase;
}
#neuralfetch-study-explorer .study-details-snippet {
  background: transparent;
  border: 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.82em;
  line-height: 1.5;
  margin: 0;
  overflow-x: auto;
  padding: 0;
  white-space: pre;
}
#neuralfetch-study-explorer .study-details-snippet code {
  background: transparent;
  color: var(--nse-text);
  padding: 0;
}
#neuralfetch-study-explorer .study-details-actions {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: flex-end;
}
#neuralfetch-study-explorer .study-details-btn {
  align-items: center;
  background: transparent;
  border: 1px solid var(--nse-border);
  border-radius: 6px;
  color: var(--nse-text);
  cursor: pointer;
  display: inline-flex;
  font: inherit;
  font-size: 0.9em;
  font-weight: 600;
  gap: 4px;
  padding: 7px 14px;
  text-decoration: none;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
}
#neuralfetch-study-explorer .study-details-btn:hover {
  background: color-mix(in srgb, var(--nse-brand) 10%, transparent);
  border-color: var(--nse-brand);
  color: var(--nse-brand);
}
#neuralfetch-study-explorer .study-details-open {
  background: var(--nse-brand);
  border-color: var(--nse-brand);
  color: #fff;
}
#neuralfetch-study-explorer .study-details-open:hover {
  background: var(--nse-muted);
  border-color: var(--nse-muted);
  color: #fff;
  filter: none;
}
""".strip()


_JS = """
(() => {
  const root = document.getElementById("neuralfetch-study-explorer");
  if (!root) {
    return;
  }
  const chips = Array.from(root.querySelectorAll(".event-chip"));
  const tabs = Array.from(root.querySelectorAll(".explorer-tab"));
  const panels = Array.from(root.querySelectorAll(".explorer-tab-panel"));
  const eventHeaders = Array.from(root.querySelectorAll("th.event-col"));
  const tooltip = root.querySelector("#study-tooltip");
  const details = root.querySelector("#study-details");
  if (!details || !tooltip) {
    return;
  }
  const detailsName = details.querySelector(".study-details-name");
  const detailsDeviceBadge = details.querySelector(".study-details-device-badge");
  const detailsAliases = details.querySelector(".study-details-aliases");
  const detailsDescription = details.querySelector(".study-details-description");
  const detailsStats = details.querySelector(".study-details-stats");
  const detailsEvents = details.querySelector(".study-details-events");
  const detailsEventsList = details.querySelector(".study-details-events-list");
  const detailsBashCode = details.querySelector(
    ".study-details-snippet--bash code"
  );
  const detailsPyCode = details.querySelector(
    ".study-details-snippet--python code"
  );
  const detailsOpen = details.querySelector(".study-details-open");
  const detailsGithub = details.querySelector(".study-details-github");
  // Two filter sets — devices (neuro types) and events (everything else).
  // Within each set we OR; across sets we AND. So "Eeg + Fmri + Sentence"
  // means (device in {Eeg, Fmri}) AND (events include Sentence).
  //
  // Both sets start "full" (every chip pre-selected) so the explorer shows
  // every study on load; a chip is "active" iff it's a member of its set.
  const activeDevices = new Set();
  const activeEvents = new Set();
  const ALL_DEVICES = chips
    .filter((c) => c.dataset.category === "device")
    .map((c) => c.dataset.event);
  const ALL_EVENTS = chips
    .filter((c) => c.dataset.category === "event")
    .map((c) => c.dataset.event);
  ALL_DEVICES.forEach((v) => activeDevices.add(v));
  ALL_EVENTS.forEach((v) => activeEvents.add(v));
  // Click vs. double-click disambiguation — a single click toggles, a
  // double click solos (or restores all if already soloed). Native dblclick
  // fires AFTER click, so we defer the click action briefly and cancel it
  // when a double-click follows.
  const CLICK_DELAY_MS = 220;
  let pendingClickTimer = null;
  let tooltipHideTimer = null;
  // Tracks the currently displayed study so a second click on the same
  // element collapses the details panel.
  let currentSelection = "";

  const NEURO_COLORS = {
    Eeg: "#2563eb",
    Meg: "#c026d3",
    Fmri: "#dc2626",
    Ieeg: "#f97316",
    Emg: "#16a34a",
    Fnirs: "#eab308",
    Multimodal: "#111827",
  };
  // Chip categoriser: the device row also exposes a synthetic "Multimodal"
  // chip. Studies that declare more than one neuro modality carry an extra
  // "Multimodal" entry in their `data-events`, so the same filter logic
  // matches them via this set membership.
  const NEURO_TYPES_SET = new Set([
    "Eeg", "Meg", "Fmri", "Ieeg", "Emg", "Fnirs", "Multimodal",
  ]);
  const PACKAGE_TO_REPO = {
    neuralset: "neuralset-repo",
    neuralfetch: "neuralfetch-repo",
    neuraltrain: "neuraltrain-repo",
    neuralbench: "neuralbench-repo",
  };

  function githubUrlFor(modulePath) {
    if (!modulePath) {
      return "";
    }
    const top = modulePath.split(".", 1)[0];
    const repo = PACKAGE_TO_REPO[top] || "neuralset-repo";
    const filepath = modulePath.replace(/\\./g, "/");
    return `https://github.com/facebookresearch/neuroai/blob/main/${repo}/${filepath}.py`;
  }

  function elementEvents(element) {
    return (element.dataset.events || "").split("|").filter(Boolean);
  }

  function matchesFilters(element) {
    const events = elementEvents(element);
    const devicePass = activeDevices.size === 0
      || events.some((e) => activeDevices.has(e));
    const eventPass = activeEvents.size === 0
      || events.some((e) => activeEvents.has(e));
    return devicePass && eventPass;
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

  function setForName(name) {
    return NEURO_TYPES_SET.has(name) ? activeDevices : activeEvents;
  }

  function updateChips() {
    chips.forEach((chip) => {
      const value = chip.dataset.event || "";
      if (!value) {
        return;
      }
      chip.classList.toggle("active", setForName(value).has(value));
    });
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
      const eventName = header.dataset.event || "";
      header.classList.toggle("active", setForName(eventName).has(eventName));
    });
    updateChips();
    updateSummary();
  }

  function allForName(name) {
    return NEURO_TYPES_SET.has(name) ? ALL_DEVICES : ALL_EVENTS;
  }

  function toggleEvent(eventName) {
    if (!eventName) {
      return;
    }
    const set = setForName(eventName);
    if (set.has(eventName)) {
      set.delete(eventName);
    } else {
      set.add(eventName);
    }
    applyFilters();
  }

  function soloEvent(eventName) {
    if (!eventName) {
      return;
    }
    const set = setForName(eventName);
    const all = allForName(eventName);
    // Second double-click on the soloed chip restores the full selection
    // — matches Plotly-style legend behavior so users can get back to "all"
    // without hunting for every chip individually.
    if (set.size === 1 && set.has(eventName)) {
      set.clear();
      all.forEach((v) => set.add(v));
    } else {
      set.clear();
      set.add(eventName);
    }
    applyFilters();
  }

  function scheduleClick(eventName) {
    if (pendingClickTimer) {
      clearTimeout(pendingClickTimer);
    }
    pendingClickTimer = setTimeout(() => {
      pendingClickTimer = null;
      toggleEvent(eventName);
    }, CLICK_DELAY_MS);
  }

  function cancelPendingClick() {
    if (pendingClickTimer) {
      clearTimeout(pendingClickTimer);
      pendingClickTimer = null;
    }
  }

  function fillStats(target) {
    detailsStats.innerHTML = "";
    const subjects = Number(target.dataset.subjects || 0);
    const timelines = Number(target.dataset.timelines || 0);
    const hours = Number(target.dataset.hours || 0);
    const frequency = Number(target.dataset.frequency || 0);
    const channels = Number(target.dataset.channels || 0);
    const neuro = target.dataset.neuro || target.dataset.device || "";
    const stats = [];
    if (neuro) {
      stats.push(["Modality", neuro]);
    }
    if (subjects > 0) {
      stats.push(["Subjects", subjects.toLocaleString()]);
    }
    if (timelines > 0) {
      stats.push(["Timelines", timelines.toLocaleString()]);
    }
    if (Number.isFinite(hours) && hours > 0) {
      stats.push(["Est. hours", formatNumber(hours)]);
    }
    if (channels > 0) {
      stats.push(["Channels", channels.toLocaleString()]);
    }
    if (frequency > 0) {
      stats.push(["Frequency", `${formatNumber(frequency)} Hz`]);
    }
    stats.forEach(([label, value]) => {
      const row = document.createElement("div");
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value;
      row.appendChild(dt);
      row.appendChild(dd);
      detailsStats.appendChild(row);
    });
    detailsStats.style.display = stats.length === 0 ? "none" : "";
  }

  function fillEvents(target) {
    // "Multimodal" is a derived device category we inject into
    // `data-events` for chip-filtering only — hide it from the actual
    // event-type pill list so users don't see it next to real events.
    const events = elementEvents(target).filter((e) => e !== "Multimodal");
    detailsEventsList.innerHTML = "";
    if (events.length === 0) {
      detailsEvents.setAttribute("hidden", "");
      return;
    }
    events.forEach((eventName) => {
      const pill = document.createElement("span");
      pill.className = "event-pill";
      pill.textContent = eventName;
      detailsEventsList.appendChild(pill);
    });
    detailsEvents.removeAttribute("hidden");
  }

  function buildBashSnippet(requirements) {
    const lines = ["pip install neuralfetch"];
    const reqs = (requirements || "")
      .split("|")
      .map((r) => r.trim())
      .filter(Boolean);
    if (reqs.length) {
      // Quote any package spec carrying `extras` brackets or version
      // specifiers so the line works in zsh (which globs unquoted `[…]`).
      const quoted = reqs.map((r) =>
        /[\\[<>=~ ]/.test(r) ? `'${r}'` : r
      );
      lines.push(`pip install ${quoted.join(" ")}`);
    }
    return lines.join("\\n");
  }

  function setHighlighted(codeEl, source, lang) {
    // Use the site-wide Prism shim if available so snippet colours match
    // the Pygments-rendered code blocks elsewhere in the docs; otherwise
    // fall back to plain (escaped) text so the snippet still renders.
    const shim = window.codeHighlight;
    if (shim && typeof shim[lang] === "function") {
      codeEl.innerHTML = shim[lang](source);
    } else {
      codeEl.textContent = source;
    }
  }

  function buildPythonSnippet(name) {
    return [
      "import neuralset as ns",
      "",
      "study = ns.Study(",
      `    name="${name}",`,
      '    path="/data",  # where original study is downloaded',
      '    infra=dict(folder="/cache"),  # cache study.run() output',
      ")",
      "print(study.requirements)",
      "study.download()",
      "events = study.run()",
    ].join("\\n");
  }

  function showDetails(target) {
    const name = target.dataset.name || "";
    // Clicking the same study a second time closes the panel — mirrors the
    // dismiss-by-second-click pattern that maps studies and tables already
    // use elsewhere in the docs.
    if (currentSelection === name && !details.hasAttribute("hidden")) {
      hideDetails();
      return;
    }
    currentSelection = name;
    const aliases = target.dataset.aliases || "";
    const description = target.dataset.description || "No description available.";
    const url = target.dataset.url || "";
    const modulePath = target.dataset.module || "";
    const device = target.dataset.device || "";
    detailsName.textContent = name || "Study";
    if (device && NEURO_COLORS[device]) {
      detailsDeviceBadge.textContent = device;
      detailsDeviceBadge.style.background = NEURO_COLORS[device];
      detailsDeviceBadge.removeAttribute("hidden");
    } else {
      detailsDeviceBadge.setAttribute("hidden", "");
      detailsDeviceBadge.style.background = "";
    }
    if (aliases) {
      detailsAliases.textContent = `a.k.a. ${aliases}`;
      detailsAliases.removeAttribute("hidden");
    } else {
      detailsAliases.setAttribute("hidden", "");
    }
    detailsDescription.textContent = description;
    fillStats(target);
    fillEvents(target);
    const bashSrc = buildBashSnippet(target.dataset.requirements || "");
    const pySrc = buildPythonSnippet(name);
    setHighlighted(detailsBashCode, bashSrc, "bash");
    setHighlighted(detailsPyCode, pySrc, "python");
    if (url) {
      detailsOpen.href = url;
      detailsOpen.removeAttribute("hidden");
    } else {
      detailsOpen.setAttribute("hidden", "");
      detailsOpen.removeAttribute("href");
    }
    const githubUrl = githubUrlFor(modulePath);
    if (githubUrl) {
      detailsGithub.href = githubUrl;
      detailsGithub.removeAttribute("hidden");
    } else {
      detailsGithub.setAttribute("hidden", "");
      detailsGithub.removeAttribute("href");
    }
    details.removeAttribute("hidden");
    hideTooltip(true);
    requestAnimationFrame(() => {
      details.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  }

  function hideDetails() {
    details.setAttribute("hidden", "");
    currentSelection = "";
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
    const target = event.currentTarget;
    const name = target.dataset.name || "";
    const aliases = target.dataset.aliases || "";
    const description = target.dataset.description || "No description available.";
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
    const hint = document.createElement("div");
    hint.style.opacity = "0.7";
    hint.style.marginTop = "6px";
    hint.textContent = "Click for full details";
    tooltip.appendChild(hint);
    tooltip.removeAttribute("hidden");
    moveTooltip(event);
  }

  function hideTooltip(immediate) {
    clearTimeout(tooltipHideTimer);
    if (immediate) {
      tooltip.setAttribute("hidden", "");
      return;
    }
    tooltipHideTimer = setTimeout(() => {
      tooltip.setAttribute("hidden", "");
    }, 120);
  }

  chips.forEach((chip) => {
    chip.addEventListener("click", () => scheduleClick(chip.dataset.event || ""));
    chip.addEventListener("dblclick", () => {
      cancelPendingClick();
      soloEvent(chip.dataset.event || "");
    });
  });
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => activateTab(tab.dataset.tab || ""));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") {
        return;
      }
      event.preventDefault();
      const idx = tabs.indexOf(tab);
      const next = event.key === "ArrowRight"
        ? tabs[(idx + 1) % tabs.length]
        : tabs[(idx - 1 + tabs.length) % tabs.length];
      activateTab(next.dataset.tab || "");
      next.focus();
    });
  });

  function activateTab(name) {
    if (!name) {
      return;
    }
    tabs.forEach((tab) => {
      const isActive = tab.dataset.tab === name;
      tab.classList.toggle("active", isActive);
      tab.setAttribute("aria-selected", isActive ? "true" : "false");
      tab.setAttribute("tabindex", isActive ? "0" : "-1");
    });
    panels.forEach((panel) => {
      if (panel.dataset.panel === name) {
        panel.removeAttribute("hidden");
      } else {
        panel.setAttribute("hidden", "");
      }
    });
  }

  eventHeaders.forEach((header) => {
    header.addEventListener("click", () => scheduleClick(header.dataset.event || ""));
    header.addEventListener("dblclick", () => {
      cancelPendingClick();
      soloEvent(header.dataset.event || "");
    });
  });
  root.querySelectorAll("[data-action='study']").forEach((element) => {
    element.addEventListener("click", (event) => {
      // Allow event-col header clicks (which propagate through their <button>)
      // to keep their toggle behavior; only study cells / circles open the panel.
      if (event.target.closest("th.event-col")) {
        return;
      }
      showDetails(element);
    });
    element.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        showDetails(element);
      }
    });
    element.addEventListener("mouseenter", showTooltip);
    element.addEventListener("mousemove", moveTooltip);
    element.addEventListener("mouseleave", () => hideTooltip(false));
  });
  details.querySelectorAll("[data-details-close]").forEach((el) => {
    el.addEventListener("click", hideDetails);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !details.hasAttribute("hidden")) {
      hideDetails();
    }
  });
  applyFilters();
})();
""".strip()


if __name__ == "__main__":
    docs_root = Path(__file__).resolve().parents[1]
    report_path = build_docs_study_explorer(
        docs_root / "neuralfetch" / "_explore_studies.html"
    )
    print(f"Saved HTML fragment to {report_path}")
