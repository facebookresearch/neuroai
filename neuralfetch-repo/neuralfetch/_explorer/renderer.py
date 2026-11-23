# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import math
import re
import typing as tp
from html import escape
from importlib.resources import files
from urllib.parse import unquote, urlsplit

import pandas as pd

import neuralset as ns
from neuralfetch._explorer import _icons
from neuralset.events import study

_COLORS = {
    "Eeg": "#437ED8",
    "Emg": "#B5B5B5",
    "Fmri": "#BA4039",
    "Fnirs": "#E8C13B",
    "Ieeg": "#6D3D92",
    "Meg": "#03B779",
    "Multimodal": "#111827",
}

_SUMMARY_COLUMNS = [
    "name",
    "module",
    "package",
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

_REPORT_ROOT_ID = "study-explorer"
_FRAGMENT_HEADER = (
    "<!-- Auto-generated from neuralfetch._explorer.renderer using StudyInfo metadata; "
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

    device_colors: dict[str, str] = {}
    documentation_url: str | None = None
    install_commands: dict[str, str] = {}
    neuro_types: str | list[str] | tp.Literal["all"] = "all"
    packages: list[str] | None = None
    report_index_url: str | None = None
    report_link_label: str = "View reports"
    source_urls: dict[str, str] = {
        "neuralfetch": "https://github.com/facebookresearch/neuroai/blob/main"
    }
    title: str = "Study explorer"

    _NEURO_TYPES = {"Eeg", "Meg", "Emg", "Fmri", "Fnirs", "Ieeg"}

    def model_post_init(self, __context: tp.Any) -> None:
        for field in ("documentation_url", "report_index_url"):
            value = getattr(self, field)
            if value is not None:
                _validate_url(value, field, relative=True)
        for package, url in self.source_urls.items():
            _validate_url(url, f"source_urls[{package!r}]", relative=False)
            if any(c in url for c in "?#"):
                raise ValueError(
                    "source_urls must contain repository/blob/ref bases without query or fragment"
                )
        for package in [
            *(self.packages or ()),
            *self.install_commands,
            *self.source_urls,
        ]:
            if not re.fullmatch(r"[A-Za-z_]\w*", package):
                raise ValueError(f"Invalid study package name: {package!r}")
        for device, color in self.device_colors.items():
            if device not in _COLORS or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                raise ValueError(
                    "device_colors requires known device names and #RRGGBB colors"
                )
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
            package = cls.__module__.split(".", 1)[0]
            if (
                self.packages is not None and package not in self.packages
            ) or cls._info is None:
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
                    "package": package,
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

    def render_html_fragment(self, preview_html: str = "") -> str:
        """Render a fragment with optional trusted preview-action markup."""
        summaries = self.get_summaries()
        if summaries.empty:
            raise RuntimeError("No studies with StudyInfo were found.")
        return "\n".join(
            [
                _FRAGMENT_HEADER,
                "<style>",
                _CSS,
                "</style>",
                self._render_body(
                    summaries, include_title=False, preview_html=preview_html
                ),
                '<script type="module">',
                _JS,
                "</script>",
            ]
        )

    def render_html_report(self, preview_html: str = "") -> str:
        """Render a standalone explorer with optional trusted report-preview markup."""
        summaries = self.get_summaries()
        if summaries.empty:
            raise RuntimeError("No studies with StudyInfo were found.")
        return "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f"<title>{escape(self._report_title())}</title>",
                "<style>",
                "body { margin: 0; padding: 24px; font-family: system-ui, sans-serif; }",
                "#study-explorer { max-width: 72rem; margin: 0 auto; }",
                _CSS,
                "</style>",
                "</head>",
                "<body>",
                self._render_body(
                    summaries, include_title=True, preview_html=preview_html
                ),
                '<script type="module">',
                _JS,
                "</script>",
                "</body>",
                "</html>",
            ]
        )

    def _render_body(
        self, summaries: pd.DataFrame, *, include_title: bool, preview_html: str = ""
    ) -> str:
        colors = _COLORS | self.device_colors
        style = "; ".join(
            f"--nse-{name.lower()}: {color}" for name, color in colors.items()
        )
        commands = {
            package: command
            for package, command in self.install_commands.items()
            if package in set(summaries.package)
        }
        pieces = [
            f'<div id="{_REPORT_ROOT_ID}" '
            f'class="{"standalone-explorer" if include_title else ""}" '
            f'style="{style}" data-device-colors="{escape(json.dumps(colors))}" '
            f'data-install-commands="{escape(json.dumps(commands))}" '
            f'data-report-index-url="{escape(self.report_index_url or "")}">',
            # Set the view before parsing the catalog so a deep link never flashes it.
            '<script>if (new URLSearchParams(location.search).has("study")) '
            'document.currentScript.parentElement.setAttribute("data-study-view", "");</script>',
            '<div class="explorer-catalog">',
        ]
        if include_title:
            links = []
            if self.report_index_url is not None:
                links.append(f'<a href="{escape(self.report_index_url)}">Reports</a>')
            if self.documentation_url is not None:
                links.append(
                    f'<a href="{escape(self.documentation_url)}">Documentation ↗</a>'
                )
            pieces.append('<header class="explorer-header">')
            pieces.append(f"<h1>{escape(self._report_title())}</h1>")
            if links:
                pieces.append(
                    '<nav class="explorer-site-nav" aria-label="Site navigation">'
                    + "".join(links)
                    + "</nav>"
                )
            pieces.append("</header>")
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
                f"{_icons.svg('chart-area')}"
                "&nbsp;Plot</button>",
                '<button type="button" class="explorer-tab" role="tab" '
                'aria-selected="false" data-tab="table" id="tab-table" '
                'aria-controls="panel-table" tabindex="-1">'
                f"{_icons.svg('table')}"
                "&nbsp;Table</button>",
                "</div>",
                '<div class="explorer-tab-panel" role="tabpanel" id="panel-plot" '
                'aria-labelledby="tab-plot" data-panel="plot">',
                self._render_scatter(summaries),
                "</div>",
                '<div class="explorer-tab-panel" role="tabpanel" id="panel-table" '
                'aria-labelledby="tab-table" data-panel="table" hidden>',
                '<p class="note">'
                f"{_icons.svg('arrows-left-right')}"
                "&nbsp;Scroll horizontally to see every event column &middot;"
                f"&nbsp;{_icons.svg('arrow-pointer')}"
                "&nbsp;<strong>click</strong> a column header to toggle a filter "
                "(<strong>double-click</strong> to solo) &middot; "
                "click a study name for details."
                "</p>",
                self._render_event_table(summaries),
                "</div>",
                "</section>",
                "</div>",
                '<nav class="study-view-nav" aria-label="Study navigation" hidden>'
                '<a class="study-view-back">← Back to study catalog</a></nav>',
                '<p class="study-view-empty" role="status" hidden>'
                "Study not found in this catalog.</p>",
                self._render_details_panel(preview_html=preview_html),
                "</div>",
            ]
        )
        return "\n".join(pieces)

    def _render_details_panel(self, *, preview_html: str = "") -> str:
        reports = preview_html
        if not reports and self.report_index_url is not None:
            reports = (
                '<a class="study-details-btn study-details-reports-link" '
                'target="_blank" rel="noopener noreferrer" hidden>'
                f"{escape(self.report_link_label)} ↗</a>"
            )
        if reports:
            reports = f'<div class="study-details-reports">{reports}</div>'
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
            f"{_icons.svg('terminal')}"
            "&nbsp;Install dependencies</div>"
            '<pre class="study-details-snippet study-details-snippet--bash">'
            "<code></code></pre>"
            "</div>"
            '<div class="study-details-snippet-wrap">'
            '<div class="study-details-snippet-label">'
            f"{_icons.svg('python')}"
            "&nbsp;Python</div>"
            '<pre class="study-details-snippet study-details-snippet--python">'
            "<code></code></pre>"
            "</div>"
            "</div>"
            f"{reports}"
            '<div class="study-details-actions">'
            '<a class="study-details-btn study-details-open" '
            'target="_blank" rel="noopener noreferrer" hidden>'
            f"{_icons.svg('up-right-from-square')}"
            "&nbsp;Original study</a>"
            '<a class="study-details-btn study-details-github" '
            'target="_blank" rel="noopener noreferrer" hidden>'
            f"{_icons.svg('github')}"
            "&nbsp;Curation code</a>"
            "</div>"
            "</div>"
        )

    def _render_controls(self, summaries: pd.DataFrame) -> str:
        device_chips: list[str] = []
        event_chips: list[str] = []
        for event_type in self._event_filter_options(summaries):
            if event_type in self._NEURO_TYPES:
                color = (_COLORS | self.device_colors).get(event_type, "#6b7280")
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
                f'style="--chip-color: {(_COLORS | self.device_colors)["Multimodal"]}">Multimodal</button>'
            )

        package_chips = [
            f'<button type="button" class="event-chip event-chip--package" '
            f'data-event="{escape(package)}" data-category="package">'
            f"{escape(package)}</button>"
            for package in sorted(set(summaries.package))
        ]

        rows: list[str] = [
            '<div class="controls-row">'
            '<span class="controls-label">Package</span>'
            '<div class="event-chips" role="group" aria-label="Filter by package">'
            + "".join(package_chips)
            + "</div>"
            "</div>"
        ]
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

        for row in tp.cast(tp.Iterable[tp.Any], rows.sort_values("name").itertuples()):
            x = x_pos(float(row.hours_per_subject))
            y = y_pos(float(row.n_subjects))
            r = radius(float(row.n_hours_plot))
            color = (_COLORS | self.device_colors).get(
                _primary_neuro_type(str(row.neuro_event_type)), "#6b7280"
            )
            event_data = _data_list(_events_for_filter(row))  # type: ignore[arg-type]
            pieces.append(
                f'<g class="study-point" data-action="study" '
                f'tabindex="0" role="button" '
                f'data-package="{escape(str(row.package))}" '
                f'data-events="{escape(event_data)}" data-name="{escape(row.name)}">'
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
                color = (_COLORS | self.device_colors).get(event_type, "#6b7280")
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
            '<th class="study-col" colspan="7"></th>'
            f'<th colspan="{len(neuro_event_types)}">Neuro events</th>'
            + (
                f'<th colspan="{len(other_event_types)}">Other events</th>'
                if other_event_types
                else ""
            )
            + "</tr><tr>"
            '<th class="study-col">Study</th>'
            "<th>Alias</th>"
            "<th>Package</th>"
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
        for row in tp.cast(
            tp.Iterable[tp.Any], summaries.sort_values("name").itertuples()
        ):
            row_event_types: list[str] = row.event_types  # type: ignore[assignment]
            device = _primary_neuro_type(str(row.neuro_event_type))
            event_data = _data_list(_events_for_filter(row))
            description = _description_attr(str(row.description))
            url = _url_attr(str(row.url))
            source = self.source_urls.get(str(row.package))
            code_url = ""
            if source is not None and re.fullmatch(
                r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*", row.module
            ):
                code_url = f"{source.rstrip('/')}/{row.package}-repo/{row.module.replace('.', '/')}.py"
            cells = [
                f'<th class="study-col study-name" data-action="study" '
                f'data-name="{escape(row.name)}" '
                f'data-module="{escape(row.module)}" '
                f'data-package="{escape(row.package)}" '
                f'data-device="{escape(device)}" '
                f'data-neuro="{escape(str(row.neuro_event_type))}" '
                f'data-url="{escape(url)}" data-code-url="{escape(code_url)}" '
                f'data-events="{escape(event_data)}" '
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
                f"<td>{escape(str(row.package))}</td>",
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
                f'<tr class="study-row device-{escape(device.lower())}" '
                f'data-device="{escape(device)}" data-events="{escape(event_data)}" '
                f'data-package="{escape(row.package)}" '
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
        return self.title

    def _neuro_filter(self) -> set[str] | None:
        if self.neuro_types == "all":
            return None
        if isinstance(self.neuro_types, list):
            return set(self.neuro_types)
        return {self.neuro_types}


def _validate_url(value: str, field: str, *, relative: bool) -> None:
    decoded = unquote(value)
    try:
        url = urlsplit(value)
        _ = url.port
    except ValueError as error:
        raise ValueError(f"{field} must be a valid URL") from error
    if (
        not value
        or any(ord(c) < 32 or ord(c) == 127 or c.isspace() or c == "\\" for c in decoded)
        or url.scheme not in ("", "http", "https")
        or (url.netloc and not url.scheme)
        or (url.scheme and not url.hostname)
        or url.username is not None
        or (not relative and not url.scheme)
        or value.startswith("//")
        or decoded.startswith("//")
    ):
        kind = "HTTP(S) or relative page" if relative else "HTTP(S) repository base"
        raise ValueError(
            f"{field} must be an {kind} URL without controls, backslashes or userinfo"
        )


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
    try:
        _validate_url(text, "study URL", relative=False)
    except ValueError:
        return ""
    return text


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


_CSS = (
    files("neuralfetch._explorer")
    .joinpath("static/study_explorer.css")
    .read_text(encoding="utf-8")
    .strip()
)
_JS = (
    files("neuralfetch._explorer")
    .joinpath("static/study_explorer.js")
    .read_text(encoding="utf-8")
    .strip()
)
