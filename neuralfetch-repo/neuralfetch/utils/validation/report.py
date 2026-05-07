# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""HTML metadata builder and MNE Report generator for validation runs."""

from __future__ import annotations

import datetime
import html
import importlib.metadata
import logging
import pprint
import typing as tp
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .config import StudyValidation
from .erp import _build_subject_epochs
from .plots import (
    _plot_drop_grid,
    _plot_grand_average,
    _plot_group_grand_average_html,
    _plot_subject_scores,
    _plot_trf_grand_average,
    _plot_trf_scores,
    _scores_summary_table,
    _subject_sort_key,
    _summary_table_html,
)

if tp.TYPE_CHECKING:
    import mne
    import neuralset as ns
    import pandas as pd
    import xarray
    from neuralset.events import study

logger = logging.getLogger(__name__)


def _get_package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _format_config_value(value: tp.Any) -> str:
    """Render a dict/list/tuple as a pretty-printed, HTML-escaped ``<pre>`` block.

    Empty containers render as a muted ``(none)`` placeholder so the
    surrounding description list stays visually balanced.
    """
    if value is None or (hasattr(value, "__len__") and len(value) == 0):
        return "<em>(none)</em>"
    formatted = pprint.pformat(value, indent=2, width=72, sort_dicts=False)
    style = (
        "margin:0; padding:6px 8px;"
        " background:rgba(0,0,0,0.04);"
        " border-radius:4px;"
        " overflow-x:auto;"
        " font-size:0.9em;"
    )
    return f'<pre style="{style}">{html.escape(formatted)}</pre>'


def _build_metadata_html(
    study_cls: tp.Type["study.Study"],
    validation: StudyValidation,
    query: str | None = None,
    infra: dict[str, tp.Any] | None = None,
) -> str:
    """Build an HTML block with study metadata, dataset info, and analysis config.

    Parameters
    ----------
    study_cls : type[Study]
        The study class being validated.
    validation : StudyValidation
        Declarative validation config.
    query : str or None
        CLI-time study-level pandas query override, rendered in a
        "Runtime Overrides" section when provided.
    infra : dict or None
        CLI-time infra overrides (cluster / slurm_partition / timeout_min
        / ...), rendered in the "Runtime Overrides" section when provided.
    """
    parts: list[str] = []

    # Environment
    parts.append("<h3>Environment</h3>")
    parts.append("<ul>")
    for pkg in ("neuralset", "neuralfetch", "neuralyze"):
        parts.append(f"<li><strong>{pkg}</strong>: {_get_package_version(pkg)}</li>")
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    parts.append(f"<li><strong>Generated at</strong>: {timestamp}</li>")
    parts.append("</ul>")

    # Study metadata
    parts.append("<h3>Study Metadata</h3>")
    parts.append("<ul>")
    if validation.description:
        parts.append(
            f"<li><strong>Description:</strong> {html.escape(validation.description)}</li>"
        )
    if study_cls.description:
        parts.append(f"<li><strong>Dataset:</strong> {study_cls.description}</li>")
    if study_cls.url:
        parts.append(
            f'<li><strong>URL:</strong> <a href="{study_cls.url}">{study_cls.url}</a></li>'
        )
    if study_cls.licence:
        parts.append(f"<li><strong>Licence:</strong> {study_cls.licence}</li>")
    parts.append("</ul>")

    # Dataset summary from _info
    info = study_cls._info
    if info is not None:
        parts.append("<h3>Dataset Summary</h3>")
        parts.append("<ul>")
        parts.append(f"<li><strong>Participants:</strong> {info.num_subjects}</li>")
        parts.append(f"<li><strong>Timelines:</strong> {info.num_timelines}</li>")
        parts.append(f"<li><strong>Sampling frequency:</strong> {info.frequency} Hz</li>")
        modalities = ", ".join(sorted(info.event_types_in_query))
        parts.append(f"<li><strong>Event types:</strong> {modalities}</li>")
        parts.append("</ul>")

    # Analysis config
    parts.append("<h3>Analysis Configuration</h3>")
    parts.append("<dl>")

    def _row(label: str, value_html: str) -> None:
        parts.append(f"<dt><strong>{html.escape(label)}</strong></dt>")
        parts.append(f"<dd>{value_html}</dd>")

    _row("Mode", html.escape(validation.mode))
    _row("Event type", html.escape(validation.event_type))
    _row("Epoch window", f"[{validation.start}, {validation.stop}] s")
    _row("Neuro config", _format_config_value(validation.neuro))
    _row("Extractor config", _format_config_value(validation.extractor))
    _row("Model", _format_config_value(validation.model))
    _row("CV", _format_config_value(validation.cv))
    _row("Scoring", _format_config_value(validation.scoring))
    if validation.train_query:
        _row("Train query", f"<code>{html.escape(validation.train_query)}</code>")
    if validation.test_query:
        _row("Test queries", _format_config_value(list(validation.test_query)))
    if validation.reference_metric is not None:
        _row(
            f"Reference ({validation.reference_metric_name})",
            f"{validation.reference_metric:.4f}",
        )
    if validation.trf is not None:
        trf_summary: dict[str, tp.Any] = {
            "tmin": validation.trf.tmin,
            "tmax": validation.trf.tmax,
            "aggregation": validation.trf.aggregation,
            "mode": "encod",
        }
        _row("TRF config", _format_config_value(trf_summary))
    parts.append("</dl>")

    # Runtime overrides (CLI flags: --query, --cluster, --slurm-partition, --slurm-time-min)
    if query is not None or infra:
        parts.append("<h3>Runtime Overrides</h3>")
        parts.append("<dl>")
        if query is not None:
            _row("Study query", f"<code>{html.escape(query)}</code>")
        if infra:
            _row("Infra", _format_config_value(infra))
        parts.append("</dl>")

    # Citation
    if study_cls.bibtex.strip():
        parts.append("<h3>Citation</h3>")
        parts.append(f"<pre>{html.escape(study_cls.bibtex.strip())}</pre>")

    return "\n".join(parts)


def generate_mne_report(
    study_cls: tp.Type["study.Study"],
    validation: StudyValidation,
    scores: "xarray.DataArray",
    output_path: Path,
    study_instance: "ns.Study | None" = None,
    events: "pd.DataFrame | None" = None,
    query: str | None = None,
    infra: dict[str, tp.Any] | None = None,
    trf_scores: "xarray.DataArray | None" = None,
) -> Path:
    """Generate an MNE Report HTML from validation scores.

    Parameters
    ----------
    study_cls : type[Study]
        The study class (provides ``_info``, ``bibtex``, ``url``, etc.).
    validation : StudyValidation
        The validation config for this study.
    scores : xarray.DataArray
        Scores returned by ``SlidingWindow.get_scores()``.
    output_path : Path
        File path for the saved HTML report.
    study_instance : neuralset.Study or None
        Instantiated study used to re-read raw data for per-subject
        ERP/ERF figures and the Subjects-by-Channels drop grid.  When
        ``None`` (e.g. unit tests), those sections are skipped.
    events : pd.DataFrame or None
        Events DataFrame returned by ``study_instance.run()``.  Required
        to build ERP/ERF figures, the drop grid, and the events map.
    query : str or None
        CLI-time study-level pandas query override, echoed in the
        "Runtime Overrides" section of the report.
    infra : dict or None
        CLI-time infra overrides (``cluster`` / ``slurm_partition`` /
        ``timeout_min`` / ...), echoed in the "Runtime Overrides" section.
    trf_scores : xarray.DataArray or None
        Per-channel TRF encoding scores from
        :meth:`neuralyze.trf.TRFScoring.get_scores`.  When provided, a
        per-participant TRF section is added at the end of the report.
    Returns
    -------
    Path
        The path to the saved report.
    """
    import mne

    study_name = study_cls.__name__

    report = mne.Report(title=f"{study_name}: Validation")

    # 1. Metadata section
    metadata_html = _build_metadata_html(study_cls, validation, query=query, infra=infra)
    report.add_html(metadata_html, title="Study Information", tags=("metadata",))

    # 1b. Events preview table (head 10)
    if events is not None:
        try:
            table_html = events.head(10).to_html(
                index=False,
                classes="table table-sm table-striped table-hover table-bordered",
                border=0,
            )
            overflow_div = (
                '<div style="overflow-x: auto; max-width: 100%; display: block;">'
                f"{table_html}"
                "</div>"
            )
            report.add_html(
                html=f"<h3>Events (first 10 rows)</h3>{overflow_div}",
                title="Events Preview",
                tags=("metadata",),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Events table build failed: %s", exc)

    # 2. Grand average figure (static)
    fig = _plot_grand_average(scores, validation)
    report.add_figure(fig, title="Grand Average", tags=("summary",))
    plt.close(fig)

    # 3. Group grand average (interactive Plotly)
    try:
        html_str = _plot_group_grand_average_html(scores, validation)
        report.add_html(
            html_str,
            title="Group Grand Average (interactive)",
            tags=("summary", "group"),
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("Group grand-average build failed: %s", exc)

    # 3b. TRF grand average summary
    if trf_scores is not None:
        trf_grand_fig = None
        try:
            trf_grand_fig = _plot_trf_grand_average(trf_scores)
            report.add_figure(
                trf_grand_fig,
                title="TRF Grand Average",
                tags=("summary", "trf"),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("TRF grand average build failed: %s", exc)
        finally:
            if trf_grand_fig is not None:
                plt.close(trf_grand_fig)

    subject_channels: dict[str, list[str]] = {}
    subject_bads: dict[str, list[str]] = {}
    failed_subjects: set[str] = set()

    # 4. Per-subject ERP/ERF (pre-walk so the drop grid gets populated
    # before it is rendered).  We build figures here but only *attach*
    # them to the report after the drop grid, matching the plan's
    # section order.
    subjects_scored = sorted(
        scores.coords["subject"].values.tolist(), key=_subject_sort_key
    )
    per_subject_figs: dict[str, tuple["mne.Evoked", str]] = {}
    if study_instance is not None and events is not None:
        # Phase 1 — load raw files and build epochs sequentially.
        # MNE's raw file readers (e.g. BrainVision) are not thread-safe:
        # concurrent reads produce "I/O operation on closed file" errors
        # because internal file handles are not safe to share across threads.
        # The two-phase structure is kept so mne.Info is stored here and
        # reused by the TRF section (section 8) without a second raw read.
        epoch_results: dict[str, tuple[tp.Any, str] | None] = {}
        for subject_id in tqdm(
            subjects_scored,
            desc="Loading ERP/ERF data",
            ncols=100,
            leave=False,
        ):
            try:
                epoch_results[subject_id] = _build_subject_epochs(
                    study_instance, events, validation, subject_id
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Evoked build failed for %s: %s", subject_id, exc)
                failed_subjects.add(subject_id)

        # Phase 2 — average epochs into evoked objects on the main thread.
        for subject_id in tqdm(
            subjects_scored,
            desc="Building ERP/ERF evokeds",
            ncols=100,
            leave=False,
        ):
            result = epoch_results.get(subject_id)
            if result is None:
                continue
            epochs, label = result
            subject_bads[subject_id] = list(epochs.info["bads"])
            subject_channels[subject_id] = list(epochs.info["ch_names"])
            try:
                evoked = epochs.average()
            except Exception as exc:  # pragma: no cover
                logger.warning("epochs.average() failed for %s: %s", subject_id, exc)
                failed_subjects.add(subject_id)
                continue
            per_subject_figs[subject_id] = (evoked, label)

    # 5. Subjects x Channels drop grid (opt-in via show_qc = true in TOML)
    if validation.show_qc and events is not None:
        try:
            all_subjects = set(events.subject.unique().tolist())
        except Exception:  # pragma: no cover
            all_subjects = set(subject_channels) | set(subjects_scored)
        scored_set = {str(s) for s in subjects_scored}
        excluded_subjects = (
            {str(s) for s in all_subjects} - scored_set
        ) | failed_subjects
        grid_fig = None
        try:
            grid_fig = _plot_drop_grid(
                subject_channels,
                subject_bads,
                excluded_subjects,
                all_subjects,
            )
            report.add_figure(
                grid_fig,
                title="Participants x Channels: drops",
                tags=("summary", "qc"),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Drop grid build failed: %s", exc)
        finally:
            if grid_fig is not None:
                plt.close(grid_fig)

    # 6. Results summary table
    summary_rows = _scores_summary_table(scores)
    summary_html = _summary_table_html(summary_rows)
    report.add_html(summary_html, title="Results Summary", tags=("summary",))

    # 7. Per-subject figures: score plot + (optional) Evoked + (optional) TRF
    trf_subject_set: set[str] = (
        set(trf_scores.coords["subject"].values.tolist())
        if trf_scores is not None
        else set()
    )
    for subject_id in tqdm(
        subjects_scored,
        desc="Attaching per-participant figures",
        ncols=100,
        leave=False,
    ):
        # Strip study-name prefix (e.g. "StudyName/1" -> "1")
        short_id = subject_id.split("/", 1)[-1] if "/" in subject_id else subject_id
        # Events map for this participant (all event types from the dataframe).
        if events is not None:
            try:
                sfreq = float(validation.erp_display_sfreq or 1000.0)
                subj_events = events[events["subject"] == subject_id]
                if not subj_events.empty and "type" in subj_events.columns:
                    unique_types = sorted(subj_events["type"].dropna().unique().tolist(), key=str)
                    type_to_id = {t: i + 1 for i, t in enumerate(unique_types)}
                    ids = subj_events["type"].map(type_to_id).to_numpy(dtype=int)
                    samples = (subj_events["start"].to_numpy(dtype=float) * sfreq).astype(int)
                    events_arr = np.column_stack(
                        [samples, np.zeros(len(samples), dtype=int), ids]
                    )
                    report.add_events(
                        events_arr,
                        title=f"Participant {short_id} — Events",
                        event_id=type_to_id,
                        sfreq=sfreq,
                        tags=("per-participant",),
                    )
            except Exception as exc:  # pragma: no cover
                logger.warning("Events map failed for %s: %s", subject_id, exc)
        if subject_id in per_subject_figs:
            evoked, label = per_subject_figs[subject_id]
            try:
                report.add_evokeds(
                    evoked,
                    titles=f"Participant {short_id} - {label}",
                    tags=("per-participant", label.lower()),
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("add_evokeds failed for %s: %s", subject_id, exc)
        fig = _plot_subject_scores(scores, subject_id, validation)
        report.add_figure(fig, title=f"Participant {short_id}", tags=("per-participant",))
        plt.close(fig)
        # TRF encoding figure after ERP and decoding for this participant.
        if trf_scores is not None and subject_id in trf_subject_set:
            # Reuse mne.Info from the evoked object built in the ERP phase so
            # the raw file is not read a second time just for the topomap.
            info: "mne.Info | None" = (
                per_subject_figs[subject_id][0].info if subject_id in per_subject_figs else None
            )
            trf_fig = None
            try:
                trf_fig = _plot_trf_scores(trf_scores, subject_id, info=info)
                report.add_figure(
                    trf_fig,
                    title=f"Participant {short_id} — TRF Encoding",
                    tags=("per-participant", "trf"),
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("TRF figure failed for %s: %s", subject_id, exc)
            finally:
                if trf_fig is not None:
                    plt.close(trf_fig)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_path), overwrite=True, open_browser=False)
    logger.info("MNE Report saved to %s", output_path)
    return output_path
