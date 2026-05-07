# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Top-level validation runner: orchestrates scoring, events loading, and report."""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import neuralset as ns

from neuralfetch.utils.base import root_study_folder

from .builders import _build_sliding_window, _build_trf_scoring
from .config import _resolve_validation
from .report import generate_mne_report

logger = logging.getLogger(__name__)


def validate_study(
    name: str,
    output_dir: str | Path,
    study_folder: str | Path | None = None,
    cache_dir: str | Path | None = None,
    query: str | None = None,
    infra: dict[str, tp.Any] | None = None,
) -> Path:
    """Run the standard validation analysis for a study and generate an MNE Report.

    Parameters
    ----------
    name : str
        Registered study name (e.g. ``"Grootswagers2022Human"``).
    output_dir : str or Path
        Directory where the HTML report is written.  Required; callers must
        pick an explicit location so generated reports are never scattered
        into an implicit home-folder default.
    study_folder : str or Path or None
        Root folder containing the study data.  Falls back to
        :func:`~neuralfetch.utils.root_study_folder` if not provided.
    cache_dir : str or Path or None
        Shared root folder for all exca caches (study timelines, sliding-window
        scores, neuro/extractor artifacts).  Defaults to
        :data:`neuralset.CACHE_FOLDER` (``~/.cache/neuralset/``).
    query : str or None
        Optional pandas query applied to the study before ``_load_timelines``
        runs, so only matching timelines are loaded.  Uses
        :func:`neuralset.events.utils.query_with_index` semantics (virtual
        columns ``subject``, ``subject_index``, ``subject_timeline_index``).
        Examples: ``"subject_index < 1"``, ``"subject_index in [0, 1, 2]"``,
        ``"subject_timeline_index < 1"``.
    infra : dict or None
        Override infrastructure config (cluster, slurm_partition, etc.).

    Returns
    -------
    Path
        Path to the generated MNE Report HTML file.
    """
    study_cls, validation = _resolve_validation(name)
    slug = name.lower()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge study-level infra defaults with caller-supplied overrides.
    # Caller (CLI) takes precedence for any key it provides.
    if validation.infra or infra:
        effective_infra: dict[str, tp.Any] = dict(validation.infra or {})
        if infra:
            effective_infra.update(infra)
        infra = effective_infra

    if study_folder is None:
        study_folder = root_study_folder()
    else:
        study_folder = Path(study_folder)

    cache_path = Path(cache_dir) if cache_dir is not None else ns.CACHE_FOLDER

    logger.info("Building SlidingWindow for %s ...", name)
    sw = _build_sliding_window(
        name,
        validation,
        study_folder,
        cache_dir=cache_path,
        query=query,
        infra=infra,
    )

    logger.info("Running get_scores() ...")
    scores = sw.get_scores()

    trf_scores = None
    if validation.trf is not None:
        logger.info("Building TRFScoring for %s ...", name)
        trf = _build_trf_scoring(
            name,
            validation,
            study_folder,
            cache_dir=cache_path,
            query=query,
            infra=infra,
        )
        logger.info("Running TRFScoring.get_scores() ...")
        try:
            trf_scores = trf.get_scores()
        except Exception:  # pragma: no cover
            logger.error("TRF get_scores() failed, skipping TRF section", exc_info=True)

    logger.info("Loading events for report figures ...")
    study_instance: ns.Study | None
    events_df = None
    try:
        # Force sequential timeline loading (see _build_sliding_window for
        # rationale -- avoids the upstream processpool __setstate__ bug).
        study_instance = study_cls(
            path=study_folder,
            infra_timelines={"cluster": None},
        )
        if query is not None:
            study_instance.query = query
        events_df = study_instance.run()
    except Exception as exc:  # pragma: no cover
        logger.warning("Skipping ERP/ERF + drop grid: could not load events: %s", exc)
        study_instance = None

    report_path = output_dir / f"{slug}_validation.html"
    generate_mne_report(
        study_cls,
        validation,
        scores,
        report_path,
        study_instance=study_instance,
        events=events_df,
        query=query,
        infra=infra,
        trf_scores=trf_scores,
    )

    logger.info("Validation complete for %s: %s", name, report_path)
    return report_path
