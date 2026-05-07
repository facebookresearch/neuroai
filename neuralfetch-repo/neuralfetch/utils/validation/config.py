# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic config models and TOML discovery for study validations."""

from __future__ import annotations

import logging
import tomllib
import typing as tp
from pathlib import Path

import pydantic

if tp.TYPE_CHECKING:
    from neuralset.events import study

logger = logging.getLogger(__name__)


class TRFConfig(pydantic.BaseModel):
    """Config for an optional TRF (Temporal Response Function) encoding analysis.

    When attached to a :class:`StudyValidation` via its ``trf`` field, a
    :class:`neuralyze.trf.TRFScoring` is run in addition to the standard
    :class:`~neuralyze.SlidingWindow` analysis.  The TRF models how the brain
    *continuously* responds to stimulus features over a lag window, rather than
    using epoch-locked decoding.

    The ``neuro``, ``extractor``, ``model``, ``cv``, ``scoring``, ``event_type``,
    and ``mode`` fields from the parent :class:`StudyValidation` are reused.
    Only the lag window and extractor aggregation strategy need to be specified
    here.

    Attributes
    ----------
    tmin : float
        Earliest lag in seconds (e.g. ``-0.2`` for 200 ms pre-stimulus).
    tmax : float
        Latest lag in seconds (e.g. ``0.5`` for 500 ms post-stimulus).
    aggregation : {"sum", "average"}
        How stimulus features are placed in the continuous time-series.
        ``"sum"`` places feature vectors at each trigger onset (standard for
        discrete image/word paradigms).  Overrides the extractor's
        ``aggregation`` key from the parent config.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    tmin: float = -0.2
    tmax: float = 0.5
    aggregation: str = "sum"


class StudyValidation(pydantic.BaseModel):
    """Declarative config for a study's decoding/encoding feasibility check.

    Describes one representative :class:`~neuralyze.SlidingWindow`
    analysis used to confirm that the end-to-end pipeline (raw ->
    events -> neuro -> features -> model -> time-resolved scores) runs
    on a study and produces sensible per-subject output.  This is *not*
    a full replication of the source paper's analysis; a config may
    optionally set ``reference_metric`` to overlay a published number
    on the plots for context, but matching a paper is not the goal.

    Attributes
    ----------
    description : str
        Human-readable summary of the analysis being exercised (e.g.
        the modality, features, and target).
    mode : {"decod", "encod"}
        Decoding (neuro -> features) or encoding (features -> neuro).
    event_type : str
        Trigger event type for epoching (e.g. ``"Image"``, ``"Word"``).
    start : float
        Epoch start relative to trigger, in seconds.
    stop : float
        Epoch end relative to trigger, in seconds.
    neuro : dict
        Neuro extractor config passed to :class:`neuralyze.SlidingWindow`.
    extractor : dict
        Feature extractor config passed to :class:`neuralyze.SlidingWindow`.
    model : dict
        Sklearn pipeline config (e.g. ``{"sklearn_model": {"name": "RidgeCV"}}``).
    cv : dict
        Cross-validation config (e.g. ``{"n_splits": 5}``).
    scoring : dict
        Metric config (e.g. ``{"name": "corr"}`` or ``{"name": "acc"}``).
    train_query : str or None
        Optional pandas query to filter training events.
    test_query : tuple of str
        Optional pandas queries for test sub-scoring.
    reference_metric : float or None
        Optional published metric to overlay on the decoding plots for
        context.  Leave unset when no comparison is intended.
    reference_metric_name : str
        Name of the reference metric used as a plot axis label and in
        the overlaid reference line (e.g. ``"Pearson r"``,
        ``"accuracy"``).  Informational only when ``reference_metric``
        is unset.
    erp_display_filter : tuple of (float or None, float or None) or None
        Band-pass applied (via :meth:`mne.io.Raw.filter`) only to the
        per-subject ERP/ERF figure in the report.  Does not affect the
        SlidingWindow cache uid.  Defaults to ``(None, 40.0)`` -- a light
        40 Hz low-pass for cleaner display.  Set to ``None`` to disable.
    erp_display_sfreq : float or None
        Resample target (via :meth:`mne.io.Raw.resample`) applied only to
        the per-subject ERP/ERF figure.  Defaults to ``200.0`` Hz.  Set
        to ``None`` to keep the native sampling rate.
    trf : TRFConfig or None
        Optional TRF encoding analysis.  When set, a
        :class:`neuralyze.trf.TRFScoring` is run in addition to the
        SlidingWindow and its per-channel scores are added to the report.
        See :class:`TRFConfig` for parameter details.
    show_qc : bool
        Whether to include the "Participants x Channels: drops" QC figure in
        the report.  Defaults to ``False`` — the drop grid needs study-specific
        tuning and is omitted by default.  Set to ``true`` in the TOML to
        enable it.
    infra : dict or None
        Default infrastructure parameters for this study's validation run.
        Acts as a base that CLI flags merge on top of (CLI takes precedence
        for any key it provides).  Useful for storing study-specific
        resource requirements (e.g. ``cluster``, ``slurm_partition``,
        ``timeout_min``, ``mem_gb``, ``cpus_per_task``) so the correct
        command to reproduce the run is self-documenting in the config.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    description: str = ""
    mode: tp.Literal["decod", "encod"] = "decod"
    event_type: str = ""
    start: float = -0.1
    stop: float = 1.0
    neuro: dict[str, tp.Any] = {}
    extractor: dict[str, tp.Any] = {}
    model: dict[str, tp.Any] = {}
    cv: dict[str, tp.Any] = {}
    scoring: dict[str, tp.Any] = {}
    train_query: str | None = None
    test_query: tuple[str, ...] = ()
    reference_metric: float | None = None
    reference_metric_name: str = ""
    erp_display_filter: tuple[float | None, float | None] | None = (None, 40.0)
    erp_display_sfreq: float | None = 200.0
    trf: TRFConfig | None = None
    show_qc: bool = False
    infra: dict[str, tp.Any] | None = None


_VALIDATIONS_CACHE: dict[str, StudyValidation] | None = None


def discover_validations() -> dict[str, StudyValidation]:
    """Scan ``neuralfetch/validations/`` for ``*.toml`` config files.

    Each TOML file must contain a top-level ``study_name`` key whose value
    becomes the registry key.  All remaining keys are passed to
    :class:`StudyValidation` via ``model_validate``.

    Results are cached after the first call.  Use :func:`_clear_cache`
    to force a re-scan (useful in tests).

    Returns
    -------
    dict[str, StudyValidation]
        Mapping from study name to its validation config.
    """
    global _VALIDATIONS_CACHE  # noqa: PLW0603
    if _VALIDATIONS_CACHE is not None:
        return _VALIDATIONS_CACHE

    import neuralfetch.validations as _validations_pkg

    pkg_dir = Path(_validations_pkg.__file__).resolve().parent
    results: dict[str, StudyValidation] = {}
    for toml_file in sorted(pkg_dir.glob("*.toml")):
        try:
            with open(toml_file, "rb") as f:
                data = tomllib.load(f)
            study_name = data.pop("study_name")
            validation = StudyValidation.model_validate(data)
            results[study_name] = validation
        except Exception:
            logger.warning("Failed to load validation TOML %s", toml_file)

    _VALIDATIONS_CACHE = results
    return _VALIDATIONS_CACHE


def _clear_cache() -> None:
    """Reset the cached result of :func:`discover_validations`."""
    global _VALIDATIONS_CACHE  # noqa: PLW0603
    _VALIDATIONS_CACHE = None


def list_validatable_studies() -> dict[str, StudyValidation]:
    """Return all studies that have a validation config."""
    return discover_validations()


def _resolve_validation(
    name: str,
    validations: dict[str, StudyValidation] | None = None,
) -> tuple[tp.Type["study.Study"], StudyValidation]:
    """Look up the study class and its validation config by name.

    Matching is case-insensitive so ``thingsopm2025expanded`` resolves to
    ``ThingsOpm2025Expanded`` without requiring the caller to know the exact
    casing.
    """
    from neuralset.events import study

    if validations is None:
        validations = discover_validations()
    if name not in validations:
        # Case-insensitive lookup: prefer exact match, fall back to lower-case.
        lower_map = {k.lower(): k for k in validations}
        canonical = lower_map.get(name.lower())
        if canonical is None:
            available = ", ".join(sorted(validations)) or "(none)"
            raise ValueError(f"No validation config for {name!r}. Available: {available}")
        name = canonical
    validation = validations[name]
    cls = study.STUDIES.get(name)
    if cls is None:
        study._resolve_study(name)
        cls = study.STUDIES.get(name)
    if cls is None:
        raise ImportError(f"Study {name!r} not found.")
    return cls, validation
