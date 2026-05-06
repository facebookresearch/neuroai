# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Study validation runner.

Run a study's validation analysis using :mod:`neuralyze` and generate
an MNE Report (interactive HTML).  The CLI wiring lives in
:mod:`neuralfetch.cli.validate`; this module is library-only.
"""

from __future__ import annotations

import datetime
import html
import importlib.metadata
import logging
import pprint
import tomllib
import typing as tp
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pydantic
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib import pyplot as plt
from tqdm import tqdm

import neuralset as ns
from neuralfetch.utils.base import root_study_folder
from neuralset.events import etypes, study
from neuralset.events.utils import extract_events

if tp.TYPE_CHECKING:
    import mne
    import pandas as pd
    import xarray

logger = logging.getLogger(__name__)

_INFRA_PROPAGATE_KEYS = (
    "cluster",
    "slurm_partition",
    "timeout_min",
    "folder",
    "mode",
    "cpus_per_task",
    "mem_gb",
)


# ---------------------------------------------------------------------------
# Validation framework
# ---------------------------------------------------------------------------


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
    n_pca : int, float, or None
        PCA applied to the stimulus features before fitting the TRF.  Required
        when the extractor output dimension is large (e.g. 384 for DINOv2-small,
        768 for DINOv2-base): the TRF delay matrix grows as
        ``n_samples × n_features × n_lags``, which becomes infeasible at full
        dimensionality.

        * **int** — keep exactly that many components (e.g. ``50``).
        * **float in (0, 1)** — keep the minimum number of components that
          explain at least that fraction of variance (e.g. ``0.95`` for 95 %).
          This is the recommended setting: the number of components adapts to
          the actual variance structure of the stimulus set.
        * **None** — disable PCA (only viable for low-dimensional features).
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    tmin: float = -0.2
    tmax: float = 0.5
    aggregation: str = "sum"
    n_pca: int | float | None = None


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
    event_label_column : str
        Name of the events DataFrame column to use as labels in the Events
        Map section of the report.  Each unique value becomes a distinct
        colour/level in ``mne.Report.add_events``.  Defaults to
        ``"description"``.  If the column is absent from the events
        DataFrame (or contains only nulls), falls back to a single level
        named after ``event_type``.
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
    event_label_column: str = "description"
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
) -> tuple[tp.Type[study.Study], StudyValidation]:
    """Look up the study class and its validation config by name.

    Matching is case-insensitive so ``thingsopm2025expanded`` resolves to
    ``ThingsOpm2025Expanded`` without requiring the caller to know the exact
    casing.
    """
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


# ---------------------------------------------------------------------------
# SlidingWindow builder
# ---------------------------------------------------------------------------


def _build_sliding_window(
    study_name: str,
    validation: StudyValidation,
    study_folder: Path,
    cache_dir: Path | None = None,
    query: str | None = None,
    infra: dict[str, tp.Any] | None = None,
) -> tp.Any:
    """Construct a :class:`neuralyze.SlidingWindow` from a validation config."""
    from neuralyze import SlidingWindow

    study_config: dict[str, tp.Any] = {
        "name": study_name,
        "path": str(study_folder),
        # Run timeline loading sequentially to avoid a known upstream bug in
        # ``neuralset.events.study.Study.__setstate__`` that crashes processpool
        # workers with ``AttributeError: ... has no attribute 'path'`` (see
        # fairinternal/brainai#2822). Timeline loading is I/O-bound and typically
        # fine sequentially for validation runs.
        "infra_timelines": {"cluster": None},
    }
    if query is not None:
        study_config["query"] = query
    if cache_dir is not None:
        infra = dict(infra) if infra is not None else {}
        infra.setdefault("folder", str(cache_dir))
        retry = infra.pop("retry", False)
        study_infra: dict[str, tp.Any] = {"backend": "Cached", "folder": str(cache_dir)}
        if retry:
            study_infra["mode"] = "retry"
        study_config["infra"] = study_infra

    neuro_dict = dict(validation.neuro)
    extractor_dict = dict(validation.extractor)
    if infra is not None:
        for d in (neuro_dict, extractor_dict):
            sub_infra = d.setdefault("infra", {})
            for key in _INFRA_PROPAGATE_KEYS:
                if key in infra:
                    sub_infra.setdefault(key, infra[key])

    data_config: dict[str, tp.Any] = {
        "study": study_config,
        "start": validation.start,
        "stop": validation.stop,
        "event_type": validation.event_type,
        "neuro": neuro_dict,
        "extractor": extractor_dict,
    }

    sw_kwargs: dict[str, tp.Any] = {
        "data": data_config,
        "mode": validation.mode,
        "model": dict(validation.model),
        "scoring": dict(validation.scoring),
    }
    if validation.cv:
        sw_kwargs["cv"] = dict(validation.cv)
    if validation.train_query is not None:
        sw_kwargs["train_query"] = validation.train_query
    if validation.test_query:
        sw_kwargs["test_query"] = list(validation.test_query)

    if infra is not None:
        sw_kwargs["infra"] = infra

    return SlidingWindow(**sw_kwargs)


def _build_trf_scoring(
    study_name: str,
    validation: StudyValidation,
    study_folder: Path,
    cache_dir: Path | None = None,
    query: str | None = None,
    infra: dict[str, tp.Any] | None = None,
) -> tp.Any:
    """Construct a :class:`neuralyze.trf.TRFScoring` from a validation config.

    Reuses ``validation.neuro``, ``validation.extractor``, ``validation.scoring``,
    and ``validation.cv`` from the parent :class:`StudyValidation`, overriding
    the extractor's ``aggregation`` key with ``validation.trf.aggregation``.  The
    CV config is also patched to include ``group="filepath"`` if not already set,
    which is required for TRF to avoid data leakage across autocorrelated segments.

    ``mode`` is always forced to ``"encod"`` because a TRF is inherently an
    encoding model (features → brain), regardless of the parent
    :class:`StudyValidation`'s ``mode``.

    When ``validation.trf.n_pca`` is set, a local subclass of
    :class:`~neuralyze.trf.TRFScoring` is used that applies PCA to the
    extractor features before each TRF fit, reducing dimensionality in a
    cross-validation-safe manner (fit on train, transform both train and test).
    """
    from neuralyze.trf import TRFScoring

    class _TRFScoringWithPCA(TRFScoring):
        n_pca: int | float | None = None

        def _fit_predict(
            self,
            X: tp.Any,
            Y: tp.Any,
            X_test: tp.Any,
            frequency: float,
        ) -> tp.Any:
            if self.n_pca is not None:
                from sklearn.decomposition import PCA

                pca = PCA(n_components=self.n_pca)
                X = pca.fit_transform(X)
                X_test = pca.transform(X_test)
            return super()._fit_predict(X, Y, X_test, frequency)

    assert validation.trf is not None, "trf config must be set"
    trf_cfg = validation.trf

    study_config: dict[str, tp.Any] = {
        "name": study_name,
        "path": str(study_folder),
        "infra_timelines": {"cluster": None},
    }
    if query is not None:
        study_config["query"] = query
    if cache_dir is not None:
        infra = dict(infra) if infra is not None else {}
        infra.setdefault("folder", str(cache_dir))
        retry = infra.pop("retry", False)
        study_infra: dict[str, tp.Any] = {"backend": "Cached", "folder": str(cache_dir)}
        if retry:
            study_infra["mode"] = "retry"
            infra["mode"] = "retry"
        study_config["infra"] = study_infra

    neuro_dict = dict(validation.neuro)
    extractor_dict = dict(validation.extractor)
    extractor_dict["aggregation"] = trf_cfg.aggregation
    if "frequency" in neuro_dict:
        extractor_dict.setdefault("frequency", neuro_dict["frequency"])
    if infra is not None:
        for d in (neuro_dict, extractor_dict):
            sub_infra = d.setdefault("infra", {})
            for key in _INFRA_PROPAGATE_KEYS:
                if key in infra:
                    sub_infra.setdefault(key, infra[key])

    data_config: dict[str, tp.Any] = {
        "study": study_config,
        "event_type": validation.event_type,
        "neuro": neuro_dict,
        "extractor": extractor_dict,
    }

    cv_dict = dict(validation.cv)
    if "group" not in cv_dict:
        cv_dict["group"] = "filepath"
        cv_dict["shuffle"] = False

    trf_kwargs: dict[str, tp.Any] = {
        "data": data_config,
        "trf": {"tmin": trf_cfg.tmin, "tmax": trf_cfg.tmax},
        "mode": "encod",
        "scoring": dict(validation.scoring),
        "cv": cv_dict,
    }
    if trf_cfg.n_pca is not None:
        trf_kwargs["n_pca"] = trf_cfg.n_pca
    if infra is not None:
        trf_kwargs["infra"] = infra

    return _TRFScoringWithPCA(**trf_kwargs)


# ---------------------------------------------------------------------------
# Plotting helpers
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


def _evoked_kind_label(neuro_event_type: str, info: "mne.Info") -> str:
    """Return ``"ERP"``, ``"ERF"``, or ``"Evoked"`` from a modality cue.

    Priority:

    1. ``neuro_event_type`` (from :meth:`Study.neuro_types`) resolves
       most cases directly -- ``"Meg"`` -> ``"ERF"``, ``"Eeg"``/``"Ieeg"``
       -> ``"ERP"``.
    2. Otherwise fall back to the dominant channel type reported by
       :func:`mne.channel_type` across ``info["ch_names"]``.
    3. If nothing matches, return ``"Evoked"`` so the report still
       renders something meaningful.
    """
    import mne

    et = (neuro_event_type or "").strip()
    if et == "Meg":
        return "ERF"
    if et in {"Eeg", "Ieeg"}:
        return "ERP"

    try:
        types = [mne.channel_type(info, i) for i in range(len(info["ch_names"]))]
    except Exception:
        return "Evoked"
    tset = set(types)
    if tset & {"mag", "grad"}:
        return "ERF"
    if tset & {"eeg", "seeg", "ecog"}:
        return "ERP"
    return "Evoked"


def _iter_subject_raws(
    study_instance: "ns.Study",
    events: "pd.DataFrame",
    subject_id: str,
    neuro_type_names: tp.Iterable[str],
    trigger_event_type: str,
) -> Iterator[tuple[tp.Any, np.ndarray, str]]:
    """Yield ``(raw, trigger_seconds, neuro_event_type_name)`` per timeline.

    Uses only public :mod:`neuralset` / :mod:`mne` entry points:
    :func:`neuralset.events.utils.extract_events` and ``event.read()``.
    ``trigger_seconds`` are trigger start times *relative to the raw's
    data array* (i.e. with ``first_samp / sfreq`` subtracted), so the
    caller can convert to sample indices via :meth:`mne.io.Raw.time_as_index`
    after any filter/resample.
    """
    del study_instance  # currently unused; kept for API symmetry / future extensions
    neuro_names = set(neuro_type_names)
    subj_df = events.loc[events.subject == subject_id]
    if subj_df.empty:
        return

    for timeline, tl_df in subj_df.groupby("timeline"):
        neuro_df = tl_df.loc[tl_df.type.isin(neuro_names)]
        trig_df = tl_df.loc[tl_df.type == trigger_event_type]
        if neuro_df.empty or trig_df.empty:
            continue

        neuro_event_list = extract_events(neuro_df)
        if not neuro_event_list:
            continue
        neuro_event = neuro_event_list[0]
        neuro_type_name = type(neuro_event).__name__
        if not isinstance(neuro_event, etypes.BaseDataEvent):
            # Only file-backed events (MneRaw, Audio, ...) expose ``.read()``;
            # anything else can't yield a usable Raw for ERP/ERF plots.
            continue

        try:
            raw = neuro_event.read()
        except Exception as exc:  # pragma: no cover - IO errors surface to caller
            logger.warning(
                "Failed to read raw for participant %s timeline %s: %s",
                subject_id,
                timeline,
                exc,
            )
            continue

        sfreq = float(raw.info["sfreq"])
        raw_onset = raw.first_samp / sfreq
        rel_times = np.asarray(trig_df["start"].to_numpy(), dtype=float) - raw_onset
        yield raw, rel_times, neuro_type_name


def _build_subject_epochs(
    study_instance: "ns.Study",
    events: "pd.DataFrame",
    validation: StudyValidation,
    subject_id: str,
) -> tuple[tp.Any, str] | None:
    """Build concatenated ``mne.Epochs`` + ``"ERP"``/``"ERF"`` label for a subject.

    Applies the display-only filter/resample from ``validation``.  Returns
    ``None`` if the subject has no usable neural timeline or zero trigger
    events.
    """
    import mne

    study_cls = type(study_instance)
    neuro_names = study_cls.neuro_types()

    epoch_list: list[tp.Any] = []
    label: str | None = None
    for raw, rel_seconds, neuro_name in _iter_subject_raws(
        study_instance,
        events,
        subject_id,
        neuro_names,
        validation.event_type,
    ):
        if rel_seconds.size == 0:
            continue

        if validation.erp_display_filter is not None:
            lo, hi = validation.erp_display_filter
            if lo is not None or hi is not None:
                raw = raw.copy().load_data().filter(lo, hi, verbose=False)
        if validation.erp_display_sfreq is not None and (
            raw.info["sfreq"] > validation.erp_display_sfreq
        ):
            raw = (
                raw.copy()
                .load_data()
                .resample(validation.erp_display_sfreq, verbose=False)
            )

        sfreq = float(raw.info["sfreq"])
        max_t = (len(raw.times) - 1) / sfreq
        mask = (rel_seconds >= 0.0) & (rel_seconds <= max_t)
        clipped = rel_seconds[mask]
        if clipped.size == 0:
            continue
        rel_samples = raw.time_as_index(clipped)
        abs_samples = np.asarray(rel_samples, dtype=int) + int(raw.first_samp)

        events_arr = np.column_stack(
            [
                abs_samples,
                np.zeros(abs_samples.size, dtype=int),
                np.ones(abs_samples.size, dtype=int),
            ]
        )
        # Apply pre-trigger baseline correction only when the decoding
        # window actually extends below 0; otherwise the single-sample
        # baseline (e.g. start=0) would trigger an MNE error.
        baseline = (None, 0.0) if validation.start < 0 else None
        try:
            epochs = mne.Epochs(
                raw,
                events_arr,
                event_id={validation.event_type: 1},
                tmin=validation.start,
                tmax=validation.stop,
                baseline=baseline,
                preload=True,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("mne.Epochs failed for %s: %s", subject_id, exc)
            continue
        if len(epochs) == 0:
            continue
        epoch_list.append(epochs)
        if label is None:
            label = _evoked_kind_label(neuro_name, epochs.info)

    if not epoch_list:
        return None
    merged = (
        epoch_list[0]
        if len(epoch_list) == 1
        else mne.concatenate_epochs(epoch_list, verbose=False)
    )
    return merged, label or "Evoked"


#: Fixed topomap times (in seconds) for the per-subject ``plot_joint``
#: figures.  Stimulus onset, 100 ms, and 200 ms were chosen because they
#: bracket the most informative short-latency ERP/ERF components
#: (N1/P1/N170 families).  Values outside an evoked's actual time range
#: are filtered out at render time.
_JOINT_TOPOMAP_TIMES: tuple[float, ...] = (0.0, 0.1, 0.2)


def _plot_subject_evoked(
    epochs: tp.Any,
    subject_id: str,
    label: str,
) -> plt.Figure:
    """Average *epochs* and render an ``evoked.plot_joint`` figure."""
    evoked = epochs.average()
    times = [
        t for t in _JOINT_TOPOMAP_TIMES if evoked.tmin - 1e-9 <= t <= evoked.tmax + 1e-9
    ]
    kwargs: dict[str, tp.Any] = {
        "show": False,
        "title": f"Participant {subject_id.split('/', 1)[-1] if '/' in subject_id else subject_id} - {label}",
    }
    if times:
        kwargs["times"] = times
    figs = evoked.plot_joint(**kwargs)
    if isinstance(figs, (list, tuple)):
        return figs[0]
    return figs


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
                legendgrouptitle_text="Participants" if full_id == subject_order[0] else "",
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

    score_label = validation.reference_metric_name or "Score"
    fig.update_layout(
        title="Group Grand Average",
        xaxis_title="Time (s)",
        yaxis_title=score_label,
        height=480,
        margin=dict(l=60, r=20, t=50, b=160),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="left",
            x=0,
            font=dict(size=11),
            tracegroupgap=4,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#ebebeb"),
        yaxis=dict(showgrid=True, gridcolor="#ebebeb"),
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
    grand_mean = float(np.mean([r["peak_score"] for r in rows]))
    rows.append({"subject": "Grand Average", "peak_score": grand_mean, "peak_time": ""})
    return rows


def _summary_table_html(rows: list[dict[str, tp.Any]]) -> str:
    """Render the scores summary as an HTML table."""
    lines = [
        "<table>",
        "<thead><tr><th>Participant</th><th>Peak Score</th><th>Peak Time (s)</th></tr></thead>",
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
        bold = ' style="font-weight:bold"' if row["subject"] == "Grand Average" else ""
        lines.append(
            f"<tr{bold}><td>{row['subject']}</td><td>{peak}</td><td>{time}</td></tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


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
    import mne

    subj_scores = trf_scores.sel(subject=subject_id)
    if "split" in subj_scores.dims:
        subj_scores = subj_scores.mean(dim="split")
    scores_np = np.atleast_1d(np.asarray(subj_scores.values).squeeze())

    short_id = subject_id.split("/", 1)[-1] if "/" in subject_id else subject_id

    has_topomap = False
    if info is not None:
        try:
            pos = mne.channels.layout._find_topomap_coords(info, picks=None)  # type: ignore[attr-defined]
            has_topomap = pos is not None and len(pos) == len(scores_np)
        except Exception:
            has_topomap = False

    ncols = 2 if has_topomap else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    ax_bar = axes[0]
    ax_bar.bar(range(len(scores_np)), scores_np, color="steelblue", width=0.8)
    ax_bar.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax_bar.set_xlabel("Channel index")
    ax_bar.set_ylabel("Pearson r")
    ax_bar.set_title(f"Participant {short_id} — TRF per-channel scores")

    if has_topomap:
        ax_topo = axes[1]
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
            logger.warning("TRF topomap failed for %s: %s", subject_id, exc)
            ax_topo.set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# MNE Report generator
# ---------------------------------------------------------------------------


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
    study_cls: tp.Type[study.Study],
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
            "n_pca": validation.trf.n_pca,
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
    study_cls: tp.Type[study.Study],
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
                classes="table table-sm",
                border=True,
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
        html = _plot_group_grand_average_html(scores, validation)
        report.add_html(
            html,
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
    per_subject_figs: dict[str, tuple[plt.Figure, str, "mne.Info | None"]] = {}
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

        # Phase 2 — build figures on the main thread.
        for subject_id in tqdm(
            subjects_scored,
            desc="Building ERP/ERF figures",
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
                fig = _plot_subject_evoked(epochs, subject_id, label)
            except Exception as exc:  # pragma: no cover
                logger.warning("plot_joint failed for %s: %s", subject_id, exc)
                failed_subjects.add(subject_id)
                continue
            per_subject_figs[subject_id] = (fig, label, epochs.info)

    # 5. Subjects x Channels drop grid
    if events is not None:
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

    # 5b. Events map
    if events is not None and validation.event_type:
        try:
            sfreq = float(validation.erp_display_sfreq or 1000.0)
            target_type = validation.event_type
            filtered = events[events["type"] == target_type]
            label_col = validation.event_label_column
            if (
                label_col
                and label_col in filtered.columns
                and filtered[label_col].notna().any()
            ):
                unique_labels = sorted(
                    filtered[label_col].dropna().unique().tolist(), key=str
                )
                label_to_id = {str(lbl): i + 1 for i, lbl in enumerate(unique_labels)}
                event_id = dict(label_to_id)
                ids = (
                    filtered[label_col]
                    .map(lambda x: label_to_id.get(str(x), 1))
                    .to_numpy(dtype=int)
                )
            else:
                event_id = {target_type: 1}
                ids = np.ones(len(filtered), dtype=int)
            samples = (filtered["start"].to_numpy(dtype=float) * sfreq).astype(int)
            events_arr = np.column_stack(
                [
                    samples,
                    np.zeros(len(samples), dtype=int),
                    ids,
                ]
            )
            report.add_events(
                events_arr,
                title="Events Map",
                event_id=event_id,
                sfreq=sfreq,
                tags=("summary",),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Events map build failed: %s", exc)

    # 6. Results summary table
    summary_rows = _scores_summary_table(scores)
    summary_html = _summary_table_html(summary_rows)
    report.add_html(summary_html, title="Results Summary", tags=("summary",))

    # 7. Per-subject figures: score plot + (optional) Evoked plot_joint
    for subject_id in tqdm(
        subjects_scored,
        desc="Attaching per-participant figures",
        ncols=100,
        leave=False,
    ):
        # Strip study-name prefix (e.g. "StudyName/1" -> "1")
        short_id = subject_id.split("/", 1)[-1] if "/" in subject_id else subject_id
        fig = _plot_subject_scores(scores, subject_id, validation)
        report.add_figure(fig, title=f"Participant {short_id}", tags=("per-participant",))
        plt.close(fig)
        if subject_id in per_subject_figs:
            evoked_fig, label, _ = per_subject_figs[subject_id]
            report.add_figure(
                evoked_fig,
                title=f"Participant {short_id} - {label}",
                tags=("per-participant", label.lower()),
            )
            plt.close(evoked_fig)

    # 8. Per-subject TRF encoding figures (only when trf_scores provided)
    if trf_scores is not None:
        trf_subjects = sorted(
            trf_scores.coords["subject"].values.tolist(), key=_subject_sort_key
        )
        for subject_id in tqdm(
            trf_subjects,
            desc="Building TRF figures",
            ncols=100,
            leave=False,
        ):
            short_id = subject_id.split("/", 1)[-1] if "/" in subject_id else subject_id
            # Reuse the mne.Info stored from the ERP pre-build loop so the
            # raw file is not read a second time just for the topomap.
            info: "mne.Info | None" = (
                per_subject_figs[subject_id][2] if subject_id in per_subject_figs else None
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


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------


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
