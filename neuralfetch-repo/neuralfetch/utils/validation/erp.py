# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Helpers for loading raw EEG/MEG data and building per-subject ERP/ERF epochs."""

from __future__ import annotations

import logging
import typing as tp
from collections.abc import Iterator

import numpy as np

from .config import StudyValidation

if tp.TYPE_CHECKING:
    import mne
    import neuralset as ns
    import pandas as pd

logger = logging.getLogger(__name__)


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
    from neuralset.events import etypes
    from neuralset.events.utils import extract_events

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
