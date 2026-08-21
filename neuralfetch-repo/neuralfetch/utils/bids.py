# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""BIDS export utility for neuralfetch studies.

Data-fetching utilities (STUDY_FOLDER, add_sentences, download_things_images,
etc.) live in ``neuralfetch.utils``.
"""

import errno
import logging
import shutil
import time
import typing as tp
from datetime import date, timedelta
from pathlib import Path

import exca
import mne
import numpy as np
import pandas as pd
import pydantic

import neuralset.events as ev
from neuralset.events import study as base

logger = logging.getLogger(__name__)

MNE_RAW_TYPES = {"Meg", "Eeg", "Emg", "Ieeg", "Fnirs"}
STIMULUS_FILE_TYPES = {"Sound", "Image", "Video"}

# mne_bids requires a concrete write format for preloaded data (our raws are
# always already loaded, never a passthrough file on disk) -- "auto" is only
# valid when writing straight from a source file. Each entry is a valid
# non-"auto" choice from mne_bids.config.CONVERT_FORMATS for that datatype.
_BIDS_WRITE_FORMAT = {
    "meg": "FIF",
    "eeg": "BrainVision",
    "ieeg": "BrainVision",
    "emg": "BDF",
}

# Class-level MapInfra instance used to wire the @apply decorator.
# Runtime execution uses the instance's infra_bids field (which callers can
# override, e.g. to switch from processpool to slurm).
_infra_bids = exca.MapInfra(cluster="processpool")


def _is_transient_lock_error(exc: BaseException) -> bool:
    """True for the NFS lock-file races seen in concurrent write_raw_bids calls.

    write_raw_bids guards its shared root-level files (participants.tsv/json,
    README) with ``<file>.lock`` files. On a cold-start export, many parallel
    jobs race to create those locks on a freshly created NFS directory, which
    surfaces transiently as either:

    - ``FileNotFoundError`` on a ``*.lock`` path (the lock's parent directory is
      not yet visible on the worker's NFS client), or
    - ``OSError`` with ``errno.ESTALE`` (Errno 116, stale NFS handle) when the
      lock-file cleanup races across processes.

    Both clear once the directory state settles, so they are safe to retry. Any
    other error is treated as genuine and re-raised immediately.
    """
    fname = str(getattr(exc, "filename", "") or "")
    if isinstance(exc, FileNotFoundError) and fname.endswith(".lock"):
        return True
    if isinstance(exc, OSError) and exc.errno == errno.ESTALE:
        return True
    return False


def _iter_subclasses(cls: type) -> tp.Iterator[type]:
    """Recursively yield all subclasses of cls."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _iter_subclasses(sub)


def _pad_label(label: str, width: int = 2) -> str:
    """Zero-pad *label* to *width* digits if it is purely numeric.

    Non-numeric labels (e.g. ``"sample"``, ``"control"``) are returned
    unchanged so that existing string-based subject/session IDs are not
    broken.
    """
    return label.zfill(width) if label.isdigit() else label


def _annotation_descriptions(df: pd.DataFrame) -> pd.Series:
    """Build annotation description strings from a categorical events DataFrame.

    Tries columns in priority order: ``description`` → ``state`` → ``stage``.
    Returns strings of the form ``"{type}/{label}"`` when a label is found,
    or just ``"{type}"`` when none of the label columns are present or non-empty.
    """

    def _label(row: pd.Series) -> str:
        for col in ("description", "state", "stage"):
            val = row.get(col)
            if val is not None and pd.notna(val) and str(val) != "":
                return f"{row['type']}/{val}"
        return str(row["type"])

    return df.apply(_label, axis=1)


def _write_participants_tsv(results: list[dict], path: Path) -> None:
    """Write participants.tsv once from per-timeline job results.

    Called by the coordinator after all parallel SLURM jobs finish.
    Each result dict must contain ``"subject"`` and ``"demographics"``
    (a dict with optional keys: birthday, sex, hand, weight, height using
    MNE integer codes).  De-duplicates by subject ID.
    """
    from mne_bids.write import _participants_tsv as _mne_participants_tsv

    seen: set[str] = set()
    tsv_path = str(path / "participants.tsv")

    for result in sorted(results, key=lambda r: r["subject"]):
        subject = result["subject"]
        if subject in seen:
            continue
        seen.add(subject)

        demographics = result.get("demographics", {})
        info = mne.create_info(ch_names=["STI"], sfreq=1000.0, ch_types=["stim"])
        subject_info = mne._fiff.meas_info.SubjectInfo(demographics)
        with info._unlock():
            info["subject_info"] = subject_info
        raw = mne.io.RawArray(np.zeros((1, 1)), info, verbose=False)

        _mne_participants_tsv(
            raw=raw,
            subject_id=subject,
            fname=tsv_path,
            overwrite=True,
        )


class BidsExporter(pydantic.BaseModel):
    """Export a Neuralset Study to BIDS format, optionally in parallel via SLURM.

    Parameters
    ----------
    path :
        Root directory for the BIDS output.
    device :
        Neurophysiology recording type.  Must be one of ``"Eeg"``,
        ``"Meg"``, ``"Ieeg"``, ``"Emg"``, or ``"Fnirs"``.
    task :
        BIDS task label. If ``None``, the ``"task"`` column in the events
        DataFrame is used.
    anonymize :
        Passed to ``mne_bids.write_raw_bids``.  Requires a ``daysback``
        key.  If ``None``, no anonymization is performed.
    overwrite :
        If ``True``, overwrite existing BIDS files.
    infra_bids :
        Caching/compute backend for per-timeline BIDS writes.  Uses a local
        process pool by default; set ``cluster="slurm"`` (with a ``folder``
        and SLURM parameters) to parallelise across cluster nodes.

    Examples
    --------
    Sequential (default)::

        BidsExporter(path="/data/bids", device="Meg", task="mytask").export(study)

    Parallel SLURM::

        BidsExporter(
            path="/data/bids",
            device="Meg",
            task="mytask",
            infra_bids=exca.MapInfra(
                cluster="slurm",
                folder="/tmp/bids_jobs",
                slurm_partition="learnfair",
                mem_gb=64,
                timeout_min=60,
            ),
        ).export(study)
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    path: Path
    device: str
    task: str | None = None
    anonymize: dict[str, tp.Any] | None = None
    overwrite: bool = False
    infra_bids: exca.MapInfra = _infra_bids

    # Private: set by export() so each SLURM job can re-register the source
    # study (required for SpecialLoader.from_json to reconstruct the raw loader).
    _study_cls_name: str = pydantic.PrivateAttr(default="")
    _study_module: str = pydantic.PrivateAttr(default="")
    _study_path: Path = pydantic.PrivateAttr(default_factory=Path)

    def export(self, study: base.Study) -> Path:
        """Run the full BIDS export for *study*.

        Returns the BIDS root path.
        """
        path = Path(self.path)
        if self.device not in MNE_RAW_TYPES:
            raise ValueError(
                f"{self.device!r} is not supported by mne_bids. "
                f"Must be one of {MNE_RAW_TYPES}."
            )

        # Store study identity so _export_timeline can register it in each job.
        self._study_cls_name = type(study).__name__
        self._study_module = type(study).__module__
        self._study_path = study.path

        events = study.run()
        grouped = [(tid, df) for tid, df in events.groupby("timeline")]

        # Create the BIDS root before dispatching jobs. write_raw_bids writes
        # shared root-level files (participants.tsv/json, README) guarded by a
        # .lock file at the root. When exporting to a fresh root in parallel, the
        # jobs would otherwise race to create that lock before any job has created
        # the root, failing with FileNotFoundError on the lock file.
        path.mkdir(parents=True, exist_ok=True)

        # _write_timelines dispatches each (timeline_id, DataFrame) pair to a
        # SLURM job (or local worker).  All per-timeline files are written
        # inside each job; participants.tsv is written once here after all
        # jobs finish to avoid concurrent read-modify-write corruption.
        results = list(self._write_timelines(grouped))
        _write_participants_tsv(results, path)

        n_timelines = events["timeline"].nunique()
        n_subjects = events["subject"].nunique()
        n_stimulus_files = (
            events[events["type"].isin(STIMULUS_FILE_TYPES)]["filepath"].nunique()
            if "filepath" in events.columns
            else 0
        )
        logger.info(
            "BIDS export complete: %d timeline(s), %d subject(s), %d stimulus file(s) "
            "written to %s",
            n_timelines,
            n_subjects,
            n_stimulus_files,
            path,
        )
        return path

    @_infra_bids.apply(
        item_uid=lambda item: item[0],  # timeline_id — unique cache key per timeline
        cache_type="Pickle",  # status cache; enables resume on failure
    )
    def _write_timelines(
        self, items: tp.Iterable[tuple[str, pd.DataFrame]]
    ) -> tp.Iterator[dict]:
        """Dispatch one _export_timeline call per timeline.

        Decorated with ``@_infra_bids.apply`` so each item is processed by a
        separate worker (SLURM job or local process).  Results are cached,
        enabling resume-on-failure without re-exporting completed timelines.
        """
        for timeline_id, timeline_df in items:
            participant_info = self._export_timeline(timeline_df)
            yield {"timeline": timeline_id, **participant_info}

    def _export_timeline(self, timeline_df: pd.DataFrame) -> dict:
        """Read one timeline and write its BIDS files. Runs in a SLURM job.

        All files written here have unique per-timeline paths, so parallel
        execution is safe without locking.  participants.tsv is intentionally
        excluded — the coordinator writes it once after all jobs finish.

        Returns per-subject demographics so the coordinator can write
        participants.tsv correctly.
        """
        import mne_bids  # deferred: not a default dependency

        # Re-register the source study in this process so SpecialLoader.from_json
        # can reconstruct it.  STUDIES is populated by importing the class module;
        # STUDY_PATHS is populated only by constructing an instance.
        # We import by exact module path (not a full package scan) to avoid
        # triggering import errors in unrelated study modules.
        if self._study_cls_name:
            import importlib

            from neuralset.events.study import STUDIES, STUDY_PATHS

            if self._study_cls_name not in STUDY_PATHS:
                if self._study_cls_name not in STUDIES and self._study_module:
                    importlib.import_module(self._study_module)
                scls = STUDIES.get(self._study_cls_name)
                if scls is not None:
                    scls(path=self._study_path)  # registers STUDY_PATHS entry

        event_cls = getattr(ev.etypes, self.device)
        datatype = self.device.lower()
        categorical_types = frozenset(
            cls.__name__ for cls in _iter_subclasses(ev.etypes.CategoricalEvent)
        )

        raw_row = timeline_df.query("type == @self.device").iloc[0]
        raw = event_cls.from_dict(raw_row).read()

        if raw.get_montage() is None:
            logger.warning(
                "No channel positions found in raw for timeline %s; "
                "electrodes.tsv will not be written.",
                timeline_df["timeline"].iloc[0],
            )

        if self.anonymize:
            raw.anonymize(**self.anonymize)

        # Build Annotations from all CategoricalEvent subtypes and attach to raw
        cat_df = timeline_df[timeline_df["type"].isin(categorical_types)]
        if not cat_df.empty:
            annotations = mne.Annotations(
                onset=cat_df["start"].values,
                duration=cat_df["duration"].values,
                description=_annotation_descriptions(cat_df).values,
                orig_time=raw.info["meas_date"],
            )
            raw.set_annotations(annotations)

        # Resolve subject — strip study-name prefix e.g. "Mne2013Sample/sample" -> "sample"
        subject = _pad_label(timeline_df["subject"].iloc[0].split("/")[-1])

        # Resolve optional BIDSPath fields from events columns
        # TODO: remove if all Studies are now BIDS compliant
        session = (
            _pad_label(str(timeline_df["session"].iloc[0]))
            if "session" in timeline_df.columns and timeline_df["session"].iloc[0]
            else None
        )
        run = (
            _pad_label(str(timeline_df["run"].iloc[0]))
            if "run" in timeline_df.columns and timeline_df["run"].iloc[0]
            else None
        )

        # TODO: consider making these enum classes
        sex_map = {"unknown": 0, "male": 1, "female": 2}
        hand_map = {"right": 1, "left": 2, "ambidextrous": 3}
        demographics: dict[str, tp.Any] = {}
        for col in ["age", "sex", "hand", "weight", "height"]:
            if col in timeline_df.columns:
                if col == "age":
                    # MNE stores a birthday, not an age; approximate it from the
                    # reported age (leap years are ignored, which is fine since
                    # this is only used for anonymized day-shifted dates).
                    demographics["birthday"] = date.today() - timedelta(
                        days=int(timeline_df["age"].iloc[0]) * 365
                    )
                elif col == "sex":
                    demographics["sex"] = sex_map.get(timeline_df[col].iloc[0], 0)
                elif col == "hand":
                    demographics["hand"] = hand_map.get(timeline_df[col].iloc[0], 0)
                else:
                    demographics[col] = float(timeline_df[col].iloc[0])

        subject_info = mne._fiff.meas_info.SubjectInfo(demographics)
        raw.info["subject_info"] = subject_info

        # Resolve task: parameter > events column > study class name
        if self.task is not None:
            resolved_task = self.task
        elif "task" in timeline_df.columns:
            resolved_task = str(timeline_df["task"].iloc[0])
        else:
            resolved_task = None
        if not resolved_task:
            raise ValueError(
                f"Task not found for timeline {timeline_df['timeline'].iloc[0]}. "
                "Please provide a task name as a parameter or add a 'task' column to the events DataFrame."
            )

        bids_path = mne_bids.BIDSPath(
            subject=subject,
            session=session,
            task=resolved_task,
            run=run,
            datatype=datatype,
            suffix=datatype,
            root=self.path,
        )

        try:
            write_format = _BIDS_WRITE_FORMAT[datatype]
        except KeyError:
            raise ValueError(
                f"No BIDS write format configured for datatype {datatype!r} "
                f"(device={self.device!r}). Add an entry to _BIDS_WRITE_FORMAT."
            ) from None

        # All files written by write_raw_bids have unique per-timeline paths —
        # fully safe to run in parallel.  participants.tsv is written by the
        # coordinator after all jobs finish (see export()).
        #
        # The shared root-level files (participants.tsv/json, README) are guarded
        # by .lock files; on a cold-start parallel export, a few jobs transiently
        # lose the race to create those locks on the fresh NFS directory. Retry
        # with backoff on just those transient lock errors — overwrite=True makes
        # the retry idempotent.
        _MAX_ATTEMPTS = 6
        for attempt in range(_MAX_ATTEMPTS):
            try:
                mne_bids.write_raw_bids(
                    raw=raw,
                    bids_path=bids_path,
                    overwrite=self.overwrite,
                    allow_preload=True,
                    format=write_format,
                )
                break
            except (FileNotFoundError, OSError) as exc:
                if not _is_transient_lock_error(exc) or attempt == _MAX_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "Transient BIDS lock error (%s) writing %s; retrying "
                    "(attempt %d/%d).",
                    exc,
                    bids_path.basename,
                    attempt + 1,
                    _MAX_ATTEMPTS,
                )
                time.sleep(0.5 * 2**attempt)

        # Copy stimulus files to <root>/stimuli/ and write their onsets to events.tsv
        path = Path(self.path)
        stim_file_df = (
            timeline_df[
                timeline_df["type"].isin(STIMULUS_FILE_TYPES)
                & timeline_df["filepath"].notna()
            ]
            if "filepath" in timeline_df.columns
            else pd.DataFrame()
        )
        if not stim_file_df.empty:
            stimuli_dir = path / "stimuli"
            stimuli_dir.mkdir(exist_ok=True)

            for src_path in stim_file_df["filepath"].unique():
                src = Path(src_path)
                dst = stimuli_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

            # Write stimulus event rows to the BIDS events.tsv sidecar.
            # Columns internal to the pipeline (timeline, subject, session, run,
            # task, type) are dropped; everything else (category, stem, split,
            # caption, …) passes through as extra BIDS columns.
            _INTERNAL_COLS = {
                "timeline",
                "subject",
                "session",
                "run",
                "task",
                "type",
                "filepath",
            }
            bids_events = stim_file_df.copy()
            bids_events["stim_file"] = bids_events["filepath"].apply(
                lambda p: f"stimuli/{Path(p).name}"
            )
            bids_events = bids_events.rename(columns={"start": "onset"})
            bids_events = bids_events.drop(
                columns=[c for c in _INTERNAL_COLS if c in bids_events.columns]
            )
            bids_events = bids_events.sort_values("onset").reset_index(drop=True)

            events_tsv_path = (
                bids_path.copy().update(suffix="events", extension=".tsv").fpath
            )
            if events_tsv_path.exists():
                # Merge with events already written by write_raw_bids (CategoricalEvents)
                existing = pd.read_csv(events_tsv_path, sep="\t")
                bids_events = (
                    pd.concat([existing, bids_events], ignore_index=True)
                    .sort_values("onset")
                    .reset_index(drop=True)
                )
            bids_events.to_csv(events_tsv_path, sep="\t", index=False)

        return {"subject": subject, "demographics": demographics}


def study_to_bids(
    study: base.Study,
    path: Path,
    device: str,
    task: str | None = None,
    anonymize: dict[str, tp.Any] | None = None,
    overwrite: bool = False,
    infra_bids: exca.MapInfra | None = None,
) -> Path:
    """Export a Neuralset Study to BIDS format.

    Convenience wrapper around :class:`BidsExporter`.  Pass ``infra_bids``
    to parallelise per-timeline writes across SLURM or local workers.

    Currently supports neurophysiology modalities only: EEG, MEG, iEEG,
    EMG, and fNIRS.  Neuroimaging modalities such as fMRI are not yet
    supported.

    Parameters
    ----------
    study :
        A neuralset Study instance.
    path :
        Root directory for the BIDS output.
    device :
        Neurophysiology recording type.  Must be one of ``"Eeg"``,
        ``"Meg"``, ``"Ieeg"``, ``"Emg"``, or ``"Fnirs"``.
    task :
        BIDS task label. If ``None``, the ``"task"`` column in the events
        DataFrame is used.
    anonymize :
        Follows the format of the ``anonymize`` parameter in
        ``mne_bids.write_raw_bids``.  Requires a ``daysback`` key.
        If ``None``, no anonymization is performed.
    overwrite :
        If ``True``, overwrite existing BIDS files.
    infra_bids :
        Optional compute/cache backend.  If ``None``, a local process pool
        is used.  Pass ``exca.MapInfra(cluster="slurm", ...)`` to dispatch
        each timeline to a SLURM job.

    Returns
    -------
    Path
        The BIDS root directory (``path``).
    """
    kw: dict[str, tp.Any] = {}
    if infra_bids is not None:
        kw["infra_bids"] = infra_bids
    return BidsExporter(
        path=path,
        device=device,
        task=task,
        anonymize=anonymize,
        overwrite=overwrite,
        **kw,
    ).export(study)
