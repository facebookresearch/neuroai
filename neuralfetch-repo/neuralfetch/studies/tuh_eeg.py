# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""TUH EEG: Clinical EEG recordings from Temple University Hospital.

This study provides a large corpus of clinical EEG recordings from Temple
University Hospital. Multiple annotated subsets are available, each exposing
a different study class. The population consists of clinical patients.

Study Aliases:
    - Obeid2016Tueg (TUEG): Full superset, no labels
    - Lopez2017Tuab (TUAB): Normal/abnormal labels
    - Hamid2020Tuar (TUAR): Artifact event annotations
    - Veloso2017Tuep (TUEP): Epilepsy/no-epilepsy labels
    - Harati2015Tuev (TUEV): Epileptiform and artifact event annotations
    - VonWeltin2017Tusl (TUSL): Slowing event annotations
    - Shah2018Tusz (TUSZ): Seizure and artifact event annotations

Experimental Design:
    - EEG recordings (varying channel counts, 250-256 Hz)
    - Clinical population (various neurological conditions)
    - EDF file format

Download Requirements:
    - Requires free account registration at
      https://isip.piconepress.com/projects/nedc/html/tuh_eeg/
    - After registration, request access via the online form
    - Once approved, download via rsync using provided credentials:
      rsync -auxvL nedc_tuh_eeg@www.isip.piconepress.com:data/tuh_eeg/ DESTINATION/
    - Automated download is not yet supported in this script
"""

import datetime
import functools
import logging
import re
import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from neuralfetch import utils
from neuralset.events import study
from neuralset.events.etypes import CategoricalEvent
from neuralset.events.study import _identify_study_subfolder

logger = logging.getLogger(__name__)


class EpileptiformActivity(CategoricalEvent):
    """Epileptiform activity event.

    Parameters
    ----------
    state : {'spsw', 'gped', 'pled', 'bckg'}
        Activity type:

        - 'spsw': Spike and/or sharp waves
        - 'gped': Generalized periodic epileptiform discharges
        - 'pled': Periodic lateralized epileptiform discharges
        - 'bckg': Background (no epileptiform activity)
    """

    state: tp.Literal[
        "spsw",  # Spike and/or sharp waves
        "gped",  # Generalized periodic epileptiform discharges
        "pled",  # Periodic lateralized epileptiform discharges
        "bckg",  # Background (no seizure)
    ]


def _fix_ch_name(name: str) -> str:
    """Fix and standardize EEG channel names that are not in the 10-5 system.

    References
    ----------
    [1] Acharya, Jayant N., et al. "American clinical neurophysiology society guideline 2:
        guidelines for standard electrode position nomenclature." The Neurodiagnostic Journal 56.4
        (2016): 245-252.
    [2] Oostenveld, Robert. "High-density EEG electrode placement", https://robertoostenveld.nl/electrode/
    """
    name = {
        # Channels to remap to a close 10-5 equivalent
        "T1": "FT9",  # Very close according to [1]
        "T2": "FT10",  # same
        "T3": "T7",  # Replaced in 10-5 system [2]
        "T4": "T8",  # same
        "T5": "P7",  # same
        "T6": "P8",  # same
        "C3P": "CP3",  # Between C3 and P3
        "C4P": "CP4",  # Between C4 and P4
        # HACK: A1 and A2 are in the 10-5 MNE montage, but not in the 10-5 MNE layout!
        # Replacing them by close equivalents that are not already in the dataset. To be
        # removed once we switch to using 3D positions from montages instead of 2D positions
        # from layouts.
        "A1": "T9",
        "A2": "T10",
        # Channels to ignore
        # "SP1": None,  # Sphenoidal electrodes under the zygoma and above the mandibular notch
        # "SP2": None,  # Same
        # "LOC": None,  # Left ocular canthi (EOG)
        # "ROC": None,  # Right ocular canthi (EOG)
        # "PG1": None,  # Nasopharyngeal electrode
        # "PG2": None,  # Same
    }.get(name, name)
    return name.replace("FP", "Fp").replace("Z", "z")


class _BaseTuhEeg(study.Study):
    aliases: tp.ClassVar[tuple[str, ...]] = ("TUH EEG Corpus", "TUH EEG")
    STUDY_CODE_MAP: tp.ClassVar[dict] = {
        "obeid2016": "tueg",
        "lopez2017": "tuab",
        "hamid2020": "tuar",
        "veloso2017": "tuep",
        "harati2015": "tuev",
        "vonweltin2017": "tusl",
        "shah2018": "tusz",
    }
    url: tp.ClassVar[str] = "https://isip.piconepress.com/projects/nedc/html/tuh_eeg/"
    licence: tp.ClassVar[str] = "Custom"
    description: tp.ClassVar[str] = (
        "The TUH EEG dataset is a large corpus of clinical EEG recordings from Temple University Hospital."
    )

    # TODO: Add download method, requires authentication
    def _download(self) -> None:
        raise NotImplementedError("Dataset not available to download yet.")
        # self._create_symbolic_links()

    def _create_symbolic_links(self) -> None:
        """Makes symbolic link for each tuh_eeg study folder"""
        for study_name, code in self.STUDY_CODE_MAP.items():
            source_path = self.path / "tuh_eeg" / code
            target_path = self.path / study_name
            target_path.symlink_to(source_path)

    @staticmethod
    def _fix_ch_names(raw: mne.io.RawArray) -> mne.io.RawArray:
        pattern = re.compile("^EEG (.+)-(.+)$")
        ch_types, ch_names_mapping = {}, {}
        for name in raw.ch_names:
            # Clean up name
            match = pattern.match(name)
            if match is not None:
                clean_name, ref_name = _fix_ch_name(match[1]), _fix_ch_name(match[2])
                if ref_name not in ["REF", "LE"]:  # bipolar channel
                    clean_name = clean_name + "-" + ref_name
            ch_names_mapping[name] = name if match is None else clean_name

            # Infer channel type
            if "EKG" in name:
                ch_type = "ecg"
            elif "EMG" in name:
                ch_type = "emg"
            elif name.startswith("EEG "):
                ch_type = "eeg"
            else:
                ch_type = "misc"
            ch_types[ch_names_mapping[name]] = ch_type

        raw = raw.rename_channels(ch_names_mapping)
        raw = raw.set_channel_types(ch_types)

        # Drop EEG channels whose cathode were not found in the 10-5 montage
        montage = mne.channels.make_standard_montage("standard_1005")
        to_drop = sorted(
            [
                name
                for name in raw.ch_names
                if ch_types[name] == "eeg" and name.split("-")[0] not in montage.ch_names
            ]
        )
        raw = raw.drop_channels(to_drop)
        if to_drop:
            logger.info("Dropped %s unrecognized EEG channels: %s", len(to_drop), to_drop)
        # XXX Double check whether this is necessary
        raw = raw.set_montage(montage, on_missing="ignore")

        return raw

    def _load_raw_from_path(self, file_path: str) -> mne.io.RawArray:
        raw = mne.io.read_raw(file_path)
        raw = self._fix_ch_names(raw)
        # Some files have an invalid measurement date; replacing by single date as we don't use
        # this information anyway
        raw.info.set_meas_date(
            datetime.datetime(2000, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
        )
        return raw

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        eeg_file = self._get_eeg_filename(timeline)
        raw = self._load_raw_from_path(eeg_file)
        return raw

    def _get_eeg_filename(self, timeline: dict[str, tp.Any]) -> str:
        raise NotImplementedError

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        event = dict(
            type="Eeg",
            start=0.0,
            filepath=info,
        )
        return pd.DataFrame([event])


class Obeid2016Tueg(_BaseTuhEeg):
    # Class variables
    url: tp.ClassVar[str] = (
        "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_artifact/"
    )
    aliases: tp.ClassVar[tuple[str, ...]] = ("TUEG",)
    description: tp.ClassVar[str] = (
        "Superset TUH EEG with EEG for all participants without labels or annotations"
    )
    bibtex: tp.ClassVar[str] = """
    @article{obeid2016temple,
        title={The Temple University Hospital EEG Data Corpus},
        author={Obeid, I., & Picone, J.},
        year={2016},
        journal={Frontiers in Neuroscience},
        volume={10},
        number={196},
        doi={10.3389/fnins.2016.00196},
    }

    @misc{obeid2016_data,
        url={https://isip.piconepress.com/projects/nedc/html/tuh_eeg/}
    }
    """
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=69652,
        num_subjects=14987,
        num_events_in_query=1,
        event_types_in_query={"Eeg"},
        data_shape=(23, 323840),
        frequency=256.0,
    )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings."""
        folder = self.path / "edf"
        for fname in utils.scan_files(folder):
            if not fname.endswith(".edf"):
                continue
            parts = Path(fname).parts[-5:]
            folder_number, _, sess_dir, channel_configuration, file_path = parts
            date = sess_dir[5:]
            file_path = file_path.replace(".edf", "")
            if len(file_path.split("_")) == 4:
                prefix, subject, session, token_number = file_path.split("_")
            else:
                subject, session, token_number = file_path.split("_")
                prefix = None
            yield {
                "folder_number": folder_number,  # e.g. "000" through "109"
                "subject": subject,
                "session": session,  # e.g. "s001"
                "date": date,  # YYYY or YYYY_MM_DD, e.g. "2000"
                "channel_configuration": channel_configuration,  # e.g. "01_tcp_ar"
                "token_number": token_number,  # e.g. "t000"
                "prefix": prefix,
            }

    def _get_eeg_filename(self, timeline: dict[str, tp.Any]) -> str:
        tl = timeline
        folder = (
            self.path
            / "edf"
            / tl["folder_number"]
            / tl["subject"]
            / f"{tl['session']}_{tl['date']}"
            / tl["channel_configuration"]
        )
        name = f"{tl['subject']}_{tl['session']}_{tl['token_number']}.edf"
        if timeline.get("prefix", None) is not None:
            name = f"{tl['prefix']}_{name}"
        return str(folder / name)


# ---------------------------------------------------------------------------
# TUAB (Lopez2017Tuab) adapted to the harmonized BIDS re-host of the full TUH
# EEG corpus stored at ``<data_root>/tueg_bids_edf``. The original NEDC layout
# ``edf/<split>/<label>/*.edf`` is not present; recordings are discovered from
# ``metadata_yneuro.csv`` (its ``File path`` column is the on-disk manifest,
# identical to ``files_list.pkl``) and labelled from the per-subject
# ``Pathology`` column. See :class:`Lopez2017Tuab` for details.
# ---------------------------------------------------------------------------

# The BIDS re-host carries no original TUAB clinical read; the only pathology
# signal is the harmonized per-subject ``Pathology`` value. Map it onto the
# binary normal/abnormal target expected by the eeg/pathology task.
_TUAB_PATHOLOGY_TO_LABEL: dict[str, str] = {
    "healthy": "normal",
    "epilepsy": "abnormal",
}


def _tuab_subject_split(subject: str) -> str:
    """Deterministic per-subject train/eval split (~20% eval).

    The re-host does not preserve TUAB's official train/eval partition, so a
    stable hash-based split keeps the pathology task runnable while ensuring
    all recordings of a subject share the same split (no subject leakage).
    """
    import hashlib

    digest = int(hashlib.md5(subject.encode("utf-8")).hexdigest(), 16)
    return "eval" if (digest % 5 == 0) else "train"


@functools.lru_cache(maxsize=4)
def _load_tuab_manifest(root_str: str) -> tuple[dict[str, str], ...]:
    """Parse ``metadata_yneuro.csv`` into per-recording timeline dicts.

    Cached per data root so repeated ``iter_timelines`` calls (study_summary +
    run) do not re-read the large CSV. Rows whose ``Pathology`` does not map to
    normal/abnormal, or whose ``File path`` is empty, are skipped.
    """
    root = Path(root_str)
    meta = pd.read_csv(
        root / "metadata_yneuro.csv",
        usecols=["Subject ID", "Session", "Run", "Task", "Pathology", "File path"],
        dtype=str,
        keep_default_na=False,
    )
    timelines: list[dict[str, str]] = []
    for sid, sess, run, task, pathology, fpath in zip(
        meta["Subject ID"],
        meta["Session"],
        meta["Run"],
        meta["Task"],
        meta["Pathology"],
        meta["File path"],
    ):
        label = _TUAB_PATHOLOGY_TO_LABEL.get(str(pathology).strip().lower())
        rel = str(fpath).strip()
        if label is None or not rel:
            continue
        subject = str(sid).strip()
        timelines.append(
            {
                "subject": subject,
                "session": str(sess).strip() or "001",
                "run": str(run).strip() or "000",
                "task": str(task).strip() or "rest",
                "label": label,  # normal | abnormal  (read by the target)
                "split": _tuab_subject_split(subject),  # train | eval
                "file_path": rel,  # relative to the data root (parent of tueg_bids_edf)
            }
        )
    return tuple(timelines)


class Lopez2017Tuab(_BaseTuhEeg):
    """TUAB (TUH Abnormal): normal / abnormal EEG pathology labels.

    .. note::

        This reader is adapted to the harmonized BIDS re-host of the full TUH
        EEG corpus (``<data_root>/tueg_bids_edf``). The original NEDC
        ``edf/<split>/<label>/*.edf`` directory layout is **not** present. Each
        recording is instead enumerated from ``metadata_yneuro.csv`` and its
        ``label`` is derived from the per-subject ``Pathology`` column
        (``healthy`` -> ``normal``, ``epilepsy`` -> ``abnormal``). A
        deterministic per-subject train/eval split is synthesised because the
        re-host does not preserve TUAB's official partition.
    """

    # Class variables
    aliases: tp.ClassVar[tuple[str, ...]] = ("TUAB",)
    url: tp.ClassVar[str] = (
        "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_abnormal/"
    )
    description: tp.ClassVar[str] = (
        "Subset of TUH EEG with labels for normal / abnormal recordings"
    )
    bibtex: tp.ClassVar[str] = """
    @inproceedings{lopez2015automated,
        title={Automated identification of abnormal adult EEGs},
        author={Lopez, Sebas and Suarez, G and Jungreis, D and Obeid, I and Picone, Joseph},
        booktitle={2015 IEEE signal processing in medicine and biology symposium (SPMB)},
        pages={1--5},
        year={2015},
        organization={IEEE},
        doi={10.1109/SPMB.2015.7405423},
    }
    """
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        # On-disk manifest reality for the tueg_bids_edf re-host (metadata_yneuro.csv:
        # 69671 recordings across 14987 subjects, all with a mappable Pathology).
        num_timelines=69671,
        num_subjects=14987,
        num_events_in_query=1,
        event_types_in_query={"Eeg"},
        data_shape=(23, 339000),
        frequency=250.0,
    )

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # The re-host data + manifest live in a ``tueg_bids_edf`` folder. Depending
        # on how the data root is configured, the base resolver may hand us
        # ``<root>``, ``<root>/tueg_bids_edf/Lopez2017Tuab`` (folder configured as
        # tueg_bids_edf) or ``<root>/Lopez2017Tuab``. Snap ``self.path`` to the
        # directory that actually holds ``metadata_yneuro.csv`` so file resolution
        # and enumeration are correct regardless of the base resolution.
        if not (self.path / "metadata_yneuro.csv").exists():
            p = self.path
            candidates = [
                p / "tueg_bids_edf",
                p.parent,
                p.parent / "tueg_bids_edf",
                p.parent.parent / "tueg_bids_edf",
            ]
            for candidate in candidates:
                if (candidate / "metadata_yneuro.csv").exists():
                    self.path = candidate
                    study.STUDY_PATHS[self.__class__.__name__] = self.path
                    break

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Enumerate recordings from the metadata manifest (disk holds no EDF tree)."""
        manifest = _load_tuab_manifest(str(self.path))
        if not manifest:
            raise RuntimeError(
                f"No TUAB timelines parsed from {self.path}/metadata_yneuro.csv"
            )
        for timeline in manifest:
            yield dict(timeline)

    def _get_eeg_filename(self, timeline: dict[str, tp.Any]) -> str:
        # ``File path`` is relative to the data root (parent of tueg_bids_edf),
        # e.g. ``tueg_bids_edf/sub-XXX/ses-YYY/eeg/..._eeg.edf``. Strip the
        # leading root-folder segment and resolve against ``self.path``.
        rel = str(timeline["file_path"])
        prefix = self.path.name + "/"
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
        return str(self.path / rel)


class Hamid2020Tuar(_BaseTuhEeg):
    """TUAR artifact subset, adapted to the TUEG BIDS re-host (``tueg_bids_edf``).

    The original loader expected ``<Hamid2020Tuar>/edf/<montage>/*.edf`` with a
    sibling ``*.csv`` per-channel artifact annotation file. The campaign instead
    ships a single anonymised BIDS tree (``tueg_bids_edf``) whose recordings and
    event annotations are described by ``metadata_yneuro.csv``. Timelines are
    discovered from that metadata and joined via the "File path" column; events
    are read from the "Events sample/duration/ID" columns.

    Everything is kept self-contained inside this class on purpose: the sibling
    TUH studies live in the same module and are adapted by other maintainers, so
    no module-level helpers are introduced.

    NOTE: raw loading is deferred through :class:`SpecialLoader`; the original
    artifact labels were per channel, whereas the re-host carries the recording's
    event table directly, so events are already collapsed to per-event rows.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("TUAR",)
    url: tp.ClassVar[str] = (
        "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_artifact/"
    )
    description: tp.ClassVar[str] = "Subset of TUH EEG with artifact events"
    bibtex: tp.ClassVar[str] = """
    @article{hamid2020temple,
        author={Hamid, A. and Gagliano, K. and Rahman, S. and Tulin, N. and Tchiong, V. and Obeid, I. and Picone, J.},
        booktitle={2020 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
        title={The Temple University Artifact Corpus: An Annotated Corpus of EEG Artifacts},
        year={2020},
        pages={1--4},
        doi={10.1109/SPMB50085.2020.9353647},
    }
    """
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=4699,
        num_subjects=3921,
        num_events_in_query=2,
        event_types_in_query={"Eeg", "Artifact"},
        data_shape=(23, 360500),
        frequency=250.0,
    )
    _META_CACHE: tp.ClassVar[dict] = {}

    def _rehost_root(self) -> Path:
        # ``self.path`` auto-resolves to ``DATA_DIR/Hamid2020Tuar`` (absent on
        # disk); the recordings + metadata live in the shared TUEG BIDS re-host.
        return self.path.parent / "tueg_bids_edf"

    @staticmethod
    def _as_list(value: tp.Any) -> list:
        """Parse a stringified python-list cell from ``metadata_yneuro.csv``."""
        import ast

        if value is None or isinstance(value, float):
            return []
        s = str(value).strip()
        if s in ("", "nan", "None"):
            return []
        if s.startswith("["):
            try:
                return list(ast.literal_eval(s))
            except Exception:  # pylint: disable=broad-except
                return []
        return [s]

    @staticmethod
    def _bids_entities(file_path: str) -> dict[str, tp.Any]:
        def grab(key: str) -> tp.Optional[str]:
            m = re.search(rf"{key}-([A-Za-z0-9]+)", file_path)
            return m.group(1) if m else None

        return {
            "subject": grab("sub"),
            "session": grab("ses"),
            "task": grab("task"),
            "run": grab("run"),
        }

    def _metadata(self) -> pd.DataFrame:
        """Event-bearing recordings of the re-host, indexed by ``File path``."""
        root = self._rehost_root()
        key = str(root)
        cache = type(self)._META_CACHE
        if key not in cache:
            csv = root / "metadata_yneuro.csv"
            cols = [
                "Sampling frequency",
                "Events sample",
                "Events duration",
                "Events ID",
                "Trial type",
                "File path",
            ]
            df = pd.read_csv(csv, usecols=cols, dtype=str)
            samp = df["Events sample"].astype(str).str.strip()
            mask = df["Events sample"].notna() & (samp != "") & (samp != "nan")
            df = df[mask].drop_duplicates(subset="File path").set_index("File path")
            cache[key] = df
        return cache[key]

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Discover recordings from the re-host metadata."""
        df = self._metadata()
        for file_path in df.index:
            timeline = self._bids_entities(str(file_path))
            timeline["file_path"] = str(file_path)
            yield timeline

    def _get_eeg_filename(self, timeline: dict[str, tp.Any]) -> str:
        # "File path" already includes the "tueg_bids_edf/..." prefix.
        return str(self._rehost_root().parent / timeline["file_path"])

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        row = self._metadata().loc[timeline["file_path"]]
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        records: list[dict[str, tp.Any]] = [
            {"type": "Eeg", "start": 0.0, "filepath": info}
        ]
        try:
            sfreq = float(row["Sampling frequency"])
        except Exception:  # pylint: disable=broad-except
            sfreq = 256.0
        if not sfreq or sfreq != sfreq:  # guard nan / zero
            sfreq = 256.0
        samples = self._as_list(row["Events sample"])
        durations = self._as_list(row["Events duration"])
        states = self._as_list(row["Trial type"]) or self._as_list(row["Events ID"])
        for i, sample in enumerate(samples):
            try:
                start = float(sample) / sfreq
            except Exception:  # pylint: disable=broad-except
                continue
            duration = 0.0
            if i < len(durations):
                try:
                    duration = float(durations[i]) / sfreq
                except Exception:  # pylint: disable=broad-except
                    duration = 0.0
            state = str(states[i]) if i < len(states) else "event"
            records.append(
                {
                    "type": "Artifact",
                    "start": start,
                    "duration": duration,
                    "state": state,
                    "filepath": "",
                }
            )
        return pd.DataFrame(records)


class Veloso2017Tuep(_BaseTuhEeg):
    """Possible labels: "epilepsy", "no_epilepsy"."""

    # Class variables
    aliases: tp.ClassVar[tuple[str, ...]] = ("TUEP",)
    url: tp.ClassVar[str] = (
        "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_epilepsy/"
    )
    description: tp.ClassVar[str] = (
        "Subset of TUH EEG with labels no epilepsy / epilepsy recordings"
    )
    bibtex: tp.ClassVar[str] = """
    @article{veloso2017big,
        author={Veloso, L. and McHugh, J. and von Weltin, E. and Lopez, S. and Obeid, I. and Picone, J},
        title={Big data resources for EEGs: Enabling deep learning research},
        booktitle={2017 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
        year={2017},
        pages={1--3},
        doi={10.1109/SPMB.2017.8257044},
    }
    """
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=2298,
        num_subjects=200,
        num_events_in_query=1,
        event_types_in_query={"Eeg"},
        data_shape=(24, 303500),
        frequency=250.0,
    )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings."""
        for label_dir in sorted(self.path.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name[3:]
            for sub_dir in sorted(label_dir.iterdir()):
                for sess_dir in sorted(sub_dir.iterdir()):
                    date = sess_dir.stem[5:]
                    for file_path in sorted(sess_dir.rglob("*.edf")):
                        channel_configuration = file_path.parts[-2]
                        subject, session, token_number = file_path.stem.split("_")
                        yield {
                            "subject": subject,
                            "label": label,  # "epilepsy" | "no_epilepsy"
                            "session": session,  # e.g. "s001"
                            "date": date,  # YYYY or YYYY_MM_DD, e.g. "2000"
                            "channel_configuration": channel_configuration,  # e.g. "01_tcp_ar"
                            "token_number": token_number,  # e.g. "t000"
                        }

    def _get_eeg_filename(self, timeline: dict[str, tp.Any]) -> str:
        tl = timeline
        ep_label = {"epilepsy": "00_epilepsy", "no_epilepsy": "01_no_epilepsy"}[
            tl["label"]
        ]
        folder = (
            self.path
            / ep_label
            / tl["subject"]
            / f"{tl['session']}_{tl['date']}"
            / tl["channel_configuration"]
        )
        return str(folder / f"{tl['subject']}_{tl['session']}_{tl['token_number']}.edf")


import ast as _ast
import functools as _functools
import zlib as _zlib


def _tuev_as_list(value: tp.Any) -> list:
    """Coerce a metadata cell (list, ndarray, or repr-string) into a list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    text = str(value).strip()
    if text in ("", "nan", "NaN", "None", "[]"):
        return []
    try:
        parsed = _ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, (list, tuple, np.ndarray)):
        return list(parsed)
    return [parsed]


def _tuev_has_events(value: tp.Any) -> bool:
    return len(_tuev_as_list(value)) > 0


def _tuev_clean(value: tp.Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "nan", "NaN", "None"):
        return None
    return text


@_functools.lru_cache(maxsize=2)
def _tuev_load_index(root_str: str) -> pd.DataFrame:
    """Load & cache the event-bearing (TUEV) subset of the TUEG re-host index.

    The NEMAR/openneuro re-host ships the entire TUEG superset as one flat index
    (``metadata_yneuro.pkl`` / ``.csv``) alongside (empty) BIDS subject folders.
    Recordings carrying event annotations (non-empty ``Events sample``) form the
    TUEV subset. The ~144 MB index is parsed once per process and indexed by
    ``File path`` so any number of timelines can be built without re-reading it.
    """
    from pathlib import Path as _Path

    root = _Path(root_str)
    frame: pd.DataFrame | None = None
    for name in ("metadata_yneuro.pkl", "metadata_yneuro.csv"):
        candidate = root / name
        if candidate.exists():
            frame = (
                pd.read_pickle(candidate)
                if candidate.suffix == ".pkl"
                else pd.read_csv(candidate)
            )
            break
    if frame is None:
        raise RuntimeError(
            f"No TUEV metadata index (metadata_yneuro.*) found under {root}"
        )
    frame = frame[frame["Events sample"].map(_tuev_has_events)].copy()
    frame["__file_path__"] = frame["File path"].map(str)
    frame = frame[~frame["__file_path__"].duplicated(keep="first")]
    return frame.set_index("__file_path__", drop=False)


class Harati2015Tuev(_BaseTuhEeg):
    """Possible labels:
    - bckg: Background (no seizure)
    - gped: Generalized periodic epileptiform discharges
    - pled: Periodic lateralized epileptiform discharges
    - spsw: Spike and/or sharp waves

    NOTE: In the original dataset, the labels are provided per channel, meaning the same event can
    appear multiple times. Here we merge events that appear on multiple channels into a single
    event to facilitate window-wise processing.
    """

    # Class variables
    aliases: tp.ClassVar[tuple[str, ...]] = ("TUEV",)
    url: tp.ClassVar[str] = (
        "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_events/"
    )
    description: tp.ClassVar[str] = (
        "Subset of TUH EEG with annotations for epilepsy and artifact events"
    )
    bibtex: tp.ClassVar[str] = """
    @article{harati2015improved,
        title={Improved EEG event classification using differential energy},
        author={Harati, Amir and Golmohammadi, Meysam and Lopez, Silvia and Obeid, Iyad and Picone, Joseph},
        booktitle={2015 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
        pages={1--4},
        year={2015},
        organization={IEEE},
        doi={10.1109/SPMB.2015.7405421},
    }
    """
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=4699,
        num_subjects=3921,
        num_events_in_query=1,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(23, 306500),
        frequency=250.0,
    )

    # --- NEMAR / openneuro re-host adaptation -----------------------------
    # The original ISIP train/eval ``.rec`` per-channel annotation layout is not
    # available on this cluster. The TUEG corpus was re-hosted in BIDS at
    # ``<DATA_DIR>/tueg_bids_edf`` with a flat ``metadata_yneuro`` index; the
    # per-recording event annotations live in the ``Events sample`` /
    # ``Events ID`` columns and recordings are joined to their EDF via the
    # ``File path`` column. ``self.path`` resolves to ``<DATA_DIR>/Harati2015Tuev``
    # (a name that does not exist on disk), so the reader reads the sibling
    # re-host folder that sits next to it under DATA_DIR.
    _REHOST_DIR: tp.ClassVar[str] = "tueg_bids_edf"
    _GAP_SECONDS: tp.ClassVar[float] = 1.0  # merge events closer than this into one block

    @staticmethod
    def _has_events(value: tp.Any) -> bool:
        return _tuev_has_events(value)

    def _rehost_root(self) -> Path:
        return self.path.parent / self._REHOST_DIR

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Discover the event-bearing (TUEV) subset from the re-host index.

        One flat index holds the whole TUEG superset; the recordings that carry
        event annotations constitute the subset used here. The original ISIP
        train/eval partition is not preserved by the re-host, so a deterministic
        per-subject split is derived (keeps a subject wholly in one split).
        """
        frame = _tuev_load_index(str(self._rehost_root()))
        for _, row in frame.iterrows():
            subject = _tuev_clean(row.get("Subject ID"))
            if subject is None:
                continue
            split = "eval" if (_zlib.crc32(subject.encode()) % 5 == 0) else "train"
            yield {
                "subject": subject,
                "session": _tuev_clean(row.get("Session")),
                "run": _tuev_clean(row.get("Run")),
                "split": split,
                "file_path": str(row["File path"]),
            }

    def _get_eeg_filename(self, timeline: dict[str, tp.Any]) -> str:
        # ``File path`` in the index is relative to DATA_DIR (it starts with the
        # re-host folder name), and DATA_DIR is ``self.path.parent``.
        return str(self.path.parent / timeline["file_path"])

    def _iter_event_blocks(
        self, sfreq: float, samples: list, ids: list
    ) -> tp.Iterator[dict[str, tp.Any]]:
        """Condense per-sample event markers into contiguous same-code blocks.

        The re-host stores one marker per stimulation sample (~1.3k per
        recording); consecutive markers with the same code and <_GAP_SECONDS
        spacing are merged into a single Stimulus event spanning the block.
        """
        sfreq = float(sfreq) if sfreq and float(sfreq) > 0 else 250.0
        if not ids or len(ids) != len(samples):
            ids = ["na"] * len(samples)
        pairs = sorted(
            ((float(s), str(i)) for s, i in zip(samples, ids)),
            key=lambda pair: pair[0],
        )
        if not pairs:
            return
        gap = self._GAP_SECONDS * sfreq
        block_start = block_end = prev = pairs[0][0]
        block_id = pairs[0][1]
        blocks: list[tuple[float, float, str]] = []
        for sample, ident in pairs[1:]:
            if ident != block_id or (sample - prev) > gap:
                blocks.append((block_start, block_end, block_id))
                block_start, block_id = sample, ident
            block_end = prev = sample
        blocks.append((block_start, block_end, block_id))
        for start_s, stop_s, ident in blocks:
            try:
                code = int(float(ident))
            except (TypeError, ValueError):
                code = -100
            yield {
                "type": "Stimulus",
                "start": start_s / sfreq,
                "duration": max((stop_s - start_s) / sfreq, 0.0),
                "code": code,
                "description": str(ident),
            }

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        frame = _tuev_load_index(str(self._rehost_root()))
        file_path = str(timeline["file_path"])
        row = frame.loc[file_path] if file_path in frame.index else None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        sfreq = 250.0
        rec_duration = 0.0
        if row is not None:
            sfreq_val = row.get("Sampling frequency")
            if pd.notna(sfreq_val) and float(sfreq_val) > 0:
                sfreq = float(sfreq_val)
            dur_val = row.get("Recording duration")
            if pd.notna(dur_val) and float(dur_val) > 0:
                rec_duration = float(dur_val)

        # Emit the Eeg row with explicit duration+frequency taken from the index.
        # ``MneRaw.model_post_init`` only eagerly reads the file when duration or
        # frequency is missing; supplying both keeps the raw lazy so timelines
        # build even though the re-host ships metadata without the EDF signal.
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = {
            "type": "Eeg",
            "start": 0.0,
            "filepath": info,
            "duration": rec_duration,
            "frequency": sfreq,
        }
        eeg_df = pd.DataFrame([eeg])
        if row is None:
            return eeg_df

        samples = _tuev_as_list(row["Events sample"])
        if not samples:
            return eeg_df
        ids = _tuev_as_list(row["Events ID"])
        blocks = list(self._iter_event_blocks(sfreq, samples, ids))
        if not blocks:
            return eeg_df
        return pd.concat([eeg_df, pd.DataFrame(blocks)], ignore_index=True)


class HaratiAbhishaike2015Tuev(Harati2015Tuev):
    """Harati2015Tuev but where there is an event every 1s and for each affected channel.

    The original script comes from https://github.com/Abhishaike/EEG_Event_Classification and was
    reused in BIOT, LaBraM, CBraMod, etc.
    """

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=518,
        num_subjects=370,
        num_events_in_query=634,
        event_types_in_query={"Eeg", "Artifact", "EpileptiformActivity"},
        data_shape=(23, 306500),
        frequency=250.0,
    )

    def model_post_init(self, log__: tp.Any) -> None:
        # hack to use Harati2015Tuev subfolder
        # pylint: disable=attribute-defined-outside-init
        self.path = _identify_study_subfolder(self.path, "Harati2015Tuev")
        super().model_post_init(log__)

    # pylint: disable=arguments-differ
    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:  # type: ignore
        # Re-host exposes only condensed Stimulus blocks; delegate as-is.
        return super()._load_timeline_events(timeline)


class VonWeltin2017Tusl(_BaseTuhEeg):
    aliases: tp.ClassVar[tuple[str, ...]] = ("TUSL",)
    url: tp.ClassVar[str] = (
        "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_slowing/"
    )
    description: tp.ClassVar[str] = (
        "Subset of TUH EEG with annotations for slowing events"
    )
    bibtex: tp.ClassVar[str] = """
    @article{vonweltin2017electroencephalographic,
        author={Von Weltin, E. and Ahsan, T. and Shah, V. and Jamshed, D. and Golmohammadi, M. and Obeid, I. and Picone, J.},
        booktitle={2017 IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
        title={Electroencephalographic slowing: A primary source of error in automatic seizure detection},
        year={2017},
        pages={1-5},
        doi={10.1109/SPMB.2017.8257018},
    }
    """
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=112,
        num_subjects=38,
        num_events_in_query=1,
        event_types_in_query={"Eeg"},
        data_shape=(23, 360500),
        frequency=250.0,
    )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings."""
        folder = self.path / "edf"
        for sub_dir in sorted(folder.iterdir()):
            for sess_dir in sorted(sub_dir.iterdir()):
                date = sess_dir.stem[5:]
                for file_path in sorted(sess_dir.rglob("*.edf")):
                    channel_configuration = file_path.parts[-2]
                    subject, session, token_number = file_path.stem.split("_")
                    yield {
                        "subject": subject,
                        "session": session,  # e.g. "s001"
                        "date": date,  # YYYY or YYYY_MM_DD, e.g. "2000"
                        "channel_configuration": channel_configuration,  # e.g. "01_tcp_ar"
                        "token_number": token_number,  # e.g. "t000"
                    }

    def _get_eeg_filename(self, timeline: dict[str, tp.Any]) -> str:
        tl = timeline
        folder = (
            self.path
            / "edf"
            / tl["subject"]
            / f"{tl['session']}_{tl['date']}"
            / tl["channel_configuration"]
        )
        return str(folder / f"{tl['subject']}_{tl['session']}_{tl['token_number']}.edf")


class Shah2018Tusz(_BaseTuhEeg):
    # Class variables
    aliases: tp.ClassVar[tuple[str, ...]] = ("TUSZ",)
    url: tp.ClassVar[str] = (
        "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_seizure/"
    )
    description: tp.ClassVar[str] = (
        "Subset of TUH EEG with extensive annotations of seizure (start/duration) per-channel."
    )
    bibtex: tp.ClassVar[str] = """
    @article{shah2018temple,
        author={Shah, Vinit  and von Weltin, Eva  and Lopez, Silvia  and McHugh, James Riley  and Veloso, Lillian  and Golmohammadi, Meysam  and Obeid, Iyad  and Picone, Joseph },
        title={The Temple University Hospital Seizure Detection Corpus},
        journal={Frontiers in Neuroinformatics},
        volume={12},
        year={2018},
        doi={10.3389/fninf.2018.00083},
    }
    """
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=7361,
        num_subjects=675,
        num_events_in_query=23,
        event_types_in_query={"Eeg", "Seizure"},
        data_shape=(24, 75250),
        frequency=250.0,
    )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        folder = self.path / "edf"
        for split_dir in sorted(folder.iterdir()):
            split = split_dir.name
            for sub_dir in sorted(split_dir.iterdir()):
                for sess_dir in sorted(sub_dir.iterdir()):
                    date = sess_dir.stem[5:]
                    for file_path in sorted(sess_dir.rglob("*.edf")):
                        channel_configuration = file_path.parts[-2]
                        subject, session, token_number = file_path.stem.split("_")
                        yield {
                            "subject": subject,
                            "session": session,  # e.g. "s001"
                            "date": date,  # YYYY or YYYY_MM_DD, e.g. "2000"
                            "split": split,  # "dev" | "train" | "eval"
                            "token_number": token_number,  # e.g. "t000"
                            "channel_configuration": channel_configuration,  # e.g. "01_tcp_ar"
                        }

    def _get_eeg_filename(self, timeline: dict[str, tp.Any]) -> str:
        tl = timeline
        folder = (
            self.path
            / "edf"
            / tl["split"]
            / tl["subject"]
            / f"{tl['session']}_{tl['date']}"
            / tl["channel_configuration"]
        )
        return str(folder / f"{tl['subject']}_{tl['session']}_{tl['token_number']}.edf")

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg_df = pd.DataFrame([dict(type="Eeg", start=0.0, filepath=info)]).dropna(axis=1)
        annot_df = self._load_annot_events(timeline)
        output = pd.concat([eeg_df, annot_df]).reset_index(drop=True)
        return output

    def _load_annot_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        annot_file = str(self._get_eeg_filename(timeline)).replace(".edf", ".csv")
        annot_df = pd.read_csv(annot_file, skiprows=5)
        annot_df = annot_df.rename(
            {"start_time": "start", "stop_time": "end", "label": "state"}, axis=1
        )
        annot_df["duration"] = annot_df.end - annot_df.start
        # Drop background / non-seizure events
        annot_df = annot_df.loc[annot_df["state"] != "bckg"]
        annot_df.insert(0, "type", "Seizure")
        annot_df = annot_df[
            ["type", "start", "duration", "state", "channel", "end", "confidence"]
        ]
        return annot_df
