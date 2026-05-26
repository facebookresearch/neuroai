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


class Lopez2017Tuab(_BaseTuhEeg):
    """Possible labels: "abnormal", "normal"."""

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
        num_timelines=2993,
        num_subjects=2329,
        num_events_in_query=1,
        event_types_in_query={"Eeg"},
        data_shape=(23, 339000),
        frequency=250.0,
    )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings."""
        folder = self.path / "edf"
        if not folder.exists():
            raise RuntimeError(f"Folder should exist: {folder}")

        for split_dir in sorted(folder.iterdir()):
            split = split_dir.name
            for label_dir in sorted(split_dir.iterdir()):
                label = label_dir.name
                for file_path in sorted(label_dir.rglob("*.edf")):
                    subject, session, token_number = file_path.stem.split("_")
                    yield {
                        "subject": subject,
                        "split": split,  # "train" | "eval"
                        "label": label,  # "abnormal" | "normal"
                        "session": session,  # e.g. "s001"
                        "token_number": token_number,  # e.g. "t000"
                        "channel_configuration": "01_tcp_ar",
                    }

    def _get_eeg_filename(self, timeline: dict[str, tp.Any]) -> str:
        tl = timeline
        folder = (
            self.path / "edf" / tl["split"] / tl["label"] / tl["channel_configuration"]
        )
        return str(folder / f"{tl['subject']}_{tl['session']}_{tl['token_number']}.edf")


class Hamid2020Tuar(_BaseTuhEeg):
    """NOTE: In the original dataset, the labels are provided per channel, meaning the same event
    can appear multiple times. Here we merge events that appear on multiple channels into a single
    event to facilitate window-wise processing. Moreover, we separate combined events (e.g.,
    'eyem_musc') into multiple events, one for each event type (e.g., 'eyem' and 'musc').
    """

    # Class variables
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
        num_timelines=310,
        num_subjects=213,
        num_events_in_query=4,
        event_types_in_query={"Eeg", "Artifact"},
        data_shape=(23, 360500),
        frequency=250.0,
    )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        folder = self.path / "edf"
        for channel_folder in sorted(folder.iterdir()):
            channel_configuration = channel_folder.stem
            for file_path in sorted(channel_folder.rglob("*.edf")):
                subject, session, token_number = file_path.stem.split("_")
                yield {
                    "channel_configuration": channel_configuration,  # e.g. "01_tcp_ar"
                    "subject": subject,
                    "session": session,  # e.g. "s001"
                    "token_number": token_number,  # e.g. "t000"
                }

    def _get_eeg_filename(self, timeline: dict[str, tp.Any]) -> str:
        tl = timeline
        file_dir = self.path / "edf" / timeline["channel_configuration"]
        name = f"{tl['subject']}_{tl['session']}_{tl['token_number']}.edf"
        return str(file_dir / name)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        tl = timeline
        file_dir = self.path / "edf" / timeline["channel_configuration"]
        basename = f"{tl['subject']}_{tl['session']}_{tl['token_number']}"
        events_file = str(file_dir / f"{basename}.csv")
        seiz_events_file = str(file_dir / f"{basename}_seiz.csv")
        events_df = self._format_artifact_events(events_file, seiz_events_file)

        # Clip events that exceed the recording duration
        raw = mne.io.read_raw(self._get_eeg_filename(timeline))
        events_stop = (events_df["start"] + events_df["duration"]).clip(
            upper=raw.duration
        )
        events_df["duration"] = events_stop - events_df["start"]

        # Merge multi-channel events into one
        groups = events_df.groupby(
            ["type", "start", "duration", "filepath", "state"], sort=False
        )
        events_df = groups.channel.apply(lambda x: ",".join(sorted(x))).reset_index()
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = {"type": "Eeg", "start": 0.0, "filepath": info}
        return pd.concat([pd.DataFrame([eeg]), events_df], ignore_index=True)

    def _load_events_csv(self, file_path: str, header_row: int = 6) -> pd.DataFrame:
        events_df = pd.read_csv(file_path, header=header_row)
        return events_df

    def _format_artifact_events(self, file_path: str, seiz_path: str) -> pd.DataFrame:
        out_cols = [
            "type",
            "start",
            "duration",
            "filepath",
            "channel",
            "state",
            "had_epileptic_event",
        ]
        events_df = self._load_events_csv(file_path)
        events_df["type"] = "Artifact"
        if Path(seiz_path).exists():
            seiz_df = self._load_events_csv(seiz_path)
            seiz_df["type"] = "Seizure"
            events_df = pd.concat([events_df, seiz_df])
            had_epileptic_event = True
        else:
            had_epileptic_event = False
        output = events_df[["channel", "start_time", "label", "type"]]
        output = output.rename(columns={"start_time": "start", "label": "state"})
        output["filepath"] = ""  # no stimulus
        output["had_epileptic_event"] = had_epileptic_event
        # Compute duration
        output["duration"] = events_df["stop_time"] - events_df["start_time"]
        # Separate combo events (e.g., 'eyem_musc') into multiple rows
        output["state"] = output.state.str.split("_")
        output = output.explode(column="state")
        output["state"] = output["state"].replace({"elec": "elpp"})
        return output[out_cols]


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
        num_timelines=518,
        num_subjects=370,
        num_events_in_query=20,
        event_types_in_query={"Eeg", "Artifact", "EpileptiformActivity"},
        data_shape=(23, 306500),
        frequency=250.0,
    )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        folder = self.path / "edf"
        for split_dir in sorted(folder.iterdir()):
            split = split_dir.name
            for sub_dir in sorted(split_dir.iterdir()):
                for file_path in sorted(sub_dir.rglob("*.edf")):
                    if split == "train":
                        subject, token_number = file_path.stem.split("_")
                        label = None
                        session = None
                    elif split == "eval":
                        label, subject, _, _ = file_path.stem.split("_")
                        session = file_path.stem[9:]
                        token_number = None
                    yield {
                        "subject": subject,
                        "split": split,  # "train" | "eval"
                        "token_number": token_number,  # e.g. "t000", only for train split
                        "label": label,  # "bckg" | "gped" | "pled" | "spsw" | None
                        "session": session,  # e.g. "s001", only for eval split
                    }

    def _get_eeg_filename(self, timeline: dict[str, tp.Any]) -> str:
        tl = timeline
        folder = self.path / "edf" / tl["split"] / tl["subject"]
        if tl["split"] == "eval":
            return str(folder / f"{tl['label']}_{tl['subject']}_{tl['session']}.edf")
        return str(folder / f"{tl['subject']}_{tl['token_number']}.edf")

    def _get_rec_filename(self, timeline: dict[str, tp.Any]) -> str:
        tl = timeline
        folder = self.path / "edf" / tl["split"] / tl["subject"]
        if tl["split"] == "eval":
            return str(folder / f"{tl['label']}_{tl['subject']}_{tl['session']}.rec")
        return str(folder / f"{tl['subject']}_{tl['token_number']}.rec")

    def _load_timeline_events(
        self,
        timeline: dict[str, tp.Any],
        condense_events: bool = True,
        merge_channels: bool = True,
    ) -> pd.DataFrame:
        eeg_df = super()._load_timeline_events(timeline)
        annots_df = self._load_annot_events(timeline)
        if condense_events:
            annots_df = self._condense_events(annots_df)
        if merge_channels:
            groups = annots_df.groupby(["type", "start", "duration", "state"], sort=False)
            annots_df = groups.channel.apply(
                lambda x: "{" + ",".join([str(int(x)) for x in sorted(x)]) + "}"
            ).reset_index()
        return pd.concat([eeg_df, annots_df], ignore_index=True)

    def _load_annot_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        rec_file = self._get_rec_filename(timeline)
        cols = ["channel", "start", "stop", "label_code"]
        event_df = pd.DataFrame(np.genfromtxt(rec_file, delimiter=","), columns=cols)
        label_dict = {1: "spsw", 2: "gped", 3: "pled", 4: "eyem", 5: "artf", 6: "bckg"}
        event_df["state"] = event_df.label_code.map(label_dict)
        event_df["start"] = event_df.start.round(1)
        event_df["stop"] = event_df.stop.round(1)
        event_df["duration"] = (event_df.stop - event_df.start).round(1)
        type_map = {
            "spsw": "EpileptiformActivity",  # Spike and/or sharp waves
            "gped": "EpileptiformActivity",  # Generalized periodic epileptiform discharges
            "pled": "EpileptiformActivity",  # Periodic lateralized epileptiform discharges
            "eyem": "Artifact",
            "artf": "Artifact",
            "bckg": "EpileptiformActivity",  # All other non-seizure cerebral signals
        }
        event_df["type"] = event_df.state.map(type_map)
        return event_df

    def _condense_events(self, event_df: pd.DataFrame) -> pd.DataFrame:
        """Annotations are labeled for every sec; group consecutive events and set duration."""
        output = pd.DataFrame(columns=["type", "start", "duration", "channel", "state"])
        groups = event_df.groupby(by=["type", "channel", "state"])
        event_times = pd.concat(
            [groups["start"].unique(), groups["stop"].unique()], axis=1
        )

        channel: float
        state: str
        for (event_type, channel, state), data in event_times.iterrows():  # type: ignore
            start_times = set(data.start) - set(data.stop)
            stop_times = set(data.stop) - set(data.start)

            # Merge possible overlapping events (over the next or previous 1s)
            for t in sorted(data.start):
                start_times -= set(np.linspace(t + 0.1, t + 1, 10).round(1))
            for t in sorted(data.stop, reverse=True):
                stop_times -= set(np.linspace(t - 1, t - 0.1, 10).round(1))

            for start_t, stop_t in zip(sorted(start_times), sorted(stop_times)):
                duration = stop_t - start_t
                if duration <= 0:
                    msg = f"Duration should be strictly positive, got {duration}"
                    raise RuntimeError(msg)
                output.loc[len(output)] = [  # type: ignore
                    event_type,  # type: ignore
                    start_t,
                    duration,
                    channel,
                    state,
                ]
        return output


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
        return super()._load_timeline_events(
            timeline, condense_events=False, merge_channels=False
        )


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
