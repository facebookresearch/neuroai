# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from neuralfetch import download
from neuralset.events import study
from neuralset.events.etypes import CategoricalEvent

logger = logging.getLogger(__name__)
logger.propagate = False


class SleepArousal(CategoricalEvent):
    state: tp.Literal[
        # Starts with "resp_" in [1]
        "hypopnea",
        "central_apnea",
        "mixed_apnea",
        "obstructive_apnea",
        "cheynestokesbreath",  # Cheyne-Stokes abnormal breathing pattern (faster, then slower)
        "hypoventilation",
        "partialobstructive",  # Partial airway obstruction
        # Starts with "arousal_" in [1]
        "rera",  # Respiratory Effort Related Arousal
        "spontaneous",  # Not directly triggered by obvious breathing event
        "plm",  # Periodic leg movement
        "bruxism",  # Grinding/clenching of teeth
        "snore",
        "noise",  # Noise caused by vocalizations, coughing, movement, external noise, etc.
    ]


# https://github.com/hubertjb/dynamic-spatial-filtering/blob/master/datasets.py
def convert_wfdb_anns_to_mne_annotations(annots) -> mne.Annotations:
    """Convert wfdb.io.Annotation format to MNE's.

    Parameters
    ----------
    annots : wfdb.io.Annotation
        Annotation object obtained by e.g. loading an annotation file with
        wfdb.rdann().

    Returns
    -------
    mne.Annotations :
        MNE Annotations object.
    """
    ann_chs = set(annots.chan)
    onsets = annots.sample / annots.fs
    new_onset: list[float] = []
    new_duration: list[float] = []
    new_description: list[str] = []
    for ch in ann_chs:
        mask = annots.chan == ch
        ch_onsets = onsets[mask]
        ch_descs = np.array(annots.aux_note)[mask]

        # Events with beginning and end, defined by '(event' and 'event)'
        if all((i.startswith("(") or i.endswith(")")) for i in ch_descs):
            for desc, onset in zip(ch_descs, ch_onsets):
                if desc.startswith("("):
                    event_name = desc[1:]
                    event_start = onset
                elif desc.endswith(")"):
                    event_end = onset
                    new_onset.append(event_start)
                    new_duration.append(event_end - event_start)
                    new_description.append(event_name)
                else:
                    event_name, event_start, event_end = "", -1.0, -1.0
        else:  # Sleep stage-like annotations
            ch_durations = np.concatenate([np.diff(ch_onsets), [30]])
            assert all(ch_durations > 0), "Negative duration"
            new_onset.extend(ch_onsets)
            new_duration.extend(ch_durations)
            new_description.extend(ch_descs)

    mne_annots = mne.Annotations(new_onset, new_duration, new_description, orig_time=None)

    return mne_annots


class Ghassemi2018You(study.Study):
    """PhysioNet Challenge 2018: EEG polysomnography recordings for sleep arousal detection.

    This study provides polysomnographic EEG recordings from participants monitored
    at the Massachusetts General Hospital (MGH) sleep laboratory for the diagnosis
    of sleep disorders. Data includes sleep stage and arousal annotations for the
    training set. Participants are clinical patients evaluated for sleep apnea.

    Experimental Design:
        - EEG recordings (13-channel, 200 Hz)
        - 1,983 participants
        - Paradigm: clinical polysomnographic sleep monitoring
            * Training set (N=994) with arousal annotations
            * Test set (N=989) without labels

    Data Format:
        - 1,983 total timelines (1 per participant)
        - WFDB (.mat/.hea) format
        - Event types: Eeg, SleepStage, SleepArousal
        - Channels include EEG, EOG, EMG, ECG, and respiratory signals
        - EEG referenced at M1 and M2 (left and right mastoid processes)
        - Participant metadata: age, sex
        - Sleep arousal-related events following nomenclature used in [1]_. Parameters ---------- state : Hypopnea/apnea state, or arousal type. References ---------- .. [1] Ghassemi, Mohammad M., et al. "You snooze, you win: the physionet/computing in cardiology challenge 2018." 2018 Computing in Cardiology Conference (CinC). Vol. 45. IEEE, 2018.
    """

    url: tp.ClassVar[str] = "https://physionet.org/files/challenge-2018/1.0.0/"
    aliases: tp.ClassVar[tuple[str, ...]] = (
        "PhysioNet/CinC Challenge 2018",
        "CinC2018",
    )

    licence: tp.ClassVar[str] = "ODC-By-1.0"
    bibtex: tp.ClassVar[str] = """
    @article{ghassemi2018you,
        title={You {{Snooze}}, {{You Win}}: The {{PhysioNet}}/{{Computing}} in {{Cardiology Challenge}} 2018},
        shorttitle={You {{Snooze}}, {{You Win}}},
        author={Ghassemi, Mohammad M and Moody, Benjamin E and Lehman, Li-wei H and Song, Christopher and Li, Qiao and Sun, Haoqi and Mark, Roger G and Westover, M Brandon and Clifford, Gari D},
        year=2018,
        month=sep,
        journal={Computing in cardiology},
        volume={45},
        pages={10.22489/cinc.2018.049},
        issn={2325-8861},
        doi={10.22489/cinc.2018.049},
        pmcid={PMC8596964},
        pmid={34796237}
    }

    @misc{ghassemi2018youa,
        title={You {{Snooze You Win}}: {{The PhysioNet}}/{{Computing}} in {{Cardiology Challenge}} 2018},
        shorttitle={You {{Snooze You Win}}},
        author={Ghassemi, Mohammad and Moody, Benjamin and Lehman, Li-wei and Mark, Roger and Clifford, Gari D.},
        year=2018,
        month=feb,
        publisher={PhysioNet},
        doi={10.13026/1Q9B-GE17},
        url={https://physionet.org/files/challenge-2018/1.0.0/}
    }
    """
    description: tp.ClassVar[str] = (
        "A corpus of 1,985 EEG recordings from a sleep laboratory annotated for arousal and sleep stage."
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1983,
        num_subjects=1983,
        num_events_in_query=171,
        event_types_in_query={"Eeg", "SleepStage", "SleepArousal"},
        data_shape=(6, 5147000),
        frequency=200.0,
        query='subject == "Ghassemi2018You/tr03-0005"',  # Skip test subjs
    )

    def _download(self) -> None:
        physionet = download.Physionet(
            study="challenge-2018",
            dset_dir=self.path,
            version="1.0.0",
        )
        physionet.download()

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        folder = self.path / "download"
        for split in ["training", "test"]:
            split_dir = folder / split
            for file_path in sorted(split_dir.rglob("*.mat")):
                # Skip arousal files in training set
                if "arousal.mat" in str(file_path):
                    continue
                subject = file_path.stem
                yield dict(
                    subject=subject,
                    split=split.replace("ing", ""),
                    split_dir=split,
                )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        subj_info = self._get_subject_info(timeline)
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = dict(
            type="Eeg",
            start=0.0,
            filepath=info,
            age=subj_info["Age"],
            gender=subj_info["Sex"],
        )

        if timeline["split"] == "test":  # No sleep labels for test set
            output = pd.DataFrame([eeg])
        else:
            sleep_df = self._load_sleep_events(timeline)
            output = pd.concat([pd.DataFrame([eeg]), sleep_df]).reset_index(drop=True)
        return output

    def _get_filepath(self, filetype: str, timeline: dict[str, tp.Any]) -> Path:
        folder = self.path / "download"
        file_dir = folder / timeline["split_dir"] / timeline["subject"]
        match filetype:
            case "eeg":
                file_path = f"{timeline['subject']}.mat"
            case "header":
                file_path = f"{timeline['subject']}.hea"
            case "arousal":
                file_path = f"{timeline['subject']}.arousal"
            case "subject_info":
                file_dir = folder
                file_path = "age-sex.csv"
        return file_dir / file_path

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        """
        Load raw PhysioNet PC18 WFDB data and return an MNE Raw object.
        """
        import wfdb  # type: ignore[import-untyped]

        # Load WFDB record
        mat_file = self._get_filepath("eeg", timeline)
        record = wfdb.rdrecord(Path(mat_file).with_suffix(""))

        # Extract signal data
        data = record.p_signal.T

        # Convert units to volts for EEG/EMG/ECG channels
        units = np.array(record.units)
        data[units == "uV"] /= 1e6
        data[units == "mV"] /= 1e3

        ch_types = [
            "eeg",
            "eeg",
            "eeg",
            "eeg",
            "eeg",
            "eeg",
            "eog",  # E1-M2
            "emg",  # Chin1-Chin2
            "misc",  # ABD  # codespell:ignore abd
            "misc",  # CHEST
            "misc",  # AIRFLOW
            "misc",  # SaO2
            "ecg",  # ECG
        ]

        # Create MNE info object
        info = mne.create_info(
            ch_names=record.sig_name, sfreq=record.fs, ch_types=ch_types
        )

        raw = mne.io.RawArray(data, info)
        return raw

    def _load_sleep_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Load sleep annotations from WFDB arousal data"""
        import wfdb  # type: ignore[import-untyped]

        arousal_file = str(self._get_filepath("arousal", timeline).with_suffix(""))
        annots = wfdb.rdann(
            arousal_file,
            "arousal",
            sampfrom=0,
            sampto=None,
            shift_samps=False,
            return_label_elements=["symbol"],
            summarize_labels=False,
        )
        mne_annots = convert_wfdb_anns_to_mne_annotations(annots)
        annots_df = mne_annots.to_data_frame()
        start = annots_df.loc[0, "onset"]  # Assumes annots start at time 0
        annots_df["start"] = (annots_df["onset"] - start).dt.total_seconds()

        # Handle sleep stage events
        stage_mask = annots_df.description.isin(["W", "N1", "N2", "N3", "R"])
        annots_df.loc[stage_mask, "stage"] = annots_df[stage_mask].description
        annots_df.loc[stage_mask, "type"] = "SleepStage"

        # Handle hypopnea/apnea events
        apnea_mask = annots_df.description.str.startswith("resp_")
        annots_df.loc[apnea_mask, "state"] = annots_df.loc[
            apnea_mask, "description"
        ].apply(lambda x: x.split("resp_")[1].replace("apnea", "_apnea"))
        annots_df.loc[apnea_mask, "type"] = "SleepArousal"

        # Handle arousal events
        arousal_mask = annots_df.description.str.startswith("arousal_")
        annots_df.loc[arousal_mask, "state"] = annots_df.loc[
            arousal_mask, "description"
        ].str.replace("arousal_", "")
        annots_df.loc[arousal_mask, "type"] = "SleepArousal"

        annots_df = annots_df[["type", "start", "duration", "stage", "state"]]
        return annots_df

    def _get_subject_info(self, timeline: dict[str, tp.Any]) -> dict:
        info_df = pd.read_csv(self._get_filepath("subject_info", timeline))
        info_df["Sex"] = info_df["Sex"].str.lower()
        # Select subject-specific information
        index = info_df[info_df.Record == timeline["subject"]].index
        return info_df.loc[index].to_dict("records")[0]
