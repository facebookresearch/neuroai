# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import mne
import pandas as pd

from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)
logger.propagate = False


class Zyma2019Electroencephalograms(study.Study):
    url: tp.ClassVar[str] = "https://physionet.org/files/eegmat/1.0.0/"
    """EEG Mental Arithmetic Tasks (EEGMAT): EEG responses during mental arithmetic.

    This study provides EEG data from participants performing serial subtraction
    tasks. Each participant has two runs: a resting-state baseline and a mental
    arithmetic task. Participant metadata includes age, gender, count quality
    group, and number of subtractions performed.

    Experimental Design:
        - EEG recordings (20-channel, 500 Hz, Neurocom)
        - 35 participants
        - 2 runs per participant
            * Run 1: resting-state baseline
            * Run 2: mental arithmetic (serial subtraction)

    Data Format:
        - 70 total timelines (35 participants x 2 runs)
        - EDF files for EEG data
        - Event types: Eeg
        - Participant metadata: age, gender, count quality, number of subtractions

    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("EEG Mental Arithmetic Tasks", "EEGMAT")

    licence: tp.ClassVar[str] = "ODC-By-1.0"
    bibtex: tp.ClassVar[str] = """
    @article{zyma2019electroencephalograms,
        url = {http://dx.doi.org/10.3390/data4010014},
        title={Electroencephalograms during Mental Arithmetic Task Performance},
        volume={4},
        issn={2306-5729},
        url={http://dx.doi.org/10.3390/data4010014},
        doi={10.3390/data4010014},
        number={1},
        journal={Data},
        publisher={MDPI AG},
        author={Zyma,  Igor and Tukaev,  Sergii and Seleznov,  Ivan and Kiyono,  Ken and Popov,  Anton and Chernykh,  Mariia and Shpenkov,  Oleksii},
        year={2019},
        month=jan,
        pages={14}
    }

    @misc{zyma2019_data,
        url={https://physionet.org/files/eegmat/1.0.0/}
    }
    """
    description: tp.ClassVar[str] = (
        "EEG recordings from 35 participants performing mental arithmetic (serial subtraction)."
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=35 * 2,
        num_subjects=35,
        num_events_in_query=1,
        event_types_in_query={"Eeg"},
        data_shape=(20, 91000),
        frequency=500.0,
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("boto3", "botocore")

    def _download(self, overwrite=False) -> None:
        physionet = download.Physionet(
            study="eegmat", version="1.0.0", dset_dir=self.path
        )
        physionet.download(overwrite=overwrite)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        task_map = {"1": "rest", "2": "mental_arithmetic"}
        for i in range(1, 36):
            for run in ["1", "2"]:
                yield dict(subject=f"Subject{i:02}", run=run, task=task_map[run])

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg_events = pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    filepath=info,
                    start=0.0,
                ),
            ]
        )
        # Add subject-level information
        subj_info = self._get_subject_info(timeline)
        eeg_events["age"] = subj_info["Age"]
        eeg_events["gender"] = {"F": "female", "M": "male"}[subj_info["Gender"]]
        eeg_events["n_subtractions"] = subj_info["Number of subtractions"]
        eeg_events["count_quality"] = subj_info["Count quality"]
        return eeg_events

    def _get_eeg_filename(self, timeline: dict[str, tp.Any]) -> Path:
        folder = self.path / "download"
        return folder / f"{timeline['subject']}_{timeline['run']}.edf"

    def _get_subject_info(self, timeline: dict[str, tp.Any]) -> pd.Series:
        file_path = self.path / "download" / "subject-info.csv"
        subj_info = pd.read_csv(file_path)
        output = subj_info[subj_info.Subject == timeline["subject"]].squeeze()
        assert isinstance(output, pd.Series)
        return output

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        # Neurocom monopolar EEG 23-channel system (Ukraine, XAI-MEDICA)
        raw = mne.io.read_raw_edf(self._get_eeg_filename(timeline))
        ch_types = {ch: ch.split(" ")[0].lower() for ch in raw.ch_names}
        ch_names = {ch: ch.split(" ")[1] for ch in raw.ch_names}
        raw.set_channel_types(ch_types)
        raw.rename_channels(ch_names)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")
        return raw
