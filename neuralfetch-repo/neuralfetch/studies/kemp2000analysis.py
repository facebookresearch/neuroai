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
from mne.datasets import sleep_physionet

from neuralset.events import study

logger = logging.getLogger(__name__)
logger.propagate = False


class Kemp2000Analysis(study.Study):
    """SleepEDF: EEG responses during sleep.

    This dataset contains ambulatory polysomnographic recordings over approximately
    two consecutive nights. The study investigates the effects of aging on sleep
    by analyzing slow-wave EEG microcontinuity across participants of varying ages.

    Experimental Design:
        - EEG recordings (2-channel, 100 Hz)
        - 78 participants
        - Up to 2 recording sessions per participant
        - Paradigm: whole-night ambulatory sleep monitoring
            * Sleep stages annotated via hypnogram (W, N1, N2, N3, R)

    Notes:
        - The description attribute reports 82 participants; the script includes 78
          (5 participant indices excluded from range 0-82).
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("Sleep-EDF Cassette Dataset", "SleepEDF")

    licence: tp.ClassVar[str] = "ODC-By-1.0"
    bibtex: tp.ClassVar[str] = """
    @article{kemp2000analysis,
        author={Kemp, B. and Zwinderman, A.H. and Tuk, B. and Kamphuisen, H.A.C. and Oberye, J.J.L.},
        journal={IEEE Transactions on Biomedical Engineering},
        title={Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG},
        year={2000},
        volume={47},
        number={9},
        pages={1185-1194},
        doi={10.1109/10.867928},
        url={https://ieeexplore.ieee.org/abstract/document/867928/}
    }

    @misc{kemp2000_data,
        url={https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/}
    }
    """
    description: tp.ClassVar[str] = (
        "Ambulatory EEG recordings over 48-hr period for 82 participants to study effects of aging on sleep."
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=153,
        num_subjects=78,
        num_events_in_query=152,
        event_types_in_query={"Artifact", "Eeg", "SleepStage"},
        data_shape=(2, 8490000),
        frequency=100.0,
        query='subject == "Kemp2000Analysis/00" and session == 2',
    )

    SUBJECTS: tp.ClassVar[list[int]] = [
        x for x in range(83) if x not in [39, 68, 69, 78, 79]
    ]
    url: tp.ClassVar[str] = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
    SUBJECTS_REC_1: tp.ClassVar[list[int]] = [
        x for x in SUBJECTS if x not in [36, 39, 52, 68, 69, 78, 79]
    ]
    SUBJECTS_REC_2: tp.ClassVar[list[int]] = [
        x for x in SUBJECTS if x not in [13, 39, 68, 69, 78, 79]
    ]

    def _download(self) -> None:
        """
        Leverages mne.datasets.fetch_dataset method to download.
        - The reported download link: https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/

        Option to download through S3:
        aws s3 sync --no-sign-request s3://physionet-open/sleep-edfx/1.0.0/ DESTINATION

        Option to download with wget:
        folder = self.path / "physionet-sleep-data"
        download_url = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
        subprocess.run((f"wget -r -N -c -np -P {folder} {download_url}"), shell=True)
        """
        sleep_physionet.age.fetch_data(
            subjects=self.SUBJECTS_REC_1, recording=[1], path=self.path
        )
        sleep_physionet.age.fetch_data(
            subjects=self.SUBJECTS_REC_2, recording=[2], path=self.path
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        folder = self.path / "physionet-sleep-data"
        for subject in self.SUBJECTS:
            subject_str = f"{subject:02}"
            for session in [1, 2]:
                eeg_file = list(folder.glob(f"SC4{subject_str}{session}*-PSG.edf"))
                if len(eeg_file) == 1:
                    suffix = eeg_file[0].stem[6:8]
                elif len(eeg_file) >= 1:
                    raise ValueError(
                        f"Must have 1 file per subject-session; got {len(eeg_file)}: {eeg_file}"
                    )
                else:
                    continue
                hypno_file = list(
                    folder.glob(f"SC4{subject_str}{session}*-Hypnogram.edf")
                )
                if len(hypno_file) != 1:
                    raise ValueError(
                        f"Must have 1 file per subject-session; got {len(hypno_file)}: {hypno_file}"
                    )
                hypno_suffix = hypno_file[0].stem[6:8]
                yield dict(
                    subject=subject_str,
                    session=session,
                    suffix=suffix,
                    hypno_suffix=hypno_suffix,
                )

    def _get_filenames(self, timeline: dict[str, tp.Any]) -> tuple[Path, Path | None]:
        study_path = self.path / "physionet-sleep-data"
        eeg_file = (
            study_path
            / f"SC4{timeline['subject']}{timeline['session']}{timeline['suffix']}-PSG.edf"
        )
        if timeline["hypno_suffix"] is not None:
            hypno_file = (
                study_path
                / f"SC4{timeline['subject']}{timeline['session']}{timeline['hypno_suffix']}-Hypnogram.edf"
            )
        else:
            hypno_file = None
        return eeg_file, hypno_file

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        events = pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    start=0.0,
                    filepath=info,
                )
            ]
        )
        if timeline["hypno_suffix"] is not None:
            hypno_df = self._load_hypnogram_annots(timeline)
            events = pd.concat([events, hypno_df], axis=0)

        return events

    def _load_hypnogram_annots(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        hypnogram_file = self._get_filenames(timeline)[1]
        annots_df = mne.read_annotations(hypnogram_file).to_data_frame(time_format=None)
        annots_df = annots_df.rename(columns={"onset": "start"})
        annots_df["stage"] = annots_df["description"].replace(
            {
                "Sleep stage W": "W",
                "Sleep stage 1": "N1",
                "Sleep stage 2": "N2",
                "Sleep stage 3": "N3",
                "Sleep stage 4": "N3",
                "Sleep stage R": "R",
                "Sleep stage ?": pd.NA,
                "Movement time": pd.NA,
            }
        )
        annots_df["state"] = annots_df["description"].replace(
            {
                "Sleep stage W": pd.NA,
                "Sleep stage 1": pd.NA,
                "Sleep stage 2": pd.NA,
                "Sleep stage 3": pd.NA,
                "Sleep stage 4": pd.NA,
                "Sleep stage R": pd.NA,
                "Sleep stage ?": pd.NA,
                "Movement time": "musc",
            }
        )
        annots_df = annots_df.dropna(subset=["stage", "state"], how="all", axis=0)
        annots_df.insert(0, "type", "")
        annots_df.loc[annots_df.stage.isin(["W", "N1", "N2", "N3", "R"]), "type"] = (
            "SleepStage"
        )
        annots_df.loc[annots_df.state == "musc", "type"] = "Artifact"
        return annots_df[["type", "start", "duration", "stage", "state"]]

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.Raw:
        # Modified Oxford 4-channel cassette recorder
        # Frequency response range (3 dB points) from 0.5 to 100 Hz
        raw = mne.io.read_raw(self._get_filenames(timeline)[0])
        montage = mne.channels.make_standard_montage("standard_1020")
        ch_types = {}
        ch_names_mapping = {}
        for raw_name in raw.ch_names:
            if raw_name == "Event marker":
                continue
            ch_type, ch_name = raw_name.split(" ")
            ch_names_mapping[raw_name] = ch_name
            ch_types[ch_name] = ch_type.lower().replace("temp", "temperature")
        raw = raw.drop_channels(["Event marker"])
        raw = raw.rename_channels(ch_names_mapping)
        raw = raw.set_channel_types(ch_types)
        raw = raw.set_montage(montage, on_missing="ignore")
        return raw
