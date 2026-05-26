# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import pandas as pd
from mne_bids import BIDSPath

from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)


class Miltiadous2023Dice(study.Study):
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds004504/"
    """Miltiadous2023Dice: EEG resting-state recordings from Alzheimer's disease, frontotemporal dementia, and healthy participants.

    This dataset contains resting-state eyes-closed EEG recordings from 88
    participants: 36 with Alzheimer's disease, 23 with frontotemporal dementia, and
    29 age-matched healthy controls. The data were collected from routine clinical
    EEG and are provided in BIDS-compliant format.

    Experimental Design:
        - EEG recordings (19-channel, 500 Hz)
        - 88 participants (36 Alzheimer's, 23 frontotemporal dementia, 29 healthy controls)
        - Single resting-state recording per participant (eyes closed)
        - Paradigm: resting-state, eyes closed

    """

    bibtex: tp.ClassVar[str] = """
    @article{miltiadous2023dice,
        title={{{DICE-Net}}: {{A Novel Convolution-Transformer Architecture}} for {{Alzheimer Detection}} in {{EEG Signals}}},
        shorttitle={{{DICE-Net}}},
        author={Miltiadous, Andreas and Gionanidis, Emmanouil and Tzimourta, Katerina D. and Giannakeas, Nikolaos and Tzallas, Alexandros T.},
        year=2023,
        journal={IEEE Access},
        volume={11},
        pages={71840--71858},
        issn={2169-3536},
        doi={10.1109/ACCESS.2023.3294618},
    }

    @article{miltiadous2023dataset,
        title={A {{Dataset}} of {{Scalp EEG Recordings}} of {{Alzheimer}}'s {{Disease}}, {{Frontotemporal Dementia}} and {{Healthy Subjects}} from {{Routine EEG}}},
        author={Miltiadous, Andreas and Tzimourta, Katerina D. and Afrantou, Theodora and Ioannidis, Panagiotis and Grigoriadis, Nikolaos and Tsalikakis, Dimitrios G. and Angelidis, Pantelis and Tsipouras, Markos G. and Glavas, Euripidis and Giannakeas, Nikolaos and Tzallas, Alexandros T.},
        year=2023,
        month=may,
        journal={Data},
        volume={8},
        number={6},
        publisher={publisher},
        issn={2306-5729},
        doi={10.3390/data8060095},
        copyright={http://creativecommons.org/licenses/by/3.0/},
        langid={english}
    }

    @misc{miltiadous2024dataset,
        title={A Dataset of {{EEG}} Recordings from: {{Alzheimer}}'s Disease, {{Frontotemporal}} Dementia and {{Healthy}} Subjects},
        shorttitle={A Dataset of {{EEG}} Recordings From},
        author={Miltiadous, Andreas and Tzimourta, Katerina D. and Afrantou, Theodora and Ioannidis, Panagiotis and Grigoriadis, Nikolaos and Tsalikakis, Dimitrios G. and Angelidis, Pantelis and Tsipouras, Markos G. and Glavas, Evripidis and Giannakeas, Nikolaos and Tzallas, Alexandros T.},
        year=2024,
        publisher={Openneuro},
        doi={10.18112/OPENNEURO.DS004504.V1.0.8},
        url={https://openneuro.org/datasets/ds004504/}
    }
    """
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "EEG resting state-closed eyes recordings from 88 participants in total, with Alzheimer's "
        "disease, dementia and healthy controls."
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=88,
        num_subjects=88,
        num_events_in_query=1,
        event_types_in_query={"Eeg"},
        data_shape=(19, 299900),
        frequency=500,
    )

    def _download(self, overwrite=False) -> None:
        download.Openneuro(study="ds004504", dset_dir=self.path).download()

    def iter_timelines(self):
        """Returns a generator of all recordings"""
        task = "eyesclosed"
        for subject in range(1, 89):
            sub_id = str(subject).zfill(3)
            dir_path = self.path / "download" / f"sub-{sub_id}" / "eeg"
            fpath = dir_path / f"sub-{sub_id}_task-{task}_eeg.set"

            if fpath.exists():
                yield dict(subject=sub_id, task=task)

    def _get_participant_info(self, subject) -> pd.DataFrame:
        """Returns participant info as a dataframe"""
        fpath_group_info = self.path / "download" / "participants.tsv"
        df = pd.read_csv(fpath_group_info, sep="\t")
        info = df.query(f"participant_id == 'sub-{subject}'").copy()
        info.reset_index(inplace=True, drop=True)

        info.rename(
            columns={
                "Age": "age",
                "Gender": "gender",
                "Group": "diagnosis",
                "MMSE": "mmse_score",
            },
            inplace=True,
        )
        dict_gender = {"M": "male", "F": "female"}
        dict_diagnosis = {"A": "alzheimers", "C": "healthy", "F": "dementia"}
        info.replace({"gender": dict_gender, "diagnosis": dict_diagnosis}, inplace=True)
        info.drop(columns="participant_id", inplace=True)

        return info

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        subject_info = self._get_participant_info(timeline["subject"])
        bids_path = BIDSPath(
            subject=timeline["subject"],
            task=timeline["task"],
            root=self.path / "download",
            datatype="eeg",
            suffix="eeg",
            extension=".set",
        )
        eeg = pd.DataFrame([dict(type="Eeg", filepath=str(bids_path.fpath), start=0)])

        return pd.concat([eeg, subject_info], axis=1)


# # # # # mini dataset # # # # #


class Miltiadous2023DiceSample(Miltiadous2023Dice):
    """Sample version of Miltiadous2023Dice EEG Alzheimer's dataset.

    Downloads only 3 subjects: one from each diagnosis group (Alzheimer's,
    dementia, healthy control) for quick testing and validation.

    - Subject 001: Alzheimer's disease (Group A)
    - Subject 037: Frontotemporal dementia (Group F)
    - Subject 067: Healthy control (Group C)

    OpenNeuro dataset: https://openneuro.org/datasets/ds004504/
    Data: 19-channel EEG, 500 Hz, resting-state eyes-closed
    """

    description: tp.ClassVar[str] = (
        "Sample Miltiadous2023Dice: 3 subjects (one per diagnosis group) for rapid testing. "
        "EEG resting-state eyes-closed recordings."
    )

    _SAMPLE_SUBJECTS: tp.ClassVar[tuple[str, ...]] = ("001", "037", "067")

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=3,
        num_subjects=3,
        num_events_in_query=1,
        event_types_in_query={"Eeg"},
        data_shape=(19, 299900),
        frequency=500,
    )

    def _download(self, overwrite=False) -> None:
        """Download only 3 subjects from the dataset."""
        include_patterns = [
            "dataset_description.json",
            "participants.tsv",
            "participants.json",
        ] + [f"sub-{sub_id}/**" for sub_id in self._SAMPLE_SUBJECTS]

        download.Openneuro(
            study="ds004504",
            dset_dir=self.path,
            include=include_patterns,
        ).download(overwrite=overwrite)

    def iter_timelines(self):
        """Yield only the 3 sample subjects."""
        task = "eyesclosed"
        for sub_id in self._SAMPLE_SUBJECTS:
            dir_path = self.path / "download" / f"sub-{sub_id}" / "eeg"
            fpath = dir_path / f"sub-{sub_id}_task-{task}_eeg.set"

            if fpath.exists():
                yield dict(subject=sub_id, task=task)
