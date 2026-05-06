# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from itertools import chain, product
from pathlib import Path

import mne
import pandas as pd

from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)


class Mumtaz2018Machine(study.Study):
    url: tp.ClassVar[str] = "https://figshare.com/articles/dataset/EEG_Data_New/4244171"
    """EEG responses from patients with major depressive disorder and healthy controls.

    This dataset contains resting-state and task EEG recordings from 34 participants
    with major depressive disorder (MDD) and 30 healthy controls. Participants
    completed three conditions: eyes closed, eyes open, and a P300 auditory oddball
    task.

    Experimental Design:
        - EEG recordings (19-channel, 256 Hz)
        - 64 participants (34 MDD, 30 healthy controls)
        - 3 conditions per participant: eyes closed (EC), eyes open (EO), P300 task (TASK)
        - Paradigm: resting-state (eyes open/closed) and P300 auditory oddball task

    Notes:
        - 12 recordings are missing from the dataset. The author and Figshare
          support have been notified.
    """

    licence: tp.ClassVar[str] = "CC-BY-4.0"
    bibtex: tp.ClassVar[str] = """
    @article{mumtaz2018machine,
        title={A Machine Learning Framework Involving {{EEG-based}} Functional Connectivity to Diagnose Major Depressive Disorder ({{MDD}})},
        author={Mumtaz, Wajid and Ali, Syed Saad Azhar and Yasin, Mohd Azhar Mohd and Malik, Aamir Saeed},
        year=2018,
        month=feb,
        journal={Medical & Biological Engineering & Computing},
        volume={56},
        number={2},
        pages={233--246},
        issn={1741-0444},
        doi={10.1007/s11517-017-1685-z},
        langid={english},
    }

    @misc{mumtaz2016mdd,
        url = "https://figshare.com/articles/dataset/EEG_Data_New/4244171",
        title={{{MDD Patients}} and {{Healthy Controls EEG Data}} ({{New}})},
        author={Mumtaz, Wajid},
        year=2016,
        month=nov,
        publisher={figshare},
        url="https://figshare.com/articles/dataset/EEG_Data_New/4244171",
        doi={10.6084/m9.figshare.4244171.v2},
        langid={english}
    }
    """
    description: tp.ClassVar[str] = "MDD Patients and Healthy Controls EEG Data"
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=180,
        num_subjects=64,
        num_events_in_query=1,
        event_types_in_query={"Eeg"},
        data_shape=(19, 154880),
        frequency=256,
    )
    # 2025-11-24: Wrote to the author of the dataset to notify these files were missing.
    # Will update with response and hopefully solution to these missing runs.
    # 2025-11-25: Wrote to Figshare support team to notify them of the missing files and
    # to see if they may have a solution to this.
    # 2025-11-26: Figshare reached out to the author to inform them of the missing files.
    # In this study, the data are missing for the following:
    _MISSING: tp.ClassVar[tp.Iterable[tp.Iterable[str]]] = [
        ("H", "S12", "EC"),
        ("H", "S18", "EC"),
        ("H", "S20", "TASK"),
        ("H", "S21", "TASK"),
        ("H", "S25", "EO"),
        ("MDD", "S4", "EC"),
        ("MDD", "S7", "EO"),
        ("MDD", "S8", "EC"),
        ("MDD", "S8", "EO"),
        ("MDD", "S9", "TASK"),
        ("MDD", "S12", "EC"),
        ("MDD", "S16", "EC"),
    ]
    _TASK_MAP: tp.ClassVar[dict[str, str]] = {
        "TASK": "P300",
        "EC": "eyeclosed",
        "EO": "eyesopen",
    }
    _DIAGNOSIS_MAP: tp.ClassVar[dict[str, str]] = {
        "H": "healthy",
        "MDD": "major_depressive_disorder",
    }

    def _download(self) -> None:
        dataset_id = "4244171"
        figshare = download.Figshare(study=dataset_id, dset_dir=self.path)
        figshare.download()

    @staticmethod
    def _get_fname(path: Path, basename: str) -> Path:
        dir_path = path / "download"
        exact = dir_path / f"{basename}.edf"
        if exact.exists():
            return exact
        # Some files have an extra space before the task code
        parts = basename.rsplit(" ", 1)
        if len(parts) == 2:
            alt = dir_path / f"{parts[0]}  {parts[1]}.edf"
            if alt.exists():
                return alt
        raise FileNotFoundError(f"Cannot find EDF file for '{basename}' in {dir_path}")

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""

        raw_tasks = ["TASK", "EC", "EO"]
        healthy = product(["H"], [f"S{num}" for num in range(1, 31)], raw_tasks)
        mdd = product(["MDD"], [f"S{num}" for num in range(1, 35)], raw_tasks)

        for diag_code, subject, task_code in chain(healthy, mdd):
            if (diag_code, subject, task_code) in self._MISSING:
                continue
            sub_id = f"{diag_code} {subject}"
            yield dict(
                subject=sub_id,
                task=self._TASK_MAP[task_code],
                task_code=task_code,
                diagnosis=self._DIAGNOSIS_MAP[diag_code],
            )

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:  # type: ignore
        tl = timeline
        basename = f"{tl['subject']} {tl['task_code']}"
        fname = self._get_fname(self.path, basename)
        raw = mne.io.read_raw(fname)
        raw = raw.rename_channels(lambda x: x.replace("EEG ", "").replace("-LE", ""))
        raw = raw.set_montage("standard_1005", on_missing="ignore")
        misc = {"A2-A1", "23A-23R", "24A-24R"} & set(raw.ch_names)
        eeg = set(raw.ch_names) - misc
        ch_types = ["eeg"] * len(eeg) + ["misc"] * len(misc)
        ch_names = list(eeg) + list(misc)
        ch_mapping = dict(zip(ch_names, ch_types))
        raw.set_channel_types(ch_mapping)

        return raw

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        raw = self._load_raw(timeline)
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = dict(type="Eeg", start=0.0, duration=raw.duration, filepath=info)

        return pd.DataFrame([eeg])
