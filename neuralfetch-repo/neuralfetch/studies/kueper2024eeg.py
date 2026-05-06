# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import mne
import numpy as np
import pandas as pd

from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)


class Kueper2024Eeg(study.Study):
    url: tp.ClassVar[str] = "https://zenodo.org/records/8345429"
    """Kueper2024Eeg: EEG responses to orthosis-induced movement errors.

    This dataset contains EEG recordings of participants whose right arm was moved
    by an active orthosis device performing elbow flexion and extension. During
    movements, errors were deliberately introduced by briefly reversing the orthosis
    direction. Participants indicated error detection by squeezing an air-filled ball.

    This dataset was part of the CC6 Challenge from the IJCAI 2023 competition:
    Intrinsic Error Evaluation during Human-Robot Interaction (IntEr-HRI).

    Experimental Design:
        - EEG recordings (64-channel, 500 Hz)
        - 8 participants
        - Multiple runs per participant (8-11 runs each)
        - Paradigm: error detection during orthosis-assisted arm movements
            * Elbow flexion and extension movements
            * Deliberate errors introduced by reversing orthosis direction
            * Button press (ball squeeze) response to detected errors

    """

    aliases: tp.ClassVar[tuple[str, ...]] = (
        "Intrinsic Error Evaluation during Human-Robot Interaction",
        "IntEr-HRI",
    )

    bibtex: tp.ClassVar[str] = """
    @article{kueper2024eeg,
        title={{{EEG}} and {{EMG}} Dataset for the Detection of Errors Introduced by an Active Orthosis Device},
        author={Kueper, Niklas and Chari, Kartik and B{\"u}tef{\"u}r, Judith and Habenicht, Julia and Rossol, Tobias and Kim, Su Kyoung and Tabie, Marc and Kirchner, Frank and Kirchner, Elsa Andrea},
        year=2024,
        month=jan,
        journal={Frontiers in Human Neuroscience},
        volume={18},
        publisher={Frontiers},
        issn={1662-5161},
        doi={10.3389/fnhum.2024.1304311},
        langid={english}
    }

    @misc{kueper2023eeg,
        title={{{EEG}} and {{EMG}} Dataset for the Detection of Errors Introduced by an Active Orthosis Device ({{IJCAI}}'23 {{CC6 Competition}})},
        author={Kueper, Niklas and Chari, Kartik and B{\"u}tef{\"u}r, Judith and Habenicht, Julia and Rossol, Tobias and Kim, Su Kyoung and Tabie, Marc and Kirchner, Frank and Kirchner, Elsa Andrea},
        year=2023,
        month=may,
        publisher={Zenodo},
        doi={10.5281/zenodo.8345429},
        url={https://zenodo.org/records/8345429}
    }
    """
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = """
    This dataset contains recordings of the electroencephalogram (EEG) data from eight participants who were assisted in moving their right arm by an active orthosis.
    The orthosis-supported movements were elbow joint movements, i.e., flexion and extension of the right arm.
    While the orthosis was actively moving the participant's arm, some errors were deliberately introduced for a short duration of time.
    During this time, the orthosis moved in the opposite direction. The errors are very simple and easy to detect.
    """
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=82,
        num_subjects=8,
        num_events_in_query=69,
        event_types_in_query={"Eeg", "Event", "Action", "Stimulus"},
        data_shape=(64, 161270),
        frequency=500,
    )
    _SUBJECT_RUNS: tp.ClassVar[dict[str, dict[str, list]]] = {
        "AA56D": {"date": ["20230427"], "runs": list(set(range(1, 12)) - set([8]))},
        "AC17D": {"date": ["20230424"], "runs": list(range(1, 11))},
        "AJ05D": {"date": ["20230426"], "runs": list(set(range(1, 13)) - set([8, 9]))},
        "AQ59D": {
            "date": ["20230421"],
            "runs": list(range(2, 12)),
            "additional_runs": [12],
        },
        # https://pmc.ncbi.nlm.nih.gov/articles/PMC10839100/#F1:~:text=AW59D%2C%20set1%3A%20Artifacts%20and%20Noise.
        # set 1 is excluded in the paper but appears in the dataset
        "AW59D": {"date": ["20230425"], "runs": list(range(2, 11))},
        "AY63D": {
            "date": ["20230425"],
            "runs": list(range(1, 11)),
            "additional_runs": [11],
        },
        "BS34D": {
            "date": ["20230426"],
            "runs": list(set(range(1, 12)) - set([9])),
            "additional_runs": [9],
        },
        "BY74D": {"date": ["20230424"], "runs": list(range(1, 11))},
    }

    def _download(self) -> None:
        zenodo = download.Zenodo(study="EEG", dset_dir=self.path, record_id="8345429")
        zenodo.download()

    def _get_basename(self, timeline: dict[str, tp.Any]):
        tl = timeline
        sub_dir = self.path / "download" / "EEG" / tl["subject"] / "data"
        sub = tl["subject"]
        sub_mapping = self._SUBJECT_RUNS[sub]
        date = sub_mapping["date"][0]
        if int(tl["run"]) not in sub_mapping["runs"]:
            sub_dir = sub_dir / "additional sets"
        infix = "orthosisErrorIjcai_multi_set"
        basename = sub_dir / f"{date}_{sub}_{infix}{tl['run']}"

        return basename

    def iter_timelines(self):
        for sub_id, info in self._SUBJECT_RUNS.items():
            for run_id in info["runs"] + info.get("additional_runs", list()):
                yield dict(subject=sub_id, run=str(run_id), task="orthosisError")

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        basename = self._get_basename(timeline)
        anns = mne.read_annotations(basename.with_suffix(".vmrk"))
        events = anns.to_data_frame(time_format=None)
        events.rename(columns={"onset": "start"}, inplace=True)
        events["description"] = events["description"].replace(
            {
                "Stimulus/S  1": "emg_sync",
                "Stimulus/S 16": "unknown",
                "Stimulus/S 64": "flexion",
                "Stimulus/S 48": "mid_movement",
                "Stimulus/S 32": "extension",
                "Stimulus/S 96": "error",  # Error was introduced in the trial
                "Stimulus/S 80": "button_press",  # Subjects squeezed an air-filled ball after recognizing an error
            }
        )
        events = events[
            np.logical_not(events.description.isin(["unknown", "New Segment/"]))
        ]
        events["type"] = events.description.map(
            {
                "emg_sync": "Event",
                "flexion": "Action",
                "extension": "Action",
                "mid_movement": "Stimulus",
                "error": "Stimulus",
                "button_press": "Action",
            }
        )
        mapping = {"mid_movement": 0, "error": 1}
        events["code"] = events.description.map(mapping)
        unmapped = events[events.type.isna()]
        if unmapped.size:
            raise RuntimeError(f"Unmapped events:\n{unmapped.to_string()}")

        # Identify false negatives
        error_inds = events[events.description == "error"].index
        is_false_negative = events.loc[error_inds + 1].description != "button_press"
        events.loc[error_inds[is_false_negative], "code"] = 2  # FN

        # Identify false positives
        button_inds = events[events.description == "button_press"].index
        is_false_positive = events.loc[button_inds - 1].description != "error"
        events.loc[button_inds[is_false_positive], "code"] = 3  # FP

        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = dict(type="Eeg", filepath=info, start=0)
        events = pd.concat([pd.DataFrame([eeg]), events], ignore_index=True)

        return events

    def _load_raw(self, timeline):
        basename = self._get_basename(timeline)
        filepath = basename.with_suffix(".vhdr")

        raw = mne.io.read_raw(filepath)
        raw.set_montage("brainproducts-RNP-BA-128", match_case=False)

        return raw
