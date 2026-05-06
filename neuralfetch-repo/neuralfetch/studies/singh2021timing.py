# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import mne
import pandas as pd

from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)


class Singh2021Timing(study.Study):
    url: tp.ClassVar[str] = (
        "https://huggingface.co/datasets/jalauer/Singh2021Timing/tree/main/data"
    )
    """Singh2021Timing: EEG responses during an interval timing task in Parkinson's disease.

    EEG recordings from 83 Parkinson's disease patients and 37 healthy controls
    during a peak-interval timing task with 3-second and 7-second target intervals.
    A subset of 9 PD patients completed sessions both ON and OFF dopaminergic
    medication.

    Experimental Design:
        - EEG recordings (63-channel, 500 Hz)
        - 120 participants (83 PD, 37 controls), 138 timelines
        - 80 trials per session (40 per interval type)
        - Paradigm: peak-interval timing task with visual distractors

    Notes:
        - Population includes Parkinson's disease patients and healthy controls.
        - 9 PD participants have two sessions (ON/OFF medication) recorded under
          different subject IDs (see _SUBJECT_ALIASES).
    """

    bibtex: tp.ClassVar[str] = """
    @article{singh2021timing,
        title={Timing Variability and Midfrontal \textasciitilde 4 {{Hz}} Rhythms Correlate with Cognition in {{Parkinson}}'s Disease},
        author={Singh, Arun and Cole, Rachel C. and Espinoza, Arturo I. and Evans, Aron and Cao, Scarlett and Cavanagh, James F. and Narayanan, Nandakumar S.},
        year=2021,
        month=feb,
        journal={npj Parkinson's Disease},
        volume={7},
        number={1},
        pages={14},
        publisher={Nature Publishing Group},
        issn={2373-8057},
        doi={10.1038/s41531-021-00158-x},
        copyright={2021 The Author(s)},
        langid={english},
        keywords={Neurophysiology,Neuroscience},
    }
        url = {https://huggingface.co/datasets/jalauer/Singh2021Timing/tree/main/data},

    @misc{singh2021_data,
        url={https://huggingface.co/datasets/jalauer/Singh2021Timing/tree/main/data}
    }
    """
    licence: tp.ClassVar[str] = "PDDL-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 83 Parkinson's disease patients and 37 controls during interval timing."
    )
    _SUBJECT_RUNS: tp.ClassVar[dict[str, tp.Iterable[int]]] = {
        "Control": set(range(1025, 1420, 10)) - set((1045, 1165, 1355)),
        "PD": (
            set(range(1005, 1870, 10))
            | set(
                [
                    2445,
                    2515,
                    2565,
                    2625,
                    2815,
                    2835,
                    2845,
                    2845,
                    2855,
                    2865,
                    3445,
                    3515,
                    3565,
                    3625,
                ]
            )
        )
        - set((1205, 1255, 1345, 1355, 1495, 1545, 1805, 1825)),
    }
    # There are 9 PD subjects with two sessions, but recorded under a different subject name in
    # session 2 (see README in url above)
    _SUBJECT_ALIASES: tp.ClassVar[dict[str, str]] = {
        "PD1815": "PD2815",
        "PD1835": "PD2835",
        "PD1845": "PD2845",
        "PD1855": "PD2855",
        "PD1865": "PD2865",
        "PD3445": "PD2445",
        "PD3515": "PD2515",
        "PD3565": "PD2565",
        "PD3625": "PD2625",
    }
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=138,
        num_subjects=129,
        num_events_in_query=882,  # query=1st timeline
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(63, 733750),
        frequency=500,
    )

    def _download(self, overwrite=False) -> None:
        # https://predict.cs.unm.edu/downloads.php d014
        # Unable to use original dataset on PRED+ct.
        # Files shared here rely on Sharepoint
        # and cannot be programmatically downloaded there.
        # Alternate source found on Hugging Face, not uploaded by original authors.
        # https://huggingface.co/jalauer/datasets
        hf_org = "jalauer"
        hf_repo = "Singh2021Timing"
        hg = download.Huggingface(org=hf_org, study=hf_repo, dset_dir=self.path)
        if hg.get_success_file().exists() and not overwrite:
            return
        hg.download(overwrite=overwrite)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        for diagnosis, subjects in self._SUBJECT_RUNS.items():
            for subject in sorted(subjects):
                sub_id = f"{diagnosis}{subject}"
                sessions = [1, 2] if sub_id in self._SUBJECT_ALIASES else [1]
                for session in sessions:
                    yield dict(subject=sub_id, session=session, diagnosis=diagnosis)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        tl = timeline
        subject = (
            self._SUBJECT_ALIASES.get(tl["subject"], tl["subject"])
            if tl["session"] == 2
            else tl["subject"]
        )
        basename = self.path / "download" / "data" / subject

        # extract annotations
        events = mne.read_annotations(basename.with_suffix(".vmrk")).to_data_frame(
            time_format=None
        )
        events.rename(columns={"onset": "start"}, inplace=True)
        events["type"] = "Stimulus"
        events["code"] = events.description.map(
            {
                "Stimulus/S  1": 1,
                "Stimulus/S  2": 2,
                "Stimulus/S  3": 3,
                "Stimulus/S  4": 4,
                "Stimulus/S  5": 5,
                "Stimulus/S  6": 6,
                "Stimulus/S  7": 7,
                "Stimulus/S255": 8,
                "New Segment/": 9,
                "Response/R  3": 10,
            }
        )
        events["description"] = events.description.map(
            {
                "Stimulus/S  1": "short_interval_instruction",  # 1 s
                "Stimulus/S  2": "long_inverval_instruction",  # 1 s
                "Stimulus/S  3": "interval_start",  # Start of interval (blue rectangle shown)
                # Lasts 8-10 s for short intervals and 18-20 s for long intervals
                "Stimulus/S  4": "spacebar_press",  # XXX Replace with Button event of right duration
                "Stimulus/S  5": "spacebar_release",
                "Stimulus/S  6": "distracting_vowel",
                # XXX Interval end is not available in the dataset
                "Stimulus/S  7": "trial_feedback",  # On 15% of the trials
                "Stimulus/S255": "end_of_last_trial",
                # Not described in README.md
                "New Segment/": "unknown",
                "Response/R  3": "unknown",
            }
        )
        events.loc[
            (
                (events.type == "Stimulus")
                & events.description.str.endswith("interval_instruction")
            ),
            "duration",
        ] = 1.0

        eeg = dict(type="Eeg", filepath=basename.with_suffix(".vhdr"), start=0)
        events = pd.concat([pd.DataFrame([eeg]), events], ignore_index=True)
        return events
