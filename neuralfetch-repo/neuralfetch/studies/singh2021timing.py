# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import pandas as pd
from mne_bids import BIDSPath

from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)


class Singh2021Timing(study.Study):
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds004579/versions/1.0.0"
    """Singh2021Timing: EEG responses during an interval timing task in Parkinson's disease.

    EEG recordings from Parkinson's disease patients and healthy controls during a
    peak-interval timing task with 3-second and 7-second target intervals. Sourced
    from the authors' OpenNeuro release (ds004579), which is BIDS-compliant with
    EEGLAB ``.set``/``.fdt`` recordings and ``_events.tsv`` sidecars.

    Experimental Design:
        - EEG recordings (63-channel, 500 Hz)
        - 139 recordings (94 Parkinson's disease, 45 healthy controls)
        - 80 trials per session (40 per interval type)
        - Paradigm: peak-interval timing task with visual distractors

    Notes:
        - Population includes Parkinson's disease patients and healthy controls.
        - BIDS uses numeric ``sub-XXX`` identifiers; the original diagnosis/subject
          label (e.g. ``PD1005``, ``Control1025``) is kept in ``participants.tsv``
          under the ``EEG`` column and exposed as ``subject_label``.
        - Single-session only: the 9 PD dual-session recordings of the earlier
          Hugging Face mirror are not part of ds004579.
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

    @misc{singh2021_data,
        title={Interval Timing Task},
        author={Singh, Arun and Cole, Rachel and Espinoza, Arturo and Wessel, Jan R. and Cavanagh, Jim and Narayanan, Nandakumar},
        publisher={OpenNeuro},
        doi={10.18112/openneuro.ds004579.v1.0.0},
        url={https://openneuro.org/datasets/ds004579/versions/1.0.0}
    }
    """
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings in 94 Parkinson's disease patients and 45 controls during "
        "interval timing."
    )
    _TASK: tp.ClassVar[str] = "IntervalTiming"
    _FREQUENCY: tp.ClassVar[int] = 500
    # ``_events.tsv`` "value" markers -> integer codes. The markers are the raw
    # BrainVision triggers, so "Stimulus/S  1" of the original recordings reads
    # as "S  1" here.
    _CODE_MAPPING: tp.ClassVar[dict[str, int]] = {
        "S  1": 1,
        "S  2": 2,
        "S  3": 3,
        "S  4": 4,
        "S  5": 5,
        "S  6": 6,
        "S  7": 7,
        "S255": 8,
        "boundary": 9,
        "R  3": 10,
    }
    _DESCRIPTION_MAPPING: tp.ClassVar[dict[str, str]] = {
        "S  1": "short_interval_instruction",  # 1 s
        "S  2": "long_inverval_instruction",  # 1 s
        "S  3": "interval_start",  # Start of interval (blue rectangle shown)
        # Lasts 8-10 s for short intervals and 18-20 s for long intervals
        "S  4": "spacebar_press",  # XXX Replace with Button event of right duration
        "S  5": "spacebar_release",
        "S  6": "distracting_vowel",
        # XXX Interval end is not available in the dataset
        "S  7": "trial_feedback",  # On 15% of the trials
        "S255": "end_of_last_trial",
        # Not described in the dataset README
        "boundary": "unknown",  # BrainVision "New Segment/"
        "R  3": "unknown",  # BrainVision "Response/R  3"
    }
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=139,
        num_subjects=139,
        num_events_in_query=1760,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(63, 793440),
        frequency=500.0,
    )

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # v2: source moved from a third-party Hugging Face mirror to the authors'
        # OpenNeuro release, changing subject ids, file formats and event parsing.
        self.version = "v2"

    def _download(self, overwrite: bool = False) -> None:
        # The lab's own distribution (https://narayanan.lab.uiowa.edu/datasets)
        # relies on Sharepoint and cannot be fetched programmatically.
        download.Openneuro(study="ds004579", dset_dir=self.path).download(
            overwrite=overwrite
        )

    def _bids_root(self) -> Path:
        return self.path / "download"

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        participants = pd.read_csv(self._bids_root() / "participants.tsv", sep="\t")
        diagnosis_map = {"PD": "parkinsons", "Control": "control"}
        missing = []
        for row in participants.itertuples():
            subject = str(row.participant_id).removeprefix("sub-")
            if not self._eeg_path(subject).fpath.exists():
                missing.append(subject)
                continue
            group = str(row.GROUP)
            yield dict(
                subject=subject,
                subject_label=str(row.EEG),
                diagnosis=diagnosis_map.get(group, group),
            )
        if missing:
            logger.warning(
                "Skipped %d of %d participants with no recording on disk; "
                "the download may be incomplete",
                len(missing),
                len(participants),
            )
            logger.debug("Participants without a recording: %s", ", ".join(missing))

    def _eeg_path(self, subject: str) -> BIDSPath:
        return BIDSPath(
            subject=subject,
            task=self._TASK,
            root=self._bids_root(),
            datatype="eeg",
            suffix="eeg",
            extension=".set",
        )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        eeg_path = self._eeg_path(timeline["subject"])
        events_path = eeg_path.copy().update(suffix="events", extension=".tsv")

        events = pd.read_csv(events_path.fpath, sep="\t")
        events.rename(columns={"onset": "start"}, inplace=True)
        events["type"] = "Stimulus"
        events["code"] = events["value"].map(self._CODE_MAPPING)
        events["description"] = events["value"].map(self._DESCRIPTION_MAPPING)
        # The sidecar "duration" is every marker's BrainVision size field -- a
        # count of *data points* -- written unconverted into a column BIDS
        # defines in seconds, so it reads 1.0 for every event. Convert it back
        # (1 sample, i.e. 2 ms), then time the instruction cues, which are the
        # only events the paradigm really does show for a second.
        durations = pd.to_numeric(events["duration"], errors="coerce")
        events["duration"] = durations.fillna(0.0) / self._FREQUENCY
        events.loc[
            events.description.str.endswith("interval_instruction", na=False),
            "duration",
        ] = 1.0
        events = events[["type", "start", "duration", "code", "description"]]

        eeg = dict(type="Eeg", filepath=eeg_path.fpath, start=0)
        events = pd.concat([pd.DataFrame([eeg]), events], ignore_index=True)
        return events
