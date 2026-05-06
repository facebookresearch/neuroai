# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import mne
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from neuralfetch import download
from neuralset.events import study


class Dan2023Bids(study.Study):
    url: tp.ClassVar[str] = "https://zenodo.org/records/10259996"
    """CHB-MIT: EEG recordings during seizure monitoring in pediatric participants.

    This BIDS-formatted version of the CHB-MIT Scalp EEG Database contains long-term
    EEG monitoring recordings from pediatric participants (ages 1.5-22) with
    intractable seizures at Children's Hospital Boston. Participants were monitored
    for up to several days following withdrawal of anti-seizure medication to
    characterize seizures and assess candidacy for surgical intervention.

    Experimental Design:
        - EEG recordings (18-channel, 256 Hz)
        - 23 participants (22 unique individuals; 5 males, 17 females)
        - Multiple recording sessions with varying numbers of runs per participant
        - Paradigm: continuous seizure monitoring
            * Long-term EEG recording during anti-seizure medication withdrawal
            * Seizure events annotated in BIDS events files

    Notes:
        - Subject 21 in the original dataset was consolidated as a second session
          of subject 01 in this BIDS version.
        - The original dataset was collected at Children's Hospital Boston and
          published on PhysioNet (doi: 10.13026/C2K01R).
        - Channel count varies across participants; 18 channels reflects the
          reference timeline (timeline_index 144).
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("CHB-MIT Scalp EEG Database", "CHB-MIT")

    licence: tp.ClassVar[str] = "ODC-By-1.0"
    bibtex: tp.ClassVar[str] = """
    @misc{dan2023bids,
        author={Dan, Jonathan and Shoeb, Ali},
        title={BIDS CHB-MIT Scalp EEG Database},
        month=Dec,
        year=2023,
        publisher={EPFL},
        version={v1.0.0},
        doi={10.5281/zenodo.10259996},
        url={https://doi.org/10.5281/zenodo.10259996},
    }

    @misc{guttag2010chb,
        title={{{CHB-MIT Scalp EEG Database}}},
        author={Guttag, John},
        date={2010},
        publisher={physionet.org},
        doi={10.13026/C2K01R},
        url={https://physionet.org/content/chbmit/},
    }
    """
    description: tp.ClassVar[str] = """
    This database, collected at the Children's Hospital Boston, consists of EEG recordings from pediatric participants with intractable seizures.
    Participants were monitored for up to several days following withdrawal of anti-seizure medication in order to characterize their seizures and assess their candidacy for surgical intervention.
    The recordings are grouped into 23 cases and were collected from 22 participants (5 males, ages 3-22; and 17 females, ages 1.5-19).
    """
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        # subject 1 was originally reported under two subject ids. this version of the dataset consolidates into one id, two sessions.
        num_timelines=686,
        num_subjects=23,
        query="timeline_index == 144",  # timeline with 1 seizure
        num_events_in_query=2,
        event_types_in_query={"Eeg", "Seizure"},
        data_shape=(18, 921600),
        frequency=256,
    )
    _SUBJECT_RUNS: tp.ClassVar[dict[str, list[list[int]]]] = {
        "01": [list(range(0, 42)), list(range(0, 33))],
        "02": [list(range(0, 36))],
        "03": [list(range(0, 38))],
        "04": [list(range(0, 42))],
        "05": [list(range(0, 39))],
        "06": [list(range(0, 18))],
        "07": [list(range(0, 19))],
        "08": [list(range(0, 20))],
        "09": [list(range(0, 19))],
        "10": [list(range(0, 25))],
        "11": [list(range(0, 35))],
        "12": [list(range(0, 24))],
        "13": [list(range(0, 33))],
        "14": [list(range(0, 26))],
        "15": [list(range(0, 40))],
        "16": [list(range(0, 19))],
        "17": [list(range(0, 21))],
        "18": [list(range(0, 36))],
        "19": [list(range(0, 30))],
        "20": [list(range(0, 29))],
        # subject 21 in the original study was actually
        # the second session of subject 1.
        "22": [list(range(0, 31))],
        "23": [list(range(0, 9))],
        "24": [list(range(0, 22))],
    }

    def _download(self) -> None:
        zenodo = download.Zenodo(
            study="CHB-MIT-EEG-Corpus", dset_dir=self.path, record_id="10259996"
        )
        zenodo.download()

    def _get_bids_path(self, timeline: dict[str, tp.Any]) -> BIDSPath:
        """Returns the BIDS path for a study"""
        tl = timeline
        bids_path = BIDSPath(
            subject=tl["subject"],
            session=tl["session"],
            task=tl["task"],
            run=tl["run"],
            root=self.path / "download" / "BIDS_CHB-MIT",
            datatype="eeg",
        )
        return bids_path

    def iter_timelines(self):
        for sub_id, sessions in self._SUBJECT_RUNS.items():
            for ii, runs in enumerate(sessions, 1):
                ses_id = f"{ii:02d}"
                for jj in runs:
                    run_id = f"{jj:02d}"
                    yield dict(
                        subject=sub_id, session=ses_id, run=run_id, task="szMonitoring"
                    )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        bids_path = self._get_bids_path(timeline)
        bids_path.update(suffix="events")

        events = pd.read_csv(bids_path.fpath, sep="\t")
        events.rename(columns={"onset": "start"}, inplace=True)
        events["state"] = events["eventType"].map(
            lambda x: "seiz" if x.startswith("sz") else x
        )
        events["type"] = "Seizure"
        events = events[["type", "start", "duration", "state"]]

        # Neuralset convention: remove "background" events as they correspond to no event.
        # Instead, use events transforms to add them back if desired.
        events = events[events.state != "bckg"]

        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = dict(type="Eeg", filepath=info, start=0)
        events = pd.concat([pd.DataFrame([eeg]), events], ignore_index=True)

        return events

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        # EEG uses a double banana bipolar montage
        raw = read_raw_bids(self._get_bids_path(timeline), verbose=False)
        mne.rename_channels(
            raw.info, mapping=lambda x: x.replace("FP", "Fp").replace("Z", "z")
        )

        return raw
