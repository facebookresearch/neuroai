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


class Hinss2023Open(study.Study):
    """COG-BCI Database (COG-BCI): EEG responses to cognitive tasks.

    The COG-BCI database contains recordings of 29 participants over 3 sessions
    performing tasks designed to elicit different cognitive states, totaling over
    100 hours of EEG and ECG data. Tasks include psychomotor vigilance (PVT),
    Flanker, N-back (0-, 1-, 2-back), and Multi-Attribute Task Battery (MATB).

    Experimental Design:
        - EEG recordings (62-channel, 500 Hz)
        - 29 participants
        - 3 sessions per participant, 8 task conditions per session
        - Paradigm: Passive BCI with cognitive tasks (PVT, Flanker, N-back, MATB)
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("COG-BCI",)

    bibtex: tp.ClassVar[str] = """
    @article{hinss2023open,
        title={Open multi-session and multi-task EEG cognitive dataset for passive brain-computer interface applications},
        author={Hinss, Marcel F and Jahanpour, Emilie S and Somon, Bertille and Pluchon, Lou and Dehais, Fr{\'e}d{\'e}ric and Roy, Rapha{\"e}lle N},
        journal={Scientific Data},
        volume={10},
        number={1},
        pages={85},
        year={2023},
        publisher={Nature Publishing Group UK London},
        doi={10.1038/s41597-022-01898-y}
    }

    @dataset{hinss2022cogdata,
        title={{COG-BCI} Database: A Multi-Session and Multi-Task {EEG} Cognitive Dataset for Passive Brain-Computer Interfaces},
        author={Hinss, Marcel F and Jahanpour, Emilie S and Somon, Bertille and Pluchon, Lou and Dehais, Fr{\'e}d{\'e}ric and Roy, Rapha{\"e}lle N},
        year={2022},
        publisher={Zenodo},
        doi={10.5281/zenodo.6874129},
        url={https://zenodo.org/records/7413650}
    }
    """
    url: tp.ClassVar[str] = "https://zenodo.org/records/7413650"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "EEG recordings from 29 participants over 3 sessions performing cognitive tasks "
        "(PVT, Flanker, N-back, MATB), totaling over 100 hours of EEG and ECG data."
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=696,
        num_subjects=29,
        num_events_in_query=273,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(62, 306060),
        frequency=500,
    )
    _SUBJECT_RUNS: tp.ClassVar[dict[str, bool]] = {
        f"sub-{key:02d}": False for key in range(1, 3)
    } | {f"sub-{key:02d}": True for key in range(3, 30)}
    _TASKS: tp.ClassVar[list[str]] = [
        "PVT",
        "Flanker",
        "twoBACK",
        "oneBACK",
        "zeroBACK",
        "MATBeasy",
        "MATBmed",
        "MATBdiff",
    ]

    def _download(self) -> None:
        zenodo = download.Zenodo(study="COG-BCI", dset_dir=self.path, record_id="7413650")
        zenodo.download()

    def _get_filepath(self, timeline: dict[str, tp.Any]):
        tl = timeline
        root = self.path / "download"
        sub_dir = root / tl["subject"]
        if self._SUBJECT_RUNS[tl["subject"]]:
            sub_dir = sub_dir / tl["subject"]
        filepath = (sub_dir / tl["session"] / "eeg" / tl["task"]).with_suffix(".set")

        return filepath

    def iter_timelines(self):
        for sub_id in self._SUBJECT_RUNS.keys():
            for task in self._TASKS:
                for ii in range(1, 4):
                    ses_id = f"ses-S{ii}"
                    yield dict(subject=sub_id, session=ses_id, task=task)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        filepath = self._get_filepath(timeline)
        events = mne.read_annotations(filepath).to_data_frame(time_format=None)
        events.rename(columns={"onset": "start", "description": "trigger"}, inplace=True)

        df = pd.read_csv(self.path / "download" / "triggerlist.txt")
        df["code"] = df["code"].astype("str")
        mapping = dict(zip(df["code"], df["content"]))
        # each task has a distinct set of events associated with them
        # the description consists of task name (in caps) and the stimulus event
        # and the events correspond to the onset of the stimulus presentation
        events["description"] = events["trigger"].map(lambda x: mapping.get(x, x))
        events["type"] = "Stimulus"

        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = dict(type="Eeg", filepath=info, start=0)
        events = pd.concat([pd.DataFrame([eeg]), events], ignore_index=True)

        return events

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        filepath = self._get_filepath(timeline)
        raw = mne.io.read_raw_eeglab(filepath, verbose=False)
        raw.set_channel_types({"ECG1": "ecg"})
        # get_chanlocs files missing from sub-25 to sub-29
        filepath_chs = filepath.parent / "chanlocs" / "get_chanlocs.txt"
        if filepath_chs.exists():
            df = pd.read_csv(
                filepath_chs, sep=r"\s+", header=None, names=["ch_names", "X", "Y", "Z"]
            )
            fids = dict()
            for ch_name, fid in [("nas", "nasion"), ("lhj", "lpa"), ("rhj", "rpa")]:
                row = df.query(f"ch_names == '{ch_name}'")
                fids[fid] = list(zip(row.X, row.Y, row.Z))[0]

            idx = df["ch_names"].isin(("nas", "lhj", "rhj", "ECG1"))
            chs = df[~idx]
            ch_pos = dict(zip(chs["ch_names"], zip(chs.X, chs.Y, chs.Z)))

            dig = mne.channels.make_dig_montage(
                ch_pos, nasion=fids["nasion"], lpa=fids["lpa"], rpa=fids["rpa"]
            )

            raw.set_montage(dig)

        return raw
