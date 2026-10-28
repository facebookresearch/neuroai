# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp
from itertools import product
from pathlib import Path

import mne
import pandas as pd

from neuralfetch import download
from neuralset.events import study


class Xu2025Alljoined(study.Study):
    url: tp.ClassVar[str] = "https://doi.org/10.82901/nemar.nm000134"
    """Alljoined-1.6M: large-scale EEG responses to static images.

    A million-trial EEG dataset from 20 participants viewing static images,
    recorded with affordable consumer-grade EEG (Emotiv). Designed for evaluating
    brain-computer interfaces and EEG-to-image decoding at scale. Each participant
    completed up to 4 sessions with up to 19 blocks each, with 100 ms image
    presentations.

    Experimental Design:
        - EEG recordings (Emotiv, standard 1005 montage, EDF format)
        - 20 participants, up to 4 sessions x 19 blocks each
        - Image presentation duration: 100 ms
        - Paradigm: passive viewing of static images

    Notes:
        - Successor to Alljoined (xu2024.py) with ~10x more data.
        - Markers sharing an onset are dropped as ambiguous; handled in code.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("Alljoined-1.6M",)

    bibtex: tp.ClassVar[str] = """
    @misc{xu2025alljoined,
        title={Alljoined-1.{{6M}}: {{A Million-Trial EEG-Image Dataset}} for {{Evaluating Affordable Brain-Computer Interfaces}}},
        shorttitle={Alljoined-1.{{6M}}},
        author={Xu, Jonathan and Nunes, Ugo Bruzadin and Jiang, Wangshu and Ryther, Samuel and Pringle, Jordan and Scotti, Paul S. and Delorme, Arnaud and Kneeland, Reese},
        year=2025,
        month=aug,
        number={arXiv:2508.18571},
        eprint={2508.18571},
        primaryclass={q-bio},
        publisher={arXiv},
        doi={10.48550/arXiv.2508.18571},
        archiveprefix={arXiv}
    }

    @misc{xu2025_data,
        title={Alljoined-1.6M},
        publisher={NEMAR},
        doi={10.82901/nemar.nm000134},
        url={https://doi.org/10.82901/nemar.nm000134}
    }
    """
    licence: tp.ClassVar[str] = "CC-BY-NC-ND-4.0"
    description: tp.ClassVar[str] = "20 participants watching static images in EEG."

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1520,
        num_subjects=20,
        num_events_in_query=1021,
        event_types_in_query={"Eeg", "Image"},
        data_shape=(32, 76032),
        frequency=256.0,
    )

    def _download(self, overwrite: bool = False) -> None:
        accept = os.environ.get("ALLJOINED_ACCEPT_LICENCE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if not accept:
            raise RuntimeError(
                f"Alljoined-1.6M is released under {self.licence} (non-commercial use). "
                "Set ALLJOINED_ACCEPT_LICENCE=1 to accept the licence before downloading."
            )
        download.Nemar(
            study="nm000134",
            dset_dir=self.path,
            version="1.0.3",
            exclude=["code/**", "sourcedata/**"],
        ).download(overwrite=overwrite)

    @staticmethod
    def _get_fname(path: str | Path, subject: int, session: int, run: int) -> Path:
        folder, suffix = "nm000134", "_eeg.edf"
        dir_path = (
            Path(path)
            / "download"
            / folder
            / f"sub-{subject:02d}"
            / f"ses-{session:02d}"
            / "eeg"
        )
        stem = f"sub-{subject:02d}_ses-{session:02d}_task-images_run-{run:02d}"
        return dir_path / f"{stem}{suffix}"

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        for subject, session, run in product(range(1, 21), range(1, 5), range(1, 20)):
            fname = self._get_fname(self.path, subject, session, run)
            if fname.exists():
                yield dict(subject=str(subject), session=session, run=run)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.BaseRaw:
        # `iter_timelines` yields `subject` as a string for the global index;
        # `_get_fname` formats it with `:02d`, so cast back to int here.
        filepath = str(
            self._get_fname(
                self.path,
                int(timeline["subject"]),
                timeline["session"],
                timeline["run"],
            )
        )
        # When loading, all the channels are marked as EEG
        raw = mne.io.read_raw(filepath)
        # For some of the files, Emotiv writes AFz as Afz
        if "Afz" in raw.ch_names:
            raw.rename_channels({"Afz": "AFz"})
        # Set the montage to channels that correspond to EEG placement
        # Emotiv doesn't use a standard 1020. it's closer to a 1010 placement
        montage = mne.channels.make_standard_montage("standard_1005")
        eeg_ch_names = set.intersection(set(raw.ch_names), set(montage.ch_names))
        misc_ch_names = set.difference(set(raw.ch_names), eeg_ch_names)
        eeg_mapping = list(zip(eeg_ch_names, ["eeg" for _ in eeg_ch_names]))
        misc_mapping = list(zip(misc_ch_names, ["misc" for _ in misc_ch_names]))
        raw.set_channel_types(dict(eeg_mapping + misc_mapping), on_unit_change="ignore")
        raw.set_montage("standard_1005", on_missing="ignore", match_case=False)
        return raw

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        raw = self._load_raw(timeline)
        frequency = raw.info["sfreq"]
        # extract annotations
        events_tsv = str(raw.filenames[0]).replace("_eeg.edf", "_events.tsv")
        events_df = pd.read_csv(events_tsv, sep="\t")
        events_df.rename(columns={"onset": "start", "trial_type": "label"}, inplace=True)
        # the marker stream stalls and flushes several markers on one timestamp
        # (29/1525 recordings, 160 markers): the evoking image is unrecoverable
        collided = events_df["start"].duplicated(keep=False)
        # behav (responses) and debug markers show no image: stim_file is n/a
        keep = ~collided & events_df["stim_file"].notna()
        events_df = events_df[keep].reset_index(drop=True)
        # stimulus presentation defined in the study as 100ms
        events_df["duration"] = 0.1
        events_df["type"] = "Image"
        stimuli = Path(self.path).resolve() / "download" / "nm000134" / "stimuli"
        events_df["filepath"] = events_df["stim_file"].apply(lambda x: str(stimuli / x))
        # add raw event
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = dict(type="Eeg", filepath=info, frequency=frequency, start=0)
        events_df = pd.concat([pd.DataFrame([eeg]), events_df], ignore_index=True)
        return events_df
