# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import typing as tp
from itertools import product
from pathlib import Path

import mne
import pandas as pd

from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)


class Xu2025Alljoined(study.Study):
    url: tp.ClassVar[str] = (
        "https://huggingface.co/datasets/Alljoined/Alljoined-1.6M/tree/main/raw_eeg"
    )
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
        - Subject 8 has mislabeled files (sessions 1/3/4); handled in code.
        - Image stimuli are extracted from a zip archive during download.
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
        url={https://huggingface.co/datasets/Alljoined/Alljoined-1.6M/tree/main/raw_eeg}
    }
    """
    licence: tp.ClassVar[str] = "CC-BY-NC-SA-4.0"
    description: tp.ClassVar[str] = "20 participants watching static images in EEG."

    def _download(self, overwrite=False) -> None:
        accept = os.environ.get("ALLJOINED_ACCEPT_LICENCE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if not accept:
            raise RuntimeError(
                "Alljoined-1.6M is released under CC-BY-NC-SA-4.0 (non-commercial use). "
                "Set ALLJOINED_ACCEPT_LICENCE=1 to accept the licence before downloading."
            )
        hf_org = "Alljoined"
        hf_repo = "Alljoined-1.6M"
        hg = download.Huggingface(org=hf_org, study=hf_repo, dset_dir=self.path)
        if hg.get_success_file().exists() and not overwrite:
            return
        hg.download(overwrite=overwrite)
        self._extract_stimuli(self.path, overwrite=overwrite)

    @classmethod
    def _extract_stimuli(cls, path: Path, overwrite=False) -> None:
        # The HF repo ships the image stimuli only as `stimuli.zip` (top-level
        # `images/*.jpg`); `_load_timeline_events` reads them from the extracted
        # `stimuli/images/` dir, so we must unpack on first download. Gate on
        # whether the images already exist (idempotent) rather than `overwrite`,
        # which previously skipped extraction on every normal download.
        zip_filepath = path / "download" / "stimuli.zip"
        stimuli_dir = path / "download" / "stimuli"
        images_dir = stimuli_dir / "images"
        if images_dir.exists() and not overwrite:
            return
        from zipfile import ZipFile

        with ZipFile(zip_filepath, "r") as FILE:
            FILE.extractall(stimuli_dir)

        logger.info("Success: Stimuli extraction complete.")

    @staticmethod
    def _get_fname(path: str | Path, subject: int, session: int, run: int) -> Path:
        folder, suffix = "raw_eeg", ".edf"
        dir_path = (
            Path(path)
            / "download"
            / folder
            / f"sub-{subject:02d}"
            / f"session_{session:02d}"
            / f"block_{run:02d}"
        )
        # This subject is mislabeled within its directory so we update
        # the logic here to find the correct file
        # Also, I made a PR to notify the team these files were mislabeled
        # and they only corrected one of the runs
        # https://huggingface.co/datasets/Alljoined/Alljoined-1.6M/discussions/3
        mislabeled = list(product([8], [1], range(2, 20))) + list(
            product([8], [3, 4], range(1, 20))
        )
        if (subject, session, run) in mislabeled:
            subject = 19
        pattern = f"Subject {subject}, Session {session}, Block {run}*{suffix}"
        matches = [f for f in dir_path.glob(pattern)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected 1 match, got {len(matches)} for {pattern} in {dir_path}"
            )
        fpath = matches[0]

        return fpath

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
        """
        NOTE: EDF annotations were validated against the json annotation files accompanying the dataset.
        """
        raw = self._load_raw(timeline)
        frequency = raw.info["sfreq"]
        # extract annotations
        events_df = raw.annotations.to_data_frame(time_format=None)
        events_df.rename(columns={"onset": "start"}, inplace=True)
        # stimulus presentation defined in the study as 100ms
        events_df["duration"] = 0.1
        events_df["type"] = "Image"
        events_df[["label", "value", "deleted", "orig_index"]] = events_df[
            "description"
        ].str.split(",", expand=True)
        events_df["filepath"] = events_df["value"].apply(
            lambda x: str(
                Path(self.path).resolve()
                / "download"
                / "stimuli"
                / "images"
                / f"{int(x):05d}.jpg"
            )
        )
        # add raw event
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = dict(type="Eeg", filepath=info, frequency=frequency, start=0)
        events_df = pd.concat([pd.DataFrame([eeg]), events_df], ignore_index=True)
        return events_df
