# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""PROTEUS BCI Bordeaux mental-command EEG, NEMAR nm000290."""

import typing as tp
from pathlib import Path

import mne
import pandas as pd
from mne_bids import BIDSPath, get_entities_from_fname, read_raw_bids

from neuralfetch import download
from neuralset.events import study


class Dreyer2026Proteus(study.Study):
    """Graz and BrainHero mental-command BCI, EEG-BIDS training release.

    The training release contains 112 runs from 14 sessions of 10
    participants. A session has an eyes-open and an eyes-closed baseline run,
    two acquisition runs and four online runs with feedback, split between the
    Graz and BrainHero interfaces. Cues ask for kinesthetic motor imagery
    (``mi``), mental subtraction (``sub``) or word generation (``word``).
    41 EEG channels are stored at 500 Hz in EDF.

    Notes
    -----
    - One ``Stimulus`` event per cue, from the cue to the end of the trial
      (8 s); ``code`` is 0/1/2 for mi/sub/word. Baseline runs have no cue
      events. ``task`` and ``run`` come from the BIDS file names.
    - Recordings are DC-coupled. This loader does not filter, re-reference,
      resample or clean artifacts.
    - No electrode coordinates are distributed; channel positions come from
      MNE's ``standard_1020`` montage.
    - The NEMAR download is pinned to 1.0.0. An existing BIDS tree directly
      under the study directory is also supported.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("proteus", "nm000290")
    bibtex: tp.ClassVar[str] = """
    @misc{dreyer2026proteus,
        author = {Dreyer, Pauline and Bourdil, Manon and Bechon, Loic and Kojima, Simon and Velut, Sebastien and Rimbert, Sebastien and Roy, Rapha{\\"e}lle and Lotte, Fabien},
        title = {{PROTEUS} {BCI} {Bordeaux} — EEG/EMG Foundation Challenge 2026, Track 02},
        year = {2026},
        publisher = {NEMAR},
        version = {1.0.0},
        doi = {10.82901/nemar.nm000290},
        url = {https://doi.org/10.82901/nemar.nm000290},
    }
    """
    url: tp.ClassVar[str] = "https://doi.org/10.82901/nemar.nm000290"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "Inria Bordeaux: 112 runs from 10 participants; cued motor imagery, mental "
        "subtraction and word generation with Graz and BrainHero interfaces; "
        "41 EEG channels at 500 Hz."
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=112,
        num_subjects=10,
        num_events_in_query=35,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(41, 210000),
        frequency=500,
    )

    _CODES: tp.ClassVar[dict[str, int]] = {"mi": 0, "sub": 1, "word": 2}
    NEMAR_DATASET_ID: tp.ClassVar[str] = "nm000290"

    def _download(self, overwrite: bool = False) -> None:
        download.Nemar(
            study=self.NEMAR_DATASET_ID,
            dset_dir=self.path,
            version="1.0.0",
        ).download(overwrite=overwrite)

    @property
    def bids_root(self) -> Path:
        if any(self.path.glob("sub-*/ses-*/eeg/*_eeg.edf")):
            return self.path
        return self.path / "download" / self.NEMAR_DATASET_ID

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        files = sorted(self.bids_root.glob("sub-*/ses-*/eeg/*_eeg.edf"))
        if not files:
            raise FileNotFoundError(
                f"No EDF recordings in {self.bids_root}; run study.download() first"
            )
        for path in files:
            entities = get_entities_from_fname(path.name)
            yield dict(
                subject=f"sub-{entities['subject']}",
                session=entities["session"],
                task=entities["task"],
                run=entities["run"],
            )

    def _bids_path(self, timeline: dict[str, tp.Any]) -> BIDSPath:
        return BIDSPath(
            root=self.bids_root,
            subject=timeline["subject"].removeprefix("sub-"),
            session=timeline["session"],
            task=timeline["task"],
            run=timeline["run"],
            datatype="eeg",
            suffix="eeg",
            extension=".edf",
        )

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.Raw:
        raw = read_raw_bids(self._bids_path(timeline), verbose="ERROR")
        raw.set_montage("standard_1020")
        return raw

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        raw = self._load_raw(timeline)
        tsv = self._bids_path(timeline).update(suffix="events", extension=".tsv")
        events = pd.read_csv(tsv.fpath, sep="\t")
        cues = events[events.trial_type.isin(self._CODES)]
        eeg = dict(
            type="Eeg",
            start=0.0,
            duration=raw.n_times / raw.info["sfreq"],
            filepath=study.SpecialLoader(
                method=self._load_raw, timeline=timeline
            ).to_json(),
        )
        stimuli = pd.DataFrame(
            dict(
                type="Stimulus",
                start=cues.onset.to_numpy(dtype=float),
                duration=cues.duration.to_numpy(dtype=float),
                code=cues.trial_type.map(self._CODES).to_numpy(),
                description=cues.trial_type.to_numpy(),
            )
        )
        return pd.concat([pd.DataFrame([eeg]), stimuli], ignore_index=True)
