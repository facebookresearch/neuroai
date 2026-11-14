# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unilateral finger- and foot-tapping fNIRS (UFFT), NEMAR nm000891."""

import typing as tp
from pathlib import Path

import pandas as pd
from mne_bids import BIDSPath, get_entities_from_fname

from neuralfetch import download
from neuralset.events import study


class Bak2019Open(study.Study):
    """Right-finger, left-finger and foot tapping fNIRS, BIDS-NIRS release.

    30 participants perform 75 tapping trials of 10 s each (25 per class).
    20 source-detector channels over the bilateral motor cortex are stored as
    HbO and HbR (40 channels, mM·cm) at 13.33 Hz in SNIRF.

    Notes
    -----
    - One ``Stimulus`` event per trial; ``code`` is 0/1/2 for
      right finger / left finger / foot tapping (``trial_type`` of
      ``events.tsv``).
    - Files already hold haemoglobin changes: no optical density or
      Beer-Lambert step. This loader does not filter or resample.
    - Recordings are found under ``sub-*/nirs/`` or ``sub-*/ses-*/nirs/``.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("UFFT", "Bak2019Unilateral", "nm000891")
    bibtex: tp.ClassVar[str] = """
    @article{bak2019open,
        title = {Open-Access {fNIRS} Dataset for Classification of Unilateral Finger- and Foot-Tapping},
        author = {Bak, Sujin and Park, Jinwoo and Shin, Jaeyoung and Jeong, Jichai},
        journal = {Electronics},
        volume = {8},
        number = {12},
        pages = {1486},
        year = {2019},
        doi = {10.3390/electronics8121486},
    }
    """
    url: tp.ClassVar[str] = "https://doi.org/10.6084/m9.figshare.9783755.v2"
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = (
        "30 participants; right-finger, left-finger and foot tapping; "
        "40 HbO/HbR fNIRS channels at 13.33 Hz."
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=30,
        num_subjects=30,
        num_events_in_query=76,
        event_types_in_query={"Fnirs", "Stimulus"},
        data_shape=(40, 30003),
        frequency=13.333333,
    )

    _CODES: tp.ClassVar[dict[str, int]] = {
        "right_finger_tapping": 0,
        "left_finger_tapping": 1,
        "foot_tapping": 2,
    }
    NEMAR_DATASET_ID: tp.ClassVar[str] = "nm000891"

    def _download(self, overwrite: bool = False) -> None:
        download.Nemar(study=self.NEMAR_DATASET_ID, dset_dir=self.path).download(
            overwrite=overwrite
        )

    @property
    def bids_root(self) -> Path:
        return self.path / "download" / self.NEMAR_DATASET_ID

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        files = sorted(self.bids_root.glob("sub-*/**/nirs/*_nirs.snirf"))
        if not files:
            raise FileNotFoundError(
                f"No SNIRF recordings in {self.bids_root}; run study.download() first"
            )
        for path in files:
            entities = get_entities_from_fname(path.name)
            yield {k: v for k, v in entities.items() if v is not None}

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        bids = BIDSPath(root=self.bids_root, **timeline, datatype="nirs", suffix="nirs")
        tsv = bids.copy().update(suffix="events", extension=".tsv").fpath
        trials = pd.read_csv(tsv, sep="\t")
        snirf = str(bids.update(extension=".snirf").fpath)
        stimuli = pd.DataFrame(
            dict(
                type="Stimulus",
                start=trials.onset,
                duration=trials.duration,
                code=trials.trial_type.map(self._CODES),
                description=trials.trial_type,
            )
        )
        fnirs = pd.DataFrame([dict(type="Fnirs", start=0.0, filepath=snirf)])
        return pd.concat([fnirs, stimuli], ignore_index=True)
