# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NM000104 (CTRL-Labs emg2qwerty) — surface-EMG keystroke decoding."""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import pandas as pd
import pydantic
from mne.utils import _soft_import
from neuralset.events import etypes, study

LOGGER = logging.getLogger(__name__)
_MNE_BIDS_MIN_VERSION = "0.19"

# mne_bids is a hard dep of neuralfetch; ``strict=False`` keeps the
# module importable in environments without it (the actual calls will
# then fail informatively at use site).
mne_bids = _soft_import(
    "mne_bids",
    "reading the BIDS-formatted NM000104 EMG recordings",
    strict=False,
    min_version=_MNE_BIDS_MIN_VERSION,
)


class Emg2qwertyRaw(etypes.Emg):
    """NM000104 EMG event — reads via ``mne_bids.read_raw_bids``.

    The sidecar load (channel types from ``channels.tsv``, units from
    ``channels.tsv``/``_emg.json``) is what we need; mne_bids ≥0.19
    handles the EMG-unit case correctly, so no manual rescaling here.
    """

    def _read(self) -> tp.Any:
        bp = mne_bids.get_bids_path_from_fname(self.filepath)
        return mne_bids.read_raw_bids(bp, verbose=False)


class Emg2qwerty(study.Study):
    """emg2qwerty (CTRL-Labs, NeurIPS 2024) — surface-EMG keystroke decoding."""

    url: tp.ClassVar[str] = (
        "https://proceedings.neurips.cc/paper_files/paper/2024/file/"
        "a64d53074d011e49af1dfc72c332fe4b-Paper-Datasets_and_Benchmarks_Track.pdf"
    )
    bibtex: tp.ClassVar[str] = (
        "@inproceedings{NEURIPS2024_a64d5307,\n"
        "  author={Sivakumar, Viswanath and Seely, Jeffrey and Du, Alan and\n"
        "          Bittner, Sean R and Berenzweig, Adam and Bolarinwa,\n"
        "          Anuoluwapo and Gramfort, Alexandre and Mandel, Michael I},\n"
        "  title={emg2qwerty: A Large Dataset with Baselines for Touch Typing\n"
        "         using Surface Electromyography},\n"
        "  booktitle={Advances in Neural Information Processing Systems},\n"
        "  editor={A. Globerson and L. Mackey and D. Belgrave and A. Fan and\n"
        "          U. Paquet and J. Tomczak and C. Zhang},\n"
        "  pages={91373--91389},\n"
        "  publisher={Curran Associates, Inc.},\n"
        "  doi={10.52202/079017-2899},\n"
        "  volume={37},\n"
        "  year={2024}}"
    )
    description: tp.ClassVar[str] = (
        "108 subjects doing surface typing with an EMG wristband on each arm."
    )
    aliases: tp.ClassVar[tuple[str, ...]] = ("emg2qwerty", "nm000104")

    NEMAR_DATASET_ID: tp.ClassVar[str] = "nm000104"

    _bids_root_cache: Path | None = pydantic.PrivateAttr(default=None)

    def _download(self) -> None:
        from neuralfetch import download

        download.Eegdash(study=self.NEMAR_DATASET_ID, dset_dir=self.path).download()

    def _bids_root(self) -> Path:
        # ``Study.download`` writes under ``self.path / "download" / ...``;
        # users with a manual BIDS tree placed directly under
        # ``self.path`` also work.  Pick the first candidate whose
        # immediate children contain ``sub-*`` entries.  mne_bids'
        # ``get_entity_vals`` / ``find_matching_paths`` both recurse,
        # so neither distinguishes "BIDS at root" from "BIDS nested
        # one level deeper" -- hence the direct-child glob here.
        # Cached on first call.
        if self._bids_root_cache is not None:
            return self._bids_root_cache
        for candidate in (Path(self.path), Path(self.path) / "download"):
            if candidate.is_dir() and any(candidate.glob("sub-*")):
                self._bids_root_cache = candidate
                return candidate
        raise FileNotFoundError(
            f"No BIDS tree found under {self.path!s}: expected ``sub-*`` "
            f"directories at the root or under ``download/``.  Run "
            f"``Study.download()`` first or point ``path`` at an existing "
            f"BIDS-formatted copy of NM000104."
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for bp in mne_bids.find_matching_paths(
            root=self._bids_root(),
            datatypes="emg", suffixes="emg", extensions=".bdf",
        ):
            yield {"subject": bp.subject, "session": bp.session}

    def _load_timeline_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        bp = mne_bids.BIDSPath(
            root=self._bids_root(),
            subject=timeline["subject"], session=timeline["session"],
            task="typing", datatype="emg", suffix="emg", extension=".bdf",
        )
        # Light path: read just the events sidecar TSV; the BDF stays
        # closed until ``Emg2qwertyRaw._read`` opens it per segment.
        ev = pd.read_csv(
            bp.copy().update(suffix="events", extension=".tsv").fpath,
            sep="\t",
        ).rename(columns={"onset": "start"})

        # NM000104 prompt_text often ends with the two-char literal
        # "\\n"; rstrip would chew real trailing 'n' / '\\'.
        text = (
            ev["prompt_text"].astype("string")
            .str.removesuffix("\\n").str.strip()
        )
        sent_mask = (ev["value"] == "prompt") & text.notna() & (text != "")
        sentences = pd.DataFrame({
            "type": "Sentence",
            "start": ev.loc[sent_mask, "start"],
            "duration": ev.loc[sent_mask, "duration"],
            "text": text[sent_mask],
            "language": "en",
        })

        key = ev["key"].astype("string").str.strip()
        ks_mask = ev["value"].str.startswith("keystroke_", na=False) & (key != "")
        keystrokes = pd.DataFrame({
            "type": "Keystroke",
            "start": ev.loc[ks_mask, "start"],
            "duration": ev.loc[ks_mask, "duration"].fillna(0.0),
            "text": key[ks_mask],
            "language": "en",
        })

        raw_row = pd.DataFrame([{
            "type": "Emg2qwertyRaw", "filepath": str(bp.fpath),
            "start": 0.0, "subject": timeline["subject"],
        }])
        return (
            pd.concat([raw_row, sentences, keystrokes], ignore_index=True)
            .sort_values("start").reset_index(drop=True)
        )
