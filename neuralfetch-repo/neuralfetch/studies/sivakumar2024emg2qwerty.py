# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NM000104 (CTRL-Labs emg2qwerty) — surface-EMG keystroke decoding."""

from __future__ import annotations

import logging
import string
import typing as tp
from pathlib import Path

import mne_bids
import pandas as pd
import pydantic

from neuralfetch import download
from neuralset.events import etypes, study

LOGGER = logging.getLogger(__name__)

# CTC vocabulary from Sivakumar et al. (NeurIPS 2024): 98 keys + blank.
# Inlined here so the events dataframe can carry pre-computed integer
# labels on the Keystroke event (no string→int mapping needed at
# encode time).
_VOCAB: tuple[str, ...] = (
    *string.ascii_letters,
    *string.digits,
    *string.punctuation,
    "Key.backspace",
    "Key.enter",
    "Key.space",
    "Key.shift",
)
PAPER_KEY_TO_LABEL: dict[str, int] = {key: i for i, key in enumerate(_VOCAB)}
PAPER_NULL_CLASS = len(_VOCAB)  # 98 — canonical CTC blank index
PAPER_NUM_CLASSES = PAPER_NULL_CLASS + 1  # 99 — CTC head width


class BidsEmg(etypes.Emg):
    """BIDS EMG event — reads via ``mne_bids.read_raw_bids``.

    mne_bids ≥ 0.19 reads the channel types from ``channels.tsv`` and the
    units from ``channels.tsv`` / ``_emg.json``, so the EMG channels are
    returned in microvolts with no manual rescaling needed.
    """

    def _read(self) -> tp.Any:
        bp = mne_bids.get_bids_path_from_fname(self.filepath)
        return mne_bids.read_raw_bids(bp, verbose=False)


class Sivakumar2024Emg2qwerty(study.Study):
    """emg2qwerty (CTRL-Labs, NeurIPS 2024) — surface-EMG keystroke decoding."""

    bibtex: tp.ClassVar[str] = """
    @inproceedings{NEURIPS2024_a64d5307,
        author = {Sivakumar, Viswanath and Seely, Jeffrey and Du, Alan and
                  Bittner, Sean R and Berenzweig, Adam and Bolarinwa, Anuoluwapo and
                  Gramfort, Alexandre and Mandel, Michael I},
        title = {emg2qwerty: A Large Dataset with Baselines for Touch Typing
                 using Surface Electromyography},
        booktitle = {Advances in Neural Information Processing Systems},
        editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and
                  U. Paquet and J. Tomczak and C. Zhang},
        pages = {91373--91389},
        publisher = {Curran Associates, Inc.},
        doi = {10.52202/079017-2899},
        url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/a64d53074d011e49af1dfc72c332fe4b-Paper-Datasets_and_Benchmarks_Track.pdf},
        volume = {37},
        year = {2024},
    }
    """
    description: tp.ClassVar[str] = (
        "108 subjects doing surface typing with an EMG wristband on each arm."
    )
    aliases: tp.ClassVar[tuple[str, ...]] = ("emg2qwerty", "nm000104")

    NEMAR_DATASET_ID: tp.ClassVar[str] = "nm000104"

    _bids_root_cache: Path | None = pydantic.PrivateAttr(default=None)

    def _download(self) -> None:
        download.Eegdash(study=self.NEMAR_DATASET_ID, dset_dir=self.path).download()

    @property
    def bids_root(self) -> Path:
        """Path to the BIDS-formatted dataset root.

        eegdash caches a dataset's BIDS tree under
        ``cache_dir / <dataset_id>`` (see ``EEGDashBaseDataset``).  With
        ``cache_dir`` set to ``self.path / "download"``, the ``sub-*``
        tree therefore lives at ``self.path / "download" / <dataset_id>``.
        Users with an existing NM000104 BIDS tree should symlink it there.
        """
        if self._bids_root_cache is not None:
            return self._bids_root_cache
        candidate = Path(self.path) / "download" / self.NEMAR_DATASET_ID
        if not (candidate.is_dir() and any(candidate.glob("sub-*"))):
            raise FileNotFoundError(
                f"No BIDS tree found under {candidate!s}: expected "
                f"``sub-*`` directories.  Run ``Study.download()`` first, "
                f"or symlink an existing BIDS-formatted copy of "
                f"{self.NEMAR_DATASET_ID} into ``{candidate!s}``."
            )
        self._bids_root_cache = candidate
        return candidate

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for bp in mne_bids.find_matching_paths(
            root=self.bids_root,
            datatypes="emg",
            suffixes="emg",
            extensions=".bdf",
        ):
            yield {"subject": bp.subject, "session": bp.session}

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        bp = mne_bids.BIDSPath(
            root=self.bids_root,
            subject=timeline["subject"],
            session=timeline["session"],
            task="typing",
            datatype="emg",
            suffix="emg",
            extension=".bdf",
        )
        # Light path: read just the events sidecar TSV; the BDF stays
        # closed until ``BidsEmg._read`` opens it per segment.
        ev = pd.read_csv(
            bp.copy().update(suffix="events", extension=".tsv").fpath,
            sep="\t",
        ).rename(columns={"onset": "start"})

        # NM000104 prompt_text often ends with the two-char literal
        # "\\n"; rstrip would chew real trailing 'n' / '\\'.
        text = ev["prompt_text"].astype("string").str.removesuffix("\\n").str.strip()
        sent_mask = (ev["value"] == "prompt") & text.notna() & (text != "")
        sentences = pd.DataFrame(
            {
                "type": "Sentence",
                "start": ev.loc[sent_mask, "start"],
                "duration": ev.loc[sent_mask, "duration"],
                "text": text[sent_mask],
                "language": "en",
            }
        )

        key = ev["key"].astype("string").str.strip()
        # OOV keys (modifier-key variants not in the paper vocab) get
        # NaN here and are dropped from the events stream, so the CTC
        # target never holds a label outside [0, PAPER_NULL_CLASS).
        label = key.map(PAPER_KEY_TO_LABEL)
        ks_mask = (
            ev["value"].str.startswith("keystroke_", na=False)
            & (key != "")
            & label.notna()
        )
        keystrokes = pd.DataFrame(
            {
                "type": "Keystroke",
                "start": ev.loc[ks_mask, "start"],
                "duration": ev.loc[ks_mask, "duration"].fillna(0.0),
                "text": key[ks_mask],
                # Pandas nullable ``Int64`` (capital I): survives the
                # downstream ``pd.concat`` with rows that lack a
                # ``label`` (raw / sentences) without being demoted to
                # ``float64``.  ``SequenceLabelEncoder`` calls
                # ``int(...)`` on read, so numpy int64 / pandas NA both
                # round-trip cleanly for the filtered keystroke rows.
                "label": label[ks_mask].astype("Int64"),
                "language": "en",
            }
        )

        raw_row = pd.DataFrame(
            [
                {
                    "type": "BidsEmg",
                    "filepath": str(bp.fpath),
                    "start": 0.0,
                    "subject": timeline["subject"],
                }
            ]
        )
        return pd.concat([raw_row, sentences, keystrokes], ignore_index=True)
