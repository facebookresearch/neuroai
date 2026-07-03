# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import mne
import pandas as pd

from neuralset.events import study

SFREQ = 500.0
TASK = "alicelistening"


class Brennan2019Hierarchical(study.Study):
    """Brennan2019Hierarchical: EEG responses to naturalistic narrative listening.

    EEG recordings from participants listening to ~12 minutes of the "Alice in
    Wonderland" audiobook in English (chapter 1). The study investigates hierarchical
    syntactic structure and rapid linguistic predictions during naturalistic language
    comprehension.

    Re-hosted on NEMAR as ``nm000180`` in BIDS (BrainVision, 60-channel EEG @ 500 Hz).
    Word/Sentence events come from the BIDS ``*_events.tsv`` (EEG-aligned word onsets
    plus linguistic annotations: sentence id, segment, lexicality, n-gram/RNN/CFG
    surprisal), so no trigger recovery is needed.

    Experimental Design:
        - EEG recordings (60-channel, 500 Hz)
        - ~33-45 participants, 1 session per participant
        - Paradigm: Naturalistic listening to "Alice in Wonderland" audiobook (English)
    """

    url: tp.ClassVar[str] = "https://nemar.org/dataexplorer/detail?dataset_id=nm000180"

    bibtex: tp.ClassVar[str] = """
    @article{brennan2019hierarchical,
        title={Hierarchical structure guides rapid linguistic predictions during naturalistic listening},
        author={Brennan, Jonathan R and Hale, John T},
        journal={PloS one},
        volume={14},
        number={1},
        pages={e0207741},
        year={2019},
        publisher={Public Library of Science San Francisco, CA USA},
        doi={10.1371/journal.pone.0207741},
        url={https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0207741}
    }

    @misc{brennan2018eeg,
        doi={10.7302/Z29C6VNH},
        url={http://deepblue.lib.umich.edu/data/concern/generic_works/bg257f92t},
        author={Brennan,  Jonathan R.},
        keywords={Social Sciences,  linguistics,  syntax,  language,  eeg},
        title={EEG Datasets for Naturalistic Listening to "Alice in Wonderland"},
        publisher={University of Michigan},
        year={2018}
    }

    @misc{brennan2024eeg_v2,
        doi={10.7302/746w-g237},
        url={https://deepblue.lib.umich.edu/data/concern/data_sets/bn999738r},
        author={Brennan,  Jonathan R.},
        keywords={Social Sciences,  linguistics,  syntax,  language,  eeg},
        title={EEG Datasets for Naturalistic Listening to "Alice in Wonderland" (v2)},
        publisher={University of Michigan},
        year={2023}
    }
    """
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[
        str
    ] = """EEG from participants listening to ~12 min of an audiobook ("Alice in Wonderland")."""
    requirements: tp.ClassVar[tuple[str, ...]] = ()
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=39,
        num_subjects=39,
        event_types_in_query={"Eeg", "Sentence", "Word"},
        frequency=500.0,
    )

    def _eeg_dir(self, subject: str) -> Path:
        return self.path / "download" / subject / "eeg"

    def _download(self, overwrite: bool = False) -> None:
        """Data is re-hosted on NEMAR nm000180 (BIDS) with per-subject events.tsv.

        The raw BIDS is fetched via nemar-py (``nemar.download(dataset='nm000180')``)
        and the events.tsv from github.com/nemarDatasets/nm000180 (v1.1.3+). If the
        BIDS EEG is already present we skip; otherwise we raise with instructions.
        """
        dl = self.path / "download"
        if list(dl.glob(f"sub-*/eeg/sub-*_task-{TASK}_eeg.vhdr")):
            return
        raise RuntimeError(
            f"Brennan2019 (nm000180) BIDS data not found under {dl}. Download with "
            "nemar-py (dataset='nm000180') and fetch the *_events.tsv from "
            "github.com/nemarDatasets/nm000180."
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """One timeline per BIDS subject that has both raw EEG and an events.tsv."""
        dl = self.path / "download"
        for sub_dir in sorted(dl.glob("sub-*")):
            subject = sub_dir.name
            vhdr = sub_dir / "eeg" / f"{subject}_task-{TASK}_eeg.vhdr"
            events = sub_dir / "eeg" / f"{subject}_task-{TASK}_events.tsv"
            if vhdr.exists() and events.exists():
                yield dict(subject=subject)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.BaseRaw:
        subject = timeline["subject"]
        vhdr = self._eeg_dir(subject) / f"{subject}_task-{TASK}_eeg.vhdr"
        raw = mne.io.read_raw_brainvision(vhdr, verbose=False)
        try:
            raw.set_montage(
                mne.channels.make_standard_montage("easycap-M10"), on_missing="warn"
            )
        except Exception:
            pass
        raw.info["subject_info"] = dict(his_id=subject)
        if abs(raw.info["sfreq"] - SFREQ) > 1e-3:
            raise RuntimeError(f"Expected sfreq {SFREQ}, got {raw.info['sfreq']}")
        return raw

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        subject = timeline["subject"]
        ev_file = self._eeg_dir(subject) / f"{subject}_task-{TASK}_events.tsv"
        df = pd.read_csv(ev_file, sep="\t")
        df = df[df["trial_type"].astype(str) == "word"].reset_index(drop=True)

        words = pd.DataFrame(
            {
                "type": "Word",
                "start": df["onset"].astype(float),
                "duration": df["duration"].astype(float),
                "text": df["word"].astype(str),
                "sequence_id": df["sentence_id"],
                "word_id": df.get("word_index", pd.Series(range(len(df)))),
                "condition": "sentence",
                "language": "english",
                "modality": "heard",
            }
        )

        sents = []
        for sid, g in words.groupby("sequence_id"):
            g = g.sort_values("start")
            start = float(g["start"].iloc[0])
            end = float(g["start"].iloc[-1] + g["duration"].iloc[-1])
            sents.append(
                {
                    "type": "Sentence",
                    "start": start,
                    "duration": end - start,
                    "text": " ".join(g["text"].astype(str).tolist()),
                    "sequence_id": sid,
                    "condition": "sentence",
                    "language": "english",
                    "modality": "heard",
                }
            )
        sentences = pd.DataFrame(sents)

        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = pd.DataFrame([{"type": "Eeg", "filepath": info, "start": 0}])

        events = pd.concat([eeg, words, sentences], ignore_index=True)
        events = _extract_sentences(events)
        return events


def _extract_sentences(events: pd.DataFrame) -> pd.DataFrame:
    """Attach the full sentence text to each Word event (grouped by sequence_id)."""
    events_out = events.copy()
    words = events.loc[events.type == "Word"]
    for _, d in words.groupby("sequence_id"):
        for uid in d.index:
            events_out.loc[uid, "sentence"] = " ".join(d.text.astype(str).values)
    return events_out
