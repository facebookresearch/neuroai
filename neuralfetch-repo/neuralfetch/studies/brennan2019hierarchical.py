# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import zipfile
from pathlib import Path

import mne
import pandas as pd
from scipy.io import loadmat

from neuralfetch import download, utils
from neuralfetch.download import success_writer
from neuralset.events import study

SFREQ = 500.0


class Brennan2019Hierarchical(study.Study):
    """Brennan2019Hierarchical: EEG responses to naturalistic narrative listening.

    EEG recordings from 33 participants listening to approximately 12 minutes of the
    "Alice in Wonderland" audiobook in English. The study investigates hierarchical
    syntactic structure and rapid linguistic predictions during naturalistic language
    comprehension.

    Experimental Design:
        - EEG recordings (60-channel, 500 Hz)
        - 33 participants
        - 1 session per participant
        - Paradigm: Naturalistic listening to "Alice in Wonderland" audiobook (English)
    """

    url: tp.ClassVar[str] = (
        "https://deepblue.lib.umich.edu/data/concern/data_sets/bn999738r"
    )

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
    ] = """EEG recordings from 33 participants listening to 12 min of an audiobook ("Alice in Wonderland").
    """
    requirements: tp.ClassVar[tuple[str, ...]] = ("globus-sdk>=4.5",)
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=33,
        num_subjects=33,
        num_events_in_query=2226,
        event_types_in_query={"Audio", "Eeg", "Sentence", "Word"},
        data_shape=(60, 366525),
        frequency=500.0,
    )

    def _download(self) -> None:
        dl_dir = self.path / "download"

        # Deep Blue Data guest collection + record-scoped path for Brennan2019
        # (https://deepblue.lib.umich.edu/data/concern/data_sets/bn999738r).
        download.Globus(
            study="/bn/99/97/38/r/",
            dset_dir=self.path,
            collection_id="cc387c09-b0e5-422b-a384-0d96e7ffdc73",
        ).download()

        with success_writer(dl_dir / "success_extract") as already_done:
            if not already_done:
                print(f"Extracting `brennan2019` audio to {dl_dir}/audio...")
                with zipfile.ZipFile(str(dl_dir / "audio.zip"), "r") as zip_:
                    zip_.extractall(str(dl_dir))

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        # S49 excluded: no corresponding timelock-preprocessing .mat file
        all_subjects = {f"S{i:02d}" for i in range(1, 49)}
        # FIXME retrieve bad subjects?
        # S02: bad trl?
        bads = {"S02", "S24", "S26", "S27", "S30", "S32", "S34", "S35", "S36"}
        for subject in sorted(all_subjects - bads):
            yield dict(subject=subject)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        dl_dir = self.path / "download"
        vhdr_file = dl_dir / f"{timeline['subject']}.vhdr"
        raw = mne.io.read_raw_brainvision(vhdr_file, eog=["VEOG"], misc=["Aux5"])
        raw.set_montage(mne.channels.make_standard_montage("easycap-M10"))
        subject_id = timeline["subject"]
        raw.info["subject_info"] = dict(his_id=subject_id, id=int(subject_id[1:]))
        if raw.info["sfreq"] != SFREQ:
            raise RuntimeError(f"Expected sfreq {SFREQ}, got {raw.info['sfreq']}")
        if len(raw.ch_names) != 62:
            raise RuntimeError(f"Expected 62 channels, got {len(raw.ch_names)}")
        return raw

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        dl_dir = self.path / "download"
        file = dl_dir / "timelock-preprocessing" / f"{timeline['subject']}.mat"
        events = self._read_meta(file)
        events = utils.add_sentences(events)
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = pd.DataFrame([{"type": "Eeg", "filepath": info, "start": 0}])
        events.loc[events.type.isin(["Word", "Sentence", "Text"]), "modality"] = "heard"
        return pd.concat([eeg, events])

    def _read_meta(self, fname: Path) -> pd.DataFrame:
        proc = loadmat(
            fname,
            squeeze_me=True,
            chars_as_strings=True,
            struct_as_record=True,
            simplify_cells=True,
        )["proc"]

        meta = proc["trl"]

        if len(meta) != proc["tot_trials"]:
            raise RuntimeError(f"Expected {proc['tot_trials']} trials, got {len(meta)}")
        if proc["tot_chans"] != 61:
            raise RuntimeError(f"Expected 61 channels, got {proc['tot_chans']}")
        bads = list(proc["impedence"]["bads"])  # codespell:ignore impedence
        bads += list(proc["rejections"]["badchans"])

        columns = list(proc["varnames"])
        if len(columns) != meta.shape[1]:
            columns = ["start_sample", "stop_sample", "offset"] + columns
            if len(columns) != meta.shape[1]:
                raise RuntimeError(
                    f"Column count mismatch: {len(columns)} vs {meta.shape[1]}"
                )
        meta = pd.DataFrame(meta, columns=["_" + i for i in columns])
        if len(meta) != 2129:
            raise RuntimeError(
                f"Expected 2129 trials, got {len(meta)}"
            )  # FIXME retrieve subjects who have less trials?

        # Add Brennan's annotations
        dl_dir = self.path / "download"
        story = pd.read_csv(dl_dir / "AliceChapterOne-EEG.csv")
        events = meta.join(story)

        events["type"] = "Word"
        events["condition"] = "sentence"
        events["duration"] = events.offset - events.onset

        rename_map = dict(Word="text", Position="word_id", Sentence="sequence_id")
        events = events.rename(columns=rename_map)
        events["start"] = events["_start_sample"] / SFREQ

        # add audio events
        wav_file = dl_dir / "audio" / "DownTheRabbitHoleFinal_SoundFile%i.wav"
        sounds = []
        for segment, d in events.groupby("Segment"):
            # Some wav files start BEFORE the onset of eeg recording...
            start = d.iloc[0].start - d.iloc[0].onset
            sound = dict(type="Audio", start=start, filepath=str(wav_file) % segment)
            sounds.append(sound)
        events = pd.concat([events, pd.DataFrame(sounds)], ignore_index=True)
        events = events.sort_values("start").reset_index()

        # clean up
        keep = [
            "start",
            "duration",
            "type",
            "word_id",
            "sequence_id",
            "condition",
            "filepath",
            "text",
        ]
        events = events[keep]
        events["language"] = "english"
        events = _extract_sentences(events)

        return events


def _extract_sentences(events: pd.DataFrame) -> pd.DataFrame:
    """
    Extract sentences from a dataframe of events.
    """

    events_out = events.copy()
    is_word = events.type == "Word"
    words = events.loc[is_word]

    for _, d in words.groupby("sequence_id"):
        for uid in d.index:
            events_out.loc[uid, "sentence"] = " ".join(d.text.values)

    return events_out
