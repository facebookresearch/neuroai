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

    def _download(self, overwrite: bool = False) -> None:
        dl_dir = self.path / "download"

        # Deep Blue Data guest collection + record-scoped path for Brennan2019
        # (https://deepblue.lib.umich.edu/data/concern/data_sets/bn999738r).
        download.Globus(
            study="/bn/99/97/38/r/",
            dset_dir=self.path,
            collection_id="cc387c09-b0e5-422b-a384-0d96e7ffdc73",
        ).download()

        # Bumped from "success_extract": the v1 marker predates proc.zip
        # extraction, so existing caches must re-extract to get the .mat files.
        with success_writer(dl_dir / "success_extract_v2") as already_done:
            if not already_done:
                # audio.zip -> audio/*.wav, proc.zip -> timelock-preprocessing/*.mat
                for archive in ("audio.zip", "proc.zip"):
                    print(f"Extracting `brennan2019` {archive} to {dl_dir}...")
                    with zipfile.ZipFile(str(dl_dir / archive), "r") as zip_:
                        members = [
                            m for m in zip_.namelist() if not m.startswith("__MACOSX")
                        ]
                        zip_.extractall(str(dl_dir), members=members)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        # 49 subjects were recorded (S01-S49); 33 remain after the exclusions below.
        all_subjects = {f"S{i:02d}" for i in range(1, 50)}
        # No timelock-preprocessing .mat file ships for these subjects.
        no_proc = {"S28", "S29", "S31", "S33", "S46", "S47", "S49"}
        # Excluded for data quality (bad trials / rejected channels).
        bad_quality = {"S02", "S24", "S26", "S27", "S30", "S32", "S34", "S35", "S36"}
        for subject in sorted(all_subjects - no_proc - bad_quality):
            yield dict(subject=subject)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        dl_dir = self.path / "download"
        vhdr_file = dl_dir / f"{timeline['subject']}.vhdr"
        raw = mne.io.read_raw_brainvision(vhdr_file)
        montage = mne.channels.make_standard_montage("easycap-M10")
        montage_chs = set(montage.ch_names)
        # The auxiliary (non-scalp) channel is named inconsistently across the
        # v2 release -- "Aux5" for some subjects, "AUD" for others -- alongside
        # the "VEOG" EOG channel. Type every channel absent from the montage so
        # set_montage positions only the scalp electrodes.
        non_scalp = {
            ch: ("eog" if "EOG" in ch.upper() else "misc")
            for ch in raw.ch_names
            if ch not in montage_chs
        }
        # on_unit_change="ignore": aux channels go from V (EEG) to NA (misc),
        # which is expected here and would otherwise warn once per subject.
        raw.set_channel_types(non_scalp, on_unit_change="ignore")
        raw.set_montage(montage)
        subject_id = timeline["subject"]
        raw.info["subject_info"] = dict(his_id=subject_id, id=int(subject_id[1:]))
        if raw.info["sfreq"] != SFREQ:
            raise RuntimeError(f"Expected sfreq {SFREQ}, got {raw.info['sfreq']}")
        n_eeg = len(mne.pick_types(raw.info, eeg=True))
        if n_eeg != 60:
            raise RuntimeError(f"Expected 60 EEG channels, got {n_eeg}")
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
