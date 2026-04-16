# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import string
import typing as tp
import warnings
from itertools import product
from pathlib import Path

import mne
import pandas as pd

from neuralfetch import download, utils
from neuralset import utils as nutils
from neuralset.events import etypes, study

mne.set_log_level(False)

CHAPTER_PATHS = [
    "ch1-3.wav",
    "ch4-6.wav",
    "ch7-9.wav",
    "ch10-12.wav",
    "ch13-14.wav",
    "ch15-19.wav",
    "ch20-22.wav",
    "ch23-25.wav",
    "ch26-27.wav",
]

# Handle the particular runs for which we need a higher tolerance
# To handle the default case, we'll use this in the later code
# abs_tol, max_missing = TOL_MISSING_DICT.get((subject, run), (10, 5))
TOL_MISSING_DICT = {
    (9, 6): (30, 5),
    (10, 6): (30, 5),
    (12, 5): (30, 5),
    (13, 3): (5, 30),
    (13, 7): (5, 30),
    (14, 9): (30, 5),
    (21, 6): (200, 5),
    (21, 8): (30, 5),
    (22, 4): (30, 5),
    (33, 2): (40, 40),
    (39, 5): (45, 5),
    (40, 2): (80, 5),
    (41, 1): (40, 5),
    (43, 4): (200, 5),
    (43, 5): (110, 5),
    (44, 9): (30, 5),
    (24, 2): (10, 20),
}

# Handle the runs that are particularly bad:
# - sub21 run 8
# - sub23 run 1-4
BAD_RUNS_LIST_LISTEN = [
    (21, 8),
    (23, 1),
    (23, 2),
    (23, 3),
    (23, 4),
]


def _clean_text(text: str) -> str:
    """Remove punctuation, lowercase, and normalize hyphens/dashes."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.replace("-", " ").replace("–", " ").replace("—", " ")
    return " ".join(text.split())


class _Bel2026PetitBase(study.Study):
    requirements: tp.ClassVar[tuple[str, ...]] = ("openneuro-py",)

    task: tp.ClassVar[str]

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self.infra_timelines.version = "v3"

    def _download(self) -> None:
        raise NotImplementedError(
            "This function is meant to be called on each independent study, Bel2026PetitRead and Bel2026PetitListen,"
            "which have different dataset IDs. See their implementations for details."
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        if not self.path.exists():
            raise RuntimeError(f"Missing folder {self.path}")
        subject_file = self.path / "participants.tsv"
        subjects_df = pd.read_csv(subject_file, sep="\t")
        subject_ids: list[str] = sorted(
            subjects_df.participant_id.apply(lambda x: x.split("-")[1]).values, key=int
        )
        runs = [f"{x:02d}" for x in range(1, 10)]
        sessions = [f"{x:02d}" for x in range(1, 2)]
        for subject, session, run in product(subject_ids, sessions, runs):
            bids_path = self._get_meg_bids_path(subject, session, run)
            if (int(subject), int(run)) in BAD_RUNS_LIST_LISTEN and self.task == "listen":
                continue
            if not Path(str(bids_path)).exists():
                continue
            yield dict(subject=subject, session=session, task=self.task, run=run)

    def _get_meg_bids_path(self, subject: str, session: str, run: str) -> tp.Any:
        from mne_bids import BIDSPath

        return BIDSPath(
            subject=subject,
            session=session,
            task=self.task,
            run=run,
            root=self.path,
            datatype="meg",
            suffix="meg",
            extension=".fif",
        )

    def _make_meg_event(self, timeline: dict[str, tp.Any]) -> etypes.Meg:
        """Build a Meg event; model_post_init reads the raw once for metadata."""
        bids_path = self._get_meg_bids_path(
            timeline["subject"], timeline["session"], timeline["run"]
        )
        meg_filepath = Path(str(bids_path.fpath))
        if not meg_filepath.exists():
            raise FileNotFoundError(
                f"Missing MEG file for {self.__class__.__name__}: {meg_filepath}. "
                "Please run study.download() first."
            )
        # start="auto": auto-resolve from raw.first_samp
        return etypes.Meg(
            timeline="",
            filepath=str(meg_filepath),
            subject=timeline["subject"],
            start="auto",  # type: ignore
        )

    def _get_seq_id_path(self, timeline: dict[str, tp.Any]) -> Path:
        return (
            self.path
            / f"sourcedata/task-{self.task}_run-{timeline['run']}_extra_info.tsv"
        )

    def _get_syntax_path(self, timeline: dict[str, tp.Any]) -> Path:
        return (
            self.path
            / f"sourcedata/stimuli/run{timeline['run']}_v2_0.25_0.5-tokenized.syntax.txt"
        )

    def _add_text_context(
        self,
        timeline: dict[str, tp.Any],
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add the full text of the session to the events dataframe."""
        text_filepath = (
            self.path
            / "sourcedata"
            / "stimuli"
            / "txt_laser"
            / f"run{int(timeline['run'])}.txt"
        )
        words = events.query('type=="Word"').copy()
        orig_text = text_filepath.read_text("utf8")
        start = min(words.start)
        stop = max(words.start + words.duration)
        duration = stop - start

        text = {
            "type": "Text",
            "text": orig_text.replace("\n\n", " ").replace("\n", " "),
            "start": start,
            "duration": duration,
            "language": "french",
        }
        events = pd.concat([pd.DataFrame([text]), events])
        return events


# # # # # actual studies # # # # #


class Bel2026PetitListen(_Bel2026PetitBase):
    """Le Petit Prince MEG (LPP-Listen): MEG responses to spoken French narrative.

    MEG recordings of participants listening to the audiobook of Le Petit Prince
    (Antoine de Saint-Exupéry) across 9 runs. Word onsets are aligned to MEG
    triggers; phoneme-level annotations come from the stimulus timing files.

    Experimental Design:
        - MEG recordings (306-channel Elekta Neuromag TRIUX, 1000 Hz)
        - 58 participants
        - 1 session, 9 runs per session
        - Paradigm: passive listening to spoken narrative
            * Le Petit Prince audiobook in French
            * Word- and phoneme-level onset annotations
            * Stimulus language: French

    Notes:
        - Some runs are excluded due to poor trigger alignment (see BAD_RUNS_LIST_LISTEN).
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("LPP-Listen", "LPP MEG Listen")
    bibtex: tp.ClassVar[str] = """
    @article{bel2026petit,
        title={Le Petit Prince MEG: Magneto-Encephalography Datasets of Reading and Listening tasks in 108 French Participants},
        author={Bel, Corentin and Bonnaire, Julie and King, Jean-R{\\'e}mi and Pallier, Christophe},
        year={2026}
    }

    @misc{bel2026petitlisten,
        author={Bel, Corentin and Bonnaire, Julie and King, Jean-R{\\'e}mi and Pallier, Christophe},
        title={Le Petit Prince MEG Listen},
        year={2025},
        doi={10.18112/openneuro.ds007523.v1.0.0},
        publisher={OpenNeuro},
        url={https://openneuro.org/datasets/ds007523}
    }
    """
    licence: tp.ClassVar[str] = "CC0-1.0"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds007523"
    description: tp.ClassVar[str] = (
        "MEG recordings from 58 participants listening to Le Petit Prince in French."
    )

    task: tp.ClassVar[str] = "listen"

    # 58 subjects × 1 session × 9 runs - bad runs = 516 timelines
    # Events: Meg + Audio + Word + Phoneme + Text + Sentence (via add_sentences)
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=516,
        num_subjects=58,
        num_events_in_query=1775,
        event_types_in_query={"Meg", "Audio", "Word", "Text", "Sentence"},
        data_shape=(306, 623000),
        frequency=1000.0,
    )

    def _download(self) -> None:
        dataset_id = "ds007523"
        client = download.Openneuro(folder="", study=dataset_id, dset_dir=self.path)
        client.download()

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Load events from the events.tsv file for the listen task."""
        tl = timeline
        error_msg_prefix = (
            f"subject {tl['subject']}, session {tl['session']}, run {tl['run']}\n"
        )

        meg = self._make_meg_event(tl)
        raw = meg.read()
        sound_triggers = mne.find_events(raw, stim_channel="STI001", shortest_event=1)

        # Load events from TSV file
        events: list[dict[str, tp.Any]] = []
        tsv = str(meg.filepath).replace("meg.fif", "events.tsv")
        words_df = pd.read_csv(tsv, delimiter="\t")

        for _, row in words_df.iterrows():
            trial_type = row["trial_type"]
            if "BAD_ACQ_SKIP" in str(trial_type):
                continue

            events.append(
                {
                    "condition": "sentence",
                    "type": (
                        trial_type.capitalize() if isinstance(trial_type, str) else "Word"
                    ),
                    "start": row["onset"],
                    "duration": row["duration"],
                    "stop": row["onset"] + row["duration"],
                    "language": "french",
                    "text": row.get("stimulus", row.get("word", "")),
                }
            )

        # extract sound annotation
        sound_triggers = sound_triggers[sound_triggers[:, 2] == 5]
        if len(sound_triggers) != 2:
            warnings.warn(
                f"No sound triggers found for subject {tl['subject']}, run {tl['run']}"
            )
        else:
            start, stop = sound_triggers[:, 0] / raw.info["sfreq"]
            events.append(
                dict(
                    type="Audio",
                    start=start,
                    duration=stop - start,
                    filepath=self.path
                    / "sourcedata/stimuli/audio"
                    / CHAPTER_PATHS[int(tl["run"]) - 1],
                )
            )

        events_df = pd.DataFrame(events)

        # Remove empty words that were included in the metadata files...
        events_df = events_df[events_df["text"] != " "]

        metadata = pd.read_csv(self._get_seq_id_path(tl))
        rows_events, rows_metadata = nutils.match_list(
            [str(word) for word in events_df["text"].values],
            [str(word) for word in metadata["word"].values],
        )

        assert len(rows_events) / len(events_df) > 0.95, (
            error_msg_prefix
            + f"only {len(rows_events) / len(events_df)} of the words were found in the metadata"
        )
        events_idx, metadata_idx = (
            events_df.index[rows_events],
            metadata.index[rows_metadata],
        )

        # Adding the information about sequence_id and n_closing
        events_df["word"] = events_df["text"]
        for col in ["sequence_id", "n_closing", "is_last_word", "pos"]:
            events_df.loc[events_idx, col] = metadata.loc[metadata_idx, col]

        # add train/test/val splits
        events_df = utils.add_sentences(events_df)

        words = events_df.loc[events_df.type == "Word"]

        # Get the word triggers from STI008, as a step so we can get the offset
        word_triggers = mne.find_stim_steps(raw, stim_channel="STI008")
        word_triggers = word_triggers[word_triggers[:, 2] == 0]

        abs_tol, max_missing = TOL_MISSING_DICT.get(
            (int(tl["subject"]), int(tl["run"])), (10, 5)
        )
        i, j = nutils.approx_match_samples(
            (words.start * 1000).tolist(),
            word_triggers[:, 0],
            abs_tol=abs_tol,
            max_missing=max_missing,
        )

        words = words.iloc[i, :]

        events_df.loc[:, "unaligned_start"] = events_df.loc[:, "start"]
        events_df.loc[words.index, "start"] = word_triggers[j, 0] / raw.info["sfreq"]

        events_df = pd.concat([pd.DataFrame([meg.to_dict()]), events_df])
        events_df = self._add_text_context(tl, events_df)

        events_df.loc[events_df.type.isin(["Word", "Sentence", "Text"]), "modality"] = (
            "heard"
        )
        return events_df.sort_values(by="start").reset_index(drop=True)


class Bel2026PetitRead(_Bel2026PetitBase):
    """Le Petit Prince MEG (LPP-Read): MEG responses to reading French narrative.

    MEG recordings of participants reading Le Petit Prince
    (Antoine de Saint-Exupéry) across 9 runs via rapid serial visual presentation
    (RSVP). Word onsets are aligned to MEG triggers via word-length matching.

    Experimental Design:
        - MEG recordings (306-channel Elekta Neuromag TRIUX, 1000 Hz)
        - 50 participants
        - 1 session, 9 runs per session
        - Paradigm: RSVP reading of narrative text (225 ms/word, 50 ms blank)
            * Le Petit Prince in French
            * Word-level onset annotations aligned via trigger matching
            * Stimulus language: French

    Notes:
        - Words that cannot be aligned to triggers are kept as MissedWord events.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("LPP-Read", "LPP MEG Read")
    bibtex: tp.ClassVar[str] = """
    @article{bel2026petit,
        title={Le Petit Prince MEG: Magneto-Encephalography Datasets of Reading and Listening tasks in 108 French Participants},
        author={Bel, Corentin and Bonnaire, Julie and King, Jean-R{\\'e}mi and Pallier, Christophe},
        year={2026}
    }

    @misc{bel2026petitread,
        author={Bel, Corentin and Bonnaire, Julie and King, Jean-R{\\'e}mi and Pallier, Christophe},
        title={Le Petit Prince MEG Read},
        year={2025},
        doi={10.18112/openneuro.ds007524.v1.0.1},
        publisher={OpenNeuro},
        url={https://openneuro.org/datasets/ds007524}
    }
    """
    licence: tp.ClassVar[str] = "CC0-1.0"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds007524"
    description: tp.ClassVar[str] = (
        "MEG recordings from 50 participants reading Le Petit Prince in French."
    )

    task: tp.ClassVar[str] = "read"

    # ~50 subjects × 1 session × 9 runs = ~450 timelines
    # Events: Meg + Word + MissedWord + Text + Sentence (via add_sentences)
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=450,
        num_subjects=50,
        num_events_in_query=1589,
        event_types_in_query={"Meg", "Word", "MissedWord", "Text", "Sentence"},
        data_shape=(306, 512_000),
        frequency=1000.0,
    )

    def _download(self) -> None:
        dataset_id = "ds007524"
        client = download.Openneuro(folder="", study=dataset_id, dset_dir=self.path)
        client.download()

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Load events from the events.tsv file for the read task."""
        tl = timeline
        error_msg_prefix = (
            f"subject {tl['subject']}, session {tl['session']}, run {tl['run']}\n"
        )

        meg = self._make_meg_event(tl)
        raw = meg.read()

        events_df = self._parse_events_tsv(meg)
        events_df = self._align_with_triggers(events_df, raw, error_msg_prefix)
        events_df = self._add_metadata(tl, events_df, error_msg_prefix)
        events_df = utils.add_sentences(events_df)
        events_df = pd.concat([pd.DataFrame([meg.to_dict()]), events_df])
        events_df = self._add_text_context(tl, events_df)

        events_df.loc[events_df.type.isin(["Word", "Sentence", "Text"]), "modality"] = (
            "read"
        )

        return events_df.sort_values(by="start").reset_index(drop=True)

    def _parse_events_tsv(self, meg: etypes.Meg) -> pd.DataFrame:
        """Parse events from the BIDS events.tsv file."""
        tsv = str(meg.filepath).replace("meg.fif", "events.tsv")
        words_df = pd.read_csv(tsv, delimiter="\t")

        events = []
        for _, row in words_df.iterrows():
            if "BAD_ACQ_SKIP" in str(row["trial_type"]):
                continue

            events.append(
                {
                    "condition": "sentence",
                    "type": (
                        row["trial_type"].capitalize()
                        if isinstance(row["trial_type"], str)
                        else "Word"
                    ),
                    "start": row["onset"],
                    "duration": row["duration"],
                    "stop": row["onset"] + row["duration"],
                    "language": "french",
                    "text": row.get("stimulus", row.get("word", "")),
                }
            )

        return pd.DataFrame(events)

    def _align_with_triggers(
        self, events_df: pd.DataFrame, raw: mne.io.RawArray, error_msg_prefix: str
    ) -> pd.DataFrame:
        """Align word events with MEG triggers based on word length."""
        word_triggers = mne.find_events(raw, stim_channel="STI101", shortest_event=1)

        # Normalize trigger values if needed
        if word_triggers[:, 2].max() > 2048:
            word_triggers[:, 2] -= 2048

        # Match triggers to words by length
        words = events_df.loc[events_df.type == "Word"].copy()
        words["wlength"] = words.text.apply(len)

        i, j = nutils.match_list(word_triggers[:, 2], words.wlength)
        words_recovered = len(j) / len(words)

        assert words_recovered > 0.9, (
            error_msg_prefix + f"only {words_recovered:.2%} of words found in triggers"
        )

        # Update aligned words and track missed words
        matched_indices = words.iloc[j].index
        missed_words = words[~words.index.isin(matched_indices)].copy()
        missed_words["type"] = "MissedWord"

        events_df["unaligned_start"] = events_df["start"]
        events_df.loc[matched_indices, "start"] = word_triggers[i, 0] / raw.info["sfreq"]

        # Remove unmatched words and add back as MissedWord
        events_df = events_df.loc[
            events_df.index.isin(matched_indices) | (events_df.type != "Word")
        ]
        events_df = pd.concat([events_df, missed_words], ignore_index=True)

        return events_df

    def _add_metadata(
        self, timeline: dict[str, tp.Any], events_df: pd.DataFrame, error_msg_prefix: str
    ) -> pd.DataFrame:
        """Add syntactic metadata by matching cleaned text."""
        metadata = pd.read_csv(self._get_seq_id_path(timeline))

        # Clean text for matching
        events_df["clean_text"] = events_df["text"].apply(_clean_text)
        metadata["word"] = metadata["word"].apply(_clean_text)

        # Match events to metadata
        rows_events, rows_metadata = nutils.match_list(
            events_df["clean_text"].astype(str).tolist(),
            metadata["word"].astype(str).tolist(),
        )

        match_rate = len(rows_events) / len(events_df)
        assert match_rate > 0.8, (
            error_msg_prefix + f"only {match_rate:.2%} of events matched with metadata"
        )

        # Add metadata columns
        events_df["word"] = events_df["text"]
        events_idx = events_df.index[rows_events]
        metadata_idx = metadata.index[rows_metadata]

        for col in ["sequence_id"]:
            events_df.loc[events_idx, col] = metadata.loc[metadata_idx, col].values

        return events_df


# # # # # Sample datasets # # # # #


class Bel2026PetitListenSample(Bel2026PetitListen):
    """Lightweight Bel2026PetitListen with subject 26, run 01 only.

    Single subject, single run, designed for quick validation
    and testing of the Bel2026Petit MEG study infrastructure.

    Events: Meg + Audio + Word + Text + Sentence
    Frequency: 1000 Hz
    MEG sensors: 306 channels

    OpenNeuro dataset: https://openneuro.org/datasets/ds007523
    """

    description: tp.ClassVar[str] = (
        "Sample Bel2026PetitListen: MEG listening to Le Petit Prince (subject 26, run 01)"
    )

    bibtex: tp.ClassVar[str] = """
    @misc{bel2026petitlisten,
        author={Bel, Corentin and Bonnaire, Julie and King, Jean-R{\\'e}mi and Pallier, Christophe},
        title={Le Petit Prince MEG Listen},
        year={2025},
        doi={10.18112/openneuro.ds007523.v1.0.0},
        publisher={OpenNeuro},
        url={https://openneuro.org/datasets/ds007523}
    }
    """
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds007523"
    licence: tp.ClassVar[str] = "CC0-1.0"

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1,
        num_subjects=1,
        num_events_in_query=1775,
        event_types_in_query={"Audio", "Meg", "Sentence", "Text", "Word"},
        data_shape=(306, 617000),
        frequency=1000.0,
    )

    def _download(self) -> None:
        # ds007523 is the full dataset; here we download only subject 26
        dataset_id = "ds007523"
        client = download.Openneuro(
            folder="",
            study=dataset_id,
            dset_dir=self.path,
            include=[
                # MEG data for subject 26, session 01, listen task, run 01
                "sub-26/ses-01/meg/sub-26_ses-01_task-listen_run-01*",
                # Metadata files for listen task run 01
                "sourcedata/task-listen_run-01_extra_info.tsv",
                # Text stimuli for run 1
                "sourcedata/stimuli/txt_laser/run1.txt",
                # Audio for run 1 (ch1-3.wav from CHAPTER_PATHS[0])
                "sourcedata/stimuli/audio/ch1-3.wav",
                # Required for iter_timelines and dataset metadata
                "participants.tsv",
                "dataset_description.json",
            ],
        )
        client.download()

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Yield only subject 26, run 01."""
        yield dict(subject="26", session="01", task=self.task, run="01")


class Bel2026PetitReadSample(Bel2026PetitRead):
    """Lightweight Bel2026PetitRead with subject 26, run 01 only.

    Single subject, single run, designed for quick validation
    and testing of the Bel2026Petit MEG study infrastructure.

    NOTE: This mini version doesn't include the extra metadata (sequence_id)
    that the full version has, as the sourcedata files are not available
    in the OpenNeuro mini dataset.

    Events: Meg + Word + MissedWord + Text + Sentence
    Frequency: 1000 Hz
    MEG sensors: 306 channels

    OpenNeuro dataset: https://openneuro.org/datasets/ds007524
    """

    description: tp.ClassVar[str] = (
        "Sample Bel2026PetitRead: MEG reading Le Petit Prince (subject 26, run 01)"
    )

    bibtex: tp.ClassVar[str] = """
    @misc{bel2026petitread,
        author={Bel, Corentin and Bonnaire, Julie and King, Jean-R{\\'e}mi and Pallier, Christophe},
        title={Le Petit Prince MEG Read},
        year={2025},
        doi={10.18112/openneuro.ds007524.v1.0.1},
        publisher={OpenNeuro},
        url={https://openneuro.org/datasets/ds007524}
    }
    """
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds007524"
    licence: tp.ClassVar[str] = "CC0-1.0"

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1,
        num_subjects=1,
        num_events_in_query=1600,
        event_types_in_query={"Meg", "MissedWord", "Sentence", "Text", "Word"},
        data_shape=(306, 475000),
        frequency=1000.0,
    )

    def _download(self) -> None:
        # ds007524 is the full dataset; here we download only subject 26 run 01
        dataset_id = "ds007524"
        client = download.Openneuro(
            folder="",
            study=dataset_id,
            dset_dir=self.path,
            include=[
                # MEG data for subject 26, session 01, read task, run 01
                "sub-26/ses-01/meg/sub-26_ses-01_task-read_run-01*",
                # Calibration files needed for MEG
                "sub-26/ses-01/meg/sub-26_ses-01_acq-*",
                "sub-26/ses-01/meg/sub-26_ses-01_coordsystem.json",
                # Required for iter_timelines and dataset metadata
                "participants.tsv",
                "dataset_description.json",
            ],
        )
        client.download()

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Yield only subject 26, run 01."""
        yield dict(subject="26", session="01", task=self.task, run="01")

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Load events - simplified version without metadata files."""
        tl = timeline
        error_msg_prefix = (
            f"subject {tl['subject']}, session {tl['session']}, run {tl['run']}\n"
        )

        meg = self._make_meg_event(tl)
        raw = meg.read()

        events_df = self._parse_events_tsv(meg)
        events_df = self._align_with_triggers(events_df, raw, error_msg_prefix)

        # Create simple sequence_id based on sentence-ending punctuation
        events_df = self._add_simple_sequence_ids(events_df)

        events_df = utils.add_sentences(events_df)
        events_df = self._add_meg_and_text(tl, meg, raw, events_df)

        events_df.loc[events_df.type.isin(["Word", "Sentence", "Text"]), "modality"] = (
            "read"
        )

        return events_df.sort_values(by="start").reset_index(drop=True)

    def _add_simple_sequence_ids(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Add sequence_id to words based on sentence-ending punctuation."""
        # Create sequence_id based on punctuation
        words = events_df.query('type=="Word"').copy()
        sequence_id = 0
        sequence_ids = []
        sentence_enders = (".", "!", "?", ":", ";")

        for text in words["text"]:
            sequence_ids.append(sequence_id)
            # Check if word ends with sentence-ending punctuation
            if any(text.rstrip().endswith(p) for p in sentence_enders):
                sequence_id += 1

        events_df.loc[words.index, "sequence_id"] = sequence_ids
        return events_df

    def _add_meg_and_text(
        self,
        timeline: dict[str, tp.Any],  # noqa: ARG002
        meg: etypes.Meg,
        raw: mne.io.RawArray,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add MEG and Text events without relying on sourcedata txt files."""
        freq = raw.info["sfreq"]
        meg_text: list[dict[str, tp.Any]] = [
            {
                "filepath": meg.filepath,
                "type": "Meg",
                "start": raw.first_samp / freq,
                "frequency": freq,
            }
        ]

        # Reconstruct text from Word events
        words = events.query('type=="Word"').copy()
        text = " ".join(words["text"].tolist())
        start = min(words.start)
        stop = max(words.start + words.duration)
        duration = stop - start

        meg_text.append(
            {
                "type": "Text",
                "text": text,
                "start": start,
                "duration": duration,
                "language": "french",
            }
        )
        events = pd.concat([pd.DataFrame(meg_text), events])
        return events
