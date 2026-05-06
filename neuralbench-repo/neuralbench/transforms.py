# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import pandas as pd
import pydantic
from sklearn.model_selection import train_test_split

import neuralset as ns
from neuralset.events import transforms as _transf


class TextPreprocessor(_transf.EventsTransform):
    """Clean and filter text-related events.

    The following operations are applied to the events:

    - Keep only events with duration >= 0
    - Keep only neuro events, Audio, or valid Word events (with text as string)
    - Clean 'text' column by removing special characters and lowercasing
    - Drop empty or blank text entries
    - For Nieuwland2018, group similar sentences together to avoid leakage
      (each sentence has two very similar versions)
    """

    neuro_event_type: str = "Eeg"

    @staticmethod
    def clean_text(x: str) -> str:
        """Remove special characters and lowercase the text."""
        return "".join(e for e in x if e.isalnum() or e in [" ", "-", "'"]).lower()

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        # Keep only valid events based on type and text validity
        is_valid_word = events.text.apply(lambda x: isinstance(x, str))
        keep_event = (events.duration >= 0) & (
            events.type.isin([self.neuro_event_type, "Audio"]) | is_valid_word
        )
        events = events.loc[keep_event].copy()

        # Clean the 'text' column for Word events
        word_mask = events.type == "Word"
        events.loc[word_mask, "text"] = events.loc[word_mask, "text"].apply(
            self.clean_text
        )

        # Drop blank or empty text entries
        events = events.loc[~events.text.isin(["", " "])].copy()

        if events.study.iloc[0] == "Nieuwland2018":
            events["sequence_id"] = events.groupby("sentence").ngroup() // 2

        return events


SklearnSplit = _transf.SklearnSplit


class SimilaritySplit(_transf.EventsTransform):
    """Perform train/val/test split based on similarity of sentence events.

    Depending on the type of stimulus event that is expected, the behavior is as follows:

    - For Audio events, propagate sentence mapping to Word events, then chunk Audio events based on
      Word events.
    - For Keystroke events, propagate sentence mapping to Keystroke events.
    - For Sentence or Word events, directly apply the similarity-based split.

    Parameters
    ----------
    use_sklearn_split :
        If True, use sklearn's `train_test_split` after computing clusters, rather than using
        `SimilaritySplitter`'s deterministic cluster assignment.
        NOTE: `valid_random_state` and `test_random_state` are ignored unless `use_sklearn_split`
        is True.
    """

    stim_event_type: tp.Literal["Sentence", "Word", "Audio", "Keystroke"]
    valid_split_ratio: float = pydantic.Field(
        0.2, strict=True, ge=0.0, le=1.0, allow_inf_nan=False
    )
    test_split_ratio: float = pydantic.Field(
        0.2, strict=True, ge=0.0, le=1.0, allow_inf_nan=False
    )
    valid_random_state: int = 33
    test_random_state: int = 33
    threshold: float = 0.2
    use_sklearn_split: bool = False

    def _split(self, events: pd.DataFrame, splitter: _transf.SimilaritySplitter):
        if self.use_sklearn_split:
            splitted_events = splitter.compute_clusters(events)
            sklearn_splitter = _transf.SklearnSplit(
                split_by="cluster_id",
                valid_split_ratio=self.valid_split_ratio,
                test_split_ratio=self.test_split_ratio,
                valid_random_state=self.valid_random_state,
                test_random_state=self.test_random_state,
            )
            splitted_events = sklearn_splitter(splitted_events)
            events.loc[splitted_events.index, "split"] = splitted_events["split"]
        else:
            # Use SimilaritySplitter's deterministic cluster assignment
            events = splitter(events)

        return events

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        ratios = {
            "train": 1.0 - (self.valid_split_ratio + self.test_split_ratio),
            "val": self.valid_split_ratio,
            "test": self.test_split_ratio,
        }

        # Initialize splitter over Sentence events
        splitter = _transf.SimilaritySplitter(
            extractor=ns.extractors.text.TfidfEmbedding(),  # TODO: parametrize
            ratios=ratios,
            threshold=self.threshold,
        )

        if self.stim_event_type == "Audio":  # Designed for Accou2023
            events = self._split(events, splitter)
            sentence_split_mapping = (
                events.loc[events.type == "Sentence", ["text", "split"]]
                .set_index("text")["split"]
                .to_dict()
            )
            events.loc[events.type == "Word", "split"] = events.loc[
                events.type == "Word", "sentence"
            ].map(sentence_split_mapping)

            # Chunk Audio events based on Word segments
            events = _transf.chunk_events(
                events,
                event_type_to_chunk="Audio",
                event_type_to_use="Word",
                min_duration=3.0,
                max_duration=120.0,
            )

        elif self.stim_event_type == "Keystroke":  # Designed for Levy2025Brain
            # For the Typing dataset, we need to transfer the clusters from the sentences to the
            # keystroke events
            if "sentence" not in events.columns:
                raise ValueError(
                    "Expected 'sentence' column in Keystroke events for similarity-based splitting."
                )
            unique_sentences = (
                events.loc[events.type == "Keystroke", "sentence"].dropna().unique()
            )

            # Fake dataset creation to be able to populate the keystroke events later
            sentence_df = pd.DataFrame(
                {
                    "text": unique_sentences,
                    "type": "Sentence",
                    "start": 0.0,
                    "duration": 0.0,
                    "timeline": "dummy_tl",
                }
            )

            # Run similarity splitter on these pseudo-Sentence rows
            sentence_df = self._split(sentence_df, splitter)

            # Mapping sentence → split
            sentence_split_map = sentence_df.set_index("text")["split"].to_dict()

            events.loc[events.type == "Keystroke", "split"] = events.loc[
                events.type == "Keystroke", "sentence"
            ].map(sentence_split_map)

        elif self.stim_event_type in ["Sentence", "Word"]:
            events = self._split(events, splitter)

        return events


class PredefinedSplit(_transf.EventsTransform):
    """Assign train/test labels based on a predefined split, and optionally split train into
    validation as well.

    Parameters
    ----------
    event_type : str | None
        If provided, only split events of this type.
    test_split_query : str | None
        If provided, query used to create a test split.
    col_name : str
        Column name to use for the created split.
    valid_split_ratio : float
        Ratio of the **training set** to use for validation. CAUTION: This is unlike the other
        splitter (e.g. SklearnSplit) where the validation split is a ratio of the entire dataset.
    test_random_state : int | None
        Unused - for compatibility with SklearnSplit.
    """

    event_type: str | None = None
    test_split_query: str | None
    col_name: str = "split"

    valid_split_by: str | None = "timeline"
    valid_split_ratio: float = 0.2
    valid_random_state: int = 33
    test_random_state: int | None = None  # Unused - for compatibility with SklearnSplit

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        if self.event_type is not None:
            unused_events = events[events.type != self.event_type]
            events = events[events.type == self.event_type]

        if self.test_split_query is not None:
            test_inds = events.query(self.test_split_query).index
            events[self.col_name] = "train"
            events.loc[test_inds, self.col_name] = "test"

        if self.col_name not in events or "test" not in events[self.col_name].values:
            raise ValueError("Predefined train/test split required.")

        # Optionally assign random samples of train to validation split
        if self.valid_split_by is not None:
            train_events = events[events[self.col_name] == "train"]
            if self.valid_split_by == "_index":
                if "_index" not in train_events.columns:
                    train_events = train_events.copy()
                    train_events["_index"] = range(len(train_events))
                    events.loc[train_events.index, "_index"] = train_events["_index"]
            train_groups = np.asarray(train_events[self.valid_split_by].dropna().unique())
            train, valid = train_test_split(
                train_groups,
                test_size=self.valid_split_ratio,
                random_state=self.valid_random_state,
            )
            if len(valid) == 0:
                raise ValueError(
                    "Empty validation set, try increasing `valid_split_ratio`."
                )
            split_mapping = {
                **{k: "train" for k in train},
                **{k: "val" for k in valid},
            }
            train_events.loc[:, self.col_name] = train_events[
                self.valid_split_by
            ].replace(split_mapping)
            events.loc[train_events.index, self.col_name] = train_events[self.col_name]

        if self.event_type is not None:
            events = pd.concat([unused_events, events], ignore_index=True, axis=0)

        return events


class CropSleepRecordings(_transf.EventsTransform):
    """Keep up to max_wake_duration_min mins of wake (W) time before and after the first and last
    sleep events.
    """

    max_wake_duration_min: float = 30.0

    def crop_first_last_wake(self, evs: pd.DataFrame) -> pd.DataFrame:
        non_wake_inds = evs[evs.stage != "W"].index
        if len(non_wake_inds) == 0:
            return evs
        if non_wake_inds[0] > evs.index[0]:
            ind = np.where(evs.index == non_wake_inds[0])[0][0] - 1
            ind = evs.index[ind]
            start, duration = evs.loc[ind, ["start", "duration"]]
            assert isinstance(start, float)
            assert isinstance(duration, float)

            new_duration = min(self.max_wake_duration_min * 60.0, duration)
            evs.loc[ind, "duration"] = new_duration
            evs.loc[ind, "start"] = start + duration - new_duration

        if non_wake_inds[-1] < evs.index[-1]:
            ind = np.where(evs.index == non_wake_inds[-1])[0][0] + 1
            ind = evs.index[ind]
            start, duration = evs.loc[ind, ["start", "duration"]]
            assert isinstance(start, float)
            assert isinstance(duration, float)

            new_duration = min(self.max_wake_duration_min * 60.0, duration)
            evs.loc[ind, "duration"] = new_duration
            evs.loc[ind, "stop"] = start + new_duration

        return evs

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        sleep_events = (
            events[events.type == "SleepStage"]
            .groupby("timeline")
            .apply(self.crop_first_last_wake, include_groups=False)  # type: ignore
            .droplevel(1)
            .reset_index()
        )
        events = pd.concat(
            [
                events[events.type != "SleepStage"],
                sleep_events,
            ],
            ignore_index=True,
            axis=0,
        ).sort_values(by=["timeline", "start"])

        return events


class CropTimelines(_transf.EventsTransform):
    """Crop neuro timelines.

    Parameters
    ----------
    event_type : str | None
        If provided, only crop events of this type.
    start_offset_s : float | None
        If provided, crop this offset from the start of the event.
    max_duration_s : float | None
        If provided, cap the event at this duration.
    """

    event_type: str | None = None
    start_offset_s: float | None = None
    max_duration_s: float | None = None

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()
        if self.event_type is not None:
            sel_events = events[events.type == self.event_type]
        else:
            sel_events = events

        if self.start_offset_s is not None:
            sel_events.loc[:, "start"] += self.start_offset_s
            sel_events.loc[:, "duration"] -= self.start_offset_s
        if self.max_duration_s is not None:
            sel_events.loc[:, "duration"] = np.minimum(
                sel_events.duration.to_numpy(), self.max_duration_s
            )
        events.update(sel_events)

        if (sel_events["duration"] <= 0).any():
            raise ValueError(
                "Cropping timelines resulted in non-positive duration for some events. "
                "Check that `start_offset_s` is not larger than the shortest event duration."
            )

        return events


class AddDefaultEvents(_transf.EventsTransform):
    """Add default events to a timeline to fill out its duration.

    E.g., useful for epilepsy recordings where seizures are sparse.

    Parameters
    ----------
    target_event_type : str
        Event type to fill out.
    default_event_type : str
        Event type to use for the added events.
    default_event_fields : dict[str, Any]
        Additional fields to add to the created events, as a dictionary of (field_name, value).
    """

    target_event_type: str
    default_event_type: str
    default_event_fields: dict[str, tp.Any] = {}

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        all_events = []
        for tl, group in events.groupby("timeline"):
            tl_start, tl_stop = group.start.min(), group.stop.max()
            target_intervals = group.loc[
                group.type == self.target_event_type, ["start", "stop"]
            ]

            new_intervals = []
            start = tl_start
            for _, interval in target_intervals.iterrows():
                if interval.start > start:
                    new_intervals.append((start, interval.start))
                start = interval.stop
            if start < tl_stop:
                new_intervals.append((start, tl_stop))

            new_events = pd.DataFrame(new_intervals, columns=["start", "stop"])
            new_events["type"] = self.default_event_type
            new_events["timeline"] = tl
            new_events["duration"] = new_events.stop - new_events.start
            for field, value in self.default_event_fields.items():
                new_events[field] = value

            all_events.extend([group, new_events])

        events = pd.concat(all_events, ignore_index=True, axis=0)
        return events


class OffsetEvents(_transf.EventsTransform):
    """Offset selected events by specified amounts.

    Parameters
    ----------
    query: str
        Query to select events to offset.
    start_offset: float | None = None
        Offset to apply starting from the start of the event.
    end_offset: float | None = None
        Offset to apply starting from the end of the event.
    start_offset_from_end: float | None = None
        Offset to apply starting from the end of the event.
    end_offset_from_start: float | None = None
        Offset to apply starting from the start of the event.
    """

    query: str
    start_offset: float | None = None
    end_offset: float | None = None
    start_offset_from_end: float | None = None
    end_offset_from_start: float | None = None

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)

        if self.start_offset is not None and self.start_offset_from_end is not None:
            raise ValueError("Cannot specify both start_offset and start_offset_from_end")
        if self.end_offset is not None and self.end_offset_from_start is not None:
            raise ValueError("Cannot specify both end_offset and end_offset_from_start")

        if (
            self.start_offset is None
            and self.start_offset_from_end is None
            and self.end_offset is None
            and self.end_offset_from_start is None
        ):
            msg = "At least one of start_offset, start_offset_from_end, end_offset, or "
            msg += "end_offset_from_start must be specified"
            raise ValueError(msg)

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        for _, group in events.groupby("timeline"):
            min_tl_start, max_tl_stop = group.start.min(), group.stop.max()
            sel_events = group.query(self.query)

            if self.start_offset is not None:
                new_start = sel_events.start + self.start_offset
            elif self.start_offset_from_end is not None:
                new_start = sel_events.stop + self.start_offset_from_end
            else:
                new_start = sel_events.start
            events.loc[sel_events.index, "start"] = np.maximum(min_tl_start, new_start)

            if self.end_offset is not None:
                new_stop = sel_events.stop + self.end_offset
            elif self.end_offset_from_start is not None:
                new_stop = sel_events.start + self.end_offset_from_start
            else:
                new_stop = sel_events.stop
            events.loc[sel_events.index, "stop"] = np.minimum(max_tl_stop, new_stop)

            events.loc[sel_events.index, "duration"] = (
                events.loc[sel_events.index, "stop"]
                - events.loc[sel_events.index, "start"]
            )

        if any(events.duration <= 0):
            msg = (
                "Offsetting events resulted in negative or zero duration for some events."
            )
            raise ValueError(msg)

        return events


class ShuffleTrainingLabels(_transf.EventsTransform):
    """Randomly permute the label field of training events (diagnostic ablation).

    Only rows where ``split == "train"`` are modified; validation and test rows
    are left untouched.  Values of the given ``event_field`` are reshuffled
    uniformly at random among training rows (of the given ``event_types``),
    preserving the overall class marginal on the training set but destroying
    the epoch-to-label correspondence.

    This is intended as a sanity check for pre-training leakage in foundation
    models: if a model fine-tuned on permuted training labels still performs
    above chance on the (intact) test labels, the encoder has memorised
    dataset-level structure during pre-training.  Clean fine-tuning lift on
    genuine features should collapse to chance under this ablation.

    Must run **after** a split transform that populates the ``split`` column.

    Parameters
    ----------
    event_field : str
        Name of the column to permute (e.g. ``"code"`` for LabelEncoder).
    event_types : str or list of str or None
        If provided, only permute values among events whose ``type`` is in this
        list.  ``None`` permutes across all training rows regardless of type.
    random_state : int
        Seed for reproducible permutation.
    """

    event_field: str
    event_types: str | list[str] | None = None
    random_state: int = 33

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if "split" not in events.columns:
            raise ValueError(
                "ShuffleTrainingLabels requires a 'split' column. Run a split "
                "transform (e.g. SklearnSplit, PredefinedSplit) first."
            )
        if self.event_field not in events.columns:
            raise ValueError(
                f"event_field {self.event_field!r} not found in events columns."
            )
        events = events.copy()
        is_train = events["split"] == "train"
        if self.event_types is not None:
            types = (
                [self.event_types]
                if isinstance(self.event_types, str)
                else list(self.event_types)
            )
            mask = is_train & events["type"].isin(types)
        else:
            mask = is_train
        n = int(mask.sum())
        if n <= 1:
            return events
        rng = np.random.default_rng(self.random_state)
        values = events.loc[mask, self.event_field].to_numpy().copy()
        rng.shuffle(values)
        events.loc[mask, self.event_field] = values
        return events
