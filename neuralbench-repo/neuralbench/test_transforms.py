# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import pandas as pd
import pytest

import neuralset as ns
from neuralset.events import standardize_events

from .main import Data
from .transforms import (
    AddDefaultEvents,
    CropSleepRecordings,
    CropTimelines,
    OffsetEvents,
    PredefinedSplit,
    ShuffleTrainingLabels,
    SimilaritySplit,
    TextPreprocessor,
)


@pytest.fixture
def events():
    return standardize_events(
        pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    start=0.0,
                    duration=60.0,
                    subject="1",
                    frequency=100.0,
                    filepath=":.fif",
                    study="Test2025",
                    timeline="blu",
                ),
                dict(type="Word", start=42.0, duration=1, text="table", timeline="blu"),
                dict(type="Phoneme", start=42.0, duration=1, text="t", timeline="blu"),
                dict(
                    type="Eeg",
                    start=43.0,
                    duration=2.0,
                    subject="1",
                    frequency=100.0,
                    filepath=":.fif",
                    study="Test2025",
                    timeline="blublu",
                ),
                dict(type="Phoneme", start=43.0, duration=1, text="t", timeline="blublu"),
                dict(type="Phoneme", start=44.0, duration=1, text="u", timeline="blublu"),
                dict(
                    type="Eeg",
                    start=44.5,
                    duration=10.0,
                    subject="1",
                    frequency=100.0,
                    filepath=":.fif",
                    study="Test2025",
                    timeline="blublublu",
                ),
                dict(
                    type="Phoneme",
                    start=45.0,
                    duration=1,
                    text="v",
                    timeline="blublublu",
                ),
            ]
        )
    )


@pytest.fixture
def text_events():
    return pd.DataFrame(
        [
            dict(
                type="Word",
                text="Hello!!",
                start=1.0,
                duration=-0.5,
                study="Test2025",
                timeline="blu",
            ),
            dict(
                type="Word",
                text="Hi!!",
                start=1.1,
                duration=0.5,
                study="Test2025",
                timeline="blu",
            ),
            dict(
                type="Word",
                text="This is... fine.",
                start=5.0,
                duration=1.0,
                study="Test2025",
                timeline="blu",
            ),
            dict(
                type="Eeg",
                text=None,
                start=0.0,
                duration=60.0,
                filepath="test.fif",
                study="Test2025",
                timeline="blu",
            ),
            dict(
                type="Audio",
                text="BEEP",
                start=0.5,
                duration=0.3,
                study="Test2025",
                timeline="blu",
            ),
            dict(
                type="Word",
                text=" ",
                start=10.0,
                duration=0.6,
                study="Test2025",
                timeline="blu",
            ),
            dict(
                type="Word",
                text="not_valid",
                start=12.0,
                duration=-1.0,
                study="Test2025",
                timeline="blu",
            ),
        ]
    )


@pytest.fixture
def sentence_events():
    return standardize_events(
        pd.DataFrame(
            [
                dict(
                    type="Sentence",
                    start=0.0,
                    duration=5.0,
                    study="Test2025",
                    text="The quick brown fox jumps over the lazy dog",
                    timeline="blu",
                ),
                dict(
                    type="Word",
                    start=0.0,
                    duration=1.0,
                    study="Test2025",
                    text="quick",
                    timeline="blu",
                ),
                dict(
                    type="Word",
                    start=1.0,
                    duration=1.0,
                    study="Test2025",
                    text="brown",
                    timeline="blu",
                ),
                dict(
                    type="Sentence",
                    start=0.0,
                    duration=10.0,
                    study="Test2025",
                    text="The quick brown fox jumps over the lazy dog",
                    timeline="blu",
                ),
                dict(
                    type="Sentence",
                    start=0.0,
                    duration=15.0,
                    study="Test2025",
                    text="The birch canoe slid on the smooth planks",
                    timeline="blu",
                ),
                dict(
                    type="Sentence",
                    start=0.0,
                    duration=20.0,
                    study="Test2025",
                    text="The birch canoe slid on the smooth planks",
                    timeline="blu",
                ),
                dict(
                    type="Sentence",
                    start=0.0,
                    duration=25.0,
                    study="Test2025",
                    text="Glue the sheet to the dark blue background",
                    timeline="blu",
                ),
                dict(
                    type="Sentence",
                    start=0.0,
                    duration=30.0,
                    study="Test2025",
                    text="Glue the sheet to the dark blue background",
                    timeline="blu",
                ),
            ]
        )
    )


@pytest.fixture
def sleep_events():
    return standardize_events(
        pd.DataFrame(
            [
                dict(
                    type="SleepStage",
                    start=0.0,
                    duration=3600.0,
                    stage="W",
                    timeline="blu",
                ),
                dict(
                    type="SleepStage",
                    start=3600.0,
                    duration=30.0,
                    stage="N1",
                    timeline="blu",
                ),
                dict(
                    type="SleepStage",
                    start=3630.0,
                    duration=30.0,
                    stage="W",
                    timeline="blu",
                ),
                dict(
                    type="SleepStage",
                    start=3660.0,
                    duration=30.0,
                    stage="R",
                    timeline="blu",
                ),
                dict(
                    type="SleepStage",
                    start=3690.0,
                    duration=3600.0,
                    stage="W",
                    timeline="blu",
                ),
            ]
        )
    )


@pytest.fixture
def eeg_events(tmp_path):
    return ns.Study(
        name="Test2023Meg",
        path=tmp_path / "data",
        query="timeline_index<3",
    ).run()


def test_predefined_train_test_split(events):
    transform = PredefinedSplit(
        event_type="Phoneme",
        test_split_query="(type == 'Phoneme') & (text == 't')",
        col_name="split",
        valid_split_by="timeline",
        valid_split_ratio=0.5,
        valid_random_state=33,
    )
    new_events = transform(events)
    assert (new_events.loc[new_events.split == "test", "text"] == ["t", "t"]).all()
    assert (new_events.loc[~new_events.split.isna(), "type"] == "Phoneme").all()
    assert (new_events.loc[new_events.split == "val", "timeline"] == ["blublublu"]).all()


def test_predefined_split_event_type(events):
    """Test that event_type parameter only splits events of specified type."""
    # Add a predefined split column to the original events
    events = events.copy()
    events["split"] = "existing"

    # Apply PredefinedSplit only to Phoneme events
    transform = PredefinedSplit(
        event_type="Phoneme",
        test_split_query="text == 't'",
        col_name="split",
        valid_split_by="timeline",
        valid_split_ratio=0.5,
        valid_random_state=33,
    )
    new_events = transform(events)

    # Check that Phoneme events have the new split applied
    phoneme_events = new_events[new_events.type == "Phoneme"]
    assert "split" in phoneme_events.columns
    assert set(phoneme_events["split"].unique()) == {"train", "val", "test"}

    # Check that test split contains the correct Phoneme events
    test_phonemes = phoneme_events[phoneme_events.split == "test"]
    assert len(test_phonemes) == 2  # Two phonemes with text 't'
    assert (test_phonemes.text == "t").all()

    # Check that Word events still have the original split value
    word_events = new_events[new_events.type == "Word"]
    assert len(word_events) == 1
    assert (word_events["split"] == "existing").all()

    # Verify total number of events is preserved
    assert len(new_events) == len(events)


def test_crop_sleep_recordings(sleep_events):
    max_wake_duration_min = 30.0
    transform = CropSleepRecordings(max_wake_duration_min=max_wake_duration_min)
    new_events = transform(sleep_events)

    assert new_events.iloc[0].duration == max_wake_duration_min * 60.0
    assert new_events.iloc[-1].duration == max_wake_duration_min * 60.0


@pytest.mark.parametrize("max_duration_s", [None, 10.0, 110.0])
def test_crop_timelines(eeg_events, max_duration_s: float | None):
    start_offset_s = 10.0
    transform = CropTimelines(
        event_type="Meg",
        start_offset_s=start_offset_s,
        max_duration_s=max_duration_s,
    )
    new_events = transform(eeg_events)

    assert new_events.shape == eeg_events.shape
    assert (new_events.loc[new_events.type == "Meg", "start"] == start_offset_s).all()
    if max_duration_s is not None:
        assert (
            new_events.loc[new_events.type == "Meg", "duration"] <= max_duration_s
        ).all()


@pytest.mark.parametrize("start_offset_s", [100.0, 200.0])
def test_crop_timelines_invalid_duration(eeg_events, start_offset_s: float):
    transform = CropTimelines(event_type="Meg", start_offset_s=start_offset_s)
    with pytest.raises(ValueError, match="non-positive duration"):
        transform(eeg_events)


def test_text_preprocessor(text_events):
    transform = TextPreprocessor(neuro_event_type="Eeg")
    output = transform(text_events)

    assert all(output["duration"] >= 0)
    assert set(output["type"]) == {"Word", "Eeg", "Audio"}
    assert not output["text"].isin(["", " "]).any()
    assert output.loc[output.type == "Word", "text"].str.islower().all()


def test_similarity_split_sentence(sentence_events):
    transform = SimilaritySplit(
        stim_event_type="Sentence",
        valid_split_ratio=0.33,
        test_split_ratio=0.33,
        threshold=0.2,
    )
    output = transform(sentence_events)

    assert "split" in output.columns
    assert set(output["split"]) == {"train", "val", "test", ""}
    assert (output[output.type == "Sentence"].groupby("split").text.nunique() == 1).all()


@pytest.mark.parametrize(
    "valid_random_state,test_random_state", [(9, 9), (99, 9), (11, 31)]
)
def test_similarity_split_sentence_use_sklearn(
    sentence_events,
    valid_random_state: int,
    test_random_state: int,
) -> None:
    transform = SimilaritySplit(
        stim_event_type="Sentence",
        valid_split_ratio=0.33,
        test_split_ratio=0.33,
        valid_random_state=valid_random_state,
        test_random_state=test_random_state,
        threshold=0.2,
        use_sklearn_split=True,
    )
    output = transform(sentence_events)

    assert "split" in output.columns
    assert set(output["split"].dropna()) == {"train", "val", "test"}
    assert (output[output.type == "Sentence"].groupby("split").text.nunique() == 1).all()


def test_add_default_events():
    tl_duration = 200.0
    events = pd.DataFrame(
        [
            dict(
                type="Eeg",
                start=0.0,
                duration=tl_duration,
                frequency=100.0,
                subject="1",
                timeline="session_1",
                filepath=":.fif",
            ),
            dict(
                type="Seizure",
                start=10.0,
                duration=5.0,
                state="seiz",
                timeline="session_1",
            ),
            dict(
                type="Seizure",
                start=75.0,
                duration=10.0,
                state="seiz",
                timeline="session_1",
            ),
            # Events that start and end with the timeline
            dict(
                type="Eeg",
                start=0.0,
                duration=tl_duration,
                frequency=100.0,
                subject="1",
                timeline="session_2",
                filepath=":.fif",
            ),
            dict(
                type="Seizure",
                start=0.0,
                duration=5.0,
                state="seiz",
                timeline="session_2",
            ),
            dict(
                type="Seizure",
                start=180.0,
                duration=20.0,
                state="seiz",
                timeline="session_2",
            ),
        ]
    )
    events = standardize_events(events)

    transform = AddDefaultEvents(
        target_event_type="Seizure",
        default_event_type="Seizure",
        default_event_fields={"state": "bckg"},
    )
    output = transform(events)

    for _, group in output.groupby("timeline"):
        seizure_events = group[group.type == "Seizure"]
        assert seizure_events.duration.sum() == tl_duration
        assert (
            (seizure_events.stop.iloc[:-1].values - seizure_events.start.iloc[1:].values)
            == 0.0
        ).all()


def test_add_default_events_with_overlap():
    tl_duration = 200.0
    events = pd.DataFrame(
        [
            dict(
                type="Eeg",
                start=0.0,
                duration=tl_duration,
                frequency=100.0,
                subject="1",
                timeline="session_3",
                filepath=":.fif",
            ),
            dict(
                type="Seizure",
                start=10.0,
                duration=20.0,
                state="seiz",
                timeline="session_3",
            ),
            dict(
                type="Seizure",
                start=20.0,
                duration=20.0,
                state="seiz",
                timeline="session_3",
            ),
        ]
    )
    events = standardize_events(events)

    transform = AddDefaultEvents(
        target_event_type="Seizure",
        default_event_type="Seizure",
        default_event_fields={"state": "bckg"},
    )
    output = transform(events)

    assert output.shape[0] == 5
    assert output[output.state == "bckg"].shape[0] == 2


def test_offset_events(events):
    transform = OffsetEvents(
        query="type == 'Phoneme'",
        start_offset=-2.0,
        end_offset=1.0,
    )
    output = transform(events)

    actual = output[["type", "timeline", "start", "duration"]].sort_values(
        ["timeline", "start"], ignore_index=True
    )
    expected_starts = [0.0, 40.0, 42.0, 43.0, 43.0, 43.0, 44.5, 44.5]
    expected_durations = [60.0, 4.0, 1.0, 2.0, 2.0, 2.0, 10.0, 2.5]
    assert list(actual.start) == expected_starts
    assert list(actual.duration) == expected_durations


def test_offset_events_negative_duration(events):
    transform = OffsetEvents(
        query="type == 'Phoneme'",
        start_offset=10.0,
        end_offset=-10.0,
    )
    with pytest.raises(
        ValueError,
        match="Offsetting events resulted in negative or zero duration for some events.",
    ):
        transform(events)


# ---------------------------------------------------------------------------
# ShuffleTrainingLabels
# ---------------------------------------------------------------------------


@pytest.fixture
def split_events_with_labels():
    """Events with a pre-populated ``split`` column and a ``code`` label field."""
    n_train, n_val, n_test = 40, 10, 10
    n = n_train + n_val + n_test
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return standardize_events(
        pd.DataFrame(
            {
                "type": ["Stimulus"] * n,
                "start": [float(i) for i in range(n)],
                "duration": [1.0] * n,
                "split": splits,
                "subject": [f"S{i % 5}" for i in range(n)],
                "timeline": [f"tl_{i % 5}" for i in range(n)],
                "code": [str(i % 4) for i in range(n)],
            }
        )
    )


def test_shuffle_training_labels_only_train_is_changed(split_events_with_labels):
    transform = ShuffleTrainingLabels(event_field="code", random_state=0)
    output = transform(split_events_with_labels)

    for split in ("val", "test"):
        before = split_events_with_labels.loc[
            split_events_with_labels.split == split, "code"
        ].tolist()
        after = output.loc[output.split == split, "code"].tolist()
        assert before == after


def test_shuffle_training_labels_preserves_class_marginal(split_events_with_labels):
    transform = ShuffleTrainingLabels(event_field="code", random_state=0)
    output = transform(split_events_with_labels)

    before = split_events_with_labels.loc[
        split_events_with_labels.split == "train", "code"
    ].value_counts()
    after = output.loc[output.split == "train", "code"].value_counts()
    pd.testing.assert_series_equal(
        before.sort_index(), after.sort_index(), check_names=False
    )


def test_shuffle_training_labels_actually_shuffles(split_events_with_labels):
    transform = ShuffleTrainingLabels(event_field="code", random_state=0)
    output = transform(split_events_with_labels)

    before = split_events_with_labels.loc[
        split_events_with_labels.split == "train", "code"
    ].tolist()
    after = output.loc[output.split == "train", "code"].tolist()
    assert before != after


def test_shuffle_training_labels_deterministic(split_events_with_labels):
    t1 = ShuffleTrainingLabels(event_field="code", random_state=7)
    t2 = ShuffleTrainingLabels(event_field="code", random_state=7)
    pd.testing.assert_frame_equal(
        t1(split_events_with_labels), t2(split_events_with_labels)
    )


def test_shuffle_training_labels_different_seeds(split_events_with_labels):
    t1 = ShuffleTrainingLabels(event_field="code", random_state=7)
    t2 = ShuffleTrainingLabels(event_field="code", random_state=42)

    train1 = t1(split_events_with_labels).query("split == 'train'")["code"].tolist()
    train2 = t2(split_events_with_labels).query("split == 'train'")["code"].tolist()
    assert train1 != train2


def test_shuffle_training_labels_respects_event_types(split_events_with_labels):
    extra = split_events_with_labels.iloc[:5].copy()
    extra["type"] = "Other"
    extra["code"] = "9"
    events = pd.concat([split_events_with_labels, extra], ignore_index=True)
    transform = ShuffleTrainingLabels(
        event_field="code", event_types="Stimulus", random_state=0
    )
    output = transform(events)

    other_codes = output.loc[output.type == "Other", "code"].tolist()
    assert all(c == "9" for c in other_codes)


def test_shuffle_training_labels_missing_field_raises(split_events_with_labels):
    transform = ShuffleTrainingLabels(event_field="not_a_column", random_state=0)
    with pytest.raises(ValueError, match="event_field"):
        transform(split_events_with_labels)


def test_data_instantiation(tmp_path: Path) -> None:
    """Data can be instantiated with Study or Chain configs."""
    common: tp.Any = dict(
        neuro={"name": "MneRaw"},
        target={"name": "LabelEncoder", "event_field": "text"},
        channel_positions={"name": "ChannelPositions"},
        trigger_event_type="Word",
        study={"name": "Test2023Meg", "path": tmp_path / "data"},
    )
    _ = Data(**common)
