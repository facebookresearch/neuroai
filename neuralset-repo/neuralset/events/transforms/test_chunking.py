# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import types
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import neuralset as ns
from neuralset.events import etypes
from neuralset.events import transforms as _transf
from neuralset.events.transforms import chunking

from .test_transforms import create_wav


@pytest.mark.parametrize(
    "tiling,min_d,max_d,expected",
    [
        ("equal", 0.0, 3.0, [(0.0, 2.5), (2.5, 5.0), (5.0, 7.5), (7.5, 10.0)]),
        ("equal", 0.0, np.inf, [(0.0, 10.0)]),  # no cap → single chunk
        ("equal", 2.5, 3.0, "min_duration"),  # 2*min > max
        ("equal", 20.0, 50.0, "cannot tile"),  # section < min_duration
        ("max", 0.0, 3.0, [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 10.0)]),
        ("max", 0.0, np.inf, [(0.0, 10.0)]),
        ("max", 0.0, 5.0, [(0.0, 5.0), (5.0, 10.0)]),  # exact fit, no trailing
        # "max" doesn't enforce 2*min<=max — trailing is the caller's to drop.
        ("max", 5.0, 3.0, [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 10.0)]),
        ("max", 20.0, 50.0, [(0.0, 10.0)]),  # section < max → whole-section trailing
    ],
)
def test_section_tile(
    tiling: chunking.Tiling,
    min_d: float,
    max_d: float,
    expected: list[tuple[float, float]] | str,
) -> None:
    section = chunking._Section(0.0, 10.0)
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            section.tile(min_d, max_d, tiling)
    else:
        assert section.tile(min_d, max_d, tiling) == expected


def _words(*specs: tuple[float, float, str]) -> pd.DataFrame:
    """Build a Word-events DataFrame from ``(start, duration, split)`` tuples."""
    const = dict(type="Word", text="a", language="english", timeline="foo")
    dicts = [dict(start=s, duration=d, split=split, **const) for s, d, split in specs]
    return pd.DataFrame(dicts)


@pytest.mark.parametrize(
    "frequency,rows,allow_leakage,expected",
    [
        # No / empty use_rows → one section spanning the whole event.
        (100.0, None, False, [(0.0, 10.0)]),
        (100.0, _words(), False, [(0.0, 10.0)]),
        # 3 train + 1 test word: coalesce same-split, split at midpoint of gap [3.5, 5.0].
        (
            100.0,
            _words(
                (0.5, 1.0, "train"),
                (1.5, 1.0, "train"),
                (2.5, 1.0, "train"),
                (5.0, 2.0, "test"),
            ),
            False,
            [(0.0, 4.25), (4.25, 10.0)],
        ),
        # Fmri-at-coarse-TR: TR=2 s ≫ 0.3 s gap
        # no sample-aligned boundary separates the differently-labeled runs
        (0.5, _words((1.0, 0.5, "train"), (1.8, 0.5, "test")), False, "label leakage"),
        (
            0.5,
            _words((1.0, 0.5, "train"), (1.8, 0.5, "test")),
            True,
            [(0.0, 1.65), (1.65, 10.0)],
        ),
    ],
)
def test_build_sections(
    frequency: float,
    rows: pd.DataFrame | None,
    allow_leakage: bool,
    expected: list[tuple[float, float]] | str,
) -> None:
    event: tp.Any = types.SimpleNamespace(start=0.0, duration=10.0, frequency=frequency)
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            chunking._build_sections(event, rows, allow_leakage)
    else:
        assert chunking._build_sections(event, rows, allow_leakage) == expected


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(event_type_to_chunk="NotAnEvent"), "not a splittable"),
        (
            dict(
                event_type_to_chunk="Audio",
                min_duration=1.0,
                max_duration=1.5,
                tiling="equal",
            ),
            "min_duration",
        ),
        # Same config under ``"max"`` is accepted (post-filter handles it).
        (
            dict(event_type_to_chunk="Audio", min_duration=1.0, max_duration=1.5),
            None,
        ),
    ],
)
def test_chunk_events_config_validation(kwargs: dict, match: str | None) -> None:
    """``model_post_init`` rejects (or accepts when ``match is None``)."""
    if match is None:
        _transf.ChunkEvents(**kwargs)
    else:
        with pytest.raises(ValueError, match=match):
            _transf.ChunkEvents(**kwargs)


def test_chunk_meg_first_samp() -> None:
    """Chunking a FIF with ``raw.first_samp > 0`` (Mne2013Sample has first_samp ≈ 6k).

    Verifies ``Event.start``/``offset`` bookkeeping when the source event's
    absolute start is nonzero, and that ``MneRaw.read()`` crops relative to the
    data (not absolute time) so chunks partition the full raw exactly — both in
    sample count and in content (catches off-by-one or wrong-position crops).
    """
    events = ns.Study(name="Mne2013Sample", path=ns.CACHE_FOLDER).run()
    row = events.loc[events.type == "Meg"].iloc[0]
    event = etypes.Meg.from_dict(row)
    assert event.start > 0, "Mne2013Sample should have first_samp > 0"
    full_data = event.read().get_data()
    target_dur = event.duration / 3.3  # off-grid to exercise boundary snapping
    chunked = _transf.ChunkEvents(event_type_to_chunk="Meg", max_duration=target_dur)(
        events
    )
    chunks = [
        etypes.Meg.from_dict(r)
        for r in chunked.loc[chunked.type == "Meg"].itertuples(index=False)
    ]
    # tiling="max": floor(3.3) = 3 chunks at target_dur + 1 trailing remainder.
    assert len(chunks) == 4
    assert chunks[0].start == event.start and chunks[0].offset == 0.0
    # First 3 are exactly target_dur (within sample-snap); last is the remainder.
    for ch in chunks[:3]:
        assert ch.duration == pytest.approx(target_dur, abs=1 / event.frequency)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start == pytest.approx(a.start + a.duration)
        assert b.offset == pytest.approx(a.offset + a.duration)
    last_chunk_stop = chunks[-1].start + chunks[-1].duration
    assert last_chunk_stop == pytest.approx(event.start + event.duration)
    # ``1/freq`` = sample-snap slack; ``1e-6`` = float-compare tol.
    assert all(ch.duration <= target_dur + 1 / event.frequency + 1e-6 for ch in chunks)
    # chunk data concatenates to the full raw exactly — proves the crop is
    # contiguous (no gap/overlap) and positioned correctly relative to first_samp.
    concat = np.concatenate([ch.read().get_data() for ch in chunks], axis=1)
    assert concat.shape == full_data.shape
    np.testing.assert_array_equal(concat, full_data)


@pytest.mark.parametrize(
    # Meg + Fnirs (low sfreq, custom ``_read``) + Fmri cover all chunking routes.
    # ``expected_by_word`` = number of same-``split`` runs: Meg has 2 (train/test);
    # Fnirs + Fmri have 4 each (train/test/val/train).
    "study_name,expected_by_word",
    [("Test2023Meg", 2), ("Test2024Fnirs", 4), ("Test2023Fmri", 4)],
)
def test_chunk_events_neuro(
    test_data_path: Path, study_name: str, expected_by_word: int
) -> None:
    """Word-split and max-duration chunking: counts, label homogeneity, exact partition.

    Label-leakage safety is especially critical for Fmri at coarse TR, where
    naive boundary rounding could pull a TR across a split transition.
    Non-SpecialLoader partitioning is covered by ``test_chunk_meg_first_samp``
    (MneRaw) and ``test_etypes.test_fmri_volume_chunked_read`` (Fmri).
    """
    event_type = study_name.removeprefix("Test")[4:]  # drop "Test" + YYYY
    events = ns.Study(
        name=study_name, path=test_data_path, query="timeline_index<1"
    ).run()

    chunker = functools.partial(_transf.ChunkEvents, event_type_to_chunk=event_type)
    by_word = chunker(event_type_to_split_by="Word")(events)
    word_chunks = by_word[by_word.type == event_type]
    assert len(word_chunks) == expected_by_word
    words = events[events.type == "Word"]
    for c_start, c_dur in zip(word_chunks.start, word_chunks.duration):
        c_stop = c_start + c_dur
        overlap = words[(words.start + words.duration > c_start) & (words.start < c_stop)]
        labels = set(overlap.split.astype(str))
        assert len(labels) <= 1, (
            f"label leakage in chunk [{c_start}, {c_stop}): overlapping splits {labels}"
        )

    # Off-grid cap to avoid accidental alignment with the sample period.
    row = events.loc[events.type == event_type].iloc[0]
    event = etypes.BaseSplittableEvent.from_dict(row)
    max_duration = row.duration / 2.7
    # ``"equal"`` here; ``"max"`` covered by ``test_chunk_meg_first_samp``.
    by_max = chunker(max_duration=max_duration, tiling="equal")(events)
    rows = by_max.loc[by_max.type == event_type].itertuples(index=False)
    chunks = [etypes.BaseSplittableEvent.from_dict(r) for r in rows]
    assert len(chunks) == 3  # ceil(2.7)
    # Chunk reads must partition the underlying data exactly (catches gaps/overlaps
    # from independent rounding). Time is the last axis for both MneRaw and Fmri.
    get = "get_fdata" if event_type == "Fmri" else "get_data"
    full = getattr(event.read(), get)()
    concat = np.concatenate([getattr(c.read(), get)() for c in chunks], axis=-1)
    np.testing.assert_array_equal(concat, full)


@pytest.mark.parametrize(
    "audio_duration,expected_durations,expected_offsets",
    [
        (22.0, [10.0, 10.0], [0.0, 10.0]),  # trailing 2 s < min_duration → dropped
        (25.0, [10.0, 10.0, 5.0], [0.0, 10.0, 20.0]),  # trailing 5 s >= min → kept
    ],
)
def test_chunk_events_max_trailing(
    tmp_path: Path,
    audio_duration: float,
    expected_durations: list[float],
    expected_offsets: list[float],
) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=audio_duration)
    events = pd.DataFrame([dict(type="Audio", start=0, timeline="t", filepath=fp)])
    events = ns.events.standardize_events(events)
    out = _transf.ChunkEvents(
        event_type_to_chunk="Audio",
        max_duration=10.0,
        min_duration=3.0,
        tiling="max",
    )(events)
    chunks = out[out.type == "Audio"].sort_values("start")
    assert list(chunks.duration) == expected_durations
    assert list(chunks.offset) == expected_offsets


def test_chunk_events_chained_time_then_split(tmp_path: Path) -> None:
    """Chaining time-based then split-based chunking on one Audio event.

    The first pass (``max_duration``) produces multiple target events in
    the same timeline; the second pass (``event_type_to_split_by``) must then
    clip sections per event — else a section preceding a later sub-event
    trips a false leakage guard.
    """
    fp = tmp_path / "s.wav"
    create_wav(fp, fs=44100, duration=18.0)
    audio = pd.DataFrame([dict(type="Audio", start=1.0, timeline="foo", filepath=fp)])
    words = _words(
        *[(s, 1.0, "train" if s < 12 else "test") for s in (0, 1, 2, 6, 11, 14, 17)]
    )
    events = ns.events.standardize_events(pd.concat([audio, words], ignore_index=True))
    events = _transf.ChunkEvents(event_type_to_chunk="Audio", max_duration=9.0)(events)
    events = _transf.ChunkEvents(
        event_type_to_chunk="Audio", event_type_to_split_by="Word"
    )(events)
    audio = events[events.type == "Audio"].sort_values("start")
    # Midpoint between last train word stop (12) and first test word start (14).
    assert list(audio.start) == [1.0, 10.0, 13.0]
    # Offsets index into the original 18s file (start=1 ⇒ offset = start - 1).
    assert list(audio.offset) == [0.0, 9.0, 12.0]
