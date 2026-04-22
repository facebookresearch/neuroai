# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import pickle
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import neuralset as ns
import neuralset.segments as seg
from neuralset import dataloader as dl
from neuralset.events import etypes
from neuralset.segments import list_segments

from . import utils
from .events import standardize_events
from .events.transforms.test_transforms import create_wav


def _make_segments(start=0.0) -> list[ns.segments.Segment]:
    events = standardize_events(
        pd.DataFrame(
            [
                dict(type="Word", start=10.0, duration=0.5, text="Hello", timeline="x"),
                dict(type="Word", start=12.0, duration=0.5, text="world", timeline="x"),
                dict(type="Event", start=10.0, duration=2.5, timeline="x"),
            ]
        )
    )
    segments = list_segments(
        events, triggers=events.type == "Word", start=start, duration=1.0
    )
    return segments


def test_make_segments() -> None:
    segments = _make_segments(start=0)
    assert len(segments) == 2
    for s in segments:
        assert sum(s.events.type == "Word") == 1


def test_segment() -> None:
    segments = _make_segments(start=0)
    s = segments[0]
    assert len(s.ns_events) > 0
    assert len(s.events) > 0
    assert set(s._to_extractor()) == {"events", "start", "duration", "trigger"}


@pytest.mark.parametrize("reloaded", (True, False))
def test_find_intersect(reloaded: bool, tmp_path: Path, validated: bool = True) -> None:
    #  |-----A----|
    #     |--B--|
    #     |-----C-----|
    # |----D----|
    #                |---E---]

    for event_type in "abcdef":
        type(event_type, (ns.events.Event,), {})
    try:
        events = pd.DataFrame(
            {
                "type": ["a", "b", "c", "d", "e"],
                "start": [-0.5, 0.0, 0.0, -1.0, 10.0],
                "duration": [2.0, 1.0, 2.0, 2.0, 2.0],
                "timeline": ["x", "x", "x", "x", "x"],
            }
        )
        if validated:
            with utils.ignore_all():
                events = standardize_events(events)

        sel = events.type == "a"

        # Test case: find overlapping events
        found = seg.find_overlap(
            events=events, triggers=sel, duration=events.loc[sel].duration.item()
        )
        expected = "dacb" if validated else "abcd"
        assert "".join(events.loc[found].type) == expected

        # Test case: same with isolated event
        sel = events.type == "e"
        found = seg.find_overlap(
            events=events, triggers=sel, duration=events.loc[sel].duration.item()
        )
        assert "".join(events.loc[found].type) == "e"

        # test ns api
        events = standardize_events(
            pd.DataFrame(
                {
                    "type": ["Word"] * 6,
                    "text": ["a", "b", "c", "d", "e", "f"],
                    "start": [-0.5, 0, 0, -1, 10, 0],
                    "duration": [2, 1, 2, 2, 2, 3],
                    "timeline": [str(k) for k in [1, 1, 1, 1, 1, 2]],
                }
            )
        )
        if reloaded:
            # dump and reload to test for changed column dtypes
            fp = tmp_path / "data.parquet"
            events.to_parquet(fp)
            events = pd.read_parquet(fp, dtype_backend="numpy_nullable")

        # check _iter index dtype:
        actual = seg.find_overlap(events, events.text == "a")
        assert "".join(events.loc[actual].text) == "dacb"

        actual = seg.find_overlap(events, events.text == "a")
        assert "".join(events.loc[actual].text) == "dacb"
        # cannot find overlap from a specific time if multiple timeline
        with pytest.raises(AssertionError):
            actual = seg.find_overlap(events, start=0.0, duration=1.0)

        actual = seg.find_overlap(events.query("timeline=='1'"), start=0.0, duration=1.0)
        assert "".join(events.loc[actual].text) == "dacb"

        actual = seg.find_overlap(events, events.text == "f")
        assert "".join(events.loc[actual].text) == "f"

        expected_list = "d", "da", "dacb", "dacb", "e", "f"
        actual_segments = seg.list_segments(
            events, pd.Series(True, index=events.index), duration=0.1
        )
        for act, exp in zip(actual_segments, expected_list):
            sub = events.loc[[e._index for e in act.ns_events]]
            assert sub.timeline.nunique() == 1
            assert "".join(sub.text) == exp

        assert isinstance(actual_segments[0].trigger, ns.events.Event)
    finally:
        for event_type in "abcdef":
            ns.events.Event._CLASSES.pop(event_type, None)


@pytest.mark.parametrize("index_offset", (0, 12))
def test_trigger(index_offset: int) -> None:
    events = standardize_events(
        pd.DataFrame(
            [
                dict(type="Word", start=42.0, duration=1, text="table", timeline="blu"),
                dict(type="Phoneme", start=42.0, duration=1, text="t", timeline="blu"),
                dict(type="Phoneme", start=42.0, duration=1, text="t", timeline="blublu"),
            ]
        )
    )
    events.index = events.index + index_offset
    with pytest.raises(ValueError):
        _ = seg.list_segments(events, events.type == "wrong")
    segs = seg.list_segments(events, events.type == "Word")
    assert len(segs) == 1
    assert len(segs[0].ns_events) == 2, "Event list should be precomputed"  # type: ignore
    assert segs[0].trigger.type == "Word"  # type: ignore
    indices = {e._index for e in segs[0].ns_events}
    assert not indices - set(events.index), "Indices should be in the original dataframe"
    string = pickle.dumps(segs[0])
    pickle.loads(string)


def test_list_segments_duration() -> None:
    data = [dict(type="Word", start=0, duration=100, text="text", timeline="blu")]
    events = standardize_events(pd.DataFrame(data))
    duration = 3
    segs = seg.list_segments(
        events, triggers=events.type == "Word", stride=1.234, duration=duration
    )
    durations = [s.duration for s in segs]
    assert all(d == duration for d in durations), f"Got {durations}"


@pytest.mark.parametrize("stride", [5.0, 10.0, 20.0])
@pytest.mark.parametrize("start", [-1.0, 5.0])
@pytest.mark.parametrize("stride_drop_incomplete", [True, False])
def test_list_segments_idx_stride(
    stride: float,
    start: float,
    stride_drop_incomplete: bool,
) -> None:
    data = [
        dict(type="Word", start=0, duration=100, text="text", timeline="blu"),
        dict(type="Word", start=200, duration=100, text="text", timeline="blu"),
        dict(type="Phoneme", start=200, duration=100, text="text", timeline="blublu"),
    ]
    events = standardize_events(pd.DataFrame(data))
    duration = 10
    segs = list_segments(
        events=events,
        triggers=events.type == "Word",
        start=start,
        duration=duration,
        stride=stride,
        stride_drop_incomplete=stride_drop_incomplete,
    )

    starts = np.array([s.start for s in segs])
    stops = starts + duration
    # starts must be within [trigger.start + offset, trigger.stop)
    assert (
        ((starts >= 0.0 + start - 1e-10) & (starts < 100.0 + 1e-10))
        | ((starts >= 200.0 + start - 1e-10) & (starts < 300.0 + 1e-10))
    ).all()
    # with drop_incomplete, no segment may extend past its trigger's stop
    if stride_drop_incomplete:
        assert ((stops <= 100.0 + 1e-10) | (stops <= 300.0 + 1e-10)).all()

    strides = np.diff(starts)
    assert sum(strides == stride) == len(strides) - 1

    durations = [s.duration for s in segs]
    assert all(d == duration for d in durations), f"Got {durations}"


def test_find_incomplete_segments(tmp_path: Path) -> None:
    sentence = ("This is a sentence for the unit tests").split(" ")
    words = [
        dict(
            type="Word",
            text=sentence[i],
            start=i,
            duration=1.0,
            language="english",
            timeline="foo",
            split="train",
        )
        for i in range(len(sentence))
    ]
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10)
    sound = dict(type="Audio", start=3, timeline="foo", filepath=fp)
    events = [sound] + words
    events_df = standardize_events(pd.DataFrame(events))

    segments = seg.list_segments(
        events_df,
        events_df.type == "Word",
        start=-0.5,
        duration=1.0,
    )
    indices = seg.find_incomplete_segments(segments, [etypes.Audio])
    # the three first words are invalid
    assert len(indices) == 3


def _make_df() -> pd.DataFrame:
    events = [
        etypes.Word(start=2, duration=3, text="a", timeline="t2"),
        etypes.Word(start=1, duration=2, text="b", timeline="t2"),
        etypes.Word(start=0, duration=2, text="c", timeline="t1"),
        etypes.Word(start=0, duration=3, text="d", timeline="t1"),
    ]
    df = pd.DataFrame([e.to_dict() for e in events])
    df = standardize_events(df)
    return df


def test_sorting() -> None:
    df = _make_df()
    assert tuple(df.text) == ("b", "a", "d", "c")
    assert tuple(df.columns[:4]) == ("type", "start", "duration", "timeline")


def test_no_duration() -> None:
    event = etypes.Word(start=12, duration=0, text="d", timeline="t1")
    df = pd.DataFrame([event.to_dict()])
    with pytest.warns(UserWarning, match="with null duration"):
        _ = standardize_events(df)


def test_prepare_strided_window() -> None:
    params: dict[str, tp.Any] = dict(start=0, stop=1, stride=0.6, duration=2)
    out = seg._prepare_strided_windows(**params, drop_incomplete=False)
    assert len(out) == 2
    with pytest.raises(RuntimeError):
        _ = seg._prepare_strided_windows(**params, drop_incomplete=True)
    # FP precision: exact boundary, drop_incomplete=False must not overshoot
    starts, _ = seg._prepare_strided_windows(
        0, 300, stride=10, duration=10, drop_incomplete=False
    )
    assert len(starts) == 30
    assert starts[-1] < 300
    # FP precision: exact boundary, drop_incomplete=True must include last fitting window
    starts, _ = seg._prepare_strided_windows(
        0, 300, stride=0.1, duration=0.1, drop_incomplete=True
    )
    assert len(starts) == 3000
    np.testing.assert_allclose(starts[-1], 299.9, atol=1e-10)


# ---------------------------------------------------------------------------
# _EventStore internals (view semantics, pickle, lightweight segments)
# ---------------------------------------------------------------------------


def _store_and_seg(
    n_events: int = 5,
    duration: float = 0.5,
    spacing: float = 1.0,
    seg_start: float = 0.0,
    seg_dur: float = 2.0,
    trigger_idx: int | None = None,
) -> tuple[seg._EventStore, seg.Segment]:
    rows = [
        dict(
            type="Word",
            start=i * spacing,
            duration=duration,
            text=f"w{i}",
            timeline="tl0",
        )
        for i in range(n_events)
    ]
    df = standardize_events(pd.DataFrame(rows))
    store = seg._EventStore.from_dataframe(df)
    return store, seg.Segment(
        start=seg_start,
        duration=seg_dur,
        timeline="tl0",
        _trigger_idx=trigger_idx,
        _store_id=store._id,
        _store_ref=store,
    )


def test_segment_ns_events_live() -> None:
    """ns_events reflects mutations; trigger and accessors work."""
    _, s = _store_and_seg(
        n_events=10,
        spacing=1.0,
        duration=0.5,
        seg_start=0.0,
        seg_dur=3.0,
        trigger_idx=2,
    )
    assert len(s.ns_events) == 3
    s.start = 2.0
    assert len(s.ns_events) == 3

    assert s.trigger is not None
    assert s.trigger.start == 2.0
    assert set(s._to_extractor()) == {"start", "duration", "events", "trigger"}
    assert isinstance(s.events, pd.DataFrame)
    assert len(s.events) == len(s.ns_events)

    # zero-duration segment: one event when inside, nothing in a gap
    s.start, s.duration = 0.25, 0.0
    assert len(s.ns_events) == 1
    s.start = 0.75
    assert len(s.ns_events) == 0


def test_pickle_roundtrip_and_registry() -> None:
    """Segment roundtrips via registry; store self-registers on unpickle."""
    store, s = _store_and_seg(
        n_events=5,
        spacing=1.0,
        duration=0.5,
        seg_start=1.0,
        seg_dur=2.0,
    )
    original = s.ns_events
    loaded = pickle.loads(pickle.dumps(s))
    assert loaded._store_ref is store, "__setstate__ re-establishes ref"
    assert loaded.ns_events == original

    sid = store._id
    del seg._EventStore._REGISTRY[sid]
    reloaded = pickle.loads(pickle.dumps(store))
    assert seg._EventStore._REGISTRY[sid] is reloaded


def test_segment_pickle_lightweight() -> None:
    _, s = _store_and_seg(
        n_events=100,
        spacing=0.1,
        duration=0.5,
        seg_start=0.0,
        seg_dur=5.0,
    )
    assert len(pickle.dumps(s)) < len(pickle.dumps(s._store)) / 5


def test_varying_durations_in_bucket() -> None:
    """Overlap query with mixed durations exercises the searchsorted max_dur window."""
    n = 200  # > 150 threshold to hit the searchsorted path
    rows = [
        dict(type="Word", start=float(i), duration=0.1, text=f"w{i}", timeline="tl")
        for i in range(n)
    ]
    # One long event starting early but extending far into the timeline
    rows.append(dict(type="Word", start=5.0, duration=50.0, text="long", timeline="tl"))
    df = standardize_events(pd.DataFrame(rows))
    store = seg._EventStore.from_dataframe(df)
    # Query window [40, 45): the long event overlaps but doesn't start here
    found = store.overlapping(start=40.0, duration=5.0, timeline="tl")
    found_starts = {e.start for e in found}
    assert 5.0 in found_starts, "Long event should be found via max_dur window"
    # Short event at start=0 is in the widened searchsorted window
    # (0 >= 40 - 50) but doesn't overlap [40, 45) — must be excluded
    assert 0.0 not in found_starts
    # Cross-check against brute-force
    expected = {
        (r.start, r.duration)  # type: ignore
        for r in df.itertuples()
        if r.start < 45.0 and r.start + r.duration > 40.0  # type: ignore
    }
    actual = {(e.start, e.duration) for e in found}
    assert actual == expected


def test_dataset_pickle_carries_store() -> None:
    """Simulates spawn: pickle the dataset, unpickle in a fresh registry."""
    rows = [
        dict(type="Word", start=i * 0.5, duration=0.5, text=f"w{i}", timeline="tl0")
        for i in range(20)
    ] + [dict(type="Stimulus", start=0.0, duration=10.0, timeline="tl0")]
    df = standardize_events(pd.DataFrame(rows))
    segments = list_segments(df, triggers=df.type == "Word", duration=2.0)
    ext = ns.extractors.Pulse(frequency=100.0, event_types="Stimulus", aggregation="sum")
    ds = dl.SegmentDataset({"pulse": ext}, segments, pad_duration=2.0)

    blob = pickle.dumps(ds)
    old_registry = dict(seg._EventStore._REGISTRY)
    seg._EventStore._REGISTRY.clear()
    try:
        ds2 = pickle.loads(blob)
        assert "pulse" in ds2[0].data
        for s in ds2.segments:
            assert len(s.ns_events) > 0
    finally:
        seg._EventStore._REGISTRY.update(old_registry)


def test_segment_copy() -> None:
    """copy() preserves store ref, adjusts fields, does not carry cache."""
    _, s = _store_and_seg(n_events=5, seg_start=1.0, seg_dur=2.0)
    _ = s.ns_events  # populate cache
    c = s.copy(offset=0.5, duration=1.0)
    assert (c.start, c.duration, c.timeline) == (1.5, 1.0, s.timeline)
    assert c._store_ref is s._store_ref
    assert c._cached_events == []
    assert c.ns_events == s._store.overlapping(1.5, 1.0, s.timeline)


def test_pickle_gc_resilience() -> None:
    """Unpickled segments survive GC: via re-established ref, or via cache."""
    store, s = _store_and_seg(n_events=5, seg_start=0.0, seg_dur=2.0)
    expected = s.ns_events  # populate cache
    blob = pickle.dumps(s)
    # copy of unpickled segment keeps store alive after GC
    c = pickle.loads(blob).copy(offset=0, duration=3.0)
    del store, s
    gc.collect()
    assert len(c.ns_events) > 0
    # cached events survive even without the store
    seg._EventStore._REGISTRY.pop(c._store_id, None)
    loaded = pickle.loads(blob)
    assert loaded.ns_events == expected
