# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import exca
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import neuralset as ns
from neuralset import base as nsbase
from neuralset.events import etypes

from . import base


class Model(ns.BaseModel):
    extractors: tp.Sequence[ns.extractors.BaseExtractor] = ()


class ExternExtractor(ns.extractors.Pulse):
    pass


def test_features_model() -> None:
    extractors: tp.Any = [
        {"name": "Pulse"},
        {"name": "MegExtractor", "frequency": 12},
        {"name": "ExternExtractor"},
    ]
    model = Model(extractors=extractors)
    cfg = exca.ConfDict.from_model(model, uid=True, exclude_defaults=True)

    assert (
        cfg.to_yaml()
        == """extractors:
- name: Pulse
- frequency: 12.0
  name: MegExtractor
- name: ExternExtractor
"""
    )
    string = repr(model.extractors[0])
    assert model.extractors[0].name == "Pulse"  # compatibility
    assert "\n  'name': 'Pulse'" in string, "Wrong custom repr"


def test_dynamic_feature() -> None:
    # word pulse
    extractor = base.Pulse(frequency=1.0, aggregation="sum", event_types="Word")

    event = dict(
        type="Word",
        text="test",
        start=0.0,
        duration=2.0,
        language="english",
        timeline="foo",
    )

    # single event and slicing
    out = extractor(event, start=0.0, duration=4.0)
    assert np.array_equal(out, [[1, 1, 0, 0]])
    # segment does not overlap with event
    out = extractor(event, start=-4.0, duration=4.0)
    assert np.array_equal(out, [[0, 0, 0, 0]])
    # overwrite event duration (legacy BaseStatic.duration = 1.0)
    event2 = dict(event)
    event2["duration"] = 1.0
    out = extractor(event2, start=0.0, duration=4.0)
    assert np.array_equal(out, [[1, 0, 0, 0]])
    # event start after segment
    event["start"] = 2.0
    out = extractor(event, start=0.0, duration=4.0)
    assert np.array_equal(out, [[0, 0, 1, 1]])
    # event start before segment and ends when segment starts
    event["start"] = -2.0
    out = extractor(event, start=0.0, duration=4.0)
    assert np.array_equal(out, [[0, 0, 0, 0]])
    # event start before segment and ends within segment
    event["duration"] = 4.0
    out = extractor(event, start=0.0, duration=4.0)
    assert np.array_equal(out, [[1, 1, 0, 0]])
    # segment starts before 0
    out = extractor(event, start=-1.0, duration=4.0)
    assert np.array_equal(out, [[1, 1, 1, 0]])

    # list of events
    segment = dict(events=pd.DataFrame([event]), start=0, duration=4.0)
    data = extractor(**segment)  # type: ignore
    assert data.max() == 1 and data.min() == 0
    assert np.array_equal(data.shape, (1, 4))

    # two sequential events
    kwargs = dict(type="Word", text="test", language="english", timeline="foo")
    events = pd.DataFrame(
        [
            dict(start=0.0, duration=1.0, **kwargs),
            dict(start=2.0, duration=1.0, **kwargs),
        ]
    )
    out = extractor(events, start=0.0, duration=4)
    assert np.array_equal(out, [[1, 0, 1, 0]])

    # check two simultaneous events
    events = pd.DataFrame(
        [
            dict(start=1.0, duration=2.0, **kwargs),
            dict(start=2.0, duration=2.0, **kwargs),
        ]
    )
    out = extractor(events, start=0.0, duration=4.0)
    assert np.array_equal(out, [[0, 1, 2, 1]])

    # test static extractor
    extractor = base.Pulse(aggregation="sum", event_types="Word")
    out = extractor(events, start=0.0, duration=4.0)
    assert np.array_equal(out, [2.0])
    # TODO wordpulse + phonemepulse?


class Time(base.BaseExtractor):
    """Simple dynamic extractor for testing fill_slice"""

    event_types: str | tuple[str, ...] = "BaseDataEvent"
    frequency: float = 50.0  # default output frequency

    def _get_timed_arrays(
        self, events: list[ns.events.Event], start: float, duration: float
    ) -> tp.Iterable[nsbase.TimedArray]:
        for event in events:
            length = max(1, base.Frequency(self.frequency).to_ind(event.duration))
            data = np.linspace(event.start, event.start + event.duration, length)[None, :]
            yield nsbase.TimedArray(
                data=data,
                start=event.start,
                duration=event.duration,
                frequency=self.frequency,
            )


@pytest.mark.parametrize(
    "start,duration,expected",
    [
        (0.5, 0.6, [0.6, 0.8, 1, 0]),  # side 1
        (-0.5, 1, [0, 0, 0, 0, 0.2, 0.4]),  # side 2
        (0.2, 0.6, [0.2, 0.4, 0.6, 0.8]),  # inside
        (-0.2, 1.4, [0, 0, 0.2, 0.4, 0.6, 0.8, 1, 0]),  # around
        (0.5, 0.1, [0.6]),  # small (half freq)
        (0.5, 0.01, [0.6]),  # smaller  (we may want to decide there is no sample?)
        (2, 1, [0, 0, 0, 0, 0, 0]),  # no overlap after
        (-2, 1, [0, 0, 0, 0, 0, 0]),  # no overlap before
    ],
)
def test_fill_slice(start: float, duration: float, expected: list[float]) -> None:
    feat = Time(frequency=6, event_types=("Image", "Text"))
    event = etypes.Image(start=0, duration=1, timeline="stuff", filepath=__file__)
    out = feat(event, start=start, duration=duration)[0]
    np.testing.assert_array_almost_equal(out, expected)


def test_fill_small_event() -> None:
    feat = Time(frequency=6)
    event = etypes.Image(start=0.5, duration=0.01, timeline="stuff", filepath=__file__)
    out = feat(event, start=0, duration=1)[0]
    np.testing.assert_array_almost_equal(out, [0, 0, 0, 0.5, 0, 0])


@pytest.mark.parametrize(
    "aggreg,expected",
    [
        ("first", [0, 0.5, 1.0]),
        ("trigger", [0, 0, 0.7]),
        ("sum", [0, 0.5, 1.0]),
        ("mean", [0, 0.5, 1.0]),
    ],
)
def test_trigger(aggreg: tp.Literal["first", "trigger"], expected: list[float]) -> None:
    feat = Time(frequency=3, aggregation=aggreg)
    event = etypes.Image(start=0, duration=1, timeline="stuff", filepath=__file__)
    trigger = etypes.Image(start=0.7, duration=1, timeline="stuff", filepath=__file__)
    out = feat(event, start=0, duration=1, trigger=trigger)[0]
    np.testing.assert_array_almost_equal(out, expected)
    # exceptions
    if aggreg == "trigger":
        with pytest.raises(ValueError):
            feat(event, start=0, duration=1, trigger=None)


@pytest.mark.parametrize(
    "aggreg,expected",
    [
        ("sum", [[0, 0.9, 1.9]]),
        ("mean", [[0, 0.45, 0.95]]),
        ("cat", np.array([[0, 0.5, 1], [0.0, 0.4, 0.9]])),
        ("stack", np.array([[[0, 0.5, 1]], [[0.0, 0.4, 0.9]]])),
    ],
)
def test_aggreg(aggreg: tp.Literal["first", "trigger"], expected: list[float]) -> None:
    feat = Time(frequency=3, aggregation=aggreg)
    events = [
        etypes.Image(start=s, duration=1, timeline="stuff", filepath=__file__)
        for s in [0, 0.4]
    ]
    # [0, 0.5, 1] from 0 to 1 and [0.4, 0.9, 1.4] from 0.4 to 1.4
    out = feat(events, start=0, duration=1, trigger=None)
    np.testing.assert_array_almost_equal(out, expected)


def test_single_aggregation_scoped_to_window() -> None:
    """``aggregation='single'`` counts events inside the segment window,
    not across the whole input (typical when the caller passes a full
    recording's DataFrame rather than a segment-scoped subset)."""
    feat = Time(frequency=3, aggregation="single")
    events = [
        etypes.Image(start=s, duration=0.5, timeline="stuff", filepath=__file__)
        for s in [0.0, 2.0, 4.0]
    ]
    # One overlaps [2, 3): picked, no raise.
    feat(events, start=2.0, duration=1.0)
    # Two overlap [0, 3): still ambiguous.
    with pytest.raises(ValueError, match="expected only one"):
        feat(events, start=0.0, duration=3.0)


def test_trigger_outside_window() -> None:
    feat = ns.extractors.WordFrequency(aggregation="trigger")
    trigger = etypes.Word(text="Hello", start=0, duration=1, timeline="stuff")
    out = feat([], start=2, duration=1, trigger=trigger).item()
    assert out > 0


def test_series_input() -> None:
    """Passing a pd.Series (e.g. events.iloc[0]) as events or trigger."""
    word = etypes.Word(text="Hello", start=0, duration=1, timeline="stuff")
    row = pd.Series(word.to_dict())
    # as events
    feat = ns.extractors.WordFrequency()
    ref = feat(word, start=0, duration=1)
    torch.testing.assert_close(feat(row, start=0, duration=1), ref)
    # as trigger
    feat_trig = ns.extractors.WordFrequency(aggregation="trigger")
    ref_trig = feat_trig([], start=2, duration=1, trigger=word).item()
    assert feat_trig([], start=2, duration=1, trigger=row).item() == ref_trig


def test_fill_slice_load_testing() -> None:
    seed = np.random.randint(2**32 - 1)
    for k in range(1000):
        print(f"Seeding with {seed + k} for reproducibility")
        rng = np.random.default_rng(seed + k)
        freq = rng.uniform(0, 50)
        if freq > 1 and rng.integers(2):
            freq = int(freq)
        feat = Time(frequency=freq)
        event = etypes.Image(
            start=rng.uniform(0, 1),
            duration=rng.uniform(0, 1),
            timeline="stuff",
            filepath=__file__,
        )
        start = event.start + rng.uniform(-0.1, 0.1)
        end = event.start + event.duration + rng.uniform(-0.1, 0.1)
        duration = end - start
        if duration < 0:
            duration = rng.uniform(0.1, 0.2)
        feat(event, start=start, duration=duration)


def test_meg_border_cases(test_data_path: Path) -> None:
    # neuro has its own fill slice to deal with mne, so
    # we need a similar check
    query = "timeline_index < 1"
    events = ns.Study(name="Test2023Meg", path=test_data_path, query=query).run()
    event = etypes.Meg.from_dict(events.query('type=="Meg"').iloc[0])
    seed = np.random.randint(2**32 - 1001)
    for k in range(20):
        print(f"Seeding with {seed + k} for reproducibility")
        rng = np.random.default_rng(seed + k)
        extractor = ns.extractors.MegExtractor(frequency=rng.uniform(10, 50))
        start = event.start + rng.uniform(-1, 2) * event.duration
        duration = event.duration * rng.uniform(0, 1)
        out = extractor(event, start=start, duration=duration)
        time_inds = max(1, base.Frequency(extractor.frequency).to_ind(duration))
        assert out.shape[1] == time_inds


class Xp(ns.BaseModel):
    param1: int = 12
    extractor: ns.extractors.BaseExtractor = ns.extractors.Pulse()
    infra: exca.TaskInfra = exca.TaskInfra()

    @infra.apply
    def run(self) -> int:
        return self.param1


def test_cfg_feature_uid(tmp_path: Path) -> None:
    xp = Xp(extractor={"name": "HuggingFaceImage", "infra": {"folder": tmp_path}})  # type: ignore
    cfg = xp.infra.config(uid=True, exclude_defaults=True)
    assert cfg["extractor.name"] == "HuggingFaceImage"


@pytest.mark.parametrize(
    "shape,agg,max_layers,out",
    [
        ((12, 13, 14, 15), "mean", 10, (10, 14, 15)),
        ((12, 13, 14, 15), "max", None, (12, 14, 15)),
        ((12, 13, 14, 15), "sum", None, (12, 14, 15)),
        ((12, 13, 14, 15), "first", None, (12, 14, 15)),
        ((13, 14, 15), "last", 15, (13, 15)),
        ((13, 14, 15), None, 5, (5, 14, 15)),
    ],
)
def test_hf_aggregate_tokens(
    tmp_path: Path,
    shape: tuple[int, ...],
    agg: str | None,
    max_layers: int | None,
    out: tuple[int, ...],
) -> None:
    extractor: tp.Any = {
        "name": "HuggingFaceImage",
        "cache_n_layers": max_layers,
        "token_aggregation": agg,
        "infra": {"folder": tmp_path},
    }
    xp = Xp(extractor=extractor)
    data_n = np.random.rand(*shape)
    data_t = torch.from_numpy(data_n)
    agged_t = xp.extractor._aggregate_tokens(data_t)  # type: ignore
    agged_n = xp.extractor._aggregate_tokens(data_n)  # type: ignore
    assert agged_t.shape == out
    np.testing.assert_array_almost_equal(agged_t.numpy(), agged_n)


def test_huggingface_model_exists():
    base.HuggingFaceMixin(model_name="gpt2")
    with pytest.raises(ValueError):
        base.HuggingFaceMixin(model_name="not_a_model")


@pytest.mark.parametrize(
    "event_field", ["duration", "timeline", "int_field", "float_field", "bool_field"]
)
def test_event_field(event_field: str) -> None:
    event = dict(
        type="Word",
        start=0.0,
        duration=2.0,
        timeline="foo",
        text="test",
        language="english",
        # Extra fields
        int_field=42,
        float_field=3.14,
        bool_field=True,
    )

    extractor = base.EventField(
        frequency=10,
        event_types="Word",
        event_field=event_field,
    )

    if event_field in ["duration", "int_field", "float_field"]:
        extractor.prepare(pd.DataFrame([event]))
        out = extractor.get_static(ns.events.Event.from_dict(event))
        assert np.allclose(out.item(), event[event_field])  # type: ignore
    else:
        with pytest.raises(
            ValueError,
            match=f"Field {event_field} has type",
        ):
            extractor.prepare(pd.DataFrame([event]))


def test_label_encoder(test_data_path: Path) -> None:
    query = "timeline_index < 2"
    events = ns.Study(name="Test2023Meg", path=test_data_path, query=query).run()

    meg_extractor = base.LabelEncoder(
        event_types="Meg",
        event_field="filepath",
        return_one_hot=False,
        allow_missing=False,
    )
    img_extractor = base.LabelEncoder(
        event_types="Image",
        event_field="filepath",
        return_one_hot=False,
        allow_missing=False,
    )
    meg_extractor.prepare(events)
    img_extractor.prepare(events)

    extractors = {"Meg": meg_extractor, "Image": img_extractor}
    segments = ns.segments.list_segments(
        events,
        triggers=events.type == "Image",
        start=0.0,
        duration=1.0,
    )
    dataset = ns.SegmentDataset(extractors=extractors, segments=segments)
    out_inds = dataset.load_all().data

    assert (out_inds["Meg"][:, 0] == torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])).all()
    assert (out_inds["Image"][:, 0] == torch.tensor([2, 3, 1, 0, 2, 3, 1, 0])).all()


@pytest.mark.parametrize("return_one_hot", [True, False])
@pytest.mark.parametrize("treat_missing_as_separate_class", [True, False])
def test_label_encoder_allow_missing(
    test_data_path: Path,
    return_one_hot: bool,
    treat_missing_as_separate_class: bool,
) -> None:
    query = "timeline_index < 1"
    events = ns.Study(name="Test2023Meg", path=test_data_path, query=query).run()

    img_extractor = base.LabelEncoder(
        event_types="Image",
        event_field="filepath",
        return_one_hot=return_one_hot,
        allow_missing=True,
        treat_missing_as_separate_class=treat_missing_as_separate_class,
    )
    img_extractor.prepare(events)

    segments = ns.segments.list_segments(
        events,
        triggers=events.type == "Meg",
        start=0.0,
        duration=0.1,
        stride=0.1,
    )
    dataset = ns.SegmentDataset(extractors={"Image": img_extractor}, segments=segments)
    out = dataset.load_all().data["Image"]

    if return_one_hot:
        if treat_missing_as_separate_class:
            assert out.shape == (1000, 5)
            assert out.sum(dim=0).tolist() == [7, 7, 7, 7, 972]
        else:
            assert out.shape == (1000, 4)
            assert out.sum(dim=0).tolist() == [7, 7, 7, 7]
            # Verify that segments without an Image event are encoded as all-zero vectors.
            assert int((out == 0).all(dim=1).sum().item()) == 972
    else:
        assert out.shape == (1000, 1)
        if treat_missing_as_separate_class:
            assert set(out.unique().tolist()) == {0, 1, 2, 3, -1}
        else:
            assert set(out.unique().tolist()) == {0, 1, 2, 3}


def test_label_encoder_reprepare() -> None:
    """prepare() on a subset is a no-op; on new labels it raises (#2507)."""
    words = ["cat", "dog", "bird"]
    events = pd.DataFrame(
        [
            {
                "type": "Word",
                "text": w,
                "start": float(i),
                "duration": 0.5,
                "timeline": "t1",
            }
            for i, w in enumerate(words)
        ]
    )
    enc = base.LabelEncoder(event_types="Word", event_field="text", return_one_hot=False)
    enc.prepare(events)
    assert enc._n_classes == 3

    enc.prepare(events[events.text == "cat"])
    assert enc._n_classes == 3
    ev_dog = etypes.Word.from_dict(events.iloc[1].to_dict())
    assert enc.get_static(ev_dog).item() == 2  # {'bird': 0, 'cat': 1, 'dog': 2}

    unknown = pd.DataFrame(
        [
            {
                "type": "Word",
                "text": "fish",
                "start": 0.0,
                "duration": 0.5,
                "timeline": "t1",
            }
        ]
    )
    with pytest.raises(ValueError, match="not in the existing mapping"):
        enc.prepare(unknown)


@pytest.mark.parametrize("return_one_hot", [False, True])
def test_label_encoder_one_hot(test_data_path: Path, return_one_hot: bool) -> None:
    query = "timeline_index < 1"
    events = ns.Study(name="Test2023Meg", path=test_data_path, query=query).run()

    event_field = "filepath"
    extractor = base.LabelEncoder(
        event_types="Image",
        event_field=event_field,
        return_one_hot=return_one_hot,
        treat_missing_as_separate_class=False,
    )
    extractor.prepare(events)
    img_events = events[events.type == "Image"]

    inds = []
    for _, ev in img_events.iterrows():
        event = ns.events.Event.from_dict(ev.to_dict())
        inds.append(extractor(event, start=0, duration=1000))

    out_inds = torch.stack(inds, dim=0)
    assert len(out_inds) == img_events.shape[0]

    gt_event_field = img_events[event_field].to_numpy().reshape(-1, 1)
    if return_one_hot:
        encoder = OneHotEncoder(dtype=int, sparse_output=False)
        gt_inds = encoder.fit_transform(gt_event_field)
        # Check that the dim
        assert out_inds.shape == (4, 4)
    else:
        gt_inds = OrdinalEncoder(dtype=int).fit_transform(gt_event_field)[:, 0]
    assert (out_inds.squeeze().numpy() == gt_inds).all()


def test_label_encoder_dynamic(test_data_path: Path) -> None:
    query = "timeline_index < 1"
    events = ns.Study(name="Test2023Meg", path=test_data_path, query=query).run()

    event_field = "filepath"

    extractor = base.LabelEncoder(
        frequency=10,
        aggregation="trigger",
        event_types="Image",
        event_field=event_field,
        return_one_hot=False,
    )

    extractor.prepare(events)
    time_segments = ns.segments.list_segments(
        events, triggers=events.type == "Image", start=0, duration=1
    )

    dataset = ns.SegmentDataset({"label": extractor}, time_segments)
    assert dataset[0].data["label"].ndim == 3
    assert dataset[0].data["label"].shape[-1] == 10
    assert (dataset[0].data["label"] != 0).sum() == 7


@pytest.mark.parametrize("allow_missing", [True, False])
def test_label_encoder_predefined_mapping(allow_missing: bool) -> None:
    mapping = {"a": 0, "b": 1, "c": 4}
    extractor = base.LabelEncoder(
        event_types="Keystroke",
        event_field="text",
        return_one_hot=False,
        predefined_mapping=mapping,
        allow_missing=allow_missing,
    )

    events = pd.DataFrame(
        [
            {"type": "Keystroke", "text": "a", "start": 0, "timeline": ""},
            {"type": "Keystroke", "text": "b", "start": 0, "timeline": ""},
            {"type": "Keystroke", "text": "c", "start": 0, "timeline": ""},
            {"type": "Keystroke", "text": "c", "start": 0, "timeline": ""},
        ]
    )

    extractor.prepare(events)
    out = [
        extractor(ns.events.Event.from_dict(e), 0, 1).item() for _, e in events.iterrows()
    ]
    assert out == [0, 1, 4, 4]

    events_bad = pd.DataFrame(
        [
            {"type": "Keystroke", "text": "a", "start": 0, "timeline": ""},
            {
                "type": "Keystroke",
                "text": "d",
                "start": 0,
                "timeline": "",
            },  # Not in mapping
        ]
    )

    with pytest.raises(ValueError, match="not in the existing mapping"):
        extractor.prepare(events_bad)


def test_event_detector():
    freq = 10
    duration = 0.7
    n_samples = nsbase.Frequency(freq).to_ind(duration)
    events = pd.DataFrame(
        [
            {
                "type": "Meg",
                "start": 0.0,
                "duration": 0.7,
                "frequency": 10,
                "timeline": "",
            },
            {
                "type": "Word",
                "text": "foo",
                "start": 0.1,
                "duration": 0.3,
                "timeline": "",
            },
            {
                "type": "Word",
                "text": "bar",
                "start": 0.2,
                "duration": 0.4,
                "timeline": "",
            },
        ]
    )

    # === Dense
    f = base.EventDetector(mode="dense", event_types="Word", frequency=freq)
    f.prepare(events)
    out = f(events, start=0.0, duration=duration).data[0]
    expected = np.zeros(n_samples)
    expected[int(0.1 * freq) : int((0.1 + 0.3) * freq)] = 1
    expected[int(0.2 * freq) : int((0.2 + 0.4) * freq)] = 1
    assert np.allclose(out, expected)

    # === Start
    f = base.EventDetector(mode="start", event_types="Word", frequency=freq)
    f.prepare(events)
    out = f(events, start=0.0, duration=duration).data[0]
    expected = np.zeros(n_samples)
    expected[int(0.1 * freq)] = 1
    expected[int(0.2 * freq)] = 1
    assert np.allclose(out, expected)

    # === Center
    f = base.EventDetector(mode="center", event_types="Word", frequency=freq)
    f.prepare(events)
    out = f(events, start=0.0, duration=duration).data[0]
    expected = np.zeros(n_samples)
    expected[int((0.1 + 0.15) * freq)] = 1
    expected[int((0.2 + 0.2) * freq)] = 1
    assert np.allclose(out, expected)

    # === Duration
    f = base.EventDetector(mode="duration", event_types="Word", frequency=freq)
    f.prepare(events)
    out = f(events, start=0.0, duration=duration).data[0]
    expected = np.zeros(n_samples)
    expected[int((0.1 + 0.15) * freq)] = 0.3 / duration
    expected[int((0.2 + 0.2) * freq)] = 0.4 / duration
    assert np.allclose(out, expected)


def test_mixed_timelines() -> None:
    event = etypes.Word(
        text="test",
        start=0.0,
        duration=2.0,
        timeline="foo",
    )
    event2 = event.model_copy()
    event2.timeline += "2"
    extractor = base.Pulse(frequency=1.0, aggregation="sum", event_types="Word")
    with pytest.raises(ValueError):
        extractor([event, event2], start=0, duration=2)
