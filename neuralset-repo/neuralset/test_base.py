# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from pathlib import Path

import exca
import numpy as np
import pandas as pd
import pytest
import yaml
from exca.cachedict import contiguous

import neuralset as ns

from . import base
from .extractors import neuro
from .segments import list_segments

logger = logging.getLogger(__name__)


class A(base._Module):
    requirements: tp.ClassVar[tuple[str, ...]] = ("req_a",)


class B(A):
    requirements: tp.ClassVar[tuple[str, ...]] = ("req_b",)


class C(B):
    pass


def test_requirements() -> None:
    assert B.requirements == ("req_a", "req_b")
    assert C.requirements == ("req_a", "req_b")


def test_frequency_yaml() -> None:
    freq = yaml.safe_dump({"data": base.Frequency(10)})
    assert freq == "data: 10.0\n"


class MyCfg(ns.BaseModel):
    freq: float = base.Frequency(12)
    path: Path
    infra: exca.TaskInfra = exca.TaskInfra()

    @infra.apply
    def stuff(self) -> None:
        return None


def test_bad_module_exclude() -> None:
    with pytest.raises(TypeError):
        # pylint: disable=unused-variable
        class BadCfg(base._Module):
            _exclude_from_cls_uid: tuple[str, ...] = ()  # type: ignore


def test_frequency_exca() -> None:
    cfg = MyCfg(path=Path(__file__))
    string = cfg.infra.config(uid=True).to_yaml()
    assert "Frequency" not in string
    assert "Path" not in string


def test_data(test_data_path: Path) -> None:
    # List all events of a study as a dataframe
    events = ns.Study(name="Test2023Meg", path=test_data_path).run()

    # Build a list of segments time-locked to specific events
    dset = list_segments(
        events, triggers=events.type == "Image", start=-0.3, duration=0.5
    )
    assert dset

    # List all events of a study as a dataframe
    events = ns.Study(
        name="Test2023Fmri",
        path=test_data_path,
        infra_timelines={"cluster": None},
    ).run()

    # Build a list of segments time-locked to specific events
    dset = list_segments(
        events,
        triggers=events.type == "Word",
        start=-0.3,
        duration=0.5,
    )  # noqa
    assert dset


def test_strict_overlap() -> None:
    events = [
        {
            "start": i,
            "duration": 1,
            "stop": i + 1,
            "type": "Word",
            "text": "foo",
            "timeline": "bar",
        }
        for i in range(3)
    ]
    events_df = pd.DataFrame(events)  # type : ignore
    # string overlap is now always True
    segments = list_segments(
        events_df, triggers=events_df.type == "Word", start=0.0, duration=1
    )
    assert [len(s.events) for s in segments] == [1, 1, 1]


@pytest.mark.parametrize(
    "aggreg,expected", [("sum", [0, 9, 10, 10, 1]), ("mean", [0, 3, 2.5, 2.5, 1])]
)
def test_timed_array(aggreg: str, expected: list[int]) -> None:
    d = base.TimedArray(data=np.ones((2, 10)), start=10, frequency=1)
    assert aggreg in ("sum", "mean")  # for mypy
    fill = base.TimedArray(start=8, duration=12, frequency=1, aggregation=aggreg)  # type: ignore
    fill += d
    assert fill.data.shape == (2, 12)
    np.testing.assert_array_equal(fill.data[0, :5], [0, 0, 1, 1, 1])
    # more for aggregation
    d = base.TimedArray(
        data=3 * np.ones((2, 3)),
        start=9,
        frequency=1,
        aggregation=aggreg,  # type: ignore
    )
    for _ in range(3):
        fill += d
    np.testing.assert_array_equal(fill.data[0, :5], expected)


@pytest.mark.parametrize(
    "start,duration,ostart,oduration,num",
    [
        (10, 10, 10, 10, 10),  # all
        (15, 10, 15, 5, 5),  # right edge
        (5, 10, 10, 5, 5),  # left edge
        (12, 6, 12, 6, 6),  # inside
        (8, 15, 10, 10, 10),  # around
        (19.9, 10, 19, 1, 1),  # right edge
        (9.1, 1, 10, 1, 1),  # left edge
        (12, 0.1, 12, 1, 1),  # mini middle
    ],
)
def test_timed_array_overlap(
    start: float, duration: float, ostart: float, oduration: float, num: int
) -> None:
    d = base.TimedArray(data=10 + np.arange(10)[None, :], start=10, frequency=1)
    o = d.overlap(start, duration)
    assert o is not None
    assert (o.start, o.duration, o.data.shape[-1]) == (ostart, oduration, num)
    np.testing.assert_equal(o.data, np.arange(o.start, o.start + o.duration)[None, :])


def test_timed_array_load_testing() -> None:
    seed = np.random.randint(2**32 - 1)
    for k in range(100):
        print(f"Seeding with {seed + k} for reproducibility")
        rng = np.random.default_rng(seed + k)
        freq = rng.uniform(0, 50)
        if freq > 1 and rng.integers(2):
            freq = int(freq)
        content = np.ones((3, 1 + rng.integers(4)))
        f = base.TimedArray(data=content, start=rng.uniform(0, 1), frequency=freq)
        start = f.start + rng.uniform(-0.1, 0.1)
        end = f.start + f.duration + rng.uniform(-0.1, 0.1)
        duration = end - start
        if duration < 0:
            duration = rng.uniform(0.1, 0.2)
        s = base.TimedArray(duration=duration, start=start, frequency=freq)
        s += f
        if f.start + f.duration < s.start or s.start + s.duration < f.start:
            assert s._overlap_slice(f.start, f.duration) is None
            assert not np.any(s.data)
        else:
            assert np.any(s.data)


def test_timed_array_case() -> None:
    out = base.TimedArray(frequency=3, start=0, duration=1.0)
    d = np.array([[0.5, 1.0, 1.5]])
    content = base.TimedArray(frequency=3.0, start=0.5, duration=1.0, data=d)
    out += content


def test_timed_array_no_freq_no_overlap() -> None:
    out = base.TimedArray(frequency=0, start=0, duration=1.0)
    d = np.array([[0.5, 1.0, 1.5]])
    content = base.TimedArray(frequency=0, start=2, duration=1.0, data=d)
    out += content
    norm = float(sum(abs(out.data.ravel())))
    assert not norm


@pytest.mark.parametrize(
    "aggreg,expected",
    [
        ("mean", [0.5, 1.0]),
        ("sum", [1.0, 2.0]),
    ],
)
def test_no_freq(aggreg: str, expected: list[float]) -> None:
    out = base.TimedArray(frequency=3, start=0, duration=1.0)
    d = np.array([0.5, 1.0])
    content = base.TimedArray(frequency=0, start=0.5, duration=1.0, data=d)
    out += content
    assert out.data.shape == (2, 3)
    np.testing.assert_array_equal(out.data, [[0.0, 0.5, 0.5], [0.0, 1.0, 1.0]])
    with pytest.raises(ValueError):
        content += out
    assert aggreg in ("sum", "mean")  # for mypy
    out = base.TimedArray(start=0, frequency=0, duration=3, aggregation=aggreg)  # type: ignore
    for _ in range(2):
        out += base.TimedArray(frequency=0, start=0.5, duration=1.0, data=d)
    np.testing.assert_array_equal(out.data, expected)


@pytest.mark.parametrize("out_freq", (0, 10))
@pytest.mark.parametrize("in_freq", (0, 10))
def test_no_duration(in_freq: float, out_freq: float) -> None:
    out = base.TimedArray(frequency=out_freq, start=0, duration=1.0)
    d = np.array([0.5, 1.0])
    if in_freq:
        d = d[..., None]
    content = base.TimedArray(frequency=0, start=0.5, duration=0, data=d)
    out += content
    norm1 = float(sum(abs(out.data.ravel())))
    assert norm1


# TODO: find a way to enable this use case where freq is approximate
# def test_approx_freq() -> None:
#     tarrays = []
#     freq = 10
#     for dur in [10, 100]:
#         data = np.random.rand(4, int(freq * dur - 1))
#         dfreq = data.shape[-1] / dur
#         tarrays.append(base.TimedArray(frequency=dfreq, start=0, duration=dur, data=data))
#     tarrays[0] += tarrays[1]
#     tarrays[1] += tarrays[0]


def test_approx_freq_brainvision() -> None:
    tarrays = []
    d = np.random.rand(4, 100)
    for freq in [120, 119.5]:
        tarrays.append(base.TimedArray(frequency=freq, start=0, data=d))
    tarrays[0] += tarrays[1]
    tarrays[1] += tarrays[0]


def test_overlap_no_duration() -> None:
    d = np.random.rand(4, 100)
    ta = base.TimedArray(frequency=100, start=0, data=d)
    sub = ta.overlap(0, 0)
    np.testing.assert_array_equal(sub.data, d[:, :1])


def test_overlap_with_no_freq() -> None:
    ta = base.TimedArray(start=0, frequency=0, duration=0.001, data=np.ones((10, 1)))
    with pytest.raises(RuntimeError):
        ta.overlap(0, 1)


def _timed_array_dump_roundtrip(tmp_path: Path, ta: base.TimedArray) -> base.TimedArray:
    from exca.cachedict import DumpContext

    ctx = DumpContext(tmp_path)
    with ctx:
        info = ctx.dump_entry("test_key", ta)
    return ctx.load(info["content"])


@pytest.mark.parametrize(
    "shape,frequency",
    [
        ((10,), 1.0),
        ((768, 406), 2.0),
        ((4, 8, 20), 0.5),
        ((257, 768, 406), 1.0),
        ((2,), 0),  # no time dimension
    ],
)
def test_timed_array_dump(
    tmp_path: Path, shape: tuple[int, ...], frequency: float
) -> None:
    data = np.random.rand(*shape).astype(np.float32)
    meta = {"ch_names": ["E1"], "space": "MNI152"}
    ta = base.TimedArray(
        data=data,
        frequency=frequency,
        start=3.5,
        duration=2.0 if not frequency else None,
        header=meta,
    )
    loaded = _timed_array_dump_roundtrip(tmp_path, ta)
    np.testing.assert_array_equal(loaded.data, ta.data)
    for attr in ("frequency", "start", "duration", "header"):
        assert getattr(loaded, attr) == getattr(ta, attr), attr
    # data must be backed by ContiguousMemmap (file-IO proxy over memmap)
    assert isinstance(loaded.data, contiguous.ContiguousMemmap)


def test_timed_array_iadd_contiguous_memmap(tmp_path: Path) -> None:
    """__iadd__ must work when self and/or other data is ContiguousMemmap."""
    data = np.ones((2, 10), dtype=np.float32)
    mm = _timed_array_dump_roundtrip(
        tmp_path, base.TimedArray(data=data, start=0, frequency=1)
    )
    assert isinstance(mm.data, contiguous.ContiguousMemmap)
    target = mm.with_start(0)
    target += mm
    np.testing.assert_array_equal(target.data, 2 * data)


def test_mne_timed_array_roundtrip(tmp_path: Path) -> None:
    """MneTimedArray must survive a dump/load cycle as MneTimedArray, not TimedArray."""
    ta = neuro.MneTimedArray(
        data=np.random.rand(3, 100).astype(np.float32),
        frequency=100.0,
        start=0.0,
        header={"ch_names": "a,b,c", "ch_types": "eeg,eeg,eeg"},
    )
    loaded = _timed_array_dump_roundtrip(tmp_path, ta)
    actual = type(loaded).__name__
    assert type(loaded) is neuro.MneTimedArray, f"expected MneTimedArray, got {actual}"
    assert loaded.ch_names == ["a", "b", "c"]


def test_timed_array_header_in_overlap() -> None:
    meta = {"ch_names": ["E1", "E2"]}
    data = np.random.rand(2, 10).astype(np.float32)
    ta = base.TimedArray(data=data, frequency=1.0, start=0.0, header=meta)
    sub = ta.overlap(2.0, 3.0)
    assert sub.header is meta


def test_unset_start() -> None:
    data = np.random.rand(3, 10).astype(np.float32)
    meta = {"ch_names": "a,b,c"}
    unset = base.TimedArray(
        data=data, frequency=5.0, start=base._UNSET_START, duration=2.0, header=meta
    )
    assert unset.start == base._UNSET_START
    # with_start resolves the sentinel and shares data/header
    resolved = unset.with_start(1.5)
    assert resolved.start == 1.5
    assert resolved.data is unset.data
    assert resolved.header is unset.header
    # overlap and iadd both reject unset start
    with pytest.raises(RuntimeError, match="unset start"):
        unset.overlap(0.0, 1.0)
    good = base.TimedArray(data=data, frequency=5.0, start=0.0, duration=2.0)
    target = base.TimedArray(
        data=np.zeros_like(data), frequency=5.0, start=0.0, duration=2.0
    )
    with pytest.raises(RuntimeError, match="unset start"):
        target += unset
    with pytest.raises(RuntimeError, match="unset start"):
        unset += good
