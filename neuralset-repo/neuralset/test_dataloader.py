# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import neuralset as ns

from . import dataloader as dl
from . import events
from .segments import list_segments
from .test_segments import _make_segments


def test_batch_and_collate() -> None:
    segments = _make_segments()
    batches = []
    for k in range(3):
        b = dl.Batch(
            data={"feat": 10 * torch.Tensor([[k, k, k, k]])}, segments=[segments[0]]
        )
        batches.append(b)
    assert batches[0].data["feat"].shape == (1, 4)
    assert len(batches[0].segments) == 1
    collate_fn = dl.SegmentDataset({}, []).collate_fn
    batch = collate_fn(batches)
    assert len(batch.segments) == 3
    np.testing.assert_array_equal(batch.data["feat"][:, 0], [0, 10, 20])
    assert batch.data["feat"].shape == (3, 4)
    # sequential
    batch = collate_fn(batches[:2])
    batch = collate_fn([batch, batches[2].to("cpu")])
    np.testing.assert_array_equal(batch.data["feat"][:, 0], [0, 10, 20])
    assert len(batch.segments) == 3


def test_dataset() -> None:
    # A study is just a dataframe of events
    segments = _make_segments()
    pulse = ns.extractors.Pulse(frequency=100.0, event_types="Word")
    extractors: tp.Any = {"single": pulse}
    ds = dl.SegmentDataset(extractors, segments)
    dataloader = DataLoader(ds, collate_fn=ds.collate_fn, batch_size=2)
    batch = next(iter(dataloader))
    assert batch.data["single"].shape == (2, 1, 100)
    # as one batch
    full_batch = ds.load_all()
    assert full_batch.data["single"].shape == (2, 1, 100)
    # slicing
    batch = ds[:2]
    assert len(batch) == 2
    with pytest.raises(ValueError):
        batch = ds[600_000:]


@pytest.mark.sandbox_skip
def test_load_all_order() -> None:
    data = [
        dict(timeline=str(k), start=k, duration=0.1, type="Stimulus", code=k)
        for k in range(130)
    ]
    df = ns.events.standardize_events(pd.DataFrame(data))
    stim = ns.extractors.EventField(event_types="Stimulus", event_field="code")
    stim.prepare(df)
    segments = list_segments(df, triggers=df.type == "Stimulus", start=0.0, duration=0.1)
    extractors = {"stim1": stim, "stim2": stim}
    ds = dl.SegmentDataset(extractors, segments).load_all(num_workers=8)
    np.testing.assert_array_equal(ds.data["stim1"], ds.data["stim2"])
    np.testing.assert_array_equal(ds.data["stim1"][:, 0], np.arange(len(df)))


def test_padded_collate_dataset() -> None:
    # A study is just a dataframe of events
    segments = _make_segments()
    segments[1].duration = 2.0
    extractors = {"Pulse": ns.extractors.Pulse(frequency=100.0, event_types="Word")}
    ds = dl.SegmentDataset(extractors, segments, pad_duration=None)
    with pytest.raises(ValueError):
        ds.load_all()
    with pytest.raises(ValueError):
        ds.build_dataloader()

    for pad_duration in ["auto", 4.0]:
        ds = dl.SegmentDataset(extractors, segments, pad_duration=pad_duration)  # type: ignore
        dataloader = DataLoader(ds, collate_fn=ds.collate_fn, batch_size=2)
        batch = next(iter(dataloader))
        assert batch.data["Pulse"].shape == (2, 1, 200 if pad_duration == "auto" else 400)


def test_select() -> None:
    segments = _make_segments()
    extractors = {"Pulse": ns.extractors.Pulse(frequency=100.0, event_types="Word")}
    ds = dl.SegmentDataset(extractors, segments)

    batch = ds.load_all()

    for segment_like in (ds, batch):
        # check triggers and events
        assert isinstance(segment_like.triggers, pd.DataFrame)
        assert isinstance(segment_like.events, pd.DataFrame)
        assert len(segment_like.triggers) == len(segment_like)
        assert len(segment_like.events) != len(segment_like)

        # check size
        assert len(segment_like) == 2
        np.testing.assert_array_equal(segment_like.triggers.index, [1, 2])

        # select by int
        segment_like_ = segment_like.select(0)
        assert isinstance(segment_like_, segment_like.__class__)
        assert len(segment_like_) == 1

        # select by int list
        segment_like_ = segment_like.select([0])
        assert isinstance(segment_like_, segment_like.__class__)

        # select by bool mask (pd.Series)
        segment_like_ = segment_like.select(segment_like.triggers.start > 11.0)
        assert all(segment_like_.triggers.start > 11.0)
        np.testing.assert_array_equal(segment_like_.triggers.index, [2])

        # select by np.ndarray[int]
        segment_like_ = segment_like.select(np.array([1]))
        assert len(segment_like_) == 1

        # select by np.ndarray[bool]
        mask = np.array([False, True])
        segment_like_ = segment_like.select(mask)
        assert len(segment_like_) == 1

    # check when the trigger is outside of the segment
    segments = _make_segments(start=2.0)
    extractors = {
        "Pulse": ns.extractors.Pulse(
            frequency=100.0, event_types="Word", aggregation="sum"
        )
    }
    ds = dl.SegmentDataset(extractors, segments, remove_incomplete_segments=True)

    batch = ds.load_all()

    for segment_like in (ds, batch):
        assert len(segment_like.triggers) == len(segment_like)
        assert len(segment_like.events) != len(segment_like)
        assert len(segment_like) == 1

        assert not any(segment_like.triggers.index.isin(segment_like.events.index))


# # # # # EXAMPLE # # # # #


class ExampleDataLoader(ns.BaseModel):
    studies: tp.List[ns.Step]
    segmenter: dl.Segmenter
    batch_size: int = 1
    num_workers: int = 0

    def build(self) -> DataLoader[dl.Batch]:
        all_studies = pd.concat([s.run() for s in self.studies])
        ds = self.segmenter.apply(all_studies)

        ds.prepare()
        return torch.utils.data.DataLoader(
            ds,
            collate_fn=ds.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


def test_example_dataloader(test_data_path: Path) -> None:
    dloader = ExampleDataLoader(
        studies=[{"name": "Test2023Meg", "path": test_data_path}],  # type: ignore
        segmenter={  # type: ignore
            "trigger_query": "type=='Image'",
            "duration": 1.0,
            "extractors": {
                "whatever": {"name": "Pulse", "frequency": 100, "aggregation": "sum"}
            },
        },
        batch_size=2,
    )
    b = next(iter(dloader.build()))
    assert len(b.segments) == 2
    assert b.data["whatever"].shape == (2, 1, 100)


def test_empty_set(test_data_path: Path) -> None:
    # test empty dataset
    studies = [{"name": "Test2023Meg", "path": test_data_path}]
    config = dict(
        studies=studies,
        segmenter={
            "trigger_query": "type=='Foo'",
            "duration": 1.0,
            "extractors": {
                "whatever": {"name": "Pulse", "frequency": 100, "aggregation": "sum"}
            },
        },
        batch_size=1,
    )

    # empty trigger query
    dloader = ExampleDataLoader(**config)  # type: ignore
    with pytest.raises(RuntimeError):
        dloader.build()

    # empty trigger query: pulse on Image but time lock to Meg
    extractor = {
        "whatever": {
            "name": "Pulse",
            "frequency": 100,
            "aggregation": "sum",
            "event_types": "Image",
            "allow_missing": True,
        }
    }
    segmenter = {
        "trigger_query": "type=='Meg'",
        "duration": 0.5,  # no Image then
        "extractors": extractor,
        "drop_incomplete": False,
    }
    # -- without drop, should pass
    dloader = ExampleDataLoader(studies=studies, segmenter=segmenter, batch_size=1)  # type: ignore
    dloader.build()

    # -- with drop, empty set should raise
    segmenter["drop_incomplete"] = True
    dloader = ExampleDataLoader(studies=studies, segmenter=segmenter, batch_size=1)  # type: ignore
    with pytest.raises(RuntimeError):
        dloader.build()


def test_drop_useless_events(test_data_path: Path) -> None:
    config = dict(
        studies=[{"name": "Test2023Meg", "path": test_data_path}],
        segmenter={
            "drop_unused_events": False,
            "trigger_query": "type=='Image'",
            "drop_incomplete": False,
            "duration": 1.0,
            "extractors": {
                "whatever": {
                    "name": "Pulse",
                    "frequency": 100,
                    "aggregation": "sum",
                    "event_types": "Image",
                    "allow_missing": True,
                }
            },
        },
        batch_size=1,
    )
    # don't drop unused events
    dloader = ExampleDataLoader(**config)  # type: ignore
    batch = next(iter(dloader.build()))
    assert "Meg" in batch.events.type.unique()  # even though Meg not used

    # drop
    config["segmenter"]["drop_unused_events"] = True  # type: ignore
    dloader = ExampleDataLoader(**config)  # type: ignore
    batch = next(iter(dloader.build()))
    assert "Meg" not in batch.events.type.unique()

    # event not in extractor, but in trigger: pulse never populated
    config["segmenter"]["drop_unused_events"] = False  # type: ignore[index]
    config["segmenter"]["trigger_query"] = "type=='Meg'"  # type: ignore[index]
    dloader = ExampleDataLoader(**config)  # type: ignore
    dset = dloader.build()
    with pytest.raises(ValueError):
        batch = next(iter(dset))

    #
    config["segmenter"]["extractors"]["whatever"]["event_types"] = "Meg"  # type: ignore[index]
    dloader = ExampleDataLoader(**config)  # type: ignore
    dset = dloader.build()
    batch = next(iter(dset))
    assert np.array_equal(["Meg"], batch.events.type.unique())


def test_segment_dataset_with_transforms() -> None:
    segments = _make_segments()
    pulse = ns.extractors.Pulse(frequency=100.0, event_types="Word")
    feats: tp.Any = {"transformed": pulse, "untransformed": pulse}
    ds = dl.SegmentDataset(
        feats,
        segments,
        transforms={"transformed": lambda a: a * 2},
    )
    dataloader: DataLoader = DataLoader(ds, collate_fn=ds.collate_fn, batch_size=2)
    batch = next(iter(dataloader))
    assert batch.data["transformed"].shape == (2, 1, 100)
    assert batch.data["untransformed"].shape == (2, 1, 100)
    assert torch.all(batch.data["transformed"] == batch.data["untransformed"][:, :1] * 2)


def test_repr(tmp_path: Path):
    segmenter = dl.Segmenter(
        trigger_query="type=='Image'",
        duration=1.0,
        extractors={"meg": {"name": "MegExtractor", "frequency": 100.0}},  # type: ignore
    )
    string = repr(segmenter)
    assert "  'name': '" in string


def test_repr_segments() -> None:
    segments = _make_segments()
    sd = dl.Batch(data={"meg": torch.randn(1, 3, 100)}, segments=[segments[0]])
    assert repr(sd) == "Batch(1 segments, 1.00s, shapes: meg: 1, 3, 100)"
    ds = dl.SegmentDataset(extractors={}, segments=_make_segments())
    assert repr(ds) == "SegmentDataset(2 segments, 1.00s, extractors: none)"


def test_segment_data_no_leak(tmp_path: Path) -> None:
    """Batch and numpy array counts must not grow across epochs.

    Note: this does NOT detect memmap page-in (RSS growth from MemmapArray,
    +10 GB/epoch with ~500 cached recordings, invisible to object counts).
    See the commented-out RSS snippet below and memmap-consolidated-report.md.
    """
    sfreq, duration_s, n_ch = 100.0, 10.0, 5
    mne_info = mne.create_info([f"MEG {j:03d}" for j in range(n_ch)], sfreq, "mag")
    meg_rows = []
    for i in range(2):
        fif = tmp_path / f"sub-{i}-raw.fif"
        data = np.zeros((n_ch, int(sfreq * duration_s)), dtype=np.float32)
        mne.io.RawArray(data, mne_info, verbose=False).save(
            fif, overwrite=True, verbose=False
        )
        meg_rows.append(
            events.etypes.Meg(
                start=0,
                duration=duration_s,
                frequency=sfreq,
                timeline=f"sub-{i}",
                filepath=str(fif),
                subject=str(i),
            ).to_dict()
        )
    events_df = events.standardize_events(pd.DataFrame(meg_rows))
    segmenter = dl.Segmenter(
        extractors={
            "meg": {  # type: ignore
                "name": "MegExtractor",
                "frequency": "native",
                "infra": {"folder": str(tmp_path), "cluster": None},
            }
        },
        trigger_query="type == 'Meg'",
        start=0,
        duration=5.0,
        stride=5.0,
    )
    ds = segmenter.apply(events_df)
    ds.prepare()
    loader = ds.build_dataloader(batch_size=2, num_workers=0)
    gc.collect()
    count = lambda tp_: sum(1 for o in gc.get_objects() if type(o) is tp_)
    sd_before, np_before = count(dl.Batch), count(np.ndarray)
    for _ in range(3):
        for batch in loader:
            pass
    del batch
    gc.collect()
    sd_leak = count(dl.Batch) - sd_before
    np_leak = count(np.ndarray) - np_before
    assert sd_leak < 5, f"Batch leak: {sd_leak} instances after 3 epochs"
    assert np_leak < 10, f"numpy array leak: {np_leak} new arrays after 3 epochs"
    # Memmap page-in is not caught by object counts — track RSS instead:
    # import resource
    # rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024)
    # rss0 = rss(); [None for _ in range(3) for _ in loader]
    # print(f"RSS delta: +{rss() - rss0} MB")  # increase n_ch/duration_s to see effect
