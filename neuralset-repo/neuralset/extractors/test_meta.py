# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import numpy as np
import pytest
import torch

import neuralset as ns
from neuralset.events import etypes

from . import meta


@pytest.mark.parametrize("time_aggregation", ["mean", "sum", "first", "last"])
@pytest.mark.parametrize("n_groups_concat", [None, 4])
def test_time_aggregated_extractor(
    test_data_path: Path,
    time_aggregation: tp.Literal["mean", "sum", "first", "last"],
    n_groups_concat: int | None,
) -> None:
    events = ns.Study(
        name="Test2023Meg", path=test_data_path, query="timeline_index<2"
    ).run()

    meg = ns.extractors.MegExtractor(frequency=100.0)
    aggregated_meg = ns.extractors.TimeAggregatedExtractor(
        extractor=meg,
        time_aggregation=time_aggregation,
        n_groups_concat=n_groups_concat,
    )
    meg.prepare(events)

    extractors = {"meg": meg, "agg": aggregated_meg}
    segments = ns.segments.list_segments(
        events,
        triggers=events.type == "Image",
        start=0.0,
        duration=3.0,
    )
    dataset = ns.SegmentDataset(extractors, segments)
    out = dataset.load_all().data

    assert out["meg"].ndim == 3
    assert out["agg"].ndim == 2
    assert out["agg"].shape[0] == out["meg"].shape[0]
    if n_groups_concat is None:
        assert out["agg"].shape[1] == out["meg"].shape[1]
        if time_aggregation == "mean":
            assert torch.allclose(out["agg"], out["meg"].mean(-1))
        elif time_aggregation == "first":
            assert torch.allclose(out["agg"], out["meg"][..., 0])
    else:
        assert out["agg"].shape[1] == out["meg"].shape[1] * n_groups_concat


@pytest.mark.parametrize("agg_method", ["mean", "sum", "cat", "stack"])
def test_aggregated_extractor(
    test_data_path: Path, agg_method: tp.Literal["mean", "sum", "cat", "stack"]
) -> None:
    events = ns.Study(
        name="Test2023Meg", path=test_data_path, query="timeline_index<2"
    ).run()

    meg = ns.extractors.MegExtractor(frequency=100.0, aggregation="single")
    meg.prepare(events)

    agg_extractor = ns.extractors.AggregatedExtractor(
        extractors=[meg, meg], extractor_aggregation=agg_method
    )
    extractors = {"meg": meg, "agg": agg_extractor}
    segments = ns.segments.list_segments(
        events,
        triggers=events.type == "Image",
        start=0.0,
        duration=3.0,
    )
    dataset = ns.SegmentDataset(extractors, segments)
    out = dataset.load_all().data

    if agg_method in ["mean", "sum", "cat"]:
        assert out["agg"].ndim == 3
    elif agg_method == "stack":
        assert out["agg"].ndim == 4
    assert out["agg"].shape[0] == out["meg"].shape[0]
    assert out["agg"].shape[-1] == out["meg"].shape[-1]

    if agg_method == "cat":
        assert out["agg"].shape[1] == 2 * out["meg"].shape[1]
    elif agg_method == "mean":
        assert out["agg"].shape[1] == out["meg"].shape[1]
        assert torch.allclose(out["agg"], out["meg"])
    elif agg_method == "sum":
        assert out["agg"].shape[1] == out["meg"].shape[1]
        assert torch.allclose(out["agg"], out["meg"] * 2)
    elif agg_method == "stack":
        assert out["agg"].shape[1] == 2
        assert out["agg"].shape[2] == out["meg"].shape[1]
        assert out["agg"].shape[3] == out["meg"].shape[2]
        assert torch.allclose(out["agg"][:, 0], out["meg"])

    f1 = ns.extractors.WordFrequency(event_types="Word", aggregation="first")
    f2 = ns.extractors.WordFrequency(event_types="Word", aggregation="last")
    f1.prepare(events)
    f2.prepare(events)
    agg = ns.extractors.AggregatedExtractor(
        extractors=[f1, f2], extractor_aggregation=agg_method, aggregation="trigger"
    )
    extractors = {"agg": agg}
    segments = ns.segments.list_segments(
        events,
        triggers=events.type == "Word",
        start=0.0,
        duration=3.0,
    )
    dataset = ns.SegmentDataset(extractors, segments)
    out = dataset.load_all().data

    batch_size = len(segments)
    if agg_method in ["mean", "sum"]:
        assert out["agg"].shape == (batch_size, 1)
    elif agg_method == "cat":
        assert out["agg"].shape == (batch_size, 2)
    else:
        assert out["agg"].shape == (batch_size, 2, 1)


def test_extractor_pca(tmp_path: Path) -> None:
    infra: tp.Any = {"infra": {"folder": tmp_path}}
    fparams: tp.Any = {"name": "HuggingFaceText", "cache_n_layers": 3, **infra}
    feat = meta.ExtractorPCA(extractor=fparams, n_components=2)  # infra folder is copied
    events: list[tp.Any] = []
    for w in ["Hello", "world", "world", "blublu", "whatever"]:
        events.append(etypes.Word(start=0, duration=1, timeline="t", text=w, context=w))
    feat.prepare(events)
    np.random.shuffle(events)
    events.append(etypes.Image(start=0, duration=1, filepath=__file__, timeline="t"))
    feat = meta.ExtractorPCA(extractor=fparams, n_components=2)
    assert len(feat.infra.cache_dict) == 4
    feat.prepare(events)
    assert len(feat.infra.cache_dict) == 4
    out = feat(events[0], 0, 1)
    assert out.shape == (2,)
    # now on a different dataset
    feat = meta.ExtractorPCA(extractor=fparams, n_components=2)
    feat.prepare(events[:3])
    assert len(feat.infra.cache_dict) in (6, 7)  # depends if "world" was repeated
    out = feat(events[0], 0, 1)
    assert out.shape == (2,)


def test_hf_pca(tmp_path: Path) -> None:
    fparams: tp.Any = {
        "name": "HuggingFaceText",
        "cache_n_layers": 3,
        "contextualized": True,
        "infra": {"folder": tmp_path},
    }
    feat = meta.HuggingFacePCA(extractor=fparams, n_components=2)
    events = []
    kwargs: tp.Any = dict(start=0, duration=1, timeline="t")
    for w in ["Hello", "world", "blublu", "whatever"]:
        c = w if w != "Hello" else ("blublu " * 64 + w)  # trigger key shortening
        events.append(etypes.Word(text=w, context=c, **kwargs))
    feat.prepare(events)
    folder = feat.infra.uid_folder()
    assert folder is not None
    np.random.shuffle(events)  # type: ignore
    fparams["layers"] = 0.1
    feat = meta.HuggingFacePCA(extractor=fparams, n_components=2)
    feat.prepare(events)
    assert feat.infra.uid_folder() == folder  # should be unchanged
    out = feat(events[0], 0, 1)
    assert out.shape == (2,)
    # check tmp cache folder is removed
    dirs = [type(f).__name__ for f in tmp_path.iterdir()]
    assert len(dirs) == 1
    dirs = [f.name for f in folder.parent.iterdir()]
    assert len(dirs) == 1


@pytest.mark.parametrize("offset", [0, 1.0])
@pytest.mark.parametrize("duration", [2.0, None])
def test_crop_extractor(
    test_data_path: Path, offset: float, duration: float | None
) -> None:
    events = ns.Study(
        name="Test2023Meg", path=test_data_path, query="timeline_index<2"
    ).run()
    meg_params: tp.Any = {
        "name": "MegExtractor",
        "frequency": 100.0,
        "aggregation": "single",
    }
    meg = ns.extractors.MegExtractor(**meg_params)
    meg.prepare(events)

    crop_extractor = ns.extractors.CroppedExtractor(
        extractor=meg_params, offset=offset, duration=duration
    )
    crop_extractor.prepare(events)

    extractors = {"meg": meg, "crop": crop_extractor}
    segments = ns.segments.list_segments(
        events,
        triggers=events.type == "Image",
        start=0.0,
        duration=3.0,
    )
    dataset = ns.SegmentDataset(extractors, segments)
    out = dataset.load_all().data

    if duration is not None:
        assert out["crop"].size(-1) == duration * meg_params["frequency"]

    elif offset != 0:
        assert out["crop"].size(-1) == (3.0 - offset) * meg_params["frequency"]

        # test that the crop matches the meg at the right offset
        ind = int(offset * meg_params["frequency"])
        assert torch.all(out["crop"][:, :, 0] == out["meg"][:, :, ind])

    with pytest.raises(ValueError):
        out = crop_extractor(events, start=0.0, duration=0.5)  # type: ignore


def test_crop_extractor_static() -> None:
    events = []
    for k, w in enumerate(["Hello", "world"]):
        e = etypes.Word(start=k, duration=1, timeline="t", text=w, context=w)
        events.append(e)
    extractor = ns.extractors.LabelEncoder(
        event_types="Word", return_one_hot=True, event_field="text", aggregation="sum"
    )
    extractor.prepare(events)
    params: tp.Any = {"start": 0, "duration": 2, "events": events}
    out = extractor(**params)
    assert tuple(out) == (1, 1)
    cropped = ns.extractors.CroppedExtractor(
        extractor=extractor, offset=1.5, duration=0.1
    )
    out = cropped(**params)
    assert tuple(out) == (0, 1)


def test_tostatic(tmp_path: Path):
    from neuralset.extractors.test_audio import create_wav

    wavfile = tmp_path / "noise.wav"
    create_wav(wavfile, fs=44100, duration=10)
    event = etypes.Audio(start=1, timeline="x", filepath=wavfile)
    mel: tp.Any = dict(name="MelSpectrum")

    feat = ns.extractors.meta.ToStatic(extractor=mel)
    trigger = etypes.Word(start=6.0, duration=0.01, timeline="x", text="y")
    out = feat([event], start=5.0, duration=2.0, trigger=trigger)
    assert out.ndim == 1

    # outside segment duration
    with pytest.raises(RuntimeError):
        feat([event], start=5.0, duration=2.0, trigger=None)  # type: ignore
    for start in 4.0, 8.0:
        trigger = etypes.Word(start=start, duration=0.01, timeline="x", text="y")
        with pytest.raises(RuntimeError):
            feat([event], start=5.0, duration=2.0, trigger=trigger)

    # trigger aggregation
    with pytest.raises(RuntimeError):
        mel_: tp.Any = dict(name="MelSpectrum", aggregation="trigger")
        feat = ns.extractors.meta.ToStatic(extractor=mel_)

    # StaticExtractor
    with pytest.raises(ValueError):
        _mel: tp.Any = dict(name="WordFrequency")
        feat = ns.extractors.meta.ToStatic(extractor=_mel)
