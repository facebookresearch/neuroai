# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import pickle
import time
import typing as tp
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest
import requests
import sklearn
import sklearn.preprocessing
import torch
import yaml
from exca import ConfDict

import neuralset as ns
from neuralset.base import TimedArray
from neuralset.events import etypes, test_etypes
from neuralset.events.utils import extract_events
from neuralset.extractors.neuro import FmriTimedArray, _overlap

# avoid processpool which requires permissions unavailable in sandboxes
_NO_CLUSTER: tp.Any = {"cluster": None}


@pytest.mark.parametrize("cls", (ns.extractors.MegExtractor, ns.extractors.FmriExtractor))
def test_neuro_pkl(cls: tp.Type[ns.extractors.BaseExtractor]) -> None:
    inst = cls()
    string = pickle.dumps(inst)
    _ = pickle.loads(string)


def make_meg_event(filepath, start=0.0):
    timeline = "foo"
    if start == 0:
        timeline = 'Test2023Meg:{"subject":"0"}'
        filepath = '{"cls":"Test2023Meg","method":"_load_raw","timeline":{"subject":"0"}}'
    return dict(
        start=start,
        duration=99.99,
        timeline=timeline,
        filepath=filepath,
        type="Meg",
        subject="janedoe",
    )


def test_fmri(test_data_path: Path, tmp_path: Path) -> None:
    # load study to create nii.gz files
    infra: tp.Any = {"folder": tmp_path / "cache", **_NO_CLUSTER}
    fmri_data = test_data_path / "Test2023Fmri"
    study = ns.Study(
        name="Test2023Fmri",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=infra,
    ).run()

    timeline = 'Test2023Fmri:{"subject":"0"}'
    fp = '{"cls":"Test2023Fmri","method":"_load_raw","timeline":{"subject":"0"}}'
    event_manual = dict(
        start=0,
        duration=20.0,
        timeline=timeline,
        filepath=fp,
        type="Fmri",
        space="custom",
        subject="janedoe",
        frequency=0.5,
    )
    extractor = ns.extractors.FmriExtractor()

    event_auto = study.query('type=="Fmri"').iloc[:1]
    event: tp.Any
    for event in (event_manual, event_auto):
        data = extractor(event, start=0.0, duration=2.0).numpy()
        assert extractor.frequency == "native", "Should be unmodified"
        assert np.array_equal(data.shape, [20, 20, 20, 1])
        assert data.max() > 0

    # fsaverage
    projection: tp.Any = {
        "name": "SurfaceProjector",
        "mesh": "fsaverage5",
        "kind": "ball",
    }
    feature2 = ns.extractors.FmriExtractor(
        projection=projection,
        infra=infra,
    )
    data = feature2(event, start=0.0, duration=20).numpy()
    assert np.array_equal(data.shape, [20484, 10])
    # stds should be 0 (no data) or close to 1 (reduced):
    stds = np.unique(np.std(data, axis=1))
    np.testing.assert_array_almost_equal(stds, [0] + [0.948] * (len(stds) - 1), decimal=3)
    # check that cached value is an FmriTimedArray with projected space:
    val = next(iter((feature2.infra.cache_dict._ram_data.values())))
    assert isinstance(val, FmriTimedArray)
    assert val.header["space"] == "fsaverage5"
    assert "affine" not in val.header
    with pytest.raises(ValueError, match="No affine"):
        val.to_native()
    # padding
    feature3 = feature2.infra.clone_obj(padding=20500)
    t = feature3(event, start=0.0, duration=20)
    assert np.array_equal(t.shape, [20500, 10])
    feature3 = feature2.infra.clone_obj(padding="auto", allow_missing=True)
    feature3.prepare(study)
    t = feature3([], start=0.0, duration=20)
    assert np.array_equal(t.shape, [20484, 10])

    # atlas
    try:
        for name, kwargs in [
            ("difumo", {"dimension": 64}),
            ("schaefer_2018", {"n_rois": 100}),
        ]:
            projection = {
                "name": "AtlasProjector",
                "atlas": name,
                "atlas_kwargs": kwargs,
            }
            feature3 = ns.extractors.FmriExtractor(
                projection=projection,
                infra=infra,
            )
            data = feature3(event, start=0.0, duration=20).numpy()
            np.array_equal(data.shape, [64 if name == "difumo" else 100, 10])
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectTimeout, OSError):
        pass  # unstable (got internal server errors)

    # MNI space
    projection = {"name": "MaskProjector", "mask": "mni152_gm", "resolution": 2}
    feature3 = ns.extractors.FmriExtractor(
        projection=projection,
    )
    data = feature3(event, start=0.0, duration=20).numpy()
    assert np.array_equal(data.shape, [204492, 10])
    # test from file
    nii = fmri_data / "sub-0.nii.gz"
    assert nii.exists()
    event["timeline"] = "foo"
    event["filepath"] = nii
    cached_extractor = ns.extractors.FmriExtractor(infra=infra)  # type: ignore
    tdata = cached_extractor(event, start=0.0, duration=2.0)
    assert tdata.shape == torch.Size([20, 20, 20, 1])
    assert tdata.max() > 0
    # cache UID must be study-relative, not an absolute path
    assert list(cached_extractor.infra.cache_dict) == [
        "Test2023Fmri/sub-0.nii.gz_0.000_20.000"
    ]
    # no-projection volumetric: header has affine, to_native works
    cached_val = next(iter(cached_extractor.infra.cache_dict._ram_data.values()))
    assert isinstance(cached_val, FmriTimedArray)
    assert "affine" in cached_val.header
    nifti = cached_val.to_native()
    assert nifti.shape == cached_val.data.shape


@pytest.mark.parametrize("frequency", [0.25, 0.33, 0.5, 1.0, 1.33, 10.01])
def test_fmri_resample(test_data_path: Path, frequency: float) -> None:
    study = ns.Study(
        name="Test2023Fmri",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()
    event = study.query('type=="Fmri"').iloc[:1]
    orig_frequency = event.frequency.item()

    duration = 20.0

    # No resampling
    extractor = ns.extractors.FmriExtractor(frequency="native")
    data = extractor(event, start=0.0, duration=duration).numpy()
    assert extractor.frequency == "native"
    assert data.shape[3] == int(duration * orig_frequency)

    # With resampling
    resamp_extractor = ns.extractors.FmriExtractor(frequency=frequency)
    resamp_data = resamp_extractor(event, start=0.0, duration=duration).numpy()
    assert resamp_extractor.frequency == frequency
    assert resamp_data.shape[:3] == data.shape[:3]
    assert resamp_data.shape[3] == np.round(duration * frequency)

    # Check no NaN values appear (test for right=np.nan bug)
    assert not np.isnan(resamp_data).any(), f"Found NaN values with frequency={frequency}"


def test_meg(test_data_path: Path, tmp_path: Path) -> None:
    cache_path = tmp_path / "cache"
    meg_data = test_data_path / "Test2023Meg"
    # test from study
    extractor = ns.extractors.MegExtractor(filter=(100, None))
    v = extractor.infra.version
    assert v == "1"
    uid = f"neuralset.extractors.neuro.MegExtractor._get_data,{v}/"
    uid += "filter=(100,None),name=MegExtractor-cea250f4"
    assert extractor.infra.uid() == uid
    assert extractor.frequency == "native"
    study = ns.Study(
        name="Test2023Meg",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()

    # check caching infra
    infra: tp.Any = {"folder": cache_path}
    extractor = ns.extractors.MegExtractor(frequency=100, infra=infra)  # type: ignore
    meg_event = study.query('type=="Meg"').iloc[:1]
    extractor.prepare(meg_event)
    infra["mode"] = "read-only"
    post_param: tp.Any
    for post_param in [{"baseline": (0, 0.2), "scale_factor": 2}]:
        ext = ns.extractors.MegExtractor(frequency=100, **post_param, infra=infra)
        out = ext(meg_event, start=0.0, duration=1.0)  # should be loaded from cache
        assert out.shape == (19, 100)
    # check missing
    infra.pop("mode")
    extractor = ns.extractors.MegExtractor(allow_missing=True, frequency=100, infra=infra)
    extractor.prepare(study)
    assert extractor._missing_default is not None
    assert extractor._missing_default.shape == (19,)
    empty = extractor([], start=0.0, duration=1.0)
    assert empty.shape == out.shape
    extractor = ns.extractors.MegExtractor(infra=infra, frequency=50.0)  # type: ignore
    extractor.prepare(study)

    # test from file (.fif generated by Test2023Meg)
    fif = meg_data / "sub-0-raw.fif"

    extractor = ns.extractors.MegExtractor()
    event = make_meg_event(fif, 0.0)
    data = extractor(event, start=0.0, duration=1.0)
    assert extractor._effective_frequency is not None
    assert extractor._effective_frequency == 100.0
    assert extractor.frequency == "native"
    assert np.array_equal(data.shape, [19, 100])
    assert data.max() > 0.0

    # meg start > 0
    data = extractor(make_meg_event(fif, start=0.5), start=0.0, duration=1.0)
    assert not any(data[:, 0])
    assert data[:, -1].max() > 0

    # meg start < 0
    data = extractor(events=make_meg_event(fif, start=-1.0), start=0.0, duration=1.0)
    assert data.min() == 1

    # meg start < segment start
    data = extractor(events=make_meg_event(fif, start=0.0), start=0.5, duration=1.0)
    assert data.min() == 0.5
    assert data.max() == 1.49

    # meg ends before end of segment
    event = make_meg_event(fif)
    data = extractor(events=event, start=event["duration"] - 0.5, duration=1.0)
    assert not any(data[:, -1])
    assert data[:, 0].max() > 0

    # decim
    extractor = ns.extractors.MegExtractor(frequency=50.0)  # raw dependent
    data = extractor(make_meg_event(fif, start=0.5), start=0.0, duration=1.0)
    assert np.array_equal(data.shape, [19, 50])
    assert not any(data[:, 0])
    assert data[:, -1].max() > 0

    # meg channels vary
    extractor = ns.extractors.MegExtractor(frequency=100.0)
    study = ns.Study(
        name="Test2023Meg",
        path=test_data_path,
        query="timeline_index<2",
        infra_timelines=_NO_CLUSTER,
    ).run()
    extractor.prepare(study)
    images = study.query('type=="Image"')
    # select the first image of 2 different recordings
    sel = [d.index[0] for tl, d in images.groupby("timeline")]
    dset = ns.segments.list_segments(
        study, triggers=pd.Series(study.index.isin(sel)), start=0.0, duration=1.0
    )
    data_list = [extractor(**seg._to_extractor()) for seg in dset]
    assert np.shape(data_list) == (2, len(extractor._channels), 100)

    # test from file (.fif generated by Test2023Meg)
    assert fif.exists()
    events = pd.DataFrame(
        [
            dict(
                type="Meg",
                start=0.0,
                duration=100,
                filepath=fif,
                timeline="fo",
                subject="someonelse",
            ),
        ]
    )
    extractor = ns.extractors.neuro.MegExtractor(frequency=100.0)
    data = extractor(events, start=10.0, duration=1.0)
    assert data.shape == torch.Size([19, 100])
    assert data.max() > 0


@pytest.mark.parametrize(
    "l_freq,h_freq",
    [
        (None, None),
        (0.1, None),
        (None, 20.0),
        (0.1, 20.0),
        (1.0, 101.0),
    ],
)
def test_meg_filter(
    caplog, test_data_path: Path, l_freq: float | None, h_freq: float | None
) -> None:
    ns.Study(
        name="Test2023Meg",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()
    fif = test_data_path / "Test2023Meg" / "sub-0-raw.fif"

    event = etypes.Meg.from_dict(make_meg_event(fif, 0.0))
    frequency = 100.0
    extractor = ns.extractors.MegExtractor(frequency=frequency, filter=(l_freq, h_freq))
    with caplog.at_level("WARNING"):
        ta = next(iter(extractor._get_data([event])))

    assert isinstance(ta, TimedArray)
    assert ta.frequency == frequency
    assert ta.header is not None
    assert ta.header["highpass"] == (0.0 if l_freq is None else l_freq)
    if h_freq is None:
        assert ta.header["lowpass"] == frequency / 2
    elif h_freq >= frequency / 2:
        assert (
            "Lowpass filter cutoff frequency is higher than or equal to the Nyquist frequency."
            in caplog.text
        )
        assert ta.header["lowpass"] == frequency / 2
    else:
        assert ta.header["lowpass"] == h_freq


@pytest.mark.parametrize(
    "notch_filter,expected",
    [
        [10.0, "[10.0, 20.0, 30.0, 40.0]"],
        [40.0, "[40.0]"],
        [[40.0, 45.0], "[40.0, 45.0]"],
        [100.0, "Not applying notch filter as no valid frequencies were found."],
        [None, None],
    ],
)
def test_meg_notch_filter(
    caplog,
    test_data_path: Path,
    notch_filter: float | list[float] | None,
    expected: str | None,
) -> None:
    ns.Study(
        name="Test2023Meg",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()
    fif = test_data_path / "Test2023Meg" / "sub-0-raw.fif"

    event = etypes.Meg.from_dict(make_meg_event(fif, 0.0))
    extractor = ns.extractors.MegExtractor(notch_filter=notch_filter)
    with caplog.at_level("INFO"):
        next(iter(extractor._get_data([event])))

    if expected is not None:
        assert expected in caplog.text


@pytest.mark.parametrize(
    "start,tmin,tmax",
    [
        [1.0, -0.2, -0.1],
        [2.0, 0.0, 0.3],
        [3.0, -0.1, 0.1],
        [4.0, -0.5, 2.0],
        [5.0, 0.1, 3.0],
        [6.0, 0.0, 4.0],
    ],
)
def test_meg_baseline(
    start: float, tmin: float, tmax: float, test_data_path: Path
) -> None:
    ns.Study(
        name="Test2023Meg",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()
    fif = test_data_path / "Test2023Meg" / "sub-0-raw.fif"

    sfreq = 100.0
    event_start = 0.0
    duration = 1.0
    event = make_meg_event(fif, start=event_start)

    # Extract window without baseline normalization
    extractor = ns.extractors.MegExtractor(frequency=sfreq, baseline=None)
    no_bl_data = extractor(event, start=start, duration=duration)

    # Extract baseline only
    extractor = ns.extractors.MegExtractor(frequency=sfreq, baseline=None)
    bl_data = extractor(event, start=start + tmin, duration=tmax - tmin)

    # Extract window with baseline normalization
    extractor = ns.extractors.MegExtractor(frequency=sfreq, baseline=(tmin, tmax))
    bl_norm_data = extractor(event, start=start, duration=duration)

    # Extract larger window without baseline normalization
    extractor = ns.extractors.MegExtractor(frequency=sfreq, baseline=None)
    no_bl_full_data = extractor(
        event, start=start + tmin, duration=max(duration, tmax) - tmin
    )

    # Ensure everything matches
    assert no_bl_data.shape == bl_norm_data.shape
    bl_len = int(sfreq * (tmax - tmin))
    assert bl_data.shape[1] == bl_len
    assert torch.allclose(bl_data, no_bl_full_data[:, :bl_len])
    bl_norm_data2 = no_bl_data - bl_data.mean(dim=1, keepdim=True)
    np.testing.assert_almost_equal(bl_norm_data.numpy(), bl_norm_data2.numpy(), decimal=6)


@pytest.mark.parametrize(
    "offset,minmax",
    [
        (0, (0.5, 1.49)),
        (0.2, (0.7, 1.69)),
    ],
)
def test_meg_offset(
    offset: float, minmax: tuple[float, float], test_data_path: Path
) -> None:
    ns.Study(
        name="Test2023Meg",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()
    fif = test_data_path / "Test2023Meg" / "sub-0-raw.fif"
    event = make_meg_event(fif, start=0)
    # Extract window without baseline normalization
    extractor = ns.extractors.MegExtractor(frequency=100, baseline=None, offset=offset)
    data = extractor(event, start=0.5, duration=1.0).numpy()
    drange = [float(f(data[:])) for f in [np.min, np.max]]
    np.testing.assert_almost_equal(drange, minmax)


def test_fnirs(test_data_path: Path) -> None:
    ns.Study(
        name="Test2024Fnirs",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()

    sfreq = 10.0
    start = 0.0
    duration = 5.0

    timeline = 'Test2024Fnirs:{"subject":"0"}'
    filepath = '{"cls":"Test2024Fnirs","method":"_load_raw","timeline":{"subject":"0"}}'
    event = dict(
        start=start,
        duration=99.99,
        timeline=timeline,
        filepath=filepath,
        type="Fnirs",
        subject="janedoe",
    )

    # Extract window without baseline normalization
    extractor = ns.extractors.FnirsExtractor(frequency=sfreq, baseline=None)
    data = extractor(event, start=start, duration=duration)
    assert data.shape == (32, sfreq * duration)

    # Extract window with different preprocessing steps
    # XXX Uncomment preprocessing steps that require montage information (would need to update the
    # Test2024Fnirs class to also contain montage information)
    tmin, tmax = 0.0, 1.0
    extractor = ns.extractors.FnirsExtractor(
        frequency=sfreq,
        baseline=(tmin, tmax),
        # distance_threshold=10,
        # compute_optical_density=True,
        # scalp_coupling_index_threshold=0.0,
        # apply_tddr=True,
        # compute_heamo_response=True,
        # partial_pathlength_factor=0.1,
        # enhance_negative_correlation=True,
        filter=(0.01, 2.5),
    )
    preproc_data = extractor(event, start=start, duration=duration)
    assert preproc_data.shape == (32, sfreq * duration)


def test_cache(test_data_path: Path, tmp_path: Path) -> None:
    # FIXME the CACHE logic will fails if there are multiple recordings
    # (meg, fmri) within the same timeline.

    fmri: dict[str, tp.Any] = dict(
        device="Fmri",
        study="Test2023Fmri",
        stim="Word",
        extractor=ns.extractors.FmriExtractor,
        duration=2.0,
        shape=(20, 20, 20, 1),
        file=test_data_path / "Test2023Fmri" / "sub-0.nii.gz",
        subject="someone",
    )

    meg: dict[str, tp.Any] = dict(
        device="Meg",
        study="Test2023Meg",
        stim="Image",
        extractor=ns.extractors.MegExtractor,
        duration=0.5,
        shape=(19, 50),
        file=test_data_path / "Test2023Meg" / "sub-0-raw.fif",
        subject="someone",
    )

    for cond in (meg, fmri):
        study = str(cond["study"])
        events = ns.Study(
            name=study, path=test_data_path, infra_timelines=_NO_CLUSTER
        ).run()
        sel = events.type == cond["stim"]
        dset = ns.segments.list_segments(
            events, triggers=sel, start=0.0, duration=cond["duration"]
        )
        segment = dset[0]

        for cache in (None, tmp_path / "cache"):
            for filtr in (None, (None, 20.0)):
                kwargs: tp.Any = dict(cache=cache, filter=filtr)
                # TODO, clean up when FMRI is dealt with
                kwargs.pop("cache")  # for now, now cache in FMRI
                if "Fmri" in study:
                    kwargs.pop("filter")
                if cond is meg:
                    kwargs["infra"] = {"folder": cache}

                extractor = cond["extractor"](**kwargs)
                data = extractor(segment.events, segment.start, segment.duration)
                assert np.array_equal(data.shape, cond["shape"])
                assert data.max() > 0

        # no cache, no reader needed
        file = cond["file"]
        assert file.exists()
        event = dict(
            type=cond["device"],
            start=0.0,
            duration=cond["duration"],
            filepath=file,
            timeline="fo",
            subject="someone",
        )
        if cond["device"] == "Fmri":
            event.update(duration=20, frequency=0.5, space="custom")
        events = pd.DataFrame(
            [
                dict(type="Word", start=10.0, duration=0.5, text="Hello", timeline="fo"),
                event,
            ]
        )
        extractor = ns.extractors.base.Pulse(
            aggregation="sum", event_types=cond["device"]
        )
        data = extractor(events, start=10.0, duration=cond["duration"])
        assert np.array_equal(data.shape, [1])

        # no cache, but reader
        for cache in (None, tmp_path / "cache"):
            for filtr in (None, (None, 20.0)):
                kwargs = dict(cache=cache, filter=filtr)
                kwargs.pop("cache")  # TODO add fmri cache?
                if "Fmri" in cond["study"]:
                    kwargs.pop("filter")
                if cond is meg:
                    kwargs["infra"] = {"folder": cache}

                extractor = cond["extractor"](**kwargs)
                data = extractor(events, start=0.0, duration=cond["duration"])
                assert np.array_equal(data.shape, cond["shape"])
                assert data.max() > 0


def test_meg_feature_cache(test_data_path: Path, tmp_path: Path) -> None:
    events = ns.Study(
        name="Test2023Meg", path=test_data_path, infra_timelines=_NO_CLUSTER
    ).run()
    cache = tmp_path / "cache"
    extractor = ns.extractors.MegExtractor(
        frequency=100.0,
        filter=(None, 20.0),
        infra={"folder": cache},  # type: ignore
    )
    dset = ns.segments.list_segments(
        events, triggers=events.type == "Image", start=0.0, duration=5
    )
    _ = extractor(dset[0].events, start=dset[0].start, duration=dset[0].duration)
    v = extractor.infra.version
    assert v == "1"
    path = cache / f"neuralset.extractors.neuro.MegExtractor._get_data,{v}"
    path = path / "frequency=100,filter=(None,20),name=MegExtractor-9a73a136"
    assert path.exists(), f"Path does not exist, got: {list(path.parent.iterdir())}"
    ta = next(extractor.infra.cache_dict.values())  # type: ignore
    assert isinstance(ta, TimedArray)
    assert ta.header is not None
    assert "ch_names" in ta.header


def test_eeg_feature_cache(test_data_path: Path, tmp_path: Path) -> None:
    events = ns.Study(
        name="Test2024Eeg", path=test_data_path, infra_timelines=_NO_CLUSTER
    ).run()
    cache = tmp_path / "cache"
    extractor = ns.extractors.EegExtractor(
        frequency=100.0,
        filter=(None, 20.0),
        infra={"folder": cache},  # type: ignore
    )
    events_ = extract_events(events, types=extractor.event_types)
    ta = next(iter(extractor._get_data(events_)))  # type: ignore
    assert isinstance(ta, TimedArray)
    assert ta.header is not None
    assert "ch_names" in ta.header


def test_base_meg(tmp_path: Path) -> None:
    extractor = ns.extractors.MegExtractor(
        frequency=100.0,
        filter=(None, 20.0),
        infra={"folder": tmp_path},  # type: ignore
    )
    assert isinstance(extractor, ns.extractors.MegExtractor)
    # check uids
    extractor_keys = set(ConfDict.from_model(extractor, uid=True).keys())

    assert extractor_keys == {
        "name",
        "allow_missing",
        "filter",
        "drop_bads",
        "picks",
        "baseline",
        "frequency",
        "apply_proj",
        "offset",
        "scaler",
        "aggregation",
        "clamp",
        "channel_order",
        "apply_hilbert",
        "notch_filter",
        "event_types",
        "scale_factor",
        "bipolar_ref",
        "infra",  # for version
        "fill_non_finite",
    }


@pytest.mark.parametrize("apply_proj", (True, False))
def test_epoch_correct(apply_proj: bool) -> None:
    # download the first time
    events = ns.Study(
        name="Mne2013Sample", path=ns.CACHE_FOLDER, infra_timelines=_NO_CLUSTER
    ).run()
    NUM = 12

    # Define segments
    segments = ns.segments.list_segments(
        events, triggers=events.type == "Stimulus", start=0.0, duration=1.0
    )
    segments = segments[:NUM]
    dset = ns.SegmentDataset(
        {"Meg": ns.extractors.MegExtractor(apply_proj=apply_proj)},
        segments,
    )

    start = time.time()
    dloader = torch.utils.data.DataLoader(
        dset,
        collate_fn=dset.collate_fn,
        batch_size=len(dset),
    )
    batch = next(iter(dloader))
    timing = {"dataloader": time.time() - start}

    # Fast Mne reader
    start = time.time()
    meg_event = events.loc[events.type == "Meg"].iloc[0]
    stims = events.loc[events.type == "Stimulus"]
    stims = stims.iloc[:NUM, :]
    raw = etypes.Meg.from_dict(meg_event).read()
    mne_events = np.ones((len(stims), 3), dtype=int)
    # print([s["start"] * raw.info["sfreq"] for s in segments])
    mne_events[:, 0] = stims.start * raw.info["sfreq"]
    raw = raw.pick(("meg",))
    epochs = mne.Epochs(
        raw, mne_events, tmin=0.0, tmax=1.0, baseline=None, proj=apply_proj
    )
    data = epochs.get_data()
    timing["epochs"] = time.time() - start

    # check for speed and correctness
    ratio = timing["dataloader"] / timing["epochs"]
    assert ratio < 15, f"dataloader is too slow {timing}"
    meg = batch.data["Meg"].cpu().numpy()
    # np.testing.assert_almost_equal(meg * 10**12, data[:, :, :-1] * 10**12, decimal=3)
    np.testing.assert_almost_equal(meg / data[:, :, :-1], np.ones(meg.shape), decimal=3)


@pytest.mark.parametrize("n_spatial_dims", [3])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "factor",
    [
        1.0,
        10.0,
    ],
)
def test_channel_positions_meg(
    test_data_path: Path, n_spatial_dims, normalize: bool, factor: float
) -> None:
    # Create and prepare Meg extractors
    meg = ns.extractors.MegExtractor(offset=1, picks="all")
    events = ns.Study(
        name="Test2023Meg",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()
    meg.prepare(events)

    # Define and prepare channel positions extractor
    invalid_value = ns.extractors.ChannelPositions.INVALID_VALUE
    ch_pos = ns.extractors.ChannelPositions(
        neuro=meg,
        layout_or_montage_name=None,
        n_spatial_dims=n_spatial_dims,
        normalize=normalize,
    )
    ch_pos.prepare(events)

    event = ns.events.Event.from_dict(events[events.type == "Meg"].iloc[0])
    assert isinstance(event, etypes.Meg)
    ta = next(meg._get_data([event]))
    assert ta.header is not None
    ch_names = ta.header["ch_names"].split(",")
    positions = ch_pos(event, start=0.0, duration=1.0)
    assert positions.shape == (len(ch_names), n_spatial_dims)
    assert (positions[ch_names.index("INVALID_CHANNEL")] == invalid_value).all()


@pytest.mark.parametrize("n_spatial_dims", [2, 3])
@pytest.mark.parametrize("normalize", [True, False])
def test_channel_positions_eeg(test_data_path: Path, n_spatial_dims, normalize) -> None:
    eeg = ns.extractors.EegExtractor()
    events = ns.Study(
        name="Test2024Eeg",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()
    eeg.prepare(events)

    # Define and prepare channel positions extractor
    invalid_value = ns.extractors.ChannelPositions.INVALID_VALUE
    ch_pos = ns.extractors.ChannelPositions(
        neuro=eeg,
        event_types="Eeg",
        layout_or_montage_name="EEG1005" if n_spatial_dims == 2 else None,
        n_spatial_dims=n_spatial_dims,
        normalize=normalize,
    )
    ch_pos.prepare(events)

    event = ns.events.Event.from_dict(events[events.type == "Eeg"].iloc[0])
    assert isinstance(event, etypes.Eeg)
    ta = next(eeg._get_data([event]))
    assert ta.header is not None
    ch_names = ta.header["ch_names"].split(",")
    positions = ch_pos(event, start=0.0, duration=1.0)
    assert positions.shape == (len(ch_names), n_spatial_dims)
    assert (positions[ch_names.index("INVALID_CHANNEL")] == invalid_value).all()


def test_channel_positions_wrong_event_type(test_data_path: Path) -> None:
    eeg = ns.extractors.EegExtractor()
    events = ns.Study(
        name="Test2024Eeg",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()
    eeg.prepare(events)

    with pytest.raises(
        ValueError, match="event_types=Meg must match neuro.event_types=Eeg"
    ):
        ns.extractors.ChannelPositions(neuro=eeg, event_types="Meg")


def test_channel_positions_build() -> None:
    eeg = ns.extractors.EegExtractor()
    original_ch_pos = ns.extractors.ChannelPositions(
        neuro=eeg,
        layout_or_montage_name="EEG1005",
    )
    ch_pos_config = ns.extractors.ChannelPositions(
        layout_or_montage_name="EEG1005",
    )
    ch_pos = ch_pos_config.build(eeg)
    assert ch_pos.model_dump() == original_ch_pos.model_dump()


@pytest.mark.parametrize("channel_order", ["unique", "original"])
def test_unique_channels_ieeg(
    channel_order: tp.Literal["unique", "original"], test_data_path: Path
) -> None:
    events = ns.Study(
        name="Test2025Ieeg",
        path=test_data_path,
        query="subject_index<2",
        infra_timelines=_NO_CLUSTER,
    ).run()

    if channel_order == "unique":
        max_idx = 100
    else:
        max_idx = 50

    ieeg = ns.extractors.IeegExtractor(channel_order=channel_order)
    ieeg.prepare(events)

    assert max(ieeg._channels.values()) == max_idx


@pytest.mark.parametrize("channel_order", ["unique", "original"])
def test_batch_size_unique_channels_ieeg(
    channel_order: tp.Literal["unique", "original"], test_data_path: Path
) -> None:
    events = ns.Study(
        name="Test2025Ieeg",
        path=test_data_path,
        query="subject_index<2",
        infra_timelines=_NO_CLUSTER,
    ).run()

    ieeg = ns.extractors.IeegExtractor(channel_order=channel_order)
    ieeg.prepare(events)

    segmenter = ns.dataloader.Segmenter(
        start=0.0,
        duration=1.0,
        trigger_query="type=='Word'",
        extractors=dict(ieeg=ieeg),
        drop_incomplete=True,
        stride=1.0,
        stride_drop_incomplete=False,
    )

    dset = segmenter.apply(events)
    dset.prepare()

    batch = dset[0].data["ieeg"]

    output_shape = 101 if channel_order == "unique" else 51
    assert batch.shape[1] == output_shape


def test_channel_positions_ieeg(test_data_path: Path) -> None:
    ieeg = ns.extractors.IeegExtractor()
    events = ns.Study(
        name="Test2025Ieeg",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()
    ieeg.prepare(events)

    # Define and prepare channel positions extractor
    invalid_value = ns.extractors.ChannelPositions.INVALID_VALUE
    ch_pos = ns.extractors.ChannelPositions(
        neuro=ieeg,
        event_types="Ieeg",
        layout_or_montage_name=None,
        n_spatial_dims=3,
        normalize=False,
    )
    ch_pos.prepare(events)

    event = ns.events.Event.from_dict(events[events.type == "Ieeg"].iloc[0])
    assert isinstance(event, etypes.Ieeg)
    ta = next(ieeg._get_data([event]))
    assert ta.header is not None
    ch_names = ta.header["ch_names"].split(",")
    positions = ch_pos(event, start=0.0, duration=1.0)
    assert positions.shape == (len(ch_names), 3)
    assert (positions[ch_names.index("INVALID_CHANNEL")] == invalid_value).all()


@pytest.fixture
def fake_bipolar_eeg() -> mne.io.Raw:
    ch_names = ["Fp1-Fp2", "Fp2-T8", "This is not a channel name", "Oz"]
    ch_types = ["eeg"] * len(ch_names)
    sfreq = 120.0
    data = np.random.rand(len(ch_names), int(sfreq * 30))
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(data, info)


def _assert_raw_equal(raw_out: mne.io.Raw, raw_in: mne.io.Raw) -> None:
    """Check that two Raw objects match on all fields preserved by MneTimedArray."""
    for key in ("ch_names", "sfreq", "highpass", "lowpass"):
        assert raw_out.info[key] == raw_in.info[key], key
    assert raw_out.first_samp == raw_in.first_samp
    np.testing.assert_array_equal(raw_out.get_data(), raw_in.get_data())
    locs = lambda r: np.array([ch["loc"] for ch in r.info["chs"]])
    np.testing.assert_array_equal(locs(raw_out), locs(raw_in))


def test_mne_timed_array_roundtrip() -> None:
    from neuralset.extractors.neuro import MneTimedArray

    ch_names = ["EEG 001", "EEG 002", "EEG 003"]
    sfreq = 256.0
    data = (
        np.random.RandomState(42).randn(len(ch_names), int(sfreq * 2)).astype(np.float32)
    )
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    raw_in = mne.io.RawArray(data, info, first_samp=512, verbose=False)

    ta = MneTimedArray.from_native(raw_in)
    _assert_raw_equal(ta.to_native(), raw_in)


def test_mne_timed_array_cache_roundtrip(tmp_path: Path) -> None:
    """MneTimedArray must survive DumpContext serialization as MneTimedArray,
    not be downgraded to a plain TimedArray (which lacks ch_names etc.).
    Also enforces the on-disk jsonl structure so format changes are caught.
    """
    from exca.cachedict import DumpContext

    from neuralset.extractors.neuro import MneTimedArray

    ch_names = ["EEG 001", "EEG 002"]
    sfreq = 128.0
    data = np.random.RandomState(0).randn(len(ch_names), int(sfreq)).astype(np.float32)
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    ta = MneTimedArray.from_native(raw)

    ctx = DumpContext(tmp_path)
    with ctx:
        dumped = ctx.dump_entry("mne_ta", ta)

    # -- verify on-disk structure --
    jsonl_files = list(tmp_path.glob("*-info.jsonl"))
    assert len(jsonl_files) == 1
    entry = json.loads(jsonl_files[0].read_text("utf8").strip())
    # strip dynamic fields (filenames, offsets) for comparison
    for blob in [entry["data"], entry["header"]["content"]["ch_locs"]]:
        del blob["filename"], blob["offset"]
    expected = yaml.safe_load(
        """
        "#type": MneTimedArray
        "#key": mne_ta
        frequency: 128.0
        start: 0.0
        duration: 1.0
        data:
            shape: [128, 2]
            dtype: float32
            "#type": MemmapArray
        header:
            "#type": Auto
            content:
                ch_names: "EEG 001,EEG 002"
                ch_types: "eeg,eeg"
                highpass: 0.0
                lowpass: 64.0
                ch_locs:
                    shape: [2, 12]
                    dtype: float64
                    "#type": MemmapArray
    """
    )
    assert entry == expected, f"Structure mismatch:\n{json.dumps(entry, indent=2)}"

    # -- verify round-trip --
    loaded = ctx.load(dumped["content"])
    assert isinstance(loaded, MneTimedArray)
    assert loaded.ch_names == ch_names
    assert loaded.ch_types == ["eeg", "eeg"]
    _assert_raw_equal(loaded.to_native(), raw)


def test_channel_positions_include_ref(fake_bipolar_eeg) -> None:
    from neuralset.extractors.neuro import MneTimedArray

    eeg = ns.extractors.EegExtractor()
    ch_pos = ns.extractors.ChannelPositions(
        neuro=eeg,
        layout_or_montage_name="EEG1005",
        include_ref_eeg=True,
    )

    ta = MneTimedArray.from_native(fake_bipolar_eeg)
    positions = ch_pos._compute_positions(ta)
    assert positions.shape == (len(fake_bipolar_eeg.ch_names), 4)
    assert (
        positions[-1, -2:] == ch_pos.INVALID_VALUE
    ).all()  # No explicit anode for last channel


def test_channel_positions_no_valid_layout(fake_bipolar_eeg) -> None:
    from neuralset.extractors.neuro import MneTimedArray

    eeg = ns.extractors.EegExtractor()
    ch_pos = ns.extractors.ChannelPositions(
        neuro=eeg,
        layout_or_montage_name=None,
        include_ref_eeg=True,
    )
    ta = MneTimedArray.from_native(fake_bipolar_eeg)
    with pytest.raises(ValueError, match="No valid layout found."):
        ch_pos._compute_positions(ta)


@pytest.fixture
def fake_invalid_eeg() -> mne.io.Raw:
    ch_names = ["Cz-Fz", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    ch_types = ["eeg"] * len(ch_names)
    sfreq = 120.0
    data = np.random.rand(len(ch_names), int(sfreq * 30))
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types=ch_types)
    return mne.io.RawArray(data, info)


def test_channel_positions_too_many_invalid(fake_invalid_eeg, caplog) -> None:
    from neuralset.extractors.neuro import MneTimedArray

    eeg = ns.extractors.EegExtractor()
    ch_pos = ns.extractors.ChannelPositions(
        neuro=eeg,
        layout_or_montage_name="EEG1005",
        include_ref_eeg=False,
    )
    ta = MneTimedArray.from_native(fake_invalid_eeg)
    ch_pos._compute_positions(ta)
    assert "Fewer than 10% of the channels have valid positions" in caplog.text


def test_channel_positions_require_eeg() -> None:
    meg = ns.extractors.MegExtractor()
    with pytest.raises(ValueError):
        ns.extractors.ChannelPositions(
            neuro=meg,
            layout_or_montage_name=None,
            include_ref_eeg=True,
        )


@pytest.fixture
def fake_eeg_on_a_line() -> mne.io.Raw:
    ch_names = ["Fpz", "Cz", "Pz"]
    ch_types = ["eeg"] * len(ch_names)
    sfreq = 120.0
    data = np.random.rand(len(ch_names), int(sfreq * 30))
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types=ch_types)

    return mne.io.RawArray(data, info)


def test_channel_positions_on_a_line(fake_eeg_on_a_line) -> None:
    from neuralset.extractors.neuro import MneTimedArray

    eeg = ns.extractors.EegExtractor()
    ch_pos = ns.extractors.ChannelPositions(
        neuro=eeg,
        layout_or_montage_name="EEG1005",
    )
    ta = MneTimedArray.from_native(fake_eeg_on_a_line)
    pos = ch_pos._compute_positions(ta)
    assert (pos[:, 0] == 0.0).all()


@pytest.mark.parametrize("montage_name", ["standard_1020", "standard_1005", "biosemi128"])
def test_channel_positions_head_frame(montage_name: str) -> None:
    """Both the named-montage and ch_locs paths must return head-frame coordinates."""
    from neuralset.extractors.neuro import MneTimedArray

    montage = mne.channels.make_standard_montage(montage_name)
    ch_names = list(montage.get_positions()["ch_pos"].keys())[:8]
    sfreq = 128.0
    data = np.zeros((len(ch_names), int(sfreq)))
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(montage_name, verbose=False)

    ta = MneTimedArray.from_native(raw)

    eeg = ns.extractors.EegExtractor()

    named = ns.extractors.ChannelPositions(
        neuro=eeg,
        layout_or_montage_name=montage_name,
        n_spatial_dims=3,
        normalize=False,
    )
    pos_named = named._get_montage_positions(ta)

    fallback = ns.extractors.ChannelPositions(
        neuro=eeg,
        layout_or_montage_name=None,
        n_spatial_dims=3,
        normalize=False,
    )
    pos_fallback = fallback._get_montage_positions(ta)

    common = sorted(set(pos_named) & set(pos_fallback))
    assert len(common) == len(ch_names)
    for name in common:
        np.testing.assert_allclose(
            pos_named[name],
            pos_fallback[name],
            atol=1e-10,
            err_msg=f"Head-frame mismatch for {name} with montage {montage_name}",
        )


def test_meg_borders(test_data_path: Path, tmp_path: Path) -> None:
    cache_path = tmp_path / "cache"
    # test from study
    extractor = ns.extractors.MegExtractor(frequency=100, filter=(100, None))
    loader = ns.Study(
        name="Test2023Meg",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    )
    events = loader.run()
    extractor = ns.extractors.MegExtractor(infra=dict(folder=cache_path))  # type: ignore
    meg_event = etypes.Meg.from_dict(events.query('type=="Meg"').iloc[0])
    out = extractor(meg_event, start=meg_event.start - 0.5, duration=1.0)
    # overlap with the beginning
    assert out[0, 0] == 0.0
    assert out[0, -1] != 0.0
    # overlap with the end
    out = extractor(
        meg_event, start=meg_event.start + meg_event.duration - 0.5, duration=1.0
    )
    assert out[0, 0] != 0.0
    assert out[0, -1] == 0.0


@pytest.mark.parametrize("scaler", ["RobustScaler", "StandardScaler"])
@pytest.mark.parametrize("clamp", [None, 0.01, 0.1])
def test_scaled_meg(
    scaler: tp.Literal["RobustScaler", "StandardScaler"],
    test_data_path: Path,
    clamp: float | None,
) -> None:
    sfreq = 100.0
    start = 0.0
    duration = 1.0

    loader = ns.Study(
        name="Test2023Meg",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    )
    events = loader.run()
    event = etypes.Meg.from_dict(events.query('type=="Meg"').iloc[0])

    extractor = ns.extractors.MegExtractor(frequency=sfreq)
    X = extractor(event, start=start, duration=duration)

    ta = next(extractor._get_data([event]))
    data = ta.data

    _scaler = getattr(sklearn.preprocessing, scaler)().fit(data.T)

    scaled_extractor = ns.extractors.MegExtractor(
        frequency=sfreq,
        scaler=scaler,
        clamp=clamp,
    )
    scaled_X1 = scaled_extractor(event, start=start, duration=duration)

    scaled_X2 = torch.Tensor(_scaler.transform(X.T).T)
    if clamp is not None:
        scaled_X2 = torch.clamp(scaled_X2, min=-clamp, max=clamp)

    assert torch.allclose(scaled_X1, scaled_X2, atol=1e-12)


def test_first_samp() -> None:
    events = ns.Study(
        name="Mne2013Sample", path=ns.CACHE_FOLDER, infra_timelines=_NO_CLUSTER
    ).run()
    meg_event = ns.events.Event.from_dict(events.loc[events.type == "Meg"].iloc[0])
    assert meg_event.start > 0
    # first samp is ~6k at 150Hz, so start is at about 42s
    extractor = ns.extractors.MegExtractor(frequency=150)
    assert not extractor(meg_event, 41, 1).norm()
    assert extractor(meg_event, 43, 1).norm()


def test_fast_event() -> None:
    events = ns.Study(
        name="Mne2013Sample", path=ns.CACHE_FOLDER, infra_timelines=_NO_CLUSTER
    ).run()
    meg_event = ns.events.Event.from_dict(events.loc[events.type == "Meg"].iloc[0])
    assert meg_event.start > 0
    extractor = ns.extractors.MegExtractor(frequency=150)
    duration = 0.001
    assert extractor(meg_event, start=meg_event.start + 1, duration=duration).norm()


def test_border_baseline(test_data_path: Path) -> None:
    events = ns.Study(
        name="Test2023Meg", path=test_data_path, infra_timelines=_NO_CLUSTER
    ).run()
    meg_event = ns.events.Event.from_dict(events.loc[events.type == "Meg"].iloc[0])
    extractor = ns.extractors.MegExtractor(frequency=150, baseline=(0, 0.6))
    _ = extractor(meg_event, meg_event.stop - 0.5, 1)  # baseline should not bug


all_meg_channels = {
    "MEG 0111",
    "MEG 0121",
    "MEG 0131",
    "MEG 0141",
    "MEG 0211",
    "MEG 0221",
    "MEG 0231",
    "MEG 0241",
    "MEG 0311",
    "MEG 0321",
    "MEG 0331",
    "MEG 0341",
    "MEG 0411",
    "MEG 0421",
    "MEG 0431",
    "MEG 0441",
    "MEG 0511",
    "MEG 0521",
    "MEG 0531",
    "INVALID_CHANNEL",
}


@pytest.mark.parametrize(
    "picks,expected",
    [
        ("all", all_meg_channels),
        ("data", all_meg_channels - {"INVALID_CHANNEL"}),
        ("meg", all_meg_channels - {"INVALID_CHANNEL"}),
        ("mag", all_meg_channels - {"MEG 0531", "INVALID_CHANNEL"}),
        (("mag",), all_meg_channels - {"MEG 0531", "INVALID_CHANNEL"}),
        ("grad", {"MEG 0531"}),
        (["MEG 0111", "MEG 0121", "MEG 0131"], {"MEG 0111", "MEG 0121", "MEG 0131"}),
        ("^MEG 05", {"MEG 0511", "MEG 0521", "MEG 0531"}),
    ],
)
def test_meg_picks(test_data_path: Path, picks, expected: set[str]) -> None:
    events = ns.Study(
        name="Test2023Meg", path=test_data_path, infra_timelines=_NO_CLUSTER
    ).run()
    meg_event = ns.events.Event.from_dict(events.loc[events.type == "Meg"].iloc[0])
    extractor = ns.extractors.MegExtractor(frequency=150, picks=picks)
    ta = next(iter(extractor._get_data([meg_event])))  # type: ignore
    assert ta.header is not None
    assert set(ta.header["ch_names"].split(",")) == expected


def test_meg_method_on_infra(test_data_path: Path, tmp_path: Path) -> None:
    events = ns.Study(
        name="Test2023Meg",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()
    infra: tp.Any = {"cluster": "local", "folder": tmp_path}
    extractor = ns.extractors.MegExtractor(frequency=150, baseline=(0, 0.6), infra=infra)
    extractor.prepare(events)


def test_fmri_cluster_local(test_data_path: Path, tmp_path: Path) -> None:
    study = ns.Study(
        name="Test2023Fmri",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines={"folder": tmp_path, **_NO_CLUSTER},  # type: ignore
    ).run()
    fmri_event = extract_events(study, types=etypes.Fmri)[0]
    assert isinstance(fmri_event, etypes.Fmri)
    assert fmri_event.space == "custom"
    projection: tp.Any = {"name": "SurfaceProjector", "mesh": "fsaverage5"}
    infra: tp.Any = {"cluster": "local", "folder": tmp_path, "mode": "force"}
    extractor = ns.extractors.FmriExtractor(projection=projection, infra=infra)  # type: ignore
    extractor.prepare(study)


class _PseudoNifti:  # hack used in nastase
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.shape = self.data.shape

    def get_fdata(self):
        return self.data


def test_fmri_mesh_from_2d() -> None:
    n_voxels = 40962 * 2  # fsaverage6
    n_times = 10
    nii = _PseudoNifti(np.random.rand(n_voxels, n_times))
    event = etypes.Fmri(
        start=0,
        duration=n_times,
        frequency=1,
        timeline="foo",
        space="custom",
        subject="blublu",
        filepath="blublu",
    )
    object.__setattr__(event, "read", lambda: nii)  # fake it
    projection: tp.Any = {"name": "SurfaceProjector", "mesh": "fsaverage5"}
    feat = ns.extractors.FmriExtractor(
        projection=projection,
    )  # type: ignore
    out = feat(event, 0, 5)
    assert out.shape == (10242 * 2, 5)  # fsaverage5 size


def test_ieeg(test_data_path: Path) -> None:
    ns.Study(
        name="Test2025Ieeg",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()

    sfreq = 2048.0
    start = 0.0
    duration = 5.0

    timeline = 'Test2025Ieeg:{"subject":"0"}'
    filepath = '{"cls":"Test2025Ieeg","method":"_load_raw","timeline":{"subject":"0"}}'

    event = dict(
        start=start,
        duration=99.99,
        timeline=timeline,
        filepath=filepath,
        type="Ieeg",
        subject="janedoe",
        drop_bads=False,
    )

    # Extract window without baseline normalization
    extractor = ns.extractors.IeegExtractor(frequency=sfreq, baseline=None)
    data = extractor(event, start=start, duration=duration)
    assert data.shape == (51, sfreq * duration)

    # Extract window with different preprocessing steps
    tmin, tmax = 0.0, 1.0
    extractor = ns.extractors.IeegExtractor(
        frequency=sfreq,
        baseline=(tmin, tmax),
        filter=(0.05, 20.0),
        reference="bipolar",
        picks=("seeg",),  # bipolar is only allowed with seeg
        notch_filter=50.0,
        drop_bads=True,
    )
    preproc_data = extractor(event, start=start, duration=duration)
    # n_chans -1 due to bipolar filter:
    assert preproc_data.shape == (24, sfreq * duration)


def test_bipolar_ieeg(tmp_path: Path) -> None:
    expected = "Bipolar reference can only be applied on seeg signals"
    with pytest.raises(ValueError, match=expected):
        ns.extractors.IeegExtractor(picks=("ecog",), reference="bipolar")


def test_ieeg_cannot_combine_bipolar_modes() -> None:
    expected = "Cannot use both reference='bipolar'"
    with pytest.raises(ValueError, match=expected):
        ns.extractors.IeegExtractor(
            picks=("seeg",),
            reference="bipolar",
            bipolar_ref=(["A1"], ["A2"]),
        )


def test_bipolar_ref_explicit() -> None:
    """MneRaw.bipolar_ref produces the expected bipolar channels."""
    ch_names = ["Fp1", "F7", "T7", "P7"]
    sfreq = 200.0
    data = np.random.RandomState(0).randn(len(ch_names), int(sfreq * 2))
    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    anodes = ["Fp1", "F7", "T7"]
    cathodes = ["F7", "T7", "P7"]
    extractor = ns.extractors.EegExtractor(
        picks=tuple(ch_names),
        bipolar_ref=(anodes, cathodes),
    )

    result = extractor._preprocess_raw(
        raw.copy(), tp.cast(etypes.MneRaw, SimpleNamespace(frequency=sfreq))
    )

    expected_ch = {"Fp1-F7", "F7-T7", "T7-P7"}
    assert set(result.ch_names) == expected_ch

    fp1_f7_idx = result.ch_names.index("Fp1-F7")
    np.testing.assert_array_almost_equal(result.data[fp1_f7_idx], data[0] - data[1])


def test_bipolar_ref_validation() -> None:
    """bipolar_ref anodes and cathodes must have equal length."""
    with pytest.raises(ValueError, match="equal length"):
        ns.extractors.EegExtractor(
            bipolar_ref=(["Fp1", "F7"], ["F7"]),
        )


def test_overlap() -> None:
    event_start = 10.0
    event_duration = 6.0
    segment_start = 8.0
    segment_duration = 7.0
    # Test case: overlap exists
    expected_overlap_start = 10.0
    expected_overlap_duration = 5.0
    assert _overlap(event_start, event_duration, segment_start, segment_duration) == (
        expected_overlap_start,
        expected_overlap_duration,
    )
    # Test case: no overlap
    assert _overlap(event_start, event_duration, 20.0, 3.0) == (20.0, 0.0)


def test_drop_bad_channels(test_data_path: Path, tmp_path: Path) -> None:
    events = ns.Study(
        name="Test2024Eeg", path=test_data_path, infra_timelines=_NO_CLUSTER
    ).run()
    cache = tmp_path / "cache"
    # Check bad channels present
    extractor = ns.extractors.EegExtractor(
        frequency=100.0,
        filter=(None, 20.0),
        drop_bads=False,
        infra={"folder": cache},  # type: ignore
    )
    events_ = extract_events(events, types=extractor.event_types)
    ta = next(iter(extractor._get_data(events_)))  # type: ignore
    assert ta.header is not None
    ch_names = set(ta.header["ch_names"].split(","))
    assert {"Fp1", "Fpz"} < ch_names
    # Check bad channels dropped
    extractor = ns.extractors.EegExtractor(
        frequency=100.0,
        filter=(None, 20.0),
        drop_bads=True,
        infra={"folder": cache},  # type: ignore
    )
    events_ = extract_events(events, types=extractor.event_types)
    ta = next(iter(extractor._get_data(events_)))  # type: ignore
    assert ta.header is not None
    ch_names = set(ta.header["ch_names"].split(","))
    assert {"Fp1", "Fpz"} & ch_names == set()


def test_pick_channel_names(test_data_path: Path, tmp_path: Path) -> None:
    events = ns.Study(
        name="Test2024Eeg", path=test_data_path, infra_timelines=_NO_CLUSTER
    ).run()
    cache = tmp_path / "cache"
    # Check that picks allows list of channel names
    extractor = ns.extractors.EegExtractor(
        frequency=100.0,
        filter=(None, 20.0),
        picks=("Fp1", "Fpz", "Fp2"),
        infra={"folder": cache},  # type: ignore
    )
    events_ = extract_events(events, types=extractor.event_types)
    ta = next(iter(extractor._get_data(events_)))  # type: ignore
    assert ta.header is not None
    assert set(["Fp1", "Fpz", "Fp2"]) == set(ta.header["ch_names"].split(","))
    # Check that it works with drop_bads
    extractor = ns.extractors.EegExtractor(
        frequency=100.0,
        filter=(None, 20.0),
        picks=("Fp1", "Fpz", "Fp2"),
        drop_bads=True,  # bads = ["Fp1", "Fpz"]
        infra={"folder": cache},  # type: ignore
    )
    events_ = extract_events(events, types=extractor.event_types)
    ta = next(iter(extractor._get_data(events_)))  # type: ignore
    assert ta.header is not None
    assert set(["Fp2"]) == set(ta.header["ch_names"].split(","))


def test_fake_meg_study() -> None:
    # TODO: replace this test with the standard info test when PR is merged
    loader = ns.Study(
        name="Fake2025Meg",
        path=ns.CACHE_FOLDER,
        query="subject_index < 1",
        infra_timelines=_NO_CLUSTER,
    )
    events = loader.run()
    assert set(events.type) == {
        "Meg",
        "Stimulus",
        "Text",
        "Word",
        "Audio",
        "Image",
        "Sentence",
    }


def test_fake_fmri_study() -> None:
    # TODO: replace this test with the standard info test when PR is merged
    loader = ns.Study(
        name="Fake2025Fmri",
        path=ns.CACHE_FOLDER,
        query="timeline_index < 1",
        infra_timelines=_NO_CLUSTER,
    )
    events = loader.run()
    try:
        nii = etypes.Fmri.from_dict(events.loc[0]).read()
    except (requests.exceptions.HTTPError, requests.exceptions.ConnectTimeout, OSError):
        pytest.skip("Atlas unreachable")
    assert nii.shape == (91, 109, 91, 80)
    assert set(events.type) == {"Fmri", "Text", "Word", "Audio", "Image"}


def test_hrf() -> None:
    sentence = "This is a sentence for the unit tests"
    events_list = []
    for timeline in ("foo", "bar"):
        for i in range(len(sentence)):
            events_list.append(
                dict(
                    type="Word",
                    text=sentence[i],
                    start=i,
                    duration=i + 1,
                    timeline=timeline,
                )
            )
    events = pd.DataFrame(events_list)

    # Working example
    extractor: tp.Any = dict(name="WordLength", frequency=100.0, aggregation="sum")
    hrf = ns.extractors.neuro.HrfConvolve(
        extractor=extractor,
        frequency=0.5,
    )

    # Prepare is compulsory
    with pytest.raises(ValueError):
        hrf(events.query('timeline=="foo"'), start=5, duration=5.0)

    hrf.prepare(events)
    hrf(events.query('timeline=="foo"'), start=5, duration=5.0)
    # TODO add numerical test to compare against manual implementation?

    # TODO add error if no prepare

    # raise if events from heterogenous timelines
    # FIXME should be an error at BaseExtractor level
    # with pytest.raises(ValueError):
    #     hrf(events, start=5, duration=5.)

    # Cannot make apply to static extractors
    with pytest.raises(ValueError):
        feature_: tp.Any = dict(name="WordLength")
        ns.extractors.neuro.HrfConvolve(
            extractor=feature_,
            frequency=0.5,
        )

    # raises if no cache
    with pytest.raises(ValueError):
        infra: tp.Any = dict(keep_in_ram=None, folder=None)
        ns.extractors.neuro.HrfConvolve(
            extractor=extractor,
            frequency=0.5,
            infra=infra,
        )


def test_fmri_space_selection(tmp_path: Path) -> None:
    """Verify space selection: None raises, 'auto' uses heuristic, explicit picks."""
    _mk = test_etypes.make_fmri_event
    mni = "MNI152NLin2009cAsym"
    events = [
        _mk(tmp_path, space="T1w"),
        _mk(tmp_path, space=mni),
        _mk(tmp_path, space="fsaverage"),
    ]

    # from_space="auto" + SurfaceProjector → prefers fsaverage
    _surf5: tp.Any = {"name": "SurfaceProjector", "mesh": "fsaverage5"}
    feat = ns.extractors.FmriExtractor(projection=_surf5, from_space="auto")
    window: tp.Any = dict(start=0.0, duration=1.0)  # dummy — we test space filtering
    relevant = feat._get_relevant_events(events, trigger=None, **window)
    assert len(relevant) == 1
    ta = feat._preprocess_event(relevant[0])  # type: ignore[arg-type]
    assert ta.header is not None
    assert ta.header["space"] == "fsaverage5"

    # explicit from_space overrides
    feat = ns.extractors.FmriExtractor(from_space=mni)
    relevant = feat._get_relevant_events(events, trigger=None, **window)
    assert len(relevant) == 1
    ta = feat._preprocess_event(relevant[0])  # type: ignore[arg-type]
    assert ta.header is not None and ta.header["space"] == mni

    # single space + from_space=None → passes through
    feat = ns.extractors.FmriExtractor()
    single = [_mk(tmp_path, space="T1w")]
    relevant = feat._get_relevant_events(single, trigger=None, **window)
    assert len(relevant) == 1
    ta = feat._preprocess_event(relevant[0])  # type: ignore[arg-type]
    assert ta.header is not None and ta.header["space"] == "T1w"

    # multiple spaces + from_space=None → raises
    feat = ns.extractors.FmriExtractor()
    with pytest.raises(ValueError, match="Multiple spaces"):
        feat._get_relevant_events(events, trigger=None, **window)

    # multiple spaces + from_space=None + projection → also raises (no silent auto)
    feat = ns.extractors.FmriExtractor(projection=_surf5)
    with pytest.raises(ValueError, match="Multiple spaces"):
        feat._get_relevant_events(events, trigger=None, **window)

    # from_preproc filters
    mixed = [
        _mk(tmp_path, space="T1w", preproc="deepprep"),
        _mk(tmp_path, space="T1w", preproc="fmriprep"),
    ]
    feat = ns.extractors.FmriExtractor(from_preproc="deepprep")
    relevant = feat._get_relevant_events(mixed, trigger=None, **window)
    assert len(relevant) == 1
    ta = feat._preprocess_event(relevant[0])  # type: ignore[arg-type]
    assert ta.header is not None and ta.header["preproc"] == "deepprep"

    # validation: unknown from_preproc / from_space
    feat = ns.extractors.FmriExtractor(from_preproc="nonexistent")
    with pytest.raises(ValueError, match="from_preproc"):
        feat.prepare(mixed)
    feat = ns.extractors.FmriExtractor(from_space="nonexistent")
    with pytest.raises(ValueError, match="from_space"):
        feat.prepare(mixed)

    # valid individually but bad combination
    cross = [
        _mk(tmp_path, space="T1w", preproc="deepprep"),
        _mk(tmp_path, space="MNI152NLin2009cAsym", preproc="fmriprep"),
    ]
    feat = ns.extractors.FmriExtractor(
        from_preproc="deepprep", from_space="MNI152NLin2009cAsym"
    )
    with pytest.raises(ValueError, match="after from_preproc"):
        feat.prepare(cross)


def test_fmri_smoothing(test_data_path: Path) -> None:
    study = ns.Study(
        name="Test2023Fmri",
        path=test_data_path,
        query="timeline_index<1",
        infra_timelines=_NO_CLUSTER,
    ).run()
    event = study.query('type=="Fmri"').iloc[:1]

    raw = ns.extractors.FmriExtractor(cleaning=None)
    smoothed = ns.extractors.FmriExtractor(fwhm=6.0, cleaning=None)

    data_raw = raw(event, start=0.0, duration=20.0).numpy()
    data_smooth = smoothed(event, start=0.0, duration=20.0).numpy()

    assert data_raw.shape == data_smooth.shape
    assert not np.allclose(data_raw, data_smooth), "Smoothing should change the data"
    assert np.std(data_smooth) < np.std(data_raw), "Smoothing should reduce variance"


def test_spikes_extractor(tmp_path: Path) -> None:
    import h5py  # type: ignore[import-untyped]

    n_units = 5
    sfreq = 50.0
    duration = 10.0
    fp = tmp_path / "spikes.nwb"
    rng = np.random.RandomState(42)
    all_spike_times: list[np.ndarray] = []
    spike_times_index: list[int] = []
    cumulative = 0
    for _ in range(n_units):
        n_spikes = rng.randint(20, 100)
        times = np.sort(rng.uniform(0, duration, size=n_spikes))
        all_spike_times.append(times)
        cumulative += n_spikes
        spike_times_index.append(cumulative)

    with h5py.File(fp, "w") as f:
        units = f.create_group("units")
        units.create_dataset("id", data=np.arange(n_units))
        units.create_dataset("spike_times", data=np.concatenate(all_spike_times))
        units.create_dataset("spike_times_index", data=np.array(spike_times_index))
        units.create_dataset("electrodes", data=np.arange(n_units))
        electrodes = f.create_group("general/extracellular_ephys/electrodes")
        electrodes.create_dataset("id", data=np.arange(n_units))

    event = dict(
        type="Spikes",
        start=0.0,
        duration=duration,
        frequency=sfreq,
        timeline="spikes_tl",
        filepath=str(fp),
        subject="sub-01",
    )

    extractor = ns.extractors.SpikesExtractor()
    data = extractor(event, start=0.0, duration=5.0)
    assert data.shape[0] == n_units
    assert data.shape[1] == int(5.0 * sfreq)
    assert data.max() > 0

    extractor = ns.extractors.SpikesExtractor(frequency=25.0)
    data = extractor(event, start=0.0, duration=5.0)
    assert data.shape[0] == n_units
    assert data.shape[1] == int(5.0 * 25.0)
    assert data.max() > 0

    extractor = ns.extractors.SpikesExtractor(baseline=(0.0, 1.0))
    data_bl = extractor(event, start=1.0, duration=3.0)
    assert data_bl.shape == (n_units, int(3.0 * sfreq))

    extractor = ns.extractors.SpikesExtractor(clamp=0.5)
    data_cl = extractor(event, start=0.0, duration=5.0)
    assert data_cl.abs().max() <= 0.5


# ---------------------------------------------------------------------------
# NaN/inf replacement in MneRaw
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fill_non_finite", [None, 0.0, -999.0])
def test_fill_non_finite(tmp_path: Path, fill_non_finite: float | None) -> None:
    sfreq = 100.0
    n_times = 200
    data = np.random.randn(3, n_times).astype(np.float32)
    data[0, 50] = np.nan
    data[1, 100] = np.inf
    data[2, 150] = -np.inf
    info = mne.create_info([f"CH{i}" for i in range(3)], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    fp = tmp_path / "nan_test_raw.fif"
    raw.save(str(fp), overwrite=True, verbose=False)

    event = ns.events.Event.from_dict(
        dict(
            type="Eeg",
            start=0.0,
            duration=n_times / sfreq,
            frequency=sfreq,
            timeline="t",
            filepath=str(fp),
            subject="sub-01",
        )
    )
    extractor = ns.extractors.EegExtractor(
        frequency=sfreq, fill_non_finite=fill_non_finite
    )
    tensor = torch.as_tensor(extractor(event, start=0.0, duration=n_times / sfreq).data)
    if fill_non_finite is None:
        assert not torch.all(torch.isfinite(tensor))
    else:
        assert torch.all(torch.isfinite(tensor))
        if fill_non_finite != 0.0:
            assert (tensor == fill_non_finite).any()


# ---------------------------------------------------------------------------
# ChannelPositions.build include_ref_eeg override for non-EEG
# ---------------------------------------------------------------------------


def test_channel_positions_build_include_ref_override() -> None:
    """build() overrides include_ref_eeg to False for MEG extractors."""
    meg = ns.extractors.MegExtractor()
    ch_pos = ns.extractors.ChannelPositions(
        neuro=meg,
        include_ref_eeg=False,
        n_spatial_dims=3,
    )
    built = ch_pos.build(meg)
    assert built.include_ref_eeg is False


# ---------------------------------------------------------------------------
# MEG + n_spatial_dims=2 is rejected
# ---------------------------------------------------------------------------


def test_channel_positions_meg_2d_rejected() -> None:
    """ChannelPositions rejects n_spatial_dims=2 for MegExtractor."""
    meg = ns.extractors.MegExtractor()
    with pytest.raises(
        NotImplementedError, match="n_spatial_dims=2 is not supported for MEG"
    ):
        ns.extractors.ChannelPositions(neuro=meg, n_spatial_dims=2)
