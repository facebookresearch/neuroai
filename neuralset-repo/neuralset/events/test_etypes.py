# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import typing as tp
from pathlib import Path

import mne
import nibabel
import numpy as np
import pandas as pd
import PIL.Image
import pytest
import scipy.io.wavfile

import neuralset as ns
from neuralset import events
from neuralset.events import etypes


def test_lru(tmp_path: Path) -> None:
    fp = tmp_path / ".testdata/test.png"
    if not fp.exists():
        fp.parent.mkdir(parents=True, exist_ok=True)
        img = PIL.Image.fromarray(255 * np.ones((100, 100, 3), dtype=np.uint8))
        img.save(str(fp))
    event = etypes.Image(filepath=fp, start=0, duration=1, timeline="foo")
    event.read()
    # repeat for lru cache
    event.read()


def test_event_from_dict_with_nan() -> None:
    data = {
        "type": "Word",
        "start": 0,
        "timeline": "t",
        "text": "hello",
        "blublu": "extra_val",
        "language": np.nan,
        "duration": np.nan,
        "sentence_char": np.nan,
        "category": "verb",
    }
    word = ns.events.Event.from_dict(data)
    assert word.extra == {"category": "verb", "blublu": "extra_val"}
    out = word.to_dict()
    assert out["text"] == "hello"
    assert out["category"] == "verb"


def test_sound(tmp_path: Path) -> None:
    fp = tmp_path / "Sub1" / "Sub2" / "Sub3" / "noise.wav"
    fp.parent.mkdir(parents=True)
    Fs = 120
    y = np.random.randn(2 * Fs)
    scipy.io.wavfile.write(fp, Fs, y)
    event = etypes.Audio.from_dict(
        dict(type="Audio", start=1, timeline="whatever", filepath=fp, duration=np.nan)
    )
    assert event.duration == 2.0
    # short path
    assert event.study_relative_path() == fp
    event.extra["study"] = "SUB2"
    assert str(event.study_relative_path()) == "Sub2/Sub3/noise.wav"
    # SpecialLoader JSON filepath should pass through unchanged
    json_fp = '{"cls": "Fake", "method": "_load", "timeline": {"subject": "s1"}}'
    event.filepath = json_fp
    assert str(event.study_relative_path()) == json_fp


def test_meg(tmp_path: Path) -> None:
    fp = tmp_path / "test_raw.fif"
    n_channels, sfreq, duration = 4, 64, 60
    data = np.random.rand(n_channels, sfreq * duration)
    info = mne.create_info(n_channels, sfreq=sfreq)
    raw = mne.io.RawArray(data, info=info)
    raw.save(fp)

    event = ns.events.Event.from_dict(
        dict(
            type="Meg",
            start=0,
            timeline="whatever",
            subject="1",
            filepath=str(fp),
        )
    )
    assert isinstance(event, etypes.Meg)
    assert event.frequency == sfreq
    assert pytest.approx(event.duration, 1 / sfreq) == duration

    raw2 = event.read()
    assert isinstance(raw2, mne.io.Raw)


def test_fmri_duration() -> None:
    dummy_fmri_timeseries = np.zeros((1, 1, 1, 10), dtype=np.float32)
    nii = nibabel.Nifti1Image(dummy_fmri_timeseries, affine=np.eye(4))
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmpfile:
        nibabel.save(nii, tmpfile.name)
    fmri_event = dict(
        start=0,
        timeline="whatever",
        filepath=tmpfile.name,
        type="Fmri",
        space="custom",
        subject="janedoe",
        frequency=0.5,
    )
    # 10 timepoints / 0.5 Hz
    assert ns.events.Event.from_dict(fmri_event).duration == 20.0


def test_categorical_event() -> None:
    expected_subclasses = [
        etypes.Action,
        etypes.Stimulus,
        etypes.EyeState,
        etypes.Artifact,
        etypes.Seizure,
        etypes.SleepStage,
        etypes.Background,
    ]
    assert issubclass(etypes.CategoricalEvent, etypes.Event)
    for cls in expected_subclasses:
        assert issubclass(cls, etypes.CategoricalEvent), f"{cls} is not a subclass"


def test_action() -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="Action",
            start=1,
            duration=2,
            timeline="whatever",
            code=1,
            description="motor_left_fist",
        )
    )
    assert isinstance(event, etypes.Action)
    assert isinstance(event, etypes.CategoricalEvent)
    assert event.code == 1
    assert event.description == "motor_left_fist"


def test_stimulus() -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="Stimulus",
            start=1,
            duration=2,
            timeline="whatever",
            code=1,
            description="right_nostril_smell",
        )
    )
    assert isinstance(event, etypes.Stimulus)


@pytest.mark.parametrize("state", ("open", "closed"))
def test_eye_state(state: str) -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="EyeState",
            start=1,
            duration=2,
            timeline="whatever",
            state=state,
        )
    )
    assert isinstance(event, etypes.EyeState)
    assert event.state == state


@pytest.mark.parametrize(
    "state",
    (
        "bckg",
        "seiz",
        "gnsz",
        "fnsz",
        "spsz",
        "cpsz",
        "absz",
        "tnsz",
        "cnsz",
        "tcsz",
        "atsz",
        "mysz",
    ),
)
def test_seizure(state: str) -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="Seizure",
            start=0,
            duration=10,
            timeline="whatever",
            state=state,
        )
    )
    assert isinstance(event, etypes.Seizure)
    assert event.state == state


@pytest.mark.parametrize("state", ("eyem", "musc", "chew", "shiv", "elpp", "artf"))
def test_artifact(state: str) -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="Artifact",
            start=1,
            duration=2,
            timeline="whatever",
            state=state,
        )
    )
    assert isinstance(event, etypes.Artifact)
    assert event.state == state


@pytest.mark.parametrize("key", ("a", "<space>"))
def test_keystroke(key: str) -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="Keystroke",
            start=1,
            duration=2,
            timeline="whatever",
            text=key,
        )
    )
    assert isinstance(event, etypes.Keystroke)
    assert event.text == key


@pytest.mark.parametrize("stage", ("W", "N1", "N2", "N3", "R"))
def test_sleepstage(stage: str) -> None:
    event = ns.events.Event.from_dict(
        dict(
            type="SleepStage",
            start=0,
            duration=1,
            timeline="whatever",
            stage=stage,
        )
    )
    assert isinstance(event, etypes.SleepStage)
    assert event.stage == stage


def test_event_type_list() -> None:
    ns.Study.catalog()
    existing = [
        # general
        "Event",
        "BaseDataEvent",
        "BaseSplittableEvent",
        # brain
        "Fmri",
        "MneRaw",
        "Eeg",
        "Meg",
        "Emg",
        "Fnirs",
        "Ieeg",
        "Spikes",
        # image/video
        "Image",
        "_VideoImage",
        "Video",
        # audio
        "Audio",
        # text
        "BaseText",
        "Text",
        "Sentence",
        "Word",
        "Phoneme",
        # actions
        "Keystroke",
        "Action",
        "EyeState",
        # annotations
        "CategoricalEvent",
        "Background",
        "SleepStage",
        "Artifact",
        "Seizure",
        "Stimulus",
    ]
    # only check event classes owned by neuroai packages
    registered = {
        name
        for name, cls in ns.events.Event._CLASSES.items()
        if cls.__module__.startswith(("neuralset", "neuralfetch", "neuraltrain"))
    }
    msg = (
        f"Missing or additional event class detected, please update the list in {__file__} "
        "if this is an intended change, so as to reference existing event classes "
        "(but check beforehand if there is not already an existing equivalent class :) )"
    )
    assert set(existing) == registered, msg


def test_split_event(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    Fs = 120
    y = np.random.randn(5 * Fs)
    scipy.io.wavfile.write(fp, Fs, y)
    event = etypes.Audio.from_dict(
        dict(type="Audio", start=1, timeline="whatever", filepath=fp, duration=np.nan)
    )

    res = event._split(timepoints=[0, 1.49, 2 * 1.49, 3 * 1.49], min_duration=1.49)
    assert {ev.offset for ev in res} == {0, 1.49, 2 * 1.49}


def test_validate_events_bids_nan_recovery() -> None:
    """NaN/None in BIDS entity columns (e.g. after CSV round-trip) are recovered."""
    event: tp.Any = {
        "type": "Motor",
        "start": 0.0,
        "duration": 1.0,
        "timeline": "tl1",
        "subject": float("nan"),
        "task": None,
    }
    recovered = ns.events.standardize_events(pd.DataFrame([event]))
    for entity in events.utils.BIDS_ENTITIES:
        assert recovered[entity].iloc[0] == events.utils.BIDS_ENTITY_DEFAULT
        assert recovered[entity].dtype == object


def test_normalize_rejects_nan_duration() -> None:
    """Events with missing or NaN duration are rejected early (lightweight path)."""
    missing: tp.Any = {"type": "Motor", "start": 0.0, "timeline": "tl1"}
    with pytest.raises(ValueError, match="duration"):
        ns.events.standardize_events(pd.DataFrame([missing]), auto_fill=False)
    nan_dur: tp.Any = {
        "type": "Motor",
        "start": 0.0,
        "duration": float("nan"),
        "timeline": "tl1",
    }
    with pytest.raises(ValueError, match="NaN duration"):
        ns.events.standardize_events(pd.DataFrame([nan_dur]), auto_fill=False)


# --- Fmri helpers (shared with extractors/test_neuro.py) ---

FMRI_SHAPE_4D = (2, 3, 4, 10)


def _make_nifti(folder: Path, name: str, shape: tuple[int, ...] = FMRI_SHAPE_4D) -> Path:
    fp = folder / name
    nii = nibabel.Nifti2Image(np.random.rand(*shape).astype(np.float32), np.eye(4))
    nibabel.save(nii, fp)
    return fp


def _make_gifti(folder: Path, name: str, n_vertices: int = 2, n_times: int = 10) -> Path:
    fp = folder / name
    darrays = [
        nibabel.gifti.GiftiDataArray(np.random.rand(n_vertices).astype(np.float32))
        for _ in range(n_times)
    ]
    nibabel.save(nibabel.GiftiImage(darrays=darrays), fp)
    return fp


def make_fmri_event(
    folder: Path,
    *,
    space: str = "T1w",
    surface: bool = False,
    mask: bool = False,
    n_vertices: int = 5,
    n_times: int = 10,
    shape: tuple[int, ...] = FMRI_SHAPE_4D,
    **kw: tp.Any,
) -> etypes.Fmri:
    """Create a fake Fmri event backed by real NIfTI/GIfTI files on disk."""
    if surface:
        name = f"sub-01_hemi-L_space-{space}_bold.func.gii"
        fp = _make_gifti(folder, name, n_vertices, n_times)
        _make_gifti(folder, name.replace("hemi-L", "hemi-R"), n_vertices, n_times)
    else:
        fp = _make_nifti(folder, f"bold_{space}.nii.gz", shape)
    event_kw: dict[str, tp.Any] = dict(
        start=0, timeline="t", filepath=fp, subject="s1", frequency=1.0, space=space
    )
    if mask:
        event_kw["mask_filepath"] = str(
            _make_nifti(folder, f"mask_{space}.nii.gz", shape[:-1])
        )
    event_kw.update(kw)
    return etypes.Fmri(**event_kw)


@pytest.mark.parametrize("mask", [False, True])
def test_fmri_volume_read(tmp_path: Path, mask: bool) -> None:
    event = make_fmri_event(tmp_path, mask=mask, space="T1w", frequency=0.5)
    assert event.read().get_fdata().shape == FMRI_SHAPE_4D
    if mask:
        assert event.read_mask() is not None
        assert event.read_mask().get_fdata().shape == FMRI_SHAPE_4D[:-1]
    else:
        assert event.read_mask() is None


def test_fmri_surface_read(tmp_path: Path) -> None:
    n_v, n_t = 5, 8
    event = make_fmri_event(
        tmp_path, surface=True, space="fsaverage", n_vertices=n_v, n_times=n_t
    )
    assert event.read().get_fdata().shape == (n_v * 2, n_t)
    assert event.read_mask() is None
