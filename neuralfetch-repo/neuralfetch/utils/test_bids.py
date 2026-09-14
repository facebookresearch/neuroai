# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for neuralfetch.utils.bids (BidsExporter / study_to_bids)."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_fake_raw(device: str = "Eeg") -> mne.io.RawArray:
    """Return a minimal in-memory Raw with one channel matching *device*."""
    ch_type = "eeg" if device in ("Eeg", "Ieeg") else "mag" if device == "Meg" else "eeg"
    info = mne.create_info(["CH 001"], sfreq=250.0, ch_types=ch_type)
    return mne.io.RawArray(np.zeros((1, 250)), info, verbose=False)


def _make_fake_events(
    fif_path: Path,
    device: str = "Eeg",
    subject: str = "01",
    task: str = "auditory",
) -> pd.DataFrame:
    """Return a minimal events DataFrame for a single timeline.

    The device row is built from a real event object so its fields match
    what ``Eeg.from_dict`` (called inside ``study_to_bids``) expects.
    """
    from neuralset.events import etypes

    device_cls = getattr(etypes, device)
    device_event = device_cls(
        start=0.0,
        duration=1.0,
        timeline="tl_0",
        subject=subject,
        filepath=str(fif_path),
        frequency=250.0,
    )
    device_row = {**device_event.to_dict(), "session": None, "task": task, "run": None}

    return pd.DataFrame(
        [
            device_row,
            {
                "type": "Stimulus",
                "start": 0.1,
                "duration": 0.5,
                "timeline": "tl_0",
                "subject": subject,
                "filepath": None,
                "frequency": 250.0,
                "session": None,
                "task": task,
                "run": None,
                "description": "left_audio",
            },
        ]
    )


def _make_study(
    events: pd.DataFrame,
    path: Path = Path("."),
):
    """Return a minimal ``Study`` subclass instance backed by *events*.

    The returned object is a genuine ``Study`` subclass so it satisfies
    ``isinstance`` checks.  ``_run()`` is overridden to return the injected
    DataFrame directly, bypassing the normal timeline-loading pipeline.
    """
    import typing as tp

    from neuralset.events import study

    _events = events

    class _FakeStudy(study.Study):
        # study_to_bids only needs study.run() to yield the events table, so
        # override run() directly: the real Study.run() is a Scatter that maps
        # over iter_timelines()/_all_timelines(), which is irrelevant here (and
        # would raise "No timeline found" for this in-memory fake).
        def run(self, value: tp.Any = None) -> pd.DataFrame:
            return _events

        def iter_timelines(self) -> tp.Iterator[dict]:
            return iter([])

        def _load_timeline_events(self, timeline: dict) -> pd.DataFrame:
            return pd.DataFrame()

    return _FakeStudy(path=path)


def _run_study_to_bids(
    tmp_path: Path,
    device: str = "Eeg",
    task_param: str | None = "auditory",
    task_in_events: str = "auditory",
    subject: str = "01",
    overwrite: bool = False,
    anonymize: dict | None = None,
    extra_rows: list[dict] | None = None,
) -> Path:
    """Call ``study_to_bids`` with a mocked study backed by a real saved FIF file."""
    from neuralfetch.utils.bids import study_to_bids

    raw = _make_fake_raw(device)
    fif_path = tmp_path / "source_fake_raw.fif"
    raw.save(str(fif_path), overwrite=True, verbose=False)

    events = _make_fake_events(
        fif_path, device=device, subject=subject, task=task_in_events
    )
    if extra_rows:
        events = pd.concat([events, pd.DataFrame(extra_rows)], ignore_index=True)

    study = _make_study(events, path=tmp_path)

    return study_to_bids(
        study,
        tmp_path,
        device=device,
        task=task_param,
        overwrite=overwrite,
        anonymize=anonymize,
    )


# ---------------------------------------------------------------------------
# _annotation_descriptions tests
# ---------------------------------------------------------------------------


def test_annotation_descriptions() -> None:
    from neuralfetch.utils.bids import _annotation_descriptions

    df = pd.DataFrame(
        [
            {"type": "EyeState", "description": "open", "state": "open", "stage": "W"},
            {"type": "SleepStage", "description": None, "state": "closed", "stage": "N2"},
            {"type": "Artifact", "description": None, "state": None, "stage": "R"},
            {"type": "Stimulus"},
        ]
    )
    result = _annotation_descriptions(df).tolist()

    # description takes priority over state and stage
    assert result[0] == "EyeState/open"
    # state is used when description is absent
    assert result[1] == "SleepStage/closed"
    # stage is used when description and state are absent
    assert result[2] == "Artifact/R"
    # falls back to type alone when no label column is present
    assert result[3] == "Stimulus"


# ---------------------------------------------------------------------------
# study_to_bids tests
# ---------------------------------------------------------------------------


def test_study_to_bids_invalid_device(tmp_path: Path) -> None:
    from neuralfetch.utils.bids import study_to_bids

    study = _make_study(pd.DataFrame(), path=tmp_path)
    with pytest.raises(ValueError, match="not supported"):
        study_to_bids(study, tmp_path, device="Fmri", task="task")


def test_study_to_bids_returns_path(tmp_path: Path) -> None:
    result = _run_study_to_bids(tmp_path)
    assert result == tmp_path


def test_study_to_bids_creates_bids_structure(tmp_path: Path) -> None:
    _run_study_to_bids(tmp_path)
    assert (tmp_path / "dataset_description.json").exists()


def test_study_to_bids_task_from_parameter(tmp_path: Path) -> None:
    """Explicit *task* parameter is used even when a task column is present."""
    _run_study_to_bids(tmp_path, task_param="mytask", task_in_events="other")
    bids_files = list(tmp_path.rglob("*task-mytask*"))
    assert bids_files, "Expected BIDS files containing 'task-mytask'"


def test_study_to_bids_task_from_column(tmp_path: Path) -> None:
    """Task is read from the events 'task' column when no parameter is given."""
    _run_study_to_bids(tmp_path, task_param=None, task_in_events="columntask")
    bids_files = list(tmp_path.rglob("*task-columntask*"))
    assert bids_files, "Expected BIDS files containing 'task-columntask'"


def test_study_to_bids_missing_task_raises(tmp_path: Path) -> None:
    """An empty task value raises ValueError."""
    with pytest.raises(ValueError, match="[Tt]ask"):
        _run_study_to_bids(tmp_path, task_param=None, task_in_events="")


def test_study_to_bids_overwrite_false_raises(tmp_path: Path) -> None:
    """Calling study_to_bids a second time without overwrite=True raises an error."""
    _run_study_to_bids(tmp_path, overwrite=False)
    with pytest.raises(Exception):
        _run_study_to_bids(tmp_path, overwrite=False)


def test_study_to_bids_overwrite_true(tmp_path: Path) -> None:
    """Calling study_to_bids twice with overwrite=True succeeds."""
    _run_study_to_bids(tmp_path, overwrite=True)
    _run_study_to_bids(tmp_path, overwrite=True)
    assert (tmp_path / "dataset_description.json").exists()


def test_study_to_bids_anonymize(tmp_path: Path) -> None:
    """Passing anonymize does not raise an error."""
    _run_study_to_bids(tmp_path, anonymize={"daysback": 200})
    assert (tmp_path / "dataset_description.json").exists()


def test_study_to_bids_stimulus_files_copied(tmp_path: Path) -> None:
    """Image events with filepaths cause stimuli/ directory to be populated."""
    fake_image = tmp_path / "fake_image.png"
    fake_image.write_bytes(b"\x89PNG\r\n")

    extra_rows = [
        {
            "type": "Image",
            "start": 0.2,
            "duration": 0.2,
            "timeline": "tl_0",
            "subject": "01",
            "filepath": str(fake_image),
            "frequency": 250.0,
            "session": None,
            "task": "auditory",
            "run": None,
        }
    ]
    bids_root = tmp_path / "bids_out"
    bids_root.mkdir()
    _run_study_to_bids(bids_root, extra_rows=extra_rows)

    stimuli_dir = bids_root / "stimuli"
    assert stimuli_dir.exists(), "stimuli/ directory should be created"
    assert (stimuli_dir / "fake_image.png").exists(), "image file should be copied"


def test_study_to_bids_subject_prefix_stripped(tmp_path: Path) -> None:
    """Subject strings like 'StudyName/01' are stripped to '01', written as 'sub-01'."""
    _run_study_to_bids(tmp_path, subject="StudyName/01")
    sub_dirs = [
        p.name for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("sub-")
    ]
    assert "sub-01" in sub_dirs, f"Expected 'sub-01' directory, found: {sub_dirs}"
