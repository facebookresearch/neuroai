# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for neuralfetch.utils.bids.export (BidsExporter / study_to_bids)."""

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
    from neuralfetch.utils.bids.export import study_to_bids

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
    from neuralfetch.utils.bids.export import _annotation_descriptions

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
    from neuralfetch.utils.bids.export import study_to_bids

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


def _stim_row(etype: str, filepath: Path, start: float) -> dict:
    return {
        "type": etype,
        "start": start,
        "duration": 0.2,
        "timeline": "tl_0",
        "subject": "01",
        "filepath": str(filepath),
        "frequency": 250.0,
        "session": None,
        "task": "auditory",
        "run": None,
    }


def _events_tsv(bids_root: Path) -> pd.DataFrame:
    (fname,) = bids_root.rglob("*_events.tsv")
    return pd.read_csv(fname, sep="\t")


def test_study_to_bids_fnirs_rejected(tmp_path: Path) -> None:
    from neuralfetch.utils.bids.export import study_to_bids

    study = _make_study(pd.DataFrame(), path=tmp_path)
    with pytest.raises(ValueError, match="not supported"):
        study_to_bids(study, tmp_path, device="Fnirs", task="task")


def test_study_to_bids_audio_stimulus_exported(tmp_path: Path) -> None:
    wav = tmp_path / "stim" / "tone.wav"
    wav.parent.mkdir()
    wav.write_bytes(b"RIFF")
    bids_root = tmp_path / "bids_out"
    bids_root.mkdir()
    _run_study_to_bids(bids_root, extra_rows=[_stim_row("Audio", wav, 0.2)])

    assert (bids_root / "stimuli" / "tone.wav").exists()
    # BIDS stim_file is relative to the stimuli/ directory
    assert "tone.wav" in _events_tsv(bids_root)["stim_file"].tolist()


def test_study_to_bids_same_basename_stimuli_kept_apart(tmp_path: Path) -> None:
    """Distinct files sharing a basename are exported to distinct paths."""
    srcs = []
    for folder in ("a", "b"):
        src = tmp_path / "stim" / folder / "img.png"
        src.parent.mkdir(parents=True)
        src.write_bytes(folder.encode())
        srcs.append(src)
    bids_root = tmp_path / "bids_out"
    bids_root.mkdir()
    rows = [_stim_row("Image", src, 0.2 + 0.3 * i) for i, src in enumerate(srcs)]
    _run_study_to_bids(bids_root, extra_rows=rows)

    assert (bids_root / "stimuli" / "a" / "img.png").read_bytes() == b"a"
    assert (bids_root / "stimuli" / "b" / "img.png").read_bytes() == b"b"
    stim_files = set(_events_tsv(bids_root)["stim_file"].dropna())
    assert {"a/img.png", "b/img.png"} <= stim_files


def test_study_to_bids_colliding_timelines_get_distinct_runs(tmp_path: Path) -> None:
    """Timelines differing only in a non-BIDS field are numbered as runs."""
    from neuralfetch.utils.bids.export import study_to_bids

    fif_path = tmp_path / "source_fake_raw.fif"
    _make_fake_raw().save(str(fif_path), overwrite=True, verbose=False)
    blocks = []
    for block in range(2):
        df = _make_fake_events(fif_path)
        df["timeline"] = f"tl_block{block}"
        df["block"] = block
        blocks.append(df)
    study = _make_study(pd.concat(blocks, ignore_index=True), path=tmp_path)
    bids_root = tmp_path / "bids_out"
    study_to_bids(study, bids_root, device="Eeg", task="auditory")

    runs = sorted(p.name for p in bids_root.rglob("*_eeg.vhdr"))
    assert runs == [
        "sub-01_task-auditory_run-01_eeg.vhdr",
        "sub-01_task-auditory_run-02_eeg.vhdr",
    ]


def test_study_to_bids_overwrite_refreshes_stimulus(tmp_path: Path) -> None:
    src = tmp_path / "stim" / "img.png"
    src.parent.mkdir()
    src.write_bytes(b"old")
    bids_root = tmp_path / "bids_out"
    bids_root.mkdir()
    rows = [_stim_row("Image", src, 0.2)]
    _run_study_to_bids(bids_root, overwrite=True, extra_rows=rows)
    src.write_bytes(b"new")
    _run_study_to_bids(bids_root, overwrite=True, extra_rows=rows)

    assert (bids_root / "stimuli" / "img.png").read_bytes() == b"new"


def test_study_to_bids_subject_prefix_stripped(tmp_path: Path) -> None:
    """Subject strings like 'StudyName/01' are stripped to '01', written as 'sub-01'."""
    _run_study_to_bids(tmp_path, subject="StudyName/01")
    sub_dirs = [
        p.name for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("sub-")
    ]
    assert "sub-01" in sub_dirs, f"Expected 'sub-01' directory, found: {sub_dirs}"
