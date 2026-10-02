# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the ``Ghassemi2018You`` sleep annotations."""

from __future__ import annotations

import numpy as np
import pytest

from neuralfetch.studies.ghassemi2018you import Ghassemi2018You

_SUBJECT = "tr99-0001"

# Challenge 2018 scoring starts minutes into the record: the first stage label of
# this synthetic record is at 360 s, the first N2 at 780 s, and an arousal runs
# from 800 s to 810 s. Samples are at the study's 200 Hz.
_SAMPLES = np.array([72000, 78000, 156000, 160000, 162000])
_LABELS = ["W", "N1", "N2", "(arousal_rera", "arousal_rera)"]
_CHANNELS = np.array([0, 0, 0, 1, 1])


@pytest.fixture
def sleep_events(tmp_path):
    """Sleep events of a record whose annotations do not start at time 0."""
    wfdb = pytest.importorskip("wfdb")
    study = Ghassemi2018You(path=str(tmp_path))
    record_dir = study._download_root() / "training" / _SUBJECT
    record_dir.mkdir(parents=True)
    wfdb.wrann(
        _SUBJECT,
        "arousal",
        _SAMPLES,
        symbol=['"'] * len(_SAMPLES),
        aux_note=_LABELS,
        chan=_CHANNELS,
        fs=200,
        write_dir=str(record_dir),
    )
    timeline = {"subject": _SUBJECT, "split": "train", "split_dir": "training"}
    return study._load_sleep_events(timeline)


def test_sleep_events_keep_the_record_time_axis(sleep_events) -> None:
    """Annotation starts are seconds from the start of the record.

    The first annotation of a Challenge 2018 record sits minutes after sample
    0; subtracting its onset shifted every stage (and the first N2) earlier by
    that record-specific lead-in.
    """
    stages = sleep_events[sleep_events["type"] == "SleepStage"]
    assert stages["stage"].tolist() == ["W", "N1", "N2"]
    assert stages["start"].tolist() == pytest.approx([360.0, 390.0, 780.0])
    assert stages["duration"].tolist() == pytest.approx([30.0, 390.0, 30.0])


def test_arousal_events_keep_the_record_time_axis(sleep_events) -> None:
    arousals = sleep_events[sleep_events["type"] == "SleepArousal"]
    assert arousals["state"].tolist() == ["rera"]
    assert arousals["start"].tolist() == pytest.approx([800.0])
    assert arousals["duration"].tolist() == pytest.approx([10.0])
