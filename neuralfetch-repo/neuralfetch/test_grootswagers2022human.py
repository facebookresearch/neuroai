# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the ``Grootswagers2022Human`` events on the NEMAR on003825 layout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from neuralfetch.studies.grootswagers2022human import Grootswagers2022Human

_TRAIN = "stimuli/aardvark/aardvark_01b.jpg"
_TEST = "stimuli/zebra/zebra_12s.jpg"
_TRAIN_IN_TEST_CATEGORY = "stimuli/zebra/zebra_01b.jpg"


def _write_events(study: Grootswagers2022Human, subject: str, **cols) -> None:
    path = study._events_file(subject)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(path, sep="\t", index=False)


@pytest.fixture
def study(tmp_path: Path) -> Grootswagers2022Human:
    study = Grootswagers2022Human(path=tmp_path)
    for stim in (_TRAIN, _TEST, _TRAIN_IN_TEST_CATEGORY):
        img = study.path / "download" / "on003825" / stim
        img.parent.mkdir(parents=True, exist_ok=True)
        img.touch()
    # sub-02 saw the validation images and flags them with ``isteststim``
    _write_events(
        study,
        "2",
        onset=[10.5, 10.6, 10.7],
        duration=[0.05] * 3,
        stim=[_TRAIN, _TEST, _TRAIN_IN_TEST_CATEGORY],
        isteststim=[0, 1, 0],
    )
    # sub-01 never saw them: its events have no ``isteststim`` column
    _write_events(
        study,
        "1",
        onset=[10.724, 10.824],
        duration=[0.05] * 2,
        stim=[_TRAIN, _TRAIN_IN_TEST_CATEGORY],
    )
    return study


def test_test_split_comes_from_the_events(study: Grootswagers2022Human) -> None:
    images = study._load_timeline_events({"subject": "2"}).query("type == 'Image'")
    assert images["split"].tolist() == ["train", "test", "train"]
    assert images["is_test_category"].tolist() == [False, True, True]
    assert images["start"].tolist() == pytest.approx([10.5, 10.6, 10.7])
    assert images["duration"].tolist() == pytest.approx([0.05] * 3)
    assert Path(images["filepath"].iloc[1]).exists()


def test_subject_without_validation_images(study: Grootswagers2022Human) -> None:
    events = study._load_timeline_events({"subject": "1"})
    eeg = events.query("type == 'Eeg'")["filepath"].item()
    assert eeg.endswith("on003825/sub-01/eeg/sub-01_task-rsvp_eeg.vhdr")
    images = events.query("type == 'Image'")
    assert images["split"].tolist() == ["train", "train"]
    assert images["is_test_category"].tolist() == [False, True]
