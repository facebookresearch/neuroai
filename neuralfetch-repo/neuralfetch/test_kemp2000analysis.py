# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for where ``Kemp2000Analysis`` looks for its Sleep-EDF recordings."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralfetch.studies.kemp2000analysis import Kemp2000Analysis

_SUBJECT, _SESSION = 0, 1


def _seed(folder: Path) -> Path:
    """One PSG/hypnogram pair; ``iter_timelines`` globs names and reads nothing."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"SC4{_SUBJECT:02}{_SESSION}E0-PSG.edf").touch()
    (folder / f"SC4{_SUBJECT:02}{_SESSION}EC-Hypnogram.edf").touch()
    return folder


@pytest.fixture
def study(tmp_path):
    return Kemp2000Analysis(path=str(tmp_path))


def test_a_pre_s3_copy_is_read_where_it_lies(study) -> None:
    """The flat folder ``mne.datasets.sleep_physionet`` wrote is still found.

    Recordings moved to the mirror's ``sleep-edfx/1.0.0/sleep-cassette`` layout;
    without this fallback a copy fetched before that move yields no timelines,
    and downloading it again costs another 7 GB.
    """
    legacy = _seed(study.path / "physionet-sleep-data")

    assert study._recordings_path == legacy
    assert [t["subject"] for t in study.iter_timelines()] == [f"{_SUBJECT:02}"]

    study._download()
    assert not study._mirror_path.exists()


def test_the_mirror_layout_wins_when_both_are_present(study) -> None:
    _seed(study.path / "physionet-sleep-data")
    _seed(study._mirror_path)

    assert study._recordings_path == study._mirror_path
