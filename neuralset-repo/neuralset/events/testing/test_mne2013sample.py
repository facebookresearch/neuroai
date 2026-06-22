# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for MNE sample-data path resolution (issue #153).

These mock ``mne.datasets.sample.data_path`` so they need no network and no
1.65 GB download: the fake records every root it is asked for and creates a
minimal tree, letting us assert that ``download()`` and ``run()`` funnel
through a single root and forward ``verbose=True`` (the progress bar).
"""

import typing as tp
from pathlib import Path

import mne
import pytest

from neuralset.events.testing.mne2013sample import Fake2025Meg, Mne2013Sample


@pytest.fixture
def data_path_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    """Patch the MNE sample downloader; return the list of (root, verbose) calls."""
    calls: list = []

    def _fake(path, *args, verbose=None, **kwargs):
        calls.append((Path(path), verbose))
        out = Path(path) / "MNE-sample-data"
        (out / "MEG" / "sample").mkdir(parents=True, exist_ok=True)
        return out

    monkeypatch.setattr(mne.datasets.sample, "data_path", _fake)
    return calls


def test_download_and_read_share_single_root(
    tmp_path: Path, data_path_calls: list
) -> None:
    infra: tp.Any = {"cluster": None}
    study = Mne2013Sample(path=tmp_path, infra_timelines=infra)
    root = study._download_root()

    study._download()  # writer
    data_path = study._get_data_path()  # reader

    # #153: writer and reader resolve to the same single root ...
    assert {called_root for called_root, _ in data_path_calls} == {root}
    assert root == (tmp_path / "Mne2013Sample" / "download").absolute()
    assert data_path == root / "MNE-sample-data" / "MEG" / "sample"
    # ... so only one copy of the dataset is ever created (not three)
    assert len(list(tmp_path.rglob("MNE-sample-data"))) == 1
    # issue (2): verbose=True forwarded -> progress bar on every fetch
    assert all(verbose is True for _, verbose in data_path_calls)


def test_fake2025meg_iter_timelines_uses_same_root(
    tmp_path: Path, data_path_calls: list
) -> None:
    infra: tp.Any = {"cluster": None}
    study = Fake2025Meg(path=tmp_path, infra_timelines=infra)
    list(study.iter_timelines())  # pre-download hook before concurrent loading
    assert {called_root for called_root, _ in data_path_calls} == {study._download_root()}
    assert all(verbose is True for _, verbose in data_path_calls)
