# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for moabb2025 module-level helpers and _BaseMoabb behaviour."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neuralfetch.studies import moabb2025
from neuralfetch.studies.moabb2025 import Tangermann2012Review


def test_get_cached_moabb_data_lru(tmp_path: Path) -> None:
    """_get_cached_moabb_data uses lru_cache(maxsize=2) and evicts correctly."""
    moabb2025._get_cached_moabb_data.cache_clear()

    fake_data = {1: {"ses": {"run": "raw1"}}}
    mock_moabb_base = MagicMock()

    with (
        patch.dict(
            sys.modules,
            {
                "moabb": MagicMock(),
                "moabb.datasets": MagicMock(),
                "moabb.datasets.base": mock_moabb_base,
            },
        ),
        patch.object(moabb2025, "find_dataset_in_moabb") as mock_find,
        patch.object(moabb2025, "temp_mne_data") as mock_ctx,
    ):
        mock_ds = MagicMock()
        mock_ds.get_data.return_value = fake_data
        mock_find.return_value = mock_ds
        mock_ctx.return_value.__enter__ = MagicMock()
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        result = moabb2025._get_cached_moabb_data("ds", tmp_path, 1)
        assert result == fake_data
        assert moabb2025._get_cached_moabb_data.cache_info().misses == 1

        moabb2025._get_cached_moabb_data("ds", tmp_path, 1)
        assert moabb2025._get_cached_moabb_data.cache_info().hits == 1

    assert moabb2025._get_cached_moabb_data.cache_info().maxsize == 2
    moabb2025._get_cached_moabb_data.cache_clear()


def test_basemoabb_disables_processpool(tmp_path: Path) -> None:
    study = Tangermann2012Review(path=tmp_path)
    assert study.infra_timelines.cluster != "processpool"

    study_pp = Tangermann2012Review(
        path=tmp_path,
        infra_timelines={"cluster": "processpool"},  # type: ignore[arg-type]
    )
    assert study_pp.infra_timelines.cluster is None


def test_download_subjects_subset(tmp_path: Path) -> None:
    """``download(subjects=[1])`` downloads only the requested subject."""
    study = Tangermann2012Review(path=tmp_path)

    mock_ds = MagicMock()
    mock_ds.subject_list = [1, 2, 3]
    mock_ds.get_data.return_value = {1: {"ses": {"run": "raw"}}}

    with (
        patch.dict(
            sys.modules,
            {
                "moabb": MagicMock(),
                "moabb.datasets": MagicMock(),
                "moabb.datasets.base": MagicMock(),
            },
        ),
        patch.object(moabb2025, "find_dataset_in_moabb", return_value=mock_ds),
        patch.object(moabb2025, "temp_mne_data") as mock_ctx,
    ):
        mock_ctx.return_value.__enter__ = MagicMock()
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        study._download(subjects=[1])

    # Only subject 1 was fetched; subjects 2 and 3 were skipped.
    called_subjects = [c.kwargs["subjects"] for c in mock_ds.get_data.call_args_list]
    assert called_subjects == [[1]]


def test_download_unknown_subject_raises(tmp_path: Path) -> None:
    """``download(subjects=[...])`` rejects subjects absent from the dataset."""
    study = Tangermann2012Review(path=tmp_path)

    mock_ds = MagicMock()
    mock_ds.subject_list = [1, 2, 3]

    with (
        patch.dict(
            sys.modules,
            {
                "moabb": MagicMock(),
                "moabb.datasets": MagicMock(),
                "moabb.datasets.base": MagicMock(),
            },
        ),
        patch.object(moabb2025, "find_dataset_in_moabb", return_value=mock_ds),
        patch.object(moabb2025, "temp_mne_data") as mock_ctx,
    ):
        mock_ctx.return_value.__enter__ = MagicMock()
        mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(ValueError, match="Unknown subject"):
            study._download(subjects=[99])

    mock_ds.get_data.assert_not_called()
