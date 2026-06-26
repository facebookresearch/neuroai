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


def test_construction_creates_no_directories(tmp_path: Path) -> None:
    """Constructing a MOABB study must not touch the filesystem.

    Regression: ``_BaseMoabb.model_post_init`` used to ``mkdir`` the study leaf
    at construction. That made ``self.path.exists()`` return True for
    never-downloaded studies (defeating the base "run study.download() first"
    guard) and littered empty directories for any study merely instantiated.
    """
    study = Tangermann2012Review(path=tmp_path)
    assert study.path == tmp_path / "moabb" / "Tangermann2012Review"
    assert not study.path.exists()
    assert not (tmp_path / "moabb").exists()


def test_undownloaded_study_raises_actionable_error(tmp_path: Path) -> None:
    """An undownloaded study surfaces the base download guidance, not a raw read error."""
    study = Tangermann2012Review(path=tmp_path)
    with pytest.raises(RuntimeError, match=r"study\.download\(\) first"):
        study._run()
