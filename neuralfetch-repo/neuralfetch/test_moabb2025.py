# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for moabb2025 module-level helpers and _BaseMoabb behaviour."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
