# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for moabb2025 module-level helpers and _BaseMoabb behaviour."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pandas as pd
from exca.steps import backends

from neuralfetch.studies import moabb2025
from neuralfetch.studies.moabb2025 import Romani2025Brainform, Tangermann2012Review


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


def test_load_raw_does_not_hand_out_the_cached_raw(tmp_path: Path) -> None:
    """Preprocessing a loaded raw in place must not change the next load."""
    raw = mne.io.RawArray(np.ones((2, 100)), mne.create_info(["C3", "C4"], 100.0, "eeg"))
    tangermann = Tangermann2012Review(path=tmp_path)
    (tangermann.path / "download").mkdir()
    romani = Romani2025Brainform(path=tmp_path)
    eeg_dir = romani._bids_root() / "sub-P01" / "ses-1" / "eeg"  # noqa: SLF001
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-P01_ses-1_task-ERP_eeg.edf").touch()
    events = pd.DataFrame({"onset": [0.5], "duration": [0.1], "value": [1]})
    events.to_csv(eeg_dir / "sub-P01_ses-1_task-ERP_events.tsv", sep="\t", index=False)
    modules = ["moabb", "moabb.datasets", "moabb.datasets.base"]
    moabb2025._get_cached_moabb_data.cache_clear()
    with (
        patch.dict(moabb2025._moabb_data_cache, clear=True),
        patch.dict(sys.modules, {name: MagicMock() for name in modules}),
        patch.object(moabb2025, "temp_mne_data"),
        patch.object(moabb2025, "find_dataset_in_moabb") as mock_find,
        patch.object(mne.io, "read_raw_edf", lambda *a, **k: raw.copy()),
    ):
        mock_find.return_value.get_data.return_value = {1: {"0": {"0": raw}}}
        for study, timeline in [
            (tangermann, {"subject": 1, "session": "0", "run": "0"}),
            (romani, {"subject": 0, "session": "1", "run": "1calibration"}),
        ]:
            first = study._load_raw(timeline)  # noqa: SLF001
            first.pick(["C3"])._data[:] = 0.0  # in place, as MneRaw preprocessing does
            second = study._load_raw(timeline)  # noqa: SLF001
            assert second.ch_names == ["C3", "C4"]
            assert np.all(second.get_data() == 1.0)
    moabb2025._get_cached_moabb_data.cache_clear()


def test_basemoabb_disables_processpool(tmp_path: Path) -> None:
    # an explicit ProcessPool is downgraded to Cached (inline) to avoid ENOLCK
    study = Tangermann2012Review(
        path=tmp_path,
        timelines={"infra": {"backend": "ProcessPool"}},  # type: ignore[arg-type]
    )
    assert isinstance(study.timelines.infra, backends.Cached)
