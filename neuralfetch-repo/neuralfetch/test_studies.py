# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for neuralfetch: study discovery and study info validation."""

import importlib.util
import sys
from pathlib import Path

import pytest
import requests

import neuralset as ns
from neuralfetch import utils
from neuralset.events import study as _study_mod

INFO_STUDIES = [n for n, c in ns.Study.catalog().items() if c._info is not None]


def test_neuralfetch_discovery() -> None:
    """Check that neuralfetch studies are discovered by neuralset."""
    fetch_studies = {
        name: cls
        for name, cls in ns.Study.catalog().items()
        if cls.__module__.startswith("neuralfetch.")
    }
    assert fetch_studies, (
        "neuralfetch is installed but no studies were discovered. "
        "Check the neuralset.studies entry point in pyproject.toml."
    )


@pytest.mark.parametrize("name", INFO_STUDIES)
def test_study_info(name: str, tmp_path: Path) -> None:
    """Validate that a study's declared ``_info`` matches its actual data.

    Loads each study that provides a ``StudyInfo``, computes real values
    (num_timelines, num_subjects, num_events, event_types, data_shape,
    frequency, fmri_spaces) and asserts they match the declared metadata.
    """
    # to run one case only, use for instance:
    # pytest neuralfetch/test_studies.py::'test_study_info[Li2022Lppc]'
    try:
        folder = utils.root_study_folder(name, test_folder=tmp_path)
    except RuntimeError as e:
        pytest.skip(str(e))
    if not folder.exists():
        pytest.skip(f"Missing folder {folder} for study {name}")
    study = _study_mod.STUDIES[name](path=folder)
    if study.path == folder and folder.name.lower() != name.lower():
        # path was not updated from generic to study-specific
        pytest.skip(f"Study data not found for {name} in {folder}")
    assert study._info is not None
    try:
        actual = utils.compute_study_info(name, folder)
    except requests.exceptions.ConnectionError:
        pytest.skip(f"Network error loading {name}")
    mismatches: list[str] = []
    for key, val in actual.items():
        exp = getattr(study._info, key)
        if isinstance(val, set):
            # types in output of compute_study_info serve
            # as expected type
            exp = set(exp)
        if isinstance(val, float):
            if val != pytest.approx(exp, rel=0.01):  # type: ignore
                mismatches.append(key)
        elif val != exp:
            mismatches.append(key)
    if mismatches:
        expected_info = {k: getattr(study._info, k) for k in actual}
        expected_str = utils.format_study_info(expected_info)
        actual_str = utils.format_study_info(actual)
        msg = (
            f"For {name}\nExpected:\n{expected_str}\n"
            f"Mismatched keys: {mismatches}\n"
            f"Actual:\n{actual_str}\n"
            f'Auto-fix: python -c "from neuralfetch.utils import update_source_info;'
            f" update_source_info('{name}')\""
        )
        raise AssertionError(msg)


_FAKE_STUDY_SOURCE = """\
import typing as tp
import pandas as pd
from neuralset.events import study

class DummyUpdateTest2099(study.Study):
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo()
    def iter_timelines(self):
        yield from ({"subject": f"s{i}"} for i in range(2))
    def _load_timeline_events(self, timeline):
        return pd.DataFrame([{"type": "Stimulus", "start": 0, "duration": 1, "code": 1}])
"""


def test_update_source_info(tmp_path: Path) -> None:
    study_file = tmp_path / "dummy_study.py"
    study_file.write_text(_FAKE_STUDY_SOURCE)
    spec = importlib.util.spec_from_file_location("dummy_study", study_file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    sys.modules["dummy_study"] = mod
    try:
        actual = utils.update_source_info("DummyUpdateTest2099", folder=tmp_path)
        assert actual["num_timelines"] == 2
        new_source = study_file.read_text("utf8")
        for key in actual:
            assert f"{key}=" in new_source, f"{key} missing from rewritten source"
    finally:
        _study_mod.STUDIES.pop("DummyUpdateTest2099", None)
        sys.modules.pop("dummy_study", None)
