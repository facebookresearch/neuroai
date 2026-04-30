# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for neuralfetch: study discovery and study info validation."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
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


# ---------------------------------------------------------------------------
# Grootswagers2022HumanSample — regression test for missing image filepaths
# ---------------------------------------------------------------------------


def _make_fake_tsv(path: Path, stim_paths: list[str]) -> None:
    """Write a minimal events TSV that _load_timeline_events can parse."""
    rows = [
        {"onset": i * 0.1, "duration": 0.05, "stim": s} for i, s in enumerate(stim_paths)
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _grootswagers_study(tmp_path: Path):
    from neuralfetch.studies.grootswagers2022human import Grootswagers2022HumanSample

    return Grootswagers2022HumanSample(path=tmp_path / "Grootswagers2022HumanSample")


def test_format_img_path_prepare(tmp_path: Path) -> None:
    """_format_img_path resolves to prepare/ when the file exists there."""
    study = _grootswagers_study(tmp_path)
    img = study.path / "prepare" / "contact_lens" / "contact_lens_08s.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"")
    result = study._format_img_path("stimuli/contact_lens/contact_lens_08s.jpg")
    assert result == str(img)
    assert Path(result).exists()


def test_format_img_path_things_fallback(tmp_path: Path) -> None:
    """_format_img_path falls back to ../THINGS-images/ when prepare/ is missing."""
    study = _grootswagers_study(tmp_path)
    things_img = (
        study.path / ".." / "THINGS-images" / "contact_lens" / "contact_lens_08s.jpg"
    ).resolve()
    things_img.parent.mkdir(parents=True)
    things_img.write_bytes(b"")
    result = study._format_img_path("stimuli/contact_lens/contact_lens_08s.jpg")
    assert Path(result).exists()
    assert "THINGS-images" in result


def test_format_img_path_missing_raises(tmp_path: Path) -> None:
    """_format_img_path raises RuntimeError when the file is in neither location."""
    study = _grootswagers_study(tmp_path)
    with pytest.raises(RuntimeError, match="does not exist"):
        study._format_img_path("stimuli/contact_lens/contact_lens_08s.jpg")


def test_sample_load_timeline_events_absolute_paths(tmp_path: Path) -> None:
    """_load_timeline_events must return absolute, existing filepaths (regression for #27).

    Before the fix, the sample stored raw TSV paths like
    'stimuli/contact_lens/contact_lens_08s.jpg' — not absolute paths —
    causing FileNotFoundError downstream.
    """

    study = _grootswagers_study(tmp_path)
    sub = "01"
    eeg_folder = study.path / "download" / f"sub-{sub}" / "eeg"
    eeg_folder.mkdir(parents=True)

    stim_paths = [
        "stimuli/contact_lens/contact_lens_08s.jpg",
        "stimuli/airliner/airliner_01s.jpg",
    ]

    # Create fake image files in THINGS-images (where sample downloads them)
    for sp in stim_paths:
        parts = sp.split("/")
        img = (study.path / ".." / "THINGS-images" / parts[1] / parts[2]).resolve()
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(b"fake")

    # Create fake EEG vhdr file (filepath checked by iter_timelines)
    vhdr = eeg_folder / f"sub-{sub}_task-rsvp_eeg.vhdr"
    vhdr.write_text("")

    # Create TSV events file
    tsv = eeg_folder / f"sub-{sub}_task-rsvp_events.tsv"
    _make_fake_tsv(tsv, stim_paths)

    events = study._load_timeline_events({"subject": "1"})
    images = events.loc[events.type == "Image"]

    # All filepaths must be absolute and point to existing files
    for fp in images.filepath:
        p = Path(fp)
        assert p.is_absolute(), f"filepath is not absolute: {fp}"
        assert p.exists(), f"filepath does not exist: {fp}"


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
