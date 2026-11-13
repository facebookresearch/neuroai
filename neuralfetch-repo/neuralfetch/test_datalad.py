# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DataLad integration tests using only small local git-annex datasets."""

import shutil
import typing as tp
from pathlib import Path

import pytest

from neuralfetch import download

pytest.importorskip("datalad", minversion="1.8")
dlad = pytest.importorskip("datalad.api")
_CONTENTS = {
    name: name.encode() * 128
    for name in (
        "derivatives/a.bin",
        "derivatives/nested/a.bin",
        "sub-01/a.bin",
        "sub-02/a.bin",
    )
}


@pytest.fixture(scope="module")
def source_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("git-annex") is None:
        pytest.skip("git-annex is required")
    root = tmp_path_factory.mktemp("datalad-source") / "source"
    dataset = dlad.create(path=root)
    dlad.create(dataset=dataset, path=root / "derivatives")
    dlad.create(dataset=root / "derivatives", path=root / "derivatives/nested")
    for name, content in _CONTENTS.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    dataset.save(recursive=True)
    records = dataset.status(recursive=True, annex="availability")
    assert all(record["has_content"] for record in records if "key" in record)
    assert sum("key" in record for record in records) == len(_CONTENTS)
    return root


def _downloader(source: Path, destination: Path, **selection: tp.Any) -> download.Datalad:
    return download.Datalad(
        study="s", dset_dir=destination, repo_url=str(source), **selection
    )


def _fetched(obj: download.Datalad) -> set[str]:
    root = obj._dl_dir / obj.repo_name
    return {
        name
        for name, content in _CONTENTS.items()
        if (root / name).exists() and (root / name).read_bytes() == content
    }


@pytest.mark.parametrize(
    "selection,expected",
    [
        ({"include": ["sub-03/**"]}, set()),
        ({"include": ["sub-*"], "exclude": ["sub-*"]}, set()),
        ({"exclude": ["*"]}, set()),
        ({"include": ["sub-01/**"]}, {"sub-01/a.bin"}),
        (
            {"include": ["derivatives/**"]},
            {"derivatives/a.bin", "derivatives/nested/a.bin"},
        ),
        ({"include": ["derivatives/nested/**"]}, {"derivatives/nested/a.bin"}),
        ({"exclude": ["derivatives"]}, {"sub-01/a.bin", "sub-02/a.bin"}),
        (
            {"include": ["derivatives"], "exclude": ["derivatives/nested"]},
            {"derivatives/a.bin"},
        ),
        ({}, set(_CONTENTS)),
    ],
)
def test_datalad_local_selection(
    source_dataset: Path, tmp_path: Path, selection: dict, expected: set[str]
) -> None:
    obj = _downloader(source_dataset, tmp_path / "target", **selection)
    obj.download()
    assert _fetched(obj) == expected


@pytest.mark.parametrize(
    "first,second,expected",
    [
        (
            {"include": ["sub-01/**"]},
            {"include": ["sub-01/**", "sub-02/**"]},
            {"sub-01/a.bin", "sub-02/a.bin"},
        ),
        (
            {"include": ["sub-*"], "exclude": ["sub-02"]},
            {"include": ["sub-*"]},
            {"sub-01/a.bin", "sub-02/a.bin"},
        ),
        ({"include": ["sub-01"]}, {}, set(_CONTENTS)),
        ({}, {"include": ["sub-01"]}, set(_CONTENTS)),
    ],
)
def test_datalad_local_selection_changes(
    source_dataset: Path, tmp_path: Path, first: dict, second: dict, expected: set[str]
) -> None:
    destination = tmp_path / "target"
    _downloader(source_dataset, destination, **first).download()
    obj = _downloader(source_dataset, destination, **second)
    obj.download()
    assert _fetched(obj) == expected


def test_datalad_local_selection_overrides_wanted_policy(
    source_dataset: Path, tmp_path: Path
) -> None:
    destination = tmp_path / "target"
    obj = _downloader(source_dataset, destination, include=["sub-01"])
    obj.download()
    root = obj._dl_dir / obj.repo_name
    original_file = root / "sub-01/a.bin"
    original_mtime = original_file.stat().st_mtime_ns
    dlad.Dataset(root).repo.set_preferred_content("wanted", "exclude=*")
    obj = _downloader(source_dataset, destination, include=["sub-*"])
    obj.download()
    assert _fetched(obj) == {"sub-01/a.bin", "sub-02/a.bin"}
    assert original_file.stat().st_mtime_ns == original_mtime


def test_datalad_local_old_marker_is_rechecked(
    source_dataset: Path, tmp_path: Path
) -> None:
    obj = _downloader(source_dataset, tmp_path / "target")
    obj.get_success_file().write_text("success")
    obj.download()
    assert _fetched(obj) == set(_CONTENTS)
