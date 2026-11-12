# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import pytest
from publish_docs import publish

_SOURCE = "https://github.com/facebookresearch/neuroai/blob/main/x.py"


def _build(path: Path, marker: str) -> Path:
    for rel in ["index.html", "sub/page.html"]:
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(
            f'<html><head></head><a href="{_SOURCE}">{marker}</a></html>'
        )
    (path / "objects.inv").write_text(marker)
    return path


def test_publish(tmp_path: Path) -> None:
    site = tmp_path / "site"
    (site / "neuralset").mkdir(parents=True)
    (site / "neuralset/index.html").write_text("legacy")
    (site / "CNAME").write_text("docs.example.org")
    publish(site, _build(tmp_path / "main", "main"), "main")
    assert (site / "neuralset/index.html").read_text() == "legacy", "kept until a release"
    page = (site / "dev/sub/page.html").read_text()
    assert 'src="../../version_switcher.js" data-version="dev"' in page
    assert _SOURCE in page and "noindex" in page
    publish(site, _build(tmp_path / "v041", "v0.4.1"), "v0.4.1")
    publish(site, _build(tmp_path / "v032", "v0.3.2"), "v0.3.2")  # backport
    assert json.loads((site / "versions.json").read_text()) == {
        "stable": "0.4",
        "versions": ["dev", "0.4", "0.3"],
        "tags": {"0.4": "v0.4.1", "0.3": "v0.3.2"},
    }
    stable = (site / "stable/index.html").read_text()
    assert "v0.4.1" in stable, "backport must not replace stable/"
    assert "blob/v0.4.1/x.py" in stable and "noindex" not in stable
    assert "../stable/sub/page.html" in (site / "sub/page.html").read_text()
    legacy = (site / "neuralset/index.html").read_text()
    assert "../stable/index.html" in legacy, "legacy route missing from stable/ dropped"
    assert (site / "objects.inv").read_text() == "v0.4.1"
    assert (site / "dev/index.html").exists(), "root cleanup deleted a version"
    assert (site / "CNAME").exists(), "root cleanup deleted CNAME"
    for ref, error in [
        ("v0.4.0", "older than the published v0.4.1"),
        ("v0.5.0rc1", "vX.Y.Z"),
    ]:
        with pytest.raises(ValueError, match=error):
            publish(site, tmp_path / "main", ref)
