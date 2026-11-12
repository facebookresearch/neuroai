# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import pytest
from publish_docs import publish


def _build(path: Path, marker: str) -> Path:
    for rel in ["index.html", "sub/page.html"]:
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(f"<html><head></head><body>{marker}</body></html>")
    (path / "objects.inv").write_text(marker)
    return path


def test_publish(tmp_path: Path) -> None:
    site = tmp_path / "site"
    (site / "neuralset").mkdir(parents=True)
    (site / "neuralset/index.html").write_text("legacy")
    publish(site, _build(tmp_path / "main", "main"), "main")
    assert (site / "neuralset/index.html").exists(), "legacy root kept until a release"
    page = (site / "dev/sub/page.html").read_text()
    assert 'src="../../version_switcher.js" data-version="dev"' in page
    publish(site, _build(tmp_path / "v040", "v0.4.0"), "v0.4.0")
    publish(site, _build(tmp_path / "v032", "v0.3.2"), "v0.3.2")  # backport
    assert json.loads((site / "versions.json").read_text()) == {
        "stable": "0.4",
        "versions": ["dev", "0.4", "0.3"],
    }
    assert "v0.4.0" in (site / "stable/index.html").read_text(), (
        "backport must not replace stable/"
    )
    assert not (site / "neuralset").exists(), "legacy root not cleaned"
    assert "../stable/sub/page.html" in (site / "sub/page.html").read_text()
    assert (site / "objects.inv").read_text() == "v0.4.0"
    assert (site / "dev/index.html").exists(), "root cleanup deleted a version"
    with pytest.raises(ValueError, match="vX.Y.Z"):
        publish(site, tmp_path / "main", "v0.5.0rc1")
