# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Publish a Sphinx HTML build into the versioned gh-pages tree: ``dev/`` (main),
``X.Y/`` (latest vX.Y.* tag), ``stable/`` (copy of the highest X.Y), ``versions.json``
and redirects from the unversioned root URLs to ``stable/``.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

_SWITCHER = Path(__file__).with_name("version_switcher.js")
_REDIRECT = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Redirecting</title>
<meta http-equiv="refresh" content="0; url={target}">
<script>location.replace("{target}" + location.search + location.hash)</script>
</head></html>
"""


def _version_dir(ref: str) -> str:
    if ref == "main":
        return "dev"
    match = re.fullmatch(r"v(\d+)\.(\d+)\.\d+", ref)
    if match is None:
        raise ValueError(
            f"Cannot publish docs for ref {ref!r}: expected 'main' or a vX.Y.Z tag"
        )
    return f"{match[1]}.{match[2]}"


def _copy_build(build: Path, dest: Path, version: str) -> None:
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(build, dest)
    for page in dest.rglob("*.html"):
        root = "../" * len(page.relative_to(dest).parts)
        tag = f'<script defer src="{root}{_SWITCHER.name}" data-version="{version}" data-root="{root}"></script>'
        html = page.read_text(encoding="utf8")
        page.write_text(html.replace("</head>", tag + "</head>", 1), encoding="utf8")


def _write_root_redirects(site: Path, keep: set[str]) -> None:
    for entry in site.iterdir():
        if entry.name not in keep:
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    stable = site / "stable"
    for page in stable.rglob("*.html"):
        rel = page.relative_to(stable)
        target = "../" * (len(rel.parts) - 1) + f"stable/{rel.as_posix()}"
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_text(_REDIRECT.format(target=target), encoding="utf8")
    shutil.copy(stable / "objects.inv", site / "objects.inv")  # intersphinx


def publish(site: Path, build: Path, ref: str) -> None:
    version = _version_dir(ref)
    _copy_build(build, site / version, version)
    releases = sorted(
        (d.name for d in site.iterdir() if re.fullmatch(r"\d+\.\d+", d.name)),
        key=lambda v: tuple(int(x) for x in v.split(".")),
        reverse=True,
    )
    versions = (["dev"] if (site / "dev").is_dir() else []) + releases
    stable = releases[0] if releases else None
    (site / "versions.json").write_text(
        json.dumps({"stable": stable, "versions": versions}, indent=2)
    )
    shutil.copy(_SWITCHER, site / _SWITCHER.name)
    (site / ".nojekyll").touch()
    if version == stable:
        _copy_build(build, site / "stable", "stable")
        keep = {".git", ".nojekyll", "versions.json", _SWITCHER.name, "stable", *versions}
        _write_root_redirects(site, keep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="gh-pages checkout")
    parser.add_argument("build", type=Path, help="Sphinx HTML output")
    parser.add_argument("ref", help="'main' or a vX.Y.Z tag")
    args = parser.parse_args()
    publish(args.site, args.build, args.ref)
