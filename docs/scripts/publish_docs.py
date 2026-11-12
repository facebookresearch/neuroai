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
_KEEP = {".git", ".nojekyll", "CNAME", "robots.txt", "404.html", "versions.json"}
_BLOB = "https://github.com/facebookresearch/neuroai/blob/"
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


def _number(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", version))


def _copy_build(build: Path, dest: Path, version: str, ref: str) -> None:
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(build, dest)
    for page in dest.rglob("*.html"):
        root = "../" * len(page.relative_to(dest).parts)
        tag = f'<script defer src="{root}{_SWITCHER.name}" data-version="{version}" data-root="{root}"></script>'
        if version != "stable":  # search engines index stable/ only
            tag += '<meta name="robots" content="noindex">'
        html = page.read_text(encoding="utf8").replace("</head>", tag + "</head>", 1)
        if ref != "main":  # source links to the tag (edit links stay on main)
            html = html.replace(f"{_BLOB}main/", f"{_BLOB}{ref}/")
        page.write_text(html, encoding="utf8")


def _write_root_redirects(site: Path, keep: set[str]) -> None:
    old = [entry for entry in site.iterdir() if entry.name not in keep]
    routes = {
        p.relative_to(site)
        for entry in old
        for p in ([entry] if entry.is_file() else entry.rglob("*.html"))
        if p.suffix == ".html"
    }
    for entry in old:
        shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
    stable = site / "stable"
    pages = {p.relative_to(stable) for p in stable.rglob("*.html")}
    for rel in pages | routes:  # routes missing from stable/ go to its index
        dest = rel if rel in pages else Path("index.html")
        target = "../" * (len(rel.parts) - 1) + f"stable/{dest.as_posix()}"
        (site / rel).parent.mkdir(parents=True, exist_ok=True)
        (site / rel).write_text(_REDIRECT.format(target=target), encoding="utf8")
    shutil.copy(stable / "objects.inv", site / "objects.inv")  # intersphinx


def publish(site: Path, build: Path, ref: str) -> None:
    version = _version_dir(ref)
    index = site / "versions.json"
    tags = json.loads(index.read_text())["tags"] if index.exists() else {}
    published = tags.get(version)
    if version != "dev" and published is not None and _number(published) > _number(ref):
        raise ValueError(f"{ref} is older than the published {published} of {version}/")
    _copy_build(build, site / version, version, ref)
    if version != "dev":
        tags[version] = ref
    releases = sorted(
        (d.name for d in site.iterdir() if re.fullmatch(r"\d+\.\d+", d.name)),
        key=_number,
        reverse=True,
    )
    versions = (["dev"] if (site / "dev").is_dir() else []) + releases
    stable = releases[0] if releases else None
    index.write_text(
        json.dumps({"stable": stable, "versions": versions, "tags": tags}, indent=2)
    )
    shutil.copy(_SWITCHER, site / _SWITCHER.name)
    (site / ".nojekyll").touch()
    if version == stable:
        _copy_build(build, site / "stable", "stable", ref)
        _write_root_redirects(site, _KEEP | {_SWITCHER.name, "stable", *versions})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path, help="gh-pages checkout")
    parser.add_argument("build", type=Path, help="Sphinx HTML output")
    parser.add_argument("ref", help="'main' or a vX.Y.Z tag")
    args = parser.parse_args()
    publish(args.site, args.build, args.ref)
