# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""One-off helper to seed ``StudyInfo.size_bytes`` from dataset host APIs.

Scans ``neuralfetch/studies/*.py`` for OpenNeuro accessions, queries the
OpenNeuro GraphQL API for each dataset's on-disk size, and prints the numbers
together with a ready-to-paste ``size_bytes=...`` line per study.

This is a developer convenience, not part of the docs build. OpenNeuro reports
the on-disk size of the latest snapshot in bytes (``latestSnapshot.size``),
which matches the "on-disk after extraction" semantics of ``size_bytes``.
Hosts without a uniform size API (Zenodo, Figshare, OSF, S3) are listed as
``unresolved`` for manual follow-up.

Usage::

    python docs/scripts/fetch_dataset_sizes.py
    python docs/scripts/fetch_dataset_sizes.py --studies-dir path/to/studies
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

_OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql"

# Matches OpenNeuro accessions in any of the forms used across study files:
#   download.Openneuro("ds004212", ...)
#   https://openneuro.org/datasets/ds004212/versions/2.0.0
#   github.com/OpenNeuroDatasets/ds004192.git
_OPENNEURO_RE = re.compile(r"(?:Openneuro\(\"|datasets/|OpenNeuroDatasets/)(ds\d{6,})")

# Hosts we cannot resolve to a single size with a uniform API call; surfaced so
# the maintainer knows which studies still need a manual or per-host number.
_OTHER_HOST_RE = re.compile(r"zenodo|figshare|osf\.io|s3://|datalad", re.IGNORECASE)


def _find_studies(studies_dir: Path) -> list[Path]:
    return sorted(path for path in studies_dir.glob("*.py") if path.name != "__init__.py")


def _openneuro_size(accession: str) -> int | None:
    """Return the latest-snapshot on-disk size in bytes, or ``None`` on failure."""
    query = "query($id: ID!) { dataset(id: $id) { latestSnapshot { size } } }"
    payload = json.dumps({"query": query, "variables": {"id": accession}}).encode()
    request = urllib.request.Request(
        _OPENNEURO_GRAPHQL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode())
    except Exception as exc:  # noqa: BLE001 - one-off script, report and move on
        print(f"  ! {accession}: request failed ({exc})", file=sys.stderr)
        return None
    snapshot = (body.get("data") or {}).get("dataset") or {}
    size = (snapshot.get("latestSnapshot") or {}).get("size")
    return int(size) if size else None


def _format_size(value: int | None) -> str:
    if not value:
        return "—"
    size = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024 or unit == "PB":
            precision = 0 if size >= 100 or unit == "B" else 1
            return f"{size:.{precision}f} {unit}"
        size /= 1024
    return f"{value} B"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = (
        Path(__file__).resolve().parents[2]
        / "neuralfetch-repo"
        / "neuralfetch"
        / "studies"
    )
    parser.add_argument(
        "--studies-dir",
        type=Path,
        default=default_dir,
        help="Directory containing the per-study modules.",
    )
    args = parser.parse_args()

    if not args.studies_dir.is_dir():
        parser.error(f"studies dir not found: {args.studies_dir}")

    # One row per (file, accession): a single file may host several study
    # classes (e.g. THINGS MEG + fMRI), each with its own dataset and size, so
    # we never sum across accessions — that mapping is the maintainer's call.
    resolved: list[tuple[str, str, int]] = []
    unresolved: list[str] = []
    for path in _find_studies(args.studies_dir):
        text = path.read_text(encoding="utf8")
        accessions = sorted(set(_OPENNEURO_RE.findall(text)))
        if accessions:
            for accession in accessions:
                size = _openneuro_size(accession)
                print(f"  {path.name}: {accession} -> {_format_size(size)}")
                if size:
                    resolved.append((path.name, accession, size))
        elif _OTHER_HOST_RE.search(text):
            unresolved.append(path.name)

    print("\n=== size_bytes (paste into the matching StudyInfo) ===")
    for name, accession, size in resolved:
        print(f"# {name}  ({accession}, {_format_size(size)})")
        print(f"size_bytes={size},")

    if unresolved:
        print("\n=== unresolved (non-OpenNeuro host; measure manually) ===")
        for name in unresolved:
            print(f"  {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
