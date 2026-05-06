# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Study download runner.

Download a registered study's raw dataset into the configured study root.
The CLI wiring lives in :mod:`neuralfetch.cli.download`; this module is
library-only.
"""

from __future__ import annotations

from pathlib import Path

from neuralfetch.utils.base import root_study_folder
from neuralset.events import study


def download_study(
    name: str,
    path: str | Path | None = None,
    overwrite: bool = False,
) -> None:
    """Download a study's raw dataset into ``path/<StudyName>/``.

    Parameters
    ----------
    name : str
        Registered study class name (e.g. ``"Grootswagers2022Human"``).
    path : str | Path | None
        Root folder for study data. Defaults to
        :func:`~neuralfetch.utils.root_study_folder` (i.e. the
        ``NEURALSET_STUDY_FOLDER`` environment variable).
    overwrite : bool
        If True, force re-download even when a success-file exists.
    """
    cls = study._resolve_study(name)
    if cls is None:
        raise ValueError(f"Unknown study: {name!r}")
    root = Path(path) if path is not None else root_study_folder()
    cls(path=root).download(overwrite=overwrite)


def list_downloadable_studies() -> list[str]:
    """Return all registered study names (sorted)."""
    study._resolve_study("")  # full scan of registered packages
    return sorted(study.STUDIES.keys())
