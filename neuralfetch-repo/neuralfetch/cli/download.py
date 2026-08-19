# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Study download runner.

Download a registered study's raw dataset into the configured study root.
The CLI wiring lives in :mod:`neuralfetch.cli.main`; this module is
library-only.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from neuralfetch.utils import root_study_folder
from neuralset.events import study

logger = logging.getLogger(__name__)

# Sub-directories of a study folder that hold derived/downloaded artifacts and
# are therefore safe to wipe on a ``--clean`` re-download.
_CLEANABLE_SUBDIRS = ("download", "prepare")


def download_study(
    name: str,
    path: str | Path | None = None,
    overwrite: bool = False,
    clean: bool = False,
    assume_yes: bool = False,
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
        If True, force re-download even when a success-file exists (files
        already on disk are re-fetched/verified rather than trusted).
    clean : bool
        If True, delete the study's ``download/`` and ``prepare/`` folders
        first, then re-download from scratch (implies ``overwrite=True``).
        This is destructive; the caller is prompted for confirmation unless
        ``assume_yes`` is set.
    assume_yes : bool
        Skip the interactive confirmation for ``clean`` (for non-interactive
        use). Ignored when ``clean`` is False.
    """
    cls = study._resolve_study(name)
    if cls is None:
        raise ValueError(f"Unknown study: {name!r}")
    root = Path(path) if path is not None else root_study_folder()
    inst = cls(path=root)
    if clean:
        _clean_study_dirs(inst.path, assume_yes=assume_yes)
        overwrite = True
    inst.download(overwrite=overwrite)


def _clean_study_dirs(study_path: Path, assume_yes: bool = False) -> None:
    """Delete a study's derived-artifact sub-directories before a fresh download.

    Only ``download/`` and ``prepare/`` are removed; anything else in the study
    folder (manually staged data, notes, etc.) is left untouched. When
    ``assume_yes`` is False the user must confirm on stdin, since this
    permanently deletes data that may be expensive to re-fetch.
    """
    targets = [study_path / sub for sub in _CLEANABLE_SUBDIRS]
    existing = [t for t in targets if t.exists()]
    if not existing:
        return
    if not assume_yes:
        print("The following directories will be permanently deleted:")
        for t in existing:
            print(f"  {t}")
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            raise SystemExit("Aborted: nothing was deleted.")
    for t in existing:
        logger.info("Removing %s", t)
        shutil.rmtree(t)


def list_downloadable_studies() -> list[str]:
    """Return all registered study names (sorted)."""
    study._resolve_study("")  # full scan of registered packages
    return sorted(study.STUDIES.keys())
