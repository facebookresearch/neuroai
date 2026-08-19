# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for neuralfetch.cli.download.

The headline test here is :func:`test_all_registered_studies_accept_overwrite`,
the anti-regression guard for the download CLI: ``neuralfetch download`` always
forwards ``overwrite=<bool>`` down to ``Study._download``, so every study's
``_download`` must accept that keyword or the CLI dies with a ``TypeError``
before doing any work.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from neuralfetch.cli.download import _clean_study_dirs, download_study
from neuralset.events import study

# Studies whose ``_download`` carries a bespoke signature (positional
# credentials, dataset-specific selectors, etc.) and are intentionally deferred
# from the overwrite standardization. Tracked separately; see the download
# overwrite plan. Keep this list tight -- anything added here is a study the
# CLI still cannot drive with ``overwrite``. Entries are skipped only if they
# resolve to a registered study, so the list can't silently mask a fixable one.
_DEFERRED_CUSTOM_SIGNATURES = frozenset(
    {
        "VanEssen2012HumanMovie",
        "VanEssen2012HumanRest",
        "VanEssen2012HumanTask",
        "Shen2019Deep",
        "Zhou2023Large",
        "Gong2023Large",
        "Zhang2024Chisco",
    }
)


def _accepts_overwrite(func: object) -> bool:
    """True if *func* can be called with an ``overwrite=`` keyword."""
    params = inspect.signature(func).parameters  # type: ignore[arg-type]
    if "overwrite" in params:
        return True
    return any(p.kind is p.VAR_KEYWORD for p in params.values())


def test_all_registered_studies_accept_overwrite() -> None:
    """Every registered study's ``_download`` must accept ``overwrite=``.

    ``Study.download(**kwargs)`` forwards ``overwrite`` verbatim to
    ``_download``; a study missing the parameter makes ``neuralfetch download
    <Study>`` raise ``TypeError`` on every invocation (even without the flag,
    since the CLI always passes ``overwrite=args.overwrite``).
    """
    study._resolve_study("")  # full scan -> populate STUDIES

    checked = 0
    failures: list[str] = []
    for name, cls in sorted(study.STUDIES.items()):
        if name in _DEFERRED_CUSTOM_SIGNATURES:
            continue
        checked += 1
        if not _accepts_overwrite(cls._download):
            module = getattr(cls, "__module__", "")
            sig = inspect.signature(cls._download)
            failures.append(f"{name} ({module}): _download{sig}")

    assert checked, "no studies were scanned -- registry scan failed?"
    assert not failures, (
        "these studies' _download() cannot accept overwrite=, so "
        "`neuralfetch download <Study>` would raise TypeError:\n  "
        + "\n  ".join(failures)
    )


def test_clean_study_dirs_removes_only_artifacts(tmp_path: Path) -> None:
    """``_clean_study_dirs`` wipes download/ and prepare/ but nothing else."""
    (tmp_path / "download").mkdir()
    (tmp_path / "download" / "sub-01.fif").write_text("data")
    (tmp_path / "prepare").mkdir()
    (tmp_path / "prepare" / "cache.tsv").write_text("cache")
    (tmp_path / "notes.md").write_text("keep me")

    _clean_study_dirs(tmp_path, assume_yes=True)

    assert not (tmp_path / "download").exists()
    assert not (tmp_path / "prepare").exists()
    assert (tmp_path / "notes.md").exists()


def test_clean_study_dirs_aborts_without_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declined prompt must delete nothing and raise SystemExit."""
    (tmp_path / "download").mkdir()
    (tmp_path / "download" / "sub-01.fif").write_text("data")
    monkeypatch.setattr("builtins.input", lambda *_: "n")

    with pytest.raises(SystemExit):
        _clean_study_dirs(tmp_path, assume_yes=False)

    assert (tmp_path / "download").exists()


def test_clean_study_dirs_noop_when_absent(tmp_path: Path) -> None:
    """No download/ or prepare/ -> no prompt, no error."""
    _clean_study_dirs(tmp_path, assume_yes=False)  # must not raise or prompt


def test_download_study_unknown_name_raises() -> None:
    # ``_resolve_study`` raises ImportError for an unknown (non-empty) name
    # after scanning every registered study package.
    with pytest.raises(ImportError, match="NotARealStudy2099"):
        download_study(name="NotARealStudy2099", path="/tmp/does-not-matter")
