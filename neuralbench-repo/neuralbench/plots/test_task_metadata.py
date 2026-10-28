# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`neuralbench.plots._task_metadata`."""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralbench import registry
from neuralbench.plots import _task_metadata

_CONFIG = """data:
  study:
    source:
      name: {study}
"""


def _write_task(root: Path, device: str, task: str, study: str) -> None:
    config = root / device / task / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(_CONFIG.format(study=study), "utf8")


@pytest.fixture
def two_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    shipped, extension = tmp_path / "shipped", tmp_path / "extension"
    _write_task(shipped, "eeg", "age", "Shirazi2024Hbn")
    _write_task(extension, "meg", "age", "McNabb2025Wand")
    roots = [shipped, extension]
    monkeypatch.setattr(registry, "_all_task_roots", lambda: roots)
    return roots


def test_default_studies_keep_one_entry_per_root(two_roots: list[Path]):
    assert _task_metadata.default_studies_per_task() == {
        "age": ["Shirazi2024Hbn", "McNabb2025Wand"]
    }


def test_plugin_dataset_inherits_split_of_its_own_task(two_roots: list[Path]):
    shipped, extension = two_roots
    other = extension / "eeg" / "aaa" / "config.yaml"
    other.parent.mkdir(parents=True)
    other.write_text(
        _CONFIG.format(study="Other2026Eeg") + "    split:\n      split_by: subject\n"
    )
    dataset = extension / "eeg" / "age" / "datasets" / "plugin2026.yaml"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(_CONFIG.format(study="Plugin2026Eeg"))
    split_map = _task_metadata.build_task_split_map()
    assert split_map[("aaa", "Other2026Eeg")] == "cross_subject"
    assert split_map[("age", "Plugin2026Eeg")] == split_map[("age", "Shirazi2024Hbn")]


def test_explicit_tasks_dir_overrides_registered_roots(two_roots: list[Path]):
    assert _task_metadata.default_studies_per_task(two_roots[1]) == {
        "age": ["McNabb2025Wand"]
    }
