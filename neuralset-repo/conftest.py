# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pytest configuration for neuralset tests."""

import multiprocessing
from pathlib import Path

import matplotlib
import pytest

import neuralset as ns

matplotlib.use("Agg")


def _in_sandbox() -> bool:
    """Detect the Cursor sandbox (macOS SemLock restriction)."""
    try:
        multiprocessing.Lock()
        return False
    except (OSError, PermissionError):
        return True


IN_SANDBOX = _in_sandbox()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "sandbox_skip: skip when running inside the Cursor sandbox"
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if not IN_SANDBOX:
        return
    skip = pytest.mark.skip(reason="sandbox: multiprocessing blocked (macOS SemLock)")
    for item in items:
        if "sandbox_skip" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def test_data_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped temp directory shared across all tests for study data.

    Test studies generate synthetic data files on the first
    ``Study.run()`` call.  Sharing a single directory avoids
    regenerating those files in every test / module.

    Sub-folders for each registered test / fake study are pre-created so that
    ``_identify_study_subfolder`` resolves them automatically — tests can
    simply pass ``path=test_data_path`` without appending the study name.

    Use the function-scoped ``tmp_path`` for cache and infra directories that
    must remain isolated between tests.
    """
    root = tmp_path_factory.mktemp("test_data")
    for name in ns.Study.catalog():
        if name.startswith(("Test", "Fake")):
            (root / name).mkdir()
    return root
