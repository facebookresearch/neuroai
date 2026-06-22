# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pytest configuration for neuralset tests."""

from pathlib import Path

import matplotlib
import pytest

import neuralset as ns

matplotlib.use("Agg")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: end-to-end pipeline test (opt-in via `pytest -m slow`)"
    )


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
