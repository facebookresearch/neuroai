# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file
# in the root directory of this source tree.

"""Pytest configuration for tests collected under docs/.

Registers the `slow` marker (reused from neuralset-repo), skips slow
tests unless `--slow` is passed, and exposes a session-scoped
``fixtures_dir`` that materializes Code Builder fixtures via the JS
renderer (no committed fixtures, no drift).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_FIXTURES_GEN = Path(__file__).parent / "_data" / "render_fixtures.mjs"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--slow",
        action="store_true",
        default=False,
        help="run end-to-end (slow) tests in docs/test_code_builder.py",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: end-to-end pipeline test (opt-in via `--slow`)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--slow"):
        return
    skip = pytest.mark.skip(reason="slow test (opt-in via --slow)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Render Code Builder fixtures into a session-scoped tmp dir.

    Calls the JS generator (``docs/_data/render_fixtures.mjs``) once per
    session and returns the output path. Tests parametrized over
    axis-pinned combos read ``script.py`` / ``install.sh`` from there.
    """
    if shutil.which("node") is None:
        pytest.skip("node not on PATH (required to render Code Builder fixtures)")
    out = tmp_path_factory.mktemp("cb_fixtures")
    proc = subprocess.run(
        ["node", str(_FIXTURES_GEN), str(out)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            "render_fixtures.mjs failed:\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )
    return out
