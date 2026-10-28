# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from . import config_manager


def test_default_config_includes_cluster() -> None:
    """The non-interactive default config exposes CLUSTER='auto'."""
    config = config_manager._default_config()
    assert config["CLUSTER"] == "auto"


def test_load_config_creates_the_configured_dirs(tmp_path: Path) -> None:
    names = ("cache", "save", "data")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({f"{n.upper()}_DIR": str(tmp_path / n) for n in names})
    )
    config_manager.load_config(config_path)
    assert all((tmp_path / name).is_dir() for name in names)


def test_load_config_warns_rather_than_fails_on_an_uncreatable_dir(
    tmp_path: Path,
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"DATA_DIR": str(blocker / "data")}))
    with pytest.warns(UserWarning, match="DATA_DIR"):
        config_manager.load_config(config_path)


def test_cluster_resolves_to_none_when_configured_null(
    patch_config: Callable[..., None],
) -> None:
    """A config with ``CLUSTER: null`` resolves the lazy module var to ``None``."""
    patch_config(CLUSTER=None)
    assert config_manager.CLUSTER is None


def test_cluster_resolves_to_configured_value(
    patch_config: Callable[..., None],
) -> None:
    """A non-null CLUSTER value is surfaced verbatim."""
    patch_config(CLUSTER="slurm")
    assert config_manager.CLUSTER == "slurm"
