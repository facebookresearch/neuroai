# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

from . import config_manager


def test_default_config_includes_cluster() -> None:
    """The non-interactive default config exposes CLUSTER='auto'."""
    config = config_manager._default_config()
    assert config["CLUSTER"] == "auto"


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
