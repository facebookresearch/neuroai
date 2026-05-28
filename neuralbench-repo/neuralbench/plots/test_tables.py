# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`neuralbench.plots.tables`."""

from __future__ import annotations

import pytest

from neuralbench.aggregator import BenchmarkAggregator
from neuralbench.plots.tables import build_results_df

_DEFAULT_MAPPING: dict[str, str] = BenchmarkAggregator.model_fields[
    "loss_to_metric_mapping"
].default


def _row(*, loss_name: str, **metric_values: float) -> dict:
    """Build one synthetic result row with the columns ``build_results_df`` reads."""
    return {
        "loss": {"name": loss_name},
        "brain_model_name": "EEGNet",
        "task_name": "sleep_onset",
        **metric_values,
    }


def test_build_results_df_resolves_multi_loss_to_bmae():
    """Sleep-onset rows logged with ``MultiLoss`` must select ``test/bmae``."""
    results = [_row(loss_name="MultiLoss", **{"test/bmae": 42.0})]
    df = build_results_df(results, _DEFAULT_MAPPING)
    assert df["metric_name"].tolist() == ["test/bmae"]
    assert df["metric_value"].tolist() == [42.0]


def test_build_results_df_raises_for_unmapped_loss():
    """An unknown loss name surfaces a clear error, not ``KeyError: nan``."""
    results = [_row(loss_name="NewlyAddedLoss", **{"test/something": 1.0})]
    with pytest.raises(KeyError, match="NewlyAddedLoss"):
        build_results_df(results, _DEFAULT_MAPPING)
