# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch

import neuralbench  # noqa: F401  # registers BinnedMAE config
from neuralbench.metrics import BinnedMAE
from neuraltrain.metrics.base import BaseMetric


@pytest.mark.parametrize(
    "preds,targets,expected,kwargs",
    [
        # Default bins [0,40,90,300,600]. Targets 5,60,100,400 land in bins 0,1,2,3.
        # Per-bin MAEs: |10-5|=5, |50-60|=10, |200-100|=100, |500-400|=100. bMAE = 53.75.
        ([10.0, 50.0, 200.0, 500.0], [5.0, 60.0, 100.0, 400.0], 53.75, {}),
        # Skips empty bins: targets only in bin 0 → bMAE = MAE of bin 0.
        ([2.0, 8.0], [5.0, 5.0], 3.0, {}),
        # Upper boundary inclusion: target equal to last upper boundary belongs to last bin.
        ([550.0], [600.0], 50.0, {}),
        # Custom bin boundaries.
        ([0.5, 1.5], [0.2, 1.0], 0.4, {"bin_boundaries": [0.0, 1.0, 2.0]}),
        # No data returns NaN.
        ([], [], float("nan"), {}),
    ],
)
def test_binned_mae_computation(preds, targets, expected, kwargs):
    metric = BinnedMAE(**kwargs)
    if preds:
        metric.update(torch.tensor(preds), torch.tensor(targets))
    val = metric.compute().item()
    if math.isnan(expected):
        assert math.isnan(val)
    else:
        assert val == pytest.approx(expected)


def test_binned_mae_accumulates_across_updates():
    """Online accumulation: two updates should give the same answer as one
    update with the concatenated tensors."""
    preds_a = torch.tensor([10.0, 50.0])
    target_a = torch.tensor([5.0, 60.0])
    preds_b = torch.tensor([200.0, 500.0])
    target_b = torch.tensor([100.0, 400.0])

    online = BinnedMAE()
    online.update(preds_a, target_a)
    online.update(preds_b, target_b)

    batch = BinnedMAE()
    batch.update(torch.cat([preds_a, preds_b]), torch.cat([target_a, target_b]))

    assert online.compute().item() == pytest.approx(batch.compute().item())


@pytest.mark.parametrize(
    "boundaries,err_match",
    [
        ([0.0], "at least 2"),
        ([1.0, 0.5, 2.0], "strictly increasing"),
        ([0.0, 0.0], "strictly increasing"),
    ],
)
def test_binned_mae_invalid_boundaries(boundaries, err_match):
    with pytest.raises(ValueError, match=err_match):
        BinnedMAE(bin_boundaries=boundaries)


def test_binned_mae_resolves_via_basemetric_discriminator():
    """The metric can be constructed from a YAML-style config dict."""
    cfg = BaseMetric.model_validate(
        {
            "log_name": "bmae",
            "name": "BinnedMAE",
            "bin_boundaries": [0.0, 40.0, 90.0, 300.0, 600.0],
        }
    )
    assert type(cfg).__name__ == "BinnedMAE"
    assert isinstance(cfg.build(), BinnedMAE)
