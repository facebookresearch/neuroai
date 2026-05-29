# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Custom torchmetrics for neuralbench tasks.

Importing this module has the side-effect of registering the metric classes
defined here with the ``BaseMetric`` discriminated-union machinery from
``neuraltrain.metrics.base``, so they can be referenced by class name in
task YAML configs (e.g. ``name: BinnedMAE``).
"""

import torch
import torchmetrics

from neuraltrain.metrics.base import BaseMetric
from neuraltrain.utils import convert_to_pydantic


class BinnedMAE(torchmetrics.Metric):
    """Mean absolute error binned by ground-truth target value.

    Targets are partitioned into bins defined by ``bin_boundaries`` and the
    mean absolute error is computed inside each bin. The reported value is the
    unweighted mean of per-bin MAEs across non-empty bins.

    Targets falling exactly on the upper boundary of the last bin are included
    in that bin (so the cap value of ``cap_s`` falls in the last bin); other
    bins use the standard ``[lo, hi)`` convention.

    Parameters
    ----------
    bin_boundaries : list of float, optional
        Strictly increasing list of ``n + 1`` floats defining ``n`` bins.
        Defaults to ``[0.0, 40.0, 90.0, 300.0, 600.0]``.
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    sum_abs_err: torch.Tensor
    count: torch.Tensor

    def __init__(self, bin_boundaries: list[float] | None = None) -> None:
        super().__init__()
        boundaries = (
            list(bin_boundaries)
            if bin_boundaries is not None
            else [0.0, 40.0, 90.0, 300.0, 600.0]
        )
        if len(boundaries) < 2:
            raise ValueError(
                f"`bin_boundaries` must have at least 2 entries; got {boundaries!r}"
            )
        if any(boundaries[i] >= boundaries[i + 1] for i in range(len(boundaries) - 1)):
            raise ValueError(
                f"`bin_boundaries` must be strictly increasing; got {boundaries!r}"
            )
        self.bin_boundaries = boundaries
        n_bins = len(boundaries) - 1
        self.add_state(
            "sum_abs_err",
            default=torch.zeros(n_bins, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count",
            default=torch.zeros(n_bins, dtype=torch.float64),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        t = target.flatten().to(self.sum_abs_err.dtype)
        e = (preds.flatten().to(self.sum_abs_err.dtype) - t).abs()
        edges = torch.as_tensor(self.bin_boundaries, device=t.device, dtype=t.dtype)

        # Assign each target to a bin index
        # right=True ensures [lo, hi) binning convention.
        # Values >= edges[-2] naturally return n_bins - 1, handling the last bin correctly.
        bin_idx = torch.bucketize(t, edges[1:-1], right=True)

        # Filter out-of-range targets
        in_range = (t >= edges[0]) & (t <= edges[-1])

        if in_range.any():
            self.sum_abs_err.scatter_add_(0, bin_idx[in_range], e[in_range])
            self.count.scatter_add_(0, bin_idx[in_range], torch.ones_like(e[in_range]))

    def compute(self) -> torch.Tensor:
        nonempty = self.count > 0
        if not bool(nonempty.any()):
            return torch.tensor(float("nan"), device=self.sum_abs_err.device)
        per_bin = self.sum_abs_err / self.count.clamp(min=1)
        return per_bin[nonempty].mean().to(torch.float32)


_BinnedMAEConfig = convert_to_pydantic(
    BinnedMAE,
    "BinnedMAE",
    parent_class=BaseMetric,
    exclude_from_build=["log_name"],
)
