# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-dataset variability artefact for NeuralBench-Full.

This package builds the two-panel figure used in the NeuralBench white
paper to characterise within-task variability across the multiple
datasets of each multi-dataset paradigm:

* :func:`plot_full_variability_panel` -- vertical 2-panel figure with
  (A) the Core-vs-Full slopegraph linking each model's mean rank on
  NeuralBench-Core to its mean rank on NeuralBench-Full, and (B) a
  compact heatmap of the per-(model, task) rank standard deviation
  across each task's datasets.

Implementations live in cohesive submodules:

* :mod:`.core_vs_full`             -- core-vs-full bump chart drawing.
* :mod:`.rank_variability_matrix`  -- per-(model, task) rank-std
  heatmap drawing and its underlying matrix helper.
* :mod:`.full_variability_panel`   -- the combined 2-panel figure.
"""

from __future__ import annotations

# ``compute_task_dataset_ranks`` is the canonical home of the rank helper
# and lives in :mod:`neuralbench.plots.ranking`; re-exporting it here
# preserves the legacy ``from neuralbench.plots.variability import
# compute_task_dataset_ranks`` import path used by ``test_variability``.
from neuralbench.plots.ranking import compute_task_dataset_ranks
from neuralbench.plots.variability.core_vs_full import (
    _BUMP_COLOR_IMPROVED,
    _BUMP_COLOR_STABLE,
    _BUMP_COLOR_WORSENED,
    _bump_line_color,
)
from neuralbench.plots.variability.full_variability_panel import (
    plot_full_variability_panel,
)
from neuralbench.plots.variability.rank_variability_matrix import (
    _GROUP_BRACKET_COLORS,
    _append_marginal_means,
    _model_group,
    _order_model_task_matrix,
    compute_model_task_rank_variability,
)

__all__ = [
    # rank helper (re-export)
    "compute_task_dataset_ranks",
    # core vs full (drawing helpers stay private; expose colour constants
    # used by tests / callers needing the bump-chart palette)
    "_BUMP_COLOR_IMPROVED",
    "_BUMP_COLOR_STABLE",
    "_BUMP_COLOR_WORSENED",
    "_bump_line_color",
    # rank variability matrix (matrix helpers and group-colour map)
    "_GROUP_BRACKET_COLORS",
    "_append_marginal_means",
    "_model_group",
    "_order_model_task_matrix",
    "compute_model_task_rank_variability",
    # combined panel (the only public plotting entry point)
    "plot_full_variability_panel",
]
