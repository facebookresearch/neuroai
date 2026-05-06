# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Backwards-compatible alias for :mod:`neuralbench.plots.normalized_summary`.

The implementation moved to ``normalized_summary.py``; importing names
through ``neuralbench.plots.utils`` keeps any out-of-tree script that
hasn't been updated yet working.
"""

from __future__ import annotations

from neuralbench.plots.normalized_summary import (  # noqa: F401
    _compute_dummy_normalized_scores,
    plot_normalized_lines_summary,
)

__all__ = [
    "_compute_dummy_normalized_scores",
    "plot_normalized_lines_summary",
]
