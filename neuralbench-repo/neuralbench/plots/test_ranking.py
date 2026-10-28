# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`neuralbench.plots.ranking`."""

import numpy as np
import pandas as pd

from neuralbench.plots.ranking import compute_row_rank_stats


def test_row_rank_stats_skip_a_task_no_model_scored():
    df = pd.DataFrame(
        {
            "task_name": ["a", "a", "b", "b"],
            "model_name": ["M1", "M2", "M1", "M2"],
            "metric_name": "test/bal_acc",
            "metric_value": [1.0, 2.0, np.nan, np.nan],
        }
    )
    stats = compute_row_rank_stats(df, ["a", "b"], ["M1", "M2"])
    assert stats["mean"].tolist() == [1.0, 0.5]
