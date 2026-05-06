# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DataFrame filters shared across the plots package.

These two helpers replace ~7 copies of the same one-liner that picks
out tasks with more than one dataset, and the per-task "core" dataset
filter that several of the multi-dataset analyses depend on.
"""

from __future__ import annotations

import warnings

import pandas as pd

from neuralbench.plots._task_metadata import default_studies_per_task


def multi_dataset_tasks(df: pd.DataFrame, *, threshold: int = 2) -> list[str]:
    """Return task names with at least *threshold* distinct datasets.

    ``df`` must have ``task_name`` and ``dataset_name`` columns; the
    helper returns ``[]`` if either column is missing or no task meets
    the threshold.
    """
    if "task_name" not in df.columns or "dataset_name" not in df.columns:
        return []
    counts = df.groupby("task_name")["dataset_name"].nunique()
    return counts[counts >= threshold].index.tolist()


def filter_core_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the core (default) dataset for each task.

    A few task names (notably ``image``) are registered under multiple
    devices with different default studies; rows whose ``dataset_name``
    matches *any* of the per-device defaults are retained so cross-device
    plots don't accidentally drop one modality.
    """
    if "dataset_name" not in df.columns:
        return df
    default_studies = default_studies_per_task()
    masks = []
    for task, group in df.groupby("task_name"):
        if group["dataset_name"].nunique() <= 1:
            masks.append(group.index)
        else:
            core_studies = default_studies.get(str(task), [])
            selected = group[group["dataset_name"].isin(core_studies)].index
            if selected.empty:
                first_ds = group["dataset_name"].iloc[0]
                warnings.warn(
                    f"No default study for multi-dataset task {task!r}; "
                    f"falling back to first dataset {first_ds!r}.",
                    stacklevel=2,
                )
                selected = group[group["dataset_name"] == first_ds].index
            masks.append(selected)
    if not masks:
        return df
    keep = masks[0]
    for m in masks[1:]:
        keep = keep.union(m)
    return df.loc[keep]
