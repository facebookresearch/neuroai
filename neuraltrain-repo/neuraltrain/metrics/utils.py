# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for computing metrics."""

import typing as tp

import pandas as pd
import torch


def agg_per_group(
    x: torch.Tensor,
    groups: list[tp.Any] | None = None,
    min_n_ex_per_group: int = 1,
    agg_func: tp.Literal["mean", "first"] = "mean",
) -> tuple[torch.Tensor, list[tp.Any]]:
    """Aggregate tensors by groups.

    Parameters
    ----------
    x :
        Tensor of shape (N, *) to aggregate per group.
    groups :
        List of length N of labels for the samples used to group `x`.
    min_n_ex_per_group :
        Minimum number of examples in a group for a group to be kept, i.e. ignore a group if there
        are fewer than this number of examples in it.
    agg_func :
        Aggregation function to use.

    Returns
    -------
    torch.Tensor :
        Aggregated version of `x`, of shape (M, *).
    list :
        Aggregated version of `groups`, of length M.
    """
    if groups is None or len(groups) == 0:
        return x, []

    # Aggregation might not be necessary if there's only one item per group
    assert (
        isinstance(groups, list) or groups is None
    )  # Not supporting a torch.Tensor to simplify
    if len(set(groups)) == len(groups):
        return x, groups

    # NOTE: Using pandas is faster than computing masks by hand.
    groups_df = pd.DataFrame({"label": groups})
    agg_x, agg_groups = [], []
    for name, group in groups_df.groupby("label", sort=False):
        if len(group) < min_n_ex_per_group:
            # Ignore group because it has too few examples
            continue

        if agg_func == "first":
            assert (
                x[group.index] == x[group.index[0]]  # type: ignore
            ).all(), (
                'Cannot use agg_func="first" as group contains different values for x.'
            )
            agg_out = x[group.index[0]]
        elif agg_func == "mean":
            assert isinstance(x, torch.Tensor)
            agg_out = x[group.index].mean(dim=0)  # type: ignore
        else:
            raise NotImplementedError

        agg_x.append(agg_out)
        agg_groups.append(name)

    out = torch.stack(agg_x)

    return out, agg_groups


def agg_retrieval_preds(
    y_pred: torch.Tensor,
    groups_pred: list,
    subjects_pred: list | None = None,
    keep_subject_group_name: bool = False,
) -> tuple[torch.Tensor, list]:
    """Aggregate predictions at the right level before retrieval (e.g. subject-level).

    Parameters
    ----------
    y_pred :
        Tensor of shape (N, *) containing the predictions.
    groups_pred :
        List of length N containing the labels (group) of each example in `y_pred`.
    subjects_pred :
        If provided, list of length M containing an additional grouping variable for `y_pred`. For
        instance, this can be the subject from which prediction comes from, such that examples can
        be aggregated within each subject.
    keep_subject_group_name :
        If True and `subjects_pred` is provided, return `_groups_pred` as a list of tuples
        (subj, group). If False, return `_groups_pred` as a list of items taken from `groups_pred`.

    Returns
    -------
    torch.Tensor :
        Aggregated version of `y_pred` using `groups_pred`, of shape (N', *).
    list :
        List of length N' containing the labels (group) for each predicted output example.
    """

    # Average predictions across instances
    groups = (
        groups_pred if subjects_pred is None else list(zip(subjects_pred, groups_pred))
    )
    _y_pred, _groups_pred = agg_per_group(y_pred, groups=groups, agg_func="mean")

    _groups_pred = (
        _groups_pred
        if subjects_pred is None or keep_subject_group_name
        else [group[1] for group in _groups_pred]
    )

    return _y_pred, _groups_pred
