# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import pytest
import torch

from .utils import agg_per_group, agg_retrieval_preds


@pytest.mark.parametrize("n_groups", [1, 4, 64])
def test_agg_per_group(n_groups: int) -> None:
    n_examples, n_features = 64, 3
    x = torch.rand(n_examples, n_features)
    groups = torch.randperm(n_examples) % n_groups
    min_n_ex_per_group = 1

    agg_x, agg_groups = agg_per_group(
        x, groups=groups.tolist(), min_n_ex_per_group=min_n_ex_per_group, agg_func="mean"
    )

    assert agg_x.shape == (n_groups, n_features)
    for g in range(n_groups):
        out = x[groups == g]
        out = out.mean(dim=0)
        assert (out == agg_x[agg_groups.index(g)]).all()

    assert set(agg_groups) == set(groups.tolist())


def test_agg_per_group_first() -> None:
    n_examples, n_features, n_groups = 64, 3, 4
    groups = torch.randperm(n_examples) % n_groups
    x = groups.repeat(n_features, 1).T

    agg_x, agg_groups = agg_per_group(
        x, groups=groups.tolist(), min_n_ex_per_group=1, agg_func="first"
    )

    assert agg_x.shape == (n_groups, n_features)
    for g in range(n_groups):
        out = x[groups == g]
        assert (out[0] == agg_x[agg_groups.index(g)]).all()

    assert set(agg_groups) == set(groups.tolist())

    # Error if groups do not have the same values
    x = torch.rand(n_examples, n_features)
    with pytest.raises(AssertionError):
        agg_per_group(x, groups=groups.tolist(), min_n_ex_per_group=1, agg_func="first")


@pytest.mark.parametrize("groups", [[], None])
def test_agg_per_group_no_groups(groups: list | None) -> None:
    n_examples, n_features = 64, 3
    x = torch.rand(n_examples, n_features)
    min_n_ex_per_group = 1

    agg_x, agg_groups = agg_per_group(
        x, groups=groups, min_n_ex_per_group=min_n_ex_per_group, agg_func="mean"
    )

    assert (agg_x == x).all()
    assert agg_groups == []


def test_agg_per_group_min_n() -> None:
    n_examples, n_features, n_groups = 64, 3, 4
    x = torch.rand(n_examples, n_features)
    groups = torch.arange(n_examples) % n_groups
    groups[: n_examples - n_groups] = 0
    min_n_ex_per_group = 2

    agg_x, agg_groups = agg_per_group(
        x,
        groups=groups.tolist(),
        min_n_ex_per_group=min_n_ex_per_group,
        agg_func="mean",
    )

    assert (agg_x == x[: n_examples - n_groups + 1].mean(dim=0)).all()
    assert agg_groups == [0]


@pytest.mark.parametrize("n_subjects", [None, 1, 3])
def test_agg_retrieval_preds(n_subjects: int | None) -> None:
    n_examples_pred = 64
    n_features = 3
    n_groups_pred = 4
    y_pred = torch.rand(n_examples_pred, n_features)

    groups_pred = torch.randperm(n_examples_pred) % n_groups_pred

    keep_subject_group_name = n_subjects is not None and n_subjects > 1

    if n_subjects is None:
        subjects_pred, n_subjects = None, 1
        n_out_groups = n_groups_pred
    else:
        subjects_pred = (torch.arange(n_examples_pred) % n_subjects).tolist()
        n_out_groups = len(set(zip(groups_pred.tolist(), subjects_pred)))

    _y_pred, _groups_pred = agg_retrieval_preds(
        y_pred,
        groups_pred.tolist(),
        subjects_pred=subjects_pred,
        keep_subject_group_name=keep_subject_group_name,
    )

    assert _y_pred.shape == (n_out_groups, n_features)
    assert {g[1] if keep_subject_group_name else g for g in _groups_pred} == set(
        groups_pred.tolist()
    )
    for s, g in product(range(n_subjects), range(n_groups_pred)):
        mask = groups_pred == g
        if subjects_pred is not None:
            mask &= torch.Tensor(subjects_pred) == s

        if mask.any():
            out = y_pred[mask].mean(dim=0)
            index = (s, g) if keep_subject_group_name else g
            assert (out == _y_pred[_groups_pred.index(index), :]).all()
