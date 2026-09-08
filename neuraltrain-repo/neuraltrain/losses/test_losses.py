# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn

from .losses import ClipLoss, MultiLoss, SigLipLoss


@pytest.mark.parametrize("norm_kind", ["x", "y", "xy"])
@pytest.mark.parametrize("temperature", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("larger_retrieval_set", [False, True])
def test_clip_loss(norm_kind, temperature, symmetric, larger_retrieval_set):
    loss = ClipLoss(norm_kind=norm_kind, temperature=temperature, symmetric=symmetric)

    batch_size, n_features = 8, 12
    y_pred = torch.randn(batch_size, n_features)
    y_true = y_pred
    if larger_retrieval_set:
        y_true = torch.cat([y_true, torch.zeros(batch_size, n_features)], dim=0)

    scores = loss.get_scores(y_pred, y_true)
    assert scores.shape == (y_pred.shape[0], y_true.shape[0])
    diag_scores = scores.diag()
    if norm_kind in ("y", "xy"):
        diag_scores = diag_scores[:, None]
    elif norm_kind == "x":
        diag_scores = diag_scores[None, :]
    assert (scores[:, :batch_size] <= diag_scores).all()

    probas = loss.get_probabilities(y_pred, y_true)
    assert probas.shape == (y_pred.shape[0], y_true.shape[0])
    assert ((probas <= 1.0) & (probas >= 0.0)).all()

    if symmetric and larger_retrieval_set:
        with pytest.raises(ValueError):
            out = loss(y_pred, y_true)
    else:
        out = loss(y_pred, y_true)
        assert not out.isnan()


@pytest.mark.parametrize("norm_kind", ["x", "y", "xy"])
@pytest.mark.parametrize("temperature", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("larger_retrieval_set", [False, True])
@pytest.mark.parametrize("identical_candidates_threshold", [None, 0.999])
def test_siglip_loss(
    norm_kind, temperature, bias, larger_retrieval_set, identical_candidates_threshold
):
    loss = SigLipLoss(
        norm_kind=norm_kind,
        temperature=temperature,
        bias=bias,
        identical_candidates_threshold=identical_candidates_threshold,
    )

    batch_size, n_features = 8, 12
    y_pred = torch.randn(batch_size, n_features)
    y_true = y_pred
    if identical_candidates_threshold is not None:
        # create two identical targets
        y_true.data[-1] = y_true.data[0]
    if larger_retrieval_set:
        y_true = torch.cat([y_true, torch.zeros(batch_size, n_features)], dim=0)

    loss_value = loss(y_pred, y_true)

    # Compare to formulation from paper (Algorithm 1)
    scores = loss.get_scores(y_pred, y_true)
    targets = 2 * torch.eye(*scores.shape) - torch.ones_like(scores)
    if identical_candidates_threshold is not None:
        targets.data[0, 7] = 1
        targets.data[7, 0] = 1
    loss_value_orig = -nn.functional.logsigmoid(targets * scores).sum()

    assert torch.isclose(loss_value, loss_value_orig, atol=1e-5)


@pytest.mark.parametrize("weights", [None, {"mse": 0.25, "clip": 0.75}])
def test_multi_loss(weights):
    batch_size, n_features = 8, 12
    y_pred = torch.randn(batch_size, n_features)
    y_true = torch.randn(batch_size, n_features)

    losses = {"mse": nn.MSELoss(), "clip": ClipLoss()}
    loss = MultiLoss(losses, weights)
    out = loss(y_pred, y_true)

    assert not out["total"].isnan()

    # Make sure it's the same as the sum of its parts
    mse_loss = losses["mse"](y_pred, y_true)
    clip_loss = losses["clip"](y_pred, y_true)
    if weights is None:
        weights = {name: 1.0 for name in losses}
    total = sum([weights[name] * out[name] for name in losses])

    assert out["total"] == total
    assert out["mse"] == mse_loss
    assert out["clip"] == clip_loss


def test_single_multi_loss():
    batch_size, n_features = 8, 12
    y_pred = torch.randn(batch_size, n_features)
    y_true = torch.randn(batch_size, n_features)
    weight = 0.3

    loss = MultiLoss({"mse": nn.MSELoss()}, {"mse": weight})
    out = loss(y_pred, y_true)

    total = weight * nn.MSELoss()(y_pred, y_true)
    assert out["total"] == total


@pytest.mark.parametrize("multi_pred_heads", [False, True])
@pytest.mark.parametrize("multi_targets", [False, True])
def test_multi_loss_multi_heads(multi_pred_heads, multi_targets):
    batch_size, n_features = 8, 12
    if multi_pred_heads:
        y_pred = {
            "mse": torch.randn(batch_size, n_features),
            "clip": torch.randn(batch_size, n_features),
        }
    else:
        y_pred = torch.randn(batch_size, n_features)

    if multi_targets:
        y_true = {
            "mse": torch.randn(batch_size, n_features),
            "clip": torch.randn(batch_size, n_features),
        }
    else:
        y_true = torch.randn(batch_size, n_features)

    losses = {"mse": nn.MSELoss(), "clip": ClipLoss()}
    weights = {"mse": 0.5, "clip": 0.5}
    loss = MultiLoss(losses, weights)

    out = loss(y_pred, y_true)
    assert not out["total"].isnan()
