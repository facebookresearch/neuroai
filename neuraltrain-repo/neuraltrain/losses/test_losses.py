# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from . import base, losses
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


class _ProjectionLoss(nn.Module):
    def __init__(self, distributed: bool) -> None:
        super().__init__()
        self.projection = nn.Linear(5, 4, bias=False)
        loss_type = losses.DistributedClipLoss if distributed else ClipLoss
        self.loss = loss_type(
            norm_kind="xy",
            temperature=True,
            symmetric=True,
            reduction="mean",
        )

    def forward(self, inputs: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return self.loss(self.projection(inputs), candidates)


def _global_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(12)
    return (
        torch.randn(6, 5, generator=generator),
        torch.randn(6, 4, generator=generator),
        torch.randn(4, 5, generator=generator),
    )


def _distributed_clip_worker(
    rank: int,
    rank_sizes: tuple[int, ...],
    rendezvous: str,
) -> None:
    world_size = len(rank_sizes)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        inputs, candidates, weight = _global_inputs()
        start = sum(rank_sizes[:rank])
        rank_slice = slice(start, start + rank_sizes[rank])
        local_inputs = inputs[rank_slice].clone().requires_grad_(True)
        local_candidates = candidates[rank_slice].clone().requires_grad_(True)

        module = _ProjectionLoss(distributed=True)
        module.projection.weight.data.copy_(weight)
        ddp_module = nn.parallel.DistributedDataParallel(module)
        actual = ddp_module(local_inputs, local_candidates)
        actual.backward()

        misaligned = losses.DistributedClipLoss(
            norm_kind="y", temperature=False, symmetric=False
        )
        extra_candidate = int(rank == 0)
        with pytest.raises(ValueError, match="as many candidates as estimates"):
            misaligned(torch.randn(2, 4), torch.randn(2 + extra_candidate, 4))

        reference_inputs = inputs.clone().requires_grad_(True)
        reference_candidates = candidates.clone().requires_grad_(True)
        reference = _ProjectionLoss(distributed=False)
        reference.projection.weight.data.copy_(weight)
        expected = reference(reference_inputs, reference_candidates)
        expected.backward()

        assert local_inputs.grad is not None
        assert local_candidates.grad is not None
        assert reference_inputs.grad is not None
        assert reference_candidates.grad is not None
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            module.projection.weight.grad, reference.projection.weight.grad
        )
        torch.testing.assert_close(
            module.loss.temperature.grad, reference.loss.temperature.grad
        )
        torch.testing.assert_close(
            local_inputs.grad / world_size, reference_inputs.grad[rank_slice]
        )
        torch.testing.assert_close(
            local_candidates.grad / world_size,
            reference_candidates.grad[rank_slice],
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rank_sizes", [(3, 3), (2, 4)])
def test_distributed_clip_loss(tmp_path, rank_sizes: tuple[int, ...]) -> None:
    config = {
        "name": "DistributedClipLoss",
        "norm_kind": "xy",
        "temperature": True,
        "symmetric": True,
        "reduction": "mean",
    }
    distributed = base.BaseLoss(**config).build()
    assert isinstance(distributed, losses.DistributedClipLoss)
    reference = ClipLoss(
        norm_kind="xy",
        temperature=True,
        symmetric=True,
        reduction="mean",
    )
    distributed.load_state_dict(reference.state_dict())

    torch.manual_seed(0)
    estimate = torch.randn(6, 4, requires_grad=True)
    candidate = torch.randn(6, 4, requires_grad=True)
    actual = distributed(estimate, candidate)
    expected = reference(estimate, candidate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_gradients = torch.autograd.grad(
        actual, (estimate, candidate), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected, (estimate, candidate))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

    assert sum(rank_sizes) == _global_inputs()[0].shape[0]
    mp.spawn(
        _distributed_clip_worker,
        args=(rank_sizes, str(tmp_path / "rendezvous")),
        nprocs=len(rank_sizes),
        join=True,
    )
