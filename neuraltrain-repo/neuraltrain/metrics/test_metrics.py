# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pydantic
import pytest
import torch
from sklearn.metrics import balanced_accuracy_score
from torchmetrics import PearsonCorrCoef

from neuraltrain.metrics import BaseMetric

from . import metrics
from .base import Accuracy, GroupedMetric  # type: ignore[attr-defined]


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("reduction", [None, "mean", "sum"])
def test_pearson_corr(dim: int, reduction):
    shape = (10, 20)
    y_true = torch.rand(shape)
    y_pred = torch.rand(shape)

    metric = metrics.OnlinePearsonCorr(dim=dim, reduction=reduction)
    out = metric(y_pred, y_true)
    metric.reset()
    assert not metric._initialized
    out_same = metric(y_true, y_true)

    if reduction is None:
        assert len(out) == shape[1 - dim]
        assert all(out.abs() <= 1)
        assert torch.allclose(out_same, torch.tensor([1.0]))
    elif reduction == "mean":
        assert out.shape == torch.Size([])
        assert (out.abs() <= 1).all()
        assert torch.allclose(out_same, torch.tensor([1.0]))
    elif reduction == "sum":
        assert out.shape == torch.Size([])
        assert torch.allclose(out_same, torch.tensor([shape[1 - dim] * 1.0]))

    # Double check results with external library
    if reduction is None:
        ext_metric = PearsonCorrCoef(num_outputs=shape[1 - dim])
        ext_out = ext_metric(
            y_pred if dim == 0 else y_pred.T,
            y_true if dim == 0 else y_true.T,
        )
        assert torch.allclose(out, ext_out, atol=1e-6)


@pytest.mark.parametrize("dim", [0, 1])
def test_multiple_updates(dim: int):
    metric = metrics.OnlinePearsonCorr(dim=dim, reduction="mean")
    for k in range(5):
        y_true = torch.rand(10 - k, 20)
        y_pred = 2 * y_true + 3
        metric.update(y_pred, y_true)
    out = metric.compute()
    assert torch.allclose(out, torch.tensor(1.0))


def test_multiple_updates_dim1_accumulation():
    """Verify that multi-batch accumulation for dim=1 matches single-batch computation."""
    torch.manual_seed(0)
    batches = [torch.rand(n, 20) for n in (8, 12, 5)]

    metric_incremental = metrics.OnlinePearsonCorr(dim=1, reduction=None)
    all_corrs = []
    for y_true in batches:
        y_pred = y_true + 0.1 * torch.randn_like(y_true)
        metric_incremental.update(y_pred, y_true)
        all_corrs.append(metrics.OnlinePearsonCorr._batch_pearsonr(y_pred, y_true))

    expected = torch.cat(all_corrs)
    result = metric_incremental.compute()
    assert result.shape == expected.shape
    assert torch.allclose(result, expected)


def test_compute_before_update_dim1():
    """compute() on a fresh dim=1 metric raises because there are no samples."""
    metric = metrics.OnlinePearsonCorr(dim=1, reduction=None)
    with pytest.raises(ValueError, match="No samples"):
        metric.compute()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dim", [0, 1])
def test_pearson_corr_gpu(dim: int):
    """OnlinePearsonCorr must work when created on CPU and moved to GPU."""
    device = torch.device("cuda:0")
    metric = metrics.OnlinePearsonCorr(dim=dim, reduction="mean").to(device)

    x = torch.randn(16, 20, device=device)
    metric.update(2 * x + 3, x)
    assert torch.allclose(metric.compute(), torch.tensor(1.0, device=device))

    metric.reset()
    x2 = torch.randn(16, 20, device=device)
    metric.update(2 * x2 + 3, x2)
    assert torch.allclose(metric.compute(), torch.tensor(1.0, device=device))


@pytest.mark.parametrize("dim", [0, 1])
def test_pearson_corr_constant_value(dim: int):
    # Also check constant value handling
    shape = (10, 20)
    y_true = torch.ones(shape)
    y_pred = torch.rand(shape)

    metric = metrics.OnlinePearsonCorr(dim=dim)
    out = metric(y_pred, y_true)
    assert torch.isnan(out).all()


@pytest.mark.parametrize("reduction", ["median", "mean", "std"])
def test_rank(reduction):
    F = 8

    # Create y_pred such that item i has rank i in y_true
    template = torch.rand(1, F)
    y_pred = template.repeat(F, 1)
    y_true = torch.triu(y_pred)
    y_true_labels = torch.arange(y_true.shape[0]).tolist()
    y_pred_labels = torch.arange(y_pred.shape[0]).tolist()

    metric = metrics.Rank(reduction)
    out1 = metric(y_pred, y_true, None, None)
    metric.reset()

    # With labels
    out2 = metric(y_pred, y_true, y_pred_labels, y_true_labels)
    metric.reset()

    # With labels and different shapes
    y_true_repeat = torch.cat([y_true, torch.zeros(F, F)])
    y_true_labels_repeat = torch.arange(y_true_repeat.shape[0]).tolist()
    out3 = metric(y_pred, y_true_repeat, y_pred_labels, y_true_labels_repeat)
    metric.reset()

    # With multiple calls to update
    for _ in range(3):
        out4 = metric(y_pred, y_true, None, None)

    true_reduced_rank = {"mean": torch.mean, "median": torch.median, "std": torch.std}[
        reduction
    ](torch.Tensor(y_pred_labels))
    assert out1 == out2 == out3 == out4 == true_reduced_rank

    # Without labels but with different shapes
    with pytest.raises(ValueError):
        metric(y_pred[: F // 2], y_true, None, None)

    # Without labels but with more y_pred than y_true
    with pytest.raises(ValueError):
        metric(y_pred, y_true[: F // 2], None, None)


def test_rank_half_bin():
    F = 4
    y_pred = torch.rand(1, F)
    y_true = torch.rand(F, F)
    y_true[:2, :] = y_pred
    y_pred_labels, y_true_labels = [0], list(range(F))

    metric = metrics.Rank("median")
    out = metric(y_pred, y_true, y_pred_labels, y_true_labels)

    assert out == 0.5


@pytest.mark.parametrize("topk", [1, 3, 5])
def test_topk_acc(topk: int):
    # Create y_pred such that item i has rank i in y_true
    F = 8
    template = torch.rand(1, F)
    y_pred = template.repeat(F, 1)
    y_true = torch.triu(y_pred)
    y_true_labels = torch.arange(y_true.shape[0]).tolist()
    y_pred_labels = torch.arange(y_pred.shape[0]).tolist()

    metric = metrics.TopkAcc(topk)
    out = metric(y_pred, y_true, y_pred_labels, y_true_labels)

    # Get true top-k accuracy
    true_ranks = torch.Tensor(
        y_pred_labels
    )  # Because of the way y_pred and y_true were defined
    true_acc = (true_ranks < topk).float().mean()

    assert out == true_acc


@pytest.mark.parametrize("topk", [1, 5])
@pytest.mark.parametrize("true_labels", ["first", "diagonal"])
def test_topk_acc_from_scores(topk: int, true_labels) -> None:
    batch_size, n_features = 10, 8
    n_negatives = 20

    # Like in contrastive tasks
    if true_labels == "first":
        y_true = torch.rand(batch_size, 1, n_features)
        negatives = torch.rand(batch_size, n_negatives, n_features)
        targets = torch.cat([y_true, negatives], dim=1)
        scores = -1 * torch.cdist(y_true, targets).squeeze(1)
    elif true_labels == "diagonal":
        y_true = torch.rand(batch_size, n_features)
        scores = -1 * torch.cdist(y_true, y_true)

    scores += 10.0  # Make test more generalizable

    metric = metrics.TopkAccFromScores(topk, true_labels=true_labels)
    out = metric(scores)

    assert out == 1.0


def test_grouped_metric():
    metric = GroupedMetric(
        log_name="test_metric",
        metric_name="Accuracy",
        kwargs={"num_classes": 2, "task": "multiclass"},
    ).build()
    groups = torch.LongTensor([0, 1, 1, 2])
    metric.update(torch.LongTensor([0, 0, 0, 0]), torch.LongTensor([0, 1, 0, 1]), groups)
    assert metric.compute() == {"0": 1.0, "1": 0.5, "2": 0.0}


class Xp(pydantic.BaseModel):
    metric: BaseMetric


@pytest.mark.parametrize(
    "metric_cfg, perfect_res_lower_bound, perfect_res_upper_bound",
    [
        (
            {
                "log_name": "clip",
                "name": "ImageSimilarity",
                "model_name": "clip",
                "layer": "final",
            },
            1.0,
            None,
        ),
        (
            {
                "log_name": "alexnet_2",
                "name": "ImageSimilarity",
                "model_name": "alexnet",
                "layer": 2,
            },
            1.0,
            None,
        ),
        (
            {
                "log_name": "alexnet_5",
                "name": "ImageSimilarity",
                "model_name": "alexnet",
                "layer": 5,
            },
            1.0,
            None,
        ),
        (
            {
                "log_name": "efficientnet",
                "name": "ImageSimilarity",
                "model_name": "efficientnet",
                "layer": "avgpool",
            },
            None,
            10e-7,
        ),
        (
            {
                "log_name": "swav",
                "name": "ImageSimilarity",
                "model_name": "swav",
                "layer": "avgpool",
            },
            None,
            10e-7,
        ),
    ],
)
def test_img_similarity_metrics_perfect_preds(
    metric_cfg: dict[str, tp.Any], perfect_res_lower_bound, perfect_res_upper_bound
) -> None:
    metric = Xp(metric=metric_cfg).metric.build()  # type: ignore
    images = torch.randn(2, 3, 224, 224)
    metric.update(images, images)  # type: ignore
    res = metric.compute()  # type: ignore
    if perfect_res_lower_bound is not None:
        lo = perfect_res_lower_bound
        assert res >= lo, f"Expected >={lo}, got {res}"
    if perfect_res_upper_bound is not None:
        hi = perfect_res_upper_bound
        assert res <= hi, f"Expected <={hi}, got {res}"


def test_bal_acc() -> None:
    batch_size, n_classes = 16, 5
    y_pred = torch.randn(batch_size, n_classes)
    y_true = torch.arange(batch_size) % n_classes

    metric = Accuracy(
        log_name="bal_acc",
        kwargs={
            "task": "multiclass",
            "num_classes": n_classes,
            "average": "macro",
        },
    ).build()

    out = metric(y_pred, y_true).float()
    out2 = metric(y_pred.argmax(dim=1), y_true).float()
    out_sklearn = balanced_accuracy_score(y_true, y_pred.argmax(dim=1))

    assert torch.isclose(out, out2, atol=1e-6)
    assert torch.isclose(out, torch.tensor(out_sklearn).float(), atol=1e-6)


@pytest.mark.parametrize(
    "torchmetrics_kwargs", [{"wrong_key": True}, {"sync_on_compute": False}]
)
def test_torchmetricskwargs_rank(torchmetrics_kwargs):
    if "wrong_key" in torchmetrics_kwargs:
        with pytest.raises(ValueError):
            _ = metrics.Rank(torchmetrics_kwargs=torchmetrics_kwargs)
    else:
        _ = metrics.Rank(torchmetrics_kwargs=torchmetrics_kwargs)
