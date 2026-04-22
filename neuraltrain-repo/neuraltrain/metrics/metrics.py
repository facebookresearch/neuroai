# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation metrics."""

# pylint: disable=attribute-defined-outside-init

import typing as tp
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy as sp
import torch
import torchmetrics
import torchvision.models as tvmodels
import torchvision.transforms as T
from scipy.stats import binom
from torchmetrics.utilities.data import dim_zero_cat


class OnlinePearsonCorr(torchmetrics.regression.PearsonCorrCoef):
    """
    Online Pearson correlation coefficient.

    This class computes the Pearson correlation coefficient in an online fashion,
    updating the metric with each new batch of predictions and targets.

    Parameters
    ----------
    dim : int
        The dimension along which to compute the correlation coefficient.
        - ``dim=0``: correlate across samples (each column is a separate output).
          Uses Welford online accumulation, so batch sizes may vary.
        - ``dim=1``: correlate across features/time within each sample, then
          reduce across samples. Computes per-sample correlations per batch and
          accumulates them, so it is correct across multiple batches.
    reduction : {"mean", "sum", "none"}, optional
        Specifies how to reduce the computed correlation coefficients. Defaults to "mean".
    torchmetrics_kwargs : dict or None
        Extra keyword arguments forwarded to the ``torchmetrics.Metric``
        constructor (e.g. ``dist_sync_on_step``).
    """

    def __init__(
        self,
        dim: int,
        reduction: tp.Literal["mean", "sum", "none"] | None = "mean",
        torchmetrics_kwargs: dict[str, tp.Any] | None = None,
    ):
        torchmetrics_kwargs = torchmetrics_kwargs or {}
        super().__init__(**torchmetrics_kwargs)
        self.dim = dim
        self.reduction = reduction
        self._initialized = False
        self.sample_corrs: list[torch.Tensor]
        if self.dim == 1:
            self.add_state(
                "sample_corrs",
                default=[],
                dist_reduce_fx="cat",
            )

    @staticmethod
    def _batch_pearsonr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute per-sample Pearson r along dim=1.

        Parameters
        ----------
        x, y : torch.Tensor of shape ``(B, T)``

        Returns
        -------
        torch.Tensor of shape ``(B,)``
        """
        x_centered = x - x.mean(dim=1, keepdim=True)
        y_centered = y - y.mean(dim=1, keepdim=True)
        cov = (x_centered * y_centered).sum(dim=1)
        std_x = x_centered.pow(2).sum(dim=1).sqrt()
        std_y = y_centered.pow(2).sum(dim=1).sqrt()
        denom = std_x * std_y
        return torch.where(denom > 0, cov / denom, float("nan"))

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if self.dim == 1:
            corrs = self._batch_pearsonr(preds, target)
            self.sample_corrs.append(corrs)
            return

        # dim=0: online Welford accumulation via parent class
        if not self._initialized:
            self.num_outputs = preds.shape[1]
            state_names = [
                "mean_x",
                "mean_y",
                "max_abs_dev_x",
                "max_abs_dev_y",
                "var_x",
                "var_y",
                "corr_xy",
                "n_total",
            ]
            for state_name in state_names:
                self.add_state(
                    state_name,
                    default=torch.zeros(self.num_outputs, device=self.device),
                    dist_reduce_fx=None,
                )
            self._initialized = True

        super().update(preds, target)

    def compute(self):
        if self.dim == 1:
            corrs = dim_zero_cat(self.sample_corrs)
            if self.reduction == "mean":
                return torch.nanmean(corrs)
            elif self.reduction == "sum":
                return torch.nansum(corrs)
            else:
                return corrs

        corrcoef = super().compute()
        if self.reduction == "mean":
            return torch.nanmean(corrcoef)
        elif self.reduction == "sum":
            return torch.nansum(corrcoef)
        else:  # No reduction
            return corrcoef

    def reset(self) -> None:
        self._initialized = False
        super().reset()


class Rank(torchmetrics.Metric):
    """Rank of predictions based on a retrieval set, using cosine similarity.

    Parameters
    ----------
    reduction : {"mean", "median", "std"}
        How to reduce the example-wise ranks.
    relative : bool
        If True, divide ranks by the retrieval-set size so that values lie
        in [0, 1].
    torchmetrics_kwargs : dict or None
        Extra keyword arguments forwarded to the ``torchmetrics.Metric``
        constructor.
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = True

    def __init__(
        self,
        reduction: tp.Literal["mean", "median", "std"] = "median",
        relative: bool = False,
        torchmetrics_kwargs: dict[str, tp.Any] | None = None,
    ):
        torchmetrics_kwargs = torchmetrics_kwargs or {}
        super().__init__(**torchmetrics_kwargs)

        self.reduction = reduction
        self.relative = relative
        self.add_state(
            "ranks",
            default=torch.Tensor([]),
            dist_reduce_fx="cat",
        )
        self.rank_count: torch.Tensor  # For mypy

    @classmethod
    def _compute_sim(cls, x, y, norm_kind="y", eps=1e-15):
        if norm_kind is None:
            eq, inv_norms = "b", torch.ones(x.shape[0])
        elif norm_kind == "x":
            eq, inv_norms = "b", 1 / (eps + x.norm(dim=(1), p=2))
        elif norm_kind == "y":
            eq, inv_norms = "o", 1 / (eps + y.norm(dim=(1), p=2))
        elif norm_kind == "xy":
            eq = "bo"
            inv_norms = 1 / (
                eps + torch.outer(x.norm(dim=(1), p=2), y.norm(dim=(1), p=2))
            )
        else:
            raise ValueError(f"norm must be None, x, y or xy, got {norm_kind}.")

        # Normalize inside einsum to avoid creating a copy of candidates which can be pretty big
        return torch.einsum(f"bc,oc,{eq}->bo", x, y, inv_norms)

    @staticmethod
    def _compute_ranks_from_scores(
        scores: torch.Tensor,
        true_scores: torch.Tensor,
        retrieval_size: int | None,
    ) -> torch.Tensor:
        """Average ranks obtained with strictly greater-than and greater-than-or-equals operations
        to account for repeated scores.

        E.g., the zero-based rank of prediction "1" in [0, 1, 1, 1, 2] will be 2 (instead of 1
        or 3).
        """
        ranks_gt = (scores > true_scores).nansum(dim=1)
        ranks_ge = (scores >= true_scores).nansum(dim=1) - 1
        ranks = (ranks_gt + ranks_ge) / 2
        ranks[ranks < 0] = len(scores) // 2  # FIXME
        if retrieval_size is not None:
            ranks /= retrieval_size
        return ranks

    def _compute_ranks(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_labels: None | list[str] = None,
        y_labels: None | list[str] = None,
    ) -> torch.Tensor:
        scores = self._compute_sim(x, y)

        if x_labels is not None and y_labels is not None:
            # Use explicit mapping to match predictions and targets
            true_inds = torch.tensor(
                [y_labels.index(x) for x in x_labels],
                dtype=torch.long,
                device=scores.device,
            )[:, None]
            true_scores = torch.take_along_dim(scores, true_inds, dim=1)
        else:
            # Assume 1:1 mapping of predictions and targets
            if x_labels is not None or y_labels is not None:
                raise ValueError(
                    "x_labels and y_labels must both be None or both provided"
                )
            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    f"x and y must have same first dim, got {x.shape=} vs {y.shape=}"
                )
            true_scores = torch.diag(scores)[:, None]

        return self._compute_ranks_from_scores(
            scores, true_scores, len(y) if self.relative else None
        )

    @torch.inference_mode()
    def update(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_labels: None | list[str] = None,
        y_labels: None | list[str] = None,
    ) -> None:
        """Update internal list of ranks.

        Parameters
        ----------
        x :
            Tensor of predictions, of shape (N, F).
        y :
            Tensor of retrieval set examples, of shape (M, F).
        x_labels, y_labels :
            If provided, used to match predictions and ground truths that don't have the same
            number of examples. Should have length of N and M, respectively
        """
        ranks = self._compute_ranks(x, y, x_labels, y_labels)
        self.ranks = torch.cat([self.ranks, ranks])  # type: ignore

    def compute(self) -> torch.Tensor:
        agg_func: tp.Callable
        if self.reduction == "mean":
            agg_func = torch.mean
        elif self.reduction == "median":
            agg_func = torch.median
        elif self.reduction == "std":
            agg_func = torch.std
        else:
            raise ValueError(
                f'Unknown aggregation {self.reduction} for computing metric. Available aggregations are: "mean", "median" or "std".'
            )
        return agg_func(self.ranks)

    def _compute_macro_average(
        self, ranks: torch.Tensor, labels: list[str]
    ) -> dict[str, float]:
        """
        Compute the average rank for each class.
        """
        if len(ranks) != len(labels):
            raise ValueError(f"ranks/labels mismatch: {len(ranks)} vs {len(labels)}")
        groups = defaultdict(list)
        agg_func = np.mean if self.reduction == "mean" else np.median
        for i, label in enumerate(labels):
            groups[label].append(ranks[i])
        return {label: agg_func(ranks) for label, ranks in groups.items()}  # type: ignore

    @classmethod
    def _compute_topk_scores(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        y_labels: list[str] | None = None,
        k: int = 5,
    ) -> tuple[list[list[str]] | None, torch.Tensor, list[list[float]]]:
        """
        Compute the top-k predictions and scores for each example in x.
        If y_labels are provided, the function will return the actual top-k labels for each input example as well as the indices and similarity scores.
        """
        scores = cls._compute_sim(x, y)
        topk_inds = torch.argsort(scores, dim=1, descending=True)[:, :k]
        scores = [
            [scores[i, ind].item() for ind in inds] for i, inds in enumerate(topk_inds)
        ]
        topk_labels = None
        if y_labels is not None:
            topk_labels = [[y_labels[ind] for ind in inds] for inds in topk_inds]
        return topk_labels, topk_inds, scores


class TopkAcc(Rank):
    """Top-k accuracy.

    Parameters
    ----------
    topk :
        K in top-k, i.e. minimal rank to classify a prediction as a success.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    def __init__(self, topk: int = 5):
        super().__init__(relative=False)
        self.topk = topk

    def _compute_macro_average(
        self, ranks: torch.Tensor, labels: list[str]
    ) -> dict[str, float]:
        """
        Compute the top-k accuracy for each class.
        """
        groups = defaultdict(list)
        for i, label in enumerate(labels):
            groups[label].append(ranks[i])
        return {
            label: float(np.mean([r < self.topk for r in ranks]))
            for label, ranks in groups.items()
        }  # type: ignore

    def compute(self) -> torch.Tensor:
        ranks = self.ranks
        return (ranks < self.topk).float().mean()


class TopkAccFromScores(TopkAcc):
    """Top-k accuracy computed from already available similarity scores.

    Parameters
    ----------
    topk :
        K in top-k, i.e. minimal rank to classify a prediction as a success.
    true_labels :
        Defines where to look for the scores of true pairs. If "first", use the scores in the first
        column of the scores matrix; if "diagonal", use the scores on the diagonal.
    """

    def __init__(
        self, topk: int = 5, true_labels: tp.Literal["first", "diagonal"] = "first"
    ):
        super().__init__(topk)
        self.true_labels = true_labels

    def _compute_ranks(self, scores: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.true_labels == "first":
            true_scores = scores[:, [0]]
        elif self.true_labels == "diagonal":
            true_scores = torch.diag(scores)
        else:
            raise RuntimeError

        return self._compute_ranks_from_scores(scores, true_scores, None)

    @torch.inference_mode()
    def update(self, scores: torch.Tensor) -> None:  # type: ignore[override]
        """Update internal list of ranks."""
        ranks = self._compute_ranks(scores)
        self.ranks = torch.cat([self.ranks, ranks])  # type: ignore


class ImageSimilarity(torchmetrics.Metric):
    """Image similarity metric based on feature extraction from a pretrained network.

    Code adapted from:
    https://github.com/ozcelikfu/brain-diffuser/blob/main/scripts/evaluate_reconstruction.py
    https://github.com/ozcelikfu/brain-diffuser/blob/main/scripts/eval_extract_features.py

    Parameters
    ----------
    model_name : {"inceptionv3", "alexnet", "clip", "efficientnet", "swav"}
        Pretrained network used for feature extraction.
    layer : str or int
        Layer of the network to extract features from.  Valid values depend
        on *model_name*.
    torchmetrics_kwargs : dict or None
        Extra keyword arguments forwarded to the ``torchmetrics.Metric``
        constructor.
    """

    def __init__(
        self,
        model_name: tp.Literal[
            "inceptionv3", "alexnet", "clip", "efficientnet", "swav"
        ] = "inceptionv3",
        layer: str | int = "avgpool",
        torchmetrics_kwargs: dict[str, tp.Any] | None = None,
    ):
        torchmetrics_kwargs = torchmetrics_kwargs or {}
        super().__init__(**torchmetrics_kwargs)

        self.feat_list: list[torch.Tensor] = []

        if model_name == "inceptionv3":
            net = tvmodels.inception_v3(pretrained=True)
        elif model_name == "alexnet":
            net = tvmodels.alexnet(pretrained=True)
        elif model_name == "clip":
            import clip

            model, _ = clip.load("ViT-L/14")
            net = model.visual
            net = net.to(torch.float32)
        elif model_name == "efficientnet":
            net = tvmodels.efficientnet_b1(weights=True)
        elif model_name == "swav":
            net = torch.hub.load("facebookresearch/swav:main", "resnet50")

        self.net = net.float().eval()

        self.add_state("pred_features", [], dist_reduce_fx=None)
        self.add_state("true_features", [], dist_reduce_fx=None)

        if model_name == "clip":
            self.normalize = T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
        else:
            self.normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        if model_name in ["efficientnet", "swav"]:
            self.distance_fn = sp.spatial.distance.correlation

        self.model_name = model_name
        self.layer = layer

    def add_forward_hook_to_net(self, layer: str | int) -> None:
        if self.model_name == "inceptionv3":
            if layer == "avgpool":
                self.net.avgpool.register_forward_hook(self.fn)
            elif layer == "lastconv":
                self.net.Mixed_7c.register_forward_hook(self.fn)
            else:
                raise ValueError(
                    f"Unknown layer {layer} for InceptionV3. Available layers are: 'avgpool' or 'lastconv'."
                )

        elif self.model_name == "alexnet":
            if layer == 2:
                self.net.features[4].register_forward_hook(self.fn)
            elif layer == 5:
                self.net.features[11].register_forward_hook(self.fn)
            elif layer == 7:
                self.net.classifier[5].register_forward_hook(self.fn)
            else:
                raise ValueError(
                    f"Unknown layer {layer} for AlexNet. Available layers are: 2, 5 or 7."
                )

        elif self.model_name == "clip":
            if layer == 7:
                self.net.transformer.resblocks[7].register_forward_hook(self.fn)
            elif layer == 12:
                self.net.transformer.resblocks[12].register_forward_hook(self.fn)
            elif layer == "final":
                self.net.register_forward_hook(self.fn)
            else:
                raise ValueError(
                    f"Unknown layer {layer} for CLIP. Available layers are: 7, 12 or 'final'."
                )

        elif self.model_name == "efficientnet":
            self.net.avgpool.register_forward_hook(self.fn)

        elif self.model_name == "swav":
            self.net.avgpool.register_forward_hook(self.fn)

    def fn(self, module, inputs, outputs):
        self.feat_list.append(outputs)

    @torch.inference_mode()
    def update(
        self,
        preds: torch.Tensor,
        trues: torch.Tensor,
    ) -> None:
        """Update internal list of ranks.

        Parameters
        ----------
        preds :
            Tensor of predictions, of shape (N, 3, H, W).
        trues :
            Tensor of retrieval set examples, of shape (N, 3, H',W').
        """
        # Make sure that the net has a forward hook --> as with lightning callbacks, forward hooks are removed...
        self.add_forward_hook_to_net(self.layer)
        if len(preds) != len(trues):
            raise ValueError(f"preds/trues length mismatch: {len(preds)} vs {len(trues)}")
        n = len(preds)
        self.feat_list: list[torch.Tensor] = []  # type: ignore

        # data transform
        preds = T.functional.resize(preds, (224, 224))
        trues = T.functional.resize(trues, (224, 224))

        preds_trues = torch.cat([preds, trues], dim=0)
        preds_trues = self.normalize(preds_trues).float()

        # make sure that the net is in float
        self.net = self.net.float().to(self.device)

        _ = self.net(preds_trues.to(self.device))
        features: torch.Tensor | None = None
        if self.model_name == "clip":
            if self.layer in (7, 12):
                features = torch.cat(self.feat_list, dim=1).permute(1, 0, 2)  # type: ignore
            else:
                features = torch.cat(self.feat_list, dim=0)  # type: ignore
        else:
            features = torch.cat(self.feat_list, dim=0)  # type: ignore

        self.pred_features.append(features[:n])  # type: ignore
        self.true_features.append(features[n:])  # type: ignore

    def _pairwise_corr_all(
        self,
        ground_truth: np.ndarray | torch.Tensor,
        predictions: np.ndarray | torch.Tensor,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.cpu()
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu()
        r = np.corrcoef(ground_truth, predictions)
        r = r[: len(ground_truth), len(ground_truth) :]
        # congruent pairs are on diagonal
        congruents = np.diag(r)

        # for each column (prediction) we count the number of rows (ground truth) for which the value is lower than the congruent (e.g. success).
        success = r < congruents
        success_cnt = np.sum(success, 0)

        # note: diagonal of 'success' is always zero so we can discard it. That's why we divide by len-1
        perf = np.mean(success_cnt) / (len(ground_truth) - 1)
        p = 1 - binom.cdf(
            perf * len(ground_truth) * (len(ground_truth) - 1),
            len(ground_truth) * (len(ground_truth) - 1),
            0.5,
        )

        return perf, p

    def compute(self) -> torch.Tensor:
        preds_feat_tensor = dim_zero_cat(self.pred_features)  # type: ignore
        trues_feat_tensor = dim_zero_cat(self.true_features)  # type: ignore
        preds_feat: np.ndarray = (
            preds_feat_tensor.reshape((len(preds_feat_tensor), -1)).cpu().numpy()
        )
        trues_feat: np.ndarray = (
            trues_feat_tensor.reshape((len(trues_feat_tensor), -1)).cpu().numpy()
        )
        n = len(preds_feat)

        if self.model_name in ["efficientnet", "swav"]:
            distances = np.array(
                [self.distance_fn(trues_feat[i], preds_feat[i]) for i in range(n)]
            )
            val = np.mean(distances)
        else:
            val = self._pairwise_corr_all(trues_feat, preds_feat)[0]

        return torch.tensor(val, device=self.pred_features[0].device)  # type: ignore


class GroupedMetric(torchmetrics.Metric):
    """
    A wrapper around a torchmetrics.Metric that allows for computing metrics per group.
    IMPORTANT: this metric does not work well with LightningModule, because the
    self.log() method does not support dictionaries of metrics.

    To use this metric, you need to add this in the on_val_epoch_end and on_test_epoch_end methods:
        metric_dict = {metric_name + "/" + k: v for k, v in grouped_metric.compute().items()}
        self.log_dict(metric_dict)
        grouped_metric.reset()
    """

    def __init__(self, metric_name: str, kwargs: dict[str, tp.Any] | None = None) -> None:
        super().__init__()
        if kwargs is None:
            kwargs = {}
        from neuraltrain.metrics.base import TORCHMETRICS_NAMES

        if metric_name in TORCHMETRICS_NAMES:
            self.base_metric_cls = TORCHMETRICS_NAMES[metric_name]
        else:
            metric_cls = globals().get(metric_name)
            if metric_cls is None:
                raise ValueError(f"Metric {metric_name} not found")
            self.base_metric_cls = metric_cls
        self.metric_kwargs = kwargs
        self.metrics = torch.nn.ModuleDict()  # store metrics per group

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        groups: torch.Tensor | None = None,
    ) -> None:
        """
        Update each group's metric separately.
        groups: a tensor or list of group identifiers, same shape as preds/target.

        """
        if groups is None:
            groups = torch.zeros(preds.shape[0])
        else:
            groups = groups.flatten()
            if len(groups) != preds.shape[0]:
                raise ValueError(
                    f"groups length ({len(groups)}) must match preds ({preds.shape[0]})"
                )

        # Use pandas.groupby, faster than computing masks by hand
        groups_df = pd.DataFrame({"label": groups.tolist()})
        for group_id, group in groups_df.groupby("label", sort=False):
            mask = group.index.to_numpy()
            group_preds = preds[mask]
            group_target = target[mask]

            group_key = str(group_id)
            if group_key not in self.metrics:
                self.metrics[group_key] = self.base_metric_cls(**self.metric_kwargs)
                self.metrics[group_key] = self.metrics[group_key].to(preds.device)

            self.metrics[group_key].update(group_preds, group_target)  # type: ignore

    def compute(self) -> dict[str, float]:
        # Return a dictionary of group_id: computed_metric
        return {
            gid: metric.compute().item()  # type: ignore[operator]
            for gid, metric in self.metrics.items()  # type: ignore
        }

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()  # type: ignore

    def __repr__(self) -> str:
        return f"GroupedMetric({self.base_metric_cls.__name__})"
