# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import typing as tp

import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics import Metric
from torchmetrics.classification import MultilabelConfusionMatrix

from neuralset.dataloader import Batch
from neuraltrain.metrics.metrics import GroupedMetric
from neuraltrain.optimizers import LightningOptimizer
from neuraltrain.utils import StandardScaler

from .modules import DownstreamWrapperModel

LOGGER = logging.getLogger(__name__)


class BrainModule(pl.LightningModule):
    """
    Pytorch-lightning module for M/EEG model training.

    Parameters
    ----------
    model : nn.Module
        The brain model to be trained.
    loss : nn.Module
        The loss function.
    metrics : dict[str, Metric]
        A dictionary of metrics to compute during validation and testing.
    test_full_metrics : dict[str, Metric] | None, optional
        A dictionary of metrics to compute on the full test set.
    test_full_retrieval_metrics : dict[str, Metric] | None, optional
        A dictionary of retrieval metrics to compute on the full test set.
    target_scaler : nn.Module | None, optional
        A scaler to apply to the target values.
    augmentation : nn.Module | None, optional
        Augmentation applied to the neuro input during training only.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        metrics: dict[str, Metric],
        lightning_optimizer_config: LightningOptimizer,
        test_full_metrics: dict[str, Metric] | None = None,
        test_full_retrieval_metrics: dict[str, Metric] | None = None,
        target_scaler: StandardScaler | None = None,
        augmentation: nn.Module | None = None,
    ):
        super().__init__()
        self._infer_forward_params(model)
        self.model = model
        self.augmentation = augmentation

        self.loss = loss
        self.target_scaler = target_scaler
        self.lightning_optimizer_config = lightning_optimizer_config

        self.metrics = self._update_metrics(
            metrics,
            split_names=["val", "test"],
        )
        self.test_full_metrics: nn.ModuleDict | None = None
        if test_full_metrics is not None:
            self.test_full_metrics = self._update_metrics(
                test_full_metrics,
                split_names=["test/full"],
            )
        self.test_full_retrieval_metrics: nn.ModuleDict | None = None
        if test_full_retrieval_metrics is not None:
            self.test_full_retrieval_metrics = self._update_metrics(
                test_full_retrieval_metrics,
                split_names=["test/full_retrieval"],
            )

    def _infer_forward_params(self, model: nn.Module) -> None:
        """Check which additional inputs the model's forward method requires."""
        inner_model = model
        has_preprocessor = False
        adapter_needs_positions = False
        if isinstance(model, DownstreamWrapperModel):
            has_preprocessor = model.preprocessor is not None
            adapter_needs_positions = (
                model.channel_adapter is not None and model._adapter_needs_positions
            )
            inner_model = model.wrapped_model
        # PeftModel.forward is a (*args, **kwargs) delegator: unwrap for the real signature
        sig_model = getattr(inner_model, "get_base_model", lambda: inner_model)()
        forward_sig = inspect.signature(sig_model.forward)
        self._input_name = list(forward_sig.parameters.keys())[0]
        self._requires_subject = (
            "subject_ids" in forward_sig.parameters or adapter_needs_positions
        )
        self._requires_channel_positions = (
            "channel_positions" in forward_sig.parameters
            or has_preprocessor
            or adapter_needs_positions
        )

    @staticmethod
    def _update_metrics(
        metrics: dict[str, Metric], split_names: list[str]
    ) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                split + "/" + k: v.clone()
                for k, v in metrics.items()
                for split in split_names
            }
        )

    def model_forward(self, batch: Batch) -> torch.Tensor:
        neuro = batch.data["neuro"]
        if self.augmentation is not None and self.training:
            neuro = self.augmentation(neuro)
        inputs = {self._input_name: neuro}
        if self._requires_subject:
            inputs["subject_ids"] = batch.data["subject_id"]
        if self._requires_channel_positions:
            inputs["channel_positions"] = batch.data["channel_positions"]
        return self.model(**inputs)

    def model_forward_embedding(self, batch: Batch) -> torch.Tensor:
        """Forward pass returning the penultimate embedding (before the probe)."""
        inputs = {self._input_name: batch.data["neuro"]}
        if self._requires_subject:
            inputs["subject_ids"] = batch.data["subject_id"]
        if self._requires_channel_positions:
            inputs["channel_positions"] = batch.data["channel_positions"]
        return self.model(**inputs, return_embedding=True)

    def _run_step(
        self, batch: Batch, step_name: str, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_true = batch.data["target"]

        if self.target_scaler is not None:
            y_true = self.target_scaler.transform(y_true)
        if y_true.ndim == 3 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)

        if isinstance(self.loss, nn.CrossEntropyLoss):
            # Convert back from one-hot to inds
            assert y_true.ndim == 2  # (batch_size, n_classes)
            y_true = y_true.argmax(dim=1)
        elif isinstance(self.loss, nn.BCEWithLogitsLoss):
            y_true = y_true.clamp(max=1.0)

        log_kwargs: dict[str, tp.Any] = {
            "on_step": step_name == "train",
            "on_epoch": True,
            "logger": True,
            "prog_bar": True,
            "batch_size": y_true.shape[0],
            "sync_dist": self.trainer.world_size > 1,
        }

        y_pred = self.model_forward(batch)

        metric_pred, metric_true = y_pred, y_true
        loss_pred, loss_true = y_pred, y_true
        metric_subjects = batch.data["subject_id"]
        if y_pred.ndim == 3 and y_true.ndim == 3:
            # A dense prediction is time-major (B, T, C) -- braindecode's
            # convention -- while an extracted target is channel-major (B, C, T).
            y_true = y_true.transpose(1, 2)
            if y_pred.shape[-1] != y_true.shape[-1]:
                raise ValueError(
                    f"Prediction carries {y_pred.shape[-1]} outputs per frame but "
                    f"its target {y_true.shape[-1]}; the head is sized from the "
                    f"target's channel axis in build_brain_model."
                )
            if y_pred.shape[1] > y_true.shape[1]:
                raise ValueError(
                    f"Prediction spans {y_pred.shape[1]} frames but its target "
                    f"only {y_true.shape[1]}."
                )
            # Models that consume left context emit only the window's valid tail.
            y_true = y_true[:, -y_pred.shape[1] :]
            # NaN marks a frame the study could not label (e.g. unresolved
            # inverse kinematics); it must reach neither the loss nor a metric.
            valid = ~y_true.isnan().any(dim=-1)
            metric_pred = loss_pred = y_pred[valid]
            metric_true = loss_true = y_true[valid]
            # Metric rows are frames, not windows: (B, 1) -> (B, T).
            metric_subjects = metric_subjects.reshape(-1, 1).expand_as(valid)[valid]
            y_pred = y_pred.masked_fill(~valid.unsqueeze(-1), torch.nan)
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
            y_true = y_true.reshape(y_true.shape[0], -1)

        if isinstance(self.loss, nn.CTCLoss):
            # ``nn.CTCLoss`` needs four arguments and a transposed time axis.
            # ``y_pred`` is ``(B, T_out, C)`` log-probs from a CTC head;
            # ``y_true`` is ``(B, max_length)`` integer labels padded with
            # the blank index (see ``SequenceLabelEncoder`` with
            # ``aggregation='cat'`` + ``max_length``).
            blank = self.loss.blank
            y_true = y_true.long()
            target_lengths = (y_true != blank).sum(dim=-1)
            input_lengths = torch.full(
                (y_true.shape[0],),
                y_pred.shape[1],
                dtype=torch.long,
                device=y_pred.device,
            )
            loss = self.loss(
                y_pred.transpose(0, 1), y_true, input_lengths, target_lengths
            )
        elif loss_true.numel() == 0:
            # Every frame unlabelled: a zero still attached to the graph.
            loss = loss_pred.sum()
        else:
            loss = self.loss(loss_pred, loss_true)

        # A loss may return a dict of named components with a ``"total"``
        # key (e.g. multi-term objectives); CTC and plain losses return a
        # single tensor and fall through to the ``else`` branch.
        if isinstance(loss, dict):
            loss_total = loss["total"]
            for k, v in loss.items():
                if k == "total":
                    self.log(f"{step_name}/loss", v, **log_kwargs)
                else:
                    self.log(f"{step_name}/loss_{k}", v, **log_kwargs)
            loss = loss_total
        else:
            self.log(f"{step_name}/loss", loss, **log_kwargs)

        # Just update metrics, don't compute or log yet
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(step_name) and metric_true.numel():
                if isinstance(metric, GroupedMetric):
                    metric.update(metric_pred, metric_true, metric_subjects)
                else:
                    if isinstance(metric, MultilabelConfusionMatrix):
                        metric.update(metric_pred, metric_true.int())
                    else:
                        metric.update(metric_pred, metric_true)
                if "confusion_matrix" not in metric_name:
                    self.log(metric_name, metric, **log_kwargs)

        return loss, y_pred, y_true

    def training_step(self, batch: Batch, batch_idx: int):
        loss, _, _ = self._run_step(batch, step_name="train", batch_idx=batch_idx)
        return loss

    def validation_step(self, batch, batch_idx: int):
        _, y_pred, y_true = self._run_step(batch, step_name="val", batch_idx=batch_idx)
        return y_pred, y_true

    def test_step(self, batch, batch_idx: int):
        _, y_pred, y_true = self._run_step(batch, step_name="test", batch_idx=batch_idx)
        return y_pred, y_true

    # Schedulers that need the total training step count at build time.
    _SCHEDULER_STEP_KWARG: tp.ClassVar[dict[type, str]] = {}

    def configure_optimizers(self):  # type: ignore[override]
        # Get scheduler-specific kwargs
        scheduler_build_kwargs = {}
        if self.lightning_optimizer_config.scheduler is not None:
            name = self.lightning_optimizer_config.scheduler.__class__.__name__
            if name == "OneCycleLR":
                scheduler_build_kwargs = {
                    "total_steps": self.trainer.estimated_stepping_batches
                }
            elif name == "CosineAnnealingLR":
                scheduler_build_kwargs = {
                    "T_max": self.trainer.estimated_stepping_batches
                }
            else:
                raise NotImplementedError(f"Scheduler {name} not implemented")

        return self.lightning_optimizer_config.build(
            self.parameters(), **scheduler_build_kwargs
        )
