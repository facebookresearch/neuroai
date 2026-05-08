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
    ):
        super().__init__()
        self._infer_forward_params(model)
        self.model = model

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
        forward_sig = inspect.signature(inner_model.forward)
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
        inputs = {self._input_name: batch.data["neuro"]}
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

        if y_pred.ndim == 3 and y_true.ndim == 3:
            y_pred = y_pred.reshape(y_pred.shape[0], -1)
            y_true = y_true.reshape(y_true.shape[0], -1)

        loss = self.loss(y_pred, y_true)
        self.log(f"{step_name}/loss", loss, **log_kwargs)

        # Just update metrics, don't compute or log yet
        for metric_name, metric in self.metrics.items():
            if metric_name.startswith(step_name):
                if isinstance(metric, GroupedMetric):
                    metric.update(y_pred, y_true, batch.data["subject_id"])
                else:
                    if isinstance(metric, MultilabelConfusionMatrix):
                        metric.update(y_pred, y_true.int())
                    else:
                        metric.update(y_pred, y_true)
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
