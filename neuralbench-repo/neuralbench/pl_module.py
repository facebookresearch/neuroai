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


class CtcMetricRegistry:
    """Per-task registry of CTC metric *factory builders*.

    A factory builder takes the active extractor and returns a no-arg
    ``Callable[[], Metric]`` for ``BrainModule`` to call.  This indirection
    lets the metric capture per-experiment state (e.g. the extractor's
    charset) at ``Experiment.prepare_pl_module`` time, so multi-experiment
    grids running different vocabularies get independent metrics.

    Tasks register a builder during their package import (triggered lazily
    by ``experiment_config._maybe_import_task_module``).  Re-registering
    with a different builder under the same task name logs a warning.
    """

    def __init__(self) -> None:
        self._builders: dict[
            str, tp.Callable[[tp.Any], tp.Callable[[], Metric]]
        ] = {}

    def register(
        self,
        task_name: str,
        factory_builder: tp.Callable[[tp.Any], tp.Callable[[], Metric]],
    ) -> None:
        prev = self._builders.get(task_name)
        if prev is not None and prev is not factory_builder:
            LOGGER.warning(
                "CTC metric factory builder for task %r overwritten: %r → %r",
                task_name, prev, factory_builder,
            )
        self._builders[task_name] = factory_builder

    def get(
        self, task_name: str
    ) -> tp.Callable[[tp.Any], tp.Callable[[], Metric]] | None:
        return self._builders.get(task_name)


# Single shared instance.  Holding the registry on a class (rather than as
# free-floating module-level functions on a global dict) keeps state out of
# module scope and makes per-test isolation possible (instantiate a fresh
# registry in a fixture) when needed.
ctc_metric_registry = CtcMetricRegistry()


def register_ctc_metric(
    task_name: str,
    factory_builder: tp.Callable[[tp.Any], tp.Callable[[], Metric]],
) -> None:
    """Convenience wrapper around :meth:`ctc_metric_registry.register`."""
    ctc_metric_registry.register(task_name, factory_builder)


def get_ctc_metric_factory_builder(
    task_name: str,
) -> tp.Callable[[tp.Any], tp.Callable[[], Metric]] | None:
    """Convenience wrapper around :meth:`ctc_metric_registry.get`."""
    return ctc_metric_registry.get(task_name)


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
        ctc_metric_factory: tp.Callable[[], Metric] | None = None,
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
        # CTC metric is injected (not constructed here) so this generic
        # module doesn't depend on any task package.  Per-step instances
        # are stored as nn.Modules via add_module, looked up by name —
        # no parallel dict needed.
        self._ctc_metric_factory = ctc_metric_factory

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

        if isinstance(self.loss, nn.CTCLoss):
            return self._run_ctc_step(batch, y_true, step_name)

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

    def _get_ctc_metric(self, step_name: str) -> Metric:
        if self._ctc_metric_factory is None:
            raise RuntimeError(
                "BrainModule was constructed with CTCLoss but no "
                "ctc_metric_factory; pass one when building the module."
            )
        attr = f"_ctc_metric_{step_name}"
        metric = getattr(self, attr, None)
        if metric is None:
            metric = self._ctc_metric_factory()
            self.add_module(attr, metric)
        return metric

    def _run_ctc_step(
        self, batch: Batch, y_true: torch.Tensor, step_name: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # y_true layout (from KeystrokeSequence): col 0 = un-padded length,
        # cols 1: = padded labels. Model output is braindecode-shaped
        # (B, T_out, C); we transpose to (T_out, B, C) for CTCLoss.
        target_lengths, targets = y_true[:, 0].long(), y_true[:, 1:].long()
        B = y_true.shape[0]
        is_train = step_name == "train"
        log_kwargs = {
            "on_step": is_train, "on_epoch": True, "logger": True,
            "prog_bar": True, "batch_size": B,
            "sync_dist": self.trainer.world_size > 1,
        }

        y_pred = self.model_forward(batch)
        if y_pred.ndim != 3:
            raise RuntimeError(
                f"CTC model must emit a 3-D log-prob tensor; "
                f"got shape {tuple(y_pred.shape)}."
            )
        # Accept both conventions: braindecode's (B, T_out, C) and the
        # nn.CTCLoss-native (T_out, B, C) (used by upstream emg2qwerty TDS).
        # CTCLoss wants (T_out, B, C) and contiguous storage.
        if y_pred.shape[0] == B and y_pred.shape[1] != B:
            log_probs = y_pred.transpose(0, 1).contiguous()
        elif y_pred.shape[1] == B:
            log_probs = y_pred if y_pred.is_contiguous() else y_pred.contiguous()
        else:
            raise RuntimeError(
                f"CTC model output {tuple(y_pred.shape)} does not match "
                f"batch size {B} on dim 0 or 1; expected (B, T, C) or (T, B, C)."
            )
        input_lengths = torch.full(
            (B,), log_probs.shape[0], dtype=torch.long, device=log_probs.device
        )

        loss = self.loss(log_probs, targets, input_lengths, target_lengths)
        self.log(f"{step_name}/loss", loss, **log_kwargs)

        # Levenshtein decode runs on CPU — skip during train to keep the
        # per-step path light; val/test update every batch.
        if not is_train:
            metric = self._get_ctc_metric(step_name)
            metric.update(log_probs.detach(), targets, target_lengths)
            self.log(f"{step_name}/CER", metric, **log_kwargs)

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
