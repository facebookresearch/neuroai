"""
Modifying the Training Loop
============================

NeuralBench uses a PyTorch Lightning ``LightningModule`` called
:class:`~neuralbench.pl_module.BrainModule` to handle training,
validation, and testing.  This tutorial explains how the training loop
works and how to customize it by subclassing ``BrainModule``.
"""

# %%
# How BrainModule works
# ---------------------
#
# ``BrainModule`` lives in ``neuralbench/pl_module.py``.  Its key
# components are:
#
# - **``model_forward(batch)``** -- runs the model on a batch,
#   automatically passing ``subject_ids`` and ``channel_positions``
#   when the model's ``forward`` signature requires them.
#
# - **``_run_step(batch, step_name, batch_idx)``** -- shared logic
#   for train, validation, and test steps:
#
#   1. Extracts ``y_true`` from ``batch.data["target"]``
#   2. Applies target scaling if configured (e.g., for regression)
#   3. Converts targets for the loss function (one-hot to argmax
#      for ``CrossEntropyLoss``, clamping for ``BCEWithLogitsLoss``)
#   4. Calls ``model_forward(batch)`` to get predictions
#   5. Computes and logs the loss
#   6. Updates all metrics matching the current step name
#
# - **``training_step``**, **``validation_step``**,
#   **``test_step``** -- thin wrappers around ``_run_step`` that
#   return the appropriate values for Lightning.
#
# - **``configure_optimizers``** -- builds the optimizer and
#   learning rate scheduler from the config, injecting
#   ``total_steps`` / ``T_max`` automatically.
#
# Here is ``_run_step``:
#
# .. code-block:: python
#
#    def _run_step(
#        self, batch: Batch, step_name: str, batch_idx: int
#    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#        y_true = batch.data["target"]
#
#        if self.target_scaler is not None:
#            y_true = self.target_scaler.transform(y_true)
#        if y_true.ndim == 3 and y_true.shape[1] == 1:
#            y_true = y_true.squeeze(1)
#
#        if isinstance(self.loss, nn.CrossEntropyLoss):
#            assert y_true.ndim == 2
#            y_true = y_true.argmax(dim=1)
#        elif isinstance(self.loss, nn.BCEWithLogitsLoss):
#            y_true = y_true.clamp(max=1.0)
#
#        log_kwargs = {
#            "on_step": step_name == "train",
#            "on_epoch": True,
#            "logger": True,
#            "prog_bar": True,
#            "batch_size": y_true.shape[0],
#            "sync_dist": self.trainer.world_size > 1,
#        }
#
#        y_pred = self.model_forward(batch)
#
#        if y_pred.ndim == 3 and y_true.ndim == 3:
#            y_pred = y_pred.reshape(y_pred.shape[0], -1)
#            y_true = y_true.reshape(y_true.shape[0], -1)
#
#        loss = self.loss(y_pred, y_true)
#        self.log(f"{step_name}/loss", loss, **log_kwargs)
#
#        for metric_name, metric in self.metrics.items():
#            if metric_name.startswith(step_name):
#                metric.update(y_pred, y_true)
#                if "confusion_matrix" not in metric_name:
#                    self.log(metric_name, metric, **log_kwargs)
#
#        return loss, y_pred, y_true
#
# The training step simply calls ``_run_step`` and returns the loss
# for backpropagation:
#
# .. code-block:: python
#
#    def training_step(self, batch: Batch, batch_idx: int):
#        loss, _, _ = self._run_step(batch, step_name="train", batch_idx=batch_idx)
#        return loss

# %%
# Subclassing BrainModule
# -----------------------
#
# To customize the training loop, subclass ``BrainModule`` and
# override the methods you need.  The rest of the pipeline
# (data loading, checkpointing, testing) remains unchanged.
#
# **Example: adding an L2 regularization term to the training loss.**

import torch

from neuralbench.pl_module import BrainModule


class RegularizedBrainModule(BrainModule):
    """BrainModule with an L2 penalty on model weights."""

    def __init__(self, *args, l2_lambda: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.l2_lambda = l2_lambda

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._run_step(batch, step_name="train", batch_idx=batch_idx)

        l2_reg = sum(p.pow(2).sum() for p in self.model.parameters())
        loss = loss + self.l2_lambda * l2_reg

        self.log("train/l2_reg", l2_reg, prog_bar=False)
        return loss


# %%
# **Example: logging additional metrics during validation.**


class VerboseValBrainModule(BrainModule):
    """BrainModule that logs extra info during validation."""

    def validation_step(self, batch, batch_idx):
        _, y_pred, y_true = self._run_step(batch, step_name="val", batch_idx=batch_idx)
        confidence = torch.softmax(y_pred, dim=-1).max(dim=-1).values.mean()
        self.log("val/mean_confidence", confidence, prog_bar=True)
        return y_pred, y_true


# %%
# What you can customize
# ----------------------
#
# By overriding different methods on ``BrainModule``, you can:
#
# - **Add auxiliary losses** (e.g., contrastive terms, regularization)
#   by overriding ``training_step``.
# - **Change how predictions are computed** by overriding
#   ``model_forward``.
# - **Modify target preprocessing** by overriding ``_run_step``.
# - **Add custom logging or callbacks** in any step method.
# - **Change the optimizer or scheduler** by overriding
#   ``configure_optimizers``.
#
# Because NeuralBench is built on PyTorch Lightning, the full
# Lightning API is available.  See the
# `Lightning docs <https://lightning.ai/docs/pytorch/stable/>`_
# for more advanced hooks (``on_train_epoch_end``,
# ``on_before_optimizer_step``, etc.).
