# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Trainer and Experiments
=======================

This tutorial shows how to wire configs into a Lightning training loop
and package everything into a reproducible experiment.

.. note::

   This tutorial requires the ``lightning`` extra::

      pip install 'neuraltrain[lightning]'

.. seealso::

   :doc:`/neuraltrain/project_example` -- full training project
"""

# %%
# Building the training pieces
# ----------------------------
#
# All training pieces start as typed configs:

import torch
import torch.nn as nn

from neuraltrain import BaseLoss, BaseMetric, LightningOptimizer, models

model_config = models.SimpleConvTimeAgg(hidden=32, depth=4, merger_config=None)
loss_config = BaseLoss(name="CrossEntropyLoss")
metric_config = BaseMetric(
    name="Accuracy",
    log_name="acc",
    kwargs={"task": "multiclass", "num_classes": 4},
)
optim_config = LightningOptimizer(
    optimizer={"name": "Adam", "lr": 1e-4},
    scheduler={"name": "OneCycleLR", "kwargs": {"max_lr": 3e-3, "pct_start": 0.2}},
)

brain_model = model_config.build(n_in_channels=204, n_outputs=4).cpu()
loss = loss_config.build()
metrics = {metric_config.log_name: metric_config.build()}

print("Model:    ", type(brain_model).__name__)
print("Loss:     ", loss)
print("Metrics:  ", metrics)
print("Optimiser:", optim_config)

# %%
# BrainModule
# -----------
#
# ``BrainModule`` is a ``LightningModule`` that wires the built pieces
# into a training loop. It lives in ``project_example/pl_module.py``
# -- you own the full training logic.
#
# Key methods:
#
# - ``forward`` -- extracts ``batch.data[x_name]`` and calls the model
# - ``_run_step`` -- computes loss, updates metrics, logs both
# - ``configure_optimizers`` -- builds the optimiser from config,
#   passing ``total_steps`` for schedulers that need it
#
# Here is the full implementation:

import lightning.pytorch as pl
from torchmetrics import Metric

from neuraltrain.optimizers import BaseOptimizer


class BrainModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optim_config: BaseOptimizer,
        metrics: dict[str, Metric],
        x_name: str = "input",
        y_name: str = "target",
        max_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.model = model
        self.x_name, self.y_name = x_name, y_name
        self.optim_config = optim_config
        self.max_epochs = max_epochs
        self.loss = loss
        self.metrics = nn.ModuleDict(
            {split + "_" + k: v for k, v in metrics.items() for split in ["val", "test"]}
        )

    def forward(self, batch):
        return self.model(batch.data[self.x_name])

    def _run_step(self, batch, batch_idx, step_name):
        y_true = batch.data[self.y_name].squeeze(-1)
        y_pred = self.forward(batch)
        loss = self.loss(y_pred, y_true)
        self.log(f"{step_name}_loss", loss, on_epoch=True, prog_bar=True)
        for name, metric in self.metrics.items():
            if name.startswith(step_name):
                metric.update(y_pred, y_true)
                self.log(name, metric, on_epoch=True, prog_bar=True)
        return loss, y_pred, y_true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._run_step(batch, batch_idx, step_name="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, y_pred, y_true = self._run_step(batch, batch_idx, step_name="val")
        return y_pred, y_true

    def test_step(self, batch, batch_idx):
        self._run_step(batch, batch_idx, step_name="test")

    def configure_optimizers(self):
        try:
            return self.optim_config.build(
                self.parameters(),
                total_steps=self.trainer.estimated_stepping_batches,
            )
        except TypeError:
            return self.optim_config.build(self.parameters())


# %%
# Let's instantiate it and run a forward pass:

brain_module = BrainModule(
    model=brain_model,
    loss=loss,
    optim_config=optim_config,
    metrics=metrics,
    max_epochs=20,
)

x = torch.randn(8, 204, 60)
target = torch.randint(0, 4, (8,))

brain_module.eval()
with torch.no_grad():
    y_pred = brain_module.model(x)
    loss_val = brain_module.loss(y_pred, target)

print(f"Predictions: {tuple(y_pred.shape)}")
print(f"Loss:        {loss_val.item():.4f}")

# %%
# The Experiment class
# --------------------
#
# ``Experiment`` is a ``pydantic.BaseModel`` that collects *all*
# training pieces -- data, model, loss, optimiser, metrics, loggers,
# and infrastructure -- in one validated object:
#
# .. code-block:: python
#
#    class Experiment(pydantic.BaseModel):
#        data: Data
#        brain_model_config: BaseModelConfig
#        loss: BaseLoss
#        optim: LightningOptimizer
#        metrics: list[BaseMetric]
#        csv_config: CsvLoggerConfig | None = None
#        wandb_config: WandbLoggerConfig | None = None
#        infra: TaskInfra = TaskInfra(version="1")
#
# Its ``run()`` method orchestrates the full pipeline:
#
# .. code-block:: python
#
#    @infra.apply
#    def run(self) -> dict[str, float | None]:
#        pl.seed_everything(self.seed, workers=True)
#        loaders = self.data.build()
#        brain_module = self._build_brain_module(loaders["train"])
#        trainer = self._setup_trainer()
#        if not self.test_only:
#            self.fit(brain_module, trainer, loaders["train"], loaders["val"])
#        return self.test(brain_module, trainer, loaders["test"])
#
# The ``@infra.apply`` decorator makes ``run()`` cacheable and
# Slurm-submittable.

# %%
# TaskInfra
# ---------
#
# ``exca.TaskInfra`` controls where and how the experiment runs:

import exca

infra = exca.TaskInfra(
    cluster=None,
    folder="/tmp/results/mne_sample_clf",
    gpus_per_node=1,
    cpus_per_task=10,
)
print(infra)

# %%
# Key settings:
#
# - ``cluster=None`` -- run locally
# - ``cluster="auto"`` -- submit to Slurm if available
# - ``folder`` -- output directory and cache root

# %%
# Hyperparameter sweeps
# ---------------------
#
# ``run_grid()`` expands a parameter grid into experiment configs:
#
# .. code-block:: python
#
#    from neuraltrain.utils import run_grid
#
#    results = run_grid(
#        Experiment, "hp_search", default_config,
#        grid, combinatorial=True,
#    )

grid = {
    "brain_model_config.hidden": [32, 64],
    "n_epochs": [20, 50],
    "seed": [33, 87],
}

n_combinations = 1
for key, values in grid.items():
    n_combinations *= len(values)
    print(f"  {key}: {values}")
print(f"\nTotal combinations: {n_combinations}")

# %%
# Callbacks and loggers
# ---------------------
#
# The experiment's ``_setup_trainer()`` wires Lightning callbacks and
# loggers:

from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from neuraltrain import utils

callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", patience=5),
    LearningRateMonitor(logging_interval="epoch"),
    ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        dirpath="/tmp/checkpoints",
        filename="best",
        monitor="val_loss",
    ),
]
for cb in callbacks:
    print(type(cb).__name__)

csv_config = utils.CsvLoggerConfig(name="mne_sample_clf")
wandb_config = utils.WandbLoggerConfig(
    log_model=False, group="mne_sample_clf", project="mne_sample_clf"
)
print("CSV:  ", csv_config)
print("W&B:  ", wandb_config)

# %%
# .. raw:: html
#
#    <div class="page-nav">
#      <a href="../objectives/03_objectives.html" class="page-nav-btn page-nav-btn--outline">&larr; Tutorial 3: Objectives</a>
#      <a href="../../project_example.html" class="page-nav-btn">Project Example &rarr;</a>
#    </div>
