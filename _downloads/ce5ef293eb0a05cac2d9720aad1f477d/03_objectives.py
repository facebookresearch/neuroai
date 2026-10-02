# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Metrics and Optimisation
=========================

Losses, metrics, and optimisers are stored as typed config objects, just like
models. That keeps the whole training recipe serialisable and sweepable.

This tutorial covers:

- configuring losses, metrics, and optimisers
- building runtime objects from configs
- scheduler integration

"""

# %%
# Loss config
# -----------
#
# ``BaseLoss`` is a discriminated pydantic model. All standard PyTorch
# losses are auto-registered, plus custom losses like ``ClipLoss`` and
# ``SigLipLoss``.

from neuraltrain import BaseLoss

loss_config = BaseLoss(name="CrossEntropyLoss", kwargs={"reduction": "mean"})
loss = loss_config.build()
print("Built:  ", loss)

# %%
# Metric config
# -------------
#
# ``BaseMetric`` is a discriminated pydantic model. All standard torchmetrics
# are auto-registered, plus custom metrics like ``Rank`` and
# ``ImageSimilarity``.

from neuraltrain import BaseMetric

metric_config = BaseMetric(
    name="Accuracy",
    log_name="acc",
    kwargs={"task": "multiclass", "num_classes": 4},
)
metric = metric_config.build()
print("Built:  ", metric)


# %%
# Optimiser config
# ----------------
#
# ``LightningOptimizer`` bundles a torch optimiser with an optional
# scheduler. All ``torch.optim`` optimisers and
# ``torch.optim.lr_scheduler`` schedulers are auto-registered:

from neuraltrain import LightningOptimizer

optim_config = LightningOptimizer(
    optimizer={"name": "Adam", "lr": 1e-4, "kwargs": {"weight_decay": 0.0}},
    scheduler={"name": "OneCycleLR", "kwargs": {"max_lr": 3e-3, "pct_start": 0.2}},
)
print(optim_config)

# %%
# ``build()`` is called inside ``BrainModule.configure_optimizers()``
# with the model parameters and scheduler kwargs:
#
# .. code-block:: python
#
#    def configure_optimizers(self):
#        try:
#            return self.optim_config.build(
#                self.parameters(),
#                total_steps=self.trainer.estimated_stepping_batches,
#            )
#        except TypeError:
#            return self.optim_config.build(self.parameters())


# %%
# All these objects serialise cleanly to JSON for experiment snapshots,
# and can be swept just like model configs:
#
# .. code-block:: python
#
#    grid = {
#        "loss.name": ["CrossEntropyLoss", "ClipLoss"],
#        "optim.optimizer.lr": [1e-3, 1e-4],
#    }

# %%
# .. raw:: html
#
#    <div class="page-nav">
#      <a href="../models/02_models.html" class="page-nav-btn page-nav-btn--outline">&larr; Tutorial 2: Models</a>
#      <a href="../trainer/04_trainer.html" class="page-nav-btn">Tutorial 4: Trainer &rarr;</a>
#    </div>
