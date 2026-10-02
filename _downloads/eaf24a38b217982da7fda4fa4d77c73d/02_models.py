# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Model Configs
=============

``neuraltrain`` represents models as pydantic configs. The config is what gets
stored in your experiment; the actual ``torch.nn.Module`` is built later, once
the input and output dimensions are known.

This tutorial covers:

- the config-first model pattern
- instantiating and building a model from config
- available model families and shared building blocks
- sweeping over model hyperparameters
"""

# %%
# Config first, module later
# --------------------------
#
# The central abstraction is ``BaseModelConfig``. Each subclass is a typed
# pydantic model that knows how to ``build()`` its ``nn.Module`` when runtime
# dimensions are available.
#
# In the example project, the model is built inside
# ``Experiment._build_brain_module()``, once the first batch reveals the
# input shape:
#
# .. code-block:: python
#
#    batch = next(iter(train_loader))
#    n_chans = batch.data["input"].shape[1]
#    n_classes = train_loader.dataset.triggers.code.nunique()
#    brain_model = self.brain_model_config.build(
#        n_chans=n_chans, n_outputs=n_classes,
#    )
#
# Let's instantiate the default model config (``SimpleConvTimeAgg``, a
# 1-D convolutional encoder with temporal aggregation):

from neuraltrain import BaseModelConfig, models

model_config = models.SimpleConvTimeAgg(hidden=32, depth=4, merger_config=None)
print(model_config)

# %%
# Building the model
# ------------------
#
# Calling ``build()`` with the runtime dimensions produces a concrete
# ``nn.Module`` that we can inspect:

model = model_config.build(n_in_channels=204, n_outputs=4).cpu()
print(model)

# %%
# Shared building blocks
# ----------------------
#
# ``neuraltrain.models`` provides reusable pieces that custom
# models can compose, as shown below.
#

import torch.nn as nn

from neuraltrain.models import common, transformer


class MyModelImpl(nn.Module):
    def __init__(
        self,
        dim: int,
        n_outputs: int,
        merger: common.ChannelMerger,
        downsampler: common.TemporalDownsampling,
        encoder: transformer.TransformerEncoder,
        projector: common.Mlp,
    ):
        super().__init__()
        self.merger = merger.build()
        self.downsampler = downsampler.build(dim=dim)
        self.encoder = encoder.build(dim=dim)
        self.projector = projector.build(input_size=dim, output_size=n_outputs)

    def forward(self, x, subject_ids, positions):
        x = self.merger(x, subject_ids, positions)  # (B, C, T) -> (B, n_virtual, T)
        x = x.transpose(1, 2).unsqueeze(1)  # (B, 1, T, n_virtual)
        x = self.downsampler(x)  # (B, 1, T', n_virtual)
        x = x.squeeze(1)  # (B, T', n_virtual)
        x = self.encoder(x)  # (B, T', n_virtual)
        x = x.mean(dim=1)  # (B, n_virtual)
        return self.projector(x)  # (B, n_outputs)


class MyModel(BaseModelConfig):
    merger: common.ChannelMerger = common.ChannelMerger(
        n_virtual_channels=256, per_subject=True
    )
    downsampler: common.TemporalDownsampling = common.TemporalDownsampling()
    encoder: transformer.TransformerEncoder = transformer.TransformerEncoder(
        heads=4, depth=2
    )
    projector: common.Mlp = common.Mlp(hidden_sizes=[128])

    def build(self, n_outputs: int) -> nn.Module:
        dim = self.merger.n_virtual_channels
        return MyModelImpl(
            dim=dim,
            n_outputs=n_outputs,
            merger=self.merger,
            downsampler=self.downsampler,
            encoder=self.encoder,
            projector=self.projector,
        )


# %%
# Because ``MyModel`` subclasses ``BaseModelConfig``, it is automatically
# registered and can be instantiated by name -- exactly like built-in
# models. Let's build it and run a forward pass with dummy inputs:

import torch

my_config = BaseModelConfig(name="MyModel")
my_model = my_config.build(n_outputs=4)

n_chans, n_times = 204, 2700
x = torch.randn(2, n_chans, n_times)
subject_ids = torch.zeros(2, dtype=torch.long)
positions = torch.rand(2, n_chans, 2)  # 2-D channel positions in [0, 1]

with torch.no_grad():
    y = my_model(x, subject_ids, positions)

print(f"Input:      {tuple(x.shape)}")
print(f"Positions:  {tuple(positions.shape)}")
print(f"Output:     {tuple(y.shape)}")
print(f"Parameters: {sum(p.numel() for p in my_model.parameters()):,}")

# %%
# Sweeping model configs
# ----------------------
#
# Because models are typed configs, you can sweep architecture
# parameters in a grid alongside everything else:
#
# .. code-block:: python
#
#    grid = {
#        "brain_model_config.hidden": [32, 64],
#        "brain_model_config.depth": [2, 4],
#    }
#
# This is one of the main advantages of keeping the model choice in a
# serialisable configuration object.

# %%
# .. raw:: html
#
#    <div class="page-nav">
#      <a href="../data/01_data.html" class="page-nav-btn page-nav-btn--outline">&larr; Tutorial 1: Data</a>
#      <a href="../objectives/03_objectives.html" class="page-nav-btn">Tutorial 3: Objectives &rarr;</a>
#    </div>
