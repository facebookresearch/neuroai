"""
Adding a New Model
===================

Register a new brain model for evaluation on the NeuralBench benchmark.

.. important::

   Models must first be **onboarded in NeuralTrain** before they can be
   used by NeuralBench.  NeuralBench YAML configs reference model
   architectures by name; the actual ``nn.Module`` lives in the
   ``neuraltrain`` package.

This tutorial covers both steps: (1) registering the architecture in
NeuralTrain, and (2) creating the NeuralBench YAML config.
"""

# %%
# Prerequisite: the model must exist in NeuralTrain
# ---------------------------------------------------
#
# NeuralBench resolves model names via NeuralTrain's
# :class:`~neuraltrain.models.base.BaseModelConfig` registry.
# Every subclass of ``BaseModelConfig`` is automatically discoverable
# when referenced by name in a YAML config.
#
# There are two ways a model can be available:
#
# **A) Via braindecode** — Models in ``braindecode.models`` (e.g.
# ``EEGNet``, ``Deep4Net``, ``EEGConformer``, ``REVE``, ``BIOT``,
# ...) are auto-registered at import time.  Nothing to do — just
# reference them by class name in your YAML.
#
# **B) As a custom NeuralTrain module** — For architectures not in
# braindecode, you define a Pydantic config + ``nn.Module`` in
# ``neuraltrain/models/``.
#
# Here is a minimal example of adding a custom model:
#
# .. code-block:: python
#
#    # neuraltrain/models/my_model.py
#    from torch import nn
#    from neuraltrain.models.base import BaseModelConfig
#
#
#    class MyModel(BaseModelConfig):
#        """Config for MyModel.  Hyperparameters are Pydantic fields."""
#
#        hidden_size: int = 256
#        dropout: float = 0.3
#
#        def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:
#            return MyModelModule(n_in_channels, n_outputs, config=self)
#
#
#    class MyModelModule(nn.Module):
#        def __init__(self, n_in_channels, n_outputs, config):
#            super().__init__()
#            self.net = nn.Sequential(
#                nn.Flatten(1),
#                nn.LazyLinear(config.hidden_size),
#                nn.ReLU(),
#                nn.Dropout(config.dropout),
#                nn.Linear(config.hidden_size, n_outputs),
#            )
#
#        def forward(self, x):
#            # x shape: (batch, channels, time)
#            return self.net(x)
#
# Then register it by adding an import in
# ``neuraltrain/models/__init__.py``:
#
# .. code-block:: python
#
#    from . import my_model as _my_model  # noqa: F401
#
# After this, ``name: "MyModel"`` can be used in any NeuralBench
# YAML.
#
# .. note::
#
#    ``build()`` receives ``n_in_channels`` (number of EEG channels)
#    and ``n_outputs`` (task-dependent output dimension) at runtime.
#    The ``forward()`` input has shape ``(batch, channels, time)``.
#    Some models optionally accept ``channel_positions`` as well.

# %%
# Model YAML layout
# ------------------
#
# Every model file lives at
# ``neuralbench/models/{model_name}.yaml`` and is
# automatically discovered by the CLI.  The ``-m`` flag loads and
# merges the file on top of the task config:
#
# .. code-block:: bash
#
#    neuralbench eeg audiovisual_stimulus -m my_model
#
# There are two main categories of models:
#
# 1. **Task-specific models** — trained from scratch on each task.
# 2. **Foundation models** — pretrained, then fine-tuned or probed.
#
# Let's walk through concrete examples of each.

# %%
# Task-specific model (trained from scratch)
# --------------------------------------------
#
# A task-specific model config is minimal: it only needs to set the
# architecture name.
#
# .. code-block:: yaml
#
#    brain_model_config:
#      =replace=: true
#      name: "EEGNet"
#
# That's it!  The ``=replace=: true`` directive tells the config
# merger to *replace* the entire ``brain_model_config`` subtree
# rather than deep-merging with the defaults.
#
# Some task-specific models accept extra keyword arguments:
#
# .. code-block:: yaml
#
#    brain_model_config:
#      =replace=: true
#      name: "Deep4Net"
#      kwargs:
#        n_filters_time: 25
#        n_filters_spat: 25

# %%
# Foundation model (pretrained)
# ------------------------------
#
# Foundation models typically override preprocessing, specify a
# checkpoint, and configure a downstream wrapper for probing or
# fine-tuning.
#
# .. code-block:: yaml
#
#    # Preprocessing must match the pretraining setup
#    data:
#      neuro:
#        frequency: 200.0
#        filter: [0.5, 99.5]
#        notch_filter: null
#        baseline: null
#        scaler: StandardScaler
#        scale_factor: null
#        clamp: 15
#      batch_size: 32
#
#    brain_model_config:
#      =replace=: true
#      name: NtReve
#      from_pretrained_name: "brain-bzh/reve-base"
#
#    # Downstream wrapper: freeze the backbone, flatten, linear probe
#    downstream_model_wrapper:
#      model_output_key: null
#      aggregation: flatten
#      probe_config: linear
#
# Key foundation-model fields
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``data.neuro`` overrides
#   Foundation models are pretrained with specific preprocessing
#   (sampling rate, filters, scaling).  These must match.
#
# ``brain_model_config.from_pretrained_name``
#   A HuggingFace model identifier or local path.  The model
#   weights are loaded automatically by braindecode.
#
# ``downstream_model_wrapper``
#   Controls how the pretrained backbone is adapted for each task.
#   Main options:
#
#   - ``aggregation``: how to reduce the temporal/spatial dimensions
#     of the backbone output (``flatten``, ``mean``, ``first``).
#   - ``probe_config``: the classification/regression head
#     (``linear``, ``mlp``, or ``identity``).
#   - ``on_the_fly_preprocessor``: optional per-sample scaling
#     applied before the model.
#   - ``channel_merger_config``: optional learned projection from
#     variable input channels to the model's expected channels.

# %%
# Advanced: custom optimizer and scheduler
# ------------------------------------------
#
# Foundation models often benefit from different training dynamics.
# Override ``trainer_config`` and ``lightning_optimizer_config``
# directly in the model YAML:
#
# .. code-block:: yaml
#
#    data:
#      neuro:
#        frequency: 256.0
#        scaler: null
#      channel_positions:
#        n_spatial_dims: 3
#        normalize: false
#        include_ref_eeg: false
#      batch_size: 512
#
#    trainer_config:
#      n_epochs: 50
#      patience: 10
#      gradient_clip_val: 1.0
#
#    lightning_optimizer_config:
#      optimizer:
#        name: "AdamW"
#        lr: 1.0e-4
#        kwargs:
#          weight_decay: 0.05
#      scheduler:
#        =replace=: true
#        name: "OneCycleLR"
#        kwargs:
#          max_lr: 1.0e-3
#          anneal_strategy: "cos"
#          pct_start: 0.1
#
#    brain_model_config:
#      =replace=: true
#      name: "NtLuna"
#      kwargs:
#        patch_size: 40
#        num_queries: 6
#        embed_dim: 96
#        depth: 10
#        num_heads: 2
#        mlp_ratio: 4.0
#        drop_path: 0.1
#
#    pretrained_weights_fname: "./LUNA_large.safetensors"
#
#    downstream_model_wrapper:
#      on_the_fly_preprocessor:
#        scaler: "StandardScaler"
#        scale_dim: -1
#      model_output_key: null
#      aggregation: flatten
#      probe_config: linear

# %%
# Running and verifying
# ----------------------
#
# After creating your YAML, verify it works in debug mode:
#
# .. code-block:: bash
#
#    neuralbench eeg audiovisual_stimulus -m my_model --debug
#
# The model will appear in ``ALL_MODELS`` and can be combined with
# any task or grid:
#
# .. code-block:: bash
#
#    neuralbench eeg all -m my_model --grid

# %%
# Summary of the config layering
# --------------------------------
#
# .. code-block:: text
#
#    defaults/config.yaml      (base: optimizer, trainer, data, EEGNet)
#        ↓ merge
#    tasks/eeg/task/config.yaml  (task: dataset, target, loss, metrics)
#        ↓ merge
#    models/model.yaml           (model: architecture, preprocessing)
#        ↓ expand
#    grid.yaml                   (seeds, hyperparameters)
#
# Each layer can use ``=replace=: true`` to fully override a subtree
# instead of deep-merging.
#
# Checklist
# ----------
#
# .. list-table::
#    :widths: 10 90
#
#    * - ☐
#      - Ensure the model architecture exists in NeuralTrain (either
#        via braindecode or as a custom module in
#        ``neuraltrain/models/``)
#    * - ☐
#      - Create ``models/{model_name}.yaml``
#    * - ☐
#      - For foundation models: match preprocessing to pretraining
#        setup
#    * - ☐
#      - Verify with ``neuralbench eeg audiovisual_stimulus -m
#        {model_name} --debug``
#    * - ☐
#      - (Optional) Add a ``.rst`` doc page under
#        ``docs/neuralbench/models/``
