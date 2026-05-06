"""
Running Your First Benchmark Task
==================================

Evaluate a deep learning model on a downstream brain-modeling task with NeuralBench.
This tutorial walks through the full workflow: listing available tasks
and models, running an experiment with the CLI, and inspecting the
results programmatically.
"""

# %%
# Available tasks and models
# --------------------------
#
# NeuralBench discovers tasks by scanning the ``tasks/{device}/``
# directory and models by scanning ``models/``.  The available
# EEG tasks and models are listed on the
# :doc:`task index </neuralbench/tasks/tasks>` and
# :doc:`model index </neuralbench/models/models>` pages.
#
# You can also list them programmatically:
#
# .. code-block:: python
#
#    from neuralbench.registry import ALL_MODELS, TASKS
#
#    print("EEG tasks:", ", ".join(TASKS["eeg"]))
#    print("Available models:", ", ".join(ALL_MODELS))

# %%
# The ``neuralbench`` CLI
# -----------------------
#
# The primary interface is the ``neuralbench`` command.  Its two
# positional arguments are the **device** (``eeg``, ``meg``,
# ``fmri``, ...) and one or more **task** names (or ``all``).
#
# .. code-block:: bash
#
#    neuralbench eeg audiovisual_stimulus          # run with the default model (EEGNet)
#    neuralbench eeg audiovisual_stimulus -m reve   # override the model
#    neuralbench eeg all                            # run every EEG task
#
# Useful flags:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 20 80
#
#    * - Flag
#      - Description
#    * - ``-d, --debug``
#      - Debug mode: run locally, subsample the dataset, 2 epochs / 5 batches.
#    * - ``-m MODEL``
#      - Override the model.  Use ``all_classic`` or ``all_fm`` for predefined groups.
#    * - ``-g, --grid``
#      - Expand the task-specific hyperparameter grid.
#    * - ``--download``
#      - Download the dataset without running experiments.
#    * - ``-p, --prepare``
#      - Run a single experiment to warm the preprocessing cache.
#    * - ``--plot-cached``
#      - Generate comparison plots and tables from cached results (no
#        retraining).
#
# Typical three-step workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. code-block:: bash
#
#    # 1. Download the data
#    neuralbench eeg audiovisual_stimulus --download
#
#    # 2. (Optional) Warm the preprocessing cache
#    neuralbench eeg audiovisual_stimulus --prepare
#
#    # 3. Run the benchmark
#    neuralbench eeg audiovisual_stimulus
#
# The ``audiovisual_stimulus`` task uses the single-subject MNE-Python
# sample dataset (~1.5 GB), which is downloaded automatically. We use it
# both as a quick sanity-check task and as a probe of model behaviour in
# very-low-data regimes (288 trials, 4-class classification).
#
# Debug mode
# ~~~~~~~~~~
#
# During development, ``--debug`` can be used to run a quick check that a task runs correctly.
# It runs locally (no SLURM), subsamples the dataset, and trains for only
# 2 epochs with 5 batches each:
#
# .. code-block:: bash
#
#    neuralbench eeg audiovisual_stimulus --debug
#

# %%
# What gets rerun (and what doesn't)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# NeuralBench uses deterministic caching so that identical experiments
# are never run twice.  Here is how it works:
#
# - Every experiment configuration is hashed into a unique **UID**.
#   The output folder is named after this UID.
# - By default (``mode="retry"``), if results for that UID already
#   exist on disk, the run is **skipped** and the cached results are
#   returned instantly.
# - **Changing any config parameter** — model, seed, preprocessing
#   option, loss, etc. — produces a different UID, so it will run
#   from scratch.
# - The ``--force`` flag (or ``-f``) overrides this behaviour and
#   **re-executes** every experiment regardless of cache.
#
# Data preparation (``--prepare``) also benefits from caching:
# preprocessed data and targets are stored on disk and reused in
# subsequent runs.
#
# Finally, ``--plot-cached`` lets you visualize only the results that
# have already completed, without blocking on pending experiments.
#

# %%
# How it works under the hood
# ----------------------------
#
# The CLI builds one :class:`~neuralbench.main.Experiment` per
# grid point.  Each ``Experiment`` is a Pydantic model created by
# layering YAML configs:
#
# 1. ``defaults/config.yaml`` — global defaults (optimizer, trainer,
#    data loading)
# 2. ``tasks/{device}/{task}/config.yaml`` — task-specific overrides
#    (dataset, target, loss, metrics)
# 3. ``models/{model}.yaml`` — model-specific overrides
#    (architecture, preprocessing, probing strategy)
# 4. ``tasks/{device}/{task}/grid.yaml`` — hyperparameter grid
#
# Here is the config for the ``audiovisual_stimulus`` task
# (``tasks/eeg/audiovisual_stimulus/config.yaml``):
#
# .. code-block:: yaml
#
#    data:
#      study:
#        source:
#          name: Mne2013SampleEeg
#        split:
#          name: SklearnSplit
#          split_by: _index
#          valid_split_ratio: 0.2
#          test_split_ratio: 0.2
#          valid_random_state: 33
#          test_random_state: 33
#          stratify_by: description
#      target:
#        =replace=: true
#        name: LabelEncoder
#        event_types: Stimulus
#        event_field: description
#        return_one_hot: true
#        aggregation: first
#      neuro.baseline: [0.0, 0.2]
#      trigger_event_type: Stimulus
#      start: -0.2
#      duration: 1.0
#
#    brain_model_output_size: &brain_model_output_size 4
#    loss:
#      name: CrossEntropyLoss
#      kwargs:
#        label_smoothing: 0.1
#    metrics: !!python/object/apply:neuralbench.defaults.metrics.get_classification_metric_configs
#      - *brain_model_output_size
#
# The ``Experiment.run()`` method returns a dictionary of test
# metrics — for example ``{"test/bal_acc": 0.72, "test/loss": 0.31,
# ...}``.  This is also what ``BenchmarkAggregator`` collects when
# producing plots and tables (see the
# :doc:`visualizing results tutorial
# </neuralbench/auto_examples/results/plot_visualize_results>`).
#
# Switching models
# ~~~~~~~~~~~~~~~~
#
# Two convenient model groups are available via the CLI:
#
# - ``all_classic`` — 8 task-specific architectures trained from scratch
# - ``all_fm`` — 6 pretrained foundation models
#
# For example:
#
# .. code-block:: bash
#
#    neuralbench eeg audiovisual_stimulus -m eegconformer
#    neuralbench eeg audiovisual_stimulus -m all_classic
#    neuralbench eeg audiovisual_stimulus -m reve labram
#
# Here is a minimal task-specific model config (``models/eegnet.yaml``):
#
# .. code-block:: yaml
#
#    brain_model_config:
#      =replace=: true
#      name: EEGNet
#
# Hyperparameter grids
# ~~~~~~~~~~~~~~~~~~~~~
#
# Pass ``-g`` to expand the task-specific grid (from the task's
# ``grid.yaml``).  The default grid sweeps over random seeds only:
#
# .. code-block:: yaml
#
#    # defaults/grid.yaml
#    seed:
#      - 33
#      - 34
#      - 35

# %%
# Next steps
# ----------
#
# - :doc:`Visualize results
#   </neuralbench/auto_examples/results/plot_visualize_results>` from
#   completed experiments.
# - :doc:`Add a new task
#   </neuralbench/auto_examples/adding_task/create_new_task>` to the
#   benchmark.
# - :doc:`Add a new model
#   </neuralbench/auto_examples/adding_model/create_new_model>` for
#   evaluation.
