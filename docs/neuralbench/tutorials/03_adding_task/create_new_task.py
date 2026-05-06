"""
Adding a New Downstream Task
==============================

Add a brain-modeling task to NeuralBench.  Every task lives in its own
directory under ``tasks/{device}/{task_name}/`` and is defined by a
single ``config.yaml`` that overrides the global defaults.
"""

# %%
# Task directory layout
# ---------------------
#
# Each task needs at minimum a ``config.yaml``.  An optional
# ``grid.yaml`` adds task-specific hyperparameter sweeps.
#
# .. code-block:: text
#
#    neuralbench/tasks/eeg/
#        my_new_task/
#            config.yaml          # required
#            grid.yaml            # optional (hyperparameter grid)
#            datasets/            # optional (per-dataset overrides)
#                variant_a.yaml
#
# Once the directory exists the CLI picks it up automatically::
#
#    neuralbench eeg my_new_task --debug
#
# Directories prefixed with ``_`` (e.g. ``_my_new_task``) are treated
# as **unvalidated** and skipped when running ``neuralbench eeg all``.

# %%
# Anatomy of a task config
# -------------------------
#
# A ``config.yaml`` overrides parts of the base config
# (``defaults/config.yaml``).  The main sections are:
#
# 1. **Data** — which dataset to load, how to split, and what to
#    predict.
# 2. **Model output & loss** — number of output neurons and loss
#    function.
# 3. **Metrics** — evaluation metrics.
#
# Here is a complete example for a **binary classification** task.
#
# .. code-block:: yaml
#
#    data:
#      study:
#        study:
#          name: Mumtaz2018Machine        # Study class in neuralfetch.studies
#        transforms:
#          split:
#            name: SklearnSplit
#            split_by: subject             # Leave-subjects-out split
#            valid_split_ratio: 0.2
#            test_split_ratio: 0.2
#            valid_random_state: 33
#            test_random_state: 33
#            stratify_by: diagnosis        # Stratify on the target label
#
#      target:
#        =replace=: true                   # Replace the default target config entirely
#        name: LabelEncoder
#        event_types: Eeg
#        event_field: diagnosis            # Column in the events DataFrame
#        return_one_hot: true
#        aggregation: first
#
#      trigger_event_type: Eeg             # Event type that triggers a segment
#      start: 0.0                          # Segment start (seconds relative to trigger)
#      duration: 5.0                       # Segment duration (seconds)
#      stride: 5.0                         # Stride for continuous windowing
#
#    model_output_size: &model_output_size 2
#
#    loss:
#      name: CrossEntropyLoss
#      kwargs:
#        label_smoothing: 0.1
#
#    metrics: !!python/object/apply:neuralbench.defaults.metrics.get_classification_metric_configs
#      - *model_output_size

# %%
# Key fields explained
# ~~~~~~~~~~~~~~~~~~~~~
#
# ``data.study.study.name``
#   The name of the Study class (from NeuralFetch) that loads the
#   dataset.
#
# ``data.study.transforms.split``
#   How to split the data.  Common splitters:
#
#   - ``SklearnSplit`` — random split, optionally stratified, by
#     subject, index, or timeline.
#   - ``PredefinedSplit`` — split on a column value (e.g.
#     ``release == 'R5'`` for a held-out set).
#
# ``data.target``
#   What the model predicts.  Common target extractors:
#
#   - ``LabelEncoder`` — classification labels from an event field.
#   - ``EventField`` — raw numeric value (for regression).
#   - ``HuggingFaceText`` / ``Wav2Vec`` / ``HuggingFaceImage`` —
#     embedding-based targets (for retrieval tasks).
#
# ``trigger_event_type``
#   Which event type marks the start of each trial/segment.
#
# ``start`` / ``duration`` / ``stride``
#   Windowing parameters.  ``stride`` enables overlapping or
#   contiguous windows; omit it for event-locked segments.
#
# ``=replace=: true``
#   Tells the config merger to *replace* the subtree entirely
#   instead of deep-merging with the defaults.
#
# ``!!python/object/apply:...``
#   Calls a Python factory function at config load time.  The
#   metric factories in ``neuralbench.defaults.metrics`` return
#   standard metric configs for classification, regression, or
#   retrieval.

# %%
# Adding a hyperparameter grid
# ------------------------------
#
# Create a ``grid.yaml`` alongside the ``config.yaml``.  Keys use
# dot-notation and map to lists of values:
#
# .. code-block:: yaml
#
#    data.batch_size: [16, 32]
#    lightning_optimizer_config.optimizer.lr: [1.0e-3, 5.0e-4, 1.0e-4]
#
# The grid is expanded with ``--grid``:
#
# .. code-block:: bash
#
#    neuralbench eeg my_new_task --grid

# %%
# Adding a documentation page
# -----------------------------
#
# Each task should have a matching ``.rst`` page under
# ``docs/neuralbench/tasks/eeg/``.  A template is provided at
# ``docs/neuralbench/tasks/_TEMPLATE.rst``.  Copy it to
# ``docs/neuralbench/tasks/eeg/{task_name}.rst`` and fill in
# the task-specific details.

# %%
# Checklist
# ---------
#
# .. list-table::
#    :widths: 10 90
#
#    * - ☐
#      - Create ``tasks/{device}/{task_name}/config.yaml``
#    * - ☐
#      - (Optional) Create ``grid.yaml`` with task-specific sweeps
#    * - ☐
#      - Add a debug query in ``defaults/debug_study.yaml`` for fast
#        iteration
#    * - ☐
#      - Verify with ``neuralbench {device} {task_name} --debug``
#    * - ☐
#      - Add a ``.rst`` doc page under ``docs/neuralbench/tasks/``
