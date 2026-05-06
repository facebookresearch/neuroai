"""
Reproducing tasks from the EEG Foundation Challenge 2025
========================================================

The `EEG Foundation Challenge 2025 <https://eeg2025.github.io/>`_ was a
`NeurIPS 2025 Competition Track challenge focused on advancing cross-task and cross-subject EEG decoding.
The competition used the
`HBN-EEG dataset <https://openneuro.org/datasets/ds005505>`_
(Shirazi et al. 2024), which includes recordings from over 3,000
participants across six cognitive tasks.

The two challenge tasks were:

1. **Challenge 1 -- Cross-Task Transfer Learning (Reaction Time)**:
   predict behavioural reaction time from EEG recorded during the
   Contrast Change Detection (CCD) active task.
2. **Challenge 2 -- Externalizing Factor Prediction**:
   predict continuous externalizing psychopathology scores from EEG
   recorded across experimental paradigms.

Both tasks are scored with **nRMSE** (normalized Root Mean Squared
Error).

NeuralBench ships two tasks that closely mirror these challenges:
``reaction_time`` and ``psychopathology``.  This tutorial shows how
to run them and highlights the differences with the actual competition
setup.
"""

# %%
# Challenge 1: Reaction Time Prediction
# ---------------------------------------
#
# The ``reaction_time`` task predicts per-trial reaction time from
# the CCD paradigm of the HBN dataset.
#
# Key design choices in ``tasks/eeg/reaction_time/config.yaml``:
#
# - **Dataset**: ``Shirazi2024Hbn``, filtered to the CCD task
#   (``task=='task-contrastChangeDetection'``).
# - **Target**: the ``reaction_time`` field on ``Keystroke`` events,
#   excluding trials where the reaction time was shorter than 500 ms.
# - **Epoch window**: 0.5 s to 2.5 s relative to the keystroke
#   (``start: 0.5``, ``duration: 2.0``).
# - **Split**: release-based with R5 held out as the test set
#   (``PredefinedSplit``).
# - **Loss / monitor**: MSE loss, early stopping on
#   ``val/pearsonr``.
#
# Here is the full task config:
#
# .. code-block:: yaml
#
#    data:
#      study:
#        source:
#          name: Shirazi2024Hbn
#        filter_task:
#          name: QueryEvents
#          query: "task=='task-contrastChangeDetection'"
#        filter_reaction_time:
#          name: QueryEvents
#          query: "(reaction_time >= 0.5) | type == 'Eeg'"
#        split:
#          name: PredefinedSplit
#          test_split_query: "release in ['R5']"
#          col_name: split
#          valid_split_by: release
#          valid_split_ratio: 0.091
#          valid_random_state: 33
#      target:
#        =replace=: true
#        name: EventField
#        event_types: Keystroke
#        event_field: reaction_time
#        aggregation: trigger
#      trigger_event_type: Keystroke
#      start: 0.5
#      duration: 2.0
#      summary_columns: [release]
#    brain_model_output_size: &brain_model_output_size 1
#    trainer_config.monitor: val/pearsonr
#    loss:
#      name: MSELoss
#    metrics: !!python/object/apply:neuralbench.defaults.metrics.get_regression_metric_configs
#      - *brain_model_output_size
#
# Running it from the CLI:
#
# .. code-block:: bash
#
#    # Download the data
#    neuralbench eeg reaction_time --download
#
#    # Prepare the data
#    neuralbench eeg reaction --prepare
#
#    # Quick debug run
#    neuralbench eeg reaction_time --debug
#
#    # Full run with the default model (EEGNet)
#    neuralbench eeg reaction_time
#
#    # Try a different model
#    neuralbench eeg reaction_time -m simpleconv_time_agg

# %%
# Challenge 2: Externalizing Factor Prediction
# ----------------------------------------------
#
# The ``psychopathology`` task predicts the continuous externalizing
# score (derived from the CBCL questionnaire) from resting-state EEG.
#
# Key design choices in ``tasks/eeg/psychopathology/config.yaml``:
#
# - **Dataset**: ``Shirazi2024Hbn``, filtered to resting-state
#   recordings longer than 180 s that have a non-null externalizing
#   score.
# - **Cropping**: the first 60 s are discarded (habituation) and the
#   next 120 s are kept (``CropTimelines``) to limit the dataset size.
# - **Target**: the ``externalizing`` CBCL dimension.
# - **Epoch window**: non-overlapping 2 s windows
#   (``start: 0.0``, ``duration: 2.0``, ``stride: 2.0``).
# - **Split**: same release-based scheme (R5 = test).
# - **Trainer**: 40 epochs, patience 7.
#
# Here is the full task config:
#
# .. code-block:: yaml
#
#    data:
#      study:
#        source:
#          name: Shirazi2024Hbn
#        filter_resting_state_with_externalizing:
#          name: QueryEvents
#          query: >-
#            (type == 'Eeg') & (task == 'task-RestingState')
#            & (duration > 180.0) & externalizing.notnull()
#        crop_timelines:
#          name: CropTimelines
#          event_type: Eeg
#          start_offset_s: 60.0
#          max_duration_s: 120.0
#        split:
#          name: PredefinedSplit
#          test_split_query: "release in ['R5']"
#          col_name: split
#          valid_split_by: release
#          valid_split_ratio: 0.091
#          valid_random_state: 33
#      target:
#        =replace=: true
#        name: EventField
#        event_types: Eeg
#        event_field: externalizing
#        aggregation: single
#      trigger_event_type: Eeg
#      start: 0.0
#      duration: 2.0
#      stride: 2.0
#      summary_columns: [release, externalizing]
#    brain_model_output_size: &brain_model_output_size 1
#    trainer_config:
#      monitor: val/pearsonr
#      strategy: auto
#      patience: 7
#      n_epochs: 40
#    loss:
#      name: MSELoss
#    metrics: !!python/object/apply:neuralbench.defaults.metrics.get_regression_metric_configs
#      - *brain_model_output_size
#
# Running it from the CLI:
#
# .. code-block:: bash
#
#    neuralbench eeg psychopathology --download  # NOTE: Same as for the reaction time task
#    neuralbench eeg psychopathology --prepare
#    neuralbench eeg psychopathology --debug
#    neuralbench eeg psychopathology

# %%
# Competition metric: nRMSE
# --------------------------
#
# The competition leaderboard ranked submissions by **nRMSE**
# (normalized RMSE), defined as:
#
# .. math::
#
#    \text{nRMSE} = \frac{\text{RMSE}}{\text{std}(y)}
#
# where :math:`y` are the ground-truth target values.  A score of
# 1.0 corresponds to a model that predicts the mean of the target
# distribution (i.e., no better than a constant baseline).  Lower
# is better.
#
# Both ``reaction_time`` and ``psychopathology`` already compute this
# metric via ``get_regression_metric_configs``, which includes a
# ``NormalizedRMSE`` entry logged as ``normalized_rmse``.  After a
# run completes, the nRMSE appears in the test metrics as
# ``test/normalized_rmse``.
#
# .. note::
#
#    NeuralBench uses ``val/pearsonr`` for early stopping (model
#    checkpoint selection), while the competition ranked by nRMSE.
#    The ``normalized_rmse`` metric is still logged and available for
#    post-hoc comparison.

# %%
# Brain & AI team solution
# -------------------------
#
# This section summarises the approach used by the Brain & AI team and maps each component to its NeuralBench equivalent.
#
# Elements already covered by NeuralBench
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The following components are directly available:
#
# - **HBN dataset** via the ``Shirazi2024Hbn`` study
# - **CCD task filtering** and reaction time target extraction
# - **Externalizing score regression** from resting-state EEG
# - **nRMSE metric** (``normalized_rmse`` in logged metrics)
# - **Release-based train / validation / test splits**
#
# Available components that require configuration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Components that exist in NeuralTrain but require further configuration:
#
# - **SimpleConv backbone** (Défossez et al., 2023): a dilated
#   residual convolutional architecture. The `simpleconv_time_agg`
#   model is already available in NeuralTrain and can be further modified.
# - **OnTheFlyPreprocessor**: a learnable preprocessor module that
#   applies peak-to-peak thresholding (to zero out bad channels)
#   and channel-wise standard scaling directly on raw EEG input
#   inside the model.  It is available in NeuralTrain and can be
#   activated through the model config.
#
# Going further
# ~~~~~~~~~~~~~~
#
# Additional design choices go beyond what NeuralBench provides out of the box:
#
# - **Probabilistic RT formulation**: instead of predicting a single
#   scalar with MSE loss, the backbone outputs a probability
#   distribution over *T* input timesteps.  This distribution is
#   aligned to a target distribution centred on the reaction time
#   using a KL divergence loss.  At inference the expected value of
#   the distribution yields the final prediction.
# - **Auxiliary multi-task heads**: separate projection heads
#   simultaneously predict stimulation side, subject response side,
#   and answer correctness alongside the main RT target.
# - **Ensemble of ~15 models**: models with varied hyperparameters
#   (decoder depth, loss term weights, preprocessing steps, data
#   splits) are combined via weighted averaging of their predictions.
#
# These techniques can be implemented by subclassing
# :class:`~neuralbench.pl_module.BrainModule` (see the
# :doc:`advanced tutorial
# </neuralbench/auto_examples/advanced/modify_training_loop>`)
# or by writing a custom training script on top of NeuralTrain.
