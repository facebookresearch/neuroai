"""
Track 2 -- EEG-to-BCI (cross-session command decoding)
========================================================

Given EEG recorded while a user performs one of three cued mental
tasks (kinesthetic motor imagery, mental calculation, or
letter/word association), decode the active command after the user
has already provided labelled calibration data. The competition tests
**within-user, cross-session** generalisation.

- **Shift**: early sessions -> later sessions (Graz + BrainHero
  contexts).
- **Headline metric**: balanced accuracy averaged over
  subject-session-context cells (higher is better).
- **Data**: 20 subjects, 6 sessions each, BrainAmp/actiCAP. Sessions
  1-3 of the 10 evaluation subjects are released as labelled
  calibration; sessions 4-6 are the hidden test set. The 10 training
  subjects have all 6 sessions released.

.. note::
   At the time of writing the official Track 2 dataset
   (Dreyer / Kojima / Lotte, 3 classes: MI / Calc / Word) is not
   publicly released. NeuralBench's :doc:`motor_imagery
   </neuralbench/tasks/eeg/motor_imagery>` task is the closest
   analog and is used here as the starter-kit baseline.
"""

# %%
# NeuralBench mapping (starter-kit analog)
# -----------------------------------------
#
# - **CLI**: ``neuralbench eeg motor_imagery``
# - **Default dataset**: ``Stieger2021Continuous`` (62 subjects,
#   60-channel EEG, 4-class motor imagery -- LH / RH / Both / Rest).
# - **Shift**: cross-subject (NeuralBench's default split), *not* the
#   cross-session shift of the competition. Use it to validate the
#   training pipeline and architecture choice.
# - **Headline metric key**: ``test/bal_acc``.
#
# .. dropdown:: Show ``tasks/eeg/motor_imagery/config.yaml``
#
#    .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/motor_imagery/config.yaml
#       :language: yaml

# %%
# Reproducing the baseline
# ------------------------
#
# .. code-block:: bash
#
#    # 1. Download Stieger2021Continuous
#    neuralbench eeg motor_imagery --download
#
#    # 2. Prepare the preprocessing cache
#    neuralbench eeg motor_imagery --prepare
#
#    # 3. Quick local sanity check
#    neuralbench eeg motor_imagery --debug
#
#    # 4. Full baseline -- task-specific model (EEGNet)
#    neuralbench eeg motor_imagery -m eegnet
#
# Other MI datasets registered in
# :doc:`/neuralbench/tasks/eeg/motor_imagery` (MOABB, Dreyer2023,
# BCI Competition IV, ...) can also be selected with ``--dataset
# <name>`` and are useful for stress-testing cross-subject behaviour.

# %%
# Adapting to the competition setup
# ----------------------------------
#
# To match the official Track 2 evaluation regime, two pieces need to
# change once the official dataset is released:
#
# 1. **Dataset source**: register the new MI / Calc / Word study and
#    set ``data.study.source.name`` to it, with
#    ``brain_model_output_size: 3``.
# 2. **Split**: replace the default ``SklearnSplit`` with a
#    predefined per-subject split where sessions 1-3 are train and
#    sessions 4-6 are test. The
#    ``neuralset.events.transforms.PredefinedSplit`` already used by
#    ``reaction_time`` and ``psychopathology`` is the right primitive
#    -- the ``test_split_query`` becomes
#    ``"subject in evaluation_subjects and session in [4, 5, 6]"``.
#
# Submissions may dispatch internally to per-subject sub-models using
# the per-example metadata dictionary ``m`` (subject, session, run,
# paradigm). Re-training on the hidden later sessions is forbidden.
