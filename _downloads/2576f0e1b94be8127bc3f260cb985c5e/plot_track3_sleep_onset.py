"""
Track 3 -- Sleep onset (cross-subject latency prediction)
===========================================================

Given a continuous wearable EEG recording, predict the latency
(in seconds, from recording start) at which the participant
transitions into stable sleep. The competition tests **cross-subject**
generalisation: train sleepers and test sleepers are disjoint.

- **Shift**: seen sleepers -> unseen sleepers.
- **Headline metric**: recording-level mean absolute error in seconds
  (lower is better). Tolerance rates within 30 / 60 / 300 s are
  reported as diagnostics.
- **Data**: continuous Muse wearable EEG, ~1000 training subjects,
  hidden evaluation set of the same order of magnitude. The reference
  onset is the first annotated N2 event (or equivalently the first
  non-Wake epoch satisfying a fixed persistence rule).

.. note::
   The Muse training set will be released by InteraXon for the
   competition. Until then, this starter kit runs on the Sleep-EDF
   dataset (``Kemp2000Analysis``) -- the data format and target
   extractor are identical, only the recording hardware differs.
"""

# %%
# NeuralBench mapping
# -------------------
#
# - **CLI**: ``neuralbench eeg sleep_onset``
# - **Default dataset**: ``Kemp2000Analysis`` (Sleep-EDF Expanded,
#   78 nights, 2 EEG channels, full polysomnography).
# - **Target**: latency from recording start to the first N2 epoch,
#   extracted by ``AddSleepOnsetTargets`` + ``SleepOnsetTargetExtractor``
#   and capped at 600 s. This matches the competition's reference
#   definition.
# - **Headline metric key**: ``test/bmae`` (binned MAE in seconds).
#
# .. dropdown:: Show ``tasks/eeg/sleep_onset/config.yaml``
#
#    .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/sleep_onset/config.yaml
#       :language: yaml

# %%
# Reproducing the baseline
# ------------------------
#
# .. code-block:: bash
#
#    # 1. Download Sleep-EDF
#    neuralbench eeg sleep_onset --download
#
#    # 2. Prepare the preprocessing cache
#    neuralbench eeg sleep_onset --prepare
#
#    # 3. Quick local sanity check
#    neuralbench eeg sleep_onset --debug
#
#    # 4. Full baseline -- task-specific model (EEGNet)
#    neuralbench eeg sleep_onset -m eegnet

# %%
# Where the competition data diverges
# ------------------------------------
#
# Sleep-EDF and the Muse competition data differ on three axes that
# matter at training time:
#
# 1. **Hardware**: research-grade PSG (Sleep-EDF) vs consumer-grade
#    Muse headband (4-channel frontal EEG, +/- accelerometer, no EOG).
#    Expect to drop or re-map channels in the dataloader.
# 2. **Cohort and recording context**: laboratory monitored sleep vs
#    home recordings with movement artifacts and impedance changes.
# 3. **Annotations**: full hypnograms vs ``n2_onset`` events only on
#    the training set. NeuralBench already trains on
#    ``SleepOnsetMarker`` events, so the model interface does not
#    change.
#
# Two additional polysomnography datasets are already registered under
# ``tasks/eeg/sleep_onset/datasets/`` and use the same
# ``AddSleepOnsetTargets`` + ``bmae`` pipeline as the default. They are
# useful for stress-testing cross-subject behaviour on more sleepers:
#
# .. code-block:: bash
#
#    neuralbench eeg sleep_onset --dataset ghassemi2018you
#    neuralbench eeg sleep_onset --dataset alvarez2022haaglanden
#
# Once the Muse study is registered, switching is a single
# ``data.study.source.name: Interaxon2026Muse`` override (or
# ``--dataset interaxon2026muse`` if a ``datasets/`` YAML ships).
#
# Submission outputs (per the competition):
#
# - a direct onset estimate ``tau_hat`` in seconds, **or**
# - per-window time-to-onset predictions, **or**
# - per-window sleep probabilities.
#
# The current NeuralBench head produces the first format directly.
