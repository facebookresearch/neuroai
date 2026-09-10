"""
Track 3 -- Sleep onset (cross-device latency prediction)
=========================================================

Given a continuous wearable EEG recording, predict the latency
(in seconds, from recording start) to the first N2 epoch. The
competition tests **cross-device** generalisation: the seed corpora are
clinical polysomnography, while the evaluation set is consumer-grade
home-wearable EEG with its own channels and montage. Precise timing
replaces full staging because the wearable signal is too sparse for
per-epoch hypnogram reconstruction.

- **Shift**: clinical polysomnography -> home wearable EEG, on unseen
  sleepers.
- **Headline metric**: ``bMAE`` in seconds -- onset error averaged with
  equal weight over four time-to-onset bins, so long and short sleep
  onsets count the same (lower is better). The competition additionally
  reports tolerance rates within 30 / 60 / 300 s; the starter kit logs
  ``bMAE`` and the usual regression metrics, not those rates.
- **Data**: continuous Muse wearable EEG, ~1000 training subjects,
  hidden evaluation set of the same order of magnitude. The onset that
  ``AddSleepOnsetTargets`` extracts is the earliest annotated N2 event
  of the recording; it applies no persistence or non-Wake-continuity
  rule, so check the competition's definition before reshaping the
  target to match it.

.. note::
   The Muse training set is released through NeuralBench when
   submissions open. Until then, this starter kit runs on the Sleep-EDF
   dataset (``Kemp2000Analysis``) -- the data format and target
   extractor are identical, only the recording hardware differs.
"""

# %%
# NeuralBench mapping
# -------------------
#
# The matching task in NeuralBench is
# :doc:`/neuralbench/tasks/eeg/sleep_onset`.
#
# - **CLI**: ``neuralbench eeg sleep_onset``
# - **Default dataset**: ``Kemp2000Analysis`` (Sleep-EDF Expanded,
#   78 participants recorded over up to two nights each, 2 EEG
#   channels, full polysomnography).
# - **Target**: the time *remaining* until the first N2 epoch, which
#   ``AddSleepOnsetTargets`` + ``SleepOnsetTargetExtractor`` recompute for
#   every window as ``clip(n2_onset - window_stop, 0, 600)`` seconds. The
#   task therefore predicts once per 5-second window rather than once per
#   recording, and the competition's single ``tau_hat`` is
#   ``window_stop + prediction`` read off any window within 600 s of
#   onset, where the cap has not saturated the target.
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
#
#    # 5. Full baseline -- foundation model (REVE)
#    neuralbench eeg sleep_onset -m reve

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
# The current NeuralBench head produces the second format directly, and
# the first by the conversion above.
