"""
Track 5 -- Foundation Transfer (one EEG encoder, three tasks)
==============================================================

Track 5 evaluates the central foundation-model claim: does a *single*
EEG encoder transfer across heterogeneous EEG tasks without
task-specific re-training? Participants submit one encoder
:math:`g_{\\phi}`; the organisers freeze it and **linear-probe** it
for each of the three EEG tracks (Image, BCI, Sleep), then auto-enrol
the submission on those leaderboards under the foundation-model
label.

- **Shift**: cross-task, cross-device, cross-montage.
- **Headline metric**: mean rank of the submission across the three
  EEG leaderboards (lower is better),
  :math:`S_{\\mathrm{FM}}(a) = \\frac{1}{3} \\sum_{t} r_{a,t}`.
- **Rule**: the encoder must be a single set of weights. It may read
  the per-example metadata :math:`m` (subject, session, track,
  channel layout) and may be channel-aware, but bundling separately
  trained sub-encoders is not allowed.
"""

# %%
# How Track 5 is scored
# ---------------------
#
# Submitting an encoder to Track 5 *automatically* produces three
# leaderboard entries. For each EEG track, the organisers freeze the
# encoder weights and fit a **linear probe** on top of the encoder
# features:
#
# 1. Track 1 (EEG-to-Image) -- linear probe on
#    :math:`g_{\phi}(X, m)`, plugged into the same Top-5 retrieval
#    scorer as Track 1.
# 2. Track 2 (EEG-to-BCI) -- linear probe on the same encoder
#    features, scored with balanced accuracy.
# 3. Track 3 (Sleep onset) -- linear probe on the same encoder
#    features, scored with onset MAE.
#
# The submission's Track 5 score is the mean of its ranks on those
# three leaderboards. Ties are broken by jointly bootstrapping the
# three leaderboards (see the competition proposal for details).
#
# Practically: a strong Track 5 submission needs to be competitive on
# **all three** EEG tracks at the same time, with frozen encoder
# weights and a single channel-handling mechanism. No track-specific
# fine-tuning of the encoder is allowed.
#
# .. note::
#    The official Track 5 protocol freezes the encoder and fits a
#    **linear probe** per track. NeuralBench currently runs foundation
#    models with **full fine-tuning** (the encoder is trained jointly
#    with the head), so the baselines below approximate Track 5 rather
#    than reproduce its exact frozen-encoder protocol. Frozen-encoder
#    linear probing is coming soon; until it lands, treat these numbers
#    as an upper-bound proxy for the transfer signal.

# %%
# Reproducing the reference baseline (REVE)
# ------------------------------------------
#
# REVE is the reference Track 5 baseline: a single set of weights
# evaluated across Tracks 1-3. To reproduce its three leaderboard
# entries with NeuralBench:
#
# .. code-block:: bash
#
#    # Image: Top-5 accuracy
#    neuralbench eeg image -m reve
#
#    # BCI: balanced accuracy (starter-kit dataset, see Track 2)
#    neuralbench eeg motor_imagery -m reve
#
#    # Sleep onset: recording-level MAE
#    neuralbench eeg sleep_onset -m reve
#
# Each command produces a ``test/<metric>`` value (``batch_top5_acc``,
# ``bal_acc``, ``bmae``). REVE's three values are reported in
# :doc:`plot_overview`.
#
# To compare against task-specific models on the same three tracks:
#
# .. code-block:: bash
#
#    neuralbench eeg image           -m all_classic all_fm
#    neuralbench eeg motor_imagery   -m all_classic all_fm
#    neuralbench eeg sleep_onset     -m all_classic all_fm
#
# Then use the
# :doc:`results tutorial </neuralbench/auto_examples/results/plot_visualize_results>`
# (or the ``--plot-cached`` workflow described in
# :doc:`plot_overview`) to aggregate ranks across the three tracks.

# %%
# Submitting your own foundation model
# -------------------------------------
#
# Track 5 submissions are evaluated by the organisers; the encoder
# itself runs locally during preparation and is then packaged for
# submission. From NeuralBench's perspective, the encoder is a
# foundation model registered under ``models/<your_model>.yaml`` with
# a ``downstream_model_wrapper`` (the same wiring used by
# ``reve.yaml``, ``labram.yaml``, ``bendr.yaml``, ``biot.yaml``,
# ``cbramod.yaml``, ``luna.yaml``). With NeuralBench's current support,
# the encoder is fine-tuned end-to-end with the downstream head;
# frozen-encoder linear probing (the official Track 5 protocol) is
# coming soon.
#
# See :doc:`/neuralbench/auto_examples/adding_model/create_new_model`
# for the model-registration walkthrough. The same encoder config can
# then be passed to all three EEG tracks via ``-m <your_model>``.

# %%
# Rules cheat-sheet (from the competition proposal)
# --------------------------------------------------
#
# - One set of weights. No separately trained sub-encoders, no
#   metadata-conditional routing among separately trained components.
# - The encoder is frozen at submission time; only the organisers'
#   linear probes adapt to each track.
# - Channel handling (e.g., padding to a canonical montage, learned
#   channel embeddings from electrode positions) is a single shared
#   component.
# - Pre-training data is unrestricted but must be disclosed in the
#   methods report.
# - A Track 5 team may **not** also submit specialised models to
#   Tracks 1-3 in their own name; the organisers handle those entries
#   on the Track 5 encoder's behalf.
# - Track 4 (EMG) is excluded from Track 5 -- EMG and EEG differ in
#   sampling rate, substrate, and montage.
