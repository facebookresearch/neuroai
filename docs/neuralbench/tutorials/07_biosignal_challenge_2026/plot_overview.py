"""
Overview: EEG/EMG Foundation Challenge 2026
============================================

The EEG/EMG Foundation Challenge 2026 is a proposed multi-track
competition on shift-robust representation learning for biosignals,
and the multi-modal successor to the 2025 EEG Foundation Challenge.

.. note::

   The competition is currently at the proposal stage and has not
   yet been accepted. This starter kit is shared so the community
   can experiment with the planned tracks while the proposal is
   under review.

The competition is organised as **4 + 1 tracks**:

1. **Track 1 -- EEG-to-Image** (cross-stimulus): retrieve a viewed image
   from EEG. Headline metric: **Top-5 retrieval accuracy** (higher is
   better).
2. **Track 2 -- EEG-to-BCI** (cross-session, within-user): decode one of
   three cued mental tasks (motor imagery, mental calculation, word
   association). Headline metric: **balanced accuracy** averaged over
   subject-session-context cells (higher is better).
3. **Track 3 -- Sleep onset** (cross-subject): estimate sleep-onset
   latency from wearable EEG. Headline metric: **recording-level mean
   absolute error in seconds** (lower is better).
4. **Track 4 -- EMG-to-Text** (cross-subject): decode keystroke
   sequences from surface EMG. Headline metric: **character error
   rate** (lower is better).
5. **Track 5 -- Foundation Transfer** (cross-task, EEG only): submit a
   *single* EEG encoder; organisers fit linear probes for Tracks 1-3.
   Headline metric: **mean rank** across the three EEG leaderboards
   (lower is better).

Tracks 1-4 accept both task-specific models and foundation models.
Track 5 is restricted to a single shared encoder.

This starter kit shows how to reproduce a baseline for each track with
NeuralBench, using the closest publicly available dataset when the
official competition data is not yet released.
"""

# %%
# Starter kit pages
# -----------------
#
# - :doc:`Track 1 -- EEG-to-Image <plot_track1_eeg_to_image>`
# - :doc:`Track 2 -- EEG-to-BCI <plot_track2_eeg_to_bci>`
# - :doc:`Track 3 -- Sleep onset <plot_track3_sleep_onset>`
# - :doc:`Track 4 -- EMG-to-Text <plot_track4_emg_to_text>`
# - :doc:`Track 5 -- Foundation Transfer <plot_track5_foundation_transfer>`
# - :doc:`How to Submit a Model <plot_submission_guide>`
#
# Each page follows the same shape:
#
# 1. What the track measures (data, shift, headline metric).
# 2. The matching ``neuralbench`` task and CLI commands.
# 3. Where the official competition data diverges from the default.

# %%
# Representative results
# ----------------------
#
# The table below summarises NeuralBench results on **publicly
# available datasets that are similar in paradigm and modality** to
# the ones the competition will use. These numbers are *not* the
# competition leaderboard -- the official tracks will use distinct
# datasets, hidden evaluation sets, and rerun protocols. They are
# useful as a sanity check that your training pipeline behaves like
# the published baselines on the closest open data.
#
# .. list-table::
#    :header-rows: 1
#    :widths: 22 18 18 18 18
#
#    * - Model
#      - Image (Top-5 %, higher)
#      - BCI (Bal. acc %, higher)
#      - Sleep (Onset MAE s, lower)
#      - EMG (CER %, lower)
#    * - Chance
#      - 2.22 +/- 0.31
#      - 24.81 +/- 1.03
#      - 205.42 +/- 0.01
#      - 96.71 +/- 0.00
#    * - Dummy
#      - 2.50 +/- 0.00
#      - 25.00 +/- 0.00
#      - 299.90 +/- 0.00
#      - 100.00 +/- 0.00
#    * - EEGNet
#      - 28.13 +/- 0.14
#      - 58.58 +/- 0.34
#      - 143.30 +/- 0.40
#      - --
#    * - REVE (FM)
#      - 84.75 +/- 0.38
#      - 68.04 +/- 0.73
#      - 134.89 +/- 2.02
#      - --
#    * - EMG2QwertyNet
#      - --
#      - --
#      - --
#      - 25.14 +/- 2.30
#
# REVE is also the reference **Track 5** baseline: its three EEG-track
# numbers come from a single set of encoder weights, so its mean rank
# on the Tracks 1-3 leaderboards is also its Track 5 score.

# %%
# Collecting and plotting your results
# -------------------------------------
#
# Every NeuralBench run caches its test-metric dictionary on disk.
# After the experiments you care about have finished, re-invoke the
# same CLI command with ``--plot-cached`` to aggregate results into
# comparison plots and CSV tables without retraining:
#
# .. code-block:: bash
#
#    # 1. Run experiments (cached automatically)
#    neuralbench eeg image motor_imagery sleep_onset -m eegnet reve
#
#    # 2. Aggregate cached results -- no retraining
#    neuralbench eeg image motor_imagery sleep_onset -m eegnet reve --plot-cached
#
# ``--plot-cached`` produces, under ``<SAVE_DIR>/outputs/``:
#
# - ``core/core_bar_chart.png`` -- bar chart per task and model.
# - ``core/core_results_table.csv`` -- raw per-task metrics.
# - ``core/core_rank_table.csv`` -- ranks per task (the input to the
#   Track 5 mean-rank score).
#
# For programmatic access to the same data, instantiate
# :class:`~neuralbench.main.BenchmarkAggregator` directly. The
# :doc:`/neuralbench/auto_examples/results/plot_visualize_results`
# tutorial walks through the full Python API, including how to
# customise the loss-to-metric mapping and the output directory.

# %%
# Resources and dependencies
# ---------------------------
#
# This starter kit relies on the following organiser-maintained
# open-source libraries:
#
# - `NeuralBench <https://facebookresearch.github.io/neuroai/>`_
#   (this package): unified benchmark suite.
# - `NeuralSet <https://kingjr.github.io/files/neuralset.pdf>`_: data
#   loading, study registry, event system.
# - `Braindecode <https://github.com/braindecode/braindecode>`_ and
#   `MOABB <https://github.com/NeuroTechX/moabb>`_: deep-learning EEG
#   architectures and BCI benchmarks.
# - `MNE-Python <https://mne.tools/>`_ and
#   `EEG-Dash <https://eegdash.org/>`_: signal-processing primitives.
#
# If ``neuralbench`` is not yet installed, follow the
# :doc:`installation guide </neuralbench/install>` and the
# :doc:`quickstart </neuralbench/auto_examples/quickstart/01_run_first_task>`
# before continuing.

# %%
# Known gaps in this starter kit
# -------------------------------
#
# Some pieces of the planned competition pipeline are not yet shipped
# with NeuralBench at the time of writing. The following items are
# being worked on and will be folded back into the relevant track
# page once they land:
#
# 1. **Official Track 2 dataset (MI / Calc / Word, 20 subjects, 6
#    sessions, Graz + BrainHero).** The corresponding study is not
#    public yet. Track 2 currently uses ``Stieger2021Continuous`` (4
#    motor-imagery classes, cross-subject) as the closest analog.
# 2. **Muse sleep-onset training set (~1000 subjects).** The Track 3
#    page currently runs on ``Kemp2000Analysis`` (Sleep-EDF) -- and the
#    additional ``Ghassemi2018You`` / ``Alvarez2022Haaglanden`` PSG
#    datasets -- with the same ``SleepOnsetTargetExtractor`` + ``bmae``
#    metric the competition will use.
# 3. **Hidden evaluation sets.** Every track is scored against a hidden
#    test set (Alljoined Emotiv for Track 1, later Graz/BrainHero
#    sessions for Track 2, the Muse cohort for Track 3, 100 new users
#    for Track 4). Only the closest public training data ships here, so
#    the numbers above are sanity checks, not leaderboard scores.
#
# Updates and corrections to this page are tracked on the GitHub
# repository linked from the (future) competition website.
