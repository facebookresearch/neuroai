"""
Overview: EEG/EMG Foundation Challenge 2026
============================================

The `EEG/EMG Foundation Challenge 2026
<https://neural-interfaces26.github.io/>`_ is a competition on shift-robust
decoding of biosignals, and the multi-modal successor to the 2025 EEG
Foundation Challenge. It runs as part of the Brain and Body Foundation Model
workshop; the website has the submission window, the rules, and the prizes.

The competition is organised as **four tracks**, each isolating one kind of
distribution shift:

1. **Track 1 -- EEG-to-Image** (cross-stimulus): retrieve the image a
   subject is viewing from a single EEG epoch, ranked against held-out
   candidates. Headline metric: **Top-5 accuracy** (higher is better).
2. **Track 2 -- BCI command decoding** (cross-session): decode one of
   three cued mental commands (motor imagery, mental calculation, word
   association) from short EEG windows, training on early sessions and
   scoring on later ones without recalibration. Headline metric:
   **balanced accuracy** (higher is better).
3. **Track 3 -- Sleep onset** (cross-device): predict the latency from
   recording start to the first N2 epoch, on consumer wearable
   EEG rather than clinical polysomnography. Headline metric: **binned
   MAE (bMAE) in seconds** (lower is better) -- the absolute error
   averaged inside time-to-onset bins and then across bins with equal
   weight, so late onsets count as much as early ones.
4. **Track 4 -- EMG-to-Pose** (cross-user and cross-stage): regress
   20 hand-joint angle trajectories from 16-channel wrist surface EMG.
   Headline metric: **mean angular error** (lower is better), logged in
   radians here and reported in degrees by the competition.

All four tracks accept both task-specific models and foundation
models.

This starter kit shows how to reproduce a baseline for each track with
NeuralBench, using publicly available reference datasets.
"""

# %%
# Starter kit pages
# -----------------
#
# - :doc:`Training a model -- masked prediction on EEG <plot_pretrain_mae>`
# - :doc:`Track 1 -- EEG-to-Image <plot_track1_eeg_to_image>`
# - :doc:`Track 2 -- BCI command decoding <plot_track2_eeg_to_bci>`
# - :doc:`Track 3 -- Sleep onset <plot_track3_sleep_onset>`
# - :doc:`Track 4 -- EMG-to-Pose <plot_track4_emg_to_pose>`
# - :doc:`How to Submit a Model <plot_submission_guide>`
#
# The training page is track-agnostic: it shows how to pretrain a
# foundation model and hand it to ``neuralbench``, whichever track you
# then score it on. Each track page follows the same shape:
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
#      - Sleep (bMAE s, lower)
#      - EMG pose (MAE deg, lower)
#    * - Chance
#      - 2.22 +/- 0.31
#      - 24.81 +/- 1.03
#      - 205.42 +/- 0.01
#      - --
#    * - Dummy
#      - 2.50 +/- 0.00
#      - 25.00 +/- 0.00
#      - 299.90 +/- 0.00
#      - --
#    * - EEGNet
#      - 28.13 +/- 0.14
#      - 58.58 +/- 0.34
#      - 143.30 +/- 0.40
#      - --
#    * - REVE (foundation model)
#      - 84.75 +/- 0.38
#      - 68.04 +/- 0.73
#      - 134.89 +/- 2.02
#      - --
#    * - VEMG2Pose
#      - --
#      - --
#      - --
#      - 25.14 +/- 2.30
#
# The pose column is in degrees, to match the published baseline, while
# the task logs ``val/mae`` in radians: multiply by 180 / pi to compare.

# %%
# Budgeting the first download
# -----------------------------
#
# ``--download`` is a one-off step per machine, but not a small one, and
# the track pages put it first for that reason: the default corpora run
# from a few gigabytes to several hundred, and the upstream server is
# usually slower than your disk. Check free space on ``DATA_DIR`` before
# starting one, and run it somewhere you can leave going for hours.
#
# Footprints measured under ``DATA_DIR`` after a full download:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 20 35 45
#
#    * - Track
#      - Default dataset
#      - Disk under ``DATA_DIR``
#    * - 1 -- Image
#      - ``Gifford2022Large``
#      - ~210 GB
#    * - 2 -- BCI
#      - ``Stieger2021Continuous``
#      - ~600 GB, plus ~280 GB for the copy MOABB converts on first read
#    * - 3 -- Sleep onset
#      - ``Kemp2000Analysis``
#      - ~7 GB
#    * - 4 -- EMG pose
#      - ``Salter2024Emg2pose``
#      - ~340 GB
#
# Among Track 1's alternatives, ``Xu2024Alljoined`` is ~24 GB and
# ``Xu2025Alljoined`` (Alljoined-1.6M) ~250 GB. Track 2's
# ``tangermann2012`` is under 1 GB, small enough to exercise the whole
# pipeline before committing to a default.
#
# ``--prepare`` then writes a separate preprocessing cache under
# ``CACHE_DIR``, so the two directories are worth pointing at different
# filesystems if only one of them is large.

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
#    # 1. Run the three EEG tracks (cached automatically)
#    neuralbench eeg image motor_imagery sleep_onset -m eegnet reve
#
#    # 2. Aggregate cached results -- no retraining
#    neuralbench eeg image motor_imagery sleep_onset -m eegnet reve --plot-cached
#
#    # 3. Track 4 lives under another device -- aggregate separately
#    neuralbench emg pose -m vemg2pose --plot-cached
#
# ``--plot-cached`` aggregates within a single device, so the EMG
# track is collected by its own invocation. It produces, under
# ``<SAVE_DIR>/outputs/``:
#
# - ``core/core_bar_chart.png`` -- bar chart per task and model.
# - ``core/core_results_table.csv`` -- raw per-task metrics.
# - ``core/core_rank_table.csv`` -- ranks per task.
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
# - :doc:`NeuralBench </neuralbench/index>` (this package): unified
#   benchmark suite.
# - :doc:`NeuralSet </neuralset/index>`
#   (`paper <https://arxiv.org/abs/2605.03169>`_): data loading, study
#   registry, event system.
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
# The competition releases its own corpora through NeuralBench when
# submissions open. Until then the track pages run on the closest open
# datasets, so three pieces are still missing here:
#
# 1. **Official Track 2 dataset (MI / Calc / Word, 20 subjects, 6
#    sessions, Graz + BrainHero).** Track 2 currently uses
#    ``Stieger2021Continuous`` (4 motor-imagery classes, cross-subject)
#    as the closest analog.
# 2. **Muse sleep-onset training set (~1000 subjects).** The Track 3
#    page currently runs on ``Kemp2000Analysis`` (Sleep-EDF) -- and the
#    additional ``Ghassemi2018You`` / ``Alvarez2022Haaglanden`` PSG
#    datasets -- with the same ``SleepOnsetTargetExtractor`` + ``bmae``
#    metric the competition will use.
# 3. **Hidden evaluation sets.** Tracks 1-3 are scored against hidden
#    test sets (the Alljoined evaluation cohort, later Graz/BrainHero
#    sessions, and the Muse cohort). Track 4 uses EMG2Pose's published
#    held-out user and stage split. The numbers above are sanity checks,
#    not leaderboard scores.
#
# Each item will be folded into the relevant track page once it lands.
# If you spot something out of date, or anything else in this starter
# kit that could be clearer, please open an issue or a pull request on
# `the neuroai repository <https://github.com/facebookresearch/neuroai>`_.
