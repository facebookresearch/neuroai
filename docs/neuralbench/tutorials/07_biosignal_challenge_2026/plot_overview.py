"""
Overview: EEG/EMG Foundation Challenge 2026
============================================

If you landed here from the competition website, start with the three boxes
below: they say what NeuralBench is, what the challenge is, and why one
documentation site covers both.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: :fas:`flask` What is NeuralBench?
      :class-card: sd-shadow-sm

      An open-source benchmark suite for NeuroAI models, developed at Meta
      FAIR and documented on this site. One CLI and one Python API download a
      dataset, preprocess it, train a model on it, and report a metric --
      across dozens of brain-modelling tasks on EEG, MEG, fMRI and EMG. It is
      a research tool that exists independently of any competition.

   .. grid-item-card:: :fas:`trophy` What is the challenge?
      :class-card: sd-shadow-sm

      A competition on shift-robust decoding of biosignals, hosted and
      operated by Yneuro, Inria and UC San Diego. Four tracks, submissions
      from 21 September to 21 November 2026 on Codabench, winners announced
      at the Brain & Body Workshop at NeurIPS 2026. The `competition website
      <https://neural-interfaces26.github.io/>`_ is the authoritative source
      for all of that.

.. admonition:: How the two fit together
   :class: important

   The challenge uses NeuralBench as its **starter kit**. Every track maps
   onto a NeuralBench task, so ``neuralbench <device> <task>`` reproduces
   that track's baseline on public data today, and the 2026 competition
   corpora are released through NeuralBench when submissions open on
   21 September 2026.

   The two are not interchangeable. NeuralBench gives you the data pipeline,
   the baselines, and a number to beat; **scoring happens on Codabench**,
   against labels that stay confidential, and the numbers on these pages are
   sanity checks rather than leaderboard entries. Nothing in the rules
   requires you to train with NeuralBench either -- see
   :doc:`Training a model <plot_pretrain_mae>` for the
   bring-your-own-model route.

The competition is organised as **four tracks**, each isolating one kind of
distribution shift:

1. **Track 1 -- EEG-to-Image** (cross-stimulus): retrieve the image a
   subject is viewing from a single EEG epoch, ranked against held-out
   candidates in a frozen DINOv2 embedding space. Headline metric:
   **Top-5 retrieval accuracy** (higher is better).
2. **Track 2 -- BCI decoding** (cross-session): decode one of three cued
   mental commands (motor imagery, mental calculation, word association)
   from short EEG windows, training on a user's earlier sessions and
   scoring on their later ones without recalibration. Headline metric:
   **balanced accuracy**, averaged over subject-session-context cells
   (higher is better).
3. **Track 3 -- Sleep onset** (cross-user): predict the seconds remaining
   until the first stable N2 epoch, from four-channel wearable EEG recorded
   at home, on sleepers never seen in training. Headline metric: **binned
   MAE (bMAE) in seconds** (lower is better) -- the absolute error averaged
   inside four time-to-onset ranges and then across them with equal weight,
   so long latencies count as much as short ones.
4. **Track 4 -- EMG-to-Pose** (cross-user): regress 20 hand-joint angle
   trajectories from 16-channel wrist sEMG, on users and movement stages
   never seen during training. Headline metric: **mean absolute angular
   error**, reported in degrees by the competition and logged in radians
   here (lower is better).

All four tracks accept both task-specific models and foundation models, and
all four are scored under the same reproducibility audit -- the organisers
re-run the top three submissions of each track.
"""

# %%
# Your first hour, in order
# -------------------------
#
# 1. :doc:`Install NeuralBench </neuralbench/install>` and let the first run
#    prompt you for ``DATA_DIR``, ``CACHE_DIR`` and ``SAVE_DIR``.
# 2. Run the :doc:`quickstart
#    </neuralbench/auto_examples/quickstart/01_run_first_task>` on the 1.5 GB
#    MNE sample dataset, to confirm the install trains at all before you
#    download anything large.
# 3. Pick your track below and read its page. Each one names the matching
#    NeuralBench task and the exact commands that reproduce its baseline.
# 4. Register for that track on Codabench, from the `competition website
#    <https://neural-interfaces26.github.io/>`_.
# 5. Start the track's ``--download`` early -- it is the long pole, measured
#    in hours (see `Budgeting disk and the first download`_).
# 6. Iterate on your model, then read :doc:`How to Submit a Model
#    <plot_submission_guide>`.

# %%
# Starter kit pages
# -----------------
#
# - :doc:`Training a model -- masked prediction on EEG <plot_pretrain_mae>`
# - :doc:`Track 1 -- EEG-to-Image <plot_track1_eeg_to_image>`
# - :doc:`Track 2 -- BCI decoding <plot_track2_eeg_to_bci>`
# - :doc:`Track 3 -- Sleep onset <plot_track3_sleep_onset>`
# - :doc:`Track 4 -- EMG-to-Pose <plot_track4_emg_to_pose>`
# - :doc:`How to Submit a Model <plot_submission_guide>`
#
# The training page is track-agnostic: it shows how to pretrain a
# foundation model and hand it to ``neuralbench``, whichever track you
# then score it on. Each track page follows the same shape:
#
# 1. What the track measures (data, shift, headline metric).
# 2. Where to find the task in ``neuralbench``, and how to change it.
# 3. The commands that reproduce its baseline, with what each one costs.
# 4. Where the official competition data diverges from the default.

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
# Budgeting disk and the first download
# --------------------------------------
#
# ``--download`` is a one-off step per machine, but not a small one, and
# the track pages put it first for that reason: the default corpora run
# from a few gigabytes to several hundred, and the upstream server is
# usually slower than your disk. Check free space before starting one, and
# run it somewhere you can leave going for hours.
#
# Budget for **both** directories. ``--download`` fills ``DATA_DIR``;
# ``--prepare`` then writes a separate preprocessing cache under
# ``CACHE_DIR``, which is not the small one of the two -- for Track 4 it is
# larger than the raw data. Point them at different filesystems if only one
# of them is large.
#
# Both columns are measured after a full download and prepare of the
# default dataset:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 16 30 27 27
#
#    * - Track
#      - Default dataset
#      - ``DATA_DIR`` (raw)
#      - ``CACHE_DIR`` (prepared)
#    * - 1 -- Image
#      - ``Gifford2022Large``
#      - ~220 GB
#      - ~13 GB
#    * - 2 -- BCI
#      - ``Stieger2021Continuous``
#      - ~940 GB
#      - ~96 GB
#    * - 3 -- Sleep onset
#      - ``Kemp2000Analysis``
#      - ~8 GB
#      - ~18 GB
#    * - 4 -- EMG pose
#      - ``Salter2024Emg2pose``
#      - ~330 GB
#      - ~440 GB
#
# Two entries need a footnote. ``Stieger2021Continuous`` ends up on disk
# three times over -- the NEMAR original, the copy MOABB converts on first
# read, and a second converted tree -- which is where the ~940 GB comes
# from rather than any one copy being that large. And Track 4's cache
# exceeds its raw data because the 20 joint angles are cached as a second
# 2 kHz pass (~230 GB) alongside the EMG itself (~185 GB).
#
# These figures are larger than the archive sizes the competition website
# lists, which describe the compressed upstream releases rather than what
# lands on your disk after download and conversion.
#
# One thing the table cannot show: the cache is keyed on the
# *preprocessing* config, and models disagree about it. ``reve`` asks for
# 200 Hz where the benchmark default is 120 Hz, so running both baselines
# warms two caches rather than reusing one. Budget per model family you
# intend to run, not per track.
#
# Among Track 1's alternatives, ``Xu2024Alljoined`` is ~25 GB,
# ``Grootswagers2022Human`` ~75 GB and ``Xu2025Alljoined``
# (Alljoined-1.6M) ~270 GB. Track 2's ``tangermann2012`` is under 1 GB,
# small enough to exercise the whole pipeline before committing to a
# default.

# %%
# How much wall-clock to expect
# ------------------------------
#
# The track pages quote measured times for each command. Three things about
# *how* the commands run explain most of the variance between what they say
# and what you will see, and only the first is under your control:
#
# - **SLURM or not.** ``--prepare`` and the training runs dispatch through
#   the ``CLUSTER`` key in your config (see :doc:`/neuralbench/install`).
#   Where SLURM is available, ``--prepare`` fans the preprocessing out over
#   10 to 128 jobs, so the quoted 15 minutes is 15 minutes of *job wall
#   time*, not of compute. On one machine the same work runs serially and
#   takes proportionally longer.
# - **Three seeds, not one.** A full run is one job per model x dataset x
#   seed, and the default grid is three seeds. On SLURM they run
#   concurrently, so the command finishes in roughly the time of its slowest
#   job; locally they run one after another, so budget three times the
#   per-seed figure the track pages quote.
# - **Cache warmth.** ``--debug`` is the odd one out: it always runs
#   in-process on one GPU with a single seed, never on SLURM. That makes it
#   a genuine end-to-end check, but it also means skipping ``--prepare``
#   does not skip the preprocessing -- it just moves it into your debug run,
#   which is how a nominally 45-second sanity check becomes a 45-minute one.

# %%
# Collecting and plotting your results
# -------------------------------------
#
# A finished run prints Lightning's test-metric table and returns the same
# numbers as a dictionary, which is what lands on disk. From a real
# ``sleep_onset`` job:
#
# .. code-block:: python
#
#    {'n_total_params': 1425, 'n_trainable_params': 1425,
#     'peak_cpu_memory_mb': 2040.26, 'peak_gpu_memory_mb': 15.89,
#     'test/bmae': 140.209, 'test/mae': 257.191, 'test/pearsonr': 0.4220,
#     'test/r2_score': -1.3576, 'test/rmse': 298.825,
#     'training_time_s': 314.687}
#
# The headline metric each track is scored on is one key of that dictionary,
# named on the track page. The rest is there to keep you honest about what
# produced it -- parameter counts, peak memory, and training time.
#
# Every run caches that dictionary under ``SAVE_DIR``.
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
#    as its default analog, and ``Scherer2015Individually`` for the
#    cross-session, multi-command side of the task.
# 2. **Muse sleep-onset training set (~1000 subjects).** The Track 3
#    page currently runs on ``Kemp2000Analysis`` (Sleep-EDF) -- and the
#    additional ``Ghassemi2018You`` / ``Alvarez2022Haaglanden`` PSG
#    datasets -- with the same ``SleepOnsetTargetExtractor`` + ``bmae``
#    metric the competition will use. Note that these are clinical
#    polysomnography, so the starter kit adds a device gap the competition
#    itself does not have: there, training and evaluation are both Muse.
# 3. **Hidden evaluation sets.** All four tracks are scored against labels
#    that stay confidential (the Alljoined evaluation cohort, later
#    Graz/BrainHero sessions, the Muse evaluation cohort, and the EMG2Pose
#    evaluation cohort). The numbers above are sanity checks, not
#    leaderboard scores.
#
# Each item will be folded into the relevant track page once it lands.
# If you spot something out of date, or anything else in this starter
# kit that could be clearer, please open an issue or a pull request on
# `the neuroai repository <https://github.com/facebookresearch/neuroai>`_.
