"""
Track 2 -- BCI decoding (cross-session)
=========================================

.. image:: https://neural-interfaces26.github.io/exports/bci-decoding.gif
   :alt: Three cued mental commands decoded on an unseen later session
   :target: https://neural-interfaces26.github.io/tracks.html
   :width: 100%

Given short EEG windows recorded while a user performs one of three
cued mental commands (kinesthetic motor imagery, mental calculation, or
word association), decode the active command. The competition tests
**cross-session** generalisation: models train on a user's earlier
sessions and are scored on their later ones, with no per-session
recalibration allowed.

- **Shift**: earlier sessions -> later sessions (Graz + BrainHero
  contexts), within the same user.
- **Headline metric**: balanced accuracy computed for each session of each
  subject, then averaged over sessions (higher is better).
- **Data**: 20 subjects, 6 sessions each, 47 channels (43 EEG, 2 EMG,
  2 EOG) at 500 Hz, ~80 hours in total. Sessions 1-3 of the 10
  evaluation subjects are released as labelled calibration; sessions
  4-6 are the hidden test set. The 10 training subjects have all 6
  sessions released.

.. note::
   The training release of the official Track 2 corpus is available as
   `NEMAR nm000290, version 1.0.0 <https://doi.org/10.82901/nemar.nm000290>`__:
   112 runs from 14 sessions of 10 participants, 41 EEG channels at
   500 Hz (CC-BY-4.0). It is the default dataset of ``neuralbench eeg
   _motor_imagery_stream``, which runs a local subject-disjoint benchmark
   on it, not the official split or the sealed evaluation.
   `Starter-kit analogs`_ below lists the other NeuralBench tasks that
   come closest, one per axis of the official task.
"""

# %%
# New to NeuralBench? Start here
# ------------------------------
#
# This page is a track guide, not an introduction to the ecosystem. If you
# arrived straight from the competition website:
#
# - :doc:`Challenge overview <plot_overview>` -- what NeuralBench is, how it
#   relates to the competition, and the baseline numbers for all four tracks.
# - :doc:`Installation </neuralbench/install>` and the :doc:`quickstart
#   </neuralbench/auto_examples/quickstart/01_run_first_task>` -- get a task
#   running on a 1.5 GB dataset before you download anything large.
# - `Official Track 2 guide <https://neural-interfaces26.github.io/tracks.html>`__
#   -- registration, rules, data access, prizes, leaderboard. Authoritative
#   on every competition matter; this page only covers the code.
# - :doc:`How to Submit a Model <plot_submission_guide>`.

# %%
# Where to find this task in NeuralBench
# --------------------------------------
#
# NeuralBench has two versions of this task. **Use ``_motor_imagery_stream``
# for the competition.**
#
# - ``_motor_imagery_stream`` scores each session one window at a time,
#   forward in time: test windows come one per batch, its runs in recording
#   order, and a model's optional ``reset_state()`` is called on a fresh copy
#   of the model at the start of each session (``data.stream_by: [subject,
#   session]``, see :doc:`Modifying the training loop
#   </neuralbench/auto_examples/advanced/modify_training_loop>`). A stateful
#   model can therefore adapt to a session as it goes, across its runs, but
#   carries nothing over to the next one. Validation stays batched; set
#   ``data.val_batch_size: 1`` to select checkpoints on streams too, at
#   several times the training cost. The leading underscore only keeps the
#   task out of ``neuralbench eeg all``.
# - :doc:`/neuralbench/tasks/eeg/motor_imagery` is the NeuralBench benchmark
#   version, which scores windows independently, in batches. The baseline
#   numbers in the :doc:`challenge overview <plot_overview>` come from it, on
#   its own default dataset, ``Stieger2021Continuous`` (62 subjects of
#   4-class MI). With ``--dataset dreyer2026proteus`` it tests on the same
#   windows as ``_motor_imagery_stream``, but with its own preprocessing.
#
# Both versions share the split, target and loss. ``_motor_imagery_stream``
# also feeds the signal in microvolts, without the default per-recording
# scaler or clamping (resampling and filters stay; to scale each window, see
# :ref:`preprocessing-inside-the-model`), adds the per-session
# balanced accuracy, defaults to the competition corpus, and ships
# only the two other dataset variants these pages use, ``dreyer2023`` and
# ``tangermann2012``; the other MI corpora are variants of ``motor_imagery``.
#
# - **CLI**: ``neuralbench eeg _motor_imagery_stream``
# - **Dataset**: ``Dreyer2026Proteus`` (NEMAR nm000290, ~1.5 GB), the
#   training release of the official corpus: 10 participants, 41-channel
#   EEG, the three cued commands -- motor imagery, mental subtraction, word
#   generation -- with the Graz and BrainHero interfaces. Each window covers
#   1-5 s after a cue. Every run mixes the three commands in random order.
# - **Shift**: held-out subjects, *not* the cross-session shift of the
#   competition. Use it to validate the training pipeline and architecture
#   choice.
# - **Headline metric key**: ``test/bal_acc_stream_mean`` (balanced accuracy
#   per subject and session, averaged over sessions).
#
# ``Dreyer2026Proteus`` is also the corpus Codabench scores Track 2 against
# during the warm-up phase. ``--dataset dreyer2023`` selects
# ``Dreyer2023Large`` (87 subjects, 27-channel EEG, 2-class motor imagery --
# left hand / right hand, ~19 GB).
#
# **What the config is.** A NeuralBench task is one ``config.yaml``, and
# nothing else: a YAML overlay on the ``neuralbench/defaults/`` configs naming
# the study to load, how to split it, what the target is, the loss, and the
# metrics. Reading it is the fastest way to know exactly what the baseline
# does.
#
# .. dropdown:: Show ``tasks/eeg/_motor_imagery_stream/config.yaml``
#
#    .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/_motor_imagery_stream/config.yaml
#       :language: yaml
#
# .. dropdown:: Show ``tasks/eeg/_motor_imagery_stream/datasets/dreyer2023.yaml``
#
#    .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/_motor_imagery_stream/datasets/dreyer2023.yaml
#       :language: yaml
#
# **How to change it**, in increasing order of effort:
#
# - ``--dataset <name>`` merges
#   ``tasks/eeg/_motor_imagery_stream/datasets/<name>.yaml`` over the base config.
# - ``-m <model>`` and ``-w <preset>`` swap the architecture and the
#   adaptation strategy (frozen probe, LoRA, full fine-tuning) without
#   touching any file.
# - Anything else -- window length, learning rate, split -- is a config edit.
#   From Python, pass dotted keys to :func:`~neuralbench.evaluate_model`
#   (``overrides={"data.duration": 3.0}``). From a source checkout
#   (``pip install -e``), edit ``config.yaml`` directly, or add your own
#   ``datasets/*.yaml`` variant beside the existing ones and select it with
#   ``--dataset``. Both routes are described in :doc:`Adding a New Task
#   </neuralbench/auto_examples/adding_task/create_new_task>`. The split is
#   the field to reach for here: see `Adapting to the competition setup`_.

# %%
# Split and model selection
# --------------------------
#
# **Split (task default ``Dreyer2026Proteus``).** Subject-level
# 60 / 20 / 20 drawn by ``SklearnSplit`` with ``split_by: subject`` and both
# seeds fixed at 33, which on 10 participants resolves to **6 train /
# 2 validation / 2 test** (1,521 / 379 / 755 windows). Every session of a
# participant lands in the same fold.
#
# That test partition is the current Codabench warm-up evaluation set, so a
# ``test/bal_acc_stream_mean`` from this configuration and a warm-up
# leaderboard score are computed on the same windows.
#
# **Split (``--dataset dreyer2023``).** Subject-level and predefined, not
# random. ``Dreyer2023Large`` pools the study's parts A (subjects 1-60),
# B (61-81) and C (82-87); ``PredefinedSplit`` assigns **all 21 subjects of
# part B to test** and everything else to train, then holds out 20 % of the
# remaining subjects as validation (``valid_split_by: subject``, seed 33).
# Part B is used for test because six part-C subjects are re-recordings of
# part-A subjects under different IDs, so carving test out of A or C could
# leak a person across folds. Every subject therefore appears in exactly one
# fold, and the partition is identical on every machine.
#
# Either way the starter-kit shift is *cross-subject*, while the sealed
# phase's is *cross-session within subject*. See `Adapting to the
# competition setup`_ for what changes.
#
# **Model selection.** The checkpoint with the highest
# **``val/bal_acc_stream_mean``** is kept -- the headline metric, just on the
# validation fold. Training runs for at most 40 epochs and stops early after
# 5 epochs without improvement; only that single best checkpoint is scored on
# test.
#
# Averaging over sessions makes every session count equally regardless of
# how many windows it holds. ``test/bal_acc``, also logged, pools all
# windows instead -- a different number from the same predictions.

# %%
# Reproducing the baseline
# ------------------------
#
# .. code-block:: bash
#
#    # 1. Download Dreyer2026Proteus into DATA_DIR: ~1.5 GB. One-off per
#    #    machine, and safe to interrupt and re-run.
#    neuralbench eeg _motor_imagery_stream --download
#
#    # 2. Preprocess into CACHE_DIR -- resample, filter and window
#    #    every recording once, so each later run reads the cache instead.
#    #    No GPU needed, and it fans out over SLURM when one is configured.
#    neuralbench eeg _motor_imagery_stream --prepare
#
#    # 3. Sanity check before you queue anything: 2 epochs, a data subset, one
#    #    seed, always in-process, so progress lands in your terminal. Name
#    #    the model you actually plan to run -- a bare --debug takes the
#    #    config default, which is EEGNet.
#    neuralbench eeg _motor_imagery_stream -m eegnet --debug
#
#    # 4. Same check for the foundation model. The first build pulls REVE's
#    #    weights from the HuggingFace Hub, which needs network access; doing
#    #    it here rather than in a queued run keeps any failure in your
#    #    terminal instead of a job log.
#    neuralbench eeg _motor_imagery_stream -m reve --debug
#
#    # 5. Full baseline -- task-specific model (EEGNet). The default grid is
#    #    three seeds (concurrent on SLURM).
#    neuralbench eeg _motor_imagery_stream -m eegnet
#
#    # 6. Full baseline -- foundation model (REVE), fine-tuned end to end.
#    #    ~69M parameters against EEGNet's ~1.5k, all of them trainable here,
#    #    so this one wants a datacentre GPU rather than a laptop; it also
#    #    applies its own preprocessing (200 Hz, scaled) instead of the
#    #    task's, warming a second cache. Its scaler is fit on each whole
#    #    recording, which a streamed submission cannot do, so its test score
#    #    is optimistic.
#    neuralbench eeg _motor_imagery_stream -m reve
#
# Add ``--dataset dreyer2023`` to any of these commands to run on
# ``Dreyer2023Large`` (~19 GB), or ``--dataset tangermann2012`` for BCI
# Competition IV-2a: 9 subjects of 22-channel four-class MI in under 1 GB,
# with the whole download-prepare-train loop in well under an hour (~15 min
# to prepare, ~2 min per training seed) against a well-known published
# baseline. Both are served by MOABB, which the base install does not pull,
# so install it first: ``pip install 'moabb>=1.7.1'``.
#
# The baseline numbers in the :doc:`challenge overview <plot_overview>` come
# from ``neuralbench eeg motor_imagery``, on ``Stieger2021Continuous``: 62
# subjects split **36 train / 13 validation / 13 test** by the same
# ``SklearnSplit``. Budget ~940 GB on disk (~640 GB, plus ~300 GB for the
# copy MOABB converts on first read), ~96 GB of cache and ~65 min of
# preparation over 75 SLURM jobs, then ~30 min per EEGNet seed and ~2.5 h
# per REVE seed. Do not skip ``--prepare`` there: a cold-cache ``--debug``
# on that corpus spent ~45 min doing the same work serially in-process.
#
# :ref:`pretrained-weights` covers the hub cache, and no run has a CPU
# fallback -- ``--debug`` included.
#
# Steps 5 and 6 cache the test-metric dictionary under ``SAVE_DIR``, and
# re-running the same command with ``--plot-cached`` turns those cached
# metrics into comparison plots and CSV tables without retraining. On SLURM
# they return as soon as the grid is queued, so the numbers appear in the job
# logs rather than your terminal; see :ref:`reading-results`.

# %%
# Evaluating a model of your own
# ------------------------------
#
# A model that lives in your own codebase needs no YAML here:
# :func:`~neuralbench.evaluate_model` takes the built instance, wraps it in a
# probe sized to the task, and returns the scores as a DataFrame.
#
# .. code-block:: python
#
#    from neuralbench import check_model, evaluate_model
#
#    print(check_model(my_model, "eeg", "_motor_imagery_stream"))  # shapes only, seconds
#    scores = evaluate_model(my_model, "eeg", "_motor_imagery_stream", name="my-fm", debug=True)
#
# See :doc:`Evaluating your own model
# </neuralbench/auto_examples/quickstart/03_evaluate_your_own_model>` for what
# ``forward`` has to accept, the adaptation presets, and how to fan the runs
# out to SLURM.

# %%
# Starter-kit analogs
# -------------------
#
# The training release of the official corpus (first row) has the three
# mental commands and both interfaces. The four public corpora the competition
# points to are split across four NeuralBench tasks; each covers a different
# axis of Track 2, and all are worth training on:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 34 30 36
#
#    * - Command
#      - Dataset
#      - What it gives you
#    * - ``neuralbench eeg _motor_imagery_stream``
#      - ``Dreyer2026Proteus`` (task default, NEMAR nm000290)
#      - The training release of the official corpus: the three commands
#        with the Graz and BrainHero interfaces, 41 EEG channels, split on
#        held-out subjects, and the corpus Codabench scores against during
#        warm-up.
#    * - ``neuralbench eeg _motor_imagery_stream --dataset dreyer2023``
#      - ``Dreyer2023Large``
#      - 87 subjects of 2-class MI, split on held-out subjects.
#    * - ``neuralbench eeg motor_imagery``
#      - ``Stieger2021Continuous`` (benchmark default)
#      - The most data by far (62 subjects, 615 h) for the motor-imagery
#        class, cross-subject, and the source of the published baseline
#        numbers.
#    * - ``neuralbench eeg mental_imagery``
#      - ``Scherer2015Individually``
#      - The closest paradigm match: five cued mental tasks including
#        arithmetic and letter association, split **cross-session**
#        (session 1 held out as test).
#    * - ``neuralbench eeg mental_arithmetic``
#      - ``Zyma2019Electroencephalograms``
#      - Mental calculation against a rest baseline, 36 subjects.
#
# Every other MI corpus registered in
# :doc:`/neuralbench/tasks/eeg/motor_imagery` (Cho2017, Lee2019, ...) can be
# selected with ``neuralbench eeg motor_imagery --dataset <name>``.

# %%
# Adapting to the competition setup
# ----------------------------------
#
# The task already runs on the official corpus, with its three classes. To
# match the official Track 2 evaluation regime, the **split** has to change:
# replace the default ``SklearnSplit`` with a predefined per-subject split
# where sessions 1-3 are train and sessions 4-6 are test. The
# ``neuralbench.transforms.PredefinedSplit`` already used by
# ``mental_imagery``, ``reaction_time`` and ``psychopathology`` is the right
# primitive -- the ``test_split_query`` becomes
# ``"subject in evaluation_subjects and session in [4, 5, 6]"``.
#
# Submissions may dispatch internally to per-subject sub-models using
# the per-example metadata dictionary ``m`` (subject, session, run,
# paradigm). Re-training on the hidden later sessions is forbidden.
