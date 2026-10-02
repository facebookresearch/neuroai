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
- **Headline metric**: balanced accuracy averaged over
  subject-session-context cells (higher is better).
- **Data**: 20 subjects, 6 sessions each, 47 channels (43 EEG, 2 EMG,
  2 EOG) at 500 Hz, ~80 hours in total. Sessions 1-3 of the 10
  evaluation subjects are released as labelled calibration; sessions
  4-6 are the hidden test set. The 10 training subjects have all 6
  sessions released.

.. note::
   The official Track 2 corpus (Graz / BrainHero, 3 classes: MI / Calc
   / Word) is released through NeuralBench when submissions open.
   Until then, `Starter-kit analogs`_ below lists the NeuralBench tasks
   that come closest, one per axis of the official task.
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
# The starter-kit baseline is
# :doc:`/neuralbench/tasks/eeg/motor_imagery`.
#
# The task's own default dataset is ``Stieger2021Continuous``: 62 subjects
# of 4-class MI, and the corpus the published NeuralBench Track 2 baseline
# numbers come from. It is also a ~640 GB download (~940 GB once MOABB has
# converted it), and it is not what Codabench scores against.
#
# So these pages build against the **recommended warm-up configuration**
# instead, selected with an explicit flag:
#
# - **CLI**: ``neuralbench eeg motor_imagery --dataset dreyer2023``
# - **Dataset**: ``Dreyer2023Large`` (87 subjects, 27-channel EEG,
#   2-class motor imagery -- left hand / right hand, ~19 GB). This is the
#   corpus Codabench scores Track 2 against during the warm-up phase, so
#   it is the one to build against first.
# - **Shift**: held-out subjects, *not* the cross-session shift of the
#   competition. Use it to validate the training pipeline and architecture
#   choice.
# - **Headline metric key**: ``test/bal_acc``.
#
# .. important::
#    ``--dataset dreyer2023`` is required. Dropping it does not fall back
#    to the warm-up corpus -- it runs ``Stieger2021Continuous``, a
#    different task (4 classes, not 2) on a very much larger download.
#    Every command on this page carries the flag for that reason.
#
# **What the config is.** A NeuralBench task is one ``config.yaml``, and
# nothing else: a YAML overlay on ``neuralbench/defaults/config.yaml`` naming
# the study to load, how to split it, what the target is, the loss, and the
# metrics. Reading it is the fastest way to know exactly what the baseline
# does.
#
# .. dropdown:: Show ``tasks/eeg/motor_imagery/config.yaml``
#
#    .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/motor_imagery/config.yaml
#       :language: yaml
#
# .. dropdown:: Show ``tasks/eeg/motor_imagery/datasets/dreyer2023.yaml``
#
#    .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/motor_imagery/datasets/dreyer2023.yaml
#       :language: yaml
#
# **How to change it**, in increasing order of effort:
#
# - ``--dataset <name>`` merges
#   ``tasks/eeg/motor_imagery/datasets/<name>.yaml`` over the base config.
#   Seventeen MI corpora ship that way, and the competition corpus will too
#   once it lands.
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
# That part-B test partition is the current Codabench warm-up evaluation
# set, so a ``test/bal_acc`` from this configuration and a warm-up
# leaderboard score measure the same thing.
#
# **Split (task default ``Stieger2021Continuous``).** Subject-level
# 60 / 20 / 20 drawn by ``SklearnSplit`` with ``split_by: subject`` and both
# seeds fixed at 33, which on 62 subjects resolves to **36 train /
# 13 validation / 13 test**.
#
# Either way the starter-kit shift is *cross-subject*, while the sealed
# phase's is *cross-session within subject*. See `Adapting to the
# competition setup`_ for what changes.
#
# **Model selection.** The checkpoint with the highest **``val/bal_acc``**
# is kept -- validation balanced (macro-averaged) accuracy, the same
# quantity as the headline ``test/bal_acc``, just on the validation fold.
# Training runs for at most 40 epochs and stops early after 5 epochs
# without improvement; only that single best checkpoint is scored on test.
#
# The warm-up scorer also ranks on balanced accuracy, but pooled over all
# evaluation windows. The sealed phase instead averages it over
# subject-session-context cells, so that every cell counts equally
# regardless of how many windows it holds -- a different number from the
# same predictions.

# %%
# Reproducing the baseline
# ------------------------
#
# ``Dreyer2023Large`` and the alternative MI datasets are served by
# MOABB, which the base install does not pull, so install it first:
# ``pip install 'moabb>=1.7.1'``.
#
# .. code-block:: bash
#
#    # 1. Download Dreyer2023Large into DATA_DIR: ~19 GB. One-off per
#    #    machine, and safe to interrupt and re-run.
#    neuralbench eeg motor_imagery --dataset dreyer2023 --download
#
#    # 2. Preprocess into CACHE_DIR -- resample, filter, scale and window
#    #    every recording once, so each later run reads the cache instead.
#    #    No GPU needed, and it fans out over SLURM when one is configured.
#    neuralbench eeg motor_imagery --dataset dreyer2023 --prepare
#
#    # 3. Sanity check before you queue anything: 2 epochs, a data subset, one
#    #    seed, always in-process, so progress lands in your terminal. Name
#    #    the model you actually plan to run -- a bare --debug takes the
#    #    config default, which is EEGNet.
#    neuralbench eeg motor_imagery --dataset dreyer2023 -m eegnet --debug
#
#    # 4. Same check for the foundation model. The first build pulls REVE's
#    #    weights from the HuggingFace Hub, which needs network access; doing
#    #    it here rather than in a queued run keeps any failure in your
#    #    terminal instead of a job log.
#    neuralbench eeg motor_imagery --dataset dreyer2023 -m reve --debug
#
#    # 5. Full baseline -- task-specific model (EEGNet). The default grid is
#    #    three seeds (concurrent on SLURM).
#    neuralbench eeg motor_imagery --dataset dreyer2023 -m eegnet
#
#    # 6. Full baseline -- foundation model (REVE), fine-tuned end to end.
#    #    ~69M parameters against EEGNet's ~1.5k, all of them trainable here,
#    #    so this one wants a datacentre GPU rather than a laptop; it also
#    #    preprocesses at 200 Hz against the 120 Hz default, warming a second
#    #    cache.
#    neuralbench eeg motor_imagery --dataset dreyer2023 -m reve
#
# .. tip::
#    Smaller still: ``--dataset tangermann2012`` is BCI Competition IV-2a,
#    9 subjects of 22-channel four-class MI in under 1 GB, with the whole
#    download-prepare-train loop in well under an hour (~15 min to prepare,
#    ~2 min per training seed) against a well-known published baseline.
#
# Dropping ``--dataset dreyer2023`` from any of the commands above runs
# ``Stieger2021Continuous`` instead. Budget ~940 GB on disk (~640 GB, plus
# ~300 GB for the copy MOABB converts on first read), ~96 GB of cache and
# ~65 min of preparation over 75 SLURM jobs, then ~30 min per EEGNet seed
# and ~2.5 h per REVE seed. Do not skip ``--prepare`` there: a cold-cache
# ``--debug`` on that corpus spent ~45 min doing the same work serially
# in-process.
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
#    print(check_model(my_model, "eeg", "motor_imagery"))  # shapes only, seconds
#    scores = evaluate_model(my_model, "eeg", "motor_imagery", name="my-fm", debug=True)
#
# See :doc:`Evaluating your own model
# </neuralbench/auto_examples/quickstart/03_evaluate_your_own_model>` for what
# ``forward`` has to accept, the adaptation presets, and how to fan the runs
# out to SLURM.

# %%
# Starter-kit analogs
# -------------------
#
# No public dataset has all of the official task at once -- three mental
# commands, six sessions, one user at a time -- so the four public corpora the
# competition points to are split across three NeuralBench tasks. Each covers
# a different axis of Track 2, and all four are worth training on:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 34 30 36
#
#    * - Command
#      - Dataset
#      - What it gives you
#    * - ``neuralbench eeg motor_imagery --dataset dreyer2023``
#      - ``Dreyer2023Large`` (recommended warm-up configuration)
#      - 87 subjects of 2-class MI, split on held-out subjects, and the
#        corpus Codabench scores against during warm-up.
#    * - ``neuralbench eeg motor_imagery``
#      - ``Stieger2021Continuous`` (task default)
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
# :doc:`/neuralbench/tasks/eeg/motor_imagery` (BCI Competition IV, Cho2017,
# Lee2019, ...) can be selected the same way with ``--dataset <name>``.

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
#    ``neuralbench.transforms.PredefinedSplit`` already used by
#    ``mental_imagery``, ``reaction_time`` and ``psychopathology`` is the
#    right primitive -- the ``test_split_query`` becomes
#    ``"subject in evaluation_subjects and session in [4, 5, 6]"``.
#
# Submissions may dispatch internally to per-subject sub-models using
# the per-example metadata dictionary ``m`` (subject, session, run,
# paradigm). Re-training on the hidden later sessions is forbidden.
