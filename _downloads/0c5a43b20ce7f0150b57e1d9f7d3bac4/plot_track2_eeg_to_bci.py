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
- **Data**: 20 subjects, 6 sessions each, 64 channels at 500 Hz,
  ~80 hours in total. Sessions 1-3 of the 10 evaluation subjects are
  released as labelled calibration; sessions 4-6 are the hidden test
  set. The 10 training subjects have all 6 sessions released.

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
# The default starter-kit baseline is
# :doc:`/neuralbench/tasks/eeg/motor_imagery`.
#
# - **CLI**: ``neuralbench eeg motor_imagery``
# - **Default dataset**: ``Stieger2021Continuous`` (62 subjects,
#   60-channel EEG, 4-class motor imagery -- LH / RH / Both / Rest).
# - **Shift**: cross-subject (NeuralBench's default ``SklearnSplit``),
#   *not* the cross-session shift of the competition. Use it to validate
#   the training pipeline and architecture choice.
# - **Headline metric key**: ``test/bal_acc``.
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
# Reproducing the baseline
# ------------------------
#
# ``Stieger2021Continuous`` and the alternative MI datasets are served by
# MOABB, which the base install does not pull, so install it first:
# ``pip install 'moabb>=1.7.1'``.
#
# .. code-block:: bash
#
#    # 1. Download Stieger2021Continuous into DATA_DIR: ~640 GB, plus ~300 GB
#    #    for the copy MOABB converts on first read. Budget ~940 GB and hours
#    #    of transfer. One-off per machine, and safe to interrupt and re-run.
#    #    Start on tangermann2012 (below, <1 GB) if that is too much for now.
#    neuralbench eeg motor_imagery --download
#
#    # 2. Preprocess into CACHE_DIR (~96 GB) -- resample, filter, scale, and
#    #    window the 598 recordings once, so every later run reads the cache
#    #    instead. ~65 min spread over 75 SLURM jobs. Do not skip it here: a
#    #    cold-cache --debug on this corpus spent ~45 min doing the same work
#    #    serially in-process.
#    neuralbench eeg motor_imagery --prepare
#
#    # 3. Sanity check before you queue anything: 2 epochs, a data subset, one
#    #    seed, always in-process, so progress lands in your terminal. ~45 s on
#    #    one V100 with the cache warm. Name the model you actually plan to run
#    #    -- a bare --debug takes the config default, which is EEGNet.
#    neuralbench eeg motor_imagery -m eegnet --debug
#
#    # 4. Same check for the foundation model. REVE's weights are gated on the
#    #    HuggingFace Hub, so this needs an account and an accepted licence;
#    #    it is the cheapest place to discover that, because a queued run
#    #    reports the failure into a job log instead of your terminal.
#    neuralbench eeg motor_imagery -m reve --debug
#
#    # 5. Full baseline -- task-specific model (EEGNet). ~30 min per seed, and
#    #    the default grid is three seeds (concurrent on SLURM).
#    neuralbench eeg motor_imagery -m eegnet
#
#    # 6. Full baseline -- foundation model (REVE), fine-tuned end to end.
#    #    ~2.5 h per seed. ~69M parameters against EEGNet's ~1.5k, all of them
#    #    trainable here, so this one wants a datacentre GPU rather than a
#    #    laptop; it also preprocesses at 200 Hz against the 120 Hz default,
#    #    warming a second cache.
#    neuralbench eeg motor_imagery -m reve
#
# :ref:`pretrained-weights` has the HuggingFace steps, and no run has a CPU
# fallback -- ``--debug`` included.
#
# Steps 5 and 6 cache the test-metric dictionary under ``SAVE_DIR``, and
# re-running the same command with ``--plot-cached`` turns those cached
# metrics into comparison plots and CSV tables without retraining. On SLURM
# they return as soon as the grid is queued, so the numbers appear in the job
# logs rather than your terminal; see :ref:`reading-results`.
#
# ``--dataset tangermann2012`` is the one to reach for first: BCI
# Competition IV-2a is 9 subjects of 22-channel four-class MI in under
# 1 GB, and the whole download-prepare-train loop runs in well under an
# hour (~15 min to prepare, ~2 min per training seed) against a well-known
# published baseline -- worth doing before committing ~940 GB and a day to
# ``Stieger2021Continuous``.

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
#    * - ``neuralbench eeg motor_imagery``
#      - ``Stieger2021Continuous`` (default)
#      - The most data by far (62 subjects, 615 h) for the motor-imagery
#        class, cross-subject.
#    * - ``neuralbench eeg motor_imagery --dataset dreyer2023``
#      - ``Dreyer2023Large``
#      - 87 subjects of 2-class MI, split on held-out subjects.
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
