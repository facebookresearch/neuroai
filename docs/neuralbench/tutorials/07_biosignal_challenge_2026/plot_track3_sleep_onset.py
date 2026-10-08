"""
Track 3 -- Sleep onset (cross-user latency prediction)
=======================================================

.. image:: https://neural-interfaces26.github.io/exports/sleep-onset.gif
   :alt: Seconds to the first N2 epoch predicted from wearable EEG
   :target: https://neural-interfaces26.github.io/tracks.html
   :width: 100%

Given continuous four-channel wearable EEG recorded at home, predict the
seconds remaining until the **first N2 epoch** -- the first epoch scored
N2, not the start of a run of consecutive N2 epochs. The competition tests
generalisation across nights and across sleepers on one consumer device:
training and evaluation use the same Muse headband, the same home protocol
and the same target. Precise onset timing replaces full hypnogram
reconstruction because a sparse wearable montage supports it poorly.

- **Shift**: nights and sleepers, on one consumer device. The evaluation
  cohort contains **both sleepers seen in training and sleepers never seen
  in training**, so a model has to hold up on new nights from familiar
  people as well as on new people. Night-to-night variation, motion
  artifacts, impedance changes and channel dropout come with the home
  setting.
- **Headline metric**: binned onset error in seconds, lower is better, but
  the binning is weighted differently in each phase. Both phases split the
  error by *true* time to onset into [0, 40), [40, 90), [90, 300) and
  [300, 600] s.

  - *Sealed Muse phase*: **W-bMAE**. The four ranges carry severity
    weights of **10x, 5x, 3x and 1x**, so an error close to onset costs
    far more than one ten minutes out. W-bMAE is then computed separately
    over seen subjects (new nights from people in the training set) and
    unseen subjects, and the ranking score is the **macro-average of those
    two**, weighting night-to-night and inter-person generalisation
    equally.
  - *Sleep-EDF proxy*: **unweighted bMAE** plus plain MAE. Publishing the
    Muse dataset does not change this recipe or the Codabench scorer.

  ``neuralbench.metrics.BinnedMAE`` implements the unweighted form, so
  ``val/bmae`` and ``test/bmae`` here match the Sleep-EDF proxy objective and
  not the sealed one. Nothing in the start kit computes the severity
  weights or the seen/unseen macro-average.
- **Data**: continuous Muse wearable EEG sampled at **128 Hz**, with
  ``n2_onset`` annotations on the training cohort and a separate hidden
  evaluation cohort recorded with the same hardware and protocol. The public
  release contains **540 recordings from 203 participants**, totaling about
  157.52 hours. Its supplied split is 500 training recordings and 40 test
  recordings, all from participants seen in training. These counts do not
  describe the sealed cohort. The onset that
  ``AddSleepOnsetTargets`` extracts is the earliest annotated N2 event of
  the recording, which is exactly the competition's definition.

.. note::
   Muse data are available as `NEMAR nm000287, version 1.0.0
   <https://doi.org/10.82901/nemar.nm000287>`__, by Muse Team under
   CC-BY-NC-SA-4.0. They are the default dataset of ``neuralbench eeg
   _sleep_onset_stream``, which keeps the starter's subject-disjoint split
   and unweighted bMAE model selection; it does not use the supplied
   session split or reproduce the sealed score. Source session labels
   remain available in NeuralFetch for experiments that need the supplied
   split.

   Every released recording ends 300 s after N2, so neither total recording
   length nor whole-recording preprocessing statistics are legitimate
   predictive inputs; the start kit's preprocessing is not causal.
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
# - `Official Track 3 guide <https://neural-interfaces26.github.io/tracks.html>`__
#   -- registration, rules, data access, prizes, leaderboard. Authoritative
#   on every competition matter; this page only covers the code.
# - :doc:`How to Submit a Model <plot_submission_guide>`.

# %%
# Where to find this task in NeuralBench
# --------------------------------------
#
# NeuralBench has two versions of this task. **Use ``_sleep_onset_stream``
# for the competition, in both phases.**
#
# - ``_sleep_onset_stream`` scores each recording one 5 s window at a time,
#   forward in time, as Codabench does, and with ``--dataset
#   kemp2000analysis`` its ``test/bmae`` is the warm-up leaderboard's
#   measurement. Validation and test streams start at a random
#   time before N2 onset, so the time since a stream began says little about
#   the target, and a model's optional ``reset_state()`` is called on a fresh
#   copy of the model at the start of each recording (see :doc:`Modifying the
#   training loop </neuralbench/auto_examples/advanced/modify_training_loop>`).
#   Validation stays batched; set ``data.val_batch_size: 1`` to select
#   checkpoints on streams too, at several times the training cost. The
#   leading underscore only keeps the task out of ``neuralbench eeg all``.
# - :doc:`/neuralbench/tasks/eeg/sleep_onset` is the NeuralBench benchmark
#   version: validation and test cover the last 20 minutes before N2 onset of
#   every recording, each window scored independently. The baseline numbers
#   in the :doc:`challenge overview <plot_overview>` come from it, on its own
#   default dataset, ``Kemp2000Analysis`` (Sleep-EDF Expanded, 78
#   participants recorded over up to two nights each, 2 EEG channels, full
#   polysomnography). Its scores are not comparable with
#   ``_sleep_onset_stream``'s, as the test windows differ.
#
# Both versions share the split, target, loss and metrics.
# ``_sleep_onset_stream`` defaults to the Muse data, and selects Sleep-EDF
# with ``--dataset kemp2000analysis``:
#
# - **CLI**: ``neuralbench eeg _sleep_onset_stream``
# - **Default dataset**: ``Interaxon2026Muse`` (NEMAR nm000287, ~1.1 GB:
#   540 at-home recordings from 203 participants, 4 EEG channels -- TP9,
#   AF7, AF8, TP10 -- at 128 Hz, first-N2 annotations only).
# - **Target**: the time *remaining* until the first N2 epoch, which
#   ``AddSleepOnsetTargets`` + ``SleepOnsetTargetExtractor`` recompute for
#   every window as ``clip(n2_onset - window_stop, 0, 600)`` seconds. The
#   task therefore predicts once per 5-second window rather than once per
#   recording, and the competition's single ``tau_hat`` is
#   ``window_stop + prediction`` read off any window within 600 s of
#   onset, where the cap has not saturated the target.
# - **Headline metric key**: ``test/bmae`` (binned MAE in seconds).
#
# **What the config is.** A NeuralBench task is one ``config.yaml``, and
# nothing else: a YAML overlay on ``neuralbench/defaults/config.yaml`` naming
# the study to load, how to split it, what the target is, the loss, and the
# metrics. Reading it is the fastest way to know exactly what the baseline
# does.
#
# .. dropdown:: Show ``tasks/eeg/_sleep_onset_stream/config.yaml``
#
#    .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/_sleep_onset_stream/config.yaml
#       :language: yaml

# %%
# Split and model selection
# --------------------------
#
# **Split.** Participant-level 60 / 20 / 20, drawn by ``SklearnSplit`` with
# ``split_by: subject``. Every recording from a participant lands in exactly
# one fold, so no sleeper is shared between train, validation and test --
# the starter kit's test score therefore measures generalisation to people
# the model has never seen. Both split seeds are fixed at 33, so the
# partition is identical on every machine and every run. On the Muse data's
# 203 participants that resolves to **121 train / 41 validation / 41 test**
# (315 / 120 / 105 recordings), and on Sleep-EDF's 78 participants to
# **46 train / 16 validation / 16 test**.
#
# That 16-participant Sleep-EDF test partition *is* the current Codabench
# warm-up evaluation set: the scorer runs on the same Sleep-EDF subset this
# split produces at random state 33. A ``_sleep_onset_stream --dataset
# kemp2000analysis`` ``test/bmae`` and a warm-up leaderboard score are
# therefore the same measurement.
#
# The sealed phase is a different story. Its Muse cohort mixes seen and
# unseen sleepers, while this split holds every sleeper out, so the sealed
# score is not something the starter kit can approximate.
#
# **Model selection.** The checkpoint with the lowest **``val/bmae``** is
# kept -- validation binned MAE in seconds, the same quantity as the
# warm-up ``test/bmae``, just on the validation fold. Training runs for at
# most 40 epochs and stops early after 7 epochs without improvement; only
# that single best checkpoint is scored on test. Note this selects on the
# unweighted metric; compared with the sealed W-bMAE weights, a model tuned
# this way will be under-weighting the near-onset range that matters most.
#
# **How to change it**, in increasing order of effort:
#
# - ``--dataset <name>`` merges
#   ``tasks/eeg/_sleep_onset_stream/datasets/<name>.yaml`` over the base config.
#   That is how Sleep-EDF and the other PSG corpora below are selected.
# - ``-m <model>`` and ``-w <preset>`` swap the architecture and the
#   adaptation strategy (frozen probe, LoRA, full fine-tuning) without
#   touching any file.
# - Anything else -- the 600 s cap, the 5 s window, the learning rate -- is a
#   config edit. From Python, pass dotted keys to
#   :func:`~neuralbench.evaluate_model` (``overrides={"data.duration":
#   30.0}``). From a source checkout (``pip install -e``), edit
#   ``config.yaml`` directly, or add your own ``datasets/*.yaml`` variant
#   beside the existing ones and select it with ``--dataset``. Both routes
#   are described in :doc:`Adding a New Task
#   </neuralbench/auto_examples/adding_task/create_new_task>`.

# %%
# Reproducing the baseline
# ------------------------
#
# This is the cheapest of the four tracks to get running end to end, which
# makes it a good first target whichever track you plan to submit to.
#
# .. code-block:: bash
#
#    # 1. Download the Muse data into DATA_DIR: ~1.1 GB. One-off per
#    #    machine, and safe to interrupt and re-run.
#    neuralbench eeg _sleep_onset_stream --download
#
#    # 2. Preprocess into CACHE_DIR -- resample, filter, scale, and cut the
#    #    540 recordings into 5 s windows once, so every later run reads the
#    #    cache instead. No GPU needed, and it fans out over SLURM when one is
#    #    configured.
#    neuralbench eeg _sleep_onset_stream --prepare
#
#    # 3. Sanity check before you queue anything: 2 epochs, a data subset, one
#    #    seed, always in-process, so progress lands in your terminal. Name
#    #    the model you actually plan to run -- a bare --debug takes the
#    #    config default, which is EEGNet.
#    neuralbench eeg _sleep_onset_stream -m eegnet --debug
#
#    # 4. Same check for the foundation model. The first build pulls REVE's
#    #    weights from the HuggingFace Hub, which needs network access; doing
#    #    it here rather than in a queued run keeps any failure in your
#    #    terminal instead of a job log.
#    neuralbench eeg _sleep_onset_stream -m reve --debug
#
#    # 5. Full baseline -- task-specific model (EEGNet). The default grid is
#    #    three seeds (concurrent on SLURM).
#    neuralbench eeg _sleep_onset_stream -m eegnet
#
#    # 6. Full baseline -- foundation model (REVE), fine-tuned end to end.
#    #    ~69M parameters against EEGNet's ~1.5k, all of them trainable here,
#    #    so this one wants a datacentre GPU rather than a laptop; it also
#    #    preprocesses at 200 Hz against the 120 Hz default, warming a second
#    #    cache.
#    neuralbench eeg _sleep_onset_stream -m reve
#
# :ref:`pretrained-weights` covers the hub cache, and no run has a CPU
# fallback -- ``--debug`` included.
#
# Steps 5 and 6 cache the test-metric dictionary under ``SAVE_DIR`` --
# ``test/bmae`` is the headline number -- and re-running the same command with
# ``--plot-cached`` turns those cached metrics into comparison plots and CSV
# tables without retraining. On SLURM they return as soon as the grid is
# queued, so the numbers appear in the job logs rather than your terminal; see
# :ref:`reading-results`.

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
#    print(check_model(my_model, "eeg", "_sleep_onset_stream"))  # shapes only, seconds
#    scores = evaluate_model(my_model, "eeg", "_sleep_onset_stream", name="my-fm", debug=True)
#
# See :doc:`Evaluating your own model
# </neuralbench/auto_examples/quickstart/03_evaluate_your_own_model>` for what
# ``forward`` has to accept, the adaptation presets, and how to fan the runs
# out to SLURM.

# %%
# Where the competition data diverges
# ------------------------------------
#
# The competition's own shift is cross-user on a single device. The default
# Muse data come from that device, home protocol and target, so on them only
# the split (every sleeper held out, see `Split and model selection`_) and
# the unweighted metric differ from what you will be scored on. Prior
# filters, exact hardware generation and N2 scoring methodology are
# undocumented. An existing copy can be placed directly at
# ``DATA_DIR/Interaxon2026Muse``; downloaded copies live under
# ``Interaxon2026Muse/download/nm000287``. Cite Muse Team, *Muse Sleep-Onset
# EEG*, version 1.0.0, DOI ``10.82901/nemar.nm000287``, and retain the
# CC-BY-NC-SA-4.0 attribution.
#
# The three PSG corpora registered for this task are exactly the three public
# datasets the competition lists for Track 3, and all use the same
# ``AddSleepOnsetTargets`` + ``bmae`` pipeline as the default:
#
# .. code-block:: bash
#
#    # Sleep-EDF Expanded, the Codabench warm-up set: ~7 GB raw (about 4 min
#    # to download on a fast link) + ~18 GB cache (~15 min to prepare over
#    # 20 SLURM jobs), then ~6 min per EEGNet seed and ~8 min per REVE seed
#    neuralbench eeg _sleep_onset_stream --dataset kemp2000analysis
#
#    # PhysioNet/CinC Challenge 2018: 994 labelled subjects, by far the most
#    # sleepers, providing a larger cross-user proxy than the public Muse set.
#    # Also by far the largest: ~285 GB raw
#    # plus ~40 GB of cache.
#    neuralbench eeg _sleep_onset_stream --dataset ghassemi2018you
#
#    # HMC Sleep Staging: 151 clinical whole-night recordings, ~17 GB raw
#    neuralbench eeg _sleep_onset_stream --dataset alvarez2022haaglanden
#
# They add a second, artificial shift on top, because the only public
# sleep-onset data with scored hypnograms is clinical polysomnography. On
# them, three more axes differ from what you will be scored on:
#
# 1. **Hardware**: research-grade PSG vs the consumer-grade Muse headband
#    (4-channel frontotemporal EEG, no EOG). Expect to drop or re-map
#    channels in the dataloader, and treat any number you get there as an
#    upper bound on signal quality.
# 2. **Cohort and recording context**: laboratory monitored sleep vs home
#    recordings with movement artifacts, impedance changes and channel
#    dropout.
# 3. **Annotations**: full hypnograms vs ``n2_onset`` events only on
#    the training set. NeuralBench already trains on
#    ``SleepOnsetMarker`` events, so the model interface does not
#    change.
#
# Submission outputs (per the competition):
#
# - a direct onset estimate ``tau_hat`` in seconds, **or**
# - per-window time-to-onset predictions, **or**
# - per-window sleep probabilities.
#
# The current NeuralBench head produces the second format directly, and
# the first by the conversion above.
