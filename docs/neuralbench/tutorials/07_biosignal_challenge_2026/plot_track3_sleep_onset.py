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
  - *Warm-up phase*: **W-bMAE** on the public Muse data, computed for each
    recording over its non-empty ranges, then averaged over recordings, with
    no seen/unseen split.

  ``_sleep_onset_stream`` logs the warm-up score as
  ``test/wbmae_stream_mean`` (``neuralbench.metrics.BinnedMAE`` with
  ``bin_weights``, per recording); ``test/bmae`` is the unweighted bMAE over
  all windows. Nothing in the start kit computes the seen/unseen
  macro-average.
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
   _sleep_onset_stream``, which uses the supplied session split and selects
   checkpoints on the per-recording W-bMAE; it does not reproduce the sealed
   score.

   The release's session tables place every N2 onset exactly 300 s before
   the recording end, so total recording length is not a legitimate
   predictive input. ``_sleep_onset_stream`` feeds the
   signal at its native rate, unfiltered and unscaled apart from the
   conversion to microvolts, so no whole-recording statistic reaches the
   model; a model config that sets its own preprocessing, such as
   ``reve``'s, overrides this.
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
#   forward in time from the start of the recording, as Codabench does, and
#   its ``test/wbmae_stream_mean`` is the warm-up leaderboard's measurement.
#   A model's optional ``reset_state()`` is called on a fresh copy of the
#   model at the start of each recording (``data.stream_by: [timeline]``,
#   see :doc:`Modifying the training loop
#   </neuralbench/auto_examples/advanced/modify_training_loop>`).
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
# Both versions share the split, target and loss. ``_sleep_onset_stream``
# also feeds the signal unprocessed (see the note above), adds the
# per-recording W-bMAE, defaults to the Muse data, and selects Sleep-EDF with
# ``--dataset kemp2000analysis``:
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
# - **Headline metric key**: ``test/wbmae_stream_mean`` (W-bMAE in seconds,
#   per recording, averaged over recordings).
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
# **Split.** The supplied session split, read by ``PredefinedSplit`` from the
# release's ``split`` column: 500 training and 40 test recordings, all test
# sleepers seen in training. Twenty percent of the subjects without a test
# recording (seed 33) are held out for validation, so every test sleeper
# keeps its training recordings: **170 train / 33 validation subjects**
# (412 / 88 recordings). The PSG variants instead split
# participant-level 60 / 20 / 20 by ``SklearnSplit`` -- on Sleep-EDF's 78
# participants that is **46 train / 16 validation / 16 test**.
#
# Those 40 Muse test recordings *are* the Codabench warm-up evaluation set.
# A ``_sleep_onset_stream`` ``test/wbmae_stream_mean`` and a warm-up
# leaderboard score are therefore the same measurement.
#
# The sealed phase is a different story. Its Muse cohort mixes seen and
# unseen sleepers, while the warm-up test holds no unseen sleeper, so the
# sealed score is not something the starter kit can approximate.
#
# **Model selection.** The checkpoint with the lowest
# **``val/wbmae_stream_mean``** is kept -- the warm-up score, just on the
# validation fold. Training runs for at most 40 epochs and stops early after
# 7 epochs without improvement; only that single best checkpoint is scored on
# test.
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
#    # 2. Prepare CACHE_DIR -- read the 540 recordings and cut them into 5 s
#    #    windows once, so every later run reads the cache instead. No GPU
#    #    needed, and it fans out over SLURM when one is configured.
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
#    #    applies its own preprocessing (200 Hz, filtered, scaled) instead of
#    #    the task's raw signal, warming a second cache.
#    neuralbench eeg _sleep_onset_stream -m reve
#
# :ref:`pretrained-weights` covers the hub cache, and no run has a CPU
# fallback -- ``--debug`` included.
#
# Steps 5 and 6 cache the test-metric dictionary under ``SAVE_DIR`` --
# ``test/wbmae_stream_mean`` is the headline number -- and re-running the
# same command with ``--plot-cached`` turns those cached metrics into comparison plots and CSV
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
# the split (no unseen sleeper in the test, see `Split and model selection`_) and
# the seen/unseen macro-average differ from what you will be scored on. Prior
# filters, exact hardware generation and N2 scoring methodology are
# undocumented. An existing copy can be placed directly at
# ``DATA_DIR/Interaxon2026Muse``; downloaded copies live under
# ``Interaxon2026Muse/download/nm000287``. Cite Muse Team, *Muse Sleep-Onset
# EEG*, version 1.0.0, DOI ``10.82901/nemar.nm000287``, and retain the
# CC-BY-NC-SA-4.0 attribution.
#
# The three PSG corpora registered for this task are exactly the three public
# datasets the competition lists for Track 3, and all use the same
# ``AddSleepOnsetTargets`` pipeline and metrics as the default. Their
# recordings run for hours before N2, so they keep only the last 20 minutes
# before onset, and validation and test streams start at a random time within
# them, so the time since a stream began says little about the target:
#
# .. code-block:: bash
#
#    # Sleep-EDF Expanded: ~7 GB raw (about 4 min
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
