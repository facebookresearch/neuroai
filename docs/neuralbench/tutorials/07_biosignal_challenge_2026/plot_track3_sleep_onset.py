"""
Track 3 -- Sleep onset (cross-user latency prediction)
=======================================================

.. image:: https://neural-interfaces26.github.io/exports/sleep-onset.gif
   :alt: Seconds to first stable N2 predicted from wearable EEG
   :target: https://neural-interfaces26.github.io/tracks.html
   :width: 100%

Given continuous four-channel wearable EEG recorded at home, predict the
seconds remaining until the first stable N2 epoch. The competition tests
**cross-user** generalisation: training and evaluation use the same Muse
headband, the same home protocol and the same target, but the sleepers in
the evaluation cohort are never seen in training. Precise onset timing
replaces full hypnogram reconstruction because a sparse wearable montage
supports it poorly.

- **Shift**: seen sleepers -> unseen sleepers, on one consumer device.
  Night-to-night variation, motion artifacts, impedance changes and channel
  dropout come with the home setting.
- **Headline metric**: ``bMAE`` in seconds -- onset error averaged with
  equal weight over four time-to-onset ranges, so long and short
  latencies count the same (lower is better). The competition additionally
  reports tolerance rates within 30 / 60 / 300 s; the starter kit logs
  ``bMAE`` and the usual regression metrics, not those rates.
- **Data**: continuous Muse wearable EEG, ~1000 training subjects with
  ``n2_onset`` annotations, and a separate hidden evaluation cohort of the
  same order of magnitude recorded with the same hardware and protocol.
  The onset that ``AddSleepOnsetTargets`` extracts is the earliest
  annotated N2 event of the recording; it applies no persistence or
  non-Wake-continuity rule, so check the competition's definition of
  *stable* N2 before reshaping the target to match it.

.. note::
   The Muse training set is released through NeuralBench when
   submissions open. Until then, this starter kit runs on polysomnography
   datasets -- the data format, the target extractor and the metric are
   identical, but the recording hardware is not, so the starter kit has a
   device gap the competition itself does not (see `Where the competition
   data diverges`_).
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
# **What the config is.** A NeuralBench task is one ``config.yaml``, and
# nothing else: a YAML overlay on ``neuralbench/defaults/config.yaml`` naming
# the study to load, how to split it, what the target is, the loss, and the
# metrics. Reading it is the fastest way to know exactly what the baseline
# does.
#
# .. dropdown:: Show ``tasks/eeg/sleep_onset/config.yaml``
#
#    .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/sleep_onset/config.yaml
#       :language: yaml
#
# **How to change it**, in increasing order of effort:
#
# - ``--dataset <name>`` merges
#   ``tasks/eeg/sleep_onset/datasets/<name>.yaml`` over the base config.
#   That is how the two extra PSG corpora below are selected, and how the
#   Muse corpus will be once it ships.
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
#    # 1. Download Sleep-EDF into DATA_DIR: ~8 GB, minutes rather than hours.
#    #    One-off per machine, and safe to interrupt and re-run.
#    neuralbench eeg sleep_onset --download
#
#    # 2. Preprocess into CACHE_DIR -- resample, filter, scale, and cut the
#    #    153 whole-night recordings into 5 s windows once, so every later run
#    #    reads the cache instead. No GPU needed. ~15 min for eegnet's cache
#    #    and ~9 min for reve's, spread over 20 SLURM jobs. Note the cache
#    #    (~18 GB) is larger than the raw download.
#    neuralbench eeg sleep_onset --prepare
#
#    # 3. Sanity check before you queue anything: 2 epochs, a data subset, one
#    #    seed, always in-process. ~1 min on one V100 with the cache warm.
#    neuralbench eeg sleep_onset --debug
#
#    # 4. Full baseline -- task-specific model (EEGNet). ~6 min per seed, and
#    #    the default grid is three seeds (concurrent on SLURM).
#    neuralbench eeg sleep_onset -m eegnet
#
#    # 5. Full baseline -- foundation model (REVE), fine-tuned end to end.
#    #    ~8 min per seed.
#    neuralbench eeg sleep_onset -m reve
#
# Steps 4 and 5 print the test-metric dictionary at the end -- ``test/bmae``
# is the headline number -- and cache it under ``SAVE_DIR``; re-running the
# same command with ``--plot-cached`` turns those cached metrics into
# comparison plots and CSV tables without retraining.

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
#    print(check_model(my_model, "eeg", "sleep_onset"))  # shapes only, seconds
#    scores = evaluate_model(my_model, "eeg", "sleep_onset", name="my-fm", debug=True)
#
# See :doc:`Evaluating your own model
# </neuralbench/auto_examples/quickstart/03_evaluate_your_own_model>` for what
# ``forward`` has to accept, the adaptation presets, and how to fan the runs
# out to SLURM.

# %%
# Where the competition data diverges
# ------------------------------------
#
# The competition's own shift is cross-user on a single device. The starter
# kit adds a second, artificial shift on top, because the only public
# sleep-onset data with scored hypnograms is clinical polysomnography. Three
# axes therefore differ from what you will be scored on:
#
# 1. **Hardware**: research-grade PSG (Sleep-EDF and the two corpora below)
#    vs the consumer-grade Muse headband (4-channel frontal EEG, no EOG).
#    Expect to drop or re-map channels in the dataloader, and treat any
#    number you get here as an upper bound on signal quality.
# 2. **Cohort and recording context**: laboratory monitored sleep vs home
#    recordings with movement artifacts, impedance changes and channel
#    dropout.
# 3. **Annotations**: full hypnograms vs ``n2_onset`` events only on
#    the training set. NeuralBench already trains on
#    ``SleepOnsetMarker`` events, so the model interface does not
#    change.
#
# The three PSG corpora registered for this task are exactly the three public
# datasets the competition lists for Track 3, and all use the same
# ``AddSleepOnsetTargets`` + ``bmae`` pipeline as the default:
#
# .. code-block:: bash
#
#    # Sleep-EDF Expanded (the default), ~8 GB raw + ~18 GB cache
#    neuralbench eeg sleep_onset
#
#    # PhysioNet/CinC Challenge 2018: 994 labelled subjects, by far the most
#    # sleepers, so the best stress test of cross-user behaviour at the scale
#    # the Muse training set will have. Also by far the largest: ~285 GB raw
#    # plus ~40 GB of cache.
#    neuralbench eeg sleep_onset --dataset ghassemi2018you
#
#    # HMC Sleep Staging: 151 clinical whole-night recordings, ~17 GB raw
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
