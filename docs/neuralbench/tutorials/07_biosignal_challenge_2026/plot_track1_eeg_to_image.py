"""
Track 1 -- EEG-to-Image (cross-stimulus retrieval)
====================================================

.. image:: https://neural-interfaces26.github.io/exports/eeg-to-image.gif
   :alt: EEG epochs ranked against a held-out image gallery in DINOv2 space
   :target: https://neural-interfaces26.github.io/tracks.html
   :width: 100%

Given EEG recorded while a participant views a natural image, decode
the image identity. The competition tests **cross-stimulus**
generalisation: training images and test images do not overlap, so the
track probes whether neural representations transfer beyond memorised
concepts.

- **Shift**: seen images -> unseen images.
- **Headline metric**: Top-5 retrieval accuracy against the held-out
  candidate set, ranked in a frozen DINOv2-giant embedding space
  (higher is better).
- **Data**: THINGS-EEG1 + THINGS-EEG2 + Alljoined-1 + Alljoined-1.6M
  (88 subjects, research- and consumer-grade hardware). The hidden
  evaluation cohort adds 11 subjects recorded by Alljoined on the same
  32-channel Emotiv hardware and natural-image paradigm as
  Alljoined-1.6M.
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
# - `Official Track 1 guide <https://neural-interfaces26.github.io/tracks.html>`__
#   -- registration, rules, data access, prizes, leaderboard. Authoritative
#   on every competition matter; this page only covers the code.
# - :doc:`How to Submit a Model <plot_submission_guide>`.

# %%
# Where to find this task in NeuralBench
# --------------------------------------
#
# The closest task in NeuralBench is :doc:`/neuralbench/tasks/eeg/image`.
#
# - **CLI**: ``neuralbench eeg image``
# - **Default dataset**: ``Gifford2022Large`` (THINGS-EEG2,
#   10 subjects, 63 channels). This is one of the four datasets the
#   competition uses.
# - **Target**: frozen ``facebook/dinov2-giant`` image embeddings
#   (1536-d), aligned with a CLIP contrastive loss -- the same
#   embedding space the competition scorer uses.
# - **Headline metric key**: ``test/full_retrieval/top5_acc_subject-agg``
#   -- Top-5 accuracy against every candidate in the test set, averaged
#   over subjects. ``val/batch_top5_acc``, which early stopping
#   monitors, ranks within a batch instead, so it reads far higher and
#   is not comparable.
#
# **What the config is.** A NeuralBench task is one ``config.yaml``, and
# nothing else: a YAML overlay on ``neuralbench/defaults/config.yaml`` naming
# the study to load, how to split it, what the target is, the loss, and the
# metrics. Reading it is the fastest way to know exactly what the baseline
# does.
#
# .. dropdown:: Show ``tasks/eeg/image/config.yaml``
#
#    .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/image/config.yaml
#       :language: yaml
#
# **How to change it**, in increasing order of effort:
#
# - ``--dataset <name>`` merges ``tasks/eeg/image/datasets/<name>.yaml`` over
#   the base config. That is how the alternative corpora below are selected,
#   and how the competition corpus will be once it ships.
# - ``-m <model>`` and ``-w <preset>`` swap the architecture and the
#   adaptation strategy (frozen probe, LoRA, full fine-tuning) without
#   touching any file.
# - Anything else -- window length, learning rate, split -- is a config edit.
#   From Python, pass dotted keys to :func:`~neuralbench.evaluate_model`
#   (``overrides={"data.duration": 1.0}``). From a source checkout
#   (``pip install -e``), edit ``config.yaml`` directly, or add your own
#   ``datasets/*.yaml`` variant beside the existing ones and select it with
#   ``--dataset``. Both routes are described in :doc:`Adding a New Task
#   </neuralbench/auto_examples/adding_task/create_new_task>`.

# %%
# Reproducing the baseline
# ------------------------
#
# .. code-block:: bash
#
#    # 1. Download THINGS-EEG2 into DATA_DIR: ~220 GB, hours over a typical
#    #    link. One-off per machine, and safe to interrupt and re-run -- it
#    #    skips files already on disk. Nothing trains in this step.
#    neuralbench eeg image --download
#
#    # 2. Build the caches under CACHE_DIR (~13 GB): the preprocessed windows,
#    #    plus one frozen DINOv2-giant embedding per unique stimulus (~100 MB
#    #    for THINGS-EEG2, content-keyed and shared with the other image
#    #    tasks). The only --prepare of the four tracks that needs a GPU.
#    #    ~15 min for eegnet's cache, ~45 min for reve's, and ~10 min to embed
#    #    the 16740 stimuli, spread over 10 and 128 SLURM jobs respectively.
#    neuralbench eeg image --prepare
#
#    # 3. Sanity check before you queue anything: 2 epochs, a data subset, one
#    #    seed, always in-process. ~2 min on one V100 with the cache warm.
#    neuralbench eeg image --debug
#
#    # 4. Full baseline -- task-specific model (EEGNet). ~2.5 h per seed, and
#    #    the default grid is three seeds (concurrent on SLURM).
#    neuralbench eeg image -m eegnet
#
#    # 5. Full baseline -- foundation model (REVE), fine-tuned end to end.
#    #    ~5.5 h per seed.
#    neuralbench eeg image -m reve
#
# Steps 4 and 5 print the test-metric dictionary at the end and cache it under
# ``SAVE_DIR``; re-running the same command with ``--plot-cached`` turns those
# cached metrics into comparison plots and CSV tables without retraining.

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
#    print(check_model(my_model, "eeg", "image"))  # shapes only, seconds
#    scores = evaluate_model(my_model, "eeg", "image", name="my-fm", debug=True)
#
# See :doc:`Evaluating your own model
# </neuralbench/auto_examples/quickstart/03_evaluate_your_own_model>` for what
# ``forward`` has to accept, the adaptation presets, and how to fan the runs
# out to SLURM.

# %%
# Where the competition data diverges
# ------------------------------------
#
# The starter kit ships four relevant datasets you can use to develop
# and evaluate your model, while the competition itself evaluates on a
# *hidden* Alljoined Emotiv cohort. The four available training
# sources are ``Gifford2022Large`` (THINGS-EEG2, the default),
# ``Grootswagers2022Human`` (THINGS-EEG1), ``Xu2024Alljoined``
# (Alljoined-1) and ``Xu2025Alljoined`` (Alljoined-1.6M). They give
# directionally correct baselines but not the exact competition numbers
# (the hidden Emotiv test set is not public).
#
# ``Xu2025Alljoined`` is the one to train on if you only pick one: the
# evaluation cohort is recorded with the same 32-channel Emotiv hardware and
# the same natural-image protocol.
#
# The three non-default sources are registered under
# ``tasks/eeg/image/datasets/`` and can be selected with ``--dataset``:
#
# .. code-block:: bash
#
#    neuralbench eeg image --dataset grootswagers2022human
#    neuralbench eeg image --dataset xu2024alljoined
#    neuralbench eeg image --dataset xu2025alljoined
