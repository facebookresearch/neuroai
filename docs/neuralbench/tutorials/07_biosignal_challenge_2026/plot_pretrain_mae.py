"""
Training a model -- masked prediction on EEG
=================================================

Every track accepts two kinds of entry: a **task-specific model**, trained on
that track's task alone, and a **foundation model**, one network reused across
tasks rather than rebuilt for each. Neither is privileged, and how you obtain a
foundation model is up to you -- any data, any objective.

This page is a working template for the foundation-model route: pretrain a
small encoder by masked prediction on unlabelled EEG with ``neuraltrain``'s
``ssl_example`` project, then score it with ``neuralbench``. Masked prediction
is used here only because it needs no labels, so it can pool datasets that
share nothing but being EEG. Expect a baseline rather than a competitive
entry: the encoder is small and the corpus is four datasets.

.. note::
   Already have a model of your own? Skip to `Evaluating a model of your
   own`_. Nothing about the competition requires ``neuraltrain``,
   ``neuralset`` or PyTorch Lightning.
"""

# %%
# 1. Install, and download the corpus
# -----------------------------------
#
# ``ssl_example`` ships in the repository rather than the wheel, its encoder
# sits behind the ``models`` extra, and ``Stieger2021Continuous`` needs
# ``moabb``:
#
# .. code-block:: bash
#
#    git clone https://github.com/facebookresearch/neuroai
#    cd neuroai
#    pip install './neuraltrain-repo[lightning,models]' 'moabb>=1.7.1'
#
# It pretrains on four EEG datasets -- those behind tracks 1-3
# (``Gifford2022Large``, ``Stieger2021Continuous``, ``Kemp2000Analysis``) plus
# resting-state ``Miltiadous2023Dice`` -- some 240 subjects in all.
#
# **Repoint the paths before the first run.** ``DATADIR``, ``CACHEDIR`` and
# ``SAVEDIR`` sit at the top of ``defaults.py``, all three under
# ``~/.cache/neuralset`` and independent of the ``DATA_DIR`` you configured
# for ``neuralbench``. The four datasets want well over 1 TB
# (``Stieger2021Continuous`` ~940 GB, ``Gifford2022Large`` ~220 GB).
#
# Training reads what is on disk and never fetches, so download first -- once
# per machine:
#
# .. code-block:: bash
#
#    python -m ssl_example.grids.download
#
# Preprocessing is then cached into ``CACHEDIR`` on first use and reused by
# every later run and grid job; changing it pays for that pass again.

# %%
# 2. Check the wiring before paying for it
# ----------------------------------------
#
# The debug config swaps the four datasets for MNE's sample recording,
# downloaded on first use, and runs a single batch:
#
# .. code-block:: bash
#
#    cd neuraltrain-repo
#    python -m ssl_example.grids.test_run

# %%
# 3. Pretrain the encoder
# -----------------------
#
# The loop it runs is `MAE <https://arxiv.org/abs/2111.06377>`_ on EEG: each
# window is cut into time patches one channel at a time, so a token is one
# channel over one patch; a random ``mask_ratio`` of those tokens is swapped
# for a learned mask token; and a linear head reconstructs the hidden patches
# from the encoder's output, scored on those patches alone. Only the encoder
# is kept. `MAEEG <https://arxiv.org/abs/2211.02625>`_ applies that objective
# to EEG at this scale, and `ST-EEGFormer
# <https://openreview.net/forum?id=5Xwm8e6vbh>`_ takes it to a foundation
# model -- it won last year's :doc:`edition of this challenge
# </neuralbench/auto_examples/eeg_challenge/plot_eeg_challenge_2025>`, which
# makes it a useful model for where to take the template below.
#
# The real run finishes on a line reading ``Pretrained encoder:
# <SAVEDIR>/ssl_example.main.Experiment.run,1/<uid>/encoder.ckpt``. The
# ``<uid>`` is a hash of the config, so copy that path rather than reconstruct
# it:
#
# .. code-block:: bash
#
#    python -m ssl_example.grids.defaults
#
# .. dropdown:: Show ``ssl_example/grids/defaults.py``
#
#    .. literalinclude:: ../../../../neuraltrain-repo/ssl_example/grids/defaults.py
#       :language: python
#
# Three parts of that config are what make it self-supervised, and the parts
# to keep when you swap in your own data:
#
# - **Windows come from a stride, not from events**, so they tile the
#   recording instead of clustering around stimuli.
# - **There is no target extractor**: the input is its own target.
# - **The split holds out whole subjects**, because striding turns one
#   recording into hundreds of near-duplicate windows and splitting over
#   windows would mostly measure memorisation.
#
# ``mask_ratio`` is the knob that matters most -- hide too little and
# reconstruction becomes trivial copying. ``ssl_example/grids/run_grid.py``
# sweeps it on SLURM, and the run works unchanged on several GPUs.
#
# One design choice explains why a single encoder can span a 63-channel cap
# and Sleep-EDF's two bipolar derivations, and why adding a dataset below
# needs no code: a channel is identified by a Fourier embedding of its **3D
# position on the head**, never by its index, and zero-padded channels are
# dropped from the attention rather than read as signal.

# %%
# 4. Score it on a downstream task
# --------------------------------
#
# Pretraining is worth only what its representations are worth. ``neuralbench``
# ships an ``mae`` model config that rebuilds this encoder, and
# ``--checkpoint`` points it at your weights:
#
# .. code-block:: bash
#
#    neuralbench eeg motor_imagery -m mae \
#        --checkpoint <the encoder.ckpt path printed above> \
#        -w linear_probe_mean
#
# ``-w linear_probe_mean`` freezes the encoder and trains only a **linear
# probe** on the mean-pooled tokens, at the learning rate the benchmark uses
# for its own probes -- which is what makes the score a property of the
# pretrained representation, and comparable to the published numbers. Leave
# ``-w`` out to fine-tune end to end; ``-w lora_r4_flatten`` sits in between.
#
# .. literalinclude:: ../../../../neuralbench-repo/neuralbench/models/mae.yaml
#    :language: yaml
#
# To confirm the pretraining bought you anything, compare against the same
# architecture with no checkpoint (drop ``--checkpoint``) and against the
# task-specific baselines on the track pages. The example pretrains on the
# data of tracks 1-3, so a probe scored there has already seen that data
# unlabelled -- allowed, but silent on generalising to an unseen dataset.
#
# .. warning::
#    ``mae.yaml``'s ``dim`` and ``patch_size`` must match the encoder you
#    pretrained, and nothing checks that they do. On a mismatch
#    ``neuralbench`` logs ``Size mismatch`` and **keeps the randomly
#    initialised layer**, which reads as a failed pretraining run rather than
#    a misconfiguration -- so check the log before trusting a score. Channel
#    count is the one thing you never have to match.

# %%
# 5. Scale it up
# --------------
#
# Change the config rather than the code:
#
# - **More data**: add any study from the :doc:`NeuralFetch catalog
#   </neuralfetch/index>` to ``STUDIES``. Unlabelled EEG is the one resource
#   pretraining scales with, so this matters more than any architecture
#   choice. The competition also points at `EEGDash <https://eegdash.org/>`_
#   for several hundred further EEG corpora.
# - **A bigger encoder**: raise ``brain_model_config.dim`` and
#   ``transformer_config.depth``, mirroring any ``dim`` or ``patch_size``
#   change into ``mae.yaml`` -- see the warning above.
# - **Longer training**: raise ``n_epochs`` and ``patience``, and run on SLURM
#   through ``run_grid.py``.
#
# Beyond that: add your own architecture to ``neuraltrain`` as a
# ``BaseBrainModelConfig`` subclass (which also means adapting
# ``mae_module.py``), or train in your own codebase and bring only the
# finished model back through the API below.

# %%
# Evaluating a model of your own
# ------------------------------
#
# ``mae.yaml`` works because the encoder lives in this repo. A model that
# lives in your own script needs no YAML here:
# :func:`~neuralbench.evaluate_model` takes the built instance.
#
# .. code-block:: python
#
#    from neuralbench import check_model, evaluate_model
#
#    model = MyFoundationModel()   # built and pretrained however you like
#
#    print(check_model(model, "eeg", "motor_imagery"))
#    scores = evaluate_model(model, "eeg", "all", name="my-fm", debug=True)
#
# Run :func:`~neuralbench.check_model` before you queue anything: it pushes
# synthetic batches of the selection's shapes through the model, so a shape
# bug surfaces in seconds rather than an hour into a real run. One instance
# serves every task in the selection, so the model must accept any channel
# count and window length, and read channel identity from a
# ``channel_positions`` argument to ``forward``. It needs no classifier head.
#
# See :doc:`Evaluating your own model
# </neuralbench/auto_examples/quickstart/03_evaluate_your_own_model>` for
# suites, running on SLURM, and changing the adaptation protocol.

# %%
# Next steps
# ----------
#
# - :doc:`Track 1 -- EEG-to-Image <plot_track1_eeg_to_image>`
# - :doc:`Track 2 -- BCI decoding <plot_track2_eeg_to_bci>`
# - :doc:`Track 3 -- Sleep onset <plot_track3_sleep_onset>`
# - :doc:`Track 4 -- EMG-to-Pose <plot_track4_emg_to_pose>`
# - :doc:`How to Submit a Model <plot_submission_guide>`
