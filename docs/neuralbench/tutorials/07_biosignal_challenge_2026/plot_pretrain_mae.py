"""
Training a model -- masked prediction on EEG
=================================================

Every track of the challenge accepts a **foundation model**: one network
pretrained once on unlabelled data, then evaluated on a downstream task
without being redesigned for it. This page walks through the smallest
version of that story end to end -- pretrain an encoder by masked
prediction on unlabelled EEG, then hand it to ``neuralbench`` -- using
``neuraltrain``'s ``ssl_example`` project.

The point is the *workflow*, not the score. The example is deliberately
small, and a competitive entry will need a bigger encoder and more data
than it ships with.

.. note::
   Already have a model of your own? Skip to `Evaluating a model of your
   own`_. Continue with :doc:`How to Submit a Model
   <plot_submission_guide>` once you have a checkpoint, and see the
   per-track pages for the downstream task each track scores.
"""

# %%
# What masked prediction does
# ---------------------------
#
# The model learns by hiding part of its input and reconstructing it:
#
# 1. Each window of EEG is cut into **time patches** of ``patch_size``
#    samples, one channel at a time, so a token is one channel over one
#    patch rather than all channels at once.
# 2. A random ``mask_ratio`` of those tokens is replaced by a learned
#    **mask token**, and the encoder reads the whole sequence.
# 3. A single **linear layer** reconstructs the hidden patches from the
#    encoder's output, and the loss is the reconstruction error on those
#    patches only.
#
# Nothing in that loop uses labels or events, so the training signal
# comes from the recording itself -- which is what lets pretraining use
# far more data than any single labelled task can offer. Only the
# **encoder** is kept at the end.
#
# The original `MAE <https://arxiv.org/abs/2111.06377>`_ differs at step
# 2 and 3: it feeds the encoder only the visible patches and restores the
# rest with a transformer **decoder**, which is cheaper per step. The
# example is encoder-only, so building that decoder is left to you and is
# a natural first thing to try.

# %%
# One encoder, many montages
# --------------------------
#
# Pretraining is worth doing across datasets, and EEG datasets rarely
# agree on their channels: the ones below range from a 63-channel cap to
# Sleep-EDF's two bipolar derivations. An encoder whose first layer is
# sized from a channel count cannot span them.
#
# So a channel is never identified by its index here. Every token carries
# a Fourier embedding of its channel's **3D position on the head**,
# alongside the sin-cos embedding of its time patch. Two datasets that
# both record Cz describe it the same way, and a montage the encoder has
# never seen is just a set of positions it has not visited.
#
# That also settles what to do about *missing* channels. Pooling studies
# makes the channel axis the union of every montage, and each recording
# is zero-padded wherever it lacks a channel. Those padded channels come
# back with invalid positions, and their tokens are **dropped from the
# attention** rather than read as signal -- they are also never chosen as
# reconstruction targets, since predicting padding teaches nothing.
#
# The cost is sequence length: one token per channel *and* patch means
# the encoder attends over ``n_channels * n_patches`` positions, so the
# window length and the montage size now both set the compute bill.
#
# The payoff applies downstream too: one checkpoint scores on a task with
# a montage it never saw during pretraining.

# %%
# Setup
# -----
#
# Pretraining needs ``neuraltrain`` (the encoder lives behind the
# ``models`` extra) alongside ``neuralbench``:
#
# .. code-block:: bash
#
#    pip install 'neuraltrain-repo/.[lightning,models]'
#
# See :doc:`/neuralbench/install` for the rest of the configuration
# (data, cache, and result directories).

# %%
# Pretraining the encoder
# ------------------------
#
# The example lives in ``neuraltrain-repo/ssl_example``. It pretrains on
# the EEG datasets behind tracks 1-3 plus one resting-state dataset that
# belongs to no track. Those are large, so check the wiring first with
# the debug config, which swaps them for one small bundled recording and
# runs a single batch:
#
# .. code-block:: bash
#
#    cd neuraltrain-repo
#    python -m ssl_example.grids.test_run
#
# Then run the real thing, which downloads the four datasets on first
# use and prints the path of the pretrained encoder when it finishes:
#
# .. code-block:: bash
#
#    python -m ssl_example.grids.defaults
#
# The whole run is driven by one config dictionary:
#
# .. dropdown:: Show ``ssl_example/grids/defaults.py``
#
#    .. literalinclude:: ../../../../neuraltrain-repo/ssl_example/grids/defaults.py
#       :language: python
#
# Four parts of that config are what make it *self-supervised*, and they
# are the parts to keep when you swap in your own data:
#
# - **Several studies are pooled.** ``data.studies`` is a list of
#   studies whose events are concatenated into one training set. A list
#   *within* one entry means something different -- a chain of steps, each
#   feeding the next -- which is why each study sits in its own.
# - **Windows come from a stride, not from events.** The segmenter
#   triggers on the recording (``"type == 'Eeg'"``) and slides a window
#   across it every ``WINDOW`` seconds, so every sample of the
#   recording is used rather than only the moments around a stimulus.
# - **There is no target extractor.** The segmenter has an ``"input"``
#   entry and channel positions, but no target, because the input is its
#   own target.
# - **The split is in time.** Striding turns one recording into many
#   correlated windows, so the tail of each recording is held out for
#   validation instead of splitting over events or subjects.
#
# The knob that matters most for pretraining quality is ``mask_ratio``:
# hide too little and reconstruction becomes trivial copying.
# ``ssl_example/grids/run_grid.py`` sweeps it on SLURM, and the run works
# unchanged on several GPUs. Logging goes to CSV and, if you leave
# ``wandb_config`` set, to Weights & Biases; set it to ``None`` to train
# without it.

# %%
# Scaling it up
# -------------
#
# To turn the example into a real pretraining run, change the config
# rather than the code:
#
# - **More data**: add any study from the :doc:`NeuralFetch catalog
#   </neuralfetch/index>` to ``STUDIES``. Unlabelled EEG is the one
#   resource pretraining scales with, so this matters more than any
#   architecture choice, and a new montage needs no code change.
# - **A bigger encoder**: raise ``brain_model_config.dim`` and
#   ``transformer_config.depth``. Copy any change to ``dim`` or
#   ``patch_size`` into ``mae.yaml`` as well -- see the warning below.
# - **Longer training**: raise ``n_epochs`` and ``patience``, and run on
#   SLURM through ``run_grid.py``.
#
# You are not required to use this encoder at all -- it is a starting
# point. Any ``neuraltrain`` model config works with the same
# ``MaeModule`` loop, and any pretraining objective works if it produces
# an encoder checkpoint.

# %%
# Evaluating the pretrained encoder
# ----------------------------------
#
# Pretraining is only worth as much as the representations it leaves
# behind, so the next step is to score the encoder on a downstream task.
# ``neuralbench`` ships an ``mae`` model config that rebuilds this
# encoder, and ``--checkpoint`` points it at your weights:
#
# .. code-block:: bash
#
#    neuralbench eeg motor_imagery -m mae \
#        --checkpoint <savedir>/encoder.ckpt
#
# ``neuralbench`` builds the encoder with no output head, loads the
# checkpoint into it, freezes it, and trains only a **linear probe** on
# the mean-pooled tokens. Training nothing but the probe is what makes
# the score a measure of the representations rather than of the probe:
#
# .. literalinclude:: ../../../../neuralbench-repo/neuralbench/models/mae.yaml
#    :language: yaml
#
# .. warning::
#    ``mae.yaml`` describes the encoder it expects, and nothing checks
#    that against yours. Its preprocessing must match the pretraining
#    extractors, and its ``dim`` and ``patch_size`` must match the
#    encoder you pretrained. On a mismatch ``neuralbench`` logs ``Size
#    mismatch`` and **keeps the randomly initialised layer** -- which
#    reads as a failed pretraining run rather than a misconfiguration.
#    Check the log before trusting a score. Channel count is the one
#    thing you never have to match, since the encoder reads the montage
#    off the channel positions.
#
# The example pretrains on the datasets of tracks 1-3, so a probe scored
# on one of those tasks has seen that data unlabelled already. That is
# allowed, but it tells you nothing about generalising to a dataset the
# encoder has never met -- use a task built on another dataset for that.
#
# To confirm the pretraining actually bought you something, compare
# against the same architecture with no checkpoint (drop
# ``--checkpoint``) and against the task-specific baselines on the
# track pages.

# %%
# Evaluating a model of your own
# ------------------------------
#
# ``mae.yaml`` works because the encoder lives in this repo. A model that
# lives in your own script has no YAML here and needs none:
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
# One instance serves every task in the selection, so the model must
# accept any channel count and any window length, and take channel
# identity from a ``channel_positions`` argument to ``forward`` rather
# than from a montage fixed at construction. It needs no classifier
# head -- ``neuralbench`` wraps it in a probe sized to each task, the
# same frozen-backbone linear probe ``mae.yaml`` configures, so the two
# routes produce comparable scores.
#
# The encoder above meets those requirements, so either route works for
# it. The YAML route additionally pins the preprocessing a checkpoint was
# trained with, which is why the example uses it.
#
# Run :func:`~neuralbench.check_model` before you queue anything: it
# pushes synthetic batches of the selection's shapes through the model
# and reads only YAML, so a shape bug surfaces in seconds rather than an
# hour into a real run. Then start with ``debug=True``, which runs
# locally on two epochs and a data subset.
#
# See :doc:`Evaluating your own model
# </neuralbench/auto_examples/quickstart/03_evaluate_your_own_model>`
# for suites, running on SLURM, and changing the protocol.

# %%
# Next steps
# ----------
#
# - :doc:`Track 1 -- EEG-to-Image <plot_track1_eeg_to_image>`
# - :doc:`Track 2 -- EEG-to-BCI <plot_track2_eeg_to_bci>`
# - :doc:`Track 3 -- Sleep onset <plot_track3_sleep_onset>`
# - :doc:`Track 4 -- EMG-to-Text <plot_track4_emg_to_text>`
# - :doc:`How to Submit a Model <plot_submission_guide>`
