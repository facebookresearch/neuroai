"""
Training a model -- masked prediction on EEG
=================================================

Every track of the challenge accepts two kinds of entry: a
**task-specific model**, trained on that track's task alone, and a
**foundation model**, one network reused across tasks rather than
rebuilt for each. Neither is privileged, and how you obtain a
foundation model is up to you -- unlabelled data, labelled data,
several datasets or one, any objective you like.

This page walks through the smallest version of the foundation-model
route end to end -- pretrain an encoder by masked prediction on
unlabelled EEG, then hand it to ``neuralbench`` -- using
``neuraltrain``'s ``ssl_example`` project. Masked prediction is chosen
here only because it needs no labels, so it can pool datasets that
share nothing but the fact that they are EEG.

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
# 2. Blocks of those tokens are replaced by a learned **mask token**,
#    each block a ``mask_radius`` cap of scalp over ``mask_duration`` of
#    signal, and enough of them are drawn to hide ``mask_ratio`` of the
#    window. The encoder then reads the whole sequence.
# 3. A single **linear layer** reconstructs the hidden patches from the
#    encoder's output, and the loss is the reconstruction error on those
#    patches only.
#
# Hiding caps rather than scattered tokens is what makes step 3 hard: a
# lone hidden patch sits between visible ones on the same channel and on
# its neighbours, so interpolating them is enough to score well on it.
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
# a Fourier embedding of its channel's **3D position on the head**, in
# MNE head-frame metres, alongside the sin-cos embedding of its time
# patch. Two datasets that both record Cz describe it the same way, a
# montage the encoder has never seen is just a set of positions it has
# not visited, and ``mask_radius`` is a distance on the scalp.
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
# Getting the data
# ----------------
#
# Pretraining adds three requirements to a working ``neuralbench``
# install (:doc:`/neuralbench/install`). ``ssl_example`` is a project in
# the repository rather than part of the ``neuraltrain`` wheel, so it
# comes from a clone; the encoder it builds lives behind
# ``neuraltrain``'s ``models`` extra; and ``Stieger2021Continuous``
# reaches its host through ``moabb``, which nothing installs for you.
#
# .. code-block:: bash
#
#    git clone https://github.com/facebookresearch/neuroai
#    cd neuroai
#    pip install './neuraltrain-repo[lightning,models]' 'moabb>=1.7.1'
#
# The example pretrains on four EEG datasets: those behind tracks 1-3
# (``Gifford2022Large``, ``Stieger2021Continuous``,
# ``Kemp2000Analysis``) plus one resting-state dataset that belongs to
# no track (``Miltiadous2023Dice``), together some 240 subjects. Each
# comes from its public source, fetched by the download step below.
#
# ``ssl_example`` keeps its own paths, set by ``DATADIR``, ``CACHEDIR``
# and ``SAVEDIR`` at the top of ``defaults.py``: all three sit under
# ``~/.cache/neuralset`` and are independent of the ``DATA_DIR`` you
# configured for ``neuralbench``. Repoint them before the first run
# unless your home directory can take ~1.1 TB: ``Stieger2021Continuous``
# alone is ~600 GB downloaded plus ~280 GB once MOABB converts it, and
# ``Gifford2022Large`` adds ~210 GB.
#
# Training reads what is already on disk and never fetches, so download
# the corpus first -- once per machine:
#
# .. code-block:: bash
#
#    python -m ssl_example.grids.download
#
# Two slow steps therefore precede the first gradient step:
#
# 1. **Download** each dataset into ``DATADIR`` (the command above).
# 2. **Preprocess and cache** it into ``CACHEDIR``: resampling,
#    filtering and scaling run once per configuration, and every later
#    run and every grid job reads the cache instead of redoing them.
#
# Both steps are cached by ``exca``, keyed on the extractor config, so
# changing the preprocessing invalidates the cache and pays for it
# again. Budget for that first pass, and prefer to warm it once on a
# machine with a good connection.
#
# Because it is a long first step, check the wiring before paying for
# it. The debug config swaps the four datasets for one small recording
# -- MNE's sample dataset, downloaded on first use and already used by
# the ``neuralbench`` quickstart -- and runs a single batch:
#
# .. code-block:: bash
#
#    cd neuraltrain-repo
#    python -m ssl_example.grids.test_run

# %%
# Pretraining the encoder
# ------------------------
#
# With the wiring checked, run the real thing. It downloads and caches
# as described above, then finishes on a line reading ``Pretrained
# encoder: <SAVEDIR>/ssl_example.main.Experiment.run,1/<uid>/encoder.ckpt``.
# The ``<uid>`` is a hash of the config, so copy that path rather than
# reconstruct it:
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
# Three parts of that config are what make it *self-supervised*, and
# they are the parts to keep when you swap in your own data:
#
# - **Windows come from a stride, not from events.** The segmenter
#   triggers on the recording and steps a ``WINDOW``-second window
#   across it every ``WINDOW`` seconds, so the windows tile the
#   recording rather than clustering around stimuli. Set ``stride``
#   below ``duration`` to overlap them instead.
# - **There is no target extractor.** The segmenter has an ``"input"``
#   entry and channel positions, but no target, because the input is its
#   own target.
# - **The split holds out whole subjects.** Striding turns one recording
#   into hundreds of windows that share its subject, session and
#   electrodes, so a split over windows would leave near-duplicates of
#   the training data in validation and report a loss that mostly
#   measures memorisation. ``SklearnSplit(split_by=
#   "subject")`` puts every recording of a subject on one side instead,
#   so the validation loss measures reconstruction of a recording the
#   encoder has never seen -- the same convention the downstream tasks
#   use.
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
# You are not required to use this encoder, this loop, or this
# repository at all -- the example is a starting point, and there are
# three ways past it:
#
# - **Keep the loop, change the encoder's shape.** ``MaeModule`` is
#   written against ``MaeEncoder`` -- it calls that encoder's patching
#   and token methods, and ``brain_model_config`` is typed to it -- so
#   vary its fields (``dim``, ``patch_size``,
#   ``transformer_config.depth``) rather than swapping in another
#   ``neuraltrain`` model.
# - **Add your own model to** ``neuraltrain``. A new architecture is a
#   ``BaseBrainModelConfig`` subclass with a ``build`` method, after
#   which ``neuralbench`` can run it by name, exactly as ``MaeEncoder``
#   is. Pretraining it here also means adapting ``mae_module.py``,
#   which reconstructs the patches ``MaeEncoder`` produces.
# - **Train wherever you like.** Nothing about the competition requires
#   ``neuraltrain``, ``neuralset``, or PyTorch Lightning. Train in your
#   own codebase, with your own data pipeline and objective, and bring
#   only the finished model to ``neuralbench`` through the
#   Bring-Your-Own-Model API described in `Evaluating a model of your
#   own`_.

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
#        --checkpoint <the encoder.ckpt path printed above> \
#        -w linear_probe_mean
#
# ``neuralbench`` builds the encoder with no output head and loads the
# checkpoint into it. ``-w linear_probe_mean`` does the rest: it freezes
# every parameter the encoder has and trains only a **linear probe** on
# the mean-pooled tokens, at the learning rate the benchmark uses for
# its own probes. Freezing is what makes the score a property of the
# pretrained representation rather than of the fine-tuning that would
# otherwise follow, and reusing the benchmark's own preset is what makes
# the number comparable to the published ones. Leave ``-w`` out and the
# encoder is fine-tuned end to end, as for every other model; ``-w
# lora_r4_flatten`` sits in between.
#
# .. literalinclude:: ../../../../neuralbench-repo/neuralbench/models/mae.yaml
#    :language: yaml
#
# Preprocessing needs no attention: ``mae.yaml`` sets none, so both
# sides inherit the benchmark defaults that ``ssl_example`` pretrains
# with. If you *do* change the extractors in ``defaults.py``, mirror
# the change under ``data.neuro`` here.
#
# .. warning::
#    ``mae.yaml`` describes the encoder it expects, and nothing checks
#    that against yours: its ``dim`` and ``patch_size`` must match the
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
# head -- ``neuralbench`` wraps it in a probe sized to each task, and
# defaults to the same frozen-backbone ``linear_probe_mean`` preset that
# ``-w`` selects above, so the two routes produce comparable scores.
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
# - :doc:`Track 2 -- BCI command decoding <plot_track2_eeg_to_bci>`
# - :doc:`Track 3 -- Sleep onset <plot_track3_sleep_onset>`
# - :doc:`Track 4 -- EMG-to-Pose <plot_track4_emg_to_pose>`
# - :doc:`How to Submit a Model <plot_submission_guide>`
