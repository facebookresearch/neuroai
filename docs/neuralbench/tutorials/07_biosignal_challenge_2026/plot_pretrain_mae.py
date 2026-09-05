"""
Training a model -- masked autoencoding on EEG
=================================================

Every track of the challenge accepts a **foundation model**: one network
pretrained once on unlabelled data, then evaluated on a downstream task
without being redesigned for it. This page walks through the smallest
version of that story end to end -- pretrain a masked autoencoder (MAE)
on unlabelled EEG, then hand the encoder to ``neuralbench`` -- using
``neuraltrain``'s ``ssl_example`` project.

The point is the *workflow*, not the score: the example runs on the
one-subject MNE sample dataset in a couple of minutes on a laptop, which
is far too little data to learn useful representations. Once the
pipeline runs, swap in a larger study and a bigger encoder.

.. note::
   Continue with :doc:`How to Submit a Model <plot_submission_guide>`
   once you have a checkpoint, and see the per-track pages for the
   downstream task each track scores.
"""

# %%
# What masked autoencoding does
# ------------------------------
#
# An MAE learns by hiding part of its input and reconstructing it:
#
# 1. Each window of EEG is cut into **time patches** of ``patch_size``
#    samples, and every patch becomes one token.
# 2. A random ``mask_ratio`` of those tokens is **dropped**, and the
#    encoder only ever sees the ones that survive.
# 3. A throwaway **decoder** is asked to reconstruct the dropped patches
#    from the encoded ones, and the loss is the reconstruction error on
#    the dropped patches only.
#
# Nothing in that loop uses labels or events, so the training signal
# comes from the recording itself -- which is what lets pretraining use
# far more data than any single labelled task can offer. The decoder is
# scaffolding and is thrown away at the end; the **encoder** is the
# artifact worth keeping.

# %%
# Setup
# -----
#
# Pretraining needs ``neuraltrain`` (the MAE encoder lives behind the
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
# The example lives in ``neuraltrain-repo/ssl_example``. Run it with:
#
# .. code-block:: bash
#
#    cd neuraltrain-repo
#    python -m ssl_example.grids.defaults
#
# It prints the path of the pretrained encoder when it finishes. The
# whole run is driven by one config dictionary:
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
#   triggers on the recording (``"type == 'Eeg'"``) and slides a window
#   across it every ``WINDOW`` seconds, so every sample of the
#   recording is used rather than only the moments around a stimulus.
# - **There is no target extractor.** The segmenter has an ``"input"``
#   entry and nothing else, because the input is its own target.
# - **The split is in time.** Striding turns one recording into many
#   correlated windows, so the tail of each recording is held out for
#   validation instead of splitting over events or subjects.
#
# The knob that matters most for pretraining quality is ``mask_ratio``:
# hide too little and reconstruction becomes trivial copying.
# ``ssl_example/grids/run_grid.py`` sweeps it on SLURM.

# %%
# Scaling it up
# -------------
#
# To turn the example into a real pretraining run, change the config
# rather than the code:
#
# - **More data**: replace ``Mne2013SampleEeg`` in ``data.study`` with
#   any study from the :doc:`NeuralFetch catalog </neuralfetch/index>`, or a list
#   of them. Unlabelled EEG is the one resource pretraining scales
#   with, so this matters more than any architecture choice.
# - **A bigger encoder**: raise ``brain_model_config.dim`` and
#   ``transformer_config.depth``. Copy any change to ``dim`` or
#   ``patch_size`` into ``mae.yaml`` as well -- see the warning below.
# - **Longer training**: raise ``n_epochs`` and ``patience``, and run on
#   SLURM through ``run_grid.py``.
#
# You are not required to use this MAE at all -- it is a starting point.
# Any ``neuraltrain`` model config works with the same
# :class:`~neuraltrain.mae_module.MaeModule` loop, and any pretraining
# objective works if it produces an encoder checkpoint.

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
#    neuralbench eeg audiovisual_stimulus -m mae \
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
#    extractor, its ``dim`` and ``patch_size`` must match the encoder
#    you pretrained, and the downstream task must have the same channel
#    count, since the first layer has shape ``dim x (n_channels *
#    patch_size)``. On a mismatch ``neuralbench`` logs ``Size mismatch``
#    and **keeps the randomly initialised layer** -- which reads as a
#    failed pretraining run rather than a misconfiguration. Check the
#    log before trusting a score.
#
# ``audiovisual_stimulus`` is the quickest task to try because it uses
# the same MNE sample recording as the example above -- which also means
# the probe is scored on data the encoder pretrained on. Treat the
# number as a smoke test, and move to a task with held-out data before
# reading anything into it.
#
# To confirm the pretraining actually bought you something, compare
# against the same architecture with no checkpoint (drop
# ``--checkpoint``) and against the task-specific baselines on the
# track pages.

# %%
# Next steps
# ----------
#
# - :doc:`Track 1 -- EEG-to-Image <plot_track1_eeg_to_image>`
# - :doc:`Track 2 -- EEG-to-BCI <plot_track2_eeg_to_bci>`
# - :doc:`Track 3 -- Sleep onset <plot_track3_sleep_onset>`
# - :doc:`Track 4 -- EMG-to-Text <plot_track4_emg_to_text>`
# - :doc:`How to Submit a Model <plot_submission_guide>`
