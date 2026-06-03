"""
Track 1 -- EEG-to-Image (cross-stimulus retrieval)
====================================================

Given EEG recorded while a participant views a natural image, decode
the image identity. The competition tests **cross-stimulus**
generalisation: training images and test images do not overlap.

- **Shift**: seen images -> unseen images.
- **Headline metric**: Top-5 retrieval accuracy in a frozen DINOv2
  embedding space (higher is better).
- **Data**: THINGS-EEG1 + THINGS-EEG2 + Alljoined-1 + Alljoined-1.6M
  (88 subjects, research- and consumer-grade hardware). The hidden
  evaluation set is provided by Alljoined and uses Emotiv hardware.
"""

# %%
# NeuralBench mapping
# -------------------
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
# - **Headline metric key**: ``test/batch_top5_acc`` (Top-5 retrieval).
#
# .. dropdown:: Show ``tasks/eeg/image/config.yaml``
#
#    .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/eeg/image/config.yaml
#       :language: yaml

# %%
# Reproducing the baseline
# ------------------------
#
# .. code-block:: bash
#
#    # 1. Download THINGS-EEG2
#    neuralbench eeg image --download
#
#    # 2. Build the preprocessing + DINOv2 target cache
#    neuralbench eeg image --prepare
#
#    # 3. Quick local sanity check (2 epochs, subsampled)
#    neuralbench eeg image --debug
#
#    # 4. Full baseline -- task-specific model (EEGNet)
#    neuralbench eeg image -m eegnet
#
# Step 2 is the most expensive: it computes one DINOv2 embedding per
# unique stimulus and caches it under ``CACHE_DIR``.

# %%
# Where the competition data diverges
# ------------------------------------
#
# The starter kit ships four relevant datasets you can use to develop
# and evaluate your model, while the competition itself evaluates on a
# *hidden* Alljoined Emotiv test set. The four available training
# sources are ``Gifford2022Large`` (THINGS-EEG2, the default),
# ``Grootswagers2022Human`` (THINGS-EEG1), ``Xu2024Alljoined``
# (Alljoined-1) and ``Xu2025Alljoined`` (Alljoined-1.6M). They give
# directionally correct baselines but not the exact competition numbers
# (the hidden Emotiv test set is not public).
#
# The three non-default sources are registered under
# ``tasks/eeg/image/datasets/`` and can be selected with ``--dataset``:
#
# .. code-block:: bash
#
#    neuralbench eeg image --dataset grootswagers2022human
#    neuralbench eeg image --dataset xu2024alljoined
#    neuralbench eeg image --dataset xu2025alljoined
