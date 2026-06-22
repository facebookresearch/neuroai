"""
How to Submit a Model
======================

This page describes how to package and submit a model to the
EEG/EMG Foundation Challenge 2026.

.. warning::
   The submission portal is **not yet open**. This page is a
   placeholder that documents the expected submission workflow.
   It will be updated with concrete instructions and URLs once
   the competition is officially launched.
"""

# %%
# Overview
# --------
#
# Each track accepts a trained model (or, for Track 5, a frozen
# encoder). The submission workflow follows three steps:
#
# 1. **Train** your model locally using NeuralBench or your own
#    pipeline.
# 2. **Package** the model weights and a minimal inference script
#    into the required format.
# 3. **Upload** the package to the competition portal, where
#    organisers run evaluation on the hidden test set.
#
# Tracks 1--4 accept both task-specific models and foundation
# models. Track 5 accepts only a single frozen EEG encoder (see
# :doc:`plot_track5_foundation_transfer`).

# %%
# Important dates (TBD)
# ---------------------
#
# .. list-table::
#    :widths: 40 60
#
#    * - Competition launch
#      - TBD
#    * - Submission portal opens
#      - TBD
#    * - Submission deadline
#      - TBD
#    * - Winners announced
#      - NeurIPS 2026 competition track

# %%
# Next steps
# ----------
#
# While the portal is not yet open, you can already:
#
# - Run the track baselines from the starter kit to familiarise
#   yourself with the tasks and metrics.
# - Register a new model in NeuralBench and iterate on your
#   architecture (see
#   :doc:`/neuralbench/auto_examples/adding_model/create_new_model`).
# - Watch the competition repository for announcements on portal
#   availability and dataset releases.
