"""
How to Submit a Model
======================

This page describes how to package and submit a model to the
EEG/EMG Foundation Challenge 2026.

Submissions are handled on `Codabench <https://www.codabench.org/>`_,
with a separate registration for each track you enter. The `competition
website <https://neural-interfaces26.github.io/>`_ carries the per-track
Codabench links, the submission window, and the rules, and is the
authoritative source for all three.
"""

# %%
# Overview
# --------
#
# Each track accepts a trained model -- task-specific or a
# foundation model. The submission workflow follows three steps:
#
# 1. **Train** your model locally using NeuralBench or your own
#    pipeline.
# 2. **Package** the model weights and a minimal inference script
#    into the required format.
# 3. **Upload** the package to your track's Codabench competition,
#    where the organisers run evaluation on the hidden test set.
#

# %%
# Important dates
# ---------------
#
# The submission window, the audit, and the award dates live on the
# `competition website <https://neural-interfaces26.github.io/>`_, which
# is the authoritative source for all of them.

# %%
# Next steps
# ----------
#
# Before you submit:
#
# - Run the track baselines from the starter kit to familiarise
#   yourself with the tasks and metrics.
# - Register a new model in NeuralBench and iterate on your
#   architecture (see
#   :doc:`/neuralbench/auto_examples/adding_model/create_new_model`).
# - Watch the competition website for announcements on the remaining
#   dataset releases.
