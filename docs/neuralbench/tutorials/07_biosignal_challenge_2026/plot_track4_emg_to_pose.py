"""
Track 4 -- EMG-to-Pose (cross-user and cross-stage regression)
===============================================================

Given 16-channel surface EMG (sEMG) recorded from a wristband, predict
the corresponding trajectory of 20 hand-joint angles. The competition
combines two shifts, so a model has to survive changes in anatomy,
device placement, and hand kinematics at once.

- **Shift**: held-out users, movement stages, and user-stage
  combinations.
- **Headline metric**: mean angular error (lower is better). The task
  trains and logs radians; the competition and the paper report the
  same quantity in degrees.
- **Data**: ``emg2pose`` / NM000281 (193 participants, 25,253
  recordings, 370 hours, 29 movement stages, 2 kHz).
"""

# %%
# NeuralBench mapping
# -------------------
#
# The matching task in NeuralBench is :doc:`/neuralbench/tasks/emg/pose`.
#
# - **CLI**: ``neuralbench emg pose``
# - **Default dataset**: ``Salter2024Emg2pose`` (16-channel sEMG paired
#   with motion-capture hand pose).
# - **Model**: ``VEMG2Pose``, the paper's regression baseline.
# - **Target**: a dense 20-joint angle trajectory for each 5-s window.
# - **Headline metric key**: ``test/mae`` (radians; x57.29578 for
#   degrees).
#
# .. dropdown:: Show ``tasks/emg/pose/config.yaml``
#
#    .. literalinclude:: ../../../../neuralbench-repo/neuralbench/tasks/emg/pose/config.yaml
#       :language: yaml

# %%
# Reproducing the baseline
# ------------------------
#
# emg2pose is served through EEG-Dash, which the base install does not
# pull, so install it first: ``pip install 'eegdash>=0.8.2'``. The full
# release is ~310 GB under ``DATA_DIR``.
#
# .. code-block:: bash
#
#    # 1. Download emg2pose / NM000281
#    neuralbench emg pose -m vemg2pose --download
#
#    # 2. Prepare the preprocessing cache
#    neuralbench emg pose -m vemg2pose --prepare
#
#    # 3. Quick local sanity check
#    neuralbench emg pose -m vemg2pose --debug
#
#    # 4. Full paper regression baseline
#    neuralbench emg pose -m vemg2pose

# %%
# Scope and data handling
# -----------------------
#
# NeuralBench implements the paper's ``regression_vemg2pose`` setting.
# The autoregressive tracking setting, which also conditions on an initial
# pose and previous predictions, is outside this task's scope.
# ``-m neuropose`` and ``-m sensingdynamics`` select the paper's other two
# regression baselines.
#
# The paper split comes from the BIDS ``scans.tsv``, falling back to the
# upstream ``emg2pose_metadata.csv`` on releases whose ``scans.tsv`` omits it.
# ``BAD_IK`` events mark intervals without inverse-kinematics labels, and any
# window overlapping one is dropped from every split; padded recording tails
# are not segmented into training windows either. Joint angles stay in the
# radians emg2pose trains on, so the loss and metrics are radians too.
#
# .. warning::
#
#    emg2pose is released under CC-BY-NC-SA-4.0, and UmeTrack under
#    CC-BY-NC-4.0. Both licenses are non-commercial.
