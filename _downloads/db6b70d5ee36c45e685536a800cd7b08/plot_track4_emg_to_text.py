"""
Track 4 -- EMG-to-Text (cross-subject keystroke decoding)
===========================================================

Given surface EMG (sEMG) recorded from two wristbands while a user
types prompted text on a physical QWERTY keyboard, decode the
keystroke sequence. The competition tests **cross-subject**
generalisation: train users and test users are disjoint.

- **Shift**: seen users -> unseen users.
- **Headline metric**: corpus-level character error rate (CER), with
  insertion / deletion / substitution rates and rendered-text CER as
  diagnostics. Lower is better.
- **Data**: ``emg2qwerty`` for training and local validation (108
  users, two differential dry-electrode sEMG wristbands, 2 kHz). The
  hidden evaluation set contains 100 new users with the same hardware
  and paradigm.

.. note::
   The official Track 4 evaluation set (100 new users on the same
   hardware) is not public yet, so this page runs on the released
   ``emg2qwerty`` training data as the closest open analog.
"""

# %%
# NeuralBench mapping
# -------------------
#
# The entry point is:
#
# - **CLI**: ``neuralbench emg typing``
# - **Default dataset**: ``Sivakumar2024Emg2qwerty`` (108 users,
#   32-channel EMG, 2 kHz, ~1135 sessions, BIDS-converted from the
#   original HDF5 release).
# - **Target**: keystroke token sequence over an alphabet of keyboard
#   symbols (space, punctuation, backspace).
# - **Headline metric key**: ``test/cer`` (character error rate).
#
# The study itself is already documented:
# :py:class:`~neuralset.studies.Sivakumar2024Emg2qwerty`. It exposes
# three event types: ``Emg`` (raw signal), ``Sentence`` (prompt), and
# ``Keystroke`` (token + timestamp).

# %%
# Reproducing the baseline
# ------------------------
#
# .. code-block:: bash
#
#    # 1. Download emg2qwerty / NM000104 (~239 GB, via eegdash/NEMAR)
#    neuralbench emg typing --download
#
#    # 2. Prepare the preprocessing cache
#    neuralbench emg typing --prepare
#
#    # 3. Quick local sanity check
#    neuralbench emg typing --debug
#
#    # 4. Full baseline -- task-specific model (EMG2QwertyNet)
#    neuralbench emg typing -m emg2qwertynet
#
# Unlike the EEG tracks, the headline metric is sequence-level: the
# model must output a *sequence* of keystrokes, not a per-window
# label. Any sequence-supervised objective is permitted (CTC, RNN-T,
# alignment-supervised, LM-assisted, ...).
