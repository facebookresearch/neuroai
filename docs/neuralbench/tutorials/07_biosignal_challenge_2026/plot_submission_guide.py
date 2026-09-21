"""
How to Submit a Model
======================

Codabench owns the submission contract and documents it in full. This page
covers only the part that is ours: turning a trained NeuralBench run into
the weights and wrapper that Codabench expects.

Read the track's Codabench **Get Started -> Submission Guide** first. It is
authoritative, it is kept current, and anything below that disagrees with
it is out of date.
"""

# %%
# New to NeuralBench? Start here
# ------------------------------
#
# - :doc:`Challenge overview <plot_overview>` -- what NeuralBench is, how it
#   relates to the competition, and the baseline numbers for all four tracks.
# - :doc:`Installation </neuralbench/install>` and the :doc:`quickstart
#   </neuralbench/auto_examples/quickstart/01_run_first_task>` -- get a task
#   running on a 1.5 GB dataset before you download anything large.
# - `Official rules and track guides
#   <https://neural-interfaces26.github.io/tracks.html>`__ -- registration,
#   data access, prizes, leaderboard. Authoritative on every competition
#   matter; these pages only cover the code.
# - The track page you are submitting to: :doc:`Track 1 -- EEG-to-Image
#   <plot_track1_eeg_to_image>`, :doc:`Track 2 -- BCI decoding
#   <plot_track2_eeg_to_bci>`, :doc:`Track 3 -- Sleep onset
#   <plot_track3_sleep_onset>`, :doc:`Track 4 -- EMG-to-Pose
#   <plot_track4_emg_to_pose>`.

# %%
# Where the instructions live
# ---------------------------
#
# Registration, the ``submission.py`` contract, local testing with benchopt,
# the phase timeline and the submission limits are all documented per track
# on Codabench, under **Get Started** and **Track description**:
#
# .. list-table::
#    :header-rows: 1
#    :widths: 45 55
#
#    * - Track
#      - Codabench competition
#    * - Track 1 -- EEG-to-Image
#      - `competitions/17974 <https://www.codabench.org/competitions/17974/>`__
#    * - Track 2 -- BCI decoding
#      - `competitions/17982 <https://www.codabench.org/competitions/17982/>`__
#    * - Track 3 -- Sleep onset
#      - `competitions/17983 <https://www.codabench.org/competitions/17983/>`__
#    * - Track 4 -- EMG-to-Pose
#      - `competitions/17984 <https://www.codabench.org/competitions/17984/>`__
#
# Two things from those pages shape what you build here, so they are worth
# knowing before you train anything:
#
# - **Evaluation is inference-only, and the worker installs nothing.** Your
#   ZIP ships a ``submission.py`` defining ``class Solver(CompetSolver)``
#   plus weight files, and ``submission.py`` may import only what the worker
#   image already carries. Extra Python modules of your own are not
#   supported, so the architecture and its preprocessing have to live in
#   that one file.
# - **The warm-up and sealed phases are not the same task** on Tracks 1-3 --
#   the data, and for some tracks the number of classes or the metric
#   weighting, change when the 2026 corpora land. The **Track description**
#   tab states the current warm-up specification and the sealed one side by
#   side.

# %%
# From a NeuralBench run to a submission
# --------------------------------------
#
# NeuralBench is a preparation environment, not part of the submission: the
# evaluation image never imports it. What crosses the boundary is the
# trained weights plus enough code to rebuild the architecture.
#
# A run checkpoints its best epoch under the track's validation metric (see
# the *Split and model selection* section on each track page) to
# ``best.ckpt`` in the run's ``SAVE_DIR`` folder, with
# ``save_weights_only=True``.
#
# .. important::
#    That file is **deleted once the test phase has read it**, so a run
#    driven from the CLI leaves metrics but no weights. To keep them, set
#    ``Experiment.delete_checkpoints_on_exit = False``, which means driving
#    the experiment from Python -- there is no CLI flag. The field is part
#    of the cache uid, so flipping it gives a fresh run rather than reusing
#    a cached one.
#
# With the checkpoint in hand, take the state dict from its ``"state_dict"``
# key and strip the prefix the ``pl_module`` added, so the tensors load into
# a bare model:
#
# .. code-block:: python
#
#    import torch
#
#    ckpt = torch.load("best.ckpt", map_location="cpu")
#    state = {k.removeprefix("model."): v
#             for k, v in ckpt["state_dict"].items()
#             if k.startswith("model.")}
#    torch.save(state, "weights.pt")
#
# Check the prefixes in your own checkpoint before trusting that filter --
# they depend on the model wrapper the task used. Then rebuild the same
# architecture inside ``load_model`` and ship ``weights.pt`` alongside
# ``submission.py``.
#
# Three things differ between what a NeuralBench run produces and what
# Codabench scores:
#
# - **Units, on Track 4 only.** The task trains on radians, as emg2pose
#   does; Codabench expects degrees. Convert (``x 57.29578``) in
#   ``predict``.
# - **Preprocessing.** The EEG tasks resample to 120 Hz and apply a
#   0.1-75 Hz bandpass, a 50/60 Hz notch, a ``RobustScaler`` and a clamp at
#   20 before the model sees anything; ``emg pose`` deliberately does none
#   of it and feeds raw 2 kHz. Codabench passes windows at whatever
#   ``meta["sfreq"]`` reports and expects model-specific preprocessing to
#   live in ``submission.py``, so a model lifted from a run needs its
#   config's chain reproduced there.
# - **Portability.** Write the wrapper against ``meta`` and the batches
#   alone. A model that reads a dataset name, a file path, a subject id or
#   a hard-coded channel count can pass warm-up and break on the sealed
#   cohort.

# %%
# Will a starter-kit score match the leaderboard?
# -----------------------------------------------
#
# During warm-up, on three of the four tracks, close to it: Codabench is
# currently scoring the same public test partitions these task configs
# produce. Each track page says so in its *Split and model selection*
# section, and where the local metric and the warm-up metric diverge it
# says that too.
#
# The sealed phase is a different cohort and, for Tracks 1-3, a different
# task, so nothing here predicts it. Warm-up data is public either way, so
# treat those scores as a pipeline check rather than a ranking.
