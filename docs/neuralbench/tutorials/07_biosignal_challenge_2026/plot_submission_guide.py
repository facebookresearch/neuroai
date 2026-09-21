"""
How to Submit a Model
======================

This page describes how to package and upload a model to the
EEG/EMG Foundation Challenge 2026.

Submissions are handled on `Codabench <https://www.codabench.org/>`_, with
a separate registration for each track you enter. The `competition website
<https://neural-interfaces26.github.io/>`_ and each track's Codabench
*Get Started* and *Track description* tabs are authoritative on every
submission matter; this page summarises them and shows how a
NeuralBench-trained model fits the contract.
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
# Register first
# --------------
#
# You cannot submit to a track until you are registered for it, and
# registration is per track. Each Codabench competition carries the
# procedure under **Get Started -> Registration Guide**, which is
# authoritative and not repeated here:
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
# Two points catch people out, so they are worth knowing before you start:
# every member of a team registers individually, and approval is matched on
# the email address you type into the registration form, which must be the
# one on your Codabench account.

# %%
# What a submission is
# --------------------
#
# A ZIP holding a ``submission.py`` plus whatever weight files your model
# needs, **every file at the root of the archive**. A nested directory is
# the single most common ingestion failure.
#
# .. code-block:: text
#
#    my_submission.zip
#    |-- submission.py   # required: all Python inference code
#    |-- weights.pt      # trained parameters, any name or format
#    +-- ...             # optional non-Python artifacts
#
# Evaluation is inference-only: the model arrives fully trained, and
# Codabench mounts the extracted files read-only. During ``load_model`` and
# ``predict`` a submission must not train, must not download competition
# data, and must not write into its own directory.
#
# .. warning::
#    **Nothing is installed at submission time.** ``submission.py`` may
#    import only what the worker image already carries, pinned in the
#    competition's `requirements.txt
#    <https://github.com/neural-interfaces26/2026-competition/blob/main/requirements.txt>`__.
#    Declaring a benchopt ``requirements`` list does not change the worker
#    image. A package of your own is not an option -- additional Python
#    modules are not supported either, so the architecture and any
#    model-specific preprocessing have to live inside ``submission.py``
#    itself.
#
# ``submission.py`` defines ``class Solver(CompetSolver)``, a `benchopt
# <https://benchopt.github.io>`__ solver. Inside it you write plain
# PyTorch; no benchopt, ``neuralset`` or ``neuralbench`` knowledge is
# needed in the model itself.
#
# - ``load_model(self, meta)`` (**required**) builds your model, loads the
#   weights you shipped from ``meta["submission_dir"]``, places it on
#   ``meta["device"]``, and returns an object exposing ``predict(X)``.
# - ``predict(X)`` receives a torch batch ``X`` of shape ``(B, C, T)``,
#   already on ``meta["device"]``, and returns the track's output.
# - ``fit(self, model, train_loader)`` (optional) trains the model. The
#   server never calls it, but it is how you train locally against the
#   exact competition data and evaluation.
# - ``save_model(self, model, path)`` (optional) writes the trained
#   weights. Implement both and a local training run assembles the
#   submission folder for you -- see `Test locally, then upload`_.
#
# ``meta`` is a plain dict the platform builds and passes in:
# ``submission_dir``, ``device``, ``n_chans``, ``n_times``, ``sfreq``,
# ``ch_names``, ``chs_info``, and the track's output-size key.
#
# .. list-table::
#    :header-rows: 1
#    :widths: 20 26 18 18
#
#    * - Track
#      - ``predict(X)`` returns
#      - Output size
#      - Sealed ranking metric
#    * - 1 -- EEG-to-Image
#      - image embeddings ``(B, D)``
#      - ``meta["n_outputs"]`` = D
#      - top-5 retrieval accuracy
#    * - 2 -- BCI decoding
#      - class index per window ``(B,)``
#      - ``meta["n_classes"]``
#      - balanced accuracy
#    * - 3 -- Sleep onset
#      - seconds to onset ``(B,)``, float
#      - ``meta["n_outputs"]`` = 1
#      - weighted binned MAE
#    * - 4 -- EMG-to-Pose
#      - joint angles ``(B, n_joints, T)``, degrees
#      - ``meta["n_joints"]``
#      - mean angular MAE
#
# Track 4 is the one place the units differ: the NeuralBench task trains on
# radians, as emg2pose does, so a model lifted from a run has to convert
# (``x 57.29578``) before returning predictions.
#
# .. code-block:: python
#
#    import torch
#
#    from benchmark_utils.base_solver import CompetSolver
#
#
#    class Solver(CompetSolver):
#        name = "MyModel"
#
#        def load_model(self, meta):
#            # MyModel has to be defined in this same file.
#            model = MyModel(
#                n_chans=meta["n_chans"], n_times=meta["n_times"],
#            )
#            state = torch.load(
#                meta["submission_dir"] / "weights.pt",
#                map_location=meta["device"],
#                weights_only=True,
#            )
#            model.load_state_dict(state)
#            return model.to(meta["device"]).eval()

# %%
# What changes between the two phases
# -----------------------------------
#
# The submission contract is stable across phases; the data is not, and for
# three of the four tracks neither is the task.
#
# .. list-table::
#    :header-rows: 1
#    :widths: 30 35 35
#
#    * -
#      - Warm-up
#      - Sealed final
#    * - Evaluation data
#      - public proxy corpus
#      - held-out cohort, never released
#    * - Task and ranking metric
#      - public proxy; Tracks 1-3 differ from sealed
#      - the specification in the table above
#    * - ZIP format, ``meta`` keys, tensor contract
#      - as documented
#      - identical, though runtime values differ
#
# Concretely, as of this writing: Track 2 warms up on a **two-class**
# motor-imagery proxy ranked by balanced accuracy pooled over windows,
# where the sealed phase is three-class and averages over
# subject-session-context cells. Track 3 warms up on **unweighted** bMAE
# where the sealed phase applies severity weights and a seen/unseen
# macro-average. Track 4 uses the same task and metric in both phases, on
# different data. Each track's Codabench **Track description** tab is
# authoritative on the current warm-up specification, which changes as the
# 2026 corpora are released.
#
# .. warning::
#    Only the *data* is hidden -- the loading, scoring and model-facing code
#    is public. So one rule keeps a submission valid on data you never see:
#    **use only** ``meta`` **and the batches you are given**. A solver that
#    reads a dataset name, a file path, a subject id, or a hard-coded
#    channel count may work during warm-up and break on the sealed cohort.

# %%
# Going from a NeuralBench run to a submission
# ---------------------------------------------
#
# NeuralBench is a preparation environment, not part of the submission: the
# evaluation image does not import it. What crosses the boundary is the
# trained weights plus enough code to rebuild the architecture.
#
# A run checkpoints the best epoch under the track's validation metric (see
# the *Split and model selection* section on each track page) to
# ``best.ckpt`` in the run's ``SAVE_DIR`` folder, with
# ``save_weights_only=True``.
#
# .. important::
#    That file is **deleted once the test phase has read it**. A run
#    driven from the CLI therefore leaves metrics but no weights. To keep
#    them, set ``Experiment.delete_checkpoints_on_exit = False``, which
#    means driving the experiment from Python -- there is no CLI flag. The
#    field is part of the cache uid, so flipping it gives a fresh run
#    rather than reusing a cached one.
#
# With the checkpoint in hand:
#
# 1. Load it and take the state dict from the ``"state_dict"`` key.
# 2. Strip the Lightning wrapper: the parameter names carry the module
#    prefix the ``pl_module`` added.
# 3. Save the bare tensors next to your ``submission.py`` as, say,
#    ``weights.pt``, and rebuild the same architecture inside
#    ``load_model``.
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
# they depend on the model wrapper the task used.

# %%
# Test locally, then upload
# --------------------------
#
# A failed upload still costs one of the day's submissions, so check the
# contract locally first. The starting kit is the benchopt benchmark
# itself, one directory per track, in the `competition repository
# <https://github.com/neural-interfaces26/2026-competition>`__:
#
# .. code-block:: bash
#
#    benchopt install tracks/<track>   # add --gpu if you need CUDA
#    cp my_submission/submission.py tracks/<track>/solvers/my_submission.py
#    COMPET_SUBMISSION_DIR="$PWD/my_submission" \
#        benchopt run tracks/<track> -d Simulated -s MyModel
#
# ``MyModel`` is your ``Solver.name``. ``COMPET_SUBMISSION_DIR`` is what
# points ``meta["submission_dir"]`` at your weights. ``Simulated`` needs no
# download, so this is a seconds-long contract check -- not a score. Its
# dimensions are smaller than the real task, so a fixed-size checkpoint may
# not even load; validate those on the public track data instead. Selectors
# are case-insensitive globs, so ``-s MyModel -s "eegnet*"`` puts your
# solver next to the track's own baselines in one run.
#
# If you implement ``fit`` and ``save_model``, benchopt will also package
# the submission for you. Train on the real data with:
#
# .. code-block:: bash
#
#    benchopt prepare tracks/<track>
#    benchopt run tracks/<track> -s MyModel -o "<objective>[training=True]"
#
# where ``<track>`` and ``<objective>`` are ``image_decoding`` /
# ``Image-decoding``, ``bci_decoding`` / ``BCI-decoding``, ``sleep_onset``
# / ``Sleep-onset``, and ``emg_pose`` / ``EMG-pose``. The run writes a
# ready-to-upload folder at ``tracks/<track>/outputs/<model-name>/``,
# holding ``submission.py`` and its weights.
#
# Either way, zip the **contents** of that folder rather than the folder
# itself:
#
# .. code-block:: bash
#
#    zip -j my_submission.zip submission.py weights.pt
#
# Then upload on the track's **My Submissions** tab, select the active
# phase, and wait for *Finished*. If ingestion fails, read the **first**
# error in the log: a later complaint about a missing ``results.parquet``
# usually just means inference had already failed.

# %%
# Timeline and submission limits
# -------------------------------
#
# The phase dates, the per-day submission limits, how the final ranking is
# taken, and the reproducibility audit that follows it all live on the
# `competition website <https://neural-interfaces26.github.io/>`__ and each
# track's Codabench *Timeline* tab. They are not repeated here, so that
# there is one place to check and no stale copy to contradict it.
#
# One consequence is worth knowing while you work, though: the warm-up
# phase scores against the **public** test partitions of THINGS-EEG2,
# Dreyer 2023, Sleep-EDF and emg2pose, so leakage is possible and a warm-up
# score is indicative only. The sealed phase swaps in the private 2026
# cohorts. Keep your training code reproducible from the start -- the audit
# reruns it.
