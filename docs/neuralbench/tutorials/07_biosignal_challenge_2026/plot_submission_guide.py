"""
How to Submit a Model
======================

This page describes how to package and upload a model to the
EEG/EMG Foundation Challenge 2026.

Submissions are handled on `Codabench <https://www.codabench.org/>`_, with
a separate registration for each track you enter. The `competition website
<https://neural-interfaces26.github.io/>`_ and each track's Codabench
*Participation* tab are authoritative on every submission matter; this page
summarises them and shows how a NeuralBench-trained model fits the
contract.
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
# Registration is per track, and **every member of a team registers
# individually**:
#
# 1. Create a `Codabench account
#    <https://www.codabench.org/accounts/signup>`__ and sign in.
# 2. Open the track's Codabench page, accept the terms, and click
#    **Register**. The request shows as *pending* at first.
# 3. Complete the `registration form <https://forms.gle/p3t2V25nuQtVXyj9A>`__
#    once, selecting every track you registered for. The email address on
#    the form must match your Codabench account -- that is what approves
#    the registration automatically.
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
# Teams additionally nominate a leader, who creates one `Codabench
# organization <https://www.codabench.org/profiles/organization/create/>`__
# and adds the other members. Each person may belong to only one team.
# Submission quotas stay individual -- everyone submits from their own
# account -- but the submissions are attributed to the team.

# %%
# What a submission is
# --------------------
#
# A zipped folder containing a ``submission.py`` plus whatever weight files
# your model needs. Evaluation is **inference-only**: the model must arrive
# fully trained, and nothing is installed at submission time -- the
# evaluation image already carries torch, scikit-learn, benchopt and the
# data stack.
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
# - ``fit(self, model, train_loader)`` (optional) trains the model. It
#   never runs on the server, but it is how you train locally against the
#   exact competition data and evaluation.
# - ``save_model(self, model, path)`` (optional) writes the trained weights
#   into ``path``. Implement it and a local training run ends by zipping
#   your solver together with those files into
#   ``outputs/submission_<track>.zip``, ready to upload.
#
# ``meta`` is a plain dict: ``sfreq``, ``ch_names``, ``chs_info``,
# ``n_chans``, ``n_times``, ``device``, ``submission_dir``, and the track's
# output size.
#
# .. list-table::
#    :header-rows: 1
#    :widths: 20 26 18 18
#
#    * - Track
#      - ``predict(X)`` returns
#      - Output size
#      - Ranking metric
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
#      - binned MAE
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
#        # torch and scikit-learn come with the evaluation environment;
#        # declare only your own extras.
#        requirements = ["pip::my-model-pkg"]
#
#        def load_model(self, meta):
#            model = build_my_model(
#                n_chans=meta["n_chans"], n_times=meta["n_times"],
#            )
#            state = torch.load(meta["submission_dir"] / "weights.pt",
#                               map_location=meta["device"])
#            model.load_state_dict(state)
#            return model.to(meta["device"]).eval()
#
# .. warning::
#    Only the *data* is hidden -- the loading, scoring and model-facing code
#    is public and identical in both phases. So one rule keeps a submission
#    valid on data you never see: **use only** ``meta`` **and the batches
#    you are given**. A solver that reads a dataset name, a file path, a
#    subject id, or a hard-coded channel count may work during warm-up and
#    break on the sealed cohort.

# %%
# Going from a NeuralBench run to a submission
# ---------------------------------------------
#
# NeuralBench is a preparation environment, not part of the submission: the
# evaluation image does not import it. What crosses the boundary is the
# trained weights plus enough code to rebuild the architecture.
#
# Every full NeuralBench run keeps exactly one checkpoint -- the best one
# under the track's validation metric (see the *Split and model selection*
# section on each track page) -- written as ``best.ckpt`` under the run's
# folder in ``SAVE_DIR``, with ``save_weights_only=True``. To turn that into
# a submission:
#
# 1. Locate ``best.ckpt`` for the run you want.
# 2. Strip the Lightning wrapper: the state dict is under the
#    ``"state_dict"`` key, and its parameter names carry the module prefix
#    the ``pl_module`` added.
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
# The starting kit for submission is the benchopt benchmark itself, one per
# track (linked from the track's Codabench *Participation* tab). From a
# checkout:
#
# .. code-block:: bash
#
#    benchopt install tracks/<track>          # CPU env (add --gpu for CUDA)
#    benchopt run tracks/<track> -d Simulated # zero-download smoke test
#
# ``Simulated`` needs no download and no data stack, so it is the fastest
# way to check that your solver loads and predicts the right shape. To try
# a submission, drop your files into the track's ``solvers/`` folder:
#
# .. code-block:: bash
#
#    cp my_submission/* tracks/<track>/solvers/
#    benchopt run tracks/<track> -d Simulated -s my-solver
#
# ``benchopt test tracks/<track> --skip-install`` is the rehearsal for the
# sealed phase: it runs your solver against a differently shaped dataset,
# the closest local stand-in for data it has never seen.
#
# To train on the real data, ``benchopt prepare tracks/<track>`` downloads
# it once, then
#
# .. code-block:: bash
#
#    benchopt run tracks/<track> -s my-solver -o "[training=True]"
#
# trains your solver through ``fit`` and evaluates it exactly as the
# platform does. This is also how the baselines shipped in ``solvers/`` are
# trained.
#
# When it passes, zip the folder and upload it on the track's **My
# Submissions** tab, then confirm that evaluation succeeds and your score
# reaches the leaderboard.

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
