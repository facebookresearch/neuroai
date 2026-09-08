"""
Evaluating your own model
=========================

The :doc:`CLI </neuralbench/auto_examples/quickstart/01_run_first_task>` and
:func:`~neuralbench.run_benchmark` both take a model *name*, which means the
model has to live in this repo with a ``models/*.yaml`` beside it.  When your
model lives in your own codebase, use :func:`~neuralbench.evaluate_model`
instead: it takes the model itself and needs no files here.

It takes your model *already built*, weights and all.  One instance then serves
every task in the selection, which is only possible for a model that adapts to
its input, so this path asks two things of it:

* it accepts any channel count and any window length, since tasks differ in
  montage and epoch length -- from a 3-electrode sensorimotor strip to a
  128-channel cap;
* it reads channel identity in ``forward`` rather than from a montage fixed at
  construction.  ``channel_positions`` is always passed; a ``forward`` that
  also names ``ch_names`` is handed the dataset's electrode names, which is
  what a model keyed by name rather than by coordinate needs.

What it does *not* need is a classifier head.  NeuralBench wraps your model in a
probe sized to each task's number of classes, so the same backbone can serve a
4-class motor imagery task and a 2-class P300 task unchanged.
"""

# %%
# A model to evaluate
# -------------------
#
# This one encodes each channel's time course into tokens, adds a per-channel
# embedding derived from that channel's 3D position, then pools over channels.
# The output keeps a time axis -- ``(batch, tokens, width)``, the usual shape a
# transformer encoder produces -- and the number of tokens follows the input
# length, which is fine: the probe pools over them.

from torch import nn


class TinyFm(nn.Module):
    """A small position-conditioned encoder, adaptive in channels and time."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.width = width
        # Channel identity comes from the position, so an unseen montage needs
        # no new parameters -- which is what lets one instance span datasets.
        self.pos_embedding = nn.Linear(3, width)
        self.encoder = nn.Conv1d(1, width, kernel_size=17, stride=8, padding=8)
        self.norm = nn.LayerNorm(width)

    def forward(self, x, channel_positions):
        # x: (batch, channels, samples)
        # channel_positions: (batch, channels, 3)
        batch, channels, samples = x.shape

        # Encode every channel with the same temporal filters.
        tokens = self.encoder(x.reshape(batch * channels, 1, samples))
        tokens = tokens.reshape(batch, channels, self.width, -1)
        tokens = tokens.permute(0, 1, 3, 2)  # (batch, channels, tokens, width)

        tokens = tokens + self.pos_embedding(channel_positions).unsqueeze(2)
        return self.norm(tokens.mean(1))  # (batch, tokens, width)


# %%
# Check it before you queue anything
# ----------------------------------
#
# :func:`~neuralbench.check_model` pushes a synthetic batch of the selection's
# shapes through the model, at several montage widths. It reads only YAML, so it
# takes seconds and needs no downloaded data -- which is the point, since the
# alternative is discovering a shape bug an hour into a real run.
#
# .. code-block:: python
#
#    from neuralbench import check_model
#
#    print(check_model(TinyFm(), "eeg", "motor_imagery"))
#
# ::
#
#             task  n_spatial_locations  n_temporal_samples status  output_shape
#    0  motor_imagery                 3                 480     ok  (2, 60, 64)
#    1  motor_imagery                19                 480     ok  (2, 60, 64)
#    2  motor_imagery                64                 480     ok  (2, 60, 64)
#    3  motor_imagery               128                 480     ok  (2, 60, 64)
#
# A model that only handles one montage width shows up as ``ok`` on one row and
# ``forward failed: ...`` on the others. Pass the same ``task`` and ``dataset``
# you intend to run: dataset variants can shorten a task's window, and each
# distinct window gets its own rows.
#
# A ``forward`` that does not accept ``channel_positions`` at all raises instead
# of filling the ``status`` column, being a property of the model rather than of
# any one task.

# %%
# Run it
# ------
#
# ``device`` and ``task`` mean what they mean on the CLI, so ``task="all"``
# runs every validated task for the device. Start with ``debug=True``: it runs
# locally on a reduced config (2 epochs, 5 batches, a data subset).
#
# .. code-block:: python
#
#    from neuralbench import evaluate_model
#
#    scores = evaluate_model(
#        TinyFm(),
#        "eeg",
#        "motor_imagery",
#        name="tiny-fm",
#        dataset=["schalk2004bci2000", "leeb2007"],
#        debug=True,
#    )
#    print(scores[["dataset_name", "test/bal_acc", "n_total_params",
#                  "n_trainable_params"]])
#
# ::
#
#            dataset_name  test/bal_acc  n_total_params  n_trainable_params
#    0  Schalk2004Bci2000         0.258             900                 132
#    1      Leeb2007Brain         0.475             834                  66
#
# Those two datasets have 64 and 3 channels respectively, and one instance
# served both. The parameter counts show the split: the backbone is identical
# across the rows, and only the probe differs -- 132 trainable parameters for
# Schalk's 4 classes against 66 for Leeb's 2.
#
# Build the model however you normally would -- ``model.load_state_dict(...)``
# and all -- and ``evaluate_model`` takes it from there. The instance is
# serialized once to the cache folder and reloaded fresh for each experiment, so
# weights that one run fine-tunes never leak into the next, and the weights
# themselves take part in the cache key: change them and you get new runs rather
# than stale cached results.

# %%
# How your model is adapted to each task
# --------------------------------------
#
# ``downstream_wrapper`` chooses the adaptation strategy, and with it where the
# task's classifier head comes from. The default, ``linear_probe_mean``, freezes
# your model and trains a linear probe on its mean-pooled output.
#
# .. code-block:: python
#
#    scores = evaluate_model(
#        TinyFm(), "eeg", "all", downstream_wrapper="finetune_mean"
#    )
#
# Prefer the ``_mean`` presets here. ``flatten`` builds the probe from the
# flattened output instead, which for an adaptive model makes the probe's size a
# function of the dataset -- on the run above it would be 8196 trainable
# parameters for the 64-channel dataset against 194 for the 3-channel one. It
# works, but head capacity then varies with montage density and window length,
# which is not something you want varying underneath a benchmark number.

# %%
# Suites
# ------
#
# The dataset selection decides which suite you are running, exactly as it does
# on the CLI (see :ref:`Benchmark suites <benchmark-suites>`):
#
# .. list-table::
#    :header-rows: 1
#    :widths: 40 60
#
#    * - Call
#      - Suite
#    * - ``evaluate_model(m, "eeg", "all")``
#      - ``NeuralBench-EEG-Core`` (one dataset per task)
#    * - ``evaluate_model(m, "eeg", "all", dataset="all")``
#      - ``NeuralBench-EEG-Full`` (every dataset per task)

# %%
# Downloading, caching, and where the runs happen
# -----------------------------------------------
#
# Two things have to happen before an experiment can train, and neither is
# automatic: the dataset has to be on disk, and the preprocessed data has to be
# in the cache.  :func:`~neuralbench.evaluate_model` does both first, in that
# order, because preprocessing needs the downloaded files.  Once warm, both
# phases are a cheap status check, so leave them on; turn them off only when you
# know the state of the cache.  Warming it for a whole device takes hours, and
# it is the same work the CLI does with ``--download`` and ``--prepare``, so a
# cache warmed either way serves the other.
#
# Everything then runs in the current process, which is usually what you want in
# a notebook, and which means a full suite can take a very long time. Pass
# ``cluster="auto"`` to fan the experiments out to SLURM where one is available:
#
# .. code-block:: python
#
#    scores = evaluate_model(
#        TinyFm(), "eeg", "all", name="my-fm", cluster="auto",
#        download=False,   # datasets already on disk
#        prepare=False,    # caches already warm
#    )
#
# That call returns as soon as the jobs are queued, so the frame holds whatever
# had already finished -- nothing, the first time. Call it again with the same
# arguments to collect: completed experiments are not resubmitted, exactly as
# when re-running the CLI. With ``prepare=True`` the first call queues the cache
# warm-up alone and stops there, since the experiments would otherwise race it.
#
# Results never raise on a failed experiment. You get the rows that exist, and
# the rest are reported in a printed table where ``!n`` marks jobs that errored
# out as opposed to ones still pending.

# %%
# Changing the protocol
# ---------------------
#
# ``overrides`` takes any config key, as dotted keys or nested dicts, and is
# applied on top of the defaults and the task config:
#
# .. code-block:: python
#
#    scores = evaluate_model(
#        TinyFm(),
#        "eeg",
#        "all",
#        overrides={
#            "data.neuro.frequency": 200.0,                    # preprocessing
#            "lightning_optimizer_config.optimizer.lr": 1e-4,  # optimization
#            "trainer_config.n_epochs": 50,
#        },
#    )
#
# Whatever you pass is recorded in the frame's ``overrides`` column, so a run
# with a changed protocol is never mistaken for a stock one.
#
# Be careful with keys that define *what a task measures* -- the epoch window
# (``data.start`` / ``data.duration``), the target extractor, the splits, the
# loss, the metrics.  Changing those produces a number that is not comparable
# to any other entry on the suite, including the published baselines.

# %%
# A worked example: pretrained REVE
# ---------------------------------
#
# The foundation models the benchmark ships with are ordinary modules too, so
# one of them makes a good end-to-end example.  REVE is a braindecode model and
# ``NtReve`` is the config that builds it: ``n_outputs=None`` asks for the
# encoder alone, since the probe supplies the head, and the width and window
# length only size the build -- the instance reads the montage it actually gets
# from ``channel_positions`` on every call.
#
# .. note::
#    The weights are pulled from the Hugging Face hub at build time, so the
#    first build needs a Hugging Face account: run ``huggingface-cli login``, or
#    set ``HF_TOKEN`` in the environment, beforehand.  Later builds read them
#    from the local hub cache and need no network.
#
# .. code-block:: python
#
#    from neuraltrain.models.reve import NtReve
#
#    model = NtReve(from_pretrained_name="brain-bzh/reve-base").build(
#        n_spatial_locations=64,
#        n_temporal_samples=400,
#        n_outputs=None,
#    )
#
# A pretrained model expects the preprocessing it was trained with rather than
# the task default, which is what ``overrides`` is for.  These values are
# REVE's, read off ``models/reve.yaml``:
#
# .. code-block:: python
#
#    scores = evaluate_model(
#        model,
#        "eeg",
#        "motor_imagery",
#        name="reve-base",
#        overrides={
#            "data.neuro.frequency": 200.0,
#            "data.neuro.filter": [0.5, 99.5],
#            "data.neuro.notch_filter": None,
#            "data.neuro.baseline": None,
#            "data.neuro.scaler": "StandardScaler",
#            "data.neuro.clamp": 15,
#            # REVE's position bank is in head-frame metres, not normalised.
#            "data.channel_positions.normalize": False,
#        },
#        debug=True,
#    )
#
# Same configuration, same number as the CLI.  The call above matches ``-m reve
# -w linear_probe_mean`` rather than a bare ``-m reve``, which reads its
# adaptation strategy and its channel mapping from ``models/reve.yaml`` instead.

# %%
# What the Slurm jobs need
# ------------------------
#
# Off ``debug=True``, experiments run as Slurm jobs in separate processes, which
# rebuild the model from the serialized instance. Classes defined in a script, a
# notebook or an uninstalled checkout are stored in it whole, so there is
# nothing to package and nothing to put on the jobs' ``PYTHONPATH``. Code from
# *installed* packages is stored by import path instead: a model that imports,
# say, ``timm`` needs ``timm`` in the environment the jobs run in.
