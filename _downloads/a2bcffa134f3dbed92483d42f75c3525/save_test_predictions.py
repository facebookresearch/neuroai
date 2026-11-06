"""
Saving and Reusing Test Predictions
====================================

By default NeuralBench only keeps the **aggregate** test metrics returned by
:meth:`~neuralbench.main.Experiment.run`.  When you want to inspect the model's
behaviour window-by-window -- to compute a new metric, plot error
distributions, or build a retrieval analysis -- you can ask the experiment to
log the **raw per-window predictions and targets** to disk.

This is controlled by the opt-in ``save_test_predictions`` flag on
:class:`~neuralbench.main.Experiment`.  When enabled, a
:class:`~neuralbench.callbacks.WindowPredictionCollector` streams the
``(y_pred, y_true)`` produced during the test loop to the experiment's cache
folder, alongside a small per-window metadata table.  The predictions live next
to the cached result, so they survive cache hits and can be read back later
without re-running the experiment.
"""

# %%
# Enabling prediction logging
# ---------------------------
#
# Set ``save_test_predictions: true`` in your experiment config (it is an
# :class:`~neuralbench.main.Experiment` field, so it sits at the top level of
# ``config.yaml`` next to ``seed``, ``infra``, ``trainer_config``, ...):
#
# .. code-block:: yaml
#
#    seed: 0
#    save_test_predictions: true
#    infra:
#      folder: /path/to/cache   # required: predictions are stored here
#      ...
#
# The flag is part of the exca cache key, so toggling it produces a *fresh*
# cache entry rather than invalidating existing results.  An ``infra.folder``
# must be configured -- the predictions are written under
# ``<uid_folder>/test_predictions``.
#
# .. note::
#
#    Logging is **opt-in** because it can use a lot of memory and disk for
#    high-dimensional outputs (e.g. word/video embeddings, fMRI voxel targets,
#    or CTC log-probs).  Arrays are streamed one batch at a time, so the *write*
#    path stays light, but reading everything back concatenates the full arrays
#    in RAM (see the caveats at the bottom).

# %%
# What gets saved
# ---------------
#
# The collector writes three things to ``<uid_folder>/test_predictions``:
#
# - ``metadata`` -- a :class:`pandas.DataFrame` with one row per test window:
#
#   * ``timeline`` -- the recording the window came from;
#   * ``batch_idx`` / ``dataloader_idx`` -- where it appeared in the test loop;
#   * ``subject_id`` -- the encoded subject (when the batch exposes one);
#   * ``group`` -- the stimulus identity for retrieval tasks (e.g. the word
#     ``text``), when available.
#
# - ``y_true`` / ``y_pred`` -- arrays of shape ``(n_windows, ...)`` aligned with
#   ``metadata``, stored with their **native** per-task shape (class logits,
#   regression vectors, retrieval embeddings, CTC log-probs, ...).  No per-task
#   aggregation is applied.
#
# For **retrieval** tasks, note that the retrieval *set* itself is not stored as
# a separate object: ``y_true`` holds the per-window target embeddings (one row
# per window, with repeats), and the ``group`` column records which stimulus
# each row corresponds to.  That is enough to reconstruct the retrieval set by
# de-duplicating ``y_true`` per ``group``.

# %%
# Reading the predictions back
# ----------------------------
#
# Rebuild the *same* experiment config (so it matches the cache entry) and call
# :meth:`~neuralbench.main.Experiment.test_predictions`.  Because the result is
# cached, ``run()`` returns immediately and the predictions are read from disk:
#
# .. code-block:: python
#
#    from neuralbench.main import Experiment
#
#    experiment = Experiment(**config)  # same config used to run, flag on
#    experiment.run()                   # cache hit -> returns instantly
#
#    preds = experiment.test_predictions()
#    metadata = preds["metadata"]       # DataFrame, one row per window
#    y_true = preds["y_true"]           # np.ndarray, shape (n_windows, ...)
#    y_pred = preds["y_pred"]
#
#    print(metadata.head())
#    print(y_pred.shape, y_true.shape)

# %%
# Recomputing a metric offline
# ----------------------------
#
# Because the raw logits and targets are available, you can compute *any*
# metric after the fact -- without re-running the model.  For a classification
# task whose ``y_pred`` are class logits and ``y_true`` are integer labels:
#
# .. code-block:: python
#
#    import numpy as np
#    import torch
#    import torchmetrics
#
#    preds = experiment.test_predictions()
#    y_pred = torch.from_numpy(np.asarray(preds["y_pred"]))
#    y_true = torch.from_numpy(np.asarray(preds["y_true"])).long()
#
#    num_classes = y_pred.shape[1]
#    balanced_acc = torchmetrics.Accuracy(
#        task="multiclass", num_classes=num_classes, average="macro"
#    )
#    balanced_acc.update(y_pred, y_true)
#    print("balanced accuracy:", balanced_acc.compute().item())
#
# The metadata columns make it easy to slice first, e.g. compute the metric
# per subject with ``metadata.groupby("subject_id")`` and indexing into
# ``y_pred`` / ``y_true`` with each group's row indices.

# %%
# Caveats
# -------
#
# - **Memory on read.** Predictions are streamed to disk per batch (light on
#   write), but :meth:`~neuralbench.main.Experiment.test_predictions`
#   concatenates the per-batch chunks, so loading materializes the full arrays
#   in RAM.  For very large outputs, read and process the chunk keys directly
#   from the underlying ``CacheDict`` instead.
# - **Multi-GPU training.** NeuralBench always runs the test loop on a single
#   device (global rank zero, ``devices=1``, after the training process group is
#   torn down), so the saved predictions cover the full test set even when
#   training used multiple GPUs.
# - **Disk usage.** The arrays are stored uncompressed; budget roughly
#   ``n_windows * output_dim * 4`` bytes each for ``y_true`` and ``y_pred``.
