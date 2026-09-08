# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate a model defined outside this repo.

The CLI and :func:`~neuralbench.run_benchmark` take a model *name*, which means
the model has to live in this repo with a ``models/*.yaml`` beside it.
:func:`evaluate_model` takes the built instance instead::

    scores = evaluate_model(my_pretrained_fm, "eeg", "all", name="my-eeg-fm")
"""

import copy
import functools
import inspect
import logging
import typing as tp
from itertools import product
from pathlib import Path

import mne
import pandas as pd
import torch
from exca import ConfDict
from torch import nn

from neuralbench.config_manager import get_config
from neuralbench.experiment_config import (
    _download_dataset,
    apply_cluster,
    build_experiment_configs,
    merge_task_config,
)
from neuralbench.external import ExternalModel, check_forward, save_instance
from neuralbench.registry import _resolve_datasets, _resolve_tasks

LOGGER = logging.getLogger(__name__)

#: Montage widths :func:`check_model` probes, spanning the range real EEG caps
#: cover, so a model that handles only one width shows up as such.
_PROBE_CHANNELS = (3, 19, 64, 128)


def _external_config(model: nn.Module) -> ExternalModel:
    """Serialize *model* to the cache folder and point a config at it."""
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Expected an instantiated torch.nn.Module, got {model!r}. Build "
            "your model first -- including loading any pretrained weights -- "
            "and pass the instance."
        )
    check_forward(model)
    folder = Path(get_config()["CACHE_DIR"]) / "external_models"
    path, digest = save_instance(model, folder)
    LOGGER.info("Serialized model to %s", path)
    return ExternalModel(pickle_path=str(path), digest=digest)


def _experiment_configs(
    model_overlay: dict[str, tp.Any],
    *,
    overrides: tp.Mapping[str, tp.Any] | None,
    cluster: str | None,
    **selection: tp.Any,
) -> list[ConfDict]:
    configs = build_experiment_configs(model=model_overlay, quiet=True, **selection)
    for config in configs:
        # Last, because the debug and prepare overlays run after a model config
        # is merged and would otherwise silently revert an override.
        if overrides:
            config.update(dict(overrides))
        apply_cluster(config, cluster)
    return configs


def _run(configs: list[ConfDict], debug: bool) -> tp.Any:
    """Submit *configs* and return the aggregator holding them."""
    from neuralbench.main import BenchmarkAggregator

    # ConfDicts, which the pydantic model coerces into Experiment instances.
    agg = BenchmarkAggregator(experiments=configs, debug=debug)  # type: ignore[arg-type]
    agg.prepare()
    return agg


def evaluate_model(
    model: nn.Module,
    device: str,
    task: str | list[str],
    *,
    name: str = "external",
    overrides: tp.Mapping[str, tp.Any] | None = None,
    downstream_wrapper: str | list[str] = "linear_probe_mean",
    download: bool = True,
    prepare: bool = True,
    cluster: str | None = None,
    debug: bool = False,
    **selection: tp.Any,
) -> pd.DataFrame:
    """Run *model* on a selection of tasks and return the results.

    Runs in-process by default, which is usually what you want in a notebook
    but means a full suite can take a very long time.  Pass ``cluster="auto"``
    to fan the experiments out to SLURM instead; that call returns once they are
    queued, so it yields whatever has finished (nothing, at first).  Call it
    again with the same arguments to collect -- already-completed experiments
    are not resubmitted, exactly as with re-running the CLI.

    Parameters
    ----------
    model
        An instantiated ``nn.Module`` accepting ``(batch, channels, samples)``
        at any width and length, plus ``channel_positions``.  A ``forward``
        that also names ``ch_names`` is given the dataset's channel names,
        which is the only way to identify electrodes for a model keyed by name
        (e.g. LaBraM).  It is serialized once to the cache folder and reloaded
        fresh for each experiment.  Run :func:`check_model` first.
    device, task
        As on the CLI; ``task="all"`` runs every validated task for the device,
        which with the default ``dataset=None`` is the device's Core suite and
        with ``dataset="all"`` its Full suite.
    name
        Label for this model in the results frame.
    overrides
        Dotted-key or nested config overrides, applied last and recorded in the
        frame's ``overrides`` column.
    downstream_wrapper
        Adaptation strategy, which is also what supplies the task's classifier
        head; the default freezes the model and trains a linear probe on its
        mean-pooled output.
    download
        Fetch the datasets first.  Nothing downloads on demand, so a missing
        dataset otherwise fails deep inside ``Study``.  Cheap once present.
    prepare
        Warm the preprocessing caches first, by running one cut-down experiment
        per task/dataset.  Hours on a cold cache for a whole device, a status
        check when already warm.  Needs the data, hence after ``download``.
        On a cluster the warm-up is queued rather than run, so the call returns
        without submitting the experiments; call it again once it has finished.
    cluster
        ``None`` runs in-process; ``"auto"`` uses SLURM when available;
        ``"slurm"`` requires it.  Applies to the runs and the caches alike.
    debug
        Reduce every experiment (2 epochs, 5 batches, a data subset) and run
        locally.  For checking that a model trains at all, not for results.
    **selection
        Remaining :func:`neuralbench.cli.run_benchmark` arguments
        (``dataset``, ``force``, ...).
    """
    config = _external_config(model)
    overlay = {
        "brain_model_name": name,
        # =replace= so the external config supplants the default model rather
        # than merging field-by-field with it.
        "brain_model_config": {"=replace=": True, **config.model_dump()},
    }
    phase = dict(
        device=device,
        task=task,
        overrides=overrides,
        downstream_wrapper=downstream_wrapper,
        cluster=cluster,
        debug=debug,
        **selection,
    )
    configs = _experiment_configs(overlay, **phase)

    if download:
        # Several configs share a study, and downloading is the same work for each.
        for cfg in {c["data.study.source"]["name"]: c for c in configs}.values():
            _download_dataset(cfg)
    if prepare:
        warming = _run(_experiment_configs(overlay, prepare=True, **phase), debug)
        if cluster is not None and any(
            exp.infra.status() != "completed" for exp in warming.experiments
        ):
            LOGGER.info("Warming the caches on %s; call again to run.", cluster)
            # Submitting now would have every experiment race the warm-up it
            # exists to avoid, and recompute the same caches once per seed.
            return pd.DataFrame()

    agg = _run(configs, debug)
    # Debug experiments ran in-process and so never reach a `completed` exca
    # status; their results come straight from the cache instead.
    frame = pd.DataFrame(agg.collect(cached_only=not debug))
    if not frame.empty and overrides:
        flat = ConfDict(dict(overrides)).flat()
        frame["overrides"] = ";".join(f"{k}={v!r}" for k, v in sorted(flat.items()))
    return frame


def check_model(
    model: nn.Module,
    device: str,
    task: str | list[str],
    *,
    dataset: str | list[str] | None = None,
    overrides: tp.Mapping[str, tp.Any] | None = None,
    batch_size: int = 2,
) -> pd.DataFrame:
    """Push synthetic batches of a selection's shapes through *model*.

    One instance runs the whole selection, so it has to survive every window
    length and montage width in it.  Reads only YAML, so it takes seconds and
    needs no downloaded data -- the alternative being to discover a shape bug
    an hour into a real run.

    Returns one row per (task, window, channel count) with ``status`` either
    ``"ok"`` or the exception that would have surfaced during the run.  Raises
    instead when ``forward`` cannot accept channel positions at all, since that
    is a property of the model rather than of any one task.

    The temporal width is computed from the task window and sampling rate and
    may be off by a sample, and the channel counts are probes rather than the
    widths a dataset will actually emit.  So this catches a model that cannot
    handle a task's shapes at all, not an exact match with the dataloader.
    Likewise the synthetic positions are random and the ``ch_names`` given to a
    model that asks for them are 10-05 names: enough to exercise the shapes,
    not a montage any dataset has.
    """
    check_forward(model)
    # Probing must not touch the caller's object: a forward pass materializes
    # lazy layers and, in train mode, updates normalization statistics.
    probe = copy.deepcopy(model).eval()
    flat_overrides = ConfDict(dict(overrides or {})).flat()

    rows: list[dict[str, tp.Any]] = []
    for task_name in _resolve_tasks(device, task):
        # Dataset variants override the task window, so a selection can span
        # several window lengths; only the distinct ones are worth probing.
        windows: set[int] = set()
        for dataset_name in _resolve_datasets(device, task_name, dataset):
            flat = merge_task_config(device, task_name, dataset_name).flat()
            flat.update(flat_overrides)
            frequency = flat.get("data.neuro.frequency")
            duration = flat.get("data.duration")
            if frequency and duration:
                windows.add(round(duration * frequency))
        if not windows:
            rows.append(
                {"task": task_name, "status": "skipped: no fixed window or sampling rate"}
            )
        for samples, width in product(sorted(windows), _PROBE_CHANNELS):
            rows.append(
                {
                    "task": task_name,
                    "n_spatial_locations": width,
                    "n_temporal_samples": samples,
                    **_try_forward(probe, width, samples, batch_size),
                }
            )
    return pd.DataFrame(rows)


@functools.lru_cache(maxsize=1)
def _standard_ch_names() -> tuple[str, ...]:
    """Electrode names to probe a model keyed by name, 10-20 ones first.

    A model may only know part of the 10-05 system, so the widest probe still
    spends some of its channels on names it drops; putting the classic ones
    first keeps the narrower probes to names every EEG model knows.
    """
    names = mne.channels.make_standard_montage("standard_1005").ch_names
    classic = set(mne.channels.make_standard_montage("standard_1020").ch_names)
    return tuple(sorted(names, key=lambda name: name not in classic))


def _try_forward(
    model: nn.Module, channels: int, samples: int, batch_size: int
) -> dict[str, tp.Any]:
    x = torch.randn(batch_size, channels, samples)
    inputs: dict[str, tp.Any] = {
        "channel_positions": torch.randn(batch_size, channels, 3)
    }
    if "ch_names" in inspect.signature(model.forward).parameters:
        inputs["ch_names"] = list(_standard_ch_names()[:channels])
    try:
        with torch.no_grad():
            out = model(x, **inputs)
    except Exception as e:  # noqa: BLE001 - reported, not raised
        return {"status": f"forward failed: {type(e).__name__}: {e}"}

    if not isinstance(out, torch.Tensor):
        return {"status": f"output is {type(out).__name__}, expected a Tensor"}
    if out.ndim < 2 or out.shape[0] != batch_size:
        # The probe pools over everything but the batch axis, so a model that
        # puts batch elsewhere would have its samples mixed together.
        return {
            "status": f"output {tuple(out.shape)} is not batch-first",
            "output_shape": tuple(out.shape),
        }
    return {"status": "ok", "output_shape": tuple(out.shape)}
