# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pack pending experiments into fewer scheduler jobs via exca's ``Parallel`` step.

Instead of one Slurm job per experiment (which hits job-array limits on large
grids, see facebookresearch/exca#179), the pending experiments are dispatched as
variants of a single :class:`exca.steps.helpers.Parallel` sweep: exca packs them
into ``max_jobs`` array elements under one submission (see facebookresearch/exca#280).

Each variant is a thin :class:`_ExperimentStep` whose ``_run`` delegates to the
experiment's own :meth:`run`, so every experiment still caches under its
individual ``exca.TaskInfra`` identity — re-running the grid *without* packing
hits those same caches.
"""

from __future__ import annotations

import logging
import math
import typing as tp
from concurrent.futures import ThreadPoolExecutor

import pydantic
from exca.steps.base import Step
from exca.steps.helpers import Parallel

from neuraltrain.utils import BaseExperiment

LOGGER = logging.getLogger(__name__)

# ``TaskInfra.cluster`` -> exca steps backend class.
_BACKEND_FOR_CLUSTER: dict[str | None, str] = {
    None: "Cached",  # in-process, no scheduler
    "debug": "Cached",  # inline, for debugging
    "local": "LocalProcess",  # submitit local executor
    "auto": "Auto",  # autodetect Slurm, else local
    "slurm": "Slurm",
}
# Backends that accept Slurm-only resource hints (partition/account/...).
_SLURM_BACKENDS = frozenset({"Slurm", "Auto"})
# ``TaskInfra`` fields carried straight through to a submitit backend (same name).
_SHARED_FIELDS = (
    "job_name",
    "timeout_min",
    "nodes",
    "tasks_per_node",
    "cpus_per_task",
    "gpus_per_node",
    "mem_gb",
)
# ``TaskInfra`` field -> renamed steps ``Slurm``/``Auto`` backend field.
_RENAMED_FIELDS = {
    "slurm_partition": "partition",
    "slurm_account": "account",
    "slurm_qos": "qos",
    "slurm_constraint": "constraint",
    "slurm_additional_parameters": "additional_parameters",
    "slurm_use_srun": "use_srun",
}


class _ExperimentStep(Step):
    """Adapt a ``TaskInfra``-based :class:`BaseExperiment` to exca's steps API.

    ``exp_uid`` (the experiment's ``TaskInfra`` uid) is the step's whole
    identity, so each variant caches distinctly. The experiment itself rides in
    a ``PrivateAttr`` — ``BaseExperiment`` is not a discriminated model and would
    not survive ``Parallel``'s step re-validation as a normal field, whereas a
    private attr is excluded from the config/uid yet pickled to workers.

    ``_run`` delegates to the experiment's ``run`` so the real result caches
    under the experiment's own ``TaskInfra``. Failures are isolated per variant
    (logged, not raised) so one broken experiment does not abort the sweep; the
    failure is still recorded in that experiment's ``TaskInfra`` (visible via
    ``status()``) and re-submitted on a later ``retry`` run.
    """

    exp_uid: str
    _experiment: BaseExperiment = pydantic.PrivateAttr()

    def bind(self, experiment: BaseExperiment) -> _ExperimentStep:
        """Attach the experiment to run (kept out of the cache identity)."""
        self._experiment = experiment
        return self

    def _run(self, value: tp.Any = None) -> None:
        try:
            self._experiment.run()  # cached by the experiment's own TaskInfra
        except Exception:
            LOGGER.exception("Packed experiment %s failed; skipping", self.exp_uid)
        return None


def _should_submit_experiment(experiment: BaseExperiment) -> bool:
    """Whether an experiment needs (re-)submission.

    Mirrors the status-and-mode filter that ``exca.TaskInfra.job_array`` applies
    internally; kept inline so already-cached experiments are dropped before the
    sweep, rather than dispatched only to return from cache.
    """
    status = experiment.infra.status()
    mode = experiment.infra.mode
    if mode == "read-only":
        return False
    return (
        status == "not submitted"
        or mode == "force"
        or (status == "failed" and mode == "retry")
    )


def _pending_experiments(
    experiments: tp.Sequence[BaseExperiment],
) -> list[BaseExperiment]:
    """Runnable experiments, sorted by uid (order-independent packing).

    ``_should_submit_experiment`` calls ``infra.status()``, which may hit the
    filesystem (uncached, NFS in the cluster case), so the check is threaded to
    avoid paying ``len(experiments)`` × stat-latency in series.
    """
    n = len(experiments)
    if n == 0:
        return []
    with ThreadPoolExecutor(max_workers=min(32, n)) as pool:
        keep = list(pool.map(_should_submit_experiment, experiments))
    keyed = [(e.infra.uid(), e) for e, k in zip(experiments, keep) if k]
    keyed.sort(key=lambda kv: kv[0])
    return [e for _, e in keyed]


def _backend_config(
    infra: tp.Any, folder: tp.Any, max_jobs: int
) -> dict[str, tp.Any]:
    """Translate an experiment's ``TaskInfra`` into a steps backend config.

    The scheduler resource budget (partition/gpus/mem/time/...) is taken from a
    single ``TaskInfra`` — group experiments by resource hints before packing if
    they differ. ``mode="force"`` makes the wrapper marker never the cache
    authority: every pending variant is dispatched and the experiment's own
    ``TaskInfra`` decides what actually recomputes.
    """
    cluster = getattr(infra, "cluster", None)
    if cluster not in _BACKEND_FOR_CLUSTER:
        raise ValueError(
            f"unsupported infra.cluster={cluster!r}; expected one of "
            f"{sorted(_BACKEND_FOR_CLUSTER, key=str)}"
        )
    backend = _BACKEND_FOR_CLUSTER[cluster]
    cfg: dict[str, tp.Any] = {"backend": backend, "folder": folder, "mode": "force"}
    if backend == "Cached":
        return cfg  # local in-process backend: no resource/packing knobs
    for name in _SHARED_FIELDS:
        val = getattr(infra, name, None)
        if val is not None:
            cfg[name] = val
    if backend in _SLURM_BACKENDS:
        for src, dst in _RENAMED_FIELDS.items():
            val = getattr(infra, src, None)
            if val not in (None, False):  # skip unset + default use_srun=False
                cfg[dst] = val
    cfg["max_jobs"] = max_jobs
    return cfg


def submit_packed(
    experiments: tp.Sequence[BaseExperiment],
    experiments_per_job: int | tp.Literal["all"] = "all",
) -> int:
    """Dispatch pending experiments as one exca ``Parallel`` sweep.

    ``experiments_per_job`` sets the target group size: ``"all"`` packs every
    pending experiment into one scheduler job; an int ``N`` caps the sweep at
    ``ceil(pending / N)`` array elements (~``N`` experiments each). Returns the
    number of experiments dispatched (``0`` when everything runnable is cached).
    """
    pending = _pending_experiments(experiments)
    if not pending:
        return 0

    n = len(pending)
    per_job = n if experiments_per_job == "all" else int(experiments_per_job)
    if per_job < 1:
        raise ValueError(f"experiments_per_job must be >= 1 or 'all', got {per_job}")
    max_jobs = max(1, math.ceil(n / per_job))

    infra = pending[0].infra
    if infra.folder is None:
        raise RuntimeError(
            "experiments need infra.folder set to pack; got infra.folder=None"
        )
    backend = _backend_config(infra, infra.folder, max_jobs)

    variants = [
        # cluster=None so run() computes in-process inside the worker instead of
        # submitting a nested job; cluster is excluded from the uid, so the
        # experiment still caches under exp.infra.uid().
        _ExperimentStep(exp_uid=e.infra.uid(), infra=backend).bind(
            e.infra.clone_obj({"infra.cluster": None})
        )
        for e in pending
    ]
    Parallel(steps=variants, infra=backend).run()
    return n
