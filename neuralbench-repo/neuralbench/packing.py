# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pack pending experiments into fewer scheduler jobs."""

from __future__ import annotations

import logging
import multiprocessing
import os
import typing as tp
from concurrent.futures import ProcessPoolExecutor

import exca
from pydantic import Field, SerializeAsAny

from neuraltrain.base import BaseModel
from neuraltrain.utils import BaseExperiment

LOGGER = logging.getLogger(__name__)


def _cpu_budget() -> int:
    """Return the CPU count available to this process.

    On Linux uses ``os.sched_getaffinity`` so Slurm cgroups are respected.
    Other platforms fall back to ``os.cpu_count``, which reports *logical*
    cores (hyperthreads). Linear speedup past the physical-core count is
    therefore not guaranteed on those platforms.
    """
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        return len(getaffinity(0))
    return os.cpu_count() or 1


class PackedExperiment(BaseModel):
    """Run a batch of :class:`BaseExperiment` instances inside one scheduler job.

    A ``PackedExperiment`` is itself an ``@exca.TaskInfra.apply``-cached
    object. Its UID is derived from ``experiments`` (via ``SerializeAsAny``,
    so subclass-specific fields contribute to the hash) and from ``infra``;
    ``n_jobs`` is intentionally excluded from the UID so the cache key is
    insensitive to local-parallelism choices.

    Each child experiment is expected to have ``infra.cluster=None`` and run
    in-process; :func:`pack_experiments_for_submission` arranges this.
    """

    # ``SerializeAsAny`` preserves subclass fields when dumping/serializing;
    # without it pydantic would only emit the declared ``BaseExperiment``
    # fields, hiding per-subclass parameters from the exca cache UID.
    experiments: list[SerializeAsAny[BaseExperiment]] = Field(min_length=1)
    n_jobs: int = Field(default=1, ge=1)

    # NOTE: bump ``version`` whenever the semantics of :meth:`run` change so
    # existing packed-job caches are invalidated.
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        # ``neuraltrain.base.BaseModel`` does not define this hook, so there
        # is no parent contribution to super() over.
        return ["n_jobs"]

    @infra.apply
    def run(self) -> list[tp.Any]:
        # BaseExperiment.run() is untyped (returns Any), so the element type
        # is widened to tp.Any here.
        if self.n_jobs == 1 or len(self.experiments) <= 1:
            return [experiment.run() for experiment in self.experiments]

        max_workers = min(self.n_jobs, len(self.experiments), _cpu_budget())
        ctx = multiprocessing.get_context("spawn")
        # Manual lifecycle (no ``with``) so we can cancel still-pending
        # futures on the first failure instead of waiting for them to drain
        # — which would otherwise delay error reporting by the full length
        # of the longest in-flight experiment.
        pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
        try:
            futures = [pool.submit(exp.run) for exp in self.experiments]
            return [f.result() for f in futures]
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


def _should_submit_experiment(experiment: BaseExperiment) -> bool:
    status = experiment.infra.status()
    mode = experiment.infra.mode
    if mode == "read-only":
        return False  # read-only mode never submits, regardless of status
    return (
        status == "not submitted"
        or mode == "force"
        or (status == "failed" and mode == "retry")
    )


def _validate_experiments_per_job(value: int | tp.Literal["all"]) -> None:
    if isinstance(value, str):
        if value != "all":
            raise ValueError(
                f"experiments_per_job must be an integer >= 1 or 'all'; "
                f"got {value!r}."
            )
        return
    if not isinstance(value, int) or isinstance(value, bool):
        # ``bool`` is a subclass of ``int``; reject explicitly so callers
        # don't pass True/False by accident.
        raise ValueError(
            f"experiments_per_job must be an integer >= 1 or 'all'; "
            f"got {value!r}."
        )
    if value < 1:
        raise ValueError("experiments_per_job must be >= 1 or 'all'.")


def _warn_heterogeneous_resources(
    pending: tp.Sequence[BaseExperiment],
) -> None:
    """Log a warning when packed experiments declare different Slurm resources.

    The packed scheduler job runs under ``pending[0]``'s budget; any peer
    asking for more CPUs/GPUs/memory/time/partition will be under-provisioned.
    """
    fields = (
        "cpus_per_task",
        "gpus_per_node",
        "mem_gb",
        "timeout_min",
        "slurm_partition",
    )
    head = pending[0].infra
    head_view = {f: getattr(head, f, None) for f in fields}
    mismatched = []
    for exp in pending[1:]:
        diff = {
            f: getattr(exp.infra, f, None)
            for f in fields
            if getattr(exp.infra, f, None) != head_view[f]
        }
        if diff:
            mismatched.append(diff)
            break  # one example is enough for the warning
    if mismatched:
        LOGGER.warning(
            "PackedExperiment: pending experiments have heterogeneous "
            "Slurm resource hints; the packed scheduler job will run "
            "under pending[0]'s budget (%s). Example mismatch: %s",
            head_view,
            mismatched[0],
        )


def pack_experiments_for_submission(
    experiments: tp.Sequence[BaseExperiment],
    experiments_per_job: int | tp.Literal["all"] = 1,
    n_jobs: int = 1,
) -> list[PackedExperiment]:
    """Pack pending experiments into fewer scheduler jobs.

    Experiments whose ``infra.mode == "read-only"`` and experiments that
    are already completed (or otherwise non-submittable per
    :func:`_should_submit_experiment`) are filtered out before packing. If
    nothing is pending, an empty list is returned.

    Each child experiment is cloned with ``infra.cluster`` forced to ``None``
    so that it executes in-process inside the packed scheduler job rather
    than being re-submitted to the cluster. The packed scheduler job itself
    has ``mode`` forced to ``"force"`` to guarantee submission of the
    aggregate job.

    Limitations
    -----------
    - The scheduler-job resource budget (CPUs, GPUs, memory, walltime,
      partition) is taken from the first pending experiment only. A warning
      is logged if peers declare different resource hints. Group experiments
      with matching resource hints before packing to avoid under-provisioning.
    - With ``n_jobs > 1`` and ``gpus_per_node > 0``, parallel children share
      the same set of visible GPUs (no automatic ``CUDA_VISIBLE_DEVICES``
      slicing). Use ``n_jobs == 1`` for GPU-bound experiments unless you
      partition devices yourself.

    Parameters
    ----------
    experiments
        Sequence of experiments to consider for packing.
    experiments_per_job
        How many experiments to place into each scheduler job. Must be an
        integer ``>= 1`` or the literal string ``"all"`` to put every
        pending experiment into a single job.
    n_jobs
        Local parallelism inside each packed scheduler job. ``n_jobs == 1``
        (the default) runs the packed experiments serially in-process;
        ``n_jobs > 1`` runs them in a ``ProcessPoolExecutor`` with a
        ``"spawn"`` start method (CUDA-safe and Slurm-cgroup-aware via
        ``os.sched_getaffinity``).

    Returns
    -------
    list[PackedExperiment]
        One :class:`PackedExperiment` per scheduler job, or an empty list
        when no experiments are pending.
    """
    _validate_experiments_per_job(experiments_per_job)
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    # Compute ``infra.uid()`` once per pending experiment so the sort doesn't
    # re-serialize. The UID lookup involves a full pydantic dump and is the
    # dominant cost for large grids.
    pending_keyed: list[tuple[str, BaseExperiment]] = [
        (exp.infra.uid(), exp)
        for exp in experiments
        if _should_submit_experiment(exp)
    ]
    if not pending_keyed:
        return []
    # Sort by per-experiment UID so the packed cache key is independent of
    # the caller-provided iteration order. Without this, the same set of
    # experiments handed in a different order produces different packed UIDs
    # and forces unnecessary re-runs.
    pending_keyed.sort(key=lambda kv: kv[0])
    pending = [exp for _, exp in pending_keyed]

    _warn_heterogeneous_resources(pending)

    scheduler_infra = pending[0].infra.model_dump(
        mode="python",
        exclude_computed_fields=True,
        exclude_defaults=True,
        exclude_unset=True,
    )
    scheduler_infra["mode"] = "force"

    step = len(pending) if experiments_per_job == "all" else experiments_per_job
    return [
        PackedExperiment(
            experiments=[exp.infra.clone_obj({"infra.cluster": None}) for exp in group],
            infra=scheduler_infra,
            n_jobs=n_jobs,
        )
        for group in (
            pending[start : start + step] for start in range(0, len(pending), step)
        )
    ]
