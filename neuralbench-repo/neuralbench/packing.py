# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pack pending experiments into fewer scheduler jobs."""

from __future__ import annotations

import multiprocessing
import typing as tp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import exca
from pydantic import Field, SerializeAsAny

from neuraltrain.base import BaseModel
from neuraltrain.utils import BaseExperiment


class PackedExperiment(BaseModel):
    """Run a batch of :class:`BaseExperiment` instances inside one scheduler job.

    The packed UID is derived from ``experiments`` and ``infra``; ``n_jobs`` is
    excluded so the cache key is insensitive to local-parallelism choices.
    ``SerializeAsAny`` keeps subclass-specific fields in the UID. Use
    :func:`pack_experiments_for_submission` to build instances.

    Not using :class:`exca.MapInfra` with ``cluster="processpool"`` here: that
    path drains in-flight tasks on failure, which delays error reporting by
    the length of the longest running experiment. We want first-failure
    cancellation, hence the manual pool + ``as_completed`` gather below.
    """

    experiments: list[SerializeAsAny[BaseExperiment]] = Field(min_length=1)
    n_jobs: int = Field(default=1, ge=1)

    # Bump ``version`` whenever :meth:`run` semantics change.
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["n_jobs"]

    @infra.apply
    def run(self) -> list[tp.Any]:
        if self.n_jobs == 1 or len(self.experiments) <= 1:
            return [exp.run() for exp in self.experiments]

        max_workers = min(self.n_jobs, len(self.experiments))
        ctx = multiprocessing.get_context("spawn")  # CUDA-safe
        # Manual lifecycle + ``as_completed`` so the first child failure
        # surfaces immediately and ``shutdown(cancel_futures=True)`` aborts
        # whatever is still pending — instead of waiting for the slowest
        # in-flight task to drain before reporting.
        pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
        try:
            futures = {pool.submit(exp.run): i for i, exp in enumerate(self.experiments)}
            results: list[tp.Any] = [None] * len(self.experiments)
            for future in as_completed(futures):
                results[futures[future]] = future.result()  # raises on first failure
            return results
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


def _should_submit_experiment(experiment: BaseExperiment) -> bool:
    """Whether an experiment needs (re-)submission.

    Mirrors the status-and-mode filter that :meth:`exca.TaskInfra.job_array`
    applies internally; kept inline so the pending set can be computed before
    handing experiments to the scheduler.
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


def pack_experiments_for_submission(
    experiments: tp.Sequence[BaseExperiment],
    experiments_per_job: int | tp.Literal["all"] = 1,
    n_jobs: int = 1,
) -> list[PackedExperiment]:
    """Pack pending experiments into fewer scheduler jobs.

    Experiments with ``infra.mode == "read-only"`` or already completed are
    filtered out (see :func:`_should_submit_experiment`). Each child is cloned
    with ``infra.cluster=None`` so it runs in-process inside the packed job,
    whose own ``mode`` is forced to ``"force"``.

    ``experiments_per_job`` is the per-job size (``"all"`` packs every pending
    experiment into one job). ``n_jobs`` controls in-process parallelism inside
    each packed job (``ProcessPoolExecutor`` with the ``spawn`` start method
    when ``> 1``). The scheduler-job resource budget comes from the first
    pending experiment.
    """
    if isinstance(experiments_per_job, str):
        if experiments_per_job != "all":
            raise ValueError(
                f"experiments_per_job must be 'all' or int >= 1; "
                f"got {experiments_per_job!r}."
            )
    elif not isinstance(experiments_per_job, int) or experiments_per_job < 1:
        raise ValueError(
            f"experiments_per_job must be 'all' or int >= 1; got {experiments_per_job!r}."
        )
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    # ``_should_submit_experiment`` calls ``infra.status()``, which may hit
    # the filesystem (uncached, NFS in the cluster case). Thread the check
    # so a 5000-experiment grid doesn't pay 5000 × stat-latency in series.
    n = len(experiments)
    if n == 0:
        return []
    with ThreadPoolExecutor(max_workers=min(32, n)) as pool:
        keep_flags = list(pool.map(_should_submit_experiment, experiments))

    # Materialize uid once per kept experiment, then sort so the packed cache
    # key is independent of caller iteration order.
    keyed = [(exp.infra.uid(), exp) for exp, keep in zip(experiments, keep_flags) if keep]
    if not keyed:
        return []
    keyed.sort(key=lambda kv: kv[0])
    pending = [exp for _, exp in keyed]

    scheduler_infra = pending[0].infra.model_copy(update={"mode": "force"})

    step = len(pending) if experiments_per_job == "all" else experiments_per_job
    return [
        PackedExperiment(
            experiments=[exp.infra.clone_obj({"infra.cluster": None}) for exp in group],
            infra=scheduler_infra,
            n_jobs=n_jobs,
        )
        for group in (pending[i : i + step] for i in range(0, len(pending), step))
    ]
