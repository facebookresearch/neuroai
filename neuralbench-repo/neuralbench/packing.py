# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pack pending experiments into fewer scheduler jobs."""

from __future__ import annotations

import multiprocessing
import typing as tp
from concurrent.futures import ProcessPoolExecutor

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
        # Manual lifecycle so we can cancel pending futures on the first
        # failure rather than wait for the longest in-flight task to drain.
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
    if experiments_per_job != "all" and (
        not isinstance(experiments_per_job, int) or experiments_per_job < 1
    ):
        raise ValueError(
            f"experiments_per_job must be 'all' or int >= 1; got {experiments_per_job!r}."
        )
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1.")

    # Materialize uid once per experiment, then sort so the packed cache key
    # is independent of caller iteration order.
    keyed = [
        (exp.infra.uid(), exp) for exp in experiments if _should_submit_experiment(exp)
    ]
    if not keyed:
        return []
    keyed.sort(key=lambda kv: kv[0])
    pending = [exp for _, exp in keyed]

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
        for group in (pending[i : i + step] for i in range(0, len(pending), step))
    ]
