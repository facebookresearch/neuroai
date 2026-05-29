# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pack pending experiments into fewer scheduler jobs."""

from __future__ import annotations

import logging
import multiprocessing
import typing as tp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import exca
from pydantic import Field, SerializeAsAny

from neuraltrain.base import BaseModel
from neuraltrain.utils import BaseExperiment

LOGGER = logging.getLogger(__name__)


class PackedExperiment(BaseModel):
    """Run a batch of :class:`BaseExperiment` instances inside one scheduler job.

    The packed UID is derived from ``experiments`` and ``infra``; ``n_jobs`` and
    ``fault_isolated`` are excluded so the cache key is insensitive to runtime
    execution choices. ``SerializeAsAny`` keeps subclass-specific fields in the
    UID. Use :func:`pack_experiments_for_submission` to build instances.

    **Fault isolation (default).** A packed job bundles heterogeneous
    experiments — different models on different tasks. One model that is
    genuinely incompatible with a task's data (wrong channel count, an
    architecture that needs a longer window, a missing kwarg) must NOT abort
    the other 14 experiments sharing the scheduler job, nor cache a
    pack-level failure that masks every retry. With ``fault_isolated=True``
    each experiment runs independently; failures are logged (with traceback)
    and skipped, and the pack still returns successfully. Each ``exp.run()``
    is individually cached by exca, so successful experiments persist and only
    genuinely-broken ones are absent from the results.

    Set ``fault_isolated=False`` to restore strict fail-fast behaviour (first
    failure aborts the batch) — appropriate for a homogeneous grid where any
    failure is a real bug worth surfacing immediately.
    """

    experiments: list[SerializeAsAny[BaseExperiment]] = Field(min_length=1)
    n_jobs: int = Field(default=1, ge=1)
    fault_isolated: bool = True

    # Bump ``version`` whenever :meth:`run` semantics change. Kept at "1":
    # fault isolation only changes behaviour when an experiment *fails* (old
    # caches were all-success → identical return, or all-failure → already
    # cleared), so cached successes remain valid.
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["n_jobs", "fault_isolated"]

    def _run_one(self, idx: int, exp: BaseExperiment) -> tp.Any:
        """Run one experiment, isolating failures when configured."""
        try:
            return exp.run()
        except Exception:
            if not self.fault_isolated:
                raise
            LOGGER.exception(
                "PackedExperiment: experiment %d/%d failed (continuing; "
                "fault_isolated=True)",
                idx + 1,
                len(self.experiments),
            )
            return None

    @infra.apply
    def run(self) -> list[tp.Any]:
        if self.n_jobs == 1 or len(self.experiments) <= 1:
            results = [self._run_one(i, exp) for i, exp in enumerate(self.experiments)]
            self._log_failures(results)
            return results

        max_workers = min(self.n_jobs, len(self.experiments))
        ctx = multiprocessing.get_context("spawn")  # CUDA-safe
        # Manual lifecycle + ``as_completed`` gather. When fault_isolated, we
        # collect every result and never cancel pending work, so one bad
        # experiment cannot abort the rest. When not fault_isolated, the first
        # failure re-raises and ``cancel_futures=True`` aborts the remainder.
        pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
        results: list[tp.Any] = [None] * len(self.experiments)
        cancel = False
        try:
            futures = {pool.submit(exp.run): i for i, exp in enumerate(self.experiments)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    if not self.fault_isolated:
                        cancel = True
                        raise
                    LOGGER.exception(
                        "PackedExperiment: experiment %d/%d failed (continuing; "
                        "fault_isolated=True)",
                        idx + 1,
                        len(self.experiments),
                    )
            self._log_failures(results)
            return results
        finally:
            pool.shutdown(wait=True, cancel_futures=cancel)

    def _log_failures(self, results: list[tp.Any]) -> None:
        n_failed = sum(1 for r in results if r is None)
        if n_failed:
            LOGGER.warning(
                "PackedExperiment: %d/%d experiment(s) failed in this pack "
                "(see tracebacks above). Successful experiments are cached.",
                n_failed,
                len(self.experiments),
            )


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
