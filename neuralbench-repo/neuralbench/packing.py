# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pack pending experiments into fewer scheduler jobs via exca's ``Parallel``.

One job per experiment hits array limits on large grids (facebookresearch/exca#179).
This dispatches the pending experiments as one ``Parallel`` sweep
(facebookresearch/exca#280) into ``ceil(pending / N)`` array elements. Each variant
delegates to the experiment's own ``run()``, so results still cache under each
experiment's ``TaskInfra`` — re-running without packing reuses them.
"""

from __future__ import annotations

import logging
import math
import typing as tp

import pydantic
from exca.steps import backends
from exca.steps.base import Step
from exca.steps.helpers import Parallel

from neuraltrain.utils import BaseExperiment

LOGGER = logging.getLogger(__name__)

# TaskInfra.cluster -> steps backend class (the values neuralbench emits).
_BACKENDS = {None: "Cached", "auto": "Auto", "slurm": "Slurm"}


class _ExperimentStep(Step):
    """A steps wrapper delegating to a ``TaskInfra`` experiment's ``run()``.

    Identity is the experiment's uid (``exp_uid``); the experiment rides in a
    ``PrivateAttr`` because ``BaseExperiment`` is not a discriminated model and
    would not survive ``Parallel``'s step re-validation as a field — a private
    attr is excluded from the uid yet pickled to workers. A failure is isolated
    (logged, not raised) so it does not abort the rest of the packed element; it
    is still recorded in the experiment's own ``TaskInfra`` (status ``failed``).
    """

    exp_uid: str
    _experiment: BaseExperiment = pydantic.PrivateAttr()

    def bind(self, experiment: BaseExperiment) -> _ExperimentStep:
        self._experiment = experiment
        return self

    def _run(self, value: tp.Any = None) -> None:
        try:
            self._experiment.run()  # caches under the experiment's own TaskInfra
        except Exception:
            LOGGER.exception("Packed experiment %s failed; skipping", self.exp_uid)


def _should_submit_experiment(experiment: BaseExperiment) -> bool:
    """Whether an experiment needs (re-)submission (TaskInfra status/mode filter)."""
    status = experiment.infra.status()
    mode = experiment.infra.mode
    if mode == "read-only":
        return False
    return (
        status == "not submitted"
        or mode == "force"
        or (status == "failed" and mode == "retry")
    )


def _backend_config(infra: tp.Any, folder: tp.Any, max_jobs: int) -> dict[str, tp.Any]:
    """Translate one experiment's ``TaskInfra`` into a steps backend config.

    Resources come from a single infra (group by resources before packing if they
    differ). Keeps every infra field the backend also accepts — ``TaskInfra``
    prefixes slurm-only fields (``slurm_partition``), the backend does not.
    ``mode="force"`` makes the wrapper marker never the cache authority: every
    pending variant is dispatched and the experiment's own ``TaskInfra`` decides
    what recomputes.
    """
    cluster = getattr(infra, "cluster", None)
    if cluster not in _BACKENDS:
        raise ValueError(f"unsupported infra.cluster={cluster!r}")
    name = _BACKENDS[cluster]
    fields = getattr(backends, name).model_fields
    shared = {
        k.removeprefix("slurm_"): v
        for k, v in infra.model_dump(exclude_defaults=True).items()
    }
    cfg = {k: v for k, v in shared.items() if k in fields}
    cfg |= {"backend": name, "folder": folder, "mode": "force"}
    if name != "Cached":  # Cached runs in-process; no packing knob
        cfg["max_jobs"] = max_jobs
    return cfg


def submit_packed(
    experiments: tp.Sequence[BaseExperiment],
    experiments_per_job: int | tp.Literal["all"] = "all",
) -> int:
    """Dispatch pending experiments as one ``Parallel`` sweep; return the count.

    ``experiments_per_job``: ``"all"`` packs everything into one job; int ``N``
    caps the sweep at ``ceil(pending / N)`` array elements. Returns 0 when every
    runnable experiment is already cached.
    """
    pending = [e for e in experiments if _should_submit_experiment(e)]
    if not pending:
        return 0
    n = len(pending)
    per_job = n if experiments_per_job == "all" else int(experiments_per_job)
    if per_job < 1:
        raise ValueError(f"experiments_per_job must be >= 1 or 'all', got {per_job}")
    infra = pending[0].infra
    if infra.folder is None:
        raise RuntimeError("experiments need infra.folder set to pack")
    # tp.Any: the dict is validated into a steps Backend at construction.
    backend: tp.Any = _backend_config(infra, infra.folder, math.ceil(n / per_job))
    variants = [
        # cluster=None so run() computes in-process in the worker (not a nested
        # submission); cluster is excluded from the uid, so caching is unchanged.
        _ExperimentStep(exp_uid=e.infra.uid(), infra=backend).bind(
            e.infra.clone_obj({"infra.cluster": None})
        )
        for e in pending
    ]
    Parallel(steps=variants, infra=backend).run()
    return n
