# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Module-level workload classes for ``bench_pack_scaling.py``.

These live in their own module (not in ``__main__``) so spawned worker
processes can re-import them by qualname — that's how ``pickle`` resolves
class identity across the ``ProcessPoolExecutor(mp_context='spawn')``
boundary.
"""

import exca
from neuraltrain.utils import BaseExperiment


class SleepExperiment(BaseExperiment):
    """Sleeps ``work_seconds`` then returns the seed. I/O-bound."""

    seed: int
    work_seconds: float = 1.5
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def run(self) -> float:
        import time as _time

        _time.sleep(self.work_seconds)
        return float(self.seed)


class CpuExperiment(BaseExperiment):
    """Spends ``work_seconds`` doing numpy matmul, single BLAS thread."""

    seed: int
    work_seconds: float = 1.5
    matrix: int = 250
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def run(self) -> float:
        import time as _time

        import numpy as _np

        rng = _np.random.default_rng(self.seed)
        a = rng.standard_normal((self.matrix, self.matrix))
        end = _time.perf_counter() + self.work_seconds
        s = 0.0
        while _time.perf_counter() < end:
            a = a @ a
            a /= max(_np.linalg.norm(a), 1e-6)
            s += float(a.sum())
        return s
