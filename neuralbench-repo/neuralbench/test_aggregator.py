# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import exca

from neuraltrain.utils import BaseExperiment

from .aggregator import BenchmarkAggregator


class MiniExperiment(BaseExperiment):
    value: int
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def run(self) -> int:
        return self.value


def test_prepare_packs_experiments_and_preserves_child_caches(tmp_path: Path) -> None:
    infra: tp.Any = {
        "cluster": None,
        "folder": tmp_path / "experiments",
        "mode": "cached",
    }
    experiments = [MiniExperiment(value=i, infra=infra) for i in range(3)]

    agg = BenchmarkAggregator.model_construct(
        experiments=experiments,  # type: ignore[arg-type]
        max_workers=2,
        collect_max_workers=1,
        debug=False,
        experiments_per_job=2,
        local_workers_per_job=1,
        output_dir=str(tmp_path / "outputs"),
    )

    agg.prepare()

    assert [experiment.infra.status() for experiment in experiments] == [
        "completed",
        "completed",
        "completed",
    ]
    assert [experiment.run() for experiment in experiments] == [0, 1, 2]
