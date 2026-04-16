# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Grid over different configurations of stimulus classification experiment."""

import exca

from neuraltrain.utils import run_grid

from ..main import Experiment  # type: ignore
from .defaults import PROJECT_NAME, SAVEDIR, default_config  # type: ignore

GRID_NAME = "hp_search_light1"

update = {
    "infra": {
        "cluster": "auto",
        "folder": SAVEDIR,
        "slurm_partition": "learnfair",
        "timeout_min": 60,
        "gpus_per_node": 1,
        "cpus_per_task": 10,
        "job_name": PROJECT_NAME,
    },
    "patience": 15,
}

grid = {
    "brain_model_config.kwargs.D": [2, 4, 16],
    "n_epochs": [100, 0],  # n_epochs=0 for chance-level
    "seed": [33, 87],
}


if __name__ == "__main__":
    updated_config = exca.ConfDict(default_config)
    updated_config.update(update)

    out = run_grid(
        Experiment,
        GRID_NAME,
        updated_config,
        grid,  # type: ignore
        job_name_keys=["infra.job_name"],
        combinatorial=True,
        overwrite=True,
        dry_run=False,
        infra_mode="force",
    )
