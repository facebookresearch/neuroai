# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Grid over different configurations of the MAE pretraining experiment."""

import exca

from neuraltrain.utils import run_grid

from ..main import Experiment  # type: ignore
from .defaults import PROJECT_NAME, SAVEDIR, default_config  # type: ignore

GRID_NAME = "mae_pretrain1"

update = {
    "infra": {
        "cluster": "auto",
        "folder": SAVEDIR,
        "slurm_partition": "learnfair",
        "timeout_min": 120,
        "gpus_per_node": 1,
        "cpus_per_task": 10,
        "job_name": PROJECT_NAME,
    },
}

# `patch_size` and `dim` are deliberately not swept: `neuralbench`'s `mae.yaml`
# pins both, and the CLI cannot override them, so a checkpoint that disagrees
# loads into a randomly initialised input layer without failing.  Change them in
# `defaults.py` and in `mae.yaml` together instead.
grid = {
    "mask_ratio": [0.25, 0.5, 0.75],
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
