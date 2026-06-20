# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Optuna hyperparameter search for the stimulus classification experiment.

Adaptive (TPE) sibling of ``run_grid.py``: instead of a fixed grid, trials are sampled
to minimise the objective, with underperforming trials pruned mid-training. Workers pull
from a shared RDB study, so the search parallelises across processes / slurm jobs.
"""

import exca

from neuraltrain.utils import run_optuna

from ..main import Experiment  # type: ignore
from .defaults import SAVEDIR, default_config  # type: ignore

STUDY_NAME = "optuna_search1"

update = {
    "infra": {
        # cluster=None runs trials in-process (n_workers=1). For parallel workers set
        # cluster="auto"/"local"/"slurm" — run_optuna builds the executor from this infra
        # block, the same vocabulary run_grid uses (so slurm_partition etc. live here).
        "cluster": None,
        "folder": SAVEDIR,
        "gpus_per_node": 1,
        "cpus_per_task": 10,
        # "slurm_partition": "learnfair",  # uncomment with cluster="slurm"
        # "timeout_min": 60,
    },
    "patience": 15,
    "save_checkpoints": False,
    # No wandb per-trial by default; Optuna's study is the record of truth.
    "wandb_config": None,
}


def search_space(trial) -> dict:
    """Map an Optuna trial to flattened-config overrides (dotted keys, as in run_grid)."""
    return {
        "optim.optimizer.lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "optim.optimizer.kwargs.weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-2, log=True
        ),
        "brain_model_config.kwargs.D": trial.suggest_categorical("D", [2, 4, 8, 16]),
        "brain_model_config.kwargs.drop_prob": trial.suggest_float("drop_prob", 0.1, 0.6),
        "data.batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


if __name__ == "__main__":
    updated_config = exca.ConfDict(default_config)
    updated_config.update(update)

    study = run_optuna(
        Experiment,
        STUDY_NAME,
        updated_config,
        search_space,
        metric="test_acc",  # key in Experiment.run() result; "maximize" accuracy
        direction="maximize",
        n_trials=30,
        # n_workers>1 needs infra.cluster set above (workers share the RDB study).
        n_workers=1,
        prune=True,
        monitor="val_acc",  # must increase with `direction="maximize"` (NOT val_loss)
        overwrite=True,
        # storage="postgresql://user:pass@host/optuna",  # for multi-node parallelism
    )

    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
