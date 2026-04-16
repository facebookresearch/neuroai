# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Default configuration dictionary for project template experiment on MNE sample dataset."""

from pathlib import Path

import neuralset as ns

PROJECT_NAME = "mne_sample_clf"
CACHEDIR = f"{ns.CACHE_FOLDER}/cache/{PROJECT_NAME}"
SAVEDIR = f"{ns.CACHE_FOLDER}/results/{PROJECT_NAME}"
DATADIR = f"{ns.CACHE_FOLDER}/data/mne2013sample"
for path in [CACHEDIR, SAVEDIR, DATADIR]:
    Path(path).mkdir(parents=True, exist_ok=True)

default_config = {
    "infra": {
        "cluster": None,  # Run example locally
        "folder": SAVEDIR,
        "gpus_per_node": 1,
        "cpus_per_task": 10,
    },
    "data": {
        "study": [
            {
                "name": "Mne2013Sample",
                "path": DATADIR,
                "query": None,
                "infra": {"backend": "Cached", "folder": CACHEDIR},
            },
            {
                "name": "SklearnSplit",
                "valid_split_ratio": 0.2,
                "test_split_ratio": 0.2,
                "valid_random_state": 87,
                "split_by": "_index",
            },
        ],
        "segmenter": {
            "extractors": {
                "input": {
                    "name": "MegExtractor",
                    "frequency": 120.0,
                    "filter": (0.5, 25.0),
                    "baseline": (0.0, 0.1),
                    "scaler": "RobustScaler",
                    "clamp": 16.0,
                    "infra": {
                        "keep_in_ram": True,
                        "folder": CACHEDIR,
                        "cluster": None,
                    },
                },
                "target": {
                    "name": "EventField",
                    "event_types": "Stimulus",
                    "event_field": "code",
                },
            },
            "trigger_query": "type == 'Stimulus'",
            "start": -0.1,
            "duration": 0.5,
        },
        "batch_size": 16,
    },
    "brain_model_config": {
        "name": "EEGNet",
        "kwargs": {"D": 2, "drop_prob": 0.5, "sfreq": 120, "input_window_seconds": 0.5},
    },
    "metrics": [
        {
            "log_name": "acc",
            "name": "Accuracy",
            "kwargs": {"task": "multiclass", "num_classes": 4},  # XXX Depends on data
        },
    ],
    "loss": {"name": "CrossEntropyLoss"},
    "optim": {
        "optimizer": {
            "name": "Adam",
            "lr": 1e-4,
            "kwargs": {
                "weight_decay": 0.0,
            },
        },
        "scheduler": {
            "name": "OneCycleLR",
            "kwargs": {
                "max_lr": 3e-3,
                "pct_start": 0.2,
            },
        },
    },
    "csv_config": {
        "name": PROJECT_NAME,
        "flush_logs_every_n_steps": 100,
    },
    "wandb_config": {
        "log_model": False,
        "group": PROJECT_NAME,
        "project": PROJECT_NAME,
    },
    "save_checkpoints": True,
    "n_epochs": 20,
    "limit_train_batches": None,
    "patience": 5,
    "fast_dev_run": False,
    "seed": 33,
}


if __name__ == "__main__":
    # The following can be used for local debugging/quick tests.

    from ..main import Experiment

    exp = Experiment(
        **default_config,
    )

    exp.infra.clear_job()
    out = exp.run()
    print(out)
