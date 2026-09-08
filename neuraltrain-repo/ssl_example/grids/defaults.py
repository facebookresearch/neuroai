# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Default configuration for MAE pretraining on the challenge datasets."""

from pathlib import Path

import neuralset as ns

PROJECT_NAME = "challenge_mae"
CACHEDIR = f"{ns.CACHE_FOLDER}/cache/{PROJECT_NAME}"
SAVEDIR = f"{ns.CACHE_FOLDER}/results/{PROJECT_NAME}"
# each study claims its own subfolder
DATADIR = f"{ns.CACHE_FOLDER}/data"
for path in [CACHEDIR, SAVEDIR, DATADIR]:
    Path(path).mkdir(parents=True, exist_ok=True)

FREQUENCY = 100.0  # lowest of the four studies, so nothing is upsampled
WINDOW = 4.0  # 12 patches of 32 samples, hence 12 * n_channels tokens

# tracks 1-3 plus a resting-state study; track 4 is EMG, a different sensor space
STUDIES = [
    "Gifford2022Large",  # track 1, image decoding, 63 ch
    "Stieger2021Continuous",  # track 2, motor imagery, 60 ch
    "Kemp2000Analysis",  # track 3, sleep staging, 2 bipolar derivations
    "Miltiadous2023Dice",  # resting-state, eyes closed, 19 ch
]

default_config = {
    "infra": {
        "cluster": None,  # Run example locally
        "folder": SAVEDIR,
        "gpus_per_node": 1,
        "cpus_per_task": 10,
    },
    "data": {
        # no split transform: `Data` splits the strided windows in time
        "studies": [
            [
                {
                    "name": name,
                    "path": DATADIR,
                    "query": None,
                    "infra": {"backend": "Cached", "folder": CACHEDIR},
                }
            ]
            for name in STUDIES
        ],
        "segmenter": {
            "extractors": {
                # No "target" extractor: the input is its own target.
                "input": {
                    "name": "EegExtractor",
                    "frequency": FREQUENCY,
                    "filter": (0.5, 25.0),
                    "scaler": "RobustScaler",
                    "clamp": 16.0,
                    "infra": {
                        "keep_in_ram": True,
                        "folder": CACHEDIR,
                        "cluster": None,
                    },
                },
            },
            # the recording itself is the trigger: windows slide, no events needed
            "trigger_query": "type == 'Eeg'",
            "stride": WINDOW,
            "duration": WINDOW,
        },
        # standard_1020 also resolves Sleep-EDF's bipolar names (Fpz-Cz -> Fpz)
        "channel_positions": {
            "n_spatial_dims": 3,
            "layout_or_montage_name": "standard_1020",
            "infra": {
                "keep_in_ram": True,
                "folder": CACHEDIR,
                "cluster": None,
            },
        },
        "val_ratio": 0.2,
        "batch_size": 16,
    },
    "brain_model_config": {
        "name": "MaeEncoder",
        "dim": 256,
        "patch_size": 32,
        # `channel_emb_config` left at its default: naming it resets n_dims to 2
    },
    "mask_ratio": 0.5,
    "loss": {"name": "MaskedReconstructionLoss"},
    "optim": {
        "optimizer": {
            "name": "Adam",
            "lr": 1e-4,
            "kwargs": {"weight_decay": 0.0},
        },
        "scheduler": {
            "name": "OneCycleLR",
            "kwargs": {"max_lr": 3e-3, "pct_start": 0.2},
        },
    },
    "csv_config": {
        "name": PROJECT_NAME,
        "flush_logs_every_n_steps": 100,
    },
    # set to None to train without Weights & Biases
    "wandb_config": {
        "log_model": False,
        "group": PROJECT_NAME,
        "project": PROJECT_NAME,
    },
    "n_epochs": 50,
    "limit_train_batches": None,
    "patience": 10,
    "fast_dev_run": False,
    "seed": 33,
}


if __name__ == "__main__":
    # The following can be used for local debugging/quick tests.

    from ..main import Experiment

    exp = Experiment(**default_config)

    exp.infra.clear_job()
    out = exp.run()
    print(out)
    print(f"Pretrained encoder: {exp.checkpoint_path}")
