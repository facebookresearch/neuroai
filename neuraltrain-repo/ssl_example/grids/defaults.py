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

# preprocessing below matches `neuralbench/defaults/config.yaml`, so the encoder
# sees the same signal downstream as it does here
FREQUENCY = 120.0
WINDOW = 4.0  # 15 patches of 32 samples, hence 15 * n_channels tokens

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
        "studies": [
            [
                {
                    "name": name,
                    "path": DATADIR,
                    "query": None,
                    "infra": {"backend": "Cached", "folder": CACHEDIR},
                },
                # whole subjects held out, so validation measures generalisation to
                # an unseen recording; pretraining leaves the test split untouched
                {
                    "name": "SklearnSplit",
                    "split_by": "subject",
                    "valid_split_ratio": 0.1,
                    "test_split_ratio": 0.1,
                    "valid_random_state": 33,
                    "test_random_state": 33,
                },
            ]
            for name in STUDIES
        ],
        "segmenter": {
            "extractors": {
                # No "target" extractor: the input is its own target.
                "input": {
                    "name": "EegExtractor",
                    "frequency": FREQUENCY,
                    "filter": (0.1, 75.0),
                    "notch_filter": [50.0, 60.0],
                    "scaler": "RobustScaler",
                    "clamp": 20.0,
                    "infra": {
                        "keep_in_ram": True,
                        "folder": CACHEDIR,
                        "cluster": None,
                    },
                },
            },
            # the recording itself is the trigger: windows tile it, no events needed
            "trigger_query": "type == 'Eeg'",
            "stride": WINDOW,
            "duration": WINDOW,
        },
        # standard_1020 also resolves Sleep-EDF's bipolar names (Fpz-Cz -> Fpz)
        "channel_positions": {
            "n_spatial_dims": 3,
            "layout_or_montage_name": "standard_1020",
            # positions are padded to the channel union of the studies, which is
            # not part of the cache key: kept on disk, they would be served at the
            # width of whichever study set filled the cache first. Recomputing
            # them costs seconds, and `keep_in_ram` still spares repeated reads.
            "infra": {
                "keep_in_ram": True,
                "folder": None,
                "cluster": None,
            },
        },
        "batch_size": 16,
    },
    "brain_model_config": {
        "name": "MaeEncoder",
        "dim": 256,
        "patch_size": 32,
        # `channel_emb_config` left at its default: naming it resets n_dims to 2
    },
    "mask_ratio": 0.5,
    # the module scores the hidden patches only, so a plain MSE is the MAE loss
    "loss": {"name": "MSELoss"},
    "optim": {
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-4,
            "kwargs": {"weight_decay": 0.05},
        },
        "scheduler": {
            "name": "OneCycleLR",
            "kwargs": {"max_lr": 1e-3, "pct_start": 0.2},
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
