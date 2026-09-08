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
# Studies each claim their own subfolder of this root.
DATADIR = f"{ns.CACHE_FOLDER}/data"
for path in [CACHEDIR, SAVEDIR, DATADIR]:
    Path(path).mkdir(parents=True, exist_ok=True)

# Common sampling rate for every study. 100 Hz is the lowest of the four, so
# nothing is upsampled, and 8 s * 100 Hz // 32 = 25 patches per window gives the
# encoder enough tokens to mask over.
FREQUENCY = 100.0
WINDOW = 8.0

# The EEG datasets of tracks 1-3, plus a resting-state one that belongs to no
# track. Track 4 is EMG, which shares no sensor space with any of these.
# Pretraining pools them raw: the model sees no labels, so a dataset is worth
# including for its signal alone.
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
        # No split transform here: `Data` splits the strided windows in time.
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
            # Slide a window across the whole recording instead of cutting at
            # events: self-supervision needs no labels, so every sample counts.
            # The trigger is the recording itself, not a stimulus.
            "trigger_query": "type == 'Eeg'",
            "stride": WINDOW,
            "duration": WINDOW,
        },
        # What lets one encoder read four montages. Naming the montage matters
        # for Sleep-EDF, whose bipolar derivations (Fpz-Cz, Pz-Oz) carry no
        # coordinates of their own: the extractor falls back to the part before
        # the dash and finds Fpz and Pz here.
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
        # `merger_config` is left at its default: naming it here would rebuild
        # it from `ChannelMerger`'s own defaults, which embed 2D positions,
        # and every field would then have to be restated in `mae.yaml` too.
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
    # Drop to None to train without Weights & Biases; nothing else depends on it.
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
