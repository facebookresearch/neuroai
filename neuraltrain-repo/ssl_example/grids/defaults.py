# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Default configuration for MAE pretraining on the MNE sample EEG dataset."""

from pathlib import Path

import neuralset as ns

PROJECT_NAME = "mne_sample_mae"
CACHEDIR = f"{ns.CACHE_FOLDER}/cache/{PROJECT_NAME}"
SAVEDIR = f"{ns.CACHE_FOLDER}/results/{PROJECT_NAME}"
DATADIR = f"{ns.CACHE_FOLDER}/data/mne2013sample"
for path in [CACHEDIR, SAVEDIR, DATADIR]:
    Path(path).mkdir(parents=True, exist_ok=True)

# Window length in seconds. At 120 Hz and patch_size=32 this gives
# 4 s * 120 Hz // 32 = 15 patches per window for the encoder to mask over.
WINDOW = 4.0

default_config = {
    "infra": {
        "cluster": None,  # Run example locally
        "folder": SAVEDIR,
        "gpus_per_node": 1,
        "cpus_per_task": 10,
    },
    "data": {
        # No split transform here: `Data` splits the strided windows in time,
        # since this dataset is a single continuous recording.
        "study": [
            {
                "name": "Mne2013SampleEeg",
                "path": DATADIR,
                "query": None,
                "infra": {"backend": "Cached", "folder": CACHEDIR},
            },
        ],
        "segmenter": {
            "extractors": {
                # No "target" extractor: the input is its own target.
                "input": {
                    "name": "EegExtractor",
                    "frequency": 120.0,
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
        "val_ratio": 0.2,
        "batch_size": 16,
    },
    "brain_model_config": {
        "name": "MaeEncoder",
        "dim": 256,
        "patch_size": 32,
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
