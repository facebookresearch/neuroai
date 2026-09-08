# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Quick test run on reduced data and number of epochs for CI."""

import copy

from exca import ConfDict

import neuralset as ns

from ..main import Experiment  # type: ignore
from .defaults import CACHEDIR, default_config  # type: ignore

update = {
    "infra.cluster": None,
    "accelerator": "cpu",
    "fast_dev_run": True,
    "wandb_config": None,
}


def debug_config() -> dict:
    """`default_config` with the challenge datasets swapped for a bundled one."""
    config = copy.deepcopy(default_config)
    config["data"]["studies"] = [  # type: ignore[index]
        [
            {
                "name": "Mne2013SampleEeg",
                # use same folder as in neuralset tests to share cache:
                "path": ns.CACHE_FOLDER,
                "query": None,
                "infra": {"backend": "Cached", "folder": CACHEDIR},
            }
        ]
    ]
    # This recording names its channels "EEG 001"..., which no standard montage
    # knows; its own coordinates are set, so read them from the file instead.
    config["data"]["channel_positions"]["layout_or_montage_name"] = None  # type: ignore[index]
    return config


def test_run(config: dict) -> None:
    task = Experiment(**config)
    task.infra.clear_job()
    task.run()
    # The point of pretraining is the encoder it leaves behind.
    assert task.checkpoint_path.exists()


if __name__ == "__main__":
    updated_config = ConfDict(debug_config())
    updated_config.update(update)
    test_run(updated_config)
