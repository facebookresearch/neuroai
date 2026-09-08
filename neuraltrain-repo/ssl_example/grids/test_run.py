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
            },
            # only the recording matters here: it is the trigger, and there is no
            # target, so dropping the stimuli makes the split below exact
            {"name": "QueryEvents", "query": "type == 'Eeg'"},
            # this study is one subject in one recording, which no grouped split
            # can divide; chunking it yields pseudo-recordings that it can
            {
                "name": "ChunkEvents",
                "event_type_to_chunk": "Eeg",
                "max_duration": 24.0,
                "tiling": "equal",
            },
            # the extractor preprocesses every chunk asked of it, so dropping all
            # but the first five is what keeps this run cheap; they still yield
            # about a batch of windows, and the preprocessing stays the shipped one
            {"name": "SelectIdx", "column": "start", "idx": [0, 1, 2, 3, 4]},
            {
                "name": "SklearnSplit",
                "split_by": "_index",
                "valid_split_ratio": 0.2,
                "test_split_ratio": 0.2,
            },
        ]
    ]
    # channels named "EEG 001"...: no montage knows them, read coords from the file
    config["data"]["channel_positions"]["layout_or_montage_name"] = None  # type: ignore[index]
    return config


def test_run(config: dict) -> None:
    task = Experiment(**config)
    task.infra.clear_job()
    task.run()
    assert task.checkpoint_path.exists(), "pretraining left no encoder behind"


if __name__ == "__main__":
    updated_config = ConfDict(debug_config())
    updated_config.update(update)
    test_run(updated_config)
