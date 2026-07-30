# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch
from torch.utils.data import DataLoader

import neuralset as ns
from neuralset.events.transforms import SklearnSplit

from . import config_manager, get_default_dataloaders


def _explicit_mne_loaders(study_path: Path) -> dict[str, DataLoader]:
    """Build the ``eeg/audiovisual_stimulus`` loaders the long way: bare study
    -> SklearnSplit -> extractors -> Segmenter -> one DataLoader per split.

    This is the reference ``get_default_dataloaders`` has to reproduce, written
    against the explicit neuralset API so the two paths share no code.
    """
    study = ns.Study(name="Mne2013SampleEeg", path=study_path)
    study.download()  # fetch the MNE sample data if not already present
    events = study.run()  # raw events -- no ``split`` column yet
    events = SklearnSplit(
        split_by="_index",
        valid_split_ratio=0.2,
        test_split_ratio=0.2,
        valid_random_state=33,
        test_random_state=33,
        stratify_by="description",
    )(events)

    neuro = ns.extractors.EegExtractor(
        picks=("eeg",),
        frequency=120.0,
        filter=(0.1, 75.0),
        notch_filter=[50.0, 60.0],
        baseline=(0.0, 0.2),
        scaler="RobustScaler",
        clamp=20.0,
    )
    target = ns.extractors.LabelEncoder(
        event_types="Stimulus",
        event_field="description",
        return_one_hot=True,
        aggregation="trigger",
    )
    neuro.prepare(events)  # channel positions depend on the prepared extractor
    segmenter = ns.dataloader.Segmenter(
        start=-0.2,
        duration=1.0,
        trigger_query="type in ['Stimulus']",
        extractors={
            "neuro": neuro,
            "target": target,
            "channel_positions": ns.extractors.ChannelPositions(
                n_spatial_dims=3,
                include_ref_eeg=False,
            ).build(neuro),
        },
    )
    dataset = segmenter.apply(events)
    dataset.prepare()

    return {
        split: DataLoader(
            sub := dataset.select(dataset.triggers.split == split),
            collate_fn=sub.collate_fn,
            batch_size=64,
            shuffle=False,
        )
        for split in ("train", "val", "test")
    }


def test_get_default_dataloaders_matches_explicit_mne_pipeline() -> None:
    """``get_default_dataloaders("eeg", "audiovisual_stimulus")`` must produce the
    same segments and tensors as the hand-built neuralset pipeline.

    Compares the *test* split, unshuffled on both sides so the batches line up
    element-wise.
    """
    study_path = Path(config_manager.DATA_DIR) / "Mne2013SampleEeg"
    # ``DATA_DIR`` is relative by default, and neuralset requires a study's
    # parent folder to already exist -- create it so a fresh checkout works.
    study_path.parent.mkdir(parents=True, exist_ok=True)
    # Build the reference first: it downloads the MNE sample study if missing,
    # so the config-driven path below always has data to read.
    expected = _explicit_mne_loaders(study_path)
    loaders = get_default_dataloaders("eeg", "audiovisual_stimulus")

    assert {s: len(ld.dataset) for s, ld in loaders.items()} == {  # type: ignore[arg-type]
        s: len(ld.dataset)  # type: ignore[arg-type]
        for s, ld in expected.items()
    }

    batch = next(iter(loaders["test"]))
    expected_batch = next(iter(expected["test"]))
    # ``Data`` additionally extracts ``subject_id`` for every segment.
    assert set(batch.data) == set(expected_batch.data) | {"subject_id"}
    for key in ("neuro", "target", "channel_positions"):
        torch.testing.assert_close(batch.data[key], expected_batch.data[key])

    # 60 EEG channels, 1.0 s at 120 Hz, 4 stimulus classes.
    assert batch.data["neuro"].shape[1:] == (60, 120)
    assert batch.data["target"].shape[1:] == (4,)
    assert batch.data["channel_positions"].shape[1:] == (60, 3)
