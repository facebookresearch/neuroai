# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import neuralset as ns

from .transforms import (  # noqa: F401
    AddDefaultEvents,
    CropSleepRecordings,
    CropTimelines,
    OffsetEvents,
    PredefinedSplit,
    ShuffleTrainingLabels,
    SimilaritySplit,
    SklearnSplit,
    TextPreprocessor,
)
from .utils import make_weighted_sampler, seed_worker

LOGGER = logging.getLogger(__name__)


class Data(ns.BaseModel):
    """Create dataloaders for brain-modeling experiments."""

    study: ns.Step
    neuro: ns.extractors.BaseExtractor
    target: ns.extractors.BaseExtractor
    channel_positions: ns.extractors.ChannelPositions
    # Segments
    trigger_event_type: str | list[str]
    start: float = -0.5
    duration: float | None = 3
    stride: float | None = None
    stride_drop_incomplete: bool = True
    # Dataloaders
    use_weighted_sampler: bool = False
    batch_size: int = 64
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = None
    seed: int | None = None
    # Others
    summary_columns: list[str] = []

    _subject_id: ns.extractors.LabelEncoder | None = None

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._subject_id = ns.extractors.LabelEncoder(
            event_types=self.neuro.event_types,
            event_field="subject",
            return_one_hot=False,
        )

    def prepare(self) -> dict[str, DataLoader]:
        """Load events, build extractors, segment data and return train/val/test DataLoaders.

        Returns
        -------
        dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to
        :class:`~torch.utils.data.DataLoader` instances.
        """
        events = self.study.run()
        if "split" not in events.columns:
            LOGGER.error(
                "No `split` column found in events. Make sure splits are defined in the study, "
                "or use an events transform (`neuralset.events.transforms`) to add them."
            )

        summary_columns = ["index", "subject"] + self.summary_columns + ["timeline"]
        summary_df = (
            events.reset_index()
            .groupby(["study", "split", "type"], dropna=False)[summary_columns]
            .nunique()
        )
        LOGGER.info("Dataset summary:\n%s", summary_df.to_string())

        extractors = {
            "neuro": self.neuro,
            "target": self.target,
            "subject_id": self._subject_id,
        }

        if isinstance(self.neuro, ns.extractors.MneRaw):
            # Prepare the neuro extractor first because the channel positions depend on it
            self.neuro.prepare(events)
            channels = self.neuro._channels
            assert channels is not None
            channel_positions = self.channel_positions.build(self.neuro)
            LOGGER.info(
                f"Found {len(channels)} different channels: {list(channels.keys())}"
            )
            extractors["channel_positions"] = channel_positions

        trigger_event_type = (
            [self.trigger_event_type]
            if isinstance(self.trigger_event_type, str)
            else self.trigger_event_type
        )
        segmenter = ns.dataloader.Segmenter(
            start=self.start,
            duration=self.duration,
            trigger_query=f"type in {trigger_event_type}",
            stride=self.stride,
            stride_drop_incomplete=self.stride_drop_incomplete,
            extractors=extractors,  # type: ignore[arg-type]
        )
        dataset = segmenter.apply(events)
        dataset.prepare()

        # Derive four independent RNG streams from ``self.seed`` so that each
        # consumer (train DataLoader shuffle + train worker base-seeds, train
        # WeightedRandomSampler multinomial draws, val worker base-seeds,
        # test worker base-seeds) is a pure function of its own sub-seed.
        # Per-split DataLoader generators matter when ``num_workers > 0``:
        # ``DataLoader.__iter__`` consumes one int64 from ``generator`` to
        # derive each worker's base seed, so sharing one generator across
        # splits would couple the train shuffle stream to how often Lightning
        # iterates val/test (sanity check, per-epoch validation, etc.).
        loader_gens: dict[str, torch.Generator | None]
        sampler_gen: torch.Generator | None
        worker_init_fn: tp.Callable[[int], None] | None
        if self.seed is None:
            LOGGER.info(
                "Data.seed=None; dataloader shuffling and weighted sampling "
                "will follow the caller's global torch RNG state."
            )
            loader_gens = {"train": None, "val": None, "test": None}
            sampler_gen = None
            worker_init_fn = None
        else:
            train_state, sampler_state, val_state, test_state = np.random.SeedSequence(
                self.seed
            ).generate_state(4)
            loader_gens = {
                "train": torch.Generator().manual_seed(int(train_state)),
                "val": torch.Generator().manual_seed(int(val_state)),
                "test": torch.Generator().manual_seed(int(test_state)),
            }
            sampler_gen = torch.Generator().manual_seed(int(sampler_state))
            worker_init_fn = seed_worker

        # Create the dataloaders
        loaders = {}
        for split in tqdm(["train", "val", "test"], desc="Preparing segments"):
            split_dataset = dataset.select(dataset.triggers.split == split)
            LOGGER.info(f"# {split} segments: {len(split_dataset)} \n")

            sampler = None
            if split == "train" and self.use_weighted_sampler:
                sampler = make_weighted_sampler(
                    split_dataset, logger=LOGGER, generator=sampler_gen
                )

            persistent_workers = self.persistent_workers and self.num_workers > 0
            loaders[split] = DataLoader(
                split_dataset,
                collate_fn=split_dataset.collate_fn,
                batch_size=self.batch_size,
                shuffle=split == "train" and sampler is None,
                sampler=sampler,
                num_workers=self.num_workers,
                drop_last=self.drop_last and split == "train",
                pin_memory=self.pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                generator=loader_gens[split],
                worker_init_fn=worker_init_fn,
            )

        return loaders
