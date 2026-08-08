# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import numpy as np
import torch
from exca import ConfDict
from pydantic import field_validator
from torch.utils.data import DataLoader
from tqdm import tqdm

import neuralset as ns

from .config_manager import _ensure_initialized
from .extractors import SleepOnsetTargetExtractor  # noqa: F401
from .registry import DEFAULTS_DIR, _resolve_task_dir, load_yaml_config
from .transforms import (  # noqa: F401
    AddDefaultEvents,
    AddSleepOnsetTargets,
    CropSleepRecordings,
    CropTimelines,
    OffsetEvents,
    PredefinedSplit,
    ShuffleTrainingLabels,
    SimilaritySplit,
    SklearnSplit,
    TextPreprocessor,
)
from .utils import make_regression_bin_sampler, make_weighted_sampler, seed_worker

LOGGER = logging.getLogger(__name__)


class BaseSampler(ns.base.NamedModel):
    """Base configuration for a train-time DataLoader sampler.

    Subclasses define how training examples are drawn from a
    :class:`~neuralset.dataloader.SegmentDataset` -- typically to counteract
    class or target imbalance -- by overriding :meth:`build`.  Concrete
    samplers are selected from YAML via the ``name`` discriminator, e.g.
    ``sampler: {name: ClassificationSampler}``.

    Only the training DataLoader uses the configured sampler; the val and
    test DataLoaders always iterate the dataset in order.
    """

    def build(
        self,
        dataset: ns.dataloader.SegmentDataset,
        generator: torch.Generator | None = None,
    ) -> torch.utils.data.Sampler:
        """Build the sampler for the training ``dataset``.

        Parameters
        ----------
        dataset
            Training segment dataset; subclasses typically materialise its
            targets to compute per-sample weights.
        generator
            Optional :class:`torch.Generator` consumed by the returned
            sampler.  When set, the sampling sequence depends only on the
            generator's seed -- not on the global ``torch`` RNG -- which is
            what :class:`~neuralbench.data.Data` relies on for reproducible
            training runs.
        """
        raise NotImplementedError


class ClassificationSampler(BaseSampler):
    """Inverse-frequency weighted sampler for class-imbalanced classification.

    Computes balanced class weights from the training targets and yields a
    :class:`torch.utils.data.WeightedRandomSampler` whose per-sample weights
    equal the inverse class frequency.  In expectation, every class
    contributes the same total mass to each training epoch.

    Use for multi-class classification with skewed label distributions
    (e.g. seizure detection, sleep arousal).  Multilabel targets are not
    yet supported -- see the TODO in :func:`make_weighted_sampler`.
    """

    def build(
        self,
        dataset: ns.dataloader.SegmentDataset,
        generator: torch.Generator | None = None,
    ) -> torch.utils.data.Sampler:
        return make_weighted_sampler(dataset, logger=LOGGER, generator=generator)


class RegressionBinSampler(BaseSampler):
    """Inverse-frequency weighted sampler stratified by binned regression targets.

    The regression counterpart of :class:`ClassificationSampler`.  Targets
    are bucketised by ``bin_edges`` and assigned inverse-frequency sampling
    weights so that, in expectation, every populated bin contributes the
    same total mass to each training epoch.

    Bin semantics match :class:`~neuralbench.metrics.BinnedMAE`:

    * Bin ``i`` covers ``[bin_edges[i], bin_edges[i + 1])`` for
      ``i < n_bins - 1``; the top bin is closed on the right so targets
      exactly at ``bin_edges[-1]`` are counted in the top bin.
    * Targets outside ``[bin_edges[0], bin_edges[-1]]`` receive **zero
      weight** -- they are excluded from sampling and do not contribute to
      any bin's count.

    Parameters
    ----------
    bin_edges
        Strictly increasing sequence of length ``>= 2`` defining the bins,
        e.g. ``[0.0, 40.0, 90.0, 300.0, 600.0]`` for the sleep-onset task.
    """

    bin_edges: list[float]

    @field_validator("bin_edges")
    @classmethod
    def _validate_bin_edges(cls, value: list[float]) -> list[float]:
        if len(value) < 2:
            raise ValueError(
                f"bin_edges must have length >= 2 to define at least one bin, "
                f"got {value}."
            )
        if any(value[i + 1] <= value[i] for i in range(len(value) - 1)):
            raise ValueError(f"bin_edges must be strictly increasing, got {value}.")
        return value

    def build(
        self,
        dataset: ns.dataloader.SegmentDataset,
        generator: torch.Generator | None = None,
    ) -> torch.utils.data.Sampler:
        return make_regression_bin_sampler(
            dataset, bin_edges=self.bin_edges, logger=LOGGER, generator=generator
        )


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
    sampler: BaseSampler | None = None
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
            if split == "train" and self.sampler is not None:
                sampler = self.sampler.build(split_dataset, generator=sampler_gen)

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


def get_default_dataloaders(
    device: str, task: str, **overrides: tp.Any
) -> dict[str, DataLoader]:
    """Return the train/val/test DataLoaders a ``device``/``task`` benchmark run uses."""

    _ensure_initialized()

    config = ConfDict(load_yaml_config(DEFAULTS_DIR / "config.yaml"))
    config.update(load_yaml_config(_resolve_task_dir(device, task) / "config.yaml"))
    data_config = ConfDict(config)["data"]
    data_config.update(overrides)
    return Data(**data_config).prepare()
