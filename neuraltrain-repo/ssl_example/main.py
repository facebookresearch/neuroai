# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Defines the main classes used in the pretraining experiment.

Mirrors ``project_example`` with the three changes self-supervision needs:
- `Data`: pools several studies, segments on a stride, and extracts no target
- `Data`: extracts channel positions, which is what lets one encoder read
  montages that share no channels
- `Experiment`: drives `MaeModule`, and writes an encoder-only checkpoint
"""

import typing as tp
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import pydantic
import torch
from exca import TaskInfra
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers.logger import DummyLogger, Logger
from lightning.pytorch.utilities import rank_zero_info
from torch.utils.data import DataLoader

import neuralset as ns
from neuraltrain import BaseLoss, LightningOptimizer
from neuraltrain.mae_module import MaeModule
from neuraltrain.models.mae import MaeEncoder
from neuraltrain.utils import CsvLoggerConfig, WandbLoggerConfig


class Data(pydantic.BaseModel):
    """Builds DataLoaders of unlabelled windows from studies and extractors."""

    model_config = pydantic.ConfigDict(extra="forbid")

    # A list of steps would be read as one *chain* (each step feeding the next),
    # so pooling studies takes a list of them, concatenated below.
    studies: list[ns.Step]
    segmenter: ns.dataloader.Segmenter
    # Kept beside the segmenter rather than among its extractors so it can be
    # wired to the "input" extractor instead of repeating its preprocessing.
    channel_positions: ns.extractors.ChannelPositions
    val_ratio: float = 0.2
    batch_size: int = 64
    num_workers: int = 0

    def build(self) -> dict[str, DataLoader]:
        # `standardize_events` renumbers the concatenated index; studies already
        # prefix `timeline` and `subject` with their own name, so pooling cannot
        # merge two recordings into one.
        events = ns.events.standardize_events(
            pd.concat([study.run() for study in self.studies], ignore_index=True)
        )

        neuro = self.segmenter.extractors["input"]
        # Positions are read off the same Raw objects as the signal, and must
        # come from the very same extractor so that both index channels alike.
        assert isinstance(neuro, ns.extractors.MneRaw)
        self.segmenter.extractors["channel_positions"] = self.channel_positions.build(
            neuro
        )
        dataset = self.segmenter.apply(events)
        # Prepares over the pooled events, so the channel axis is the union of
        # every montage and a given channel keeps one index throughout.
        # Recordings lacking a channel are zero-padded there and get invalid
        # positions, which the model's merger masks out.
        dataset.prepare()
        rank_zero_info(
            f"Segmented {len(dataset)} unlabelled windows over "
            f"{len(neuro._channels)} channels"
        )

        # Striding produces many windows per recording, so the split has to be
        # over windows rather than over the trigger events an event-level
        # transform like `SklearnSplit` sees.  Holding out the tail of each
        # timeline keeps neighbouring (hence correlated) windows on one side.
        segments = pd.DataFrame(
            [{"timeline": s.timeline, "start": s.start} for s in dataset.segments]
        )
        cutoff = segments.groupby("timeline")["start"].transform(
            lambda s: s.quantile(1 - self.val_ratio)
        )
        is_val = (segments["start"] >= cutoff).to_numpy()

        loaders = {}
        for split, mask, shuffle in [("train", ~is_val, True), ("val", is_val, False)]:
            ds = dataset.select(mask)
            loaders[split] = DataLoader(
                ds,
                collate_fn=ds.collate_fn,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
            )
        return loaders


class Experiment(pydantic.BaseModel):
    """Pretrains an MAE encoder and saves it for downstream evaluation."""

    data: Data
    # Reproducibility
    seed: int = 33
    # Model
    brain_model_config: MaeEncoder
    mask_ratio: float = 0.5
    # Loss
    loss: BaseLoss
    # Optimization
    optim: LightningOptimizer
    # Hardware
    strategy: str | None = "auto"
    accelerator: str = "gpu"
    # Training
    n_epochs: int = 50
    patience: int = 10
    limit_train_batches: int | None = None
    fast_dev_run: bool = False
    # Logging
    csv_config: CsvLoggerConfig | None = None
    wandb_config: WandbLoggerConfig | None = None

    # Others
    infra: TaskInfra = TaskInfra(version="1")

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["strategy", "accelerator"]

    def model_post_init(self, __context: tp.Any) -> None:
        if self.infra.folder is None:
            msg = "infra.folder needs to be specified to save the results."
            raise ValueError(msg)

    @property
    def checkpoint_path(self) -> Path:
        """Encoder weights, in the form ``neuralbench --checkpoint`` expects."""
        folder = self.infra.uid_folder()
        assert folder is not None  # guaranteed by model_post_init
        return folder / "encoder.ckpt"

    def _setup_trainer(self) -> pl.Trainer:
        loggers: list[Logger] = []
        if self.csv_config is not None:
            loggers.append(self.csv_config.build(save_dir=self.infra.folder))
        if self.wandb_config is not None:
            # `build` logs in and starts the run; under DDP it tolerates being
            # called from a non-zero rank, where it yields a no-op logger.
            loggers.append(
                self.wandb_config.build(
                    save_dir=str(self.infra.folder), xp_config=self.model_dump()
                )
            )
        if not loggers:
            loggers.append(DummyLogger())

        return pl.Trainer(
            strategy=self.strategy,
            devices=self.infra.gpus_per_node,
            accelerator=self.accelerator,
            max_epochs=self.n_epochs,
            limit_train_batches=self.limit_train_batches,
            fast_dev_run=self.fast_dev_run,
            callbacks=[
                EarlyStopping(monitor="val_loss", mode="min", patience=self.patience),
                LearningRateMonitor(logging_interval="epoch"),
                # Early stopping runs `patience` epochs past the best one, so
                # the weights left in memory are not the ones worth keeping.
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    # Per-run folder, not `infra.folder`: a grid runs many of
                    # these at once and they would otherwise share one file.
                    dirpath=self.checkpoint_path.parent,
                    filename="best",
                ),
            ],
            logger=loggers,
        )

    def _build_mae_module(self) -> MaeModule:
        return MaeModule(
            # n_outputs=None: pretraining and downstream probing both want the
            # encoder alone, with no classification head.  No channel count
            # either -- the encoder's merger makes it montage-independent.
            model=self.brain_model_config.build(n_outputs=None),
            loss=self.loss.build(),
            optim_config=self.optim,
            mask_ratio=self.mask_ratio,
        )

    @staticmethod
    def _best_encoder(trainer: pl.Trainer, module: MaeModule) -> dict[str, torch.Tensor]:
        """Encoder weights of the best-validation epoch.

        ``MaeModule`` holds the encoder as ``self.model``, so its checkpoint
        keys carry a ``"model."`` prefix that a bare encoder does not.
        """
        callback = trainer.checkpoint_callback
        assert isinstance(callback, ModelCheckpoint)  # set in `_setup_trainer`
        if not callback.best_model_path:
            # `fast_dev_run` suppresses checkpointing, and there is no best
            # epoch to pick from anyway.
            return module.model.state_dict()
        state = torch.load(callback.best_model_path, weights_only=True)["state_dict"]
        prefix = "model."
        return {
            k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)
        }

    @infra.apply
    def run(self) -> dict[str, float | None]:
        pl.seed_everything(self.seed, workers=True)
        loaders = self.data.build()

        mae_module = self._build_mae_module()
        trainer = self._setup_trainer()
        trainer.fit(
            model=mae_module,
            train_dataloaders=loaders["train"],
            val_dataloaders=loaders["val"],
        )

        # Save the encoder alone: the mask token and reconstruction layer are
        # pretraining scaffolding, and `neuralbench --checkpoint` matches
        # against a bare encoder state dict.  Ranks hold identical weights
        # after DDP has synchronised them, so only one of them writes.
        if trainer.is_global_zero:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            encoder = self._best_encoder(trainer, mae_module)
            torch.save({"state_dict": encoder}, self.checkpoint_path)
            rank_zero_info(f"\nSaved pretrained encoder to {self.checkpoint_path}\n")
        # Every rank returns the same metrics, since `MaeModule` logs them with
        # `sync_dist=True`, but none may return before the checkpoint exists.
        trainer.strategy.barrier()

        return {k: float(v) for k, v in trainer.logged_metrics.items()}
