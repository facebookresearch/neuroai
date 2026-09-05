# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Defines the main classes used in the pretraining experiment.

Mirrors ``project_example`` with the two changes self-supervision needs:
- `Data`: segments on a stride, and extracts no target
- `Experiment`: drives `MaeModule`, and writes an encoder-only checkpoint
"""

import typing as tp
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import pydantic
import torch
from exca import TaskInfra
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers.logger import DummyLogger, Logger
from torch.utils.data import DataLoader

import neuralset as ns
from neuraltrain import BaseLoss, LightningOptimizer
from neuraltrain.mae_module import MaeModule
from neuraltrain.models.mae import MaeEncoder
from neuraltrain.utils import CsvLoggerConfig


class Data(pydantic.BaseModel):
    """Builds DataLoaders of unlabelled windows from a study and extractors."""

    model_config = pydantic.ConfigDict(extra="forbid")

    study: ns.Step
    segmenter: ns.dataloader.Segmenter
    val_ratio: float = 0.2
    batch_size: int = 64
    num_workers: int = 0

    def build(self) -> dict[str, DataLoader]:
        events = self.study.run()
        dataset = self.segmenter.apply(events)
        dataset.prepare()
        print(f"Segmented {len(dataset)} unlabelled windows")

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
            ],
            logger=loggers,
            enable_checkpointing=False,
        )

    def _build_mae_module(self, train_loader: DataLoader) -> MaeModule:
        batch = next(iter(train_loader))
        n_chans = batch.data["input"].shape[1]
        return MaeModule(
            # n_outputs=None: pretraining and downstream probing both want the
            # encoder alone, with no classification head.
            model=self.brain_model_config.build(
                n_spatial_locations=n_chans, n_outputs=None
            ),
            loss=self.loss.build(),
            optim_config=self.optim,
            mask_ratio=self.mask_ratio,
        )

    @infra.apply
    def run(self) -> dict[str, float | None]:
        pl.seed_everything(self.seed, workers=True)
        loaders = self.data.build()

        mae_module = self._build_mae_module(loaders["train"])
        trainer = self._setup_trainer()
        trainer.fit(
            model=mae_module,
            train_dataloaders=loaders["train"],
            val_dataloaders=loaders["val"],
        )

        # Save the encoder alone: the decoder is pretraining scaffolding, and
        # `neuralbench --checkpoint` matches against a bare encoder state dict.
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": mae_module.model.state_dict()}, self.checkpoint_path)
        print(f"\nSaved pretrained encoder to {self.checkpoint_path}\n")

        return {k: float(v) for k, v in trainer.logged_metrics.items()}
