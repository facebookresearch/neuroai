# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Defines the main classes used in the experiment.

We suggest the following structure:
- `Data`: configures dataset and extractors to return DataLoaders
- `Experiment`: main class that defines the experiment to run by using `Data`
"""

import typing as tp
from pathlib import Path

import lightning.pytorch as pl
import pydantic
import wandb
from exca import TaskInfra
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger, Logger
from torch.utils.data import DataLoader

import neuralset as ns
from neuraltrain import BaseLoss, BaseMetric, BaseModelConfig, LightningOptimizer
from neuraltrain.utils import CsvLoggerConfig, WandbLoggerConfig

from .pl_module import BrainModule


class Data(pydantic.BaseModel):
    """Handles configuration and creation of DataLoaders from study and extractors."""

    model_config = pydantic.ConfigDict(extra="forbid")

    study: ns.Step
    segmenter: ns.dataloader.Segmenter
    batch_size: int = 64
    num_workers: int = 0

    def build(self) -> dict[str, DataLoader]:
        events = self.study.run()

        event_summary = (
            events.reset_index()
            .groupby(["split", "type"])[["index", "subject", "filepath", "code"]]
            .nunique()
        )
        print("Event summary: \n", event_summary)

        dataset = self.segmenter.apply(events)
        dataset.prepare()

        loaders = {}
        for split, shuffle in [("train", True), ("val", False), ("test", False)]:
            ds = dataset.select(dataset.triggers["split"] == split)
            loaders[split] = DataLoader(
                ds,
                collate_fn=ds.collate_fn,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
            )
        return loaders


class Experiment(pydantic.BaseModel):
    """Defines the main experiment pipeline including data loading and training/evaluation."""

    data: Data
    # Reproducibility
    seed: int = 33
    # Model
    brain_model_config: BaseModelConfig
    load_checkpoint: bool = True
    save_checkpoints: bool = False
    # Loss
    loss: BaseLoss
    # Optimization
    optim: LightningOptimizer
    # Metrics
    metrics: list[BaseMetric]
    # Hardware
    strategy: str | None = "auto"
    accelerator: str = "gpu"
    # Training
    n_epochs: int = 10
    patience: int = 5
    limit_train_batches: int | None = None
    fast_dev_run: bool = False
    # Eval
    checkpoint_path: str | None = None
    test_only: bool = False
    # Logging
    csv_config: CsvLoggerConfig | None = None
    wandb_config: WandbLoggerConfig | None = None

    # Others
    infra: TaskInfra = TaskInfra(version="1")

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return [
            "strategy",
            "accelerator",
            "save_checkpoints",
            "checkpoint_path",
        ]

    def model_post_init(self, __context: tp.Any) -> None:
        if self.infra.folder is None:
            msg = "infra.folder needs to be specified to save the results."
            raise ValueError(msg)

    def _get_checkpoint_path(self) -> Path | None:
        if not self.load_checkpoint:
            return None
        if self.checkpoint_path is not None:
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        else:
            checkpoint_path = Path(self.infra.folder) / "last.ckpt"
        if checkpoint_path.exists() and self.load_checkpoint:
            print(f"\nLoading model from {checkpoint_path}\n")
            return checkpoint_path
        else:
            print(f"\nNo checkpoint found at {checkpoint_path}\n")
            return None

    def setup_wandb_logger(
        self,
        wandb_config: WandbLoggerConfig,
        savedir: str,
    ) -> WandbLogger:
        """Setup wandb logger and launch initialization."""

        wandb.login(host=wandb_config.host)
        logger = wandb_config.build(
            save_dir=savedir,
            xp_config=self.model_dump(),
        )
        try:
            logger.experiment.config["_dummy"] = None  # To launch initialization
        except TypeError:
            pass  # Crashes if called in a second process, e.g. with DDP

        return logger

    def _setup_trainer(self) -> pl.Trainer:
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=self.patience),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        if self.save_checkpoints:
            callbacks.append(
                ModelCheckpoint(
                    save_last=True,
                    save_top_k=1,
                    dirpath=self.infra.folder,
                    filename="best",
                    monitor="val_loss",
                    save_on_train_epoch_end=True,
                )
            )
        loggers: list[Logger] = []
        if self.csv_config is not None:
            loggers.append(self.csv_config.build(save_dir=self.infra.folder))
        if self.wandb_config is not None:
            loggers.append(
                self.setup_wandb_logger(self.wandb_config, str(self.infra.folder))
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
            callbacks=callbacks,
            logger=loggers,
            enable_checkpointing=self.save_checkpoints,
        )

    def _build_brain_module(self, train_loader: DataLoader) -> BrainModule:
        batch = next(iter(train_loader))
        n_chans = batch.data["input"].shape[1]
        n_classes = train_loader.dataset.triggers.code.nunique()
        brain_model = self.brain_model_config.build(n_chans=n_chans, n_outputs=n_classes)
        return BrainModule(
            model=brain_model,
            loss=self.loss.build(),
            optim_config=self.optim,
            metrics={metric.log_name: metric.build() for metric in self.metrics},
            max_epochs=self.n_epochs,
        )

    @infra.apply
    def run(self) -> dict[str, float | None]:
        pl.seed_everything(self.seed, workers=True)
        loaders = self.data.build()

        brain_module = self._build_brain_module(loaders["train"])
        trainer = self._setup_trainer()

        ckpt_path = self._get_checkpoint_path()
        if not self.test_only:
            trainer.fit(
                model=brain_module,
                train_dataloaders=loaders["train"],
                val_dataloaders=loaders["val"],
                ckpt_path=ckpt_path,
            )
        results = trainer.test(
            brain_module,
            dataloaders=loaders["test"],
            ckpt_path=ckpt_path if self.test_only else None,
        )
        return dict(results[0])
