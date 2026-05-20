# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import platform
import resource
import time
import typing as tp
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from exca import TaskInfra
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger, Logger
from pydantic import model_validator
from torch.utils.data import DataLoader
from tqdm import tqdm

import neuralset as ns
from neuraltrain.losses import BaseLoss
from neuraltrain.metrics import BaseMetric
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.optimizers import LightningOptimizer
from neuraltrain.utils import (
    BaseExperiment,
    CsvLoggerConfig,
    StandardScaler,
    WandbLoggerConfig,
)

from .aggregator import (  # noqa: F401
    BenchmarkAggregator as BenchmarkAggregator,
)
from .callbacks import (
    PlotConfusionMatrix,
    PlotRegressionScatter,
    PlotRegressionVectors,
    RecordingLevelEval,
    TestFullRetrievalMetrics,
)
from .data import Data as Data  # noqa: F401
from .model_factory import build_brain_model
from .modules import DownstreamWrapper
from .pl_module import BrainModule
from .utils import TrainerConfig, compute_class_weights_from_dataset

LOGGER = logging.getLogger(__name__)


class Experiment(BaseExperiment):
    """Brain-modeling experiment with support for loading pretrained weights."""

    task_name: str = ""

    # Data
    data: Data
    target_scaler: StandardScaler | None = None
    compute_class_weights: bool = False

    # Model
    brain_model_config: BaseModelConfig
    brain_model_output_size: int | None = (
        None  # For convenience, e.g. when making yaml configs
    )
    pretrained_weights_fname: str | None = None
    downstream_model_wrapper: DownstreamWrapper | None = None

    # Optim
    trainer_config: TrainerConfig
    loss: BaseLoss
    lightning_optimizer_config: LightningOptimizer

    # Evaluation
    eval_only: bool = False
    metrics: list[BaseMetric]
    validate_before_training: bool = True
    test_full_metrics: list[BaseMetric] = []
    test_full_retrieval_metrics: list[BaseMetric] = []

    # Weights & Biases
    csv_config: CsvLoggerConfig | None = None
    wandb_config: WandbLoggerConfig | None = None

    # Internal properties
    _brain_module: pl.LightningModule | None = None
    _trainer: pl.Trainer | None = None
    _csv_logger: CSVLogger | None = None
    _wandb_logger: WandbLogger | None = None
    _n_total_params: int | None = None
    _n_trainable_params: int | None = None

    # Others
    seed: int = 0
    delete_checkpoints_on_exit: bool = True
    infra: TaskInfra = TaskInfra(version="1")
    dummy: dict[str, tp.Any] = {}  # Useful to avoid overwriting experiments between grids
    brain_model_name: str = ""

    @model_validator(mode="after")
    def _populate_brain_model_name(self) -> "Experiment":
        """Auto-populate brain_model_name from brain_model_config class when unset."""
        if not self.brain_model_name:
            self.brain_model_name = type(self.brain_model_config).__name__
        return self

    @model_validator(mode="after")
    def _validate_metrics_num_classes(self) -> "Experiment":
        """Guard against stale num_classes in metrics after cross-file YAML merging.

        YAML anchors are resolved at parse time within a single file.  When a
        dataset override changes ``brain_model_output_size`` without redefining
        ``metrics``, the inherited metric dicts still contain the old
        ``num_classes`` value.  This validator catches such mismatches early.
        """
        if self.brain_model_output_size is None:
            return self
        for metric in [*self.metrics, *self.test_full_metrics]:
            kwargs = getattr(metric, "kwargs", None)
            if kwargs is None:
                continue
            for key in ("num_classes", "num_labels"):
                value = kwargs.get(key)
                if value is not None and value != self.brain_model_output_size:
                    raise ValueError(
                        f"Metric '{metric.log_name}' has {key}={value} but "
                        f"brain_model_output_size={self.brain_model_output_size}. "
                        f"Dataset override likely changed brain_model_output_size "
                        f"without redefining metrics."
                    )
        return self

    def prepare_pl_module(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> None:
        brain_model, self._n_total_params, self._n_trainable_params = build_brain_model(
            brain_model_config=self.brain_model_config,
            downstream_model_wrapper=self.downstream_model_wrapper,
            pretrained_weights_fname=self.pretrained_weights_fname,
            train_loader=train_loader,
            val_loader=val_loader,
            wandb_logger=self._wandb_logger,
        )

        if self.target_scaler is not None:
            assert hasattr(train_loader.dataset, "extractors")
            neuro_extractor = train_loader.dataset.extractors.pop("neuro")
            for batch in tqdm(train_loader, "Fitting target scaler"):
                self.target_scaler.partial_fit(batch.data["target"])
                if self.target_scaler._n_samples_seen > 5e5:
                    break
            train_loader.dataset.extractors["neuro"] = neuro_extractor

        loss_kwargs: dict[str, tp.Any] = {}
        if self.compute_class_weights:
            loss_kwargs, _ = compute_class_weights_from_dataset(
                train_loader.dataset,  # type: ignore[arg-type]
                task=(
                    "multiclass"
                    if self.loss.__class__.__name__ == "CrossEntropyLoss"
                    else "multilabel"
                ),
                logger=LOGGER,
            )

        self._brain_module = BrainModule(
            model=brain_model,
            target_scaler=self.target_scaler,
            loss=self.loss.build(**loss_kwargs),
            lightning_optimizer_config=self.lightning_optimizer_config,
            metrics={metric.log_name: metric.build() for metric in self.metrics},
            test_full_metrics={
                metric.log_name: metric.build() for metric in self.test_full_metrics
            },
            test_full_retrieval_metrics={
                metric.log_name: metric.build()
                for metric in self.test_full_retrieval_metrics
            },
        )
        pl.seed_everything(self.seed)

    def fit(
        self,
        trainer: pl.Trainer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
    ) -> None:
        msg = "Prepare the BrainModule first with self.prepare_pl_module()"
        assert self._brain_module is not None, msg

        if self.validate_before_training:
            LOGGER.info("Validating once before starting training...")
            trainer.validate(model=self._brain_module, dataloaders=[valid_loader])

        # Train model
        trainer.fit(
            model=self._brain_module,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )

    def setup_wandb_logger(
        self,
        wandb_config: WandbLoggerConfig,
        savedir: str,
    ) -> WandbLogger:
        """Setup wandb logger and launch initialization."""
        import wandb

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

    def setup_run(self):
        """Setup paths and wandb logger."""
        savedir = self.infra.uid_folder()
        LOGGER.info(f"UID folder: {savedir}")

        # Save full config as yaml
        if not savedir.exists():
            savedir.mkdir(parents=True, exist_ok=False)
            with open(savedir / "config.yaml", "w") as outfile:
                yaml.dump(self.model_dump(), outfile, indent=4, default_flow_style=False)

        if self.wandb_config is not None:
            self._wandb_logger = self.setup_wandb_logger(self.wandb_config, str(savedir))
        if self.csv_config is not None:
            self._csv_logger = self.csv_config.build(save_dir=savedir)

    def setup_trainer(self, is_test: bool = False) -> pl.Trainer:
        """Create callbacks and setup Trainer."""
        callbacks: list[Callback] = []
        if "confusion_matrix" in [metric.log_name for metric in self.metrics]:
            labels: list[str] | None = None
            if isinstance(self.data.target, ns.extractors.LabelEncoder):
                ind_to_label: dict[int, str] = {}
                for lbl, idx in self.data.target._label_to_ind.items():
                    ind_to_label.setdefault(idx, lbl)
                labels = [ind_to_label[i] for i in sorted(ind_to_label)]
            callbacks.append(PlotConfusionMatrix(labels=labels))
        if is_test:
            if self.test_full_metrics:
                callbacks.append(RecordingLevelEval())
            if self.test_full_retrieval_metrics:
                callbacks.append(
                    TestFullRetrievalMetrics(
                        event_type=self.data.target.event_types,  # type: ignore[arg-type]
                        retrieval_set_sizes=(None, 250),
                        logger=LOGGER,
                        eval_val=False,
                    )
                )
            # Add regression vector visualization for multi-dimensional outputs
            if (
                self.brain_model_output_size is not None
                and self.brain_model_output_size > 1
            ):
                loss_name = (
                    self.loss.name
                    if hasattr(self.loss, "name")
                    else self.loss.__class__.__name__
                )
                if loss_name in ["MSELoss", "ClipLoss"]:
                    callbacks.append(PlotRegressionVectors(num_samples=10))
            # Add scatter plot for 1D regression outputs
            if (
                self.brain_model_output_size is not None
                and self.brain_model_output_size == 1
            ):
                loss_name = (
                    self.loss.name
                    if hasattr(self.loss, "name")
                    else self.loss.__class__.__name__
                )
                if loss_name == "MSELoss":
                    callbacks.append(PlotRegressionScatter())
        else:
            callbacks.append(LearningRateMonitor(logging_interval="step"))
            callbacks.append(
                EarlyStopping(
                    monitor=self.trainer_config.monitor,
                    patience=self.trainer_config.patience,
                    mode=self.trainer_config.mode,
                    verbose=True,
                )
            )
            callbacks.append(
                ModelCheckpoint(
                    dirpath=self.infra.uid_folder(),
                    filename="best",
                    monitor=self.trainer_config.monitor,
                    save_last=False,
                    mode=self.trainer_config.mode,
                    save_weights_only=True,
                    save_on_train_epoch_end=None,
                    save_top_k=1,
                    enable_version_counter=False,
                )
            )

        loggers: list[Logger] = []
        if self._wandb_logger is not None:
            loggers.append(self._wandb_logger)
        if self._csv_logger is not None:
            loggers.append(self._csv_logger)
        if not loggers:
            loggers.append(DummyLogger())

        return self.trainer_config.build(
            logger=loggers,
            callbacks=callbacks,
            accelerator="cpu" if self.infra.gpus_per_node == 0 else "auto",
            devices=1 if is_test else self.infra.gpus_per_node,
            num_nodes=1,
        )

    def _test(
        self,
        loaders: dict[str, DataLoader],
        best_model_path: str | None,
    ) -> dict[str, float | None]:
        """Run the test phase on rank 0 with a dedicated trainer."""
        tester = self.setup_trainer(is_test=True)
        return dict(
            tester.test(
                self._brain_module,
                dataloaders=loaders["test"],
                ckpt_path=best_model_path,
            )[0]
        )

    def _cleanup(self, trainer: pl.Trainer) -> None:
        """Delete checkpoint and finalize W&B."""
        if (
            self.delete_checkpoints_on_exit
            and not self.eval_only
            and hasattr(trainer.checkpoint_callback, "best_model_path")
        ):
            best_model_path = getattr(
                trainer.checkpoint_callback, "best_model_path", None
            )
            assert best_model_path is not None
            Path(best_model_path).unlink(missing_ok=True)
            LOGGER.info("Deleted checkpoint: %s", best_model_path)

        if self._wandb_logger is not None:
            import wandb

            wandb.finish()

    @infra.apply(
        exclude_from_cache_uid=(
            "wandb_config",
            "csv_config",
            "brain_model_name",
        )
    )
    def run(self) -> dict[str, tp.Any]:
        """Execute the full experiment lifecycle: setup, train, test, cleanup.

        Returns a dict of test metrics (e.g. ``{"test/bal_acc": 0.85, ...}``)
        plus ``n_total_params`` and ``n_trainable_params``.
        """
        logging.basicConfig(level=logging.INFO)
        logging.getLogger("numexpr").setLevel(logging.WARNING)
        logging.getLogger("fontTools").setLevel(logging.WARNING)
        logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
        logging.getLogger("fontTools.ttLib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
        logging.getLogger("exca").propagate = False
        logging.getLogger("neuralset").propagate = False
        self.setup_run()
        loaders = self.data.prepare()
        trainer = self.setup_trainer()
        self.prepare_pl_module(loaders["train"], loaders.get("val"))

        test_results: dict[str, tp.Any] = {}

        training_time_s: float | None = None
        peak_gpu_memory_mb: float | None = None
        peak_cpu_memory_mb: float | None = None

        if not self.eval_only:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            self.fit(trainer, loaders["train"], loaders["val"])
            training_time_s = time.perf_counter() - t0

            if torch.cuda.is_available():
                peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

            rusage = resource.getrusage(resource.RUSAGE_SELF)
            # macOS reports bytes, Linux reports KB
            divisor = 1024 if platform.system() != "Darwin" else 1024**2
            peak_cpu_memory_mb = rusage.ru_maxrss / divisor

            if isinstance(trainer.checkpoint_callback, ModelCheckpoint):
                best_model_path = trainer.checkpoint_callback.best_model_path
                assert best_model_path is not None
                best_ckpt = torch.load(
                    best_model_path,
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
                best_epoch = best_ckpt["epoch"]
                LOGGER.info("Best epoch: %i", best_epoch)
                for logger in [self._wandb_logger, self._csv_logger]:
                    if logger is not None:
                        logger.log_metrics({"best_epoch": best_epoch})
        else:
            best_model_path = None

        if (
            getattr(self.infra, "gpus_per_node", 0) > 1
            and torch.distributed.is_initialized()
        ):
            torch.distributed.destroy_process_group()

        if trainer.global_rank == 0:
            test_results.update(self._test(loaders, best_model_path))

        self._cleanup(trainer)

        test_results.update(
            {
                "n_total_params": self._n_total_params,
                "n_trainable_params": self._n_trainable_params,
                "training_time_s": training_time_s,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "peak_cpu_memory_mb": peak_cpu_memory_mb,
            }
        )
        return test_results


# BenchmarkAggregator lives in aggregator.py and forward-references Experiment
# (under TYPE_CHECKING to avoid a circular import).  Pydantic needs the real
# class at validation time, so we rebuild the schema here once both are defined.
BenchmarkAggregator.model_rebuild(_types_namespace={"Experiment": Experiment})
