# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import typing as tp
import warnings
from collections import defaultdict
from pathlib import Path

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchmetrics
from lightning.pytorch.callbacks import Callback
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from neuraltrain.metrics.utils import agg_per_group, agg_retrieval_preds

if tp.TYPE_CHECKING:
    from neuraltrain.utils import StandardScaler

LOGGER = logging.getLogger(__name__)


def _set_plot_theme() -> None:
    """Apply the plotting theme used by neuralbench callbacks."""
    sns.set_theme(context="paper", style="white")


class TestFullRetrievalMetrics(Callback):
    """Accumulate predictions on entire test set before evaluating metrics.

    Requires the pl.LightningModule object this callback is attached to to define a
    torch.ModuleDict named `retrieval_metrics` containing the metrics to evaluate.
    """

    def __init__(
        self,
        event_type: tp.Literal["Word", "Image"] = "Word",
        event_field: tp.Literal["text", "category"] = "text",
        retrieval_set_sizes: tuple = (None, 250),
        save_outputs: bool = False,
        logger: tp.Any = None,
        eval_val: bool = False,
    ):
        self.event_type = event_type
        self.event_field = event_field
        self.retrieval_set_sizes = retrieval_set_sizes
        self.save_outputs = save_outputs
        self.logger = logger
        self.eval_val = eval_val
        self.full_outputs: dict[str | int, tp.Any] = {}

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        # Ensure retrieval metrics are valid
        if not hasattr(pl_module, "test_full_retrieval_metrics") or not isinstance(
            pl_module.test_full_retrieval_metrics, nn.ModuleDict
        ):
            raise ValueError(
                "The LightningModule needs a test_full_retrieval_metrics ModuleDict "
                "that contains the retrieval metrics to evaluate on the full test set."
            )
        test_full_metrics = {
            k: v
            for k, v in pl_module.test_full_retrieval_metrics.items()
            if ((k.startswith("val") or k.startswith("test")) and "retrieval" in k)
        }
        if not test_full_metrics:
            raise ValueError(
                "The LightningModule needs at least one retrieval metric with 'retrieval' in its name."
            )

    def get_n_loaders(
        self, trainer: pl.Trainer, step_name: tp.Literal["val", "test"]
    ) -> int:
        loader = getattr(trainer, step_name + "_dataloaders")
        if loader is None:
            n_loaders = 0
        elif isinstance(loader, DataLoader):
            n_loaders = 1
        else:
            n_loaders = len(loader)
        return n_loaders

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        assert trainer.val_dataloaders is not None
        self.full_outputs = {
            idx: defaultdict(list) for idx in range(len(trainer.val_dataloaders))
        }
        self.full_outputs["y_true"] = dict()

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        assert trainer.test_dataloaders is not None
        self.full_outputs = {
            idx: defaultdict(list) for idx in range(len(trainer.test_dataloaders))
        }
        self.full_outputs["y_true"] = dict()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if self.eval_val:
            self._collate_outputs(outputs, batch, dataloader_idx)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        self._collate_outputs(outputs, batch, dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.eval_val:
            self._on_epoch_end(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        self._on_epoch_end(trainer, pl_module, "test")

    def _on_epoch_end(self, trainer, pl_module, step_name):
        if self.logger is not None:
            self.logger.info(
                f"{step_name.capitalize()} segments: {len(self.full_outputs[0]['subject'])}"
            )
        self._compute_metrics(trainer, pl_module, step_name=step_name)
        if self.save_outputs:
            self._save_outputs(trainer, step_name=step_name)
        del self.full_outputs

    def _save_outputs(self, trainer, step_name):
        n_loaders = self.get_n_loaders(trainer, step_name)
        for dataloader_idx in range(n_loaders):
            full = self.full_outputs[dataloader_idx]
            for key in ["y_pred", "y_true"]:
                full[key] = torch.cat(full[key], dim=0)

            save_dir = Path(trainer.logger.save_dir) / "retrieval_outputs"
            save_dir.mkdir(parents=True, exist_ok=True)
            file = save_dir / f"{step_name}_{dataloader_idx}.pt"
            torch.save(full, file)

    def _collate_outputs(self, outputs, batch, dataloader_idx):
        y_pred, y_true = outputs
        full = self.full_outputs[dataloader_idx]

        full["y_pred"].append(y_pred.cpu())
        full["y_true"].append(y_true.cpu())

        # for each prediction, we retrieve the corresponding extractor item (category for image
        # datasets, text for word datasets)
        for segment in batch.segments:
            trigger = segment.trigger
            if hasattr(trigger, self.event_field):
                full[self.event_field].append(getattr(trigger, self.event_field))
            elif hasattr(trigger, "category"):
                full["category"].append(trigger.category)
            else:
                pass

            extra = trigger.extra
            full["subject"].append(extra["subject"])
            if "sequence_id" in extra:
                full["sequence"].append(extra["sequence_id"])
            elif hasattr(trigger, "filepath"):
                full["filepath"].append(trigger.filepath)
            else:
                pass

    def _compute_metrics(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        step_name: tp.Literal["val", "test"],
    ) -> None:
        n_loaders = self.get_n_loaders(trainer, step_name)
        test_full_metrics = {
            k: v
            for k, v in pl_module.test_full_retrieval_metrics.items()  # type: ignore
            if (k.startswith(step_name) and "retrieval" in k)
        }
        for dataloader_idx in range(n_loaders):
            full = self.full_outputs[dataloader_idx]
            y_pred = torch.cat(full["y_pred"], dim=0)
            y_true = torch.cat(full["y_true"], dim=0)

            subjects_pred = full["subject"]
            if "text" in full.keys():
                groups_pred = full["text"]
            else:
                groups_pred = full["filepath"]

            for retrieval_set_size in self.retrieval_set_sizes:
                out = self._get_test_full_metrics(
                    y_pred=y_pred,
                    y_true=y_true,
                    groups_pred=groups_pred,
                    subjects_pred=subjects_pred,
                    metrics=test_full_metrics,
                    retrieval_set_size=retrieval_set_size,
                    step_name=step_name,
                )
                for key, value in out.items():
                    if n_loaders > 1:
                        key += f"_{dataloader_idx}"
                    pl_module.log(key, value)
                    if self.logger is not None:
                        self.logger.info(f"{key}: {value}")

    @staticmethod
    def _get_test_full_metrics(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        groups_pred: list,
        subjects_pred: list,
        metrics: dict[str, torchmetrics.Metric],
        step_name: tp.Literal["val", "test"],
        retrieval_set_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute retrieval metrics with optional aggregation of predictions.

        This method processes the predictions and true labels to compute the specified metrics,
        potentially aggregating results by groups or subjects.

        Parameters
        ----------
        y_pred:
            predicted labels or scores from the model
        y_true:
            true labels for the data.
        groups_pred:
            group labels for each prediction.
        subjects_pred:
            subject id for each prediction.
        metrics:
            a dictionary of metric functions to be computed.
        step_name:
            the name of the current step (e.g. 'val', 'test').
        retrieval_set_size:
            the number of most frequent words to consider for metrics. Defaults to None.

        Returns
        -------
        dict:
            A dictionary containing computed metric values.

        Note
        ----
        The function can handle different aggregation strategies based on the end of the metric name:
          - "subject-agg": Aggregates predictions by subject first for a given group.
          - "instance-agg": Aggregates all predictions for a given group.
          - "subject-ind": Saves the metrics for each subject individually.
        """
        out = {}

        # Keep only the most frequent groups
        if retrieval_set_size is not None:
            groups_df = pd.DataFrame({"label": groups_pred})
            counts = groups_df.label.value_counts(sort=True, ascending=False)
            most_frequent = set(counts.index[:retrieval_set_size])
            if len(counts) < retrieval_set_size:
                msg = f"The number of unique labels ({len(counts)}) is lower than the retrieval size ({retrieval_set_size})."
                warnings.warn(msg, UserWarning)
            msg = f"Retrieval set items (and count) in test set:\n{counts.iloc[:retrieval_set_size].to_dict()}"
            LOGGER.warning(msg)
            mask = groups_df.label.isin(most_frequent).to_numpy()
            y_pred, y_true = y_pred[mask], y_true[mask]
            indices = np.where(mask)[0]
            groups_pred = [groups_pred[i] for i in indices]
            subjects_pred = [subjects_pred[i] for i in indices]

        # Remove repetitions in retrieval set
        agg_y_true, agg_groups_true = agg_per_group(
            y_true,
            groups=groups_pred,
            agg_func="mean",  # Use "mean" instead of "first" because events can have different
            # latent representation in speech decoding experiments
        )

        for metric_name, metric in tqdm(
            metrics.items(), "Evaluating full test set retrieval metrics"
        ):
            metric = metric.to("cpu")

            subjects: list | None
            if metric_name.endswith("subject-agg"):
                subjects = subjects_pred
            elif metric_name.endswith("instance-agg"):
                subjects = None
            else:
                subjects = torch.arange(y_pred.shape[0], device=y_pred.device).tolist()

            agg_y_pred, agg_groups_pred = agg_retrieval_preds(
                y_pred,
                groups_pred=groups_pred,
                subjects_pred=subjects,
            )
            if retrieval_set_size is not None:
                metric_name += f"_size={retrieval_set_size}"

            if "subject-ind" in metric_name and step_name == "test":
                # Compute performance per subject
                individual_subjects = pd.DataFrame({"id": subjects_pred})
                results = {}
                for subj, grp in individual_subjects.groupby("id"):
                    metric.reset()
                    metric.update(
                        agg_y_pred[grp.index.values],
                        agg_y_true,
                        [
                            agg_grp_pred
                            for i, agg_grp_pred in enumerate(agg_groups_pred)
                            if i in grp.index
                        ],
                        agg_groups_true,
                    )
                    results[subj] = metric.compute().item()
                out[metric_name] = torch.tensor(np.mean(list(results.values())))
            else:
                metric.reset()
                metric.update(agg_y_pred, agg_y_true, agg_groups_pred, agg_groups_true)
                out[metric_name] = metric.compute()

            if "subject-ind" in metric_name and "agg" not in metric_name:
                # Compute frequency-corrected average
                if hasattr(metric, "_compute_macro_average"):
                    ranks = metric._compute_ranks(  # type: ignore
                        agg_y_pred, agg_y_true, agg_groups_pred, agg_groups_true
                    )
                    macro_average = np.mean(
                        list(
                            metric._compute_macro_average(ranks, agg_groups_pred).values()  # type: ignore
                        )
                    )
                    out[metric_name + "_macro"] = torch.tensor(macro_average)
                else:
                    warnings.warn(
                        f"Metric {metric} does not implement `_compute_macro_average`."
                    )

        return out


class RecordingLevelEval(Callback):
    """Callback to evaluate average prediction over each recording (timeline)."""

    def __init__(self) -> None:
        self.test_outputs: dict[str, dict[str, tp.Any]] = {}
        self.num_classes: int | None = None

    def setup(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        if not hasattr(pl_module, "test_full_metrics") or not isinstance(
            pl_module.test_full_metrics, nn.ModuleDict
        ):
            raise ValueError(
                "The LightningModule needs a test_full_metrics ModuleDict that contains the "
                "metrics to evaluate on the full test set."
            )

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self.test_outputs = {}
        self.num_classes = None

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        y_pred_logits = outputs[0].cpu()
        y_pred = y_pred_logits.argmax(dim=1)
        y_true = outputs[1].cpu()
        timelines = [segment.events.timeline.iloc[0] for segment in batch.segments]

        # Infer number of classes from model output shape
        if self.num_classes is None:
            self.num_classes = int(y_pred_logits.shape[1])

        for timeline, y_true_, y_pred_ in zip(timelines, y_true, y_pred):
            predicted_class = y_pred_.item()

            # Initialize entry for new timeline
            if timeline not in self.test_outputs:
                self.test_outputs[timeline] = {
                    "y_true": y_true_.item(),
                    "class_counts": [0] * self.num_classes,
                }

            # Update class count for the predicted class
            self.test_outputs[timeline]["class_counts"][predicted_class] += 1

    def on_test_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        outputs_df = pd.DataFrame(self.test_outputs).T

        # Compute total count per recording
        outputs_df["total_count"] = outputs_df["class_counts"].apply(sum)

        # Compute probability for each class
        assert isinstance(self.num_classes, int)
        for i in range(self.num_classes):
            outputs_df[f"y_pred{i}"] = outputs_df["class_counts"].apply(
                lambda counts: counts[i] / sum(counts) if sum(counts) > 0 else 0.0
            )

        # Create prediction tensor with shape (n_recordings, num_classes)
        pred_columns = [f"y_pred{i}" for i in range(self.num_classes)]
        y_pred_probs = torch.from_numpy(outputs_df[pred_columns].values)
        # Convert y_true to int64 explicitly to avoid object dtype issues
        y_true = torch.tensor(outputs_df.y_true.tolist(), dtype=torch.int64)

        for metric_name, metric in pl_module.test_full_metrics.items():  # type: ignore
            metric.to("cpu")
            metric.update(y_pred_probs, y_true)
            pl_module.log(
                metric_name,
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=outputs_df.shape[0],
            )


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None,
    cmap: str = "viridis",
    xticks_rotation: float | str = 45,
) -> Figure:
    """
    Plot confusion matrix/matrices with a shared colorbar.

    Parameters
    ----------
    cm : array-like
        Confusion matrix. Can be:
        - 2D array of shape (n_classes, n_classes) for multiclass case
        - 3D array of shape (n_labels, 2, 2) for multilabel case
    labels : list of str, optional
        Labels for the classes/categories. If None, uses numeric indices.
    cmap : str, default='viridis'
        Colormap to use for the heatmap.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    _set_plot_theme()
    cm = np.asarray(cm)

    # Detect if multiclass (2D) or multilabel (3D with 2x2 matrices)
    is_multilabel = cm.ndim == 3 and cm.shape[1] == 2 and cm.shape[2] == 2

    if is_multilabel:
        # Multilabel case: multiple 2x2 confusion matrices
        n_labels = cm.shape[0]
        n_cols = int(np.ceil(np.sqrt(n_labels)))
        n_rows = int(np.ceil(n_labels / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), constrained_layout=True
        )
        axes = np.atleast_1d(axes).flatten()

        # Find global min/max for consistent colorbar
        vmin = cm.min()
        vmax = cm.max()

        # Plot each confusion matrix using ConfusionMatrixDisplay
        displays = []
        for i in range(n_labels):
            label = labels[i] if labels is not None else f"Label {i}"
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm[i], display_labels=["0", "1"]
            )
            disp.plot(
                ax=axes[i],
                cmap=cmap,
                colorbar=False,
                im_kw={"vmin": vmin, "vmax": vmax},
                xticks_rotation=xticks_rotation,
            )
            axes[i].set_title(label)
            displays.append(disp)

        # Hide unused subplots
        for i in range(n_labels, len(axes)):
            axes[i].set_visible(False)

        # Add a common colorbar (using list of visible axes only)
        visible_axes = [axes[i] for i in range(n_labels)]
        fig.colorbar(displays[0].im_, ax=visible_axes, label="Count")

    else:
        # Multiclass case: single confusion matrix
        fig, ax = plt.subplots(figsize=(8, 8))

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap=cmap, xticks_rotation=xticks_rotation)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()

    return fig


class PlotConfusionMatrix(Callback):
    def __init__(self, labels: list[str] | None = None):
        self.labels = labels

    def _plot_and_log(
        self,
        step_name: tp.Literal["val", "test"],
        trainer,
        pl_module,
    ):
        if isinstance(trainer.logger, pl.loggers.wandb.WandbLogger):
            import wandb

            metric_name = step_name + "/confusion_matrix"
            metric = getattr(pl_module.metrics, metric_name, None)
            if metric is not None:
                cm = metric.compute().detach().cpu().numpy()
                cm_fig = plot_confusion_matrix(cm, labels=self.labels)
                metric.reset()
                try:
                    wandb.log({metric_name: wandb.Image(cm_fig)})
                except wandb.errors.Error as e:
                    LOGGER.warning("Failed to log confusion matrix to W&B: %s", e)
                plt.close("all")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._plot_and_log("val", trainer, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._plot_and_log("test", trainer, pl_module)


class PlotRegressionVectors(Callback):
    """Visualize predictions vs ground truth for multi-dimensional regression tasks.

    This callback samples examples from the test set and creates visualizations
    showing the predicted and ground truth vectors overlaid for comparison.
    Each sample is shown in a separate subplot with dimension index on x-axis
    and magnitude on y-axis.

    Parameters
    ----------
    num_samples : int, default=10
        Number of samples to visualize from the test set.
    """

    def __init__(self, num_samples: int = 10):
        self.num_samples = num_samples
        self.samples_collected: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.sample_indices: set[int] | None = None

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Initialize sample collection at the start of testing."""
        self.samples_collected = []
        self.sample_indices = None

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        """Collect samples uniformly across the test set."""
        if len(self.samples_collected) >= self.num_samples:
            return

        y_pred, y_true = outputs

        # On first batch, calculate which batch indices to sample from
        if self.sample_indices is None and trainer.num_test_batches[dataloader_idx] > 0:
            total_batches = trainer.num_test_batches[dataloader_idx]
            # Calculate evenly spaced indices to collect exactly num_samples
            self.sample_indices = set(
                int(i * total_batches / self.num_samples) for i in range(self.num_samples)
            )

        # Collect samples at predetermined indices
        if self.sample_indices is not None and batch_idx in self.sample_indices:
            # Take the first sample from the batch
            if len(self.samples_collected) < self.num_samples:
                self.samples_collected.append(
                    (y_pred[0].detach().cpu(), y_true[0].detach().cpu())
                )

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate and log visualization at the end of testing."""
        if not trainer.is_global_zero:
            return

        if len(self.samples_collected) == 0:
            LOGGER.warning("No samples collected for regression vector visualization")
            return

        # Only log to wandb
        if not isinstance(trainer.logger, pl.loggers.wandb.WandbLogger):
            return

        import wandb

        try:
            fig = self._create_figure()
            wandb.log({"test/regression_vectors": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            LOGGER.error(f"Failed to create regression vector visualization: {e}")
            plt.close("all")

    def _create_figure(self) -> Figure:
        """Create the visualization figure with subplots for each sample."""
        n_samples = len(self.samples_collected)

        # Determine grid layout (prefer 2 columns)
        n_cols = min(2, n_samples)
        n_rows = int(np.ceil(n_samples / n_cols))

        # Create figure with appropriate size
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), constrained_layout=True
        )

        # Handle single subplot case
        if n_samples == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_samples > 1 else axes

        for idx, (y_pred, y_true) in enumerate(self.samples_collected):
            ax = axes[idx] if n_samples > 1 else axes[0]

            # Convert tensors to numpy
            y_pred_np = y_pred.numpy().flatten()
            y_true_np = y_true.numpy().flatten()

            # Create dimension indices
            dimensions = np.arange(len(y_true_np))

            # Plot ground truth and prediction
            ax.plot(
                dimensions,
                y_true_np,
                label="Ground Truth",
                color="tab:blue",
                linewidth=1.5,
            )
            ax.plot(
                dimensions,
                y_pred_np,
                label="Prediction",
                color="tab:orange",
                linewidth=1.5,
                alpha=0.8,
            )

            ax.set_xlabel("Dimension")
            ax.set_ylabel("Magnitude")
            ax.set_title(f"Sample {idx + 1}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_samples, len(axes)):
            axes[idx].set_visible(False)

        return fig


class PlotRegressionScatter(Callback):
    """Scatter plot of predicted vs ground truth for 1D regression tasks.

    Accumulates all test predictions, then produces a scatter plot with an
    identity line, a linear fit, and annotations for slope, intercept,
    Pearson r, and R^2.  If the ``pl_module`` has a fitted ``target_scaler``,
    predictions and targets are inverse-transformed back to original units
    before plotting.

    The figure is logged to W&B under ``"test/regression_scatter"``.
    """

    def __init__(self):
        self.y_preds: list[torch.Tensor] = []
        self.y_trues: list[torch.Tensor] = []

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.y_preds = []
        self.y_trues = []

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        y_pred, y_true = outputs
        self.y_preds.append(y_pred.detach().cpu())
        self.y_trues.append(y_true.detach().cpu())

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero:
            return

        if len(self.y_preds) == 0:
            LOGGER.warning("No samples collected for regression scatter visualization")
            return

        if not isinstance(trainer.logger, pl.loggers.wandb.WandbLogger):
            return

        import wandb

        y_pred = torch.cat(self.y_preds).squeeze()
        y_true = torch.cat(self.y_trues).squeeze()

        target_scaler = getattr(pl_module, "target_scaler", None)
        if target_scaler is not None and target_scaler._mean is not None:
            y_pred = self._inverse_transform(y_pred, target_scaler)
            y_true = self._inverse_transform(y_true, target_scaler)

        try:
            fig = self._create_figure(y_true.numpy().flatten(), y_pred.numpy().flatten())
            wandb.log({"test/regression_scatter": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            LOGGER.error(f"Failed to create regression scatter visualization: {e}")
            plt.close("all")

    @staticmethod
    def _inverse_transform(
        x: torch.Tensor, target_scaler: "StandardScaler"
    ) -> torch.Tensor:
        """Reverse a StandardScaler transform: x_orig = x * scale + mean."""
        assert target_scaler._mean is not None and target_scaler._scale is not None
        mean = target_scaler._mean.cpu()
        scale = target_scaler._scale.cpu()
        return x * scale + mean

    @staticmethod
    def _create_figure(y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150, constrained_layout=True)

        ax.scatter(y_true, y_pred, s=8, alpha=0.4, edgecolors="none")

        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        margin = 0.05 * (hi - lo) if hi > lo else 1.0
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            ls="--",
            color="0.5",
            lw=1,
            label="Identity",
        )

        slope, intercept = np.polyfit(y_true, y_pred, 1)
        fit_x = np.array([lo - margin, hi + margin])
        ax.plot(fit_x, slope * fit_x + intercept, color="tab:red", lw=1.5, label="Fit")

        r = np.corrcoef(y_true, y_pred)[0, 1]
        r2 = r**2
        textstr = (
            f"slope = {slope:.3f}\n"
            f"intercept = {intercept:.3f}\n"
            f"r = {r:.3f}\n"
            f"R² = {r2:.3f}"
        )
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        return fig
