# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions."""

import logging
import typing as tp
from copy import copy
from hashlib import sha1
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch import nn

import neuralset as ns
from neuralset.dataloader import SegmentDataset

LOGGER = logging.getLogger(__name__)


def model_hash(model: nn.Module) -> str:
    hasher = sha1()
    for p in model.parameters():
        hasher.update(p.data.cpu().numpy().tobytes())
    return hasher.hexdigest()


_PACKAGE_DIR = Path(__file__).resolve().parent


def load_checkpoint(
    brain_model: nn.Module,
    checkpoint_path: str | Path,
    logger: logging.Logger,
) -> nn.Module:
    """Load checkpoint through state_dicts.

    Note
    ----
        While pytorch-lightning exposes ways to do this, we implement checkpoint loading
        directly for more fine-grained control.
    """
    checkpoint_path = Path(checkpoint_path).expanduser()
    suffix = checkpoint_path.suffix
    if checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_path.resolve()
    else:
        checkpoint_path = (_PACKAGE_DIR / checkpoint_path).resolve()
    assert suffix in (
        ".ckpt",
        ".pth",
        ".pt",
        ".safetensors",
    ), f"Expected .ckpt, .pth, .pt or .safetensors extension but got {checkpoint_path}"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} not found.")
    logger.info(f"Reloading checkpoint from {checkpoint_path}")
    logger.info(f"Initial model hash: {model_hash(brain_model)}")

    if suffix == ".safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(checkpoint_path, device="cpu")
        # For Braindecode models with explicit channel mapping (e.g. LUNA)s
        if (mapping := getattr(brain_model, "mapping", None)) is not None:
            checkpoint = {mapping.get(k, k): v for k, v in checkpoint.items()}
    else:
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")

    # Load checkpoint and update state dict
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]  # type: ignore[assignment]

    stripped_state_dict = {}
    for name, v in checkpoint.items():
        # PyTorch Lightning uses "model." in front of each layer
        if name.startswith("model."):
            name = name.replace("model.", "")
        stripped_state_dict[name] = v

    model_dict = brain_model.state_dict()

    # When the model is a wrapper (e.g. _LunaEncoderWrapper stores the inner
    # model as self.model), state dict keys are prefixed with "model." while
    # checkpoint keys are not.  Try to auto-prefix to match.
    if not (set(stripped_state_dict) & set(model_dict)):
        for prefix in ["model."]:
            prefixed = {f"{prefix}{k}": v for k, v in stripped_state_dict.items()}
            if set(prefixed) & set(model_dict):
                logger.info(
                    "Auto-prefixed checkpoint keys with %r to match model state dict.",
                    prefix,
                )
                stripped_state_dict = prefixed
                break

    missing = set(model_dict) - set(stripped_state_dict)
    additional = set(stripped_state_dict) - set(model_dict)
    logger.info(f"Missing keys in checkpoint: {sorted(missing)}")
    logger.info(f"Additional keys in checkpoint: {sorted(additional)}")

    stripped_state_dict = {
        k: v for k, v in stripped_state_dict.items() if k in model_dict
    }
    keys_to_remove = []
    for k, v in stripped_state_dict.items():
        if model_dict[k].size() != v.size():
            logger.info(
                f"Size mismatch for {k}, checkpoint has shape {v.size()} and current model has shape {model_dict[k].size()}."
            )
            keys_to_remove.append(k)
    for k in keys_to_remove:
        stripped_state_dict.pop(k, None)

    model_dict.update(stripped_state_dict)
    brain_model.load_state_dict(model_dict)
    logger.info(f"Loaded model hash: {model_hash(brain_model)}")

    return brain_model


def get_targets_from_dataset(dataset: SegmentDataset) -> torch.Tensor:
    feat_dataset = copy(dataset)
    # Drop neuro as it takes the most time to process
    feat_dataset.extractors = {"target": feat_dataset.extractors["target"]}
    return feat_dataset.load_all().data["target"]


def get_neuro_and_targets_from_dataset(
    dataset: SegmentDataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialise ``(neuro, target)`` tensors for the whole dataset.

    Used by fit-once baselines (e.g. :class:`SklearnBaseline`) that need
    the full training set as a single NumPy array rather than mini-batches.
    Other extractors (e.g. ``channel_positions``, ``subject_id``) are
    dropped to minimise memory and load time.
    """
    feat_dataset = copy(dataset)
    feat_dataset.extractors = {
        "neuro": feat_dataset.extractors["neuro"],
        "target": feat_dataset.extractors["target"],
    }
    data = feat_dataset.load_all().data
    return data["neuro"], data["target"]


def compute_class_weights_from_dataset(
    train_dataset: SegmentDataset,
    logger: logging.Logger,
    task: tp.Literal["multiclass", "multilabel", "auto"] = "auto",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Compute class weights from training dataset for handling class imbalance."""
    targets = get_targets_from_dataset(train_dataset)
    if isinstance(train_dataset.extractors["target"], ns.extractors.LabelEncoder):
        class_mapping = list(train_dataset.extractors["target"]._label_to_ind.keys())
    else:
        class_mapping = [str(i) for i in range(int(targets.shape[-1]))]
    logger.info("Computing class weights...")

    if task == "auto":
        task = "multiclass"
        if targets.ndim == 2:
            n_classes_per_example = targets.sum(dim=1)
            if (n_classes_per_example > 1).any() or (n_classes_per_example == 0).any():
                task = "multilabel"

    loss_kwargs = {}
    if task == "multilabel":
        if targets.ndim != 2:
            raise ValueError("Expected 2D targets for multilabel")
        n_classes_per_example = targets.sum(dim=1)
        has_multi = (n_classes_per_example > 1).any()
        has_zero = (n_classes_per_example == 0).any()
        if not (has_multi or has_zero):
            raise ValueError("Expected some examples with multiple or zero classes")

        y_true = targets.clamp(max=1.0).bool().squeeze(dim=1)
        pos_weight = (~y_true).sum(dim=0) / y_true.sum(dim=0)  # n_negatives / n_positives
        pos_weight = torch.nan_to_num(pos_weight, posinf=1.0)
        pos_weight_dict = dict(zip(class_mapping, pos_weight.tolist()))
        logger.info(f"Positive class weights: {pos_weight_dict}")
        loss_kwargs["pos_weight"] = pos_weight  # For BCEWithLogitsLoss

    elif task == "multiclass":
        if targets.ndim == 2:
            n_classes = targets.shape[1]
            if n_classes == 1:
                y_true = targets.squeeze(dim=1)
            else:
                y_true = targets.argmax(dim=-1)
        else:
            y_true = targets
            n_classes = int(y_true.max().item()) + 1
        observed_classes = np.unique(y_true)
        observed_weights = torch.tensor(
            compute_class_weight(
                class_weight="balanced",
                classes=observed_classes,
                y=y_true.tolist(),
            )
        ).float()
        if len(observed_classes) < n_classes:
            class_weights = torch.ones(n_classes, dtype=torch.float32)
            for cls, w in zip(observed_classes, observed_weights):
                class_weights[int(cls)] = w
        else:
            class_weights = observed_weights
        class_weights_dict = dict(zip(class_mapping, class_weights.tolist()))
        logger.info(f"Class weights: {class_weights_dict}")
        loss_kwargs["weight"] = class_weights

    return loss_kwargs, y_true


def make_weighted_sampler(
    dataset: SegmentDataset,
    logger: logging.Logger,
) -> torch.utils.data.WeightedRandomSampler:
    """Create a weighted random sampler for the given dataset to handle class imbalance."""
    loss_kwargs, y_true = compute_class_weights_from_dataset(
        dataset,
        logger=logger,
        task="multiclass",
    )
    # TODO: Adapt to work with multilabel case as well
    weights = loss_kwargs["weight"][y_true]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=True,
    )
    return sampler


class TrainerConfig(ns.BaseModel):
    """Joint configuration for Trainer and some callbacks."""

    n_epochs: int = 100
    enable_progress_bar: bool = True
    log_every_n_steps: int = 20
    fast_dev_run: bool = False
    gradient_clip_val: float = 0.0
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None
    num_sanity_val_steps: int = 2
    accumulate_grad_batches: int = 1

    # Hardware
    strategy: str = "auto"
    precision: str = "32-true"
    accelerator: str = "auto"
    devices: int = 1
    num_nodes: int = 1

    # Callbacks
    patience: int = 5
    monitor: str = "val/loss"
    mode: str = "min"

    def build(
        self,
        logger,
        callbacks,
        accelerator: str | None = None,
        devices: int | None = None,
        num_nodes: int | None = None,
    ) -> pl.Trainer:
        return pl.Trainer(
            strategy=self.strategy,
            precision=self.precision,  # type: ignore[arg-type]
            accelerator=self.accelerator if accelerator is None else accelerator,
            devices=self.devices if devices is None else devices,
            num_nodes=self.num_nodes if num_nodes is None else num_nodes,
            gradient_clip_val=self.gradient_clip_val,
            limit_train_batches=self.limit_train_batches,
            limit_val_batches=self.limit_val_batches,
            max_epochs=self.n_epochs,
            enable_progress_bar=self.enable_progress_bar,
            log_every_n_steps=self.log_every_n_steps,
            num_sanity_val_steps=self.num_sanity_val_steps,
            fast_dev_run=self.fast_dev_run,
            accumulate_grad_batches=self.accumulate_grad_batches,
            logger=logger,
            callbacks=callbacks,
            enable_model_summary=False,
        )
