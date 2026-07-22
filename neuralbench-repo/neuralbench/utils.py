# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions."""

import logging
import random
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
from neuralset.events import etypes
from neuralset.extractors import LabelEncoder
from neuralset.utils import warn_once

LOGGER = logging.getLogger(__name__)


def seed_worker(worker_id: int) -> None:
    """Seed the per-worker ``numpy`` and ``random`` RNGs from torch's per-worker seed.

    PyTorch already seeds the per-worker ``torch`` RNG from the parent
    ``DataLoader``'s ``generator`` (when set) at worker spawn time, but
    ``numpy.random`` and Python's ``random`` modules are left untouched.  This
    helper mirrors :func:`lightning.pytorch.utilities.seed.pl_worker_init_function`
    but without relying on the ``PL_SEED_WORKERS`` environment variable, so that
    :class:`~neuralbench.data.Data` can make worker-side determinism a function
    of its own ``seed`` field rather than a Lightning side-effect.
    """
    del worker_id  # Pulled from worker_info to match PyTorch's per-worker seed
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info is not None, (
        "seed_worker must be called inside a DataLoader worker"
    )
    seed = worker_info.seed % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def model_hash(model: nn.Module) -> str:
    hasher = sha1()
    for p in model.parameters():
        hasher.update(p.data.cpu().numpy().tobytes())
    return hasher.hexdigest()


def run_probe_hook(
    model: nn.Module,
    submodule: nn.Module,
    model_batch: dict[str, torch.Tensor | None],
    probe_layer: str,
) -> torch.Tensor:
    """Run one dummy forward and return the tensor captured at ``submodule``.

    A temporary forward hook on ``submodule`` records its last output during a
    single ``model(**model_batch)`` pass; the hook is always removed afterwards.
    ``probe_layer`` is used only to make error messages actionable.

    Raises
    ------
    RuntimeError
        If the hook never fires (submodule unreachable from this input).
    TypeError
        If the captured output is not a tensor (e.g. an ``nn.MultiheadAttention``
        or ``nn.RNN`` that returns a tuple/dict).
    """
    # Single-slot buffer: keep only the last fire so a submodule that runs
    # many times (e.g. recurrent) costs O(1) memory rather than O(n_fires).
    probed_activation: list[torch.Tensor] = []

    def _hook(_m, _i, out):
        probed_activation[:] = [out]

    handle = submodule.register_forward_hook(_hook)
    try:
        model(**model_batch)
    finally:
        handle.remove()
    if not probed_activation:
        raise RuntimeError(
            f"probe_layer={probe_layer!r} hook did not fire during "
            f"the dummy forward; the submodule is unreachable from this input."
        )
    capture = probed_activation[0]
    if not isinstance(capture, torch.Tensor):
        raise TypeError(
            f"probe_layer={probe_layer!r} returned "
            f"{type(capture).__name__}; only tensor-returning "
            f"submodules are supported (e.g. probe a parent of "
            f"nn.MultiheadAttention, not the MHA itself)."
        )
    return capture


def detect_batch_dim(
    model: nn.Module,
    submodule: nn.Module,
    model_batch: dict[str, torch.Tensor | None],
    probe_layer: str,
) -> int:
    """Find the batch axis of ``submodule``'s output via a doubled-batch forward.

    Captures the activation at the input batch size and again at a doubled batch
    size: the batch axis is the only one whose size scales with the input batch,
    while sequence/feature axes stay constant. ``probe_layer`` is used only for
    the error message.

    Raises
    ------
    ValueError
        When the layout is genuinely ambiguous (zero or multiple candidate
        axes), suggesting the caller pass an explicit batch dim.
    """
    batch_size = next(
        v.shape[0] for v in model_batch.values() if isinstance(v, torch.Tensor)
    )
    capture = run_probe_hook(model, submodule, model_batch, probe_layer)
    doubled = {
        k: torch.cat([v, v], dim=0)
        if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == batch_size
        else v
        for k, v in model_batch.items()
    }
    capture2 = run_probe_hook(model, submodule, doubled, probe_layer)
    candidates = [
        i
        for i in range(capture.ndim)
        if i < capture2.ndim
        and capture.shape[i] == batch_size
        and capture2.shape[i] == 2 * batch_size
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"probe_layer={probe_layer!r} batch axis is ambiguous: "
            f"capture shape {tuple(capture.shape)} scaled to "
            f"{tuple(capture2.shape)} when the batch size doubled, giving "
            f"candidate axes {candidates}. Set probe_batch_dim explicitly."
        )
    return candidates[0]


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
    generator: torch.Generator | None = None,
) -> torch.utils.data.WeightedRandomSampler:
    """Create a weighted random sampler for the given dataset to handle class imbalance.

    Parameters
    ----------
    dataset
        Training dataset whose targets drive the class-weight computation.
    logger
        Logger forwarded to :func:`compute_class_weights_from_dataset`.
    generator
        Optional ``torch.Generator`` used by the returned sampler.  When set,
        successive iterations of the sampler draw from this generator instead
        of the global ``torch`` RNG, so the sampling sequence is determined
        solely by the generator's seed.
    """
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
        generator=generator,
    )
    return sampler


def _compute_regression_bin_weights(
    targets: torch.Tensor,
    bin_edges: tp.Sequence[float],
) -> torch.Tensor:
    """Compute per-sample inverse-frequency weights from a regression target tensor.

    Targets are bucketised into ``len(bin_edges) - 1`` bins defined by
    ``bin_edges``.  Bin ``i`` covers ``[bin_edges[i], bin_edges[i + 1])`` for
    ``i < n_bins - 1``; the top bin is closed on the right
    (``[bin_edges[-2], bin_edges[-1]]``) so that targets exactly at
    ``bin_edges[-1]`` (e.g. the cap value in ``AddSleepOnsetTargets``) are
    counted in the top bin.  This matches the bin semantics of
    :class:`~neuralbench.metrics.BinnedMAE`.

    Targets falling outside ``[bin_edges[0], bin_edges[-1]]`` receive a weight
    of ``0`` (they are effectively excluded from sampling) and do not
    contribute to any bin's count.

    Each sample's weight is ``1 / count_in_its_bin`` so that, in expectation,
    every populated bin contributes the same total mass to a weighted sampler.

    Parameters
    ----------
    targets : torch.Tensor
        1-D float tensor of shape ``(n_samples,)``.  Trailing singleton dims are
        not handled here; squeeze upstream.
    bin_edges : Sequence[float]
        Strictly increasing sequence of length ``>= 2``.

    Returns
    -------
    torch.Tensor
        1-D float tensor of shape ``(n_samples,)`` with per-sample sampling
        weights.  Empty bins and out-of-range samples contribute zero weight.
    """
    if targets.ndim != 1:
        raise ValueError(f"Expected 1-D targets, got shape {tuple(targets.shape)}.")
    if len(bin_edges) < 2:
        raise ValueError(
            f"bin_edges must have length >= 2, got {len(bin_edges)}: {list(bin_edges)}"
        )

    inner_edges = torch.as_tensor(list(bin_edges)[1:-1], dtype=targets.dtype)
    n_bins = len(bin_edges) - 1
    bin_idx = torch.bucketize(targets, inner_edges, right=False).clamp_(0, n_bins - 1)

    in_range = (targets >= bin_edges[0]) & (targets <= bin_edges[-1])
    counts = torch.bincount(bin_idx[in_range], minlength=n_bins).to(dtype=torch.float32)
    inv_counts = torch.where(counts > 0, 1.0 / counts, torch.zeros_like(counts))

    weights = torch.zeros(targets.shape[0], dtype=torch.float32, device=targets.device)
    weights[in_range] = inv_counts[bin_idx[in_range]]
    return weights


def make_regression_bin_sampler(
    dataset: SegmentDataset,
    bin_edges: tp.Sequence[float],
    logger: logging.Logger,
    generator: torch.Generator | None = None,
) -> torch.utils.data.WeightedRandomSampler:
    """Create a regression-bin stratified weighted sampler.

    Materialises the dataset's targets, bins them by ``bin_edges`` and assigns
    inverse-frequency sampling weights so that, in expectation, every populated
    bin contributes equally to each training epoch.  This is the regression
    counterpart of :func:`make_weighted_sampler`.

    Parameters
    ----------
    dataset : SegmentDataset
        Training segment dataset; its ``target`` extractor must yield a scalar
        regression target per sample.
    bin_edges : Sequence[float]
        Strictly increasing bin edges.  Targets outside
        ``[bin_edges[0], bin_edges[-1]]`` receive zero sampling weight (matches
        the bin semantics of :class:`~neuralbench.metrics.BinnedMAE`).
    logger : logging.Logger
        Logger used to report per-bin counts and the number of out-of-range
        targets.
    """
    targets = get_targets_from_dataset(dataset)
    if targets.ndim == 2 and targets.shape[-1] == 1:
        targets = targets.squeeze(-1)
    weights = _compute_regression_bin_weights(targets, bin_edges)

    n_bins = len(bin_edges) - 1
    inner_edges = torch.as_tensor(list(bin_edges)[1:-1], dtype=targets.dtype)
    bin_idx = torch.bucketize(targets, inner_edges, right=False).clamp_(0, n_bins - 1)
    in_range = (targets >= bin_edges[0]) & (targets <= bin_edges[-1])
    counts = torch.bincount(bin_idx[in_range], minlength=n_bins).tolist()
    bin_labels = [
        f"[{bin_edges[i]:g}, {bin_edges[i + 1]:g}" + ("]" if i == n_bins - 1 else ")")
        for i in range(n_bins)
    ]
    logger.info(
        "Regression-bin sampler counts: %s (out-of-range: %d)",
        dict(zip(bin_labels, counts)),
        int((~in_range).sum().item()),
    )

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
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


class SequenceLabelEncoder(LabelEncoder):
    """Fixed-length integer-label sequence extractor for CTC training.

    Reads a pre-computed integer ``event_field`` from events,
    concatenates the per-segment values, and pads or truncates the
    result to ``max_length`` with ``pad_value`` (the canonical CTC
    blank index).  Labels are assumed already in ``[0, pad_value)`` --
    no string→int mapping happens at encode time; upstream code (e.g.
    the study's events-dataframe construction) is responsible for
    dropping OOV rows and assigning the canonical paper labels.

    Inheriting from :class:`neuralset.extractors.LabelEncoder` keeps
    the shared Pydantic surface (``event_types``, ``event_field``,
    ``aggregation``, ``allow_missing``) while replacing the
    string-encoding machinery with a direct integer read.
    """

    max_length: int
    pad_value: int
    aggregation: tp.Literal["cat"] = "cat"

    @property
    def n_classes(self) -> int:
        """CTC head width: ``pad_value`` labels + one blank = ``pad_value + 1``."""
        return self.pad_value + 1

    def model_post_init(self, log__: tp.Any) -> None:
        # Skip ``LabelEncoder.model_post_init``: its "missing events ->
        # all-zero default" warning is a false positive here (missing
        # segments are encoded as an empty sequence padded with
        # ``pad_value`` -- see ``__call__`` -- not zeros), and its
        # ``predefined_mapping`` validation doesn't apply (labels are read
        # as integers, no string->int mapping).  Run the grandparent
        # (``EventField``) init only -- mirrors how ``prepare`` deliberately
        # bypasses ``LabelEncoder.prepare``.
        super(LabelEncoder, self).model_post_init(log__)
        if self.max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {self.max_length}.")
        if self.pad_value < 0:
            raise ValueError(f"pad_value must be >= 0, got {self.pad_value}.")

    def prepare(self, obj: tp.Any) -> None:
        # Skip ``LabelEncoder.prepare`` which would try to learn a
        # string→int mapping from the integer ``label`` field — it
        # builds a *sorted-enumerate* mapping, so missing values in
        # ``obj`` would corrupt the canonical ``[0, pad_value)``
        # layout.  ``EventField.prepare`` (one level up the MRO) still
        # validates the field's dtype, and the one-shot ``self(...)``
        # call populates the output shape used when
        # ``allow_missing=True``.
        super(LabelEncoder, self).prepare(obj)
        events = self._event_types_helper.extract(obj)
        if events:
            self(events[0], events[0].start, duration=0.001, trigger=events[0])

    def get_static(self, event: etypes.Event) -> torch.Tensor:  # type: ignore[override]
        return torch.tensor(
            [int(event._get_field_or_extra(self.event_field))],
            dtype=torch.long,
        )

    def __call__(  # type: ignore[override]
        self,
        events: tp.Any,
        start: float,
        duration: float,
        trigger: tp.Any = None,
    ) -> torch.Tensor:
        # CTC accepts length-0 targets; short-circuit empty segments
        # instead of letting ``BaseExtractor`` synthesize a 1-label
        # ``_missing_default`` fallback (which would land in the
        # middle of the vocabulary).
        if self.allow_missing and not any(
            start <= e.start < start + duration
            for e in self._event_types_helper.extract(events)
        ):
            return torch.full((self.max_length,), self.pad_value, dtype=torch.long)
        seq = super().__call__(events, start, duration, trigger=trigger)
        n = int(seq.numel())
        if n > self.max_length:
            warn_once(f"{self.name}: truncating labels to max_length={self.max_length}")
            seq, n = seq[: self.max_length], self.max_length
        return torch.nn.functional.pad(
            seq, (0, self.max_length - n), value=self.pad_value
        )
