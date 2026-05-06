# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Functions for building and initialising brain models.

Extracted from :class:`neuralbench.main.Experiment` to keep the experiment
lifecycle class focused on orchestration.
"""

import inspect
import logging
import typing as tp

import numpy as np
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchinfo import summary

from neuraltrain.models.base import BaseBrainDecodeModel, BaseModelConfig
from neuraltrain.models.common import ChannelMerger
from neuraltrain.models.dummy_predictor import DummyPredictor

from .modules import DownstreamWrapper
from .sklearn_baseline import SklearnBaseline
from .utils import (
    get_neuro_and_targets_from_dataset,
    get_targets_from_dataset,
    load_checkpoint,
)

LOGGER = logging.getLogger(__name__)


def build_braindecode_model(
    brain_model_config: BaseBrainDecodeModel,
    downstream_model_wrapper: DownstreamWrapper | None,
    train_loader: DataLoader,
    n_in_channels: int,
    n_times: int,
    n_outputs: int,
) -> torch.nn.Module:
    """Build a braindecode model, handling channel name extraction.

    Some braindecode models (REVE, BENDR, LaBraM) require explicit channel
    name information via ``chs_info``.  Model-specific adaptation (e.g.
    temporal embedding resizing for LaBraM) is handled by the model
    config's own ``build()`` method.

    When *downstream_model_wrapper* is set, ``n_outputs`` is not passed to
    the model (the wrapper's probe handles output projection).
    """
    build_kwargs: dict[str, tp.Any] = {}
    _needs_ch_names = getattr(brain_model_config, "chs_info_required", False)
    if downstream_model_wrapper is None:
        assert brain_model_config.__class__.__name__ != "BIOT", (
            "BIOT requires a downstream_model_wrapper"
        )
        build_kwargs["n_outputs"] = n_outputs
    if _needs_ch_names:
        neuro_extractor = train_loader.dataset.extractors["neuro"]  # type: ignore[attr-defined]
        ch_names = list(neuro_extractor._channels.keys())
        assert len(ch_names) == n_in_channels, (
            f"Expected {n_in_channels} channels, but got {len(ch_names)} channel names."
        )
        build_kwargs["chs_info"] = [{"ch_name": name} for name in ch_names]

    if brain_model_config.from_pretrained_name is not None:
        n_adapter_chans = (
            downstream_model_wrapper.n_adapter_target_channels
            if downstream_model_wrapper is not None
            else None
        )
        build_kwargs["n_chans"] = (
            n_adapter_chans if n_adapter_chans is not None else n_in_channels
        )
        if getattr(brain_model_config, "needs_n_times", False):
            build_kwargs["n_times"] = n_times
    else:
        build_kwargs.update(n_chans=n_in_channels, n_times=n_times)

    return brain_model_config.build(**build_kwargs)


def build_dummy_batch(
    brain_model: torch.nn.Module,
    batch: tp.Any,
    downstream_model_wrapper: DownstreamWrapper | None,
) -> tuple[dict[str, torch.Tensor | None], str]:
    """Build a single-sample dummy batch from *batch* for lazy-layer init and model summary.

    Returns the dummy batch dict and the name of the primary input parameter.
    """
    forward_sig = inspect.signature(brain_model.forward)
    input_name = list(forward_sig.parameters.keys())[0]
    dummy_batch: dict[str, torch.Tensor | None] = {
        input_name: batch.data["neuro"][:1].to("cpu"),
    }
    if "subject_ids" in forward_sig.parameters:
        dummy_batch["subject_ids"] = batch.data["subject_id"][:1].to("cpu")
    if "channel_positions" in forward_sig.parameters:
        dummy_batch["channel_positions"] = batch.data["channel_positions"][:1].to("cpu")

    # ChannelMerger-based adapters need channel_positions and subject_ids
    # even if the inner model doesn't require them.
    if downstream_model_wrapper is not None and isinstance(
        downstream_model_wrapper.channel_adapter_config, ChannelMerger
    ):
        if "channel_positions" not in dummy_batch:
            dummy_batch["channel_positions"] = batch.data["channel_positions"][:1].to(
                "cpu"
            )
        if "subject_ids" not in dummy_batch:
            dummy_batch["subject_ids"] = batch.data["subject_id"][:1].to("cpu")

    return dummy_batch, input_name


def init_lazy_layers(
    brain_model: torch.nn.Module,
    dummy_batch: dict[str, torch.Tensor | None],
    input_name: str,
    downstream_model_wrapper: DownstreamWrapper | None,
) -> None:
    """Run a forward pass to materialise lazy layers.

    When a channel adapter is configured the input tensor is replaced
    with one that has the adapter's target channel count.
    Only parameters accepted by ``brain_model.forward`` are passed.
    """
    with torch.no_grad():
        brain_model.eval()
        model_sig = inspect.signature(brain_model.forward)
        model_param_names = set(model_sig.parameters.keys())
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in model_sig.parameters.values()
        )
        if has_var_kwargs:
            init_batch = dict(dummy_batch)
        else:
            init_batch = {k: v for k, v in dummy_batch.items() if k in model_param_names}
        n_adapted = (
            downstream_model_wrapper.n_adapter_target_channels
            if downstream_model_wrapper is not None
            else None
        )
        if n_adapted is not None:
            x = init_batch[input_name]
            assert x is not None
            init_batch[input_name] = torch.randn(x.shape[0], n_adapted, *x.shape[2:])
        brain_model(**init_batch)
        brain_model.train()


def build_brain_model(
    *,
    brain_model_config: BaseModelConfig,
    downstream_model_wrapper: DownstreamWrapper | None,
    pretrained_weights_fname: str | None,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    wandb_logger: WandbLogger | None = None,
) -> tuple[torch.nn.Module, int, int]:
    """Build, initialise and optionally wrap a brain model.

    This is the main entry point that orchestrates braindecode/generic model
    construction, lazy-layer initialisation, pretrained-weight loading,
    downstream wrapping, and model summary logging.

    ``val_loader`` is only consumed by the fit-once baselines
    (:class:`~neuralbench.sklearn_baseline.SklearnBaseline` and
    :class:`~neuraltrain.models.dummy_predictor.DummyPredictor`) which have no
    early-stopping-style use for it.  When provided, they are fit on the
    concatenation of the train and val splits so they see the same pool of
    labelled data as the DL models (which get val via early stopping).

    Returns ``(model, n_total_params, n_trainable_params)``.
    """
    batch = next(iter(train_loader))
    n_in_channels, n_times = batch.data["neuro"].shape[1:]
    LOGGER.info(f"Neuro shape: {batch.data['neuro'].shape}")
    LOGGER.info(f"Target shape: {batch.data['target'].shape}")

    feat = batch.data["target"]
    n_outputs = feat.shape[-1]

    # 1) Build the brain model
    if isinstance(brain_model_config, BaseBrainDecodeModel):
        brain_model = build_braindecode_model(
            brain_model_config,
            downstream_model_wrapper,
            train_loader,
            n_in_channels,
            n_times,
            n_outputs,
        )
    elif isinstance(brain_model_config, DummyPredictor):
        # TODO: share computations with compute_class_weights_from_dataset and/or target_scaler
        LOGGER.info("Preparing DummyPredictor model...")
        y_train = get_targets_from_dataset(train_loader.dataset)  # type: ignore[arg-type]
        n_train = int(y_train.shape[0])
        if val_loader is not None:
            y_val = get_targets_from_dataset(val_loader.dataset)  # type: ignore[arg-type]
            y_fit = torch.cat([y_train, y_val], dim=0)
            LOGGER.info(
                "DummyPredictor: fitting on N_train + N_val = %d + %d = %d targets.",
                n_train,
                int(y_val.shape[0]),
                int(y_fit.shape[0]),
            )
        else:
            y_fit = y_train
            LOGGER.info("DummyPredictor: fitting on N_train = %d targets.", n_train)
        brain_model = brain_model_config.build(y_train=y_fit)
    elif isinstance(brain_model_config, SklearnBaseline):
        LOGGER.info(
            "Preparing SklearnBaseline model (%s)...",
            type(brain_model_config).__name__,
        )
        X_train, y_train = get_neuro_and_targets_from_dataset(train_loader.dataset)  # type: ignore[arg-type]
        X_train_np = X_train.cpu().numpy()
        y_train_np = y_train.cpu().numpy()
        X_fit_np: np.ndarray
        y_fit_np: np.ndarray
        if val_loader is not None:
            X_val, y_val_tensor = get_neuro_and_targets_from_dataset(
                val_loader.dataset  # type: ignore[arg-type]
            )
            X_val_np = X_val.cpu().numpy()
            y_val_np = y_val_tensor.cpu().numpy()
            LOGGER.info(
                "SklearnBaseline: fitting on N_train + N_val = %d + %d = %d trials.",
                X_train_np.shape[0],
                X_val_np.shape[0],
                X_train_np.shape[0] + X_val_np.shape[0],
            )
            X_fit_np = np.concatenate([X_train_np, X_val_np], axis=0)
            y_fit_np = np.concatenate([y_train_np, y_val_np], axis=0)
        else:
            LOGGER.info(
                "SklearnBaseline: fitting on N_train = %d trials.",
                X_train_np.shape[0],
            )
            X_fit_np = X_train_np
            y_fit_np = y_train_np
        brain_model = brain_model_config.build(
            X_train=X_fit_np,
            y_train=y_fit_np,
        )
    else:
        brain_model = brain_model_config.build(
            n_in_channels=n_in_channels,
            n_outputs=(None if downstream_model_wrapper is not None else n_outputs),
        )

    # 2) Initialize lazy layers
    dummy_batch, input_name = build_dummy_batch(
        brain_model, batch, downstream_model_wrapper
    )
    init_lazy_layers(brain_model, dummy_batch, input_name, downstream_model_wrapper)

    # 3) Load pretrained weights
    if brain_model_config is not None and pretrained_weights_fname is not None:
        brain_model = load_checkpoint(brain_model, pretrained_weights_fname, LOGGER)

    # 4) Wrap for downstream task
    if downstream_model_wrapper is not None:
        LOGGER.info("Wrapping brain model for downstream task...")
        input_channel_names: list[str] | None = None
        neuro_extractor = getattr(train_loader.dataset, "extractors", {}).get("neuro")
        if neuro_extractor is not None and hasattr(neuro_extractor, "_channels"):
            input_channel_names = list(neuro_extractor._channels.keys())
        brain_model = downstream_model_wrapper.build(
            brain_model,
            dummy_batch,
            n_outputs,
            input_channel_names=input_channel_names,
        )

    # 5) Log model summary
    model_summary = summary(
        brain_model,
        input_data=dummy_batch,
        row_settings=("hide_recursive_layers",),
        verbose=0,
    )
    LOGGER.info("Model summary:\n%s", model_summary)
    n_total_params: int = model_summary.total_params
    n_trainable_params: int = model_summary.trainable_params
    if wandb_logger is not None:
        wandb_logger.experiment.config["n_total_params"] = n_total_params
        wandb_logger.experiment.config["n_trainable_params"] = n_trainable_params

    return brain_model, n_total_params, n_trainable_params
