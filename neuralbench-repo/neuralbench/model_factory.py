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

import torch
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchinfo import summary

from neuraltrain.losses import BaseLoss
from neuraltrain.losses.base import BaseTorchLoss
from neuraltrain.models.base import (
    BaseBrainModelConfig,
    BaseModelConfig,
    BrainModelBuildContext,
)
from neuraltrain.models.common import ChannelMerger

from .baselines import BaseFitOnceModelConfig, FitOnceBuildContext
from .modules import DownstreamWrapper
from .utils import (
    get_neuro_and_targets_from_dataset,
    get_targets_from_dataset,
    load_checkpoint,
)

LOGGER = logging.getLogger(__name__)


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
    loss: BaseLoss | None = None,
) -> tuple[torch.nn.Module, int, int]:
    """Build, initialise and optionally wrap a brain model.

    This is the main entry point that orchestrates braindecode/generic model
    construction, lazy-layer initialisation, pretrained-weight loading,
    downstream wrapping, and model summary logging.

    ``val_loader`` is only consumed by the fit-once baselines
    (:class:`~neuralbench.baselines.SklearnBaseline` and
    :class:`~neuralbench.baselines.DummyPredictor`) which have no
    early-stopping-style use for it.  When provided, they are fit on the
    concatenation of the train and val splits so they see the same pool of
    labelled data as the DL models (which get val via early stopping).

    Returns ``(model, n_total_params, n_trainable_params)``.
    """
    batch = next(iter(train_loader))
    n_spatial_locations, n_temporal_samples = batch.data["neuro"].shape[1:]
    LOGGER.info(f"Neuro shape: {batch.data['neuro'].shape}")
    LOGGER.info(f"Target shape: {batch.data['target'].shape}")

    feat = batch.data["target"]
    # Some target extractors expose an explicit head width via
    # ``n_classes`` because their target shape doesn't reveal it -- e.g.
    # ``SequenceLabelEncoder`` for CTC, whose ``(max_length,)`` target is
    # padded class indices, not a one-hot row.  Unwrap meta-extractors
    # (e.g. ``CroppedExtractor``, which nests its inner extractor under
    # ``.extractor``) to reach the leaf that may carry it; fall back to
    # ``feat.shape[-1]`` for one-hot / scalar targets.
    leaf = getattr(train_loader.dataset, "extractors", {}).get("target")
    while (
        leaf is not None and not hasattr(leaf, "n_classes") and hasattr(leaf, "extractor")
    ):
        leaf = leaf.extractor
    n_outputs = (
        leaf.n_classes
        if leaf is not None and hasattr(leaf, "n_classes")
        else feat.shape[-1]
    )

    # Derive sampling rate / channel names from the neuro extractor so models
    # that need them (frequency: Green/FreqBandNet/CoSpectra; ch_names:
    # LaBraM/REVE) get data-correct values instead of hardcoded config ones.
    frequency: float | None = None
    ch_names: list[str] | None = None
    mesh: str | None = None
    neuro_extractor = getattr(train_loader.dataset, "extractors", {}).get("neuro")
    if neuro_extractor is not None:
        freq = getattr(neuro_extractor, "frequency", None)
        if isinstance(freq, (int, float)) and not isinstance(freq, bool):
            frequency = float(freq)
        if hasattr(neuro_extractor, "_channels"):
            ch_names = list(neuro_extractor._channels.keys())
        # Surface-sampled data (e.g. fMRI on fsaverage) exposes a mesh
        # resolution that surface models need; carry its enum name as a string.
        mesh_attr = getattr(neuro_extractor, "mesh_resolution", None)
        if mesh_attr is not None:
            mesh = mesh_attr.name if hasattr(mesh_attr, "name") else str(mesh_attr)

    # The dataset channel names always describe the adapter's *input*, so the
    # downstream wrapper needs them for name-matched adapter init (identity /
    # bipolar) regardless of any adapter reshaping below.
    dataset_ch_names = ch_names

    # A channel adapter changes the spatial size the model sees: use the
    # adapter's output width and clear ch_names (its outputs no longer map to
    # dataset channel names).
    if downstream_model_wrapper is not None:
        adapter_target = downstream_model_wrapper.n_adapter_target_channels
        if adapter_target is not None:
            n_spatial_locations = adapter_target
            ch_names = None

    # CTC sequence tasks (e.g. emg/typing) carry a blank-token index on the
    # loss; derive the scalar here so the context stays free of the loss
    # config object (only DummyPredictor's CTC baseline consumes it).
    ctc_blank_idx: int | None = None
    if isinstance(loss, BaseTorchLoss) and type(loss).__name__ == "CTCLoss":
        ctc_blank_idx = int(loss.kwargs.get("blank", 0))

    # BIOT produces no standalone classification output suitable for the
    # benchmark; it must be paired with a downstream wrapper.
    if downstream_model_wrapper is None:
        assert type(brain_model_config).__name__ != "BIOT", (
            "BIOT requires a downstream_model_wrapper"
        )

    # Lazy fit-data materializers for the fit-once baselines, closing over the
    # (train, val) loaders so neuraltrain needn't depend on neuralbench's
    # dataset utils; only the baselines that need the full pool invoke them.
    def _load_targets() -> torch.Tensor:
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
            return y_fit
        LOGGER.info("DummyPredictor: fitting on N_train = %d targets.", n_train)
        return y_train

    def _load_neuro_targets() -> tuple[torch.Tensor, torch.Tensor]:
        X_train, y_train = get_neuro_and_targets_from_dataset(train_loader.dataset)  # type: ignore[arg-type]
        if val_loader is not None:
            X_val, y_val = get_neuro_and_targets_from_dataset(val_loader.dataset)  # type: ignore[arg-type]
            LOGGER.info(
                "SklearnBaseline: fitting on N_train + N_val = %d + %d = %d trials.",
                int(X_train.shape[0]),
                int(X_val.shape[0]),
                int(X_train.shape[0] + X_val.shape[0]),
            )
            return torch.cat([X_train, X_val], dim=0), torch.cat([y_train, y_val], dim=0)
        LOGGER.info(
            "SklearnBaseline: fitting on N_train = %d trials.", int(X_train.shape[0])
        )
        return X_train, y_train

    # 1) Build the brain model. Fit-once baselines get the richer
    # FitOnceBuildContext (lazy fit-data materializers + CTC blank index);
    # all other models get the general context.
    assert isinstance(brain_model_config, BaseBrainModelConfig), (
        f"{type(brain_model_config).__name__} is not a BaseBrainModelConfig; "
        "only brain-model configs can be built by the neuralbench factory."
    )
    resolved_n_outputs = None if downstream_model_wrapper is not None else n_outputs
    ctx: BrainModelBuildContext
    if isinstance(brain_model_config, BaseFitOnceModelConfig):
        ctx = FitOnceBuildContext(
            n_spatial_locations=int(n_spatial_locations),
            n_temporal_samples=int(n_temporal_samples),
            n_outputs=resolved_n_outputs,
            frequency=frequency,
            ch_names=ch_names,
            mesh=mesh,
            ctc_blank_idx=ctc_blank_idx,
            load_targets=_load_targets,
            load_neuro_targets=_load_neuro_targets,
        )
    else:
        ctx = BrainModelBuildContext(
            n_spatial_locations=int(n_spatial_locations),
            n_temporal_samples=int(n_temporal_samples),
            n_outputs=resolved_n_outputs,
            frequency=frequency,
            ch_names=ch_names,
            mesh=mesh,
        )
    brain_model = brain_model_config.build_from_context(ctx)

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
        brain_model = downstream_model_wrapper.build(
            brain_model,
            dummy_batch,
            n_outputs,
            input_channel_names=dataset_ch_names,
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
