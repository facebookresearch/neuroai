# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`neuralbench.model_factory`."""

import typing as tp
from collections.abc import Callable

import pytest
import torch
from torch import nn

from neuraltrain.models.base import BaseBrainModelConfig

from . import model_factory as _model_factory
from .data import Data
from .model_factory import build_brain_model
from .modules import ChannelProjection, DownstreamWrapper


class _Passthrough(BaseBrainModelConfig):
    """Encoder-only config returning the input unchanged (for wrapper tests)."""

    def build(self, n_spatial_locations: int, n_outputs: int | None = None) -> nn.Module:
        return nn.Identity()


_RECORDED_N_OUTPUTS: list[int | None] = []


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((2, 7, 11), (7, 11)),
        ((2, 3, 4, 5, 11), (3 * 4 * 5, 11)),
    ],
)
def test_infer_neuro_shape(
    shape: tuple[int, ...],
    expected: tuple[int, int],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger=_model_factory.LOGGER.name)
    assert _model_factory._infer_neuro_shape(torch.empty(shape)) == expected
    if len(shape) > 3:
        assert "input tensor remains unflattened" in caplog.text


class _RecordHeadWidth(BaseBrainModelConfig):
    """Config recording the head width the factory derives from the target."""

    def build(self, n_spatial_locations: int, n_outputs: int | None = None) -> nn.Module:
        _RECORDED_N_OUTPUTS.append(n_outputs)
        return nn.Identity()


class _NameReader(nn.Module):
    """Model reading channel identity by name, as LaBraM does."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] | None = None

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: torch.Tensor | None = None,
        ch_names: list[str] | None = None,
    ) -> torch.Tensor:
        self.seen = ch_names
        return x


class _NameReaderConfig(BaseBrainModelConfig):
    """Config building a model whose ``forward`` names ``ch_names``."""

    def build(self, n_spatial_locations: int, n_outputs: int | None = None) -> nn.Module:
        return _NameReader()


def test_build_brain_model_names_channels_for_a_model_that_reads_them(
    build_data: Callable[..., Data],
) -> None:
    loader = build_data(seed=0).prepare()["train"]
    ch_names = list(loader.dataset.extractors["neuro"]._channels.keys())  # type: ignore[attr-defined]

    model, _, _, returned = build_brain_model(
        brain_model_config=_NameReaderConfig(),
        downstream_model_wrapper=None,
        pretrained_weights_fname=None,
        train_loader=loader,
    )

    # Returned for the training loop to pass on every batch, and already
    # carried by the init pass: a model that asks for names never runs without.
    assert returned == ch_names
    assert tp.cast(_NameReader, model).seen == ch_names


def test_build_brain_model_forwards_dataset_channel_names_to_adapter(
    build_data: Callable[..., Data],
) -> None:
    loader = build_data(seed=0).prepare()["train"]
    neuro_extractor = loader.dataset.extractors["neuro"]  # type: ignore[attr-defined]
    ch_names = list(neuro_extractor._channels.keys())

    wrapper = DownstreamWrapper(
        channel_adapter_config=ChannelProjection(
            n_target_channels=len(ch_names),
            init="identity",
            target_channel_names=ch_names,
            max_norm=None,
        ),
        aggregation="flatten",
    )

    model, _, _, _ = build_brain_model(
        brain_model_config=_Passthrough(),
        downstream_model_wrapper=wrapper,
        pretrained_weights_fname=None,
        train_loader=loader,
    )

    # An identity-init channel adapter clears the *context* ch_names (the brain
    # model sees the adapter output), but the wrapper must still receive the
    # dataset channel names -- the adapter's input -- to build its name-matched
    # projection. Target names equal the dataset names, so the identity init
    # yields the identity weight; reusing the cleared ch_names would raise.
    expected = torch.eye(len(ch_names)).unsqueeze(-1)
    adapter = tp.cast(nn.Conv1d, model.channel_adapter)
    assert torch.allclose(adapter.weight, expected), (
        "identity-init adapter weight is not the identity; dataset channel "
        "names did not reach the adapter."
    )


def test_build_brain_model_sizes_head_from_dense_target_channel_axis(
    build_data: Callable[..., Data],
) -> None:
    # A dense target (an extractor's raw ``(C, T)`` output rather than an
    # encoded label) is channel-major, so its last axis is time.  Sizing the
    # head from it gives a model as wide as the window is long, which only
    # surfaces later as a shape mismatch in the loss.
    data = build_data(seed=0, target={"name": "MneRaw", "event_types": "Eeg"})
    loader = data.prepare()["train"]
    batch = next(iter(loader))
    target = batch.data["target"]
    assert target.ndim == 3, "fixture no longer yields a dense target"

    # Exercise the full factory with a volumetric neuro tensor while preserving
    # the same number of spatial locations and temporal samples.
    neuro = batch.data["neuro"]
    batch.data["neuro"] = neuro.reshape(
        neuro.shape[0],
        1,
        1,
        neuro.shape[1],
        neuro.shape[2],
    )

    class _SingleBatchLoader:
        def __init__(self) -> None:
            self.dataset = loader.dataset

        def __iter__(self):
            return iter([batch])

    _RECORDED_N_OUTPUTS.clear()
    model, _, _, _ = build_brain_model(
        brain_model_config=_RecordHeadWidth(),
        downstream_model_wrapper=None,
        pretrained_weights_fname=None,
        train_loader=tp.cast(tp.Any, _SingleBatchLoader()),
    )

    n_channels, n_times = target.shape[1], target.shape[2]
    assert _RECORDED_N_OUTPUTS == [n_channels], (
        f"head sized {_RECORDED_N_OUTPUTS} rather than {[n_channels]} channels "
        f"(the window is {n_times} samples long)."
    )
    assert model(batch.data["neuro"]).shape == batch.data["neuro"].shape
