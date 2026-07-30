# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`neuralbench.model_factory`."""

import typing as tp
from collections.abc import Callable

import torch
from torch import nn

from neuraltrain.models.base import BaseBrainModelConfig

from .data import Data
from .model_factory import build_brain_model
from .modules import ChannelProjection, DownstreamWrapper


class _Passthrough(BaseBrainModelConfig):
    """Encoder-only config returning the input unchanged (for wrapper tests)."""

    def build(self, n_spatial_locations: int, n_outputs: int | None = None) -> nn.Module:
        return nn.Identity()


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

    model, _, _ = build_brain_model(
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
