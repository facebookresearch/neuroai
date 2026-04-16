# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchvision.ops import MLP

from .common import ChannelMergerModel
from .simpleconv import SimpleConv, SimpleConvTimeAgg


@pytest.fixture
def fake_meg():
    batch_size = 2
    n_channels = 4
    n_times = 120
    meg = torch.randn(batch_size, n_channels, n_times)
    return meg


@pytest.mark.parametrize(
    "depth,use_merger,subject_layers,initial_linear",
    [
        [4, False, None, 0],
        [6, True, {}, 4],
    ],
)
def test_simple_conv_shape(fake_meg, depth, use_merger, subject_layers, initial_linear):
    batch_size, n_in_channels, n_times = fake_meg.shape
    n_out_channels = 6

    config_kwargs = {
        "hidden": 5,
        "depth": depth,
        "subject_layers_config": subject_layers,
        "initial_linear": initial_linear,
    }
    if not use_merger:
        config_kwargs["merger_config"] = None

    model = SimpleConv(**config_kwargs).build(n_in_channels, n_out_channels)
    if use_merger:
        assert isinstance(model.merger, ChannelMergerModel)

    subject_ids = torch.arange(batch_size)
    channel_positions = torch.randn(batch_size, n_in_channels, 2)

    out = model(fake_meg, subject_ids=subject_ids, channel_positions=channel_positions)
    assert out.shape == (batch_size, n_out_channels, n_times)


@pytest.mark.parametrize("n_pos_dims", [2, 3])
def test_simple_conv_different_n_in_channels(fake_meg, n_pos_dims):
    batch_size, n_in_channels, _ = fake_meg.shape
    n_out_channels = 6

    config_kwargs = {
        "hidden": 5,
        "depth": 2,
        "merger_config": {
            "fourier_emb_config": {
                "n_freqs": None,
                "total_dim": (8**n_pos_dims) * 2,
                "n_dims": n_pos_dims,
            },
        },
        "subject_layers_config": None,
        "initial_linear": False,
    }
    model = SimpleConv(**config_kwargs).build(n_in_channels, n_out_channels)
    channel_positions = torch.randn(batch_size, n_in_channels, n_pos_dims)

    out1 = model(fake_meg, channel_positions=channel_positions)
    out2 = model(fake_meg[:, :3, :], channel_positions=channel_positions[:, :3, :])
    assert out1.shape == out2.shape


@pytest.mark.parametrize("time_agg_out", ["gap", "linear", "att"])
@pytest.mark.parametrize("n_time_groups", [None, 4, 7])
def test_simple_conv_time_agg_shape(
    fake_meg, time_agg_out: str, n_time_groups: int
) -> None:
    batch_size, n_in_channels, _ = fake_meg.shape
    n_out_channels = 64

    model_kwargs = {
        "hidden": 64,
        "merger_config": None,
        "subject_layers_config": None,
        "time_agg_out": time_agg_out,
        "n_time_groups": n_time_groups,
    }
    model = SimpleConvTimeAgg(**model_kwargs).build(n_in_channels, n_out_channels)  # type: ignore
    out = model(fake_meg)
    if n_time_groups is None:
        assert out.shape == (batch_size, n_out_channels)
    else:
        assert out.shape == (batch_size, n_out_channels, n_time_groups)


def test_simple_conv_time_agg_zero_depth_shape(fake_meg):
    batch_size, n_in_channels, _ = fake_meg.shape
    n_out_channels = 64

    model_kwargs = {
        "hidden": 64,
        "depth": 0,
        "merger_config": None,
        "subject_layers_config": None,
        "initial_linear": n_out_channels,
        "backbone_out_channels": n_out_channels,
        "time_agg_out": "gap",
        "output_head_config": None,
    }
    model = SimpleConvTimeAgg(**model_kwargs).build(n_in_channels, n_out_channels)
    out = model(fake_meg)
    assert out.shape == (batch_size, n_out_channels)


@pytest.mark.parametrize("output_size", [-1, 2])
def test_simple_conv_time_agg_output_head(fake_meg, output_size: int):
    batch_size, n_in_channels, _ = fake_meg.shape
    head_out = 8

    model_kwargs = {
        "hidden": 5,
        "merger_config": None,
        "subject_layers_config": None,
        "time_agg_out": "gap",
        "backbone_out_channels": 10,
        "output_head_config": {
            "hidden_sizes": [2, output_size],
            "norm_layer": "layer",
            "activation_layer": "relu",
            "dropout": 0.5,
        },
    }
    model = SimpleConvTimeAgg(**model_kwargs).build(  # type: ignore
        n_in_channels, head_out
    )

    out = model(fake_meg)
    assert isinstance(model.output_head, MLP)
    assert out.shape == (batch_size, head_out if output_size == -1 else output_size)


def test_simple_conv_time_agg_output_heads(fake_meg):
    batch_size, n_in_channels, _ = fake_meg.shape
    n_out_channels = 6

    clip_head_out = 8

    model_kwargs = {
        "hidden": 16,
        "merger_config": None,
        "subject_layers_config": None,
        "time_agg_out": "gap",
        "output_head_config": {
            "clip": {
                "hidden_sizes": [clip_head_out],
                "norm_layer": "layer",
                "activation_layer": "relu",
                "dropout": 0.5,
            },
            "mse": {
                "hidden_sizes": [32, -1],
                "norm_layer": "instance",
                "activation_layer": "gelu",
                "dropout": 0.0,
            },
        },
    }
    model = SimpleConvTimeAgg(**model_kwargs).build(n_in_channels, n_out_channels)

    out = model(fake_meg)
    assert out["clip"].shape == (batch_size, clip_head_out)
    assert out["mse"].shape == (batch_size, n_out_channels)
