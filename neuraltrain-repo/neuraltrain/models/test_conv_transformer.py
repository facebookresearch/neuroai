# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .conv_transformer import ConvTransformer, ConvTransformerModel
from .simpleconv import SimpleConvModel
from .simplerconv import SimplerConvModel


@pytest.fixture
def fake_eeg():
    batch_size = 16
    n_channels = 8
    n_times = 800
    meg = torch.randn(batch_size, n_channels, n_times)
    return meg


def test_conv_transformer_with_simpler_conv(fake_eeg):
    batch_size, n_in_channels, n_times = fake_eeg.shape
    channel_positions = torch.rand((batch_size, n_in_channels, 2))

    dim = 512
    model = ConvTransformer(
        encoder_config=dict(
            name="SimplerConv",
            merger_config={},
            kernel_sizes=(5, 2),
            strides=(5, 2),
            dropout=0.5,
        ),
        temporal_downsampling_config=dict(
            kernel_size=2,
            stride=2,
            layer_norm=True,
            layer_norm_affine=True,
            gelu=True,
        ),
        conv_pos_emb_kernel_size=5,
        transformer_config=None,
    ).build(n_in_channels=n_in_channels, n_outputs=dim)
    assert isinstance(model, ConvTransformerModel)
    assert isinstance(model.encoder, SimplerConvModel)

    out = model(fake_eeg, channel_positions=channel_positions)
    assert out.keys() == {"z", "c_out"}

    receptive_field = 20  # (5 * 2) * 2
    assert out["z"].shape == (batch_size, n_times / receptive_field, dim)
    assert out["c_out"].shape == (batch_size, n_times / receptive_field, dim)


def test_conv_transformer_with_simple_conv(fake_eeg):
    batch_size, n_in_channels, n_times = fake_eeg.shape

    dim, output_layer_dim = 512, 4
    model = ConvTransformer(
        encoder_config=dict(
            name="SimpleConv",
            merger_config=None,
        ),
        temporal_downsampling_config=dict(
            kernel_size=4,
            stride=4,
        ),
        conv_pos_emb_kernel_size=5,
        add_cls_token=True,
        pre_transformer_layer_norm=True,
        transformer_config=dict(
            name="TransformerEncoder",
            heads=1,
            depth=1,
        ),
        output_layer_dim=output_layer_dim,
    ).build(n_in_channels=n_in_channels)
    assert isinstance(model, ConvTransformerModel)
    assert isinstance(model.encoder, SimpleConvModel)

    out = model(fake_eeg)
    assert out.keys() == {"z", "c_out"}

    assert out["z"].shape == (batch_size, n_times / 4, dim)
    assert out["c_out"].shape == (batch_size, 1 + n_times / 4, output_layer_dim)


@pytest.mark.parametrize("output_layer_dim", [None, 0, 4])
def test_conv_transformer_output_layer(fake_eeg, output_layer_dim: int | None):
    batch_size, n_in_channels, n_times = fake_eeg.shape

    dim = 512
    model = ConvTransformer(
        encoder_config=dict(  # type: ignore
            name="SimpleConv",
            merger_config=None,
        ),
        transformer_config=dict(  # type: ignore
            name="TransformerEncoder",
            heads=1,
            depth=1,
        ),
        output_layer_dim=output_layer_dim,
    ).build(n_in_channels=n_in_channels)
    assert isinstance(model, ConvTransformerModel)
    assert isinstance(model.encoder, SimpleConvModel)

    out = model(fake_eeg)
    assert out.keys() == {"z", "c_out"}

    if output_layer_dim is None:
        assert isinstance(model.output_layer, torch.nn.Linear)
        output_dim = dim
    elif output_layer_dim == 0:
        assert model.output_layer is None
        output_dim = dim
    else:
        assert isinstance(model.output_layer, torch.nn.Linear)
        output_dim = output_layer_dim

    assert out["c_out"].shape == (batch_size, n_times, output_dim)


def test_conv_transformer_with_neuro_device_tokens(fake_eeg):
    batch_size, n_in_channels, n_times = fake_eeg.shape

    neuro_device_types = ["Eeg", "Meg"]
    dim, output_layer_dim = 512, 4
    model = ConvTransformer(
        encoder_config=dict(
            name="SimpleConv",
            merger_config=None,
        ),
        neuro_device_types=neuro_device_types,
        transformer_config={
            "name": "TransformerEncoder",
            "heads": 1,
            "depth": 1,
        },
        output_layer_dim=output_layer_dim,
    ).build(n_in_channels=n_in_channels)
    assert isinstance(model, ConvTransformerModel)
    assert isinstance(model.encoder, SimpleConvModel)
    assert model.neuro_device_emb.weight.shape == (len(neuro_device_types), dim)

    for neuro_device_type in neuro_device_types:
        out = model(fake_eeg, neuro_device_type=neuro_device_type)
        assert out.keys() == {"z", "c_out"}

        assert out["z"].shape == (batch_size, n_times, dim)
        assert out["c_out"].shape == (batch_size, n_times, output_layer_dim)

    with pytest.raises(ValueError):
        out = model(fake_eeg, neuro_device_type="invalid_type")

    # Make sure neuro device embeddings are used correctly
    model.eval()
    with torch.inference_mode():
        out_eeg1 = model(fake_eeg, neuro_device_type="Eeg")
        out_eeg2 = model(fake_eeg, neuro_device_type="Eeg")
        out_meg = model(fake_eeg, neuro_device_type="Meg")

        assert (out_eeg1["c_out"] == out_eeg2["c_out"]).all()
        assert not (out_eeg1["c_out"] == out_meg["c_out"]).all()


def test_conv_transformer_with_conformer(fake_eeg):
    from torchaudio.models import Conformer

    batch_size, n_in_channels, n_times = fake_eeg.shape
    channel_positions = torch.rand((batch_size, n_in_channels, 2))

    dim = 512
    model = ConvTransformer(
        encoder_config=dict(
            name="SimplerConv",
            kernel_sizes=(5, 2),
            strides=(5, 2),
        ),
        temporal_downsampling_config=None,
        conv_pos_emb_kernel_size=5,
        transformer_config={
            "name": "Conformer",
            "num_heads": 2,
            "ffn_dim": 12,
            "num_layers": 1,
            "depthwise_conv_kernel_size": 5,
            "dropout": 0.1,
            "use_group_norm": True,
            "convolution_first": False,
        },
    ).build(n_in_channels=n_in_channels, n_outputs=dim)
    assert isinstance(model, ConvTransformerModel)
    assert isinstance(model.transformer, Conformer)

    out = model(fake_eeg, channel_positions=channel_positions)
    assert out.keys() == {"z", "c_out"}

    receptive_field = 10
    assert out["z"].shape == (batch_size, n_times / receptive_field, dim)
    assert out["c_out"].shape == (batch_size, n_times / receptive_field, dim)
