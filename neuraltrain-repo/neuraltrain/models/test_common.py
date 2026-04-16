# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from math import sqrt

import pytest
import torch
from torch import nn
from torchvision.ops import MLP

from .common import (
    INVALID_POS_VALUE,
    ChannelMerger,
    ChannelMergerModel,
    FourierEmb,
    FourierEmbModel,
    Mean,
    Mlp,
    NormDenormScaler,
    SubjectLayersModel,
    TemporalDownsampling,
    UnitNorm,
    apply_temporal_adjustment,
    compute_temporal_adjustment,
)


@pytest.fixture
def fake_meg() -> torch.Tensor:
    batch_size = 4
    n_channels = 8
    n_times = 120
    meg = torch.randn(batch_size, n_channels, n_times)
    return meg


def test_invalid_pos_value_matches_neuralset() -> None:
    from neuralset.extractors.neuro import ChannelPositions

    assert INVALID_POS_VALUE == ChannelPositions.INVALID_VALUE


@pytest.mark.parametrize("init_id", [False, True])
@pytest.mark.parametrize("out_channels", [8, 9])
@pytest.mark.parametrize("mode", ["gather", "for_loop"])
def test_subject_layers(
    fake_meg: torch.Tensor,
    init_id: bool,
    out_channels: int,
    mode: tp.Literal["gather", "for_loop"],
) -> None:
    batch_size, in_channels, n_times = fake_meg.shape
    n_subjects = 4
    subjects = torch.arange(0, batch_size) % n_subjects

    if init_id and in_channels != out_channels:
        with pytest.raises(ValueError):
            layer = SubjectLayersModel(
                in_channels,
                out_channels,
                init_id=init_id,
                mode=mode,
                n_subjects=n_subjects,
            )
    else:
        layer = SubjectLayersModel(
            in_channels, out_channels, init_id=init_id, mode=mode, n_subjects=n_subjects
        )
        out = layer(fake_meg, subjects)
        assert out.shape == (batch_size, out_channels, n_times)
        out = layer(fake_meg.mean(dim=2), subjects)
        assert out.shape == (batch_size, out_channels)


def test_subject_layers_invalid_subject(fake_meg: torch.Tensor) -> None:
    batch_size, in_channels, _ = fake_meg.shape
    n_subjects = 1
    layer = SubjectLayersModel(in_channels, 2, n_subjects=n_subjects, init_id=False)

    subjects = torch.ones(batch_size)
    with pytest.raises(ValueError):
        layer(fake_meg, subjects)


@pytest.mark.parametrize("bias", [False, True])
def test_subject_layers_average_and_bias(fake_meg: torch.Tensor, bias: bool) -> None:
    batch_size, in_channels, n_times = fake_meg.shape
    n_subjects = 4
    layer = SubjectLayersModel(
        in_channels,
        2,
        n_subjects=n_subjects,
        init_id=False,
        subject_dropout=0.5,
        average_subjects=True,
        bias=bias,
    )

    subjects = torch.zeros(batch_size, dtype=torch.long)
    layer.eval()
    out = layer(fake_meg, subjects)
    assert out.shape == (batch_size, 2, n_times)


@pytest.mark.parametrize("mode", ["gather", "for_loop"])
def test_subject_layers_dropout(
    fake_meg: torch.Tensor, mode: tp.Literal["gather", "for_loop"]
) -> None:
    batch_size, in_channels, n_times = fake_meg.shape
    n_subjects = 4
    layer = SubjectLayersModel(
        in_channels,
        2,
        n_subjects=n_subjects,
        init_id=False,
        subject_dropout=0.5,
        mode=mode,
    )

    subjects = torch.arange(batch_size) % n_subjects
    layer.train()
    out = layer(fake_meg, subjects)
    assert out.shape == (batch_size, 2, n_times)
    assert layer.weights.shape[0] == n_subjects + 1


def test_fourier_emb() -> None:
    D = 288
    emb = FourierEmb(
        n_freqs=None,
        total_dim=D,
        margin=0.1,
    ).build()
    assert isinstance(emb, FourierEmbModel)

    B, C = 8, 16
    positions = torch.zeros((B, C, 2))
    out = emb(positions)
    assert out.shape == (B, C, D)

    # Avoid previous bug where positions were modified in-place in FourierEmb
    assert positions.shape == (B, C, 2)
    assert (positions == 0.0).all()


@pytest.mark.parametrize("n_dims", [1, 2, 3, 4])
def test_fourier_emb_n_dims(n_dims: int) -> None:
    n_freqs = 12
    emb = FourierEmb(
        n_freqs=n_freqs,
        total_dim=None,
        n_dims=n_dims,
        margin=0.2,
    ).build()
    assert isinstance(emb, FourierEmbModel)

    B, C = 8, 16
    positions = torch.rand((B, C, n_dims))
    out = emb(positions)
    assert out.shape == (B, C, (n_freqs**n_dims) * 2)


@pytest.mark.parametrize(
    "dropout,usage_penalty,per_subject,n_pos_dims",
    [[0.0, 0.0, False, 2], [0.5, 0.5, True, 3]],
)
def test_channel_merger_shape(fake_meg, dropout, usage_penalty, per_subject, n_pos_dims):
    n_virtual_channels = 8
    pos_dim = (12**n_pos_dims) * 2

    batch_size, n_in_channels, n_times = fake_meg.shape
    positions = torch.rand(batch_size, n_in_channels, n_pos_dims)
    subject_ids = torch.arange(batch_size)

    merger = ChannelMerger(
        n_virtual_channels=n_virtual_channels,
        fourier_emb_config=FourierEmb(
            n_freqs=None,
            total_dim=pos_dim,
            n_dims=n_pos_dims,
        ),
        dropout=dropout,
        usage_penalty=usage_penalty,
        n_subjects=10,
        per_subject=per_subject,
    ).build()
    assert isinstance(merger, ChannelMergerModel)

    out = merger(fake_meg, subject_ids, positions)
    assert out.shape == (batch_size, n_virtual_channels, n_times)


@pytest.mark.parametrize("n_pos_dims", [1, 2, 3])
def test_channel_merger_bipolar(fake_meg, n_pos_dims):
    n_virtual_channels = 10
    pos_dim = (12**n_pos_dims) * 2

    batch_size, n_in_channels, n_times = fake_meg.shape
    positions = torch.rand(batch_size, n_in_channels, n_pos_dims * 2)
    subject_ids = torch.arange(batch_size)

    merger = ChannelMerger(
        n_virtual_channels=n_virtual_channels,
        fourier_emb_config=FourierEmb(
            n_freqs=None,
            total_dim=pos_dim,
            n_dims=n_pos_dims,
        ),
        dropout=0.0,
        usage_penalty=0.0,
        n_subjects=10,
        per_subject=False,
        embed_ref=True,
    ).build()
    assert isinstance(merger, ChannelMergerModel)

    out = merger(fake_meg, subject_ids, positions)
    assert out.shape == (batch_size, n_virtual_channels, n_times)


@pytest.mark.parametrize("n_pos_dims", [2, 3])
@pytest.mark.parametrize("embed_ref", [True, False])
@pytest.mark.parametrize("dropout_around_channel", [True, False])
def test_channel_merger_dropout(
    fake_meg,
    n_pos_dims: int,
    embed_ref: bool,
    dropout_around_channel: bool,
) -> None:
    n_virtual_channels = 10
    pos_dim = (12**n_pos_dims) * 2

    batch_size, n_in_channels, _ = fake_meg.shape
    positions = torch.rand(
        batch_size, n_in_channels, n_pos_dims * 2 if embed_ref else n_pos_dims
    )
    subject_ids = torch.arange(batch_size)

    merger = ChannelMerger(
        n_virtual_channels=n_virtual_channels,
        fourier_emb_config=FourierEmb(
            n_freqs=None,
            total_dim=pos_dim,
            n_dims=n_pos_dims,
        ),
        dropout=sqrt(n_pos_dims),  # To make sure we drop everything in [0, 1] ^ D
        dropout_around_channel=dropout_around_channel,
        embed_ref=embed_ref,
    ).build()

    weights = merger._get_weights(subject_ids, positions, positions.device)
    assert weights.shape == (batch_size, n_virtual_channels, n_in_channels)
    assert (weights == 0.0).all()


def test_channel_unmerger_shape(fake_meg):
    chout = 12
    pos_dim = 288

    batch_size, n_virtual_channels, n_times = fake_meg.shape
    positions = torch.randn(batch_size, chout, 2)
    subject_ids = torch.arange(batch_size)

    merger = ChannelMerger(
        n_virtual_channels=n_virtual_channels,
        fourier_emb_config=FourierEmb(
            n_freqs=None,
            total_dim=pos_dim,
        ),
        unmerge=True,
    ).build()
    assert isinstance(merger, ChannelMergerModel)

    out = merger(fake_meg, subject_ids, positions)
    assert out.shape == (batch_size, chout, n_times)


@pytest.mark.parametrize("affine", [True, False])
def test_norm_denorm_scaler(affine: bool) -> None:
    n_feats = 10
    x1 = torch.rand(100, n_feats)
    x2 = torch.rand(20, n_feats)
    scaler = NormDenormScaler(x1, affine=affine)

    out = scaler(x2)

    if affine:
        assert torch.allclose(out.mean(dim=0), x1.mean(dim=0))
        assert torch.allclose(out.std(dim=0, correction=0), x1.std(dim=0, correction=0))
        assert torch.allclose(out.mean(dim=0), scaler.scaler.bias)
        assert torch.allclose(out.std(dim=0, correction=0), scaler.scaler.weight)
    else:
        torch.allclose(out.mean(dim=0), torch.zeros(n_feats))
        torch.allclose(out.mean(dim=0), torch.ones(n_feats))


@pytest.mark.skip(reason="TODO")
def test_bahdanau_attention():
    raise NotImplementedError


def test_unit_norm() -> None:
    x = torch.rand(10, 16)
    norm = UnitNorm()
    y = norm(x)
    assert torch.allclose(y.norm(dim=-1), torch.tensor(1.0))


@pytest.mark.parametrize("n_layers", [1, 3, 0])
@pytest.mark.parametrize("hidden_size", [2, 4])
@pytest.mark.parametrize("norm_kind", [None, "layer", "batch", "instance", "unit"])
@pytest.mark.parametrize("nonlin_kind", [None, "gelu", "relu", "elu", "prelu"])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_mlp(n_layers, hidden_size, norm_kind, nonlin_kind, dropout) -> None:
    batch_size = 32
    input_size = 8
    output_size = 16
    x = torch.rand(batch_size, input_size)

    mlp = Mlp(
        hidden_sizes=[hidden_size] * n_layers,
        norm_layer=norm_kind,
        activation_layer=nonlin_kind,
        dropout=dropout,
    ).build(
        input_size,
        output_size,
    )

    if n_layers == 0:
        assert isinstance(mlp, nn.Linear)
    else:
        assert isinstance(mlp, MLP)
    out = mlp(x)
    assert out.shape == (batch_size, output_size)


@pytest.mark.parametrize("output_size", [None, 2])
def test_mlp_input_output_sizes(output_size) -> None:
    batch_size = 32
    input_size = 8
    hidden_size = 16
    x = torch.rand(batch_size, input_size)

    mlp = Mlp(
        input_size=input_size,
        hidden_sizes=[hidden_size] * 3,
    ).build(
        output_size=output_size,
    )

    out = mlp(x)
    assert out.shape == (batch_size, hidden_size if output_size is None else output_size)


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_mean(dim: int, keepdim: bool) -> None:
    x = torch.rand((10, 9, 8))
    mean_layer = Mean(dim=dim, keepdim=keepdim)
    out = mean_layer(x)
    assert torch.allclose(out, x.mean(dim=dim, keepdim=keepdim))


@pytest.mark.parametrize("use_default_config", [True, False])
def test_temporal_downsampling(use_default_config: bool) -> None:
    batch_size, n_times, dim = 8, 64, 32
    x = torch.rand((batch_size, 1, n_times, dim))

    config_kwargs = {}
    if not use_default_config:
        config_kwargs = dict(
            kernel_size=16,
            stride=16,
            layer_norm=False,
            layer_norm_affine=False,
            gelu=False,
        )

    model = TemporalDownsampling(**config_kwargs).build(dim=dim)  # type: ignore
    out = model(x)
    out_n_times = 1 if use_default_config else n_times // 16
    assert out.shape == (batch_size, 1, out_n_times, dim)


# -- temporal adjustment helpers ---------------------------------------------------


@pytest.mark.parametrize(
    "n_times, patch_size, expected",
    [
        (200, 200, (0, 0)),  # exact match
        (400, 200, (0, 0)),  # exact multiple
        (50, 200, (150, 0)),  # shorter than one patch -> pad
        (210, 200, (0, 10)),  # not divisible -> truncate remainder
        (1, 200, (199, 0)),  # minimal input -> pad
    ],
)
def test_compute_temporal_adjustment(
    n_times: int, patch_size: int, expected: tuple[int, int]
) -> None:
    assert compute_temporal_adjustment(n_times, patch_size) == expected


def test_apply_temporal_adjustment_pad() -> None:
    x = torch.ones(2, 3, 10)
    out = apply_temporal_adjustment(x, pad_right=5)
    assert out.shape == (2, 3, 15)
    assert (out[:, :, :10] == 1).all()
    assert (out[:, :, 10:] == 0).all()


def test_apply_temporal_adjustment_truncate() -> None:
    x = torch.ones(2, 3, 10)
    out = apply_temporal_adjustment(x, truncate_right=3)
    assert out.shape == (2, 3, 7)


def test_apply_temporal_adjustment_noop() -> None:
    x = torch.ones(2, 3, 10)
    out = apply_temporal_adjustment(x)
    assert out is x


def test_channel_merger_embed_ref_auto_pad(fake_meg: torch.Tensor) -> None:
    """ChannelMergerModel auto-pads positions from n_dims to 2*n_dims when embed_ref=True."""
    n_virtual_channels = 8
    n_pos_dims = 2
    pos_dim = (12**n_pos_dims) * 2

    batch_size, n_in_channels, n_times = fake_meg.shape
    # Provide positions with only n_dims columns (not 2*n_dims)
    positions = torch.rand(batch_size, n_in_channels, n_pos_dims)
    subject_ids = torch.arange(batch_size)

    merger = ChannelMerger(
        n_virtual_channels=n_virtual_channels,
        fourier_emb_config=FourierEmb(
            n_freqs=None,
            total_dim=pos_dim,
            n_dims=n_pos_dims,
        ),
        dropout=0.0,
        usage_penalty=0.0,
        n_subjects=10,
        embed_ref=True,
    ).build()
    assert isinstance(merger, ChannelMergerModel)
    assert merger.embed_ref is True

    out = merger(fake_meg, subject_ids, positions)
    assert out.shape == (batch_size, n_virtual_channels, n_times)
