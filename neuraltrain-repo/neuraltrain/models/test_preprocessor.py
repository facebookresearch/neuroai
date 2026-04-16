# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pytest
import torch

from .common import INVALID_POS_VALUE
from .preprocessor import OnTheFlyPreprocessor, OnTheFlyPreprocessorModel


@pytest.fixture
def fake_meg() -> torch.Tensor:
    batch_size = 4
    n_channels = 8
    n_times = 120
    meg = torch.randn(batch_size, n_channels, n_times)
    return meg


@pytest.mark.parametrize("update_ch_pos", [True, False])
def test_on_the_fly_preprocessor_ptp_threshold(
    fake_meg: torch.Tensor,
    update_ch_pos: bool,
) -> None:
    batch_size, in_channels, _ = fake_meg.shape
    fake_meg[0, 0, :] *= 100  # One obviously bad channel

    channel_positions = torch.randn(batch_size, in_channels, 2)

    preproc = OnTheFlyPreprocessor(
        ptp_threshold=10.0,
        update_ch_pos=update_ch_pos,
    ).build()
    assert isinstance(preproc, OnTheFlyPreprocessorModel)

    x_out, ch_pos_out = preproc(fake_meg, channel_positions=channel_positions)
    assert x_out.shape == fake_meg.shape

    if update_ch_pos:
        assert all(ch_pos_out[0, 0, :] == INVALID_POS_VALUE)


@pytest.mark.parametrize(
    "scaler,scale_dim",
    [
        [None, None],
        [0.5, None],
        [2.0, None],
        ["StandardScaler", 1],
        ["StandardScaler", [1, 2]],
        ["StandardScaler", None],
        ["RobustScaler", 1],
        ["RobustScaler", None],
        ["QuantileAbsScaler", -1],
        ["QuantileAbsScaler", None],
    ],
)
def test_on_the_fly_preprocessor_scaler(
    fake_meg: torch.Tensor,
    scaler: (
        None | float | tp.Literal["RobustScaler", "StandardScaler", "QuantileAbsScaler"]
    ),
    scale_dim: None | int | list[int],
) -> None:
    preproc = OnTheFlyPreprocessor(
        scaler=scaler,
        scale_dim=scale_dim,
    ).build()
    assert isinstance(preproc, OnTheFlyPreprocessorModel)

    x_out, _ = preproc(fake_meg)
    if isinstance(scaler, float):
        assert torch.allclose(x_out, fake_meg * scaler)
        return

    if scaler == "RobustScaler":
        stats = x_out.quantile(
            torch.tensor([0.25, 0.5, 0.75]),
            dim=scale_dim,
            keepdim=True,
        )
        loc, scale = stats[1], stats[2] - stats[0]
    elif scaler == "StandardScaler":
        loc = x_out.mean(dim=scale_dim, keepdim=True)
        scale = x_out.std(dim=scale_dim, keepdim=True)
    else:
        return

    assert torch.allclose(loc, torch.zeros_like(loc), atol=1e-5)
    assert torch.allclose(scale, torch.ones_like(loc), atol=1e-5)


def test_on_the_fly_preprocessor_clamp(fake_meg: torch.Tensor) -> None:
    fake_meg[:, :, 0] = 10.0
    clamp = 3.0

    preproc = OnTheFlyPreprocessor(
        clamp=clamp,
    ).build()
    assert isinstance(preproc, OnTheFlyPreprocessorModel)

    x_out, _ = preproc(fake_meg)
    assert torch.allclose(x_out[:, :, 0], torch.tensor(clamp))


@pytest.mark.parametrize("quantile", [0.5, 0.75, 0.95])
def test_quantile_abs_scaler(quantile: float) -> None:
    """Verify QuantileAbsScaler divides each channel by the correct quantile."""
    torch.manual_seed(42)
    x = torch.randn(2, 3, 200)

    preproc = OnTheFlyPreprocessor(
        scaler="QuantileAbsScaler",
        scaler_quantile=quantile,
        scale_dim=-1,
    ).build()
    x_out, _ = preproc(x)

    expected_q = torch.quantile(x.abs(), quantile, dim=-1, keepdim=True)
    expected_q[expected_q == 0.0] = 1.0
    expected = x / expected_q

    assert x_out.shape == x.shape
    assert torch.allclose(x_out, expected, atol=1e-6)


def test_quantile_abs_scaler_zero_channel() -> None:
    """A channel that is all zeros should remain zero (no division by zero)."""
    x = torch.randn(1, 2, 100)
    x[:, 0, :] = 0.0

    preproc = OnTheFlyPreprocessor(
        scaler="QuantileAbsScaler",
        scaler_quantile=0.95,
        scale_dim=-1,
    ).build()
    x_out, _ = preproc(x)

    assert torch.all(x_out[:, 0, :] == 0.0)
    assert not torch.any(torch.isnan(x_out))
    assert not torch.any(torch.isinf(x_out))


def test_minmax_scaler() -> None:
    """MinMaxScaler rescales each channel to [-1, 1] along the time dim."""
    x = torch.randn(2, 3, 50)
    x[:, 0, :] *= 100.0
    x[:, 1, :] += 500.0

    preproc = OnTheFlyPreprocessor(scaler="MinMaxScaler").build()
    x_out, _ = preproc(x)

    assert x_out.min() >= -1.0 - 1e-6
    assert x_out.max() <= 1.0 + 1e-6
    assert not torch.any(torch.isnan(x_out))


def test_minmax_scaler_constant_channel() -> None:
    """A constant channel should be mapped to -1 (not NaN)."""
    x = torch.randn(1, 2, 50)
    x[:, 0, :] = 5.0

    preproc = OnTheFlyPreprocessor(scaler="MinMaxScaler").build()
    x_out, _ = preproc(x)

    assert torch.allclose(x_out[:, 0, :], torch.tensor(-1.0))
    assert not torch.any(torch.isnan(x_out))
