# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch
from torch import nn
from torch.nn.functional import conv2d

from .base import BaseModelConfig

EEG_FREQ_LIMS_HZ = [
    0.1,
    0.5,
    4.0,
    8.0,
    12.0,
    30.0,
    60.0,
]


class FreqBandNet(BaseModelConfig):
    """Parametrized filterbank feature extractor (sinc filters + power + log).

    Parameters
    ----------
    sfreq : float or None
        Sampling frequency of the input time series, in Hz.  Can also be
        provided at build time.
    freq_lims_hz : list of float or None
        Bandpass cutoff frequencies used to initialize the sinc filters.
        ``len(freq_lims_hz) - 1`` filters are created.  Mutually exclusive
        with *n_filters_conv*.
    n_filters_conv : int or None
        Number of filters to initialize with log-spaced cutoffs.  Mutually
        exclusive with *freq_lims_hz*.
    conv_kernel_len : int
        Kernel length (in samples) of each sinc convolution filter.
    conv_stride : int
        Stride of the sinc convolution.
    conv_padding : {"valid", "same"}
        Padding mode for the sinc convolution.
    pool_kernel_len : int
        Kernel length for average-pooling after the filterbank.
    flat_out : {"channels", "channels_and_time"} or None
        How to flatten the filterbank output before the classifier.
        ``"channels"`` merges filters and channels → ``(B, F*C, T)``.
        ``"channels_and_time"`` flattens everything → ``(B, F*C*T)``.
    n_outputs : int or None
        If set, append a linear output layer with this many units.
    """

    sfreq: float | None = None
    freq_lims_hz: list[float] | None = EEG_FREQ_LIMS_HZ
    n_filters_conv: int | None = None
    conv_kernel_len: int = 65
    conv_stride: int = 1
    conv_padding: tp.Literal["valid", "same"] = "valid"
    pool_kernel_len: int = 30
    flat_out: tp.Literal["channels", "channels_and_time"] | None = None
    n_outputs: int | None = None

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if not (self.freq_lims_hz is None) ^ (self.n_filters_conv is None):
            raise ValueError(
                "Exactly one of freq_lims_hz and n_filters_conv must be specified."
            )

    def build(
        self,
        n_in_channels: int | None = None,  # Unused; for compatibility with other models
        n_outputs: int | None = None,
        sfreq: float | None = None,
    ) -> nn.Module:
        sfreq = sfreq or self.sfreq
        if sfreq is None:
            raise ValueError("sfreq must be provided to build the model.")

        return FreqBandNetModel(
            n_outputs=n_outputs or self.n_outputs,
            sfreq=sfreq,
            config=self,
        )


class FreqBandNetModel(nn.Module):
    """Simple parametrized filterbank feature extractor (bandpass filters + power extraction +
    log nonlinearity + optional MLP output head).
    """

    def __init__(
        self,
        sfreq: float,
        config: FreqBandNet,
        n_outputs: int | None = None,
    ):
        super().__init__()

        self.sinc_conv = SincConv(
            1,
            config.n_filters_conv,
            kernel_size=config.conv_kernel_len,
            stride=config.conv_stride,
            sfreq=sfreq,
            padding=config.conv_padding,
            freq_lims_hz=config.freq_lims_hz,
        )
        self.pool = nn.AvgPool2d(
            kernel_size=(1, config.pool_kernel_len),
        )
        self.flat_out = config.flat_out
        self.classifier = None if n_outputs is None else nn.LazyLinear(n_outputs)

    def forward(
        self,
        x: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Apply filterbank, compute log-power, and optionally classify.

        Parameters
        ----------
        x : Tensor
            Input time series of shape ``(B, C, T)``.
        eps : float
            Small constant for numerical stability in ``log``.
        """
        B, C, T = x.shape
        x = x.reshape(B, 1, C, T)

        x = self.sinc_conv(x)  # Apply filterbank
        x = self.pool(x**2)  # Compute average power -> (B, F, C, T')
        x = torch.log(x + eps)  # Get log-power

        if self.flat_out is not None:
            B, F, C, Tp = x.shape
            x = (
                x.reshape(B, F * C, Tp)
                if self.flat_out == "channels"
                else x.reshape(B, -1)
            )

        if self.classifier is not None:
            out = self.classifier(x)
        else:
            out = x

        return out


def sinc_from_half(x_left: torch.Tensor) -> torch.Tensor:
    """Return sinc values given the left half of the indices."""
    out_left = torch.sin(x_left) / x_left
    out = torch.cat(
        [
            out_left,
            torch.ones((out_left.shape[0], 1), device=x_left.device),
            torch.flip(out_left, dims=[-1]),
        ],
        dim=1,
    )
    return out


class SincConv(nn.Module):
    """Parametrized windowed Sinc-based convolution.

    See https://arxiv.org/pdf/1808.00158

    XXX Use only half the window?
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None,
        kernel_size: int,
        stride: int,
        sfreq: float,
        padding: tp.Literal["valid", "same"] = "valid",
        min_low_hz: float = 0.1,
        max_high_hz: float = 50.0,
        freq_lims_hz: list[float] | None = None,
    ) -> None:
        super().__init__()

        if in_channels != 1:
            raise NotImplementedError()
        if kernel_size % 2 != 1:
            raise ValueError(f"{kernel_size=} must be odd")
        if not ((freq_lims_hz is None) ^ (out_channels is None)):
            raise ValueError(
                f"Exactly one of {freq_lims_hz=} and {out_channels=} must be specified."
            )

        self.stride = stride
        self.sfreq = sfreq
        self.padding = padding

        # Filter cutoff parameters
        if out_channels is not None:
            freq_lims = (
                torch.logspace(
                    torch.tensor(min_low_hz).log10(),
                    torch.tensor(max_high_hz).log10(),
                    steps=out_channels + 1,
                    base=10.0,
                )
                / sfreq
            )
        elif freq_lims_hz is not None:
            freq_lims = torch.tensor(freq_lims_hz) / sfreq

        if freq_lims[-1] > 0.5:
            cutoff = freq_lims[-1] * sfreq
            raise ValueError(f"Cutoff {cutoff:.2f} Hz exceeds Nyquist {sfreq / 2} Hz")

        self.low_freqs = nn.Parameter(freq_lims[:-1], requires_grad=True)
        self.bandwidths = nn.Parameter(freq_lims[1:] - freq_lims[:-1], requires_grad=True)

        # Define Hamming window
        n_lin = torch.linspace(
            torch.tensor(0.0),
            torch.tensor(kernel_size) - 1,
            steps=kernel_size,
        )
        _window = 0.54 - 0.46 * torch.cos(2 * torch.pi * n_lin / kernel_size)
        self.register_buffer("_window", _window)

        # Prepare sinc function arguments
        n = (kernel_size - 1) / 2.0
        _n = 2 * torch.pi * torch.arange(-n, 0).view(1, -1)
        self.register_buffer("_n", _n)

    def forward(
        self, x: torch.Tensor, return_filters: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x :
            Input time series of shape (B, 1, C, T).
        return_filters :
            If True, also return the windowed sinc filters.
        """

        # Compute filters
        low_freqs = torch.abs(self.low_freqs)[:, None]
        high_freqs = low_freqs + torch.abs(self.bandwidths)[:, None]

        g_low = 2 * low_freqs * sinc_from_half(low_freqs @ self._n)  # type: ignore
        g_high = 2 * high_freqs * sinc_from_half(high_freqs @ self._n)  # type: ignore
        filters = (g_high - g_low) * self._window  # type: ignore

        # Apply filters
        out = conv2d(
            x,
            filters[:, None, None, :],
            stride=(1, self.stride),
            padding=self.padding,
        )

        if return_filters:
            return out, filters
        return out
