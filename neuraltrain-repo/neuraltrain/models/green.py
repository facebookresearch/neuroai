# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from torch import nn

from .base import BaseModelConfig


class Green(BaseModelConfig):
    """
    Reference: GREEN: A lightweight architecture using learnable
    wavelets and Riemannian geometry for biomarker exploration with EEG signals.
    See https://arxiv.org/pdf/1808.00158.

    Parameters
    ----------
    sfreq: int | None, default=None
        Sampling frequency of the signal.
    n_freqs: int, default=15
        Number of main frequencies in the wavelet family.
    kernel_width_s: int, default=5
        Width of the kernel in seconds for the wavelets.
    conv_stride: int, default=5
        Stride of the convolution operation for the wavelets.
    oct_min: float, default=0
        Minimum frequency of interest in octave.
    oct_max: float, default=5.5
        Maximum frequency of interest in octave.
    random_f_init: bool, default=False
        Whether to randomly initialize the frequency of interest.
    shrinkage_init: float, default=-3.0
        Initial shrinkage value before applying sigmoid function.
    logref: str, default='logeuclid'
        Reference matrix used for LogEig layer.
    dropout: float, default=0.333
        Dropout rate for FC layers.
    hidden_dim: list[int], default=[32]
        Dimension of the hidden layer. If None, no hidden layer.
    pool_layer: str, default='RealCovariance'
        Pooling layer type. Options: 'RealCovariance', 'PW_PLV', 'CombinedPooling',
        'CrossCovariance', 'CrossPW_PLV', 'WaveletConv'.
    bi_out: list[int] | None, default=None
        Dimension of the output layer after BiMap.
    use_age: bool, default=False
        Whether to include age in the model.
    orth_weights: bool, default=True
        Whether to use orthogonal weight initialization.
    """

    sfreq: int | None = None
    n_freqs: int = 15
    kernel_width_s: int = 5
    conv_stride: int = 5
    oct_min: float = 0
    oct_max: float = 5.5
    random_f_init: bool = False
    shrinkage_init: float = -3.0
    logref: str = "logeuclid"
    dropout: float = 0.333
    hidden_dim: list[int] = [32]
    pool_layer: tp.Literal[
        "RealCovariance",
        "PW_PLV",
        "CombinedPooling",
        "CrossCovariance",
        "CrossPW_PLV",
        "WaveletConv",
    ] = "RealCovariance"
    bi_out: list[int] | None = None
    use_age: bool = False
    orth_weights: bool = True

    def build(
        self, n_in_channels: int, n_outputs: int, sfreq: int | None = None
    ) -> nn.Module:
        import green.wavelet_layers as wl  # type: ignore
        from green.research_code.pl_utils import get_green  # type: ignore

        kwargs = self.model_dump()
        del kwargs["name"]
        sfreq = sfreq or self.sfreq
        if sfreq is None:
            raise ValueError("sfreq must be provided to build the model.")
        kwargs["sfreq"] = sfreq
        kwargs["pool_layer"] = getattr(wl, self.pool_layer)()
        kwargs["out_dim"] = n_outputs
        kwargs["n_ch"] = n_in_channels
        return get_green(**kwargs)
