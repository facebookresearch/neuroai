# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import pydantic
import torch
from torch import nn


class ChannelsDropoutConfig(pydantic.BaseModel):
    """Configuration for braindecode's ``ChannelsDropout`` augmentation.

    Parameters
    ----------
    probability : float
        Probability of applying the augmentation to a given example.
    p_drop : float
        Fraction of channels to drop.
    """

    probability: float
    p_drop: float
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")

    def build(self) -> nn.Module:
        from braindecode.augmentation import ChannelsDropout

        return ChannelsDropout(
            probability=self.probability,
            p_drop=self.p_drop,
        )


class FrequencyShiftConfig(pydantic.BaseModel):
    """Configuration for braindecode's ``FrequencyShift`` augmentation.

    Parameters
    ----------
    probability : float
        Probability of applying the augmentation to a given example.
    sfreq : float
        Sampling frequency of the recording, in Hz.
    max_delta_freq : float
        Maximum frequency shift (in Hz) applied to the signal.
    """

    probability: float
    sfreq: float
    max_delta_freq: float
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")

    def build(self) -> nn.Module:
        from braindecode.augmentation import FrequencyShift

        return FrequencyShift(
            probability=self.probability,
            sfreq=self.sfreq,
            max_delta_freq=self.max_delta_freq,
        )


class GaussianNoiseConfig(pydantic.BaseModel):
    """Configuration for braindecode's ``GaussianNoise`` augmentation.

    Parameters
    ----------
    probability : float
        Probability of applying the augmentation to a given example.
    std : float
        Standard deviation of the additive Gaussian noise.
    """

    probability: float
    std: float
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")

    def build(self) -> nn.Module:
        from braindecode.augmentation import GaussianNoise

        return GaussianNoise(
            probability=self.probability,
            std=self.std,
        )


class SmoothTimeMaskConfig(pydantic.BaseModel):
    """Configuration for braindecode's ``SmoothTimeMask`` augmentation.

    Parameters
    ----------
    probability : float
        Probability of applying the augmentation to a given example.
    mask_len_samples : int
        Length (in samples) of the time block to mask.
    """

    probability: float
    mask_len_samples: int
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")

    def build(self) -> nn.Module:
        from braindecode.augmentation import SmoothTimeMask

        return SmoothTimeMask(
            probability=self.probability,
            mask_len_samples=self.mask_len_samples,
        )


class BandstopFilterFFTConfig(pydantic.BaseModel):
    """Configuration for :class:`BandstopFilterFFT`.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the recording, in Hz.
    bandwidth : float
        Bandwidth (in Hz) of the bandstop filter.  Must be less than half the
        sampling frequency.
    """

    sfreq: float
    bandwidth: float
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")

    def build(self) -> nn.Module:
        return BandstopFilterFFT(
            sfreq=self.sfreq,
            bandwidth=self.bandwidth,
        )


class BandstopFilterFFT(nn.Module):
    """
    Bandstop data augmentation, applying a bandstop filter to the data using Fourier transform.

    Parameters
    ----------
    sfreq: Sampling frequency of the recording
    bandwidth: Bandwidth of the bandstop filter
    """

    def __init__(
        self,
        sfreq: float,
        bandwidth: float,
    ):
        super().__init__()
        if bandwidth * 2 > sfreq:
            raise ValueError(
                "Bandwidth needs to be smaller than half of sampling frequency."
            )
        self.sfreq = sfreq
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor):
        ffted = torch.fft.rfft(
            x,
        )
        n_bins = int(np.round(self.bandwidth * 2 * ffted.shape[-1] / self.sfreq))
        i_bins = torch.randint(ffted.shape[-1] - n_bins, (len(ffted),))
        for i_example, i_bin in enumerate(i_bins):
            ffted[i_example, :, i_bin : i_bin + n_bins] = 0
        iffted = torch.fft.irfft(ffted)
        return iffted


class TrivialBrainAugmentConfig(pydantic.BaseModel):
    sfreq: float
    min_max_ch_drop: tuple[float, float] = (0.05, 0.4)
    min_max_gauss_noise: tuple[float, float] = (0.01, 0.3)
    min_max_time_mask: tuple[float, float] = (2, 32)
    min_max_bandstop: tuple[float, float] = (1, 8)
    min_max_freq_shift: tuple[float, float] = (-1, 1)
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")

    def build(self) -> nn.Module:
        return TrivialBrainAugment(self)


class TrivialBrainAugment(nn.Module):
    """
    Inspired by TrivialAugment [1], sample augmentations and strength randomly on each minibatch/forward pass.

    Parameters
    ----------
    config: Configuration that contains values for:
        sfreq: Sampling frequency of the recording
        min_max_ch_drop: Min/Max for linspace of channel dropout probabilities
        min_max_gauss_noise: Min/Max for linspace of gaussian noise standard deviation
        min_max_time_mask: Min/Max for logspace of length of timeblock to be masked
        min_max_bandstop: Min/Max  for logspace of frequency width of bandstop filter
        min_max_freq_shift: Min/Max for linspace of frequency shift

    References
    ----------
    .. [1] Mueller, Samuel and Hutter, Frank. "TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation"
    """

    def __init__(self, cfg: TrivialBrainAugmentConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from braindecode.augmentation import (
            ChannelsDropout,
            FrequencyShift,
            GaussianNoise,
            SmoothTimeMask,
        )

        num_strengths = 32
        strength = int(torch.randint(0, num_strengths, (1,)).item())

        transforms = [
            ChannelsDropout(
                1,
                p_drop=np.linspace(
                    self.cfg.min_max_ch_drop[0],
                    self.cfg.min_max_ch_drop[1],
                    num=num_strengths,
                )[strength],
            ),
            GaussianNoise(
                1,
                std=np.linspace(
                    self.cfg.min_max_gauss_noise[0],
                    self.cfg.min_max_gauss_noise[1],
                    num=num_strengths,
                )[strength],
            ),
            SmoothTimeMask(
                1,
                mask_len_samples=int(
                    np.logspace(
                        np.log2(self.cfg.min_max_time_mask[0]),
                        np.log2(self.cfg.min_max_time_mask[1]),
                        base=2,
                        num=num_strengths,
                    ).round()[strength]
                ),
            ),
            BandstopFilterFFT(
                sfreq=self.cfg.sfreq,
                bandwidth=np.logspace(
                    np.log2(self.cfg.min_max_bandstop[0]),
                    np.log2(self.cfg.min_max_bandstop[1]),
                    base=2,
                    num=num_strengths,
                )[strength],
            ),
            FrequencyShift(
                1,
                sfreq=self.cfg.sfreq,
                max_delta_freq=np.linspace(
                    self.cfg.min_max_freq_shift[0],
                    self.cfg.min_max_freq_shift[1],
                    num=num_strengths,
                )[strength],
            ),
        ]

        i_transform = int(torch.randint(0, len(transforms), (1,)).item())
        transform = transforms[i_transform]

        return transform(x)
