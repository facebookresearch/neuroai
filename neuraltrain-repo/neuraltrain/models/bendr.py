# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Neuraltrain wrapper for BENDR (Kostas et al., Front. Hum. Neurosci. 2021)."""

import logging
import typing as tp

import torch
from torch import nn

from .base import BaseBrainDecodeModel

logger = logging.getLogger(__name__)


_DEFAULT_AMPLITUDE_RANGE: float = 4861.0  # TUEG data_max - data_min, pretraining config

BENDR_N_EEG_CHANNELS: int = 19  # excluding the relative-amplitude channel


class _BendrAllTokensWrapper(nn.Module):
    """Wraps a braindecode ``BENDR`` to add MinMax preprocessing,
    the relative-amplitude 20th channel, and an all-tokens output.

    The input is assumed to already be an EEG tensor with
    ``BENDR_N_EEG_CHANNELS`` (19) channels (produced by an upstream
    ``channel_adapter``).  The wrapper:

    1. Computes a global ``(min, max)`` per window across channels+time.
    2. Scales the 19 EEG channels to ``[-1, 1]``.
    3. Appends a 20th amplitude channel ``2 * (clamp(window_range / R, 1.0) - 0.5)``.
    4. Runs the BENDR encoder + contextualizer.
    5. Returns the full contextualiser token sequence ``(B, T+1, D)``.

    Parameters
    ----------
    model : nn.Module
        A braindecode ``BENDR`` model instance (with ``final_layer=False``).
    amplitude_range : float
        Approximate global amplitude range ``R_ds`` for the relative
        amplitude 20th channel.
    """

    encoder: nn.Module
    contextualizer: nn.Module

    def __init__(
        self,
        model: nn.Module,
        amplitude_range: float = _DEFAULT_AMPLITUDE_RANGE,
    ) -> None:
        super().__init__()
        self.encoder = model.encoder  # type: ignore[assignment]
        self.contextualizer = model.contextualizer  # type: ignore[assignment]
        self.amplitude_range = amplitude_range

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global min-max scaling and append the amplitude channel.

        Input ``x`` has shape ``(B, 19, T)``; output has shape ``(B, 20, T)``.
        """
        B, _, T = x.shape

        x_flat = x.reshape(B, -1)
        x_min = x_flat.amin(dim=1, keepdim=True)  # (B, 1)
        x_max = x_flat.amax(dim=1, keepdim=True)  # (B, 1)
        window_range = (x_max - x_min).squeeze(-1)  # (B,)
        denom = (x_max - x_min).clamp_min(1e-12).unsqueeze(-1)  # (B, 1, 1)
        x_scaled = 2.0 * (x - x_min.unsqueeze(-1)) / denom - 1.0

        rel_amp = (window_range / self.amplitude_range).clamp(max=1.0)
        amp_val = 2.0 * (rel_amp - 0.5)
        amp_channel = amp_val.view(B, 1, 1).expand(B, 1, T)

        return torch.cat([x_scaled, amp_channel], dim=1)  # (B, 20, T)

    def forward(self, x: torch.Tensor, **kwargs: tp.Any) -> torch.Tensor:
        x = self._prepare_input(x)
        encoded = self.encoder(x)  # (B, encoder_h, T_enc)
        context = self.contextualizer(encoded)  # (B, encoder_h, T_enc + 1)
        return context.permute(0, 2, 1)  # (B, T_enc + 1, encoder_h)


class NtBendr(BaseBrainDecodeModel):
    """Config for the braindecode BENDR model.

    Extends :class:`BaseBrainDecodeModel` with:

    1. **Preprocessing** -- each window is scaled to ``[-1, 1]`` with a single
       ``(min, max)`` across channels and time (keeping the inter-channel
       amplitude ratios the encoder relies on), and the relative-amplitude
       channel is appended, as in DN3's ``MappingDeep1010``.
    2. **Encoder-only output** -- returns all contextualiser tokens
       ``(B, T+1, encoder_h)``, start token included.

    Channels are not mapped here: the downstream wrapper's
    ``channel_adapter_config`` must project to :data:`BENDR_N_EEG_CHANNELS`.

    Parameters
    ----------
    amplitude_range : float
        Denominator of the relative-amplitude channel.
    """

    _MODEL_CLASS_PATH: tp.ClassVar[str] = "braindecode.models.BENDR"
    amplitude_range: float = _DEFAULT_AMPLITUDE_RANGE

    def build(
        self,
        n_spatial_locations: int,
        n_temporal_samples: int,
        n_outputs: int | None = None,
        chs_info: list[dict[str, tp.Any]] | None = None,
        frequency: float | None = None,
    ) -> nn.Module:
        construct_kwargs = self._bd_shape_kwargs(
            n_spatial_locations=n_spatial_locations,
            n_temporal_samples=n_temporal_samples,
            n_outputs=n_outputs,
            chs_info=chs_info,
            frequency=frequency,
        )
        construct_kwargs["n_chans"] = BENDR_N_EEG_CHANNELS + 1  # + amplitude channel

        model = self._construct(**construct_kwargs)
        return _BendrAllTokensWrapper(model, amplitude_range=self.amplitude_range)
