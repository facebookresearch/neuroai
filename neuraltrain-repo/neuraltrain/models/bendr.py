# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Neuraltrain custom configuration for BENDR.

Includes the following adaptations over the raw braindecode BENDR model:

* **Global per-window min-max scaling** to ``[-1, 1]`` (single ``(min, max)``
  pair across channels and time), matching the original BENDR pretraining
  recipe (DN3 ``MappingDeep1010``).  Per-channel scaling would destroy
  inter-channel amplitude relationships, which BENDR's channel-unmixed 1D
  convolutional feature encoder relies on.
* **Relative amplitude 20th channel** ``2 * (clamp(window_range / R, 1.0) - 0.5)``
  producing values in ``[-1, 1]``, reproducing the DN3 ``MappingDeep1010``
  20th channel from the BENDR paper.
* **All-tokens output** -- the wrapper returns the full contextualiser token
  sequence ``(B, T+1, D)`` instead of just the CLS token, letting the
  :class:`DownstreamWrapper` handle aggregation (e.g. ``aggregation: "mean"``)
  consistently with other foundation models (CBraMod, LaBraM, LUNA, REVE).

Channel remapping is **not** handled here: the downstream wrapper's
``channel_adapter_config`` (a learned ``Conv1d(kernel_size=1)`` projection,
see :class:`neuralbench.modules.ChannelProjection`) is expected to project from
arbitrary input channels to the 19 channels BENDR consumes.  This mirrors
BIOT's handling and avoids the information loss of zero-filling missing
standard 10/20 channels.
"""

import logging
import typing as tp

import torch
from torch import nn

from .base import BaseBrainDecodeModel

logger = logging.getLogger(__name__)


# Approximate global amplitude range (max - min over the entire pretraining
# dataset) used as the denominator for the relative-amplitude 20th channel.
# Derived from the BENDR pretraining config for TUEG:
#   data_max=3276.7, data_min=-1583.93  =>  R_ds ~ 4860.6
# The exact value is not critical; it acts as a normalising constant.
_DEFAULT_AMPLITUDE_RANGE: float = 4861.0

# Number of EEG channels the pretrained BENDR encoder expects (before the
# internal relative-amplitude channel is appended).
BENDR_N_EEG_CHANNELS: int = 19


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

    Extends :class:`BaseBrainDecodeModel` with BENDR-specific logic:

    1. **Preprocessing** -- global per-window min-max to ``[-1, 1]`` with a
       20th relative-amplitude channel, matching the BENDR paper.
    2. **All-tokens output** -- the model is wrapped to return the full
       contextualiser sequence ``(B, T+1, encoder_h)`` so the downstream
       wrapper can pool it (e.g. ``aggregation: "mean"``) uniformly with
       the other FMs.

    Channel mapping is delegated to the downstream wrapper's
    ``channel_adapter_config`` (a 1x1 Conv1d projection), which must project
    to :data:`BENDR_N_EEG_CHANNELS` (19) channels.

    Parameters
    ----------
    amplitude_range : float
        Approximate global amplitude range used as the denominator of the
        relative amplitude 20th channel.  Defaults to the value derived from
        the TUEG pretraining dataset (~4861).
    """

    amplitude_range: float = _DEFAULT_AMPLITUDE_RANGE

    @classmethod
    def _ensure_model_class(cls) -> None:
        """Resolve ``_MODEL_CLASS`` on first use with a friendlier error.

        BENDR is an optional braindecode dependency; the default base-class
        path-based loader would raise ``AttributeError`` if it is missing.
        """
        if cls._MODEL_CLASS is not None:
            return
        import braindecode.models

        resolved = getattr(braindecode.models, "BENDR", None)
        if resolved is None:
            raise ImportError(
                "braindecode.models.BENDR is not available. "
                "Upgrade braindecode or install missing dependencies."
            )
        cls._MODEL_CLASS = resolved

    def build(
        self,
        n_chans: int | None = None,
        n_times: int | None = None,
        n_outputs: int | None = None,
        **kwargs: tp.Any,
    ) -> nn.Module:
        # The wrapper prepends an amplitude channel internally, so the
        # underlying BENDR encoder always receives 20 channels.  The upstream
        # channel adapter is responsible for producing the 19 EEG channels;
        # here we hard-set n_chans=20 for the inner BENDR regardless of
        # whatever was passed in (the adapter target + our amp channel
        # combine into 20).
        kwargs["n_chans"] = BENDR_N_EEG_CHANNELS + 1
        if n_times is not None:
            kwargs["n_times"] = n_times
        if n_outputs is not None:
            kwargs["n_outputs"] = n_outputs

        model = super().build(**kwargs)
        return _BendrAllTokensWrapper(model, amplitude_range=self.amplitude_range)
