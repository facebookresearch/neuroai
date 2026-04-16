# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Neuraltrain custom configuration for BENDR.

Includes the following adaptations over the raw braindecode BENDR model:

* **Channel name remapping** -- an explicit ``channel_mapping`` dict maps
  dataset channel names to the 19 standard 10/20 channel names that BENDR
  expects (matching the original DN3 ``deep1010`` mapping).  Channels whose
  names already match (case-insensitively) need no entry.  Missing channels
  are zero-filled; surplus channels are discarded.
* **Relative amplitude channel** -- the 20th input channel reproduces the
  per-window relative amplitude signal from the BENDR paper: a constant
  value ``(max(window) - min(window)) / R_ds`` broadcast over the time
  dimension, where ``R_ds`` is an approximate global amplitude range.
* **Integrated MinMaxScaler** -- per-window min-max normalisation to
  ``[-1, 1]`` is applied inside the wrapper (before the amplitude channel
  is appended), so that the ``OnTheFlyPreprocessor`` does **not** need a
  separate scaler.
* **All-tokens output** -- the wrapper returns the full contextualiser
  token sequence ``(B, T+1, D)`` instead of just the CLS token, letting
  the ``DownstreamWrapper`` handle aggregation consistently with other
  foundation models.
"""

import logging
import typing as tp

import torch
from torch import nn

from .base import BaseBrainDecodeModel
from .common import parse_bipolar_name

logger = logging.getLogger(__name__)

# 19 standard 10/20 EEG channels in the order BENDR's encoder expects,
# matching the DN3 deep1010 reduced channel set used for pretraining.
BENDR_CHANNEL_ORDER: list[str] = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]

# Approximate global amplitude range (max - min over the entire pretraining
# dataset) used as the denominator for the relative-amplitude 20th channel.
# Derived from the BENDR pretraining config for TUEG:
#   data_max=3276.7, data_min=-1583.93  =>  R_ds ~ 4860.6
# The exact value is not critical; it acts as a normalising constant.
_DEFAULT_AMPLITUDE_RANGE: float = 4861.0

_BENDR_UPPER_TO_IDX: dict[str, int] = {
    ch.upper(): i for i, ch in enumerate(BENDR_CHANNEL_ORDER)
}


def _build_channel_indices(
    union_ch_names: list[str],
    channel_mapping: dict[str, str] | None,
) -> list[tuple[int, int]]:
    """Map dataset channel names to BENDR channel-order indices.

    Returns a list of ``(dataset_idx, bendr_idx)`` pairs for every dataset
    channel that could be resolved.  Channels not in the mapping or not
    matching any of the 19 standard names are silently skipped.
    """
    pairs: list[tuple[int, int]] = []
    n_mapped = 0
    n_bipolar = 0
    for ds_idx, name in enumerate(union_ch_names):
        resolved = channel_mapping.get(name, name) if channel_mapping else name
        upper = resolved.upper()
        if upper in _BENDR_UPPER_TO_IDX:
            pairs.append((ds_idx, _BENDR_UPPER_TO_IDX[upper]))
            if channel_mapping and name in channel_mapping:
                n_mapped += 1
            continue
        bp = parse_bipolar_name(resolved)
        if bp is not None and bp[0].upper() in _BENDR_UPPER_TO_IDX:
            pairs.append((ds_idx, _BENDR_UPPER_TO_IDX[bp[0].upper()]))
            n_bipolar += 1

    if n_mapped:
        logger.info("Mapped %d channel(s) via explicit channel_mapping.", n_mapped)
    if n_bipolar:
        logger.info("Mapped %d bipolar channel(s) via anode fallback.", n_bipolar)

    matched = {p[1] for p in pairs}
    missing = [
        BENDR_CHANNEL_ORDER[i]
        for i in range(len(BENDR_CHANNEL_ORDER))
        if i not in matched
    ]
    if missing:
        logger.warning(
            "BENDR: %d of 19 standard channels missing (will be zero-filled): %s",
            len(missing),
            missing,
        )
    logger.info(
        "[BENDR_CHANNELS] n_dataset=%d  n_resolved=%d/19",
        len(union_ch_names),
        len(matched),
    )
    return pairs


class _BendrAllTokensWrapper(nn.Module):
    """Wraps a braindecode ``BENDR`` to add channel mapping, MinMaxScaler,
    relative amplitude channel, and all-tokens output.

    Parameters
    ----------
    model : nn.Module
        A braindecode ``BENDR`` model instance (with ``final_layer=False``).
    channel_pairs : list of (int, int) or None
        ``(dataset_channel_idx, bendr_channel_idx)`` pairs produced by
        :func:`_build_channel_indices`.  When ``None``, the input is
        forwarded as-is (assumed to already have 20 channels).
    amplitude_range : float
        Approximate global amplitude range ``R_ds`` for the relative
        amplitude 20th channel.
    """

    encoder: nn.Module
    contextualizer: nn.Module
    _ds_indices: torch.Tensor | None
    _bendr_indices: torch.Tensor | None

    def __init__(
        self,
        model: nn.Module,
        channel_pairs: list[tuple[int, int]] | None = None,
        amplitude_range: float = _DEFAULT_AMPLITUDE_RANGE,
    ) -> None:
        super().__init__()
        self.encoder = model.encoder  # type: ignore[assignment]
        self.contextualizer = model.contextualizer  # type: ignore[assignment]
        self.amplitude_range = amplitude_range

        if channel_pairs is not None:
            ds_indices = torch.tensor([p[0] for p in channel_pairs], dtype=torch.long)
            bendr_indices = torch.tensor([p[1] for p in channel_pairs], dtype=torch.long)
            self.register_buffer("_ds_indices", ds_indices)
            self.register_buffer("_bendr_indices", bendr_indices)
        else:
            self.register_buffer("_ds_indices", None)
            self.register_buffer("_bendr_indices", None)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Reorder channels, apply MinMaxScaler, append amplitude channel.

        Input ``x`` has shape ``(B, n_dataset_chans, T)``.
        Output has shape ``(B, 20, T)``.
        """
        B, _, T = x.shape

        # Compute per-window range before normalisation
        x_min, x_max = torch.aminmax(x, dim=-1, keepdim=True)  # over time
        x_min_global = x_min.min(dim=1, keepdim=True).values  # over channels
        x_max_global = x_max.max(dim=1, keepdim=True).values
        window_range = (x_max_global - x_min_global).squeeze(-1).squeeze(-1)  # (B,)

        # MinMaxScaler: per-channel normalisation to [-1, 1]
        denom = x_max - x_min
        denom[denom == 0.0] = 1.0
        x_scaled = 2.0 * (x - x_min) / denom - 1.0

        if self._ds_indices is not None:
            # Reorder into 19 standard BENDR channels (zero-fill missing)
            eeg = x_scaled.new_zeros(B, 19, T)
            eeg[:, self._bendr_indices] = x_scaled[:, self._ds_indices]
        else:
            eeg = x_scaled

        # 20th channel: relative amplitude = window_range / R_ds
        rel_amp = (window_range / self.amplitude_range).unsqueeze(-1).unsqueeze(-1)
        amp_channel = rel_amp.expand(B, 1, T)

        return torch.cat([eeg, amp_channel], dim=1)  # (B, 20, T)

    def forward(self, x: torch.Tensor, **kwargs: tp.Any) -> torch.Tensor:
        if self._ds_indices is not None:
            x = self._prepare_input(x)
        encoded = self.encoder(x)  # (B, encoder_h, T_enc)
        context = self.contextualizer(encoded)  # (B, encoder_h, T_enc + 1)
        return context.permute(0, 2, 1)  # (B, T_enc + 1, encoder_h)


class NtBendr(BaseBrainDecodeModel):
    """Config for the braindecode BENDR model with channel mapping support.

    Extends :class:`BaseBrainDecodeModel` with BENDR-specific logic:

    1. **Channel remapping** -- an explicit ``channel_mapping`` dict maps
       dataset channel names to standard 10/20 names.  Channels whose names
       already match (case-insensitively) need no entry.
    2. **Relative amplitude channel** -- the 20th input channel is computed
       at forward time from the per-window amplitude range, matching the
       original BENDR paper.
    3. **All-tokens output** -- the model is wrapped to return the full
       contextualiser sequence ``(B, T+1, encoder_h)`` instead of just the
       CLS token.

    Parameters
    ----------
    channel_mapping : dict or None
        Explicit mapping from dataset channel names to standard 10/20 names
        (e.g. ``{"EEG 042": "Cz"}``).  Channels that already match a name
        in :data:`BENDR_CHANNEL_ORDER` (case-insensitively) need no entry.
    amplitude_range : float
        Approximate global amplitude range used as the denominator of the
        relative amplitude 20th channel.  Defaults to the value derived from
        the TUEG pretraining dataset (~4861).
    """

    _MODEL_CLASS: tp.ClassVar[tp.Any] = None  # resolved lazily
    chs_info_required: tp.ClassVar[bool] = True
    channel_mapping: dict[str, str] | None = None
    amplitude_range: float = _DEFAULT_AMPLITUDE_RANGE

    @classmethod
    def _ensure_model_class(cls) -> None:
        """Resolve ``_MODEL_CLASS`` on first use (lazy for optional dep)."""
        if cls._MODEL_CLASS is None:
            import braindecode.models

            resolved = getattr(braindecode.models, "BENDR", None)
            if resolved is None:
                raise ImportError(
                    "braindecode.models.BENDR is not available. "
                    "Upgrade braindecode or install missing dependencies."
                )
            cls._MODEL_CLASS = resolved

    def model_post_init(self, __context__: tp.Any) -> None:
        type(self)._ensure_model_class()
        super().model_post_init(__context__)

    def build(
        self,
        n_chans: int | None = None,
        n_times: int | None = None,
        n_outputs: int | None = None,
        chs_info: list[dict[str, tp.Any]] | None = None,
        **kwargs: tp.Any,
    ) -> nn.Module:
        type(self)._ensure_model_class()

        channel_pairs: list[tuple[int, int]] | None = None
        if chs_info is not None:
            union_ch_names = [ch["ch_name"] for ch in chs_info]
            channel_pairs = _build_channel_indices(union_ch_names, self.channel_mapping)

        effective_n_chans = 20 if channel_pairs is not None else n_chans

        if self.from_pretrained_name is not None:
            model = super().build(**kwargs)
        else:
            if effective_n_chans is not None:
                kwargs["n_chans"] = effective_n_chans
            if n_times is not None:
                kwargs["n_times"] = n_times
            if n_outputs is not None:
                kwargs["n_outputs"] = n_outputs
            model = super().build(**kwargs)

        return _BendrAllTokensWrapper(
            model,
            channel_pairs=channel_pairs,
            amplitude_range=self.amplitude_range,
        )
