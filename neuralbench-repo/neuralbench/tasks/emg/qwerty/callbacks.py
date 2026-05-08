# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Train-time augmentation callbacks for the emg2qwerty CTC task (paper Sec 5.2).

* :class:`SpecAugmentCallback` — frequency / time masking on the
  log-spectrogram via a forward hook on ``pl_module.model.spectrogram``.
* :class:`BandRotationCallback` — per-band electrode roll + temporal
  jitter on ``batch.data["neuro"]`` in ``on_train_batch_start``.

Both are no-ops when ``pl_module.training`` is False.
"""

from __future__ import annotations

import typing as tp

import numpy as np
import torch
from lightning.pytorch import Callback


class SpecAugmentCallback(Callback):
    """SpecAugment masking on the log-spectrogram during training.

    Up to ``n_time_masks`` × ``time_mask_param``-frame time bands and
    ``n_freq_masks`` × ``freq_mask_param``-bin frequency bands, applied
    IID per ``(sample × band)``.  Skipped until
    ``trainer.current_epoch >= start_epoch``.  Paper defaults match
    Sivakumar et al. Sec 5.2 (3×25 / 2×4 frames-bins, prob=1.0).
    """

    def __init__(
        self,
        n_time_masks: int = 3, time_mask_param: int = 25,
        n_freq_masks: int = 2, freq_mask_param: int = 4,
        prob: float = 1.0, start_epoch: int = 0,
    ):
        super().__init__()
        self.n_time_masks, self.time_mask_param = n_time_masks, time_mask_param
        self.n_freq_masks, self.freq_mask_param = n_freq_masks, freq_mask_param
        self.prob, self.start_epoch = prob, start_epoch
        self._handle: tp.Any = None
        self._time_mask: tp.Any = None
        self._freq_mask: tp.Any = None
        self._enabled = True

    def on_train_start(self, trainer, pl_module) -> None:
        import torchaudio.transforms as ta

        # iid_masks=True so every (batch × band) gets its own mask; without
        # this, the same time-window is masked across all bands and the CTC
        # head collapses to all-blank during warmup.
        self._time_mask = ta.TimeMasking(self.time_mask_param, iid_masks=True)
        self._freq_mask = ta.FrequencyMasking(self.freq_mask_param, iid_masks=True)

        spectro = getattr(pl_module.model, "spectrogram", None)
        if spectro is None:
            raise RuntimeError(
                "SpecAugmentCallback expects the model to expose a "
                "``spectrogram`` submodule (e.g. EMG2QwertyNet)."
            )

        def _hook(module, _inputs, output):
            if not (module.training and self._enabled):
                return output
            if self.prob < 1.0 and float(np.random.rand()) >= self.prob:
                return output
            # ``output``: (T_spec, B, num_bands, electrodes, freq). Reshape
            # to (B*num_bands, electrodes, freq, T_spec) so iid_masks draws
            # one independent mask per (sample × band), shared across the
            # 16 electrodes within a band — per-electrode masking is 32×
            # more aggressive and collapses the model.
            T_spec, B, n_bands, n_elec, n_freq = output.shape
            flat = output.movedim(0, -1).reshape(B * n_bands, n_elec, n_freq, T_spec)
            # Mask to per-window mean: in log-spec space 0 = log(power=1),
            # well above typical distribution → artificial spikes.
            mv = float(flat.mean().item())
            for _ in range(int(np.random.randint(self.n_time_masks + 1))):
                flat = self._time_mask(flat, mask_value=mv)
            for _ in range(int(np.random.randint(self.n_freq_masks + 1))):
                flat = self._freq_mask(flat, mask_value=mv)
            return flat.reshape(B, n_bands, n_elec, n_freq, T_spec).movedim(-1, 0)

        self._handle = spectro.register_forward_hook(_hook)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._enabled = trainer.current_epoch >= self.start_epoch

    def on_train_end(self, trainer, pl_module) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class BandRotationCallback(Callback):
    """Per-band electrode rotation + inter-band temporal jitter (paper Sec 5.2).

    Thin Lightning-callback adapter around
    :func:`braindecode.augmentation.functional.band_rotation` — the
    per-batch math lives in braindecode so it's reusable outside the
    Lightning training loop.  The callback injects the augmented tensor
    back into ``batch.data["neuro"]`` during training.

    ``start_epoch`` defers augmentation until ``trainer.current_epoch >=
    start_epoch`` — useful for fine-tuning runs where applying band
    rotation from epoch 0 corrupts pretrained features before the
    optimizer has time to adapt.
    """

    def __init__(
        self,
        num_bands: int = 2, electrodes_per_band: int = 16,
        band_offsets: tuple[int, ...] = (-1, 0, 1),
        max_temporal_jitter: int = 120,  # 60 ms @ 2 kHz; paper value
        start_epoch: int = 0,
    ):
        super().__init__()
        self.num_bands, self.electrodes_per_band = num_bands, electrodes_per_band
        self.band_offsets = tuple(band_offsets)
        self.max_temporal_jitter = max_temporal_jitter
        self.start_epoch = start_epoch

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if not pl_module.training:
            return
        if trainer is not None and trainer.current_epoch < self.start_epoch:
            return
        x = batch.data.get("neuro")
        # x: (B, C, T) with C == num_bands * electrodes_per_band; otherwise
        # silently skip (model uses a different channel layout).
        if x is None or x.shape[1] != self.num_bands * self.electrodes_per_band:
            return

        from braindecode.augmentation.functional import band_rotation

        x_aug, _ = band_rotation(
            x, torch.zeros(x.shape[0], device=x.device),
            num_bands=self.num_bands,
            electrodes_per_band=self.electrodes_per_band,
            band_offsets=self.band_offsets,
            max_temporal_jitter=self.max_temporal_jitter,
        )
        batch.data["neuro"] = x_aug
