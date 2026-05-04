# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Neuraltrain custom configuration for LaBraM.

Includes the following adaptations:

* Channel name remapping via an explicit user-provided mapping.
* Dynamic channel resolution at forward time using ``channel_positions`` to
  detect which channels are valid per sample.
* Temporal-embedding adaptation so pretrained weights can be reused with a
  different ``n_times`` (truncation for fewer patches, linear interpolation
  for more patches than the pretrained model).
"""

import logging
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBrainDecodeModel
from .common import (
    INVALID_POS_VALUE,
    apply_temporal_adjustment,
    compute_temporal_adjustment,
    parse_bipolar_name,
)

logger = logging.getLogger(__name__)


class _LabramChannelWrapper(nn.Module):
    """Wraps a braindecode ``Labram`` to resolve channel names at forward time.

    Channel handling is precomputed in union-channel order:

    * ``_labram_names`` -- the LaBraM-cased name to forward to the inner
      model for each union channel (channels with no LaBraM mapping fall
      back to their original name and are dropped by braindecode's
      case-insensitive ``ch_names`` matching).
    * ``_positionless_mask`` -- a boolean buffer marking channels that have
      a valid LaBraM mapping but no montage position (bipolar derivations
      resolved via anode fallback, or channels added via an explicit
      ``channel_mapping``).  These are kept regardless of their per-sample
      ``channel_positions`` row, so e.g. SleepEDF's bipolar ``Fpz-Cz`` or
      Geodesic E-numbers reach LaBraM even when ``set_montage`` could not
      assign them coordinates.

    At forward time the per-sample mask ``(channel_positions != sentinel)``
    is OR-combined with ``_positionless_mask`` and intersected across the
    batch (LaBraM's ``ch_names`` is per-batch, so heterogeneous batches
    fall back to the channels valid in every sample).

    Parameters
    ----------
    model : nn.Module
        The braindecode ``Labram`` model instance.
    union_ch_names : list of str
        Ordered channel names from the dataset union (matching the channel
        dimension of the input tensor).
    ch_name_to_labram : dict mapping str to str
        Mapping from dataset channel names to LaBraM channel names, as
        returned by :meth:`NtLabram._build_channel_remapping`.  Channels not
        in this dict are forwarded with their original name and dropped by
        braindecode's case-insensitive ``ch_names`` matching.
    positionless_ch_names : set of str, optional
        Subset of ``ch_name_to_labram`` keys whose channels are not expected
        to carry montage positions.  Defaults to an empty set.
    pad_right, truncate_right : int
        Time-axis padding/truncation applied just before the inner model
        (see :func:`apply_temporal_adjustment`).
    """

    def __init__(
        self,
        model: nn.Module,
        union_ch_names: list[str],
        ch_name_to_labram: dict[str, str],
        positionless_ch_names: set[str] | None = None,
        pad_right: int = 0,
        truncate_right: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self._pad_right = pad_right
        self._truncate_right = truncate_right

        positionless = set(positionless_ch_names) if positionless_ch_names else set()
        self._labram_names: list[str] = [
            ch_name_to_labram.get(name, name) for name in union_ch_names
        ]
        self.register_buffer(
            "_positionless_mask",
            torch.tensor(
                [name in positionless for name in union_ch_names],
                dtype=torch.bool,
            ),
        )

    def forward(self, x: torch.Tensor, channel_positions: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic channel selection.

        Parameters
        ----------
        x : (B, n_channels, n_times)
        channel_positions : (B, n_channels, n_spatial_dims)
        """
        valid = (channel_positions != INVALID_POS_VALUE).any(dim=-1)
        valid = valid | self._positionless_mask  # type: ignore[operator]
        # Intersect across the batch: braindecode's ``ch_names`` is per-batch.
        common = valid.all(dim=0)

        indices = common.nonzero(as_tuple=True)[0].tolist()
        ch_names = [self._labram_names[i] for i in indices]

        x_valid = x[:, common, :]
        x_valid = apply_temporal_adjustment(
            x_valid, self._pad_right, self._truncate_right
        )

        return self.model(x_valid, ch_names=ch_names, return_all_tokens=True)


class NtLabram(BaseBrainDecodeModel):
    """Config for the braindecode LaBraM model with pretrained-model support.

    Extends :class:`BaseBrainDecodeModel` with LaBraM-specific logic:

    1. **Channel remapping** -- an explicit ``channel_mapping`` dict maps
       dataset channel names to LaBraM channel names.  Channels whose names
       already match ``LABRAM_CHANNEL_ORDER`` (case-insensitively) need no
       entry.
    2. **Dynamic channel resolution** -- the model is wrapped in
       :class:`_LabramChannelWrapper` so that valid channels are detected from
       ``channel_positions`` at each forward call, then remapped and passed to
       the inner braindecode model via ``ch_names``.

    Parameters
    ----------
    channel_mapping : dict or None
        Explicit mapping from dataset channel names to LaBraM channel names.
        Useful for EEG systems with known correspondences (e.g. Geodesic
        E-number to 10-10).
    """

    _MODEL_CLASS_PATH: tp.ClassVar[str] = "braindecode.models.Labram"
    chs_info_required: tp.ClassVar[bool] = True
    needs_n_times: tp.ClassVar[bool] = True
    channel_mapping: dict[str, str] | None = None

    def _build_channel_remapping(
        self,
        union_ch_names: list[str],
    ) -> tuple[dict[str, str], set[str]]:
        """Build a mapping from dataset channel names to LaBraM channel names.

        Mapping priority (per channel):

        1. ``self.channel_mapping`` (explicit user override)
        2. Direct case-insensitive name match against
           ``LABRAM_CHANNEL_ORDER`` -- the dataset name maps to the
           canonical LABRAM-cased version.
        3. Bipolar fallback -- for names like ``"Fp1-F3"``, try matching the
           anode (``"Fp1"``) against ``LABRAM_CHANNEL_ORDER``.

        Returns
        -------
        remap : dict
            Mapping from every matched union channel name to its LaBraM
            counterpart (always in ``LABRAM_CHANNEL_ORDER`` casing for cases
            2 and 3; whatever the user supplied for case 1).  Channels that
            cannot be mapped are **not** included (they will be passed
            through as-is and handled by braindecode's ``on_unknown_chs``
            policy).
        positionless : set
            Subset of ``remap`` keys for channels that are not expected to
            carry montage positions -- i.e. those resolved via the bipolar
            anode fallback (case 3) or explicit user ``channel_mapping``
            (case 1).  Regular case-insensitive matches (case 2) are excluded
            because such channels do have known positions and should be gated
            by the per-sample position validity check at forward time.
        """
        from braindecode.models.labram import LABRAM_CHANNEL_ORDER

        labram_upper = {ch.upper(): ch for ch in LABRAM_CHANNEL_ORDER}

        result: dict[str, str] = {}
        positionless: set[str] = set()
        n_bipolar_fallback = 0
        for name in union_ch_names:
            if self.channel_mapping and name in self.channel_mapping:
                result[name] = self.channel_mapping[name]
                positionless.add(name)
                continue
            canonical = labram_upper.get(name.upper())
            if canonical is not None:
                result[name] = canonical
                continue
            pair = parse_bipolar_name(name)
            if pair is not None:
                anode_canonical = labram_upper.get(pair[0].upper())
                if anode_canonical is not None:
                    result[name] = anode_canonical
                    positionless.add(name)
                    n_bipolar_fallback += 1

        if n_bipolar_fallback:
            logger.info(
                "Mapped %d bipolar channel(s) to LaBraM via anode fallback.",
                n_bipolar_fallback,
            )

        return result, positionless

    @staticmethod
    def _adapt_pretrained_dimensions(model: nn.Module, n_times: int) -> tuple[int, int]:
        """Adapt a pretrained LaBraM model to a different ``n_times`` in-place.

        When the input ``n_times`` is not directly compatible with the model's
        ``patch_size``, the method computes the effective number of time
        samples to use (``effective_n_times``) and returns padding/truncation
        hints so that the wrapper can adjust inputs at forward time:

        * If ``n_times < patch_size``, the input will be zero-padded to
          ``patch_size`` (1 patch).
        * If ``n_times`` is not divisible by ``patch_size``, the input is
          truncated to the largest multiple of ``patch_size`` that fits.
        * If the resulting number of patches exceeds the pretrained model's
          capacity, the temporal embedding is interpolated (linearly).
        * If fewer patches are needed, the temporal embedding is truncated.

        Returns ``(pad_right, truncate_right)`` -- exactly one is non-zero.
        """
        # Adaptation reaches into private braindecode internals; guard the
        # attributes we touch so a future braindecode rename surfaces here
        # rather than as a confusing forward-time error.
        for attr in ("patch_size", "n_times", "patch_embed"):
            if not hasattr(model, attr):
                raise AttributeError(
                    f"NtLabram._adapt_pretrained_dimensions: braindecode "
                    f"Labram has no attribute {attr!r}.  Has braindecode's "
                    f"internal layout changed?"
                )
        patch_embed_first = model.patch_embed[0]  # type: ignore[index]
        if not hasattr(patch_embed_first, "n_patchs"):
            raise AttributeError(
                "NtLabram._adapt_pretrained_dimensions: "
                "model.patch_embed[0] has no attribute 'n_patchs'.  Has "
                "braindecode's internal layout changed?"
            )

        patch_size: int = model.patch_size  # type: ignore[assignment]

        pad_right, truncate_right = compute_temporal_adjustment(n_times, patch_size)
        effective_n_times = n_times + pad_right - truncate_right

        if pad_right:
            logger.info(
                "n_times=%d < patch_size=%d; will zero-pad %d samples on "
                "the right at forward time.",
                n_times,
                patch_size,
                pad_right,
            )
        elif truncate_right:
            logger.info(
                "n_times=%d is not divisible by patch_size=%d; will "
                "truncate %d samples on the right at forward time.",
                n_times,
                patch_size,
                truncate_right,
            )

        actual_n_patches = effective_n_times // patch_size
        pretrained_n_patches: int = patch_embed_first.n_patchs  # type: ignore[assignment]

        if actual_n_patches == pretrained_n_patches:
            return pad_right, truncate_right

        logger.info(
            "Adapting pretrained LaBraM from n_times=%d (%d patches) to "
            "n_times=%d (%d patches)",
            model.n_times,
            pretrained_n_patches,
            effective_n_times,
            actual_n_patches,
        )

        if hasattr(model, "temporal_embedding"):
            if actual_n_patches < pretrained_n_patches:
                n_keep = actual_n_patches + 1
                model.temporal_embedding = nn.Parameter(  # type: ignore[index]
                    model.temporal_embedding.data[:, :n_keep, :]  # type: ignore[index]
                )
            else:
                old_emb: torch.Tensor = model.temporal_embedding.data  # type: ignore[index,assignment]
                cls_token = old_emb[:, :1, :]
                patch_tokens = old_emb[:, 1:, :]
                patch_tokens = patch_tokens.permute(0, 2, 1)
                patch_tokens = F.interpolate(
                    patch_tokens,
                    size=actual_n_patches,
                    mode="linear",
                    align_corners=False,
                )
                patch_tokens = patch_tokens.permute(0, 2, 1)
                model.temporal_embedding = nn.Parameter(
                    torch.cat([cls_token, patch_tokens], dim=1)
                )

        model._n_times = effective_n_times  # type: ignore[assignment]
        model.n_path = actual_n_patches  # type: ignore[assignment]
        patch_embed_first.n_patchs = actual_n_patches  # type: ignore[assignment]
        patch_embed_first.n_times = effective_n_times  # type: ignore[assignment,union-attr]

        return pad_right, truncate_right

    def build(
        self,
        n_chans: int | None = None,
        n_times: int | None = None,
        n_outputs: int | None = None,
        chs_info: list[dict[str, tp.Any]] | None = None,
        **kwargs: tp.Any,
    ) -> nn.Module:
        # Precompute channel remapping
        ch_name_to_labram: dict[str, str] = {}
        positionless_ch_names: set[str] = set()
        union_ch_names: list[str] = []
        if chs_info is not None:
            union_ch_names = [ch["ch_name"] for ch in chs_info]
            ch_name_to_labram, positionless_ch_names = self._build_channel_remapping(
                union_ch_names
            )

        pad_right = 0
        truncate_right = 0

        if self.from_pretrained_name is not None:
            model = super().build(**kwargs)
            if n_times is not None:
                pad_right, truncate_right = self._adapt_pretrained_dimensions(
                    model, n_times
                )
        else:
            if n_chans is not None:
                kwargs["n_chans"] = n_chans
            if n_times is not None:
                kwargs["n_times"] = n_times
            if n_outputs is not None:
                kwargs["n_outputs"] = n_outputs
            model = super().build(**kwargs)

        if chs_info is not None:
            model = _LabramChannelWrapper(
                model,
                union_ch_names,
                ch_name_to_labram,
                positionless_ch_names=positionless_ch_names,
                pad_right=pad_right,
                truncate_right=truncate_right,
            )

        return model
