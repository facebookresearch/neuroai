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

    At each forward call the wrapper inspects the per-sample
    ``channel_positions`` tensor (produced by the ``ChannelPositions``
    extractor) to determine which channels are valid (i.e. not marked with the
    sentinel ``CHANNEL_POSITIONS_INVALID_VALUE``).  The corresponding channel
    names are looked up from the stored union channel list, remapped to LaBraM
    channel names using a precomputed dictionary, and finally passed to the
    inner model via its ``ch_names`` argument.

    Parameters
    ----------
    model : nn.Module
        The braindecode ``Labram`` model instance.
    union_ch_names : list of str
        Ordered channel names from the dataset union (matching the channel
        dimension of the input tensor).
    ch_name_to_labram : dict mapping str to str
        Precomputed mapping from dataset channel names to LaBraM channel
        names.  Channels that already match (case-insensitively) may map to
        themselves.
    pad_right : int
        Number of zero-padding samples to add on the right.
    truncate_right : int
        Number of samples to truncate from the right.
    """

    def __init__(
        self,
        model: nn.Module,
        union_ch_names: list[str],
        ch_name_to_labram: dict[str, str],
        pad_right: int = 0,
        truncate_right: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self._union_ch_names = union_ch_names
        self._ch_name_to_labram = ch_name_to_labram
        self._pad_right = pad_right
        self._truncate_right = truncate_right

    def forward(self, x: torch.Tensor, channel_positions: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic channel selection.

        Parameters
        ----------
        x : (B, n_channels, n_times)
        channel_positions : (B, n_channels, n_spatial_dims)
        """
        valid_mask = (channel_positions != INVALID_POS_VALUE).any(
            dim=-1
        )  # (B, n_channels)

        # Use the intersection of valid channels across the batch so that
        # recordings with slightly different channel sets can coexist.
        common_mask = valid_mask.all(dim=0)  # (n_channels,)

        valid_indices = common_mask.nonzero(as_tuple=True)[0].tolist()
        valid_ch_names = [self._union_ch_names[i] for i in valid_indices]
        remapped_names = [
            self._ch_name_to_labram.get(name, name) for name in valid_ch_names
        ]

        x_valid = x[:, common_mask, :]
        x_valid = apply_temporal_adjustment(
            x_valid, self._pad_right, self._truncate_right
        )

        return self.model(x_valid, ch_names=remapped_names, return_all_tokens=True)


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

    _MODEL_CLASS: tp.ClassVar[tp.Any] = None  # resolved lazily; see _ensure_model_class
    chs_info_required: tp.ClassVar[bool] = True
    needs_n_times: tp.ClassVar[bool] = True
    channel_mapping: dict[str, str] | None = None

    @classmethod
    def _ensure_model_class(cls) -> None:
        """Resolve ``_MODEL_CLASS`` on first use.

        Must also be called in ``build()`` because ``model_post_init`` is not
        invoked after submitit deserialization on SLURM workers.
        """
        if cls._MODEL_CLASS is None:
            import braindecode.models

            cls._MODEL_CLASS = braindecode.models.Labram

    def model_post_init(self, __context__: tp.Any) -> None:
        type(self)._ensure_model_class()
        super().model_post_init(__context__)

    def _build_channel_remapping(
        self,
        union_ch_names: list[str],
    ) -> dict[str, str]:
        """Build a mapping from dataset channel names to LaBraM channel names.

        Mapping priority (per channel):

        1. ``self.channel_mapping`` (explicit user override)
        2. Direct case-insensitive name match against
           ``LABRAM_CHANNEL_ORDER`` -- the name is kept as-is (braindecode
           handles the case-insensitive lookup internally)
        3. Bipolar fallback -- for names like ``"Fp1-F3"``, try matching the
           anode (``"Fp1"``) against ``LABRAM_CHANNEL_ORDER``

        Returns a dict mapping every matched union channel name to its LaBraM
        counterpart.  Channels that cannot be mapped are **not** included in
        the returned dict (they will be passed through as-is and handled by
        braindecode's ``on_unknown_chs`` policy).
        """
        from braindecode.models.labram import LABRAM_CHANNEL_ORDER

        labram_upper = {ch.upper(): ch for ch in LABRAM_CHANNEL_ORDER}

        result: dict[str, str] = {}
        n_bipolar_fallback = 0
        for name in union_ch_names:
            if self.channel_mapping and name in self.channel_mapping:
                result[name] = self.channel_mapping[name]
            elif name.upper() in labram_upper:
                result[name] = name
            else:
                pair = parse_bipolar_name(name)
                if pair is not None and pair[0].upper() in labram_upper:
                    result[name] = labram_upper[pair[0].upper()]
                    n_bipolar_fallback += 1

        if n_bipolar_fallback:
            logger.info(
                "Mapped %d bipolar channel(s) to LaBraM via anode fallback.",
                n_bipolar_fallback,
            )

        return result

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
        pretrained_n_patches: int = model.patch_embed[0].n_patchs  # type: ignore[index,union-attr]

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
        model.patch_embed[0].n_patchs = actual_n_patches  # type: ignore[index,union-attr]
        model.patch_embed[0].n_times = effective_n_times  # type: ignore[index,union-attr]

        return pad_right, truncate_right

    def build(
        self,
        n_chans: int | None = None,
        n_times: int | None = None,
        n_outputs: int | None = None,
        chs_info: list[dict[str, tp.Any]] | None = None,
        **kwargs: tp.Any,
    ) -> nn.Module:
        type(self)._ensure_model_class()

        # Precompute channel remapping
        ch_name_to_labram: dict[str, str] = {}
        union_ch_names: list[str] = []
        if chs_info is not None:
            union_ch_names = [ch["ch_name"] for ch in chs_info]
            ch_name_to_labram = self._build_channel_remapping(union_ch_names)

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
                pad_right=pad_right,
                truncate_right=truncate_right,
            )

        return model
