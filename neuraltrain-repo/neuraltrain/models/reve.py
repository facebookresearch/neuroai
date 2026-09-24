# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Neuraltrain custom configuration for REVE.

Includes the following adaptations over the raw braindecode REVE model:

* **Channel name remapping** -- an explicit ``channel_mapping`` dict maps
  dataset channel names to names recognised by REVE's position bank.
  Channels whose names already appear in the bank need no entry.
* **Per-batch channel positions** -- the wrapper takes the dataset's
  ``channel_positions`` as REVE's ``pos`` for any montage the build could not
  resolve from the position bank, so one instance serves montages it was not
  built for.
* **Pretrained loading** -- REVE's ``__init__`` requires ``n_times`` and
  ``n_outputs`` to build its ``final_layer``, but
  :class:`BaseBrainDecodeModel.build` blocks ``n_times`` for pretrained
  models.  ``NtReve.build()`` calls ``from_pretrained`` directly.
* **Encoder-only support** -- when ``n_outputs`` is not provided (i.e. when
  a ``DownstreamWrapperModel`` handles the classification head), the wrapper
  calls ``forward(return_output=True)`` and extracts the final transformer
  layer's output, bypassing REVE's ``final_layer`` entirely.
"""

import logging
import typing as tp

import torch
import torch.nn as nn

from .base import BaseBrainDecodeModel, RequiredBuildField
from .common import INVALID_POS_VALUE, parse_bipolar_name

logger = logging.getLogger(__name__)


class _ReveWrapper(nn.Module):
    """Thin wrapper around REVE for channel handling and encoder-only output.

    Combines three adaptations in a single module:

    * **Channel subsetting** -- when *channel_indices* is not ``None``, the EEG
      tensor and the channel positions are sliced along the channel dimension
      before forwarding to REVE.  No-op when ``None``.
    * **Channel positions** -- when the build named every channel,
      *bank_positions* wins: those are the coordinates REVE was pretrained
      with, and using them keeps a result independent of how the montage was
      digitised.  Otherwise the batch's ``channel_positions`` become REVE's
      ``pos``, and channels the montage could not place (they arrive as
      ``INVALID_POS_VALUE``) are dropped.  A bank only lines up with the
      montage it was built from, so an instance that must span montages of the
      same width should be built without ``chs_info``.
    * **Encoder-only output** -- when *encoder_only* is ``True``, the
      forward call passes ``return_output=True`` to REVE and returns the
      final transformer layer output (index ``-1``), bypassing REVE's
      ``final_layer``.  When ``False``, forwards normally.
    """

    channel_indices: torch.Tensor | None
    bank_positions: torch.Tensor | None

    def __init__(
        self,
        model: nn.Module,
        channel_indices: list[int] | None = None,
        encoder_only: bool = False,
        bank_positions: torch.Tensor | None = None,
    ):
        super().__init__()
        self.model = model
        self.encoder_only = encoder_only
        self.register_buffer("bank_positions", bank_positions)
        if channel_indices is not None:
            self.register_buffer(
                "channel_indices",
                torch.tensor(channel_indices, dtype=torch.long),
            )
        else:
            self.register_buffer("channel_indices", None)

    def forward(
        self,
        eeg: torch.Tensor,
        channel_positions: torch.Tensor | None = None,
        **kwargs: tp.Any,
    ) -> torch.Tensor:
        if self.channel_indices is not None:
            eeg = eeg[:, self.channel_indices]
            if channel_positions is not None:
                channel_positions = channel_positions[:, self.channel_indices]
        bank = self.bank_positions
        if bank is not None and eeg.shape[1] == bank.shape[0]:
            channel_positions = bank.to(eeg).expand(eeg.shape[0], -1, -1)
        elif channel_positions is not None:
            unplaced = (channel_positions == INVALID_POS_VALUE).all(dim=-1)
            keep = ~unplaced.any(dim=0)
            eeg, channel_positions = eeg[:, keep], channel_positions[:, keep]
        if self.encoder_only:
            return self.model(eeg, pos=channel_positions, return_output=True, **kwargs)[
                -1
            ]
        return self.model(eeg, pos=channel_positions, **kwargs)


class NtReve(BaseBrainDecodeModel):
    """Config for the braindecode REVE model with channel-mapping support.

    Extends :class:`BaseBrainDecodeModel` with REVE-specific logic:

    1. **Channel remapping** -- an explicit ``channel_mapping`` dict maps
       dataset channel names to REVE position-bank names.  Channels whose
       names already appear in the bank (exact match) need no entry.
    2. **Pretrained loading** -- bypasses the base-class restriction on
       ``n_times`` for pretrained models, since REVE needs it to size its
       ``final_layer``.
    3. **Encoder-only output** -- when ``n_outputs`` is ``None`` (downstream
       wrapper handles the head), the model is wrapped to call
       ``forward(return_output=True)`` and return the final transformer
       layer output, bypassing REVE's ``final_layer``.

    Parameters
    ----------
    channel_mapping : dict or None
        Explicit mapping from dataset channel names to REVE position-bank
        names.  Useful for EEG systems whose naming convention is absent
        from the bank (e.g. Neuromag ``"EEG 005"`` or easycap-M10 numeric
        ``"2"``).
    """

    _MODEL_CLASS_PATH: tp.ClassVar[str] = "braindecode.models.REVE"
    required_fields: tp.ClassVar[list[RequiredBuildField]] = ["ch_names", "n_times"]
    channel_mapping: dict[str, str] | None = None

    def _remap_chs_info(
        self,
        chs_info: list[dict[str, tp.Any]],
    ) -> list[dict[str, tp.Any]]:
        """Apply ``channel_mapping`` to *chs_info*, returning a new list."""
        if not self.channel_mapping:
            return chs_info
        return [
            {**ch, "ch_name": self.channel_mapping.get(ch["ch_name"], ch["ch_name"])}
            for ch in chs_info
        ]

    @staticmethod
    def _derive_bipolar_position(
        name: str,
        bank: tp.Any,
    ) -> torch.Tensor | None:
        """Look up the anode position for a bipolar channel name.

        Returns ``None`` when *name* is not a valid bipolar pair or the anode
        electrode is missing from *bank*.
        """
        pair = parse_bipolar_name(name)
        if pair is None:
            return None
        anode = pair[0]
        if anode not in bank.mapping:
            return None
        return bank.embedding[bank.mapping[anode]]

    def build(
        self,
        n_spatial_locations: int,
        n_temporal_samples: int,
        n_outputs: int | None = None,
        chs_info: list[dict[str, tp.Any]] | None = None,
        frequency: float | None = None,
    ) -> nn.Module:
        if chs_info is not None:
            chs_info = self._remap_chs_info(chs_info)

        # Spatial size the model receives (already the adapter target when a
        # channel adapter is present).  Reduced below if some channels cannot
        # be resolved.
        n_chans = n_spatial_locations

        channel_indices: list[int] | None = None
        bank_positions: torch.Tensor | None = None
        n_derived = 0

        if chs_info is not None:
            from braindecode.models.reve import RevePositionBank

            bank = RevePositionBank()
            n_original = len(chs_info)

            valid_indices: list[int] = []
            valid_chs: list[dict[str, tp.Any]] = []
            positions: list[torch.Tensor] = []
            dropped: list[str] = []

            for i, ch in enumerate(chs_info):
                name = ch["ch_name"]
                if name in bank.mapping:
                    valid_indices.append(i)
                    valid_chs.append(ch)
                    positions.append(bank.embedding[bank.mapping[name]])
                else:
                    derived = self._derive_bipolar_position(name, bank)
                    if derived is not None:
                        valid_indices.append(i)
                        valid_chs.append(ch)
                        positions.append(derived)
                        n_derived += 1
                    else:
                        dropped.append(name)

            if dropped:
                logger.warning(
                    "Dropping %d channel(s) not resolvable from REVE position bank: %s",
                    len(dropped),
                    dropped,
                )
            if n_derived:
                logger.info(
                    "Mapped %d bipolar channel(s) to REVE via anode fallback.",
                    n_derived,
                )

            if len(valid_chs) < n_original:
                channel_indices = valid_indices
                chs_info = valid_chs
                n_chans = len(chs_info)

            if not valid_chs:
                raise ValueError(
                    "No dataset channels match the REVE position bank "
                    "(directly or via anode fallback). "
                    "Consider adding a `channel_mapping`."
                )

            logger.info(
                "[REVE_CHANNELS] n_dataset=%d n_resolved=%d (n_derived=%d)",
                n_original,
                len(valid_chs),
                n_derived,
            )

            bank_positions = torch.stack(positions)

        build_kwargs: dict[str, tp.Any] = {
            "n_chans": n_chans,
            "n_times": n_temporal_samples,
        }
        if chs_info is not None and n_derived == 0:
            build_kwargs["chs_info"] = chs_info

        encoder_only = n_outputs is None

        if self.from_pretrained_name is not None:
            build_kwargs["n_outputs"] = n_outputs if n_outputs is not None else 2
        elif n_outputs is not None:
            build_kwargs["n_outputs"] = n_outputs

        model = self._construct(**build_kwargs)

        if bank_positions is not None and n_derived > 0:
            model.default_pos = bank_positions

        return _ReveWrapper(model, channel_indices, encoder_only, bank_positions)
