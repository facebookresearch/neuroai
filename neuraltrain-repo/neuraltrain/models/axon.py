# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Neuraltrain custom configuration for AXON.

AXON (``braindecode.models.AXON``, checkpoint ``MannasAI/axon-eeg``) identifies
electrodes only by their 3D scalp position, which is fixed at build time from
``chs_info[i]["loc"]``.  NeuralBench passes channel *names*, so this wrapper adds:

* **Name-based electrode placement** -- each name is looked up in MNE's
  ``standard_1005`` montage in head coordinates (the frame used by the reference
  evaluation), extended with the EGI ``GSN-HydroCel-129`` names (``E1`` ...
  ``E128``, e.g. HBN).  The lookup uses braindecode's
  :func:`~braindecode.modules.channels.resolve_channel_positions`: it ignores
  case, strips an ``EEG`` prefix or ``-REF`` suffix, maps legacy aliases
  (``T3`` -> ``T7``) and places a bipolar ``A-B`` channel at the midpoint of its
  two electrodes.  An optional ``channel_mapping`` renames channels first.
* **Channel subsetting** -- channels that cannot be placed are dropped, and the
  input is sliced to the kept channels.
* **Encoder-only output** -- when ``n_outputs`` is ``None`` (a
  ``DownstreamWrapperModel`` provides the head), the model returns AXON's token
  grid ``(batch, n_chans, n_patches, embed_dim)`` from ``AXON.encode``.
  NeuralBench's ``aggregation: mean`` over that grid is AXON's own pooling.

AXON requires ``braindecode >= 2.0``.
"""

import functools
import logging
import typing as tp

import numpy as np
import torch
import torch.nn as nn

from .base import BaseBrainDecodeModel, RequiredBuildField

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _position_bank() -> dict[str, np.ndarray]:
    """Lower-case channel name -> head-frame position in metres.

    MNE ``standard_1005`` (the montage braindecode falls back to for AXON),
    plus the EGI ``GSN-HydroCel-129`` names it lacks.  Both montages are moved
    to the head frame, as ``raw.set_montage`` would do.
    """
    import mne
    from braindecode.util import resolve_montage_name

    bank: dict[str, np.ndarray] = {}
    # resolve_montage_name: "standard_1005" is "colin27_1005" from MNE 1.13 on.
    for montage_name in (resolve_montage_name("standard_1005"), "GSN-HydroCel-129"):
        montage = mne.channels.make_standard_montage(montage_name)
        montage.apply_trans(mne.channels.compute_native_head_t(montage))
        for name, pos in montage.get_positions()["ch_pos"].items():
            bank.setdefault(name.lower(), np.asarray(pos, dtype=float))
    return bank


class _AxonWrapper(nn.Module):
    """Thin wrapper around AXON for channel subsetting and encoder-only output.

    * **Channel subsetting** -- when *channel_indices* is not ``None``, the input
      is sliced along the channel dimension before AXON sees it.
    * **Encoder-only output** -- when *encoder_only* is ``True``, ``forward``
      returns ``AXON.encode(eeg)`` (``(B, C, P, D)`` tokens) and AXON's
      ``final_layer`` is replaced by :class:`~torch.nn.Identity`, so the unused,
      randomly initialised head holds no parameters.
    """

    channel_indices: torch.Tensor | None

    def __init__(
        self,
        model: nn.Module,
        channel_indices: list[int] | None = None,
        encoder_only: bool = False,
    ):
        super().__init__()
        self.model = model
        self.encoder_only = encoder_only
        if encoder_only:
            self.model.final_layer = nn.Identity()
        if channel_indices is not None:
            self.register_buffer(
                "channel_indices", torch.tensor(channel_indices, dtype=torch.long)
            )
        else:
            self.register_buffer("channel_indices", None)

    def forward(self, eeg: torch.Tensor, *args: tp.Any, **kwargs: tp.Any) -> torch.Tensor:
        if self.channel_indices is not None:
            eeg = eeg[:, self.channel_indices]
        if self.encoder_only:
            return self.model.encode(eeg)  # type: ignore[operator]
        return self.model(eeg)


class NtAxon(BaseBrainDecodeModel):
    """Config for the braindecode AXON model with name-based electrode placement.

    Parameters
    ----------
    channel_mapping : dict or None
        Explicit renames from dataset channel names to ``standard_1005`` names,
        applied before the position lookup.  Useful for systems whose names are
        not 10-05 labels (e.g. Neuromag ``"EEG 005"``).  Channels whose names
        already resolve need no entry.
    """

    _MODEL_CLASS_PATH: tp.ClassVar[str] = "braindecode.models.AXON"
    required_fields: tp.ClassVar[list[RequiredBuildField]] = ["ch_names", "n_times"]
    channel_mapping: dict[str, str] | None = None

    def _remap_chs_info(
        self, chs_info: list[dict[str, tp.Any]]
    ) -> list[dict[str, tp.Any]]:
        """Apply ``channel_mapping`` to *chs_info*, returning a new list."""
        if not self.channel_mapping:
            return chs_info
        return [
            {**ch, "ch_name": self.channel_mapping.get(ch["ch_name"], ch["ch_name"])}
            for ch in chs_info
        ]

    def build(
        self,
        n_spatial_locations: int,
        n_temporal_samples: int,
        n_outputs: int | None = None,
        chs_info: list[dict[str, tp.Any]] | None = None,
        frequency: float | None = None,
    ) -> nn.Module:
        from braindecode.modules.channels import resolve_channel_positions

        if chs_info is None:
            raise ValueError("NtAxon needs channel names to place the electrodes.")
        chs_info = self._remap_chs_info(chs_info)

        no_match = ValueError(
            "No dataset channel matches a standard_1005 or GSN-HydroCel-129 "
            "position. Consider adding a `channel_mapping`."
        )
        try:
            positions, kept = resolve_channel_positions(
                [{"ch_name": ch["ch_name"]} for ch in chs_info],
                bank=_position_bank(),
                on_missing="drop",
            )
        except ValueError as err:  # braindecode raises when nothing resolves
            raise no_match from err
        if not kept:
            raise no_match
        kept_set = set(kept)
        dropped = [ch["ch_name"] for i, ch in enumerate(chs_info) if i not in kept_set]
        if dropped:
            logger.warning(
                "Dropping %d channel(s) without a known AXON position: %s",
                len(dropped),
                dropped,
            )
        logger.info(
            "[AXON_CHANNELS] n_dataset=%d n_resolved=%d", len(chs_info), len(kept)
        )

        # AXON reads positions from chs_info[i]["loc"][:3] (MNE layout: 12 values).
        axon_chs_info = [
            {
                "ch_name": chs_info[i]["ch_name"],
                "kind": "eeg",
                "loc": np.r_[np.asarray(pos, dtype=float), np.full(9, np.nan)],
            }
            for i, pos in zip(kept, positions)
        ]
        encoder_only = n_outputs is None
        model = self._construct(
            n_chans=len(kept),
            n_times=n_temporal_samples,
            chs_info=axon_chs_info,
            # The pretrained checkpoint has no head; a placeholder size is needed
            # to build one, and encoder-only mode discards it.
            n_outputs=n_outputs if n_outputs is not None else 2,
        )
        channel_indices = None if len(kept) == len(chs_info) else list(kept)
        return _AxonWrapper(model, channel_indices, encoder_only)
