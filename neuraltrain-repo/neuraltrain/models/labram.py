# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Neuraltrain custom configuration for LaBraM.

Includes the following adaptations:

* Channel name remapping via an explicit user-provided mapping.
* Dynamic channel resolution at forward time using ``channel_positions`` to
  detect which channels are valid per sample, and ``ch_names`` to name them
  when the caller's montage differs from the one built with.
* Dynamic temporal resolution at forward time, so one instance serves any
  window length: the pretrained temporal embedding is sliced per call
  (interpolated when a window needs more patches than pretraining had).
"""

import logging
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBrainDecodeModel, RequiredBuildField
from .common import (
    INVALID_POS_VALUE,
    apply_temporal_adjustment,
    compute_temporal_adjustment,
    parse_bipolar_name,
)

logger = logging.getLogger(__name__)


def _build_channel_remapping(
    ch_names: list[str],
    channel_mapping: dict[str, str] | None = None,
) -> tuple[dict[str, str], set[str]]:
    """Map dataset channel names to LaBraM channel names.

    Mapping priority (per channel):

    1. ``channel_mapping`` (explicit user override)
    2. Direct case-insensitive name match against ``LABRAM_CHANNEL_ORDER`` --
       the dataset name maps to the canonical LABRAM-cased version.
    3. Bipolar fallback -- for names like ``"Fp1-F3"``, try matching the anode
       (``"Fp1"``) against ``LABRAM_CHANNEL_ORDER``.

    Returns
    -------
    remap : dict
        Mapping from every matched channel name to its LaBraM counterpart
        (always in ``LABRAM_CHANNEL_ORDER`` casing for cases 2 and 3;
        whatever the user supplied for case 1).  Channels that cannot be
        mapped are **not** included (they are then dropped by
        :func:`_resolve_channels`).
    positionless : set
        Subset of ``remap`` keys for channels that are not expected to carry
        montage positions -- i.e. those resolved via the bipolar anode
        fallback (case 3) or explicit user ``channel_mapping`` (case 1).
        Regular case-insensitive matches (case 2) are excluded because such
        channels do have known positions and should be gated by the
        per-sample position validity check at forward time.
    """
    from braindecode.models.labram import LABRAM_CHANNEL_ORDER

    labram_upper = {ch.upper(): ch for ch in LABRAM_CHANNEL_ORDER}

    result: dict[str, str] = {}
    positionless: set[str] = set()
    n_bipolar_fallback = 0
    for name in ch_names:
        if channel_mapping and name in channel_mapping:
            result[name] = channel_mapping[name]
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


def _resolve_channels(
    ch_names: list[str],
    channel_mapping: dict[str, str] | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Derive what :class:`_LabramChannelWrapper` needs from *ch_names*.

    Returns, in the order of *ch_names*:

    * ``labram_names`` -- the LaBraM-cased name to forward to the inner model
      for each channel (channels with no LaBraM mapping keep their original
      name; see ``known`` below for how those are then filtered out).
    * ``positionless`` -- channels that have a valid LaBraM mapping but no
      montage position (bipolar derivations resolved via anode fallback, or
      channels added via an explicit ``channel_mapping``).  These are kept
      regardless of their per-sample ``channel_positions`` row, so e.g.
      SleepEDF's bipolar ``Fpz-Cz`` or Geodesic E-numbers reach LaBraM even
      when ``set_montage`` could not assign them coordinates.
    * ``known`` -- channels whose resolved name is in
      ``LABRAM_CHANNEL_ORDER`` (case-insensitively).  Channels that fail this
      check (e.g. unmapped EGI ``E5``, ``E7``, ...) are filtered out entirely
      so braindecode never sees them.  This matters because braindecode
      dropped its ``on_unknown_chs`` parameter in ``>=1.5`` and now
      hard-raises a ``ValueError`` on the first unknown name; doing the
      filtering here keeps the wrapper's behaviour ("warn-and-drop") stable
      across braindecode versions.
    """
    from braindecode.models.labram import LABRAM_CHANNEL_ORDER

    remap, positionless = _build_channel_remapping(ch_names, channel_mapping)
    labram_names = [remap.get(name, name) for name in ch_names]

    labram_upper = {ch.upper() for ch in LABRAM_CHANNEL_ORDER}
    known = [name.upper() in labram_upper for name in labram_names]
    if not all(known):
        unknown = sorted({ch_names[i] for i, ok in enumerate(known) if not ok})
        logger.warning(
            "%d channel(s) not in LABRAM_CHANNEL_ORDER will be dropped at "
            "forward time: %s",
            len(unknown),
            unknown,
        )

    return (
        labram_names,
        torch.tensor([name in positionless for name in ch_names], dtype=torch.bool),
        torch.tensor(known, dtype=torch.bool),
    )


class _LabramChannelWrapper(nn.Module):
    """Wraps a braindecode ``Labram`` to resolve its inputs at forward time.

    braindecode addresses electrodes by name (``ch_names`` picks rows of the
    channel embedding) and reads the patch count from build-time state, so a
    bare ``Labram`` serves exactly the montage and window length it was built
    for.  This wrapper resolves both per call instead.

    **Channels.**  Names and masks come from :func:`_resolve_channels`,
    memoized per name list so the common case -- every batch carrying the
    montage built with -- resolves once.  At forward time the per-sample
    position mask is OR-combined with the positionless mask, AND-combined
    with the known mask, then intersected across the batch (LaBraM's
    ``ch_names`` is per-batch, so heterogeneous batches fall back to the
    channels valid in every sample).

    **Window length.**  ``temporal_embedding`` moves onto the wrapper so it
    can be cut to the number of patches the input actually carries, while the
    input itself is padded up to one patch or truncated to a whole number of
    patches (:func:`compute_temporal_adjustment`).

    Parameters
    ----------
    model : nn.Module
        The braindecode ``Labram`` model instance.
    union_ch_names : list of str
        Ordered channel names from the dataset union, used for any forward
        call that does not name its own channels.
    channel_mapping : dict mapping str to str, optional
        Explicit mapping from dataset channel names to LaBraM channel names,
        passed to :func:`_build_channel_remapping`.
    """

    temporal_embedding: nn.Parameter | None

    def __init__(
        self,
        model: nn.Module,
        union_ch_names: list[str],
        channel_mapping: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        # Forward-time adaptation reaches into private braindecode internals;
        # guard the attributes we touch so a future braindecode rename
        # surfaces here rather than as a confusing forward-time error.
        for attr in ("patch_size", "patch_embed"):
            if not hasattr(model, attr):
                raise AttributeError(
                    f"_LabramChannelWrapper: braindecode Labram has no "
                    f"attribute {attr!r}.  Has braindecode's internal layout "
                    f"changed?"
                )
        if not hasattr(model.patch_embed[0], "n_patchs"):  # type: ignore[index]
            raise AttributeError(
                "_LabramChannelWrapper: model.patch_embed[0] has no attribute "
                "'n_patchs'.  Has braindecode's internal layout changed?"
            )

        self.model = model
        self.channel_mapping = channel_mapping
        self.patch_size: int = model.patch_size  # type: ignore[assignment]
        # Owned by the wrapper rather than by ``model`` so that a slice of it
        # can be handed back per call; braindecode reads it as a plain
        # attribute either way, and gradients still reach the full parameter.
        self.temporal_embedding = model._parameters.pop("temporal_embedding", None)

        self._union_ch_names = list(union_ch_names)
        self._resolved: dict[
            tuple[str, ...], tuple[list[str], torch.Tensor, torch.Tensor]
        ] = {}
        self._resolve(self._union_ch_names)

    def _resolve(
        self, ch_names: list[str]
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        key = tuple(ch_names)
        if key not in self._resolved:
            self._resolved[key] = _resolve_channels(ch_names, self.channel_mapping)
        return self._resolved[key]

    def _temporal_embedding(self, n_patches: int) -> torch.Tensor:
        """The pretrained temporal embedding, cut or stretched to *n_patches*.

        braindecode tiles ``temporal_embedding[:, :-1]`` across channels, so
        its length is what fixes the number of time tokens.
        """
        embedding = tp.cast(torch.Tensor, self.temporal_embedding)
        if n_patches + 1 <= embedding.shape[1]:
            return embedding[:, : n_patches + 1]
        patches = F.interpolate(
            embedding[:, 1:].permute(0, 2, 1),
            size=n_patches,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)
        return torch.cat([embedding[:, :1], patches], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: torch.Tensor,
        ch_names: list[str] | None = None,
    ) -> torch.Tensor:
        """Forward pass with dynamic channel and window-length selection.

        Parameters
        ----------
        x : (B, n_channels, n_times)
        channel_positions : (B, n_channels, n_spatial_dims)
        ch_names : list of str, optional
            Names of the channels of *x*, defaulting to the montage the
            wrapper was built with.  Pass it whenever the incoming montage
            differs -- LaBraM cannot name an electrode from its coordinates.
        """
        labram_names, positionless, known = self._resolve(
            ch_names if ch_names is not None else self._union_ch_names
        )
        if len(labram_names) != x.shape[1]:
            raise ValueError(
                f"Got {len(labram_names)} channel name(s) for {x.shape[1]} input "
                "channels. LaBraM selects its channel embedding by name, so a "
                "montage other than the one built with must be passed as "
                "'ch_names' at forward time."
            )

        valid = (channel_positions != INVALID_POS_VALUE).any(dim=-1)
        valid = valid | positionless.to(valid.device)
        valid = valid & known.to(valid.device)
        # Intersect across the batch: braindecode's ``ch_names`` is per-batch.
        common = valid.all(dim=0)
        names = [labram_names[i] for i in common.nonzero(as_tuple=True)[0].tolist()]

        pad_right, truncate_right = compute_temporal_adjustment(
            x.shape[2], self.patch_size
        )
        x_valid = apply_temporal_adjustment(x[:, common, :], pad_right, truncate_right)

        n_patches = x_valid.shape[2] // self.patch_size
        self.model.patch_embed[0].n_patchs = n_patches  # type: ignore[index,union-attr]
        if self.temporal_embedding is not None:
            self.model.temporal_embedding = self._temporal_embedding(n_patches)

        return self.model(x_valid, ch_names=names, return_all_tokens=True)


class NtLabram(BaseBrainDecodeModel):
    """Config for the braindecode LaBraM model with pretrained-model support.

    Extends :class:`BaseBrainDecodeModel` with LaBraM-specific logic:

    1. **Channel remapping** -- an explicit ``channel_mapping`` dict maps
       dataset channel names to LaBraM channel names.  Channels whose names
       already match ``LABRAM_CHANNEL_ORDER`` (case-insensitively) need no
       entry.
    2. **Forward-time adaptation** -- the model is wrapped in
       :class:`_LabramChannelWrapper`, which picks the valid channels and the
       number of time patches from each batch, so one instance serves any
       montage and window length.

    Parameters
    ----------
    channel_mapping : dict or None
        Explicit mapping from dataset channel names to LaBraM channel names.
        Useful for EEG systems with known correspondences (e.g. Geodesic
        E-number to 10-10).
    """

    _MODEL_CLASS_PATH: tp.ClassVar[str] = "braindecode.models.Labram"
    required_fields: tp.ClassVar[list[RequiredBuildField]] = ["ch_names", "n_times"]
    channel_mapping: dict[str, str] | None = None

    def build(
        self,
        n_spatial_locations: int,
        n_temporal_samples: int,
        n_outputs: int | None = None,
        chs_info: list[dict[str, tp.Any]] | None = None,
        frequency: float | None = None,
    ) -> nn.Module:
        if self.from_pretrained_name is not None:
            # Built at the pretrained shape; the wrapper adapts every batch to
            # it, so the requested one is not passed on.
            model = self._construct()
        else:
            construct_kwargs: dict[str, tp.Any] = {
                "n_chans": n_spatial_locations,
                "n_times": n_temporal_samples,
            }
            if n_outputs is not None:
                construct_kwargs["n_outputs"] = n_outputs
            model = self._construct(**construct_kwargs)

        if chs_info is None:
            return model

        return _LabramChannelWrapper(
            model,
            [ch["ch_name"] for ch in chs_info],
            channel_mapping=self.channel_mapping,
        )
