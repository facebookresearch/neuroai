# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Neuraltrain custom configuration for LUNA.

Includes the following adaptations over the raw braindecode LUNA model:

* **Keyword mapping** — the data pipeline provides per-sample 3-D
  electrode positions as ``channel_positions``, whereas braindecode's LUNA
  forward signature expects ``channel_locations``.  The wrapper transparently
  maps between the two.
* **Time padding** — zero-pads the time dimension so it is a multiple of
  LUNA's ``patch_size``.
* **Encoder-only output** — when ``n_outputs`` is not provided (i.e. the
  downstream classification head is handled externally), the classification
  head is replaced with ``nn.Identity()`` so the model returns the normalised
  encoder latent of shape ``(B, n_patches, num_queries * embed_dim)``.
"""

import logging
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBrainDecodeModel

logger = logging.getLogger(__name__)


class _LunaEncoderWrapper(nn.Module):
    """Wraps a braindecode ``LUNA`` to accept ``channel_positions`` instead of
    ``channel_locations`` and to zero-pad the time dimension.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @property
    def mapping(self) -> dict[str, str]:
        """Expose LUNA's key mapping, prefixed for the wrapper's state dict."""
        inner_mapping = getattr(self.model, "mapping", None) or {}
        return {k: f"model.{v}" for k, v in inner_mapping.items()}

    @staticmethod
    def _pad_to_patch_size(X: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Zero-pad the time dimension to the next multiple of *patch_size*."""
        remainder = X.shape[-1] % patch_size
        if remainder == 0:
            return X
        return F.pad(X, (0, patch_size - remainder))

    def forward(
        self,
        X: torch.Tensor,
        channel_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        X = self._pad_to_patch_size(X, self.model.patch_size)  # type: ignore[union-attr,arg-type]
        return self.model(X, channel_locations=channel_positions)


class NtLuna(BaseBrainDecodeModel):
    """Config for the braindecode LUNA model.

    Extends :class:`BaseBrainDecodeModel` with LUNA-specific logic:

    1. **Keyword mapping** — wraps the model so its forward accepts
       ``channel_positions`` and maps it to LUNA's ``channel_locations``.
    2. **Time padding** — zero-pads the time dimension to a multiple of
       ``patch_size``.
    3. **Encoder-only output** — when ``n_outputs`` is not passed (i.e. when
       a ``DownstreamWrapperModel`` handles the classification head), the
       classification head is replaced with ``nn.Identity()`` so the model
       returns the encoder latent.

    Parameters
    ----------
    pretrained_filename : str or None
        When ``from_pretrained_name`` points to a Hub repository containing
        multiple weight files (e.g. ``PulpBio/LUNA``), this selects which
        file to download.  Requires braindecode >= 1.5 which natively
        supports the ``filename`` kwarg in ``from_pretrained``.
    """

    _MODEL_CLASS_PATH: tp.ClassVar[str] = "braindecode.models.LUNA"
    pretrained_filename: str | None = None

    def build(
        self,
        n_chans: int | None = None,
        n_times: int | None = None,
        n_outputs: int | None = None,
        **kwargs: tp.Any,
    ) -> nn.Module:
        build_kwargs: dict[str, tp.Any] = {}
        if n_chans is not None:
            build_kwargs["n_chans"] = n_chans
        if n_times is not None:
            build_kwargs["n_times"] = n_times

        encoder_only = n_outputs is None
        build_kwargs["n_outputs"] = 1 if encoder_only else n_outputs

        if self.from_pretrained_name is not None and self.pretrained_filename is not None:
            kwargs["filename"] = self.pretrained_filename
        model = super().build(**build_kwargs, **kwargs)

        if encoder_only:
            model.final_layer = nn.Identity()

        return _LunaEncoderWrapper(model)
