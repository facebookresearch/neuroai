# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Padded keystroke-sequence extractor for CTC targets."""

from __future__ import annotations

import logging
import typing as tp

import pydantic
import torch

import neuralset as ns
from neuralset.extractors.base import BaseStatic

from .charset import CharacterSet, VocabPreset

LOGGER = logging.getLogger(__name__)


class KeystrokeSequence(BaseStatic):
    """CTC-friendly keystroke sequence extractor.

    Output shape ``(max_target_length + 1,)``: ``out[0]`` is the un-padded
    length ``L``, ``out[1:1+L]`` the labels, the rest ``pad_value`` (defaults
    to the CTC blank index).

    Optional ``core_start_offset`` / ``core_duration`` filter target events
    to ``[start + core_start_offset, start + core_start_offset + core_duration)``
    so a padded EMG window can keep its label set aligned to the un-padded
    "core".  Defaults: no filtering.
    """

    event_types: str = "Keystroke"
    event_field: str = "text"
    max_target_length: int = pydantic.Field(default=128, gt=0)
    pad_value: int | None = None
    aggregation: tp.Literal["cat"] = "cat"
    core_start_offset: float = 0.0
    core_duration: float | None = None
    # Vocabulary preset (see :mod:`charset`).  ``"paper"`` is the 99-class
    # paper-faithful default (Sivakumar et al., NeurIPS 2024);
    # ``"qwerty_compact"`` collapses case + US-QWERTY shift variants and
    # drops Key.shift for a 51-class output.
    vocab_preset: VocabPreset = "paper"

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # Per-instance charset (no process-global mutation).
        self._charset = (
            CharacterSet.qwerty_compact()
            if self.vocab_preset == "qwerty_compact"
            else CharacterSet.paper()
        )
        self._effective_pad = (
            self._charset.null_class if self.pad_value is None else self.pad_value
        )
        self._truncation_warned = False

    @property
    def charset(self) -> CharacterSet:
        """Active vocabulary for this extractor.  Read by the qwerty task's
        CTC metric factory builder so the metric and extractor stay in
        sync per experiment."""
        return self._charset

    def get_static(self, event: ns.events.etypes.Event) -> torch.Tensor:
        cleaned = self._charset.clean_keys([getattr(event, self.event_field, "")])
        if not cleaned:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor([self._charset.key_to_label(cleaned[0])], dtype=torch.long)

    def _filter_to_core(self, events: tp.Any, segment_start: float) -> tp.Any:
        if self.core_duration is None and self.core_start_offset == 0.0:
            return events  # no filter configured
        if not isinstance(events, list):
            return events  # neuralset's standard collate path is List[Event]
        lo = segment_start + self.core_start_offset
        hi = lo + self.core_duration if self.core_duration is not None else float("inf")
        return [e for e in events if lo <= float(e.start) < hi]

    def __call__(self, *args, **kwargs) -> torch.Tensor:  # type: ignore[override]
        events = args[0] if args else kwargs.get("events")
        segment_start = (
            float(args[1]) if len(args) > 1 else float(kwargs.get("start", 0.0))
        )
        events = self._filter_to_core(events, segment_start)

        # BaseStatic's missing-event default is a 1-label dummy; we want a
        # length-0 target for empty core windows. Handle it here.
        if not len(self._event_types_helper.extract(events)):
            if not self.allow_missing:
                return super().__call__(events, *args[1:], **kwargs)
            return self._build_padded(torch.empty(0, dtype=torch.long))

        seq = super().__call__(events, *args[1:], **kwargs).to(torch.long).flatten()
        return self._build_padded(seq)

    def _build_padded(self, seq: torch.Tensor) -> torch.Tensor:
        n = int(seq.numel())
        if n > self.max_target_length and not self._truncation_warned:
            LOGGER.warning(
                "KeystrokeSequence: truncating %d keys to max_target_length=%d "
                "(further occurrences silenced).", n, self.max_target_length,
            )
            self._truncation_warned = True
        L = min(n, self.max_target_length)
        out = torch.full(
            (self.max_target_length + 1,), self._effective_pad, dtype=torch.long
        )
        out[0] = L
        if L > 0:
            out[1 : 1 + L] = seq[:L]
        return out
