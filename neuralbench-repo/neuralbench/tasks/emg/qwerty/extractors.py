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
    """CTC target extractor for typed keystroke sequences.

    Output is a fixed-shape ``(max_target_length + 1,)`` tensor:

    * ``out[0]`` — the un-padded sequence length ``L``;
    * ``out[1 : 1 + L]`` — the encoded labels;
    * remainder — ``pad_value`` (defaults to the CTC blank index).

    With ``core_start_offset`` / ``core_duration`` set, only events
    falling inside the un-padded core of a padded EMG window are kept,
    so a 0.9 s + 4 s + 0.1 s window produces a 4 s target.
    """

    event_types: str = "Keystroke"
    event_field: str = "text"
    max_target_length: int = pydantic.Field(default=128, gt=0)
    pad_value: int | None = None
    aggregation: tp.Literal["cat"] = "cat"
    core_start_offset: float = 0.0
    core_duration: float | None = None
    vocab_preset: VocabPreset = "paper"

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self._charset = (
            CharacterSet.qwerty_compact()
            if self.vocab_preset == "qwerty_compact"
            else CharacterSet.paper()
        )
        self._truncation_warned = False

    @property
    def charset(self) -> CharacterSet:
        """Active vocabulary for this extractor."""
        return self._charset

    @property
    def _effective_pad(self) -> int:
        return self._charset.null_class if self.pad_value is None else self.pad_value

    def get_static(self, event: ns.events.etypes.Event) -> torch.Tensor:
        """Encode one event as a length-1 ``(label,)`` tensor (or empty)."""
        labels = self._charset.encode([getattr(event, self.event_field, "")])
        return torch.tensor(labels, dtype=torch.long)

    def __call__(self, *args, **kwargs) -> torch.Tensor:  # type: ignore[override]
        events = args[0] if args else kwargs.get("events")
        start = float(args[1]) if len(args) > 1 else float(kwargs.get("start", 0.0))
        events = self._restrict_to_core(events, start)

        # BaseStatic's missing-event default is a 1-label dummy; emit a
        # length-0 target instead so CTC sees an empty sequence.
        empty = self.allow_missing and len(
            self._event_types_helper.extract(events)
        ) == 0
        if empty:
            return self._pad(torch.empty(0, dtype=torch.long))
        seq = super().__call__(events, *args[1:], **kwargs)
        return self._pad(seq.to(torch.long).flatten())

    def _restrict_to_core(self, events: tp.Any, start: float) -> tp.Any:
        if self.core_duration is None and self.core_start_offset == 0.0:
            return events
        if not isinstance(events, list):
            return events  # the standard collate path is always List[Event]
        lo = start + self.core_start_offset
        hi = lo + (self.core_duration or float("inf"))
        return [e for e in events if lo <= float(e.start) < hi]

    def _pad(self, seq: torch.Tensor) -> torch.Tensor:
        n = int(seq.numel())
        if n > self.max_target_length and not self._truncation_warned:
            LOGGER.warning(
                "KeystrokeSequence: truncating %d keys to max_target_length=%d "
                "(further occurrences silenced).", n, self.max_target_length,
            )
            self._truncation_warned = True
        length = min(n, self.max_target_length)
        out = torch.full(
            (self.max_target_length + 1,), self._effective_pad, dtype=torch.long,
        )
        out[0] = length
        if length:
            out[1 : 1 + length] = seq[:length]
        return out
