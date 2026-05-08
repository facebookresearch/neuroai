# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Levenshtein-based CER metric for CTC heads."""

from __future__ import annotations

import logging
from collections import Counter

import torch
from neuraltrain.metrics.base import BaseMetric
from neuraltrain.utils import convert_to_pydantic
from torchmetrics import Metric

from .charset import CharacterSet

LOGGER = logging.getLogger(__name__)


class CharacterErrorRates(Metric):
    """CTC character error rate.

    Accepts the standard ``update(y_pred, y_true)`` signature so the
    metric flows through ``BrainModule.metrics`` like any other.

    * ``y_pred``: ``(B, T_out, num_classes)`` log-probs from the model.
    * ``y_true``: ``(B, max_target_length + 1)`` length-prefixed labels
      from :class:`KeystrokeSequence` — col 0 is the un-padded length.

    Greedy decoding (argmax + collapse-repeats + drop-blanks) runs
    on-device before D2H copy.  ``compute`` returns total CER (%) as a
    scalar tensor; raw insertions / deletions / substitutions / target
    lengths are kept as accumulator state for post-hoc IER/DER/SER.
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(self, vocab_preset: str = "paper", **kwargs) -> None:
        super().__init__(**kwargs)
        self._charset = (
            CharacterSet.qwerty_compact()
            if vocab_preset == "qwerty_compact"
            else CharacterSet.paper()
        )
        for name in ("insertions", "deletions", "substitutions", "target_len"):
            self.add_state(name, default=torch.tensor(0), dist_reduce_fx="sum")

    def update(  # type: ignore[override]
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> None:
        # Lazy import keeps Levenshtein off the YAML-resolution path.
        import Levenshtein

        target_lengths = y_true[:, 0].long().detach().cpu().tolist()
        targets = y_true[:, 1:].long().detach().cpu().tolist()
        argmax = y_pred.argmax(dim=-1).detach().cpu().tolist()  # (B, T_out)
        blank = self._charset.null_class

        for i, row in enumerate(argmax):
            preds: list[int] = []
            prev = blank
            for lbl in row:
                if lbl != blank and lbl != prev:
                    preds.append(lbl)
                prev = lbl
            pred_str = self._charset.labels_to_str(preds)
            tgt_str = self._charset.labels_to_str(targets[i][: target_lengths[i]])
            edits = Counter(op for op, _, _ in Levenshtein.editops(pred_str, tgt_str))
            self.insertions += edits["insert"]
            self.deletions += edits["delete"]
            self.substitutions += edits["replace"]
            self.target_len += len(tgt_str)

    def compute(self) -> torch.Tensor:
        """Total CER (%) as a scalar tensor on the metric's device."""
        edits = (self.insertions + self.deletions + self.substitutions).float()
        return edits * 100.0 / self.target_len.clamp(min=1).float()


# Auto-register a BaseMetric config so YAML can wire it via the standard
# ``metrics: [{name: CharacterErrorRates, log_name: CER, ...}]`` block.
# Same pattern neuraltrain uses for its own custom metrics; the assignment
# keeps the generated class alive so exca's discriminator finds it.
_CharacterErrorRatesConfig = convert_to_pydantic(
    CharacterErrorRates,
    "CharacterErrorRates",
    parent_class=BaseMetric,
    exclude_from_build=["log_name"],
)
