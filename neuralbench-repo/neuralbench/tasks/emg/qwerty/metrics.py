# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Levenshtein-based CER/IER/DER/SER metric for CTC heads."""

from __future__ import annotations

import logging
from collections import Counter

import torch
from torchmetrics import Metric

from .charset import CharacterSet

LOGGER = logging.getLogger(__name__)


class CharacterErrorRates(Metric):
    """CTC character error rate.

    Update accepts batched CTC tensors:
    ``update(emissions, targets, target_lengths)`` where ``emissions`` is a
    ``(T_out, B, num_classes)`` log-prob tensor. Greedy decoding runs
    on-device (argmax) before D2H copy to keep the per-step traffic at
    ``(T, B)`` instead of ``(T, B, C)``.

    :meth:`compute` returns total CER as a scalar tensor so the metric
    integrates with Lightning's ``log()`` directly.  Insertions /
    deletions / substitutions are kept as separate accumulators
    (`self.insertions`, etc.) for callers that want the IER/DER/SER
    breakdown post-hoc.
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(self, charset_: CharacterSet | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._charset = charset_ or CharacterSet.paper()
        for name in ("insertions", "deletions", "substitutions", "target_len"):
            self.add_state(name, default=torch.tensor(0), dist_reduce_fx="sum")

    def update(  # type: ignore[override]
        self,
        emissions: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> None:
        # Lazy import: keeps Levenshtein out of config-discovery time.
        import Levenshtein

        argmax = emissions.argmax(dim=-1).detach().cpu().tolist()
        targets_cpu = targets.detach().cpu().tolist()
        lengths_cpu = target_lengths.detach().cpu().tolist()
        T_out = len(argmax)
        batch_size = len(argmax[0]) if T_out else 0
        blank = self._charset.null_class

        for i in range(batch_size):
            preds: list[int] = []
            prev = blank
            for t in range(T_out):
                lbl = argmax[t][i]
                if lbl != blank and lbl != prev:
                    preds.append(lbl)
                prev = lbl
            pred_str = self._charset.labels_to_str(preds)
            tgt_str = self._charset.labels_to_str(
                targets_cpu[i][: int(lengths_cpu[i])]
            )
            edits = Counter(op for op, _, _ in Levenshtein.editops(pred_str, tgt_str))
            self.insertions += edits["insert"]
            self.deletions += edits["delete"]
            self.substitutions += edits["replace"]
            self.target_len += len(tgt_str)

    def compute(self) -> torch.Tensor:
        """Total CER (%) as a scalar tensor on the metric's device."""
        edits = (self.insertions + self.deletions + self.substitutions).float()
        return edits * 100.0 / self.target_len.clamp(min=1).float()
