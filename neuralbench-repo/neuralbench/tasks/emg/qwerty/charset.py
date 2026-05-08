# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Vocabulary presets for the emg2qwerty CTC task.

Two presets ship out of the box:

* ``paper`` (default) — the 98-key vocabulary from Sivakumar et al.
  (NeurIPS 2024): ``string.ascii_letters`` + ``string.digits`` +
  ``string.punctuation`` + 4 modifier keys (backspace, enter, space,
  shift).  ``num_classes = 99`` (98 keys + CTC blank).
* ``qwerty_compact`` — a 50-key US-QWERTY-folded variant that case-folds
  uppercase letters into their lowercase equivalents, folds shifted
  symbols (``!@#$%^&*()`` and the punctuation row) into their unshifted
  forms, and drops the ``Key.shift`` modifier (which is now redundant).
  ``num_classes = 51``.  Recommended over ``paper`` for typing-decoding
  experiments where output capitalization / punctuation can be recovered
  by a downstream language model: the smaller output space gives more
  samples per class and trains faster, at the cost of losing
  shift-state information in the raw decoder output.
"""

from __future__ import annotations

import string
import typing as tp
from collections import OrderedDict
from collections.abc import Sequence

# Single source of truth for the supported preset names.  Extractors and
# tests both read this so adding a preset requires touching one place.
VocabPreset = tp.Literal["paper", "qwerty_compact"]


_PAPER_KEY_TO_UNICODE: OrderedDict[str, int] = OrderedDict(
    [(c, ord(c)) for c in string.ascii_letters + string.digits + string.punctuation]
    + [
        ("Key.backspace", 9003),  # ⌫
        ("Key.enter", 9166),      # ⏎
        ("Key.space", 32),
        ("Key.shift", 8679),      # ⇧
    ]
)
# Stray unicode-only inputs ("⏎", " ", "\n", ...) get normalized to the
# canonical KEY_TO_UNICODE entry by ``clean_keys``.  ``Key.shift_l`` /
# ``Key.shift_r`` are dropped — only plain ``Key.shift`` is in-vocab.
_PAPER_UNICHAR_TO_KEY: dict[str, str] = {
    " ": "Key.space",
    "\n": "Key.enter",
    "\r": "Key.enter",
    "\b": "Key.backspace",
    "⇧": "Key.shift",
    "⏎": "Key.enter",
    "⌫": "Key.backspace",
}

# US-QWERTY shift folds for the compact preset.  Built once at import time.
# Pairs are (shifted, unshifted) on a standard US layout.
_QWERTY_COMPACT_BASE = string.ascii_lowercase + string.digits + "`-=[]\\;',./"

_COMPACT_KEY_TO_UNICODE: OrderedDict[str, int] = OrderedDict(
    [(c, ord(c)) for c in _QWERTY_COMPACT_BASE]
    + [
        ("Key.backspace", 9003),
        ("Key.enter", 9166),
        ("Key.space", 32),
        # Key.shift dropped: shifted variants are folded to their
        # unshifted base in clean_keys, so the modifier is unreachable.
    ]
)

_COMPACT_INPUT_FOLDS: dict[str, str | None] = {
    # Uppercase → lowercase
    **dict(zip(string.ascii_uppercase, string.ascii_lowercase)),
    # Shifted-digit → digit
    **dict(zip("!@#$%^&*()", "1234567890")),
    # Other shifted punctuation → unshifted punctuation
    **dict(zip('~_+{}|:"<>?', "`-=[]\\;',./")),
    # Shift modifiers are no longer in-vocab; map to None to drop entirely.
    "Key.shift": None,
    "Key.shift_l": None,
    "Key.shift_r": None,
}

_COMPACT_UNICHAR_TO_KEY: dict[str, str | None] = {
    " ": "Key.space",
    "\n": "Key.enter",
    "\r": "Key.enter",
    "\b": "Key.backspace",
    "⏎": "Key.enter",
    "⌫": "Key.backspace",
    # Shift unicode sentinel is dropped in compact mode.
    "⇧": None,
}


class CharacterSet:
    """Configurable typing vocabulary plus blank → ``num_classes`` for CTC.

    Use :meth:`paper` (default, 99 classes) for paper-faithful runs and
    :meth:`qwerty_compact` (51 classes) for the case-folded
    shift-collapsed variant.
    """

    def __init__(
        self,
        key_to_unicode: OrderedDict[str, int] | None = None,
        unichar_to_key: dict[str, str | None] | None = None,
        input_folds: dict[str, str | None] | None = None,
    ) -> None:
        self.KEY_TO_UNICODE = key_to_unicode if key_to_unicode is not None else _PAPER_KEY_TO_UNICODE
        self.UNICHAR_TO_KEY = unichar_to_key if unichar_to_key is not None else _PAPER_UNICHAR_TO_KEY
        # Folds normalize *input* keys before vocabulary lookup.  Mapping a
        # key to ``None`` means "drop entirely".  Empty dict => no folding.
        self._input_folds: dict[str, str | None] = input_folds or {}
        self._key_to_index = {k: i for i, k in enumerate(self.KEY_TO_UNICODE)}

    @classmethod
    def paper(cls) -> "CharacterSet":
        """98-key paper-faithful vocabulary (Sivakumar et al., NeurIPS 2024)."""
        return cls()

    @classmethod
    def qwerty_compact(cls) -> "CharacterSet":
        """50-key US-QWERTY-folded vocabulary; ``num_classes = 51``.

        Uppercase letters and shifted symbols fold to their unshifted
        forms in :meth:`clean_keys`; the ``Key.shift`` modifier is dropped.
        """
        return cls(
            key_to_unicode=_COMPACT_KEY_TO_UNICODE,
            unichar_to_key=_COMPACT_UNICHAR_TO_KEY,
            input_folds=_COMPACT_INPUT_FOLDS,
        )

    @property
    def null_class(self) -> int:
        """Index of the CTC blank class (= ``len(KEY_TO_UNICODE)``)."""
        return len(self.KEY_TO_UNICODE)

    @property
    def num_classes(self) -> int:
        """Vocabulary size + 1 for blank."""
        return len(self.KEY_TO_UNICODE) + 1

    def key_to_label(self, key: str) -> int:
        return self._key_to_index[key]

    def labels_to_str(self, labels: Sequence[int]) -> str:
        keys = tuple(self.KEY_TO_UNICODE.keys())
        return "".join(chr(self.KEY_TO_UNICODE[keys[label]]) for label in labels)

    def clean_keys(self, keys: Sequence[str]) -> list[str]:
        """Normalize input keys and filter to the active vocabulary.

        Per-character folds (e.g. case + shift collapse for the compact
        preset) run first, so an upstream ``"A"`` event becomes ``"a"``
        before the lookup.  Folded-to-``None`` entries are dropped.
        """
        out: list[str] = []
        for k in keys:
            if k in self._input_folds:
                folded = self._input_folds[k]
                if folded is None:
                    continue  # explicit drop (e.g. Key.shift in compact)
                k = folded
            if k in self.KEY_TO_UNICODE:
                out.append(k)
                continue
            if len(k) == 1:
                normalized = self.UNICHAR_TO_KEY.get(k, k)
                if normalized is None:
                    continue
                if normalized in self.KEY_TO_UNICODE:
                    out.append(normalized)
        return out

    def encode(self, keys: Sequence[str]) -> list[int]:
        """Clean ``keys`` and return their integer labels in one call.

        Equivalent to ``[key_to_label(k) for k in clean_keys(keys)]``;
        unknown / folded-out keys are simply absent from the result.
        Mirrors the MNE pattern of single-call accessors that handle
        normalization + lookup together.
        """
        return [self.key_to_label(k) for k in self.clean_keys(keys)]


