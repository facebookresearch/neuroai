# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""emg2qwerty CTC vocabulary tables.

* ``paper`` — Sivakumar et al. NeurIPS 2024: 98 keys + CTC blank.
* ``qwerty_compact`` — paper + US-QWERTY shift fold: 50 keys + blank.
"""

from __future__ import annotations

import string
from typing import Literal

VocabPreset = Literal["paper", "qwerty_compact"]

# --- paper -----------------------------------------------------------------

_PAPER_VOCAB: tuple[str, ...] = (
    *string.ascii_letters, *string.digits, *string.punctuation,
    "Key.backspace", "Key.enter", "Key.space", "Key.shift",
)
_PAPER_UNICHAR: dict[str, str | None] = {
    " ": "Key.space", "\n": "Key.enter", "\r": "Key.enter", "\b": "Key.backspace",
    "⇧": "Key.shift", "⏎": "Key.enter", "⌫": "Key.backspace",
}

# --- qwerty_compact = paper + this fold map -------------------------------

_COMPACT_FOLDS: dict[str, str | None] = {
    **{c: c.lower() for c in string.ascii_uppercase},          # case
    **dict(zip("!@#$%^&*()", "1234567890")),                   # shift+digit
    **dict(zip('~_+{}|:"<>?', "`-=[]\\;',./")),                # shift+punct
    "Key.shift": None, "Key.shift_l": None, "Key.shift_r": None,
}
# Order follows US-QWERTY physical layout (preserves existing label assignment).
_COMPACT_VOCAB: tuple[str, ...] = (
    *string.ascii_lowercase, *string.digits, *"`-=[]\\;',./",
    "Key.backspace", "Key.enter", "Key.space",
)
assert set(_COMPACT_VOCAB) <= set(_PAPER_VOCAB)
_COMPACT_ALIASES: dict[str, str | None] = {
    **{u: k for u, k in _PAPER_UNICHAR.items() if k != "Key.shift"},
    "⇧": None,
    **_COMPACT_FOLDS,
}


def _by_len(aliases: dict[str, str | None], one: bool) -> list[tuple[str, str | None]]:
    return [(k, v) for k, v in aliases.items() if (len(k) == 1) == one]


# --- YAML-facing constants (consumed by ``KeystrokeSequence``) -----------

PAPER_KEY_TO_LABEL: list[tuple[str, int]] = [(k, i) for i, k in enumerate(_PAPER_VOCAB)]
PAPER_UNICHAR_TO_KEY: list[tuple[str, str | None]] = _by_len(_PAPER_UNICHAR, one=True)
PAPER_INPUT_FOLDS: list[tuple[str, str | None]] = _by_len(_PAPER_UNICHAR, one=False)
PAPER_NULL_CLASS = len(_PAPER_VOCAB)        # 98
PAPER_NUM_CLASSES = PAPER_NULL_CLASS + 1    # 99

COMPACT_KEY_TO_LABEL: list[tuple[str, int]] = [(k, i) for i, k in enumerate(_COMPACT_VOCAB)]
COMPACT_UNICHAR_TO_KEY: list[tuple[str, str | None]] = _by_len(_COMPACT_ALIASES, one=True)
COMPACT_INPUT_FOLDS: list[tuple[str, str | None]] = _by_len(_COMPACT_ALIASES, one=False)
COMPACT_NULL_CLASS = len(_COMPACT_VOCAB)        # 50
COMPACT_NUM_CLASSES = COMPACT_NULL_CLASS + 1    # 51


def vocab_kwargs(preset: VocabPreset = "paper") -> dict:
    """Kwargs for ``KeystrokeSequence(**vocab_kwargs(preset))``."""
    tables = {
        "paper": (PAPER_KEY_TO_LABEL, PAPER_UNICHAR_TO_KEY, PAPER_INPUT_FOLDS),
        "qwerty_compact": (
            COMPACT_KEY_TO_LABEL, COMPACT_UNICHAR_TO_KEY, COMPACT_INPUT_FOLDS,
        ),
    }
    k2l, u2k, folds = tables[preset]
    return {"key_to_label": dict(k2l), "unichar_to_key": dict(u2k), "input_folds": dict(folds)}
