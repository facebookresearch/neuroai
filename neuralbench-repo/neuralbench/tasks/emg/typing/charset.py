# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""emg2qwerty CTC vocabulary tables.

Two presets ship out of the box:

* ``paper`` — Sivakumar et al. NeurIPS 2024: 98 keys + CTC blank.
* ``qwerty_compact`` — paper + case-fold + US-QWERTY shift-fold: 50 keys + blank.

Each preset exposes three list-of-pairs constants
(``*_KEY_TO_LABEL``, ``*_UNICHAR_TO_KEY``, ``*_INPUT_FOLDS``) that the
task YAML wires into :class:`neuralset.extractors.text.KeystrokeSequence`
via ``!!python/name:``.  ``vocab_kwargs(preset)`` returns the equivalent
Python-friendly dict form for tests / inline use.
"""

from __future__ import annotations

import string
from typing import Literal

from neuralset.extractors.text import Folds, Vocab

VocabPreset = Literal["paper", "qwerty_compact"]


# --- paper -----------------------------------------------------------------

# Ordered key list; label = position in this tuple.
_PAPER_VOCAB: tuple[str, ...] = (
    *string.ascii_letters, *string.digits, *string.punctuation,
    "Key.backspace", "Key.enter", "Key.space", "Key.shift",
)
# Single-char aliases applied in KeystrokeSequence._encode's "Unicode
# fallback" step (e.g. literal " " arrives → use "Key.space" label).
_PAPER_UNICHAR_ALIASES: dict[str, str | None] = {
    " ": "Key.space", "\n": "Key.enter", "\r": "Key.enter", "\b": "Key.backspace",
    "⇧": "Key.shift", "⏎": "Key.enter", "⌫": "Key.backspace",
}
# Multi-char input folds applied first (paper has none).
_PAPER_INPUT_FOLDS: dict[str, str | None] = {}


# --- qwerty_compact = paper minus case + shift-fold + drop ``Key.shift`` ---

# Order follows the US-QWERTY physical layout (preserves the label
# assignment used by published checkpoints).
_COMPACT_VOCAB: tuple[str, ...] = (
    *string.ascii_lowercase, *string.digits, *"`-=[]\\;',./",
    "Key.backspace", "Key.enter", "Key.space",
)
assert set(_COMPACT_VOCAB) <= set(_PAPER_VOCAB)

# Inherit paper's literal-character aliases (minus the shift sentinel,
# which is explicitly dropped to ``None`` in compact mode), then add
# case + shifted-char folds.
_COMPACT_UNICHAR_ALIASES: dict[str, str | None] = {
    **{u: k for u, k in _PAPER_UNICHAR_ALIASES.items() if k != "Key.shift"},
    "⇧": None,
    **{c: c.lower() for c in string.ascii_uppercase},        # case fold
    **dict(zip("!@#$%^&*()", "1234567890")),                 # shift+digit
    **dict(zip('~_+{}|:"<>?', "`-=[]\\;',./")),              # shift+punct
}
# Multi-char input folds: explicit-drop the three shift variants.
_COMPACT_INPUT_FOLDS: dict[str, str | None] = {
    "Key.shift": None, "Key.shift_l": None, "Key.shift_r": None,
}


# --- YAML-facing constants (consumed by ``KeystrokeSequence``) ------------

PAPER_KEY_TO_LABEL: Vocab = list(zip(_PAPER_VOCAB, range(len(_PAPER_VOCAB))))
PAPER_UNICHAR_TO_KEY: Folds = list(_PAPER_UNICHAR_ALIASES.items())
PAPER_INPUT_FOLDS: Folds = list(_PAPER_INPUT_FOLDS.items())
PAPER_NULL_CLASS = len(_PAPER_VOCAB)        # 98
PAPER_NUM_CLASSES = PAPER_NULL_CLASS + 1    # 99

COMPACT_KEY_TO_LABEL: Vocab = list(zip(_COMPACT_VOCAB, range(len(_COMPACT_VOCAB))))
COMPACT_UNICHAR_TO_KEY: Folds = list(_COMPACT_UNICHAR_ALIASES.items())
COMPACT_INPUT_FOLDS: Folds = list(_COMPACT_INPUT_FOLDS.items())
COMPACT_NULL_CLASS = len(_COMPACT_VOCAB)        # 50
COMPACT_NUM_CLASSES = COMPACT_NULL_CLASS + 1    # 51


_PRESETS: dict[VocabPreset, tuple] = {
    "paper": (_PAPER_VOCAB, _PAPER_UNICHAR_ALIASES, _PAPER_INPUT_FOLDS),
    "qwerty_compact": (_COMPACT_VOCAB, _COMPACT_UNICHAR_ALIASES, _COMPACT_INPUT_FOLDS),
}


def vocab_kwargs(preset: VocabPreset = "paper") -> dict:
    """Kwargs for ``KeystrokeSequence(**vocab_kwargs(preset))``.

    Returns the dict form (natural for Python callers); YAML configs
    should use the list-of-pairs ``*_KEY_TO_LABEL`` /
    ``*_UNICHAR_TO_KEY`` / ``*_INPUT_FOLDS`` constants above.
    """
    vocab, unichar, folds = _PRESETS[preset]
    return {
        "key_to_label": {k: i for i, k in enumerate(vocab)},
        "unichar_to_key": dict(unichar),
        "input_folds": dict(folds),
    }
