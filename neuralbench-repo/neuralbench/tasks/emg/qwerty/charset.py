# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Vocabulary tables for the emg2qwerty CTC task.

Two presets — both consumed by
:class:`neuralset.extractors.text.KeystrokeSequence` via
``!!python/name:`` references in the task YAML configs:

* ``paper`` — Sivakumar et al. NeurIPS 2024, **98 keys + CTC blank**
  (99 classes): ``string.ascii_letters`` + ``string.digits`` +
  ``string.punctuation`` + 4 modifier keys (``Key.backspace``,
  ``Key.enter``, ``Key.space``, ``Key.shift``).
* ``qwerty_compact`` — paper folded down to **50 unique
  US-QWERTY base symbols** (51 classes incl. blank). Defined as a
  small replace mapping over the paper preset: case-fold uppercase
  letters, collapse shifted digits / punctuation to their unshifted
  siblings, drop ``Key.shift{,_l,_r}``. Recommended over ``paper``
  when output capitalization can be recovered downstream — fewer
  classes ⇒ more samples per class ⇒ faster training.

Encoding (raw key → label int) is the only direction needed at
runtime; ``KeystrokeSequence._encode`` owns the cleaning loop.
``CharacterErrorRates`` (CTC metric) intentionally uses ``chr(label)``
directly — distinct labels stay distinct, which is all Levenshtein
needs — so no label → display-char table lives here. Tables are
emitted as ``list[tuple[str, ...]]`` (not dicts) so they survive
``exca.ConfDict``'s YAML key-flattening on dot-containing entries
like ``"Key.backspace"`` and ``"."``.
"""

from __future__ import annotations

import string
from typing import Literal

VocabPreset = Literal["paper", "qwerty_compact"]

# --- paper preset --------------------------------------------------------

_PAPER_VOCAB: tuple[str, ...] = (
    *string.ascii_letters,
    *string.digits,
    *string.punctuation,
    "Key.backspace",
    "Key.enter",
    "Key.space",
    "Key.shift",
)

# Input substitutions paper applies to raw event keys (all 1-char).
_PAPER_UNICHAR: dict[str, str | None] = {
    " ": "Key.space",
    "\n": "Key.enter",
    "\r": "Key.enter",
    "\b": "Key.backspace",
    "⇧": "Key.shift",
    "⏎": "Key.enter",
    "⌫": "Key.backspace",
}

# --- qwerty_compact preset = paper + US-QWERTY shift fold ----------------
# A single fold dict defines everything compact differs from paper in.
# Mapping a key to ``None`` drops it from the input stream.

_COMPACT_FOLDS: dict[str, str | None] = {
    **{c: c.lower() for c in string.ascii_uppercase},          # case fold
    **dict(zip("!@#$%^&*()", "1234567890")),                   # shifted digits
    **dict(zip('~_+{}|:"<>?', "`-=[]\\;',./")),                # shifted punctuation
    "Key.shift": None,
    "Key.shift_l": None,
    "Key.shift_r": None,
}

# Compact's punctuation row follows the **US-QWERTY physical layout**
# (left-to-right unshifted keys ``` `-=[]\;',./ ```), not
# ``string.punctuation`` order — that's the assignment compact-preset
# checkpoints (and the existing tests) are pinned against. Listed
# explicitly rather than derived from ``_PAPER_VOCAB`` minus the fold
# map. Every key kept here must be a paper-vocab key (asserted below).
_COMPACT_VOCAB: tuple[str, ...] = (
    *string.ascii_lowercase,
    *string.digits,
    *"`-=[]\\;',./",
    "Key.backspace",
    "Key.enter",
    "Key.space",
)
assert set(_COMPACT_VOCAB) <= set(_PAPER_VOCAB), (
    "compact vocab must be a subset of paper vocab"
)

# Compact aliases = paper unichar map (minus the shift sentinel) merged
# with the fold map. ``"⇧": None`` overrides paper's ``"⇧": "Key.shift"``
# — later-key-wins semantics of dict merge.
_COMPACT_ALIASES: dict[str, str | None] = {
    **{u: k for u, k in _PAPER_UNICHAR.items() if k != "Key.shift"},
    "⇧": None,
    **_COMPACT_FOLDS,
}


def _by_len(
    aliases: dict[str, str | None], one: bool
) -> list[tuple[str, str | None]]:
    """Partition aliases by ``len(key) == 1`` so ``KeystrokeSequence``'s
    two normalization fields stay semantically equivalent.

    1-char entries → ``unichar_to_key`` (matches the ``len(k) == 1``
    fallback branch in ``KeystrokeSequence._encode``); multi-char
    entries → ``input_folds`` (the unconditional fold branch). The
    consumer's existing 3-step lookup (folds → key_to_label → 1-char
    unichar) is preserved by this split.
    """
    return [(k, v) for k, v in aliases.items() if (len(k) == 1) == one]


# --- YAML-facing constants (names + types pinned for ``!!python/name:``) -

PAPER_KEY_TO_LABEL: list[tuple[str, int]] = [
    (k, i) for i, k in enumerate(_PAPER_VOCAB)
]
PAPER_UNICHAR_TO_KEY: list[tuple[str, str | None]] = _by_len(_PAPER_UNICHAR, one=True)
PAPER_INPUT_FOLDS: list[tuple[str, str | None]] = _by_len(_PAPER_UNICHAR, one=False)
PAPER_NULL_CLASS = len(_PAPER_VOCAB)        # 98
PAPER_NUM_CLASSES = PAPER_NULL_CLASS + 1    # 99 (98 keys + CTC blank)

COMPACT_KEY_TO_LABEL: list[tuple[str, int]] = [
    (k, i) for i, k in enumerate(_COMPACT_VOCAB)
]
COMPACT_UNICHAR_TO_KEY: list[tuple[str, str | None]] = _by_len(
    _COMPACT_ALIASES, one=True
)
COMPACT_INPUT_FOLDS: list[tuple[str, str | None]] = _by_len(
    _COMPACT_ALIASES, one=False
)
COMPACT_NULL_CLASS = len(_COMPACT_VOCAB)        # 50
COMPACT_NUM_CLASSES = COMPACT_NULL_CLASS + 1    # 51


# --- single helper for tests + programmatic callers ----------------------


def vocab_kwargs(preset: VocabPreset = "paper") -> dict:
    """Build the kwargs dict ``KeystrokeSequence(**vocab_kwargs(preset))`` consumes."""
    tables = {
        "paper":          (PAPER_KEY_TO_LABEL,   PAPER_UNICHAR_TO_KEY,   PAPER_INPUT_FOLDS),
        "qwerty_compact": (COMPACT_KEY_TO_LABEL, COMPACT_UNICHAR_TO_KEY, COMPACT_INPUT_FOLDS),
    }
    k2l, u2k, folds = tables[preset]
    return {
        "key_to_label":   dict(k2l),
        "unichar_to_key": dict(u2k),
        "input_folds":    dict(folds),
    }
