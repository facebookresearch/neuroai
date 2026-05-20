# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests specific to the emg/typing task — vocabulary tables (this
package owns ``charset.py``).

``Sivakumar2024Emg2qwerty`` / ``KeystrokeSequence`` / ``CharacterErrorRates`` /
``CtcSeqLoss`` each have their own test files in the package that owns
them (``neuralfetch.test_sivakumar2024emg2qwerty``,
``neuralset.extractors.test_text``, ``neuraltrain.metrics.test_metrics``,
``neuraltrain.losses.test_losses``).
"""

from __future__ import annotations

from .charset import (
    COMPACT_KEY_TO_LABEL,
    COMPACT_NULL_CLASS,
    COMPACT_NUM_CLASSES,
    PAPER_KEY_TO_LABEL,
    PAPER_NULL_CLASS,
    PAPER_NUM_CLASSES,
)


def test_charset_class_count_invariants():
    paper_keys = {k for k, _ in PAPER_KEY_TO_LABEL}
    assert (PAPER_NULL_CLASS, PAPER_NUM_CLASSES, len(paper_keys)) == (98, 99, 98)

    compact_keys = {k for k, _ in COMPACT_KEY_TO_LABEL}
    assert (COMPACT_NULL_CLASS, COMPACT_NUM_CLASSES, len(compact_keys)) == (50, 51, 50)
    # compact drops uppercase, shifted symbols, and Key.shift.
    assert {"Key.shift", "A", "!"}.isdisjoint(compact_keys)
