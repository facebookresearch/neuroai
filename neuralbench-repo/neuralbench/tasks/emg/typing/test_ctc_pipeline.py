# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests specific to the emg/typing task — vocabulary tables (this
package owns ``vocab.py``).

The CTC target is now produced by ``LabelEncoder`` with
``aggregation='cat'`` + ``max_length`` and consumed by stock
``nn.CTCLoss`` (length-recovery branch in ``BrainModule._run_step``).
``Sivakumar2024Emg2qwerty`` / ``LabelEncoder`` /
``CharacterErrorRates`` each have their own test files
(``neuralfetch.test_sivakumar2024emg2qwerty``,
``neuralset.extractors.test_text``, ``neuraltrain.metrics.test_metrics``).
"""

from __future__ import annotations

from .vocab import PAPER_KEY_TO_LABEL, PAPER_NULL_CLASS, PAPER_NUM_CLASSES


def test_charset_class_count_invariants():
    paper_keys = set(PAPER_KEY_TO_LABEL)
    assert (PAPER_NULL_CLASS, PAPER_NUM_CLASSES, len(paper_keys)) == (98, 99, 98)
    # Label values are dense in [0, PAPER_NULL_CLASS); the blank sits at
    # PAPER_NULL_CLASS and the CTC head emits PAPER_NUM_CLASSES logits.
    assert set(PAPER_KEY_TO_LABEL.values()) == set(range(PAPER_NULL_CLASS))
