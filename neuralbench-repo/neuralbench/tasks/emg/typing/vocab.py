# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""emg2qwerty CTC vocabulary (Sivakumar et al., NeurIPS 2024): 98 keys + blank.

Wired into the task config (``config.yaml``) via ``!!python/name:`` so
``LabelEncoder`` can resolve ``PAPER_KEY_TO_LABEL`` at YAML load time.
"""

from __future__ import annotations

import string

_VOCAB: tuple[str, ...] = (
    *string.ascii_letters,
    *string.digits,
    *string.punctuation,
    "Key.backspace",
    "Key.enter",
    "Key.space",
    "Key.shift",
)

PAPER_KEY_TO_LABEL: dict[str, int] = {key: i for i, key in enumerate(_VOCAB)}
PAPER_NULL_CLASS = len(_VOCAB)  # 98
PAPER_NUM_CLASSES = PAPER_NULL_CLASS + 1  # 99
