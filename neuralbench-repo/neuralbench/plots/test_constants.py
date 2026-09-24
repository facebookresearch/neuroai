# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`neuralbench.plots._constants`."""

from __future__ import annotations

import pytest

from neuralbench.plots._constants import AdaptationMode


@pytest.mark.parametrize(
    "tag",
    [
        "finetune",
        "finetune_mean",
        "linear_probe_flatten",
        "attentive_probe",
        "lora_r4",
        "lora_r32_mean",
        "some_future_strategy",
    ],
)
def test_adaptation_mode_round_trips(tag: str) -> None:
    assert AdaptationMode.parse(tag).tag == tag
