# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`neuralbench.aggregator`."""

import typing as tp
from types import SimpleNamespace

from .aggregator import _infer_eval_mode
from .modules import DownstreamWrapper
from .registry import ALL_DOWNSTREAM_WRAPPERS


def _eval_mode(wrapper: DownstreamWrapper | None) -> str:
    return _infer_eval_mode(
        tp.cast(tp.Any, SimpleNamespace(downstream_model_wrapper=wrapper))
    )


def test_infer_eval_mode_tags_each_shipped_preset_distinctly():
    modes = {
        name: _eval_mode(DownstreamWrapper(**preset["downstream_model_wrapper"]))
        for name, preset in ALL_DOWNSTREAM_WRAPPERS.items()
    }
    # presets sharing a tag would merge everywhere downstream: plots, tables, folders
    assert modes == {name: name for name in ALL_DOWNSTREAM_WRAPPERS}


def test_infer_eval_mode_without_wrapper_is_finetune():
    assert _eval_mode(None) == "finetune"
