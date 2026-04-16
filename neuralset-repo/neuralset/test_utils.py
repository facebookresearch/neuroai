# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import pandas as pd
import pytest

from . import events, utils


@pytest.mark.parametrize(
    "value1,value2,expected",
    [
        (100, 110, True),
        (100, 111, False),
        (2000, 2011, True),
    ],
)
@pytest.mark.parametrize("revert", (True, False))
def test_tolerance(value1: float, value2: float, expected: bool, revert: bool):
    if revert:
        value1, value2 = value2, value1
    tol = utils.Tolerance(10, 0.01)
    out = tol(value1, value2)
    assert out is expected


def _base_seq(start: int, stop: int) -> np.ndarray:
    # [0, 1100, 2300, 3600, 5000, 6500, 8100, 9800, 11600, 13500]
    return np.array([k * 1000 + 50 * k * (k + 1) for k in range(start, stop)])


@pytest.mark.parametrize("revert", (True, False))
@pytest.mark.parametrize(
    "seq,num_matches",
    [
        (_base_seq(0, 10), 10),
        # missing start and end
        (_base_seq(1, 9), 8),
        (_base_seq(2, 9), 0),
        # increasing offset
        (_base_seq(0, 10) + np.arange(10) * 10, 10),
        (_base_seq(0, 10) + np.arange(10) * 11, 0),
        # every other
        ([1100, 3600, 6500], 3),
    ],
)
def test_approx_match_sample(seq: tp.Any, num_matches: int, revert: bool) -> None:
    seq0: tp.Any = _base_seq(0, 10).tolist()
    if not isinstance(seq, list):
        seq = seq.tolist()
    seq1 = seq
    if revert:
        seq1, seq0 = seq0, seq1
    try:
        out: tp.Any = utils.approx_match_samples(seq0, seq1, abs_tol=10, max_missing=1)
        actual_num_matches = len(out[0])
    except utils.NoApproximateMatch:
        out = ()
        actual_num_matches = 0
    msg = f"Bad matching: {out} for\n{seq0} and\n{seq1}"
    assert actual_num_matches == num_matches, msg


def test_approx_match_sample_specific_case() -> None:
    seq0 = [1100, 2300, 3600]
    seq1 = [0, 1110, 3620, 6500]
    out = utils.approx_match_samples(seq0, seq1, abs_tol=10, max_missing=1)
    np.testing.assert_array_equal(out, ([0, 2], [1, 2]))


def test_approx_match_sample_with_close_matches() -> None:
    seq0 = [1000, 5000, 7000, 7011]
    seq1 = [1000, 5000, 7010]
    out = utils.approx_match_samples(seq0, seq1, abs_tol=10, max_missing=1)
    np.testing.assert_array_equal(out, ([0, 1, 3], [0, 1, 2]))


def test_approx_match_sample_specific_case2() -> None:
    # PallierListen: Sub2 - run1
    # fmt: off
    seq0 = [    0,   370,   431,   480,   532,  880,  936, 1130, 1186,
             1350,  1405,  1543,  1809,  1864, 2100, 2155, 2450, 2505,
             3090,  3145,  3550,  3605,  4161, 4216, 4811, 4866, 5132,
             5187,  5387,  5658,  5857,  5952, 6283, 6338, 6763, 6818,
             6997,  7125,  7763,  7818,  8304, 8358, 9594, 9649, 9904,
             9960, 10495, 10550, 10676, 11115]
    seq1 = [    0,   370,   480,   880,  1130,  1350,
             1490,  1809,  2099,  2450,  3090,  3550,
             4160,  4810,  5130,  5330,  5600,  5800,
             5900,  6280,  6760,  6940,  7070,  7760,
             8300,  9590,  9900, 10490, 10620, 11110,
            11720, 11900, 12360, 12500, 13670, 14080,
            14210, 14600, 14750, 16420, 16480, 16890,
            17060, 17210, 17930, 18120, 18600, 19090, 19640, 19830]
    # fmt: on
    # should work (but not by matching 0 with 0 as one is an offset that disappears later)
    _ = utils.approx_match_samples(seq0, seq1, abs_tol=10, max_missing=3)


def test_event_types_helper() -> None:
    h = events.etypes.EventTypesHelper(events.etypes.BaseText)
    assert h.classes == (events.etypes.BaseText,)
    assert not {"Word", "Sentence", "Text"} - set(h.names)
    h = events.etypes.EventTypesHelper(("Fmri", "Meg"))
    assert h.classes == (events.etypes.Fmri, events.etypes.Meg)
    assert not {"Fmri", "Meg"} - set(h.names)
    # extraction
    kwargs = {"start": 0, "duration": 1, "timeline": "xxx"}
    df = pd.DataFrame(
        [
            {"type": "Word", "text": "blublu", **kwargs},
            {"type": "Keystroke", "text": "a", **kwargs},
        ]
    )
    extracted = events.etypes.EventTypesHelper("Word").extract(df)
    assert len(extracted) == 1


def test_train_test_split_by_group():
    n_examples, n_features = 100, 10
    X = np.random.rand(n_examples, n_features)
    y = np.random.rand(n_examples) > 0.5
    groups = np.arange(n_examples) // 10

    X_train, X_test, y_train, y_test, group_train, group_test = (
        utils.train_test_split_by_group(
            X, y, groups=groups, test_size=0.2, random_state=33
        )
    )

    assert not set(group_train) & set(group_test)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[0] + X_test.shape[0] == n_examples
