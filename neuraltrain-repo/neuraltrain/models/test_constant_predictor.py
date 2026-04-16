# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .constant_predictor import ConstantPredictor, ConstantPredictorModel


def test_constant_predictor_clf():
    y_train = torch.cat(
        [
            torch.zeros(10, dtype=torch.long),  # Class 0: 10 samples
            torch.ones(30, dtype=torch.long),  # Class 1: 30 samples (most frequent)
            torch.full((5,), 2, dtype=torch.long),  # Class 2: 5 samples
        ]
    )

    config = ConstantPredictor()
    predictor = config.build(y_train)

    assert isinstance(predictor, ConstantPredictorModel)

    X_test = torch.randn(20, 10, 100)  # batch_size, n_channels, n_times
    y_pred = predictor(X_test)

    assert y_pred.shape == (20, 3)
    assert torch.allclose(y_pred, torch.Tensor([0.0, 1.0, 0.0]))


def test_constant_predictor_reg():
    y_train = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ],
        dtype=torch.float,
    )

    config = ConstantPredictor()
    predictor = config.build(y_train)

    assert isinstance(predictor, ConstantPredictorModel)

    X_test = torch.randn(10, 10, 100)  # batch_size, n_channels, n_times
    y_pred = predictor(X_test)

    expected = y_train.mean(dim=0)
    assert y_pred.shape == (10, 3)
    assert torch.allclose(y_pred, expected)


def test_constant_predictor_multilabel():
    # Multi-label classification: each sample can have multiple active classes
    # Format: binary indicators where 1 = class is active, 0 = class is inactive
    y_train = torch.tensor(
        [
            [1, 0, 1, 0],  # Classes 0 and 2 active
            [1, 1, 0, 0],  # Classes 0 and 1 active
            [0, 1, 1, 0],  # Classes 1 and 2 active
            [1, 0, 1, 1],  # Classes 0, 2, and 3 active
            [1, 0, 0, 0],  # Only class 0 active
        ],
        dtype=torch.long,
    )
    # Most frequent values per class: [1, 0, 1, 0]
    # Class 0: appears 4 times (most frequent: 1)
    # Class 1: appears 2 times (most frequent: 0)
    # Class 2: appears 3 times (most frequent: 1)
    # Class 3: appears 1 time (most frequent: 0)

    config = ConstantPredictor()
    predictor = config.build(y_train)

    assert isinstance(predictor, ConstantPredictorModel)

    X_test = torch.randn(10, 10, 100)  # batch_size, n_channels, n_times
    y_pred = predictor(X_test)

    expected = y_train.mode(dim=0)[0].float()
    assert y_pred.shape == (10, 4)
    assert torch.allclose(y_pred, expected)
    assert torch.allclose(y_pred, torch.tensor([1.0, 0.0, 1.0, 0.0]))


@pytest.mark.parametrize(
    "y_train, expected_mode",
    [
        pytest.param(
            torch.tensor([0, 1, 1, 2, 1], dtype=torch.long),
            "most_frequent",
            id="1d_long_single_label",
        ),
        pytest.param(
            torch.tensor(
                [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]],
                dtype=torch.long,
            ),
            "most_frequent",
            id="2d_long_one_hot",
        ),
        pytest.param(
            torch.tensor(
                [[1, 0, 1], [1, 1, 0], [0, 0, 0], [1, 0, 1]],
                dtype=torch.long,
            ),
            "most_frequent_multilabel",
            id="2d_long_multilabel",
        ),
        pytest.param(
            torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                dtype=torch.float,
            ),
            "mean",
            id="2d_float_regression",
        ),
    ],
)
def test_constant_predictor_auto_mode(y_train, expected_mode):
    """Verify that mode='auto' resolves identically to the expected explicit mode."""
    auto_predictor = ConstantPredictor(mode="auto").build(y_train)
    explicit_predictor = ConstantPredictor(mode=expected_mode).build(y_train)

    X_test = torch.randn(5, 2, 10)
    assert torch.allclose(auto_predictor(X_test), explicit_predictor(X_test))


def test_constant_predictor_auto_mode_unsupported_dtype():
    y_train = torch.tensor([True, False, True])
    with pytest.raises(ValueError, match="Unsupported dtype"):
        ConstantPredictor(mode="auto").build(y_train)
