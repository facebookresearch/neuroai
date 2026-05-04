# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .dummy_predictor import DummyPredictor, DummyPredictorModel


def test_dummy_predictor_clf():
    y_train = torch.cat(
        [
            torch.zeros(10, dtype=torch.long),  # Class 0: 10 samples
            torch.ones(30, dtype=torch.long),  # Class 1: 30 samples (most frequent)
            torch.full((5,), 2, dtype=torch.long),  # Class 2: 5 samples
        ]
    )

    config = DummyPredictor()
    predictor = config.build(y_train)

    assert isinstance(predictor, DummyPredictorModel)

    X_test = torch.randn(20, 10, 100)  # batch_size, n_channels, n_times
    y_pred = predictor(X_test)

    assert y_pred.shape == (20, 3)
    assert torch.allclose(y_pred, torch.Tensor([0.0, 1.0, 0.0]))


def test_dummy_predictor_reg():
    y_train = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ],
        dtype=torch.float,
    )

    config = DummyPredictor()
    predictor = config.build(y_train)

    assert isinstance(predictor, DummyPredictorModel)

    X_test = torch.randn(10, 10, 100)  # batch_size, n_channels, n_times
    y_pred = predictor(X_test)

    expected = y_train.mean(dim=0)
    assert y_pred.shape == (10, 3)
    assert torch.allclose(y_pred, expected)


def test_dummy_predictor_most_frequent_multilabel():
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

    config = DummyPredictor(mode="most_frequent_multilabel")
    predictor = config.build(y_train)

    assert isinstance(predictor, DummyPredictorModel)

    X_test = torch.randn(10, 10, 100)
    y_pred = predictor(X_test)

    expected = y_train.mode(dim=0)[0].float()
    assert y_pred.shape == (10, 4)
    assert torch.allclose(y_pred, expected)
    assert torch.allclose(y_pred, torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_dummy_predictor_stratified_multilabel():
    # Per-class prevalences: [4/5, 2/5, 3/5, 1/5] = [0.8, 0.4, 0.6, 0.2]
    y_train = torch.tensor(
        [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    expected_probs = torch.tensor([0.8, 0.4, 0.6, 0.2])

    config = DummyPredictor(mode="stratified_multilabel", random_state=0)
    predictor = config.build(y_train)

    assert isinstance(predictor, DummyPredictorModel)
    assert torch.allclose(predictor.probs, expected_probs)

    # Output is 0/1 with the right shape, and empirical means across a large
    # batch converge to the stored probabilities.
    n_samples = 50_000
    X_test = torch.randn(n_samples, 10, 100)
    y_pred = predictor(X_test)
    assert y_pred.shape == (n_samples, 4)
    assert set(torch.unique(y_pred).tolist()) <= {0.0, 1.0}
    assert torch.allclose(y_pred.mean(dim=0), expected_probs, atol=1e-2)

    # Unlike the constant modes, two forward passes are not bit-identical.
    first = predictor(X_test[:100])
    second = predictor(X_test[:100])
    assert not torch.equal(first, second)


def test_dummy_predictor_stratified_random_state():
    y_train = torch.tensor(
        [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    X_test = torch.randn(128, 10, 100)

    pred_a = DummyPredictor(mode="stratified_multilabel", random_state=0).build(y_train)
    pred_b = DummyPredictor(mode="stratified_multilabel", random_state=0).build(y_train)
    pred_c = DummyPredictor(mode="stratified_multilabel", random_state=1).build(y_train)

    out_a = pred_a(X_test)
    out_b = pred_b(X_test)
    out_c = pred_c(X_test)

    assert torch.equal(out_a, out_b)
    assert not torch.equal(out_a, out_c)


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
            "stratified_multilabel",
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
def test_dummy_predictor_auto_mode(y_train, expected_mode):
    """Verify that mode='auto' resolves identically to the expected explicit mode."""
    # Use a shared seed so the stratified auto/explicit outputs match bit-for-bit.
    auto_predictor = DummyPredictor(mode="auto", random_state=0).build(y_train)
    explicit_predictor = DummyPredictor(mode=expected_mode, random_state=0).build(y_train)

    X_test = torch.randn(5, 2, 10)
    assert torch.allclose(auto_predictor(X_test), explicit_predictor(X_test))


def test_dummy_predictor_auto_mode_unsupported_dtype():
    y_train = torch.tensor([True, False, True])
    with pytest.raises(ValueError, match="Unsupported dtype"):
        DummyPredictor(mode="auto").build(y_train)
