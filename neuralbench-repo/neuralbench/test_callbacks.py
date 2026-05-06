# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

from .callbacks import PlotRegressionScatter, RecordingLevelEval

matplotlib.use("Agg")


class MockSegment:
    """Mock segment with events containing timeline information."""

    def __init__(self, timeline: str):
        self.events = pd.DataFrame({"timeline": [timeline]})


class MockBatch:
    """Mock batch containing segments."""

    def __init__(self, segments: list[MockSegment]):
        self.segments = segments


@pytest.fixture
def mock_trainer(mocker):
    """Create a mock trainer."""
    return mocker.Mock()


@pytest.fixture
def mock_pl_module(mocker):
    """Create a mock PyTorch Lightning module with test_full_metrics."""
    pl_module = mocker.Mock()
    # Create a mock metric instead of real one to avoid actual computation
    mock_metric = mocker.Mock()
    mock_metric.to = mocker.Mock(return_value=mock_metric)
    mock_metric.update = mocker.Mock()
    pl_module.test_full_metrics = {"test/acc": mock_metric}
    pl_module.log = mocker.Mock()
    return pl_module


@pytest.fixture
def mock_pl_module_3class(mocker):
    """Create a mock PyTorch Lightning module with 3-class metrics."""
    pl_module = mocker.Mock()
    # Create a mock metric instead of real one
    mock_metric = mocker.Mock()
    mock_metric.to = mocker.Mock(return_value=mock_metric)
    mock_metric.update = mocker.Mock()
    pl_module.test_full_metrics = {"test/acc": mock_metric}
    pl_module.log = mocker.Mock()
    return pl_module


def test_binary_single_window_per_recording(mock_trainer, mock_pl_module):
    """Test binary classification with one window per recording."""
    callback = RecordingLevelEval()
    callback.on_test_epoch_start(mock_trainer, mock_pl_module)

    # Batch with 3 recordings, 1 window each
    # Recording 1: true=0, pred logits favor class 1
    # Recording 2: true=1, pred logits favor class 0
    # Recording 3: true=1, pred logits favor class 1
    batch = MockBatch(
        [
            MockSegment("rec1"),
            MockSegment("rec2"),
            MockSegment("rec3"),
        ]
    )
    outputs = (
        torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.3, 0.7]]),  # logits
        torch.tensor([0, 1, 1]),  # true labels
    )

    callback.on_test_batch_end(mock_trainer, mock_pl_module, outputs, batch, batch_idx=0)

    # Check internal state
    assert len(callback.test_outputs) == 3
    assert callback.test_outputs["rec1"]["y_true"] == 0
    assert callback.test_outputs["rec2"]["y_true"] == 1
    assert callback.test_outputs["rec3"]["y_true"] == 1
    # Check class counts: rec1 pred class 1, rec2 pred class 0, rec3 pred class 1
    assert callback.test_outputs["rec1"]["class_counts"] == [0, 1]
    assert callback.test_outputs["rec2"]["class_counts"] == [1, 0]
    assert callback.test_outputs["rec3"]["class_counts"] == [0, 1]

    # Run epoch end
    callback.on_test_epoch_end(mock_trainer, mock_pl_module)

    # Verify metric.update was called
    mock_pl_module.test_full_metrics["test/acc"].update.assert_called_once()
    # Get the call arguments
    call_args = mock_pl_module.test_full_metrics["test/acc"].update.call_args[0]
    y_pred_probs = call_args[0]
    y_true = call_args[1]

    # Check shapes
    assert y_pred_probs.shape == (3, 2)
    assert y_true.shape == (3,)

    # Check true labels (convert to same dtype for comparison)
    assert torch.equal(y_true.long(), torch.tensor([0, 1, 1]))


def test_binary_multiple_windows_per_recording(mock_trainer, mock_pl_module):
    """Test binary classification with multiple windows per recording - key test for averaging."""
    callback = RecordingLevelEval()
    callback.on_test_epoch_start(mock_trainer, mock_pl_module)

    # Recording A has 3 windows: pred classes [1, 1, 0]
    # Recording B has 2 windows: pred classes [0, 1]
    batch = MockBatch(
        [
            MockSegment("recA"),
            MockSegment("recA"),
            MockSegment("recA"),
            MockSegment("recB"),
            MockSegment("recB"),
        ]
    )
    outputs = (
        torch.tensor(
            [
                [0.2, 0.8],  # recA window 1 -> pred class 1
                [0.3, 0.7],  # recA window 2 -> pred class 1
                [0.9, 0.1],  # recA window 3 -> pred class 0
                [0.8, 0.2],  # recB window 1 -> pred class 0
                [0.4, 0.6],  # recB window 2 -> pred class 1
            ]
        ),
        torch.tensor([1, 1, 1, 0, 0]),  # true labels
    )

    callback.on_test_batch_end(mock_trainer, mock_pl_module, outputs, batch, batch_idx=0)

    # Check internal state before epoch end
    assert len(callback.test_outputs) == 2
    # recA: pred classes [1, 1, 0] -> class_counts = [1, 2]
    assert callback.test_outputs["recA"]["class_counts"] == [1, 2]
    # recB: pred classes [0, 1] -> class_counts = [1, 1]
    assert callback.test_outputs["recB"]["class_counts"] == [1, 1]

    # Run epoch end
    callback.on_test_epoch_end(mock_trainer, mock_pl_module)

    # Get the call arguments
    call_args = mock_pl_module.test_full_metrics["test/acc"].update.call_args[0]
    y_pred_probs = call_args[0]

    # Check shapes
    assert y_pred_probs.shape == (2, 2)

    # Fixed: all windows are counted:
    # recA: all windows [1, 1, 0] -> counts [1, 2] -> prob = [1/3, 2/3]
    # recB: all windows [0, 1] -> counts [1, 1] -> prob = [1/2, 1/2]
    assert torch.allclose(
        y_pred_probs[0], torch.tensor([1 / 3, 2 / 3], dtype=torch.float64)
    )
    assert torch.allclose(y_pred_probs[1], torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_binary_batch_processing_order(mock_trainer, mock_pl_module):
    """Test that batch order doesn't matter - windows from same recording are aggregated."""
    callback = RecordingLevelEval()
    callback.on_test_epoch_start(mock_trainer, mock_pl_module)

    # First batch
    batch1 = MockBatch([MockSegment("rec1"), MockSegment("rec2")])
    outputs1 = (
        torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        torch.tensor([1, 0]),
    )
    callback.on_test_batch_end(
        mock_trainer, mock_pl_module, outputs1, batch1, batch_idx=0
    )

    # Second batch - more windows from same recordings
    batch2 = MockBatch([MockSegment("rec1"), MockSegment("rec2")])
    outputs2 = (
        torch.tensor([[0.2, 0.8], [0.9, 0.1]]),
        torch.tensor([1, 0]),
    )
    callback.on_test_batch_end(
        mock_trainer, mock_pl_module, outputs2, batch2, batch_idx=1
    )

    # Both recordings should have accumulated counts from both batches
    assert len(callback.test_outputs) == 2
    # rec1: 2 windows both pred class 1 -> counts [0, 2]
    assert callback.test_outputs["rec1"]["class_counts"] == [0, 2]
    # rec2: 2 windows both pred class 0 -> counts [2, 0]
    assert callback.test_outputs["rec2"]["class_counts"] == [2, 0]


def test_multiclass_three_class_single_window(mock_trainer, mock_pl_module_3class):
    """Test 3-class classification with single window per recording."""
    callback = RecordingLevelEval()
    callback.on_test_epoch_start(mock_trainer, mock_pl_module_3class)

    # 3 recordings, 1 window each
    # Recording 1: true=0, pred=class 2
    # Recording 2: true=1, pred=class 1
    # Recording 3: true=2, pred=class 0
    batch = MockBatch(
        [
            MockSegment("rec1"),
            MockSegment("rec2"),
            MockSegment("rec3"),
        ]
    )
    outputs = (
        torch.tensor(
            [
                [0.1, 0.2, 0.7],  # pred class 2
                [0.2, 0.6, 0.2],  # pred class 1
                [0.8, 0.1, 0.1],  # pred class 0
            ]
        ),
        torch.tensor([0, 1, 2]),  # true labels
    )

    callback.on_test_batch_end(
        mock_trainer, mock_pl_module_3class, outputs, batch, batch_idx=0
    )

    # Now should correctly handle 3-class classification
    callback.on_test_epoch_end(mock_trainer, mock_pl_module_3class)

    # Verify the output has correct shape (3 columns for 3-class problem)
    call_args = mock_pl_module_3class.test_full_metrics["test/acc"].update.call_args[0]
    y_pred_probs = call_args[0]
    # Should be (3 recordings, 3 classes)
    assert y_pred_probs.shape == (3, 3), "Should produce 3 columns for 3-class problem"

    # Verify predictions: each recording has 1 window, so probabilities should be one-hot
    # rec1 pred class 2 -> [0, 0, 1]
    # rec2 pred class 1 -> [0, 1, 0]
    # rec3 pred class 0 -> [1, 0, 0]
    assert torch.allclose(
        y_pred_probs[0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    )
    assert torch.allclose(
        y_pred_probs[1], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    )
    assert torch.allclose(
        y_pred_probs[2], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )


def test_multiclass_three_class_multiple_windows(mock_trainer, mock_pl_module_3class):
    """Test 3-class classification with multiple windows per recording."""
    callback = RecordingLevelEval()
    callback.on_test_epoch_start(mock_trainer, mock_pl_module_3class)

    # Recording A has 4 windows: pred classes [0, 1, 1, 2]
    # Should average to probabilities: [0.25, 0.50, 0.25]
    batch = MockBatch(
        [
            MockSegment("recA"),
            MockSegment("recA"),
            MockSegment("recA"),
            MockSegment("recA"),
        ]
    )
    outputs = (
        torch.tensor(
            [
                [0.8, 0.1, 0.1],  # pred class 0
                [0.1, 0.7, 0.2],  # pred class 1
                [0.2, 0.6, 0.2],  # pred class 1
                [0.1, 0.1, 0.8],  # pred class 2
            ]
        ),
        torch.tensor([1, 1, 1, 1]),  # true label (all same for recording)
    )

    callback.on_test_batch_end(
        mock_trainer, mock_pl_module_3class, outputs, batch, batch_idx=0
    )

    # Now should correctly handle 3-class with multiple windows
    callback.on_test_epoch_end(mock_trainer, mock_pl_module_3class)

    # Verify the output has correct shape
    call_args = mock_pl_module_3class.test_full_metrics["test/acc"].update.call_args[0]
    y_pred_probs = call_args[0]
    # Should be (1 recording, 3 classes)
    assert y_pred_probs.shape == (1, 3), "Should produce 3 columns for 3-class problem"

    # Verify averaged predictions: pred classes [0, 1, 1, 2] -> counts [1, 2, 1]
    # Probabilities: [1/4, 2/4, 1/4] = [0.25, 0.5, 0.25]
    expected = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
    assert torch.allclose(y_pred_probs[0], expected)


def test_edge_case_empty_epoch(mock_trainer, mock_pl_module):
    """Test when no batches are processed."""
    callback = RecordingLevelEval()
    callback.on_test_epoch_start(mock_trainer, mock_pl_module)

    # Call epoch end without processing any batches
    with pytest.raises(KeyError, match="class_counts"):
        callback.on_test_epoch_end(mock_trainer, mock_pl_module)


def test_edge_case_all_same_prediction(mock_trainer, mock_pl_module):
    """Test when all windows predict the same class."""
    callback = RecordingLevelEval()
    callback.on_test_epoch_start(mock_trainer, mock_pl_module)

    # All windows predict class 1
    batch = MockBatch([MockSegment("rec1")] * 5)
    outputs = (
        torch.tensor([[0.1, 0.9]] * 5),
        torch.tensor([0] * 5),
    )

    callback.on_test_batch_end(mock_trainer, mock_pl_module, outputs, batch, batch_idx=0)
    callback.on_test_epoch_end(mock_trainer, mock_pl_module)

    # Should complete without error
    mock_pl_module.test_full_metrics["test/acc"].update.assert_called_once()


# ---------------------------------------------------------------------------
# PlotRegressionScatter tests
# ---------------------------------------------------------------------------


class TestPlotRegressionScatterAccumulation:
    """Verify that predictions are accumulated correctly across batches."""

    def test_accumulates_across_batches(self, mock_trainer):
        callback = PlotRegressionScatter()
        callback.on_test_epoch_start(mock_trainer, None)

        outputs_1 = (torch.tensor([1.0, 2.0]), torch.tensor([1.1, 2.1]))
        outputs_2 = (torch.tensor([3.0]), torch.tensor([3.1]))
        callback.on_test_batch_end(mock_trainer, None, outputs_1, None, batch_idx=0)
        callback.on_test_batch_end(mock_trainer, None, outputs_2, None, batch_idx=1)

        assert len(callback.y_preds) == 2
        assert len(callback.y_trues) == 2
        all_preds = torch.cat(callback.y_preds)
        all_trues = torch.cat(callback.y_trues)
        assert torch.equal(all_preds, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.equal(all_trues, torch.tensor([1.1, 2.1, 3.1]))

    def test_epoch_start_resets(self, mock_trainer):
        callback = PlotRegressionScatter()
        callback.y_preds = [torch.tensor([99.0])]
        callback.y_trues = [torch.tensor([99.0])]

        callback.on_test_epoch_start(mock_trainer, None)
        assert callback.y_preds == []
        assert callback.y_trues == []


class TestPlotRegressionScatterFigure:
    """Verify figure creation with known data."""

    def test_creates_figure_with_fit(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        fig = PlotRegressionScatter._create_figure(y_true, y_pred)

        ax = fig.axes[0]
        assert ax.get_xlabel() == "Ground Truth"
        assert ax.get_ylabel() == "Prediction"

        texts = ax.get_children()
        text_objs = [t for t in texts if isinstance(t, matplotlib.text.Text)]
        annotation_text = "".join(t.get_text() for t in text_objs)
        assert "slope" in annotation_text
        assert "intercept" in annotation_text
        assert "R²" in annotation_text

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_perfect_prediction(self):
        y = np.linspace(0, 10, 50)
        fig = PlotRegressionScatter._create_figure(y, y)

        texts = [
            t for t in fig.axes[0].get_children() if isinstance(t, matplotlib.text.Text)
        ]
        annotation = "".join(t.get_text() for t in texts)
        assert "slope = 1.000" in annotation
        assert "R² = 1.000" in annotation

        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotRegressionScatterInverseTransform:
    """Verify inverse-transform logic."""

    def test_inverse_transform(self):
        class FakeScaler:
            _mean = torch.tensor([10.0])
            _scale = torch.tensor([2.0])

        scaled = torch.tensor([0.0, 1.0, -1.0])
        result = PlotRegressionScatter._inverse_transform(scaled, FakeScaler())
        expected = torch.tensor([10.0, 12.0, 8.0])
        assert torch.allclose(result, expected)

    def test_no_inverse_when_scaler_absent(self, mocker):
        """Epoch end with no target_scaler should plot raw values."""
        callback = PlotRegressionScatter()
        callback.y_preds = [torch.tensor([1.0, 2.0])]
        callback.y_trues = [torch.tensor([1.0, 2.0])]

        pl_module = mocker.Mock(spec=[])  # no target_scaler attr
        trainer = mocker.Mock()
        trainer.is_global_zero = True
        trainer.logger = mocker.Mock(spec=["__class__"])
        # Not a WandbLogger, so epoch_end should return early without error
        trainer.logger.__class__ = type("NotWandB", (), {})

        callback.on_test_epoch_end(trainer, pl_module)
