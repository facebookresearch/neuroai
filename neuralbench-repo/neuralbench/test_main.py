# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from types import SimpleNamespace

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch
from exca import TaskInfra
from exca.cachedict import CacheDict
from torch import nn
from torch.utils.data import DataLoader

from neuraltrain.augmentations import BandRotationConfig
from neuraltrain.losses import BaseLoss
from neuraltrain.metrics.metrics import GroupedMetric
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.optimizers import LightningOptimizer

from .callbacks import WindowPredictionCollector
from .data import Data
from .main import Experiment
from .pl_module import BrainModule
from .utils import TrainerConfig


class _DummyLoss:
    def build(self, **kwargs) -> nn.Module:
        return nn.MSELoss()


class _DummyBrainModule:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def _sequence_module(monkeypatch, metrics: dict[str, tp.Any]) -> BrainModule:
    module = BrainModule(
        model=nn.Identity(),
        loss=nn.L1Loss(),
        metrics=metrics,
        lightning_optimizer_config=tp.cast(LightningOptimizer, object()),
    )
    module._trainer = SimpleNamespace(world_size=1)  # type: ignore[assignment]
    monkeypatch.setattr(module, "log", lambda *args, **kwargs: None)
    return module


@pytest.mark.parametrize("grouped", [False, True])
def test_unlabelled_sequence_frames_are_masked(monkeypatch, grouped: bool) -> None:
    metrics: dict[str, tp.Any] = {}
    if grouped:
        metrics["val/mae"] = GroupedMetric(metric_name="MeanAbsoluteError")
    module = _sequence_module(monkeypatch, metrics)
    nan = float("nan")
    batch = SimpleNamespace(
        data={
            # Identity model, so the prediction is time-major (B, T, C) while
            # the extracted target is channel-major (B, C, T).
            "neuro": torch.tensor([[[2.0, 2.0], [999.0, 999.0], [1.0, 1.0]]]),
            "target": torch.tensor([[[1.0, nan, 3.0], [1.0, nan, 3.0]]]),
            # The loader emits (B, 1), not (B,).
            "subject_id": torch.tensor([[0]]),
        }
    )

    loss, prediction, _ = module._run_step(
        tp.cast(tp.Any, batch), step_name="val", batch_idx=0
    )

    assert loss.item() == 1.5, "the unlabelled frame contributes nothing"
    assert prediction[0].isnan().tolist() == [False, False, True, True, False, False]


def test_sequence_target_wider_than_prediction_is_reported(monkeypatch) -> None:
    module = _sequence_module(monkeypatch, {})
    batch = SimpleNamespace(
        data={
            # Identity model, so the prediction is (B, T=3, C=2) against a
            # target whose channel axis is 3 rather than 2.
            "neuro": torch.zeros(1, 3, 2),
            "target": torch.zeros(1, 3, 3),
            "subject_id": torch.tensor([[0]]),
        }
    )

    with pytest.raises(ValueError, match="2 outputs per frame but its target 3"):
        module._run_step(tp.cast(tp.Any, batch), step_name="val", batch_idx=0)


@pytest.mark.parametrize("training", [True, False])
def test_augmentation_rolls_the_channel_axis_in_training_only(training: bool) -> None:
    module = BrainModule(
        model=nn.Identity(),
        loss=nn.L1Loss(),
        metrics={},
        lightning_optimizer_config=tp.cast(LightningOptimizer, object()),
        augmentation=BandRotationConfig(
            probability=1.0, num_bands=1, electrodes_per_band=4, band_offsets=(1,)
        ).build(),
    )
    module.train(training)
    # (B=1, C=4, T=1): a roll of the channel axis reorders this column.
    neuro = torch.arange(4, dtype=torch.float32)[None, :, None]

    out = module.model_forward(tp.cast(tp.Any, SimpleNamespace(data={"neuro": neuro})))

    expected = [3.0, 0.0, 1.0, 2.0] if training else [0.0, 1.0, 2.0, 3.0]
    assert out[0, :, 0].tolist() == expected, "augmentation ran on the wrong axis/split"


@pytest.mark.parametrize("global_rank", [0, 1])
def test_only_rank_zero_deletes_the_checkpoint(tmp_path, global_rank: int) -> None:
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.touch()
    # Only the two flags ``_cleanup`` reads; the rest of the config is unused.
    experiment = Experiment.model_construct(  # type: ignore[call-arg]
        delete_checkpoints_on_exit=True, eval_only=False
    )
    trainer = SimpleNamespace(
        global_rank=global_rank,
        checkpoint_callback=SimpleNamespace(best_model_path=str(checkpoint)),
    )

    experiment._cleanup(tp.cast(tp.Any, trainer))

    assert checkpoint.exists() == (global_rank != 0), (
        "a non-zero rank must leave the checkpoint for rank zero to test with"
    )


def _make_experiment_with_capturing_build(
    monkeypatch, seed: int
) -> tuple[Experiment, list[torch.Tensor]]:
    """Build a minimal ``Experiment`` whose ``build_brain_model`` captures the
    first ``torch.rand(4)`` it draws after each ``prepare_pl_module`` call."""
    build_draws: list[torch.Tensor] = []

    def fake_build_brain_model(**kwargs):
        del kwargs
        build_draws.append(torch.rand(4))
        _ = torch.rand(11)
        return nn.Identity(), 0, 0

    monkeypatch.setattr("neuralbench.main.build_brain_model", fake_build_brain_model)
    monkeypatch.setattr("neuralbench.main.BrainModule", _DummyBrainModule)

    experiment = Experiment.model_construct(
        brain_model_config=tp.cast(BaseModelConfig, object()),
        downstream_model_wrapper=None,
        pretrained_weights_fname=None,
        data=tp.cast(Data, object()),
        target_scaler=None,
        compute_class_weights=False,
        trainer_config=tp.cast(TrainerConfig, object()),
        loss=tp.cast(BaseLoss, _DummyLoss()),
        lightning_optimizer_config=tp.cast(LightningOptimizer, object()),
        metrics=[],
        test_full_metrics=[],
        test_full_retrieval_metrics=[],
        seed=seed,
    )
    return experiment, build_draws


def test_prepare_pl_module_seeds_before_and_after_model_build(monkeypatch) -> None:
    """Model construction should ignore prior RNG usage and reset training RNG."""
    seed = 123
    experiment, build_draws = _make_experiment_with_capturing_build(monkeypatch, seed)
    post_draws: list[torch.Tensor] = []

    for pre_draws in (3, 17):
        torch.manual_seed(999)
        _ = torch.rand(pre_draws)
        experiment.prepare_pl_module(train_loader=tp.cast(DataLoader, object()))
        post_draws.append(torch.rand(4))

    pl.seed_everything(seed)
    expected = torch.rand(4)

    assert torch.allclose(build_draws[0], expected)
    assert torch.allclose(build_draws[1], expected)
    assert torch.allclose(post_draws[0], expected)
    assert torch.allclose(post_draws[1], expected)


def test_prepare_pl_module_different_seeds_diverge(monkeypatch) -> None:
    """Different ``Experiment.seed`` values should drive different model-build RNGs.

    Companion to :func:`test_prepare_pl_module_seeds_before_and_after_model_build`,
    which proves the same-seed determinism direction.  This one locks in that
    changing the seed actually changes the model-construction stream, catching
    future regressions where ``prepare_pl_module`` would accidentally hardcode
    a constant seed.
    """
    experiment_a, draws_a = _make_experiment_with_capturing_build(monkeypatch, seed=7)
    experiment_a.prepare_pl_module(train_loader=tp.cast(DataLoader, object()))

    experiment_b, draws_b = _make_experiment_with_capturing_build(monkeypatch, seed=8)
    experiment_b.prepare_pl_module(train_loader=tp.cast(DataLoader, object()))

    assert not torch.allclose(draws_a[0], draws_b[0])


def test_run_seeds_before_preparing_dataloaders(monkeypatch) -> None:
    """``Experiment.run()`` calls ``pl.seed_everything(self.seed, workers=True)``
    as its first action, before ``setup_run`` and ``data.prepare``.

    Data-side determinism (shuffle, weighted sampler, worker RNGs) is driven
    by ``Data.seed`` via explicit ``torch.Generator``s, but this call still
    matters for (a) the ``Data.seed=None`` fallback path, where the shuffle
    inherits the global torch RNG, and (b) any RNG-consuming code in
    ``setup_run`` / ``data.prepare`` that runs before ``prepare_pl_module``
    reseeds for model build.
    """
    events: list[str] = []
    seed_calls: list[tuple[int | None, bool]] = []

    class _DummyData:
        def prepare(self) -> dict[str, object]:
            events.append("data.prepare")
            return {"train": object(), "val": object(), "test": object()}

    def fake_seed_everything(
        seed: int | None = None, workers: bool = False, verbose: bool = True
    ) -> int:
        del verbose
        seed_calls.append((seed, workers))
        events.append("seed")
        return 0 if seed is None else seed

    def fake_setup_run(self) -> None:
        del self
        events.append("setup_run")

    def fake_setup_trainer(self):
        events.append("setup_trainer")
        return SimpleNamespace(global_rank=1)

    def fake_prepare_pl_module(self, train_loader, val_loader=None) -> None:
        del self, train_loader, val_loader
        events.append("prepare_pl_module")

    def fake_cleanup(self, trainer) -> None:
        del self, trainer
        events.append("cleanup")

    monkeypatch.setattr("neuralbench.main.pl.seed_everything", fake_seed_everything)
    monkeypatch.setattr(Experiment, "setup_run", fake_setup_run)
    monkeypatch.setattr(Experiment, "setup_trainer", fake_setup_trainer)
    monkeypatch.setattr(Experiment, "prepare_pl_module", fake_prepare_pl_module)
    monkeypatch.setattr(Experiment, "_cleanup", fake_cleanup)

    seed = 456
    experiment = Experiment.model_construct(
        data=tp.cast(Data, _DummyData()),
        brain_model_config=tp.cast(BaseModelConfig, object()),
        trainer_config=tp.cast(TrainerConfig, object()),
        loss=tp.cast(BaseLoss, _DummyLoss()),
        lightning_optimizer_config=tp.cast(LightningOptimizer, object()),
        metrics=[],
        eval_only=True,
        infra=TaskInfra(version="1", gpus_per_node=0),
        seed=seed,
    )

    result = experiment.run()

    assert seed_calls == [(seed, True)]
    assert events[:4] == ["seed", "setup_run", "data.prepare", "setup_trainer"]
    assert events[-2:] == ["prepare_pl_module", "cleanup"]
    assert result["n_total_params"] is None
    assert result["n_trainable_params"] is None


_PREDS: dict[str, tp.Any] = {
    "metadata": pd.DataFrame(
        {"timeline": ["rec0", "rec1", "rec2"], "batch_idx": [0, 0, 1]}
    ),
    "y_true": np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]),
    "y_pred": np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]),
}


def _make_eval_experiment(save_test_predictions: bool) -> Experiment:
    class _DummyData:
        def prepare(self) -> dict[str, object]:
            return {"train": object(), "val": object(), "test": object()}

    return Experiment.model_construct(
        data=tp.cast(Data, _DummyData()),
        brain_model_config=tp.cast(BaseModelConfig, object()),
        trainer_config=tp.cast(TrainerConfig, object()),
        loss=tp.cast(BaseLoss, _DummyLoss()),
        lightning_optimizer_config=tp.cast(LightningOptimizer, object()),
        metrics=[],
        eval_only=True,
        infra=TaskInfra(version="1", gpus_per_node=0),
        save_test_predictions=save_test_predictions,
    )


def _write_streamed_predictions(folder: str) -> None:
    """Mimic WindowPredictionCollector's on-disk layout: per-batch array chunks
    plus a single metadata table."""
    cache: CacheDict = CacheDict(folder=folder)
    with cache.write():
        # Two "batches": rows [0:2] then [2:3], matching ``batch_idx``.
        for chunk, sl in enumerate([slice(0, 2), slice(2, 3)]):
            tag = f"{chunk:08d}"
            cache[WindowPredictionCollector._Y_PRED_PREFIX + tag] = _PREDS["y_pred"][sl]
            cache[WindowPredictionCollector._Y_TRUE_PREFIX + tag] = _PREDS["y_true"][sl]
        cache[WindowPredictionCollector._METADATA_KEY] = _PREDS["metadata"]


def test_test_predictions_roundtrip(monkeypatch, tmp_path) -> None:
    """Streamed per-batch chunks round-trip through the accessor: arrays are
    concatenated in order and metadata is returned as a DataFrame."""
    experiment = _make_eval_experiment(save_test_predictions=True)
    _write_streamed_predictions(str(tmp_path / Experiment._TEST_PREDICTIONS_DIR))
    monkeypatch.setattr(type(experiment.infra), "uid_folder", lambda self: tmp_path)

    out = experiment.test_predictions()

    pd.testing.assert_frame_equal(out["metadata"], _PREDS["metadata"])
    assert np.array_equal(out["y_true"], _PREDS["y_true"])
    assert np.array_equal(out["y_pred"], _PREDS["y_pred"])


def test_test_predictions_accessor_errors_when_unsaved(monkeypatch, tmp_path) -> None:
    """The accessor errors when the prediction folder is empty (flag was off)."""
    experiment = _make_eval_experiment(save_test_predictions=False)
    monkeypatch.setattr(type(experiment.infra), "uid_folder", lambda self: tmp_path)
    with pytest.raises(ValueError, match="save_test_predictions=True"):
        experiment.test_predictions()
