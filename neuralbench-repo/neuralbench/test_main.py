# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from types import SimpleNamespace

import lightning.pytorch as pl
import torch
from exca import TaskInfra
from torch import nn
from torch.utils.data import DataLoader

from neuraltrain.losses import BaseLoss
from neuraltrain.models.base import BaseModelConfig
from neuraltrain.optimizers import LightningOptimizer

from .data import Data
from .main import Experiment
from .utils import TrainerConfig


class _DummyLoss:
    def build(self, **kwargs) -> nn.Module:
        return nn.MSELoss()


class _DummyBrainModule:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


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
