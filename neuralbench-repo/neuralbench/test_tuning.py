# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from exca import ConfDict

from .tuning import OptunaTrial, _storage, load_search_space, tune

optuna = pytest.importorskip("optuna")
pytest.importorskip("optuna_integration")

SPACE = {"lr": {"low": 1e-4, "high": 1e-1, "log": True}, "batch": [16, 32, 64]}


def _run_batch(configs: list[ConfDict]) -> list[dict[str, tp.Any] | None]:
    # batch 16 crashes, batch 64 gets pruned, otherwise the score peaks at lr=1e-2
    out: dict[int, dict[str, tp.Any] | None] = {16: None, 64: {"optuna_pruned": True}}
    return [out.get(c["batch"], {"val/score": -abs(c["lr"] - 1e-2)}) for c in configs]


@pytest.mark.parametrize("mode", ["max", "min"])
def test_tune_batches_resumes_and_tells_states(tmp_path: Path, mode: str) -> None:
    monitor = {"monitor": "val/score", "mode": mode}
    config = ConfDict({"task_name": "t", "trainer_config": monitor})
    (study,) = tune([config], SPACE, _run_batch, 7, tmp_path, batch_size=3)
    assert len(study.trials) == 7 and study.direction.name == mode.upper() + "IMIZE"
    expected = {16: "FAIL", 32: "COMPLETE", 64: "PRUNED"}
    assert all(t.state.name == expected[t.params["batch"]] for t in study.trials)
    # the journal keeps the study: re-running only runs the missing trials
    sizes: list[int] = []

    def counting(configs: list[ConfDict]) -> list[dict[str, tp.Any] | None]:
        sizes.append(len(configs))
        return _run_batch(configs)

    uid = study.user_attrs["uid"]
    (study,) = tune([config], SPACE, counting, 9, tmp_path, batch_size=3)
    assert sizes == [2] and len(study.trials) == 9 and study.user_attrs["uid"] == uid
    # a study recreated from scratch gets a new uid, so its runs are cached apart
    for journal in tmp_path.glob("*.journal"):
        journal.unlink()
    (study,) = tune([config], SPACE, _run_batch, 1, tmp_path)
    assert study.user_attrs["uid"] != uid


class _Module(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))

    def training_step(self, batch: tp.Any, _: int) -> torch.Tensor:
        return (self.w - 1).pow(2).sum()

    def validation_step(self, batch: tp.Any, _: int) -> None:
        self.log("val/score", 0.0)  # always below the completed trial

    def configure_optimizers(self) -> tp.Any:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_pruning_callback_stops_a_bad_run_from_the_journal(tmp_path: Path) -> None:
    path = tmp_path / "s.journal"
    study = optuna.create_study(
        study_name="s", storage=_storage(path), direction="maximize"
    )
    for _ in range(5):  # the pruner waits for 5 completed trials
        good = study.ask()
        for epoch in range(5):
            good.report(1.0, epoch)
        study.tell(good, 1.0)
    # the training job only gets this reference and rebuilds the trial from the journal
    trial_id = study.ask()._trial_id
    ref = OptunaTrial(storage=str(path), study_name="s", study_uid="u", trial_id=trial_id)
    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[ref.pruning_callback("val/score")],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.zeros(2, 1))
    )
    with pytest.raises(optuna.TrialPruned) as error:
        trainer.fit(_Module(), loader, loader)
    assert OptunaTrial.pruned(error.value)


def test_run_benchmark_optuna_gives_each_trial_its_own_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch_config: tp.Callable[..., None]
) -> None:
    from .cli import run_benchmark

    patch_config(WANDB_HOST="", SLURM_PARTITION="dummy", SAVE_DIR=str(tmp_path))
    seen: list[ConfDict] = []

    class _Result:
        def run(self) -> dict[str, float]:
            return {"val/bal_acc": 0.5}

    class _FakeAggregator:
        def __init__(self, experiments: list[tp.Any], **_: tp.Any) -> None:
            seen.extend(ConfDict(c) for c in experiments)
            self.experiments = [_Result() for _ in experiments]

        def prepare(self) -> None:
            pass

    monkeypatch.setattr("neuralbench.main.BenchmarkAggregator", _FakeAggregator)
    assert run_benchmark("eeg", "motor_imagery", optuna_trials=3, optuna_batch=2) == []
    assert len(seen) == 3 and {c["seed"] for c in seen} == {33}
    assert len({c["optuna_trial.trial_id"] for c in seen}) == 3  # cached apart
    assert "data.batch_size" in load_search_space("eeg", "motor_imagery")
