# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pydantic
import pytest
from exca import ConfDict

from neuraltrain.optimizers import LightningOptimizer

from .evaluate import _external_config, check_model, evaluate_model
from .external import ExternalModel
from .test_external import AdaptiveFm, FixedWidth, NoPositions, NotBatchFirst


def test_instance_is_serialized(
    patch_config: Callable[..., None], tmp_path: Path
) -> None:
    patch_config(CACHE_DIR=str(tmp_path))
    model = AdaptiveFm()
    config = _external_config(model)
    assert config.pickle_path == str(tmp_path / "external_models" / f"{config.digest}.pt")
    # Serializing is not byte-reproducible, so the digest has to come from the
    # model itself; a call that reused nothing would resubmit everything.
    assert _external_config(model).digest == config.digest
    assert _external_config(AdaptiveFm()).digest != config.digest  # other weights


def test_an_unusable_model_is_rejected_before_anything_is_submitted(
    patch_config: Callable[..., None], tmp_path: Path
) -> None:
    patch_config(CACHE_DIR=str(tmp_path))
    with pytest.raises(TypeError, match="Build your model first"):
        _external_config(AdaptiveFm)  # type: ignore[arg-type]
    # Otherwise the forward contract only fails once a worker loads the pickle.
    with pytest.raises(ValueError, match="does not accept 'channel_positions'"):
        _external_config(NoPositions())


def _stub_aggregator(
    monkeypatch: pytest.MonkeyPatch,
    phases: list[list[ConfDict]],
    status: list[str] | None = None,
) -> None:
    """Record the experiments of each phase instead of running them."""

    class _FakeAggregator:
        def __init__(self, experiments: list[ConfDict], debug: bool = False) -> None:
            phases.append(experiments)
            if status is None:
                self.experiments = experiments
            else:  # only the infra status is read from the warm-up phase
                infra = SimpleNamespace(status=lambda: status[0])
                self.experiments = [SimpleNamespace(infra=infra)]  # type: ignore[list-item]

        def prepare(self) -> None:
            pass

        def collect(self, cached_only: bool = True) -> list[dict[str, tp.Any]]:
            return []

    monkeypatch.setattr("neuralbench.main.BenchmarkAggregator", _FakeAggregator)


def _assembled(monkeypatch: pytest.MonkeyPatch, **kwargs: tp.Any) -> list[ConfDict]:
    """Run against a stubbed aggregator, returning the configs it was handed."""
    phases: list[list[ConfDict]] = []
    _stub_aggregator(monkeypatch, phases)
    kwargs.setdefault("dataset", "schalk2004bci2000")
    kwargs.setdefault("download", False)
    kwargs.setdefault("prepare", False)
    evaluate_model(AdaptiveFm(), "eeg", "motor_imagery", **kwargs)
    assert len(phases) == 1, "expected only the run phase"
    assert phases[0]
    return phases[0]


def test_assembles_external_model_configs(
    monkeypatch: pytest.MonkeyPatch, patch_config: Callable[..., None], tmp_path: Path
) -> None:
    patch_config(SLURM_PARTITION="dummy", CACHE_DIR=str(tmp_path))
    configs = _assembled(
        monkeypatch,
        name="my-fm",
        overrides={
            "data.neuro.frequency": 200.0,
            "lightning_optimizer_config.optimizer.lr": 1e-4,
        },
    )

    for cfg in configs:
        flat = cfg.flat()
        assert flat["brain_model_name"] == "my-fm"
        assert flat["brain_model_config.name"] == "ExternalModel"
        assert flat["brain_model_config.pickle_path"].endswith(".pt")
        # Overrides beat the task config they are layered onto.
        assert flat["data.neuro.frequency"] == 200.0
        assert flat["lightning_optimizer_config.optimizer.lr"] == 1e-4
        # A fixed instance has a fixed output width, so the head comes from the
        # wrapper for one model to span tasks of differing class counts.
        assert flat["downstream_model_wrapper.aggregation"] == "mean"
        assert flat["downstream_model_wrapper.probe_config"] == "linear"


def test_runs_in_process_by_default(
    monkeypatch: pytest.MonkeyPatch, patch_config: Callable[..., None], tmp_path: Path
) -> None:
    patch_config(SLURM_PARTITION="dummy", CLUSTER="slurm", CACHE_DIR=str(tmp_path))
    for cfg in _assembled(monkeypatch):
        assert cfg.flat()["infra.cluster"] is None
        assert cfg.flat()["data.neuro.infra.cluster"] is None

    for cfg in _assembled(monkeypatch, cluster="auto"):
        assert cfg.flat()["infra.cluster"] == "auto"
        assert cfg.flat()["data.neuro.infra.cluster"] == "auto"


def test_overrides_survive_the_debug_overlay(
    monkeypatch: pytest.MonkeyPatch, patch_config: Callable[..., None], tmp_path: Path
) -> None:
    # The overlays run after a model config is merged, so an override has to be
    # applied later still or debug mode would silently revert it.
    patch_config(SLURM_PARTITION="dummy", CACHE_DIR=str(tmp_path))
    for cfg in _assembled(
        monkeypatch, debug=True, overrides={"trainer_config.n_epochs": 7}
    ):
        assert cfg.flat()["trainer_config.n_epochs"] == 7


def test_phases_run_in_order(
    monkeypatch: pytest.MonkeyPatch, patch_config: Callable[..., None], tmp_path: Path
) -> None:
    patch_config(SLURM_PARTITION="dummy", CACHE_DIR=str(tmp_path))
    phases: list[list[ConfDict]] = []
    downloaded: list[str] = []
    _stub_aggregator(monkeypatch, phases)
    monkeypatch.setattr(
        "neuralbench.evaluate._download_dataset",
        lambda config: downloaded.append(config["task_name"]),
    )
    evaluate_model(
        AdaptiveFm(), "eeg", "motor_imagery", dataset="schalk2004bci2000", prepare=True
    )
    assert downloaded == ["motor_imagery"], "downloads once per study, before running"
    # A prepare pass is cut down to a single epoch; the real one is not.
    epochs = [phase[0].flat()["trainer_config.n_epochs"] for phase in phases]
    assert epochs[0] == 1 and epochs[1] != 1, "prepare pass, then the real one"


def test_a_queued_cache_warm_up_holds_the_experiments_back(
    monkeypatch: pytest.MonkeyPatch, patch_config: Callable[..., None], tmp_path: Path
) -> None:
    # On a cluster the warm-up is submitted, not run, so going straight on would
    # have every experiment race it and recompute the caches once per seed.
    patch_config(SLURM_PARTITION="dummy", CACHE_DIR=str(tmp_path))
    phases: list[list[ConfDict]] = []
    status = ["running"]
    _stub_aggregator(monkeypatch, phases, status)
    call = partial(
        evaluate_model,
        AdaptiveFm(),
        "eeg",
        "motor_imagery",
        dataset="schalk2004bci2000",
        download=False,
        cluster="slurm",
    )
    assert call().empty
    assert len(phases) == 1, "only the warm-up should have been submitted"

    status[0] = "completed"
    call()
    assert len(phases) == 3, "a warm cache no longer holds the experiments back"


def test_assembled_config_validates_against_the_real_models(
    monkeypatch: pytest.MonkeyPatch, patch_config: Callable[..., None], tmp_path: Path
) -> None:
    # Assembly only merges dicts, so an override naming a field that does not
    # exist is caught by pydantic rather than here; check the merged config
    # against the models that will consume it.
    patch_config(SLURM_PARTITION="dummy", CACHE_DIR=str(tmp_path))
    configs = _assembled(
        monkeypatch, overrides={"lightning_optimizer_config.optimizer.lr": 1e-4}
    )
    optimizer = LightningOptimizer(**configs[0]["lightning_optimizer_config"])
    # Via the dump because `optimizer` is typed as the BaseOptimizer union.
    assert optimizer.model_dump()["optimizer"]["lr"] == 1e-4
    model_config = ExternalModel(**configs[0]["brain_model_config"])
    assert model_config.pickle_path.endswith(".pt")

    bad = _assembled(monkeypatch, overrides={"lightning_optimizer_config.lr": 1e-4})
    with pytest.raises(pydantic.ValidationError, match="lr"):
        LightningOptimizer(**bad[0]["lightning_optimizer_config"])


def test_check_model_accepts_an_adaptive_model() -> None:
    model = AdaptiveFm()
    model.train()
    rows = check_model(model, "eeg", "motor_imagery")
    assert len(rows) == 4, "expected one row per probed channel count"
    assert set(rows["status"]) == {"ok"}
    # Pooled over channels, so the shape is the same at every width.
    assert set(rows["output_shape"]) == {(2, 4)}
    assert model.training, "probing left the caller's model in eval mode"


def test_check_model_reports_what_it_could_not_run() -> None:
    by_width = check_model(FixedWidth(), "eeg", "motor_imagery").set_index(
        "n_spatial_locations"
    )["status"]
    assert by_width[64] == "ok"
    assert by_width[19].startswith("forward failed:")

    rows = check_model(NotBatchFirst(), "eeg", "motor_imagery")
    assert rows["status"].str.startswith("output").all()

    # A contract violation rather than a per-task shape problem, so it raises
    # instead of filling every row with the same message.
    with pytest.raises(ValueError, match="does not accept 'channel_positions'"):
        check_model(NoPositions(), "eeg", "motor_imagery")


def test_check_model_probes_the_shapes_the_run_will_use() -> None:
    # motor_imagery uses a 4 s window, so an override of the sampling rate
    # changes the width the model will see.
    at_200hz = check_model(
        AdaptiveFm(), "eeg", "motor_imagery", overrides={"data.neuro.frequency": 200.0}
    )
    assert at_200hz["n_temporal_samples"].eq(800).all()

    # cvep's dataset variants cut the task's 4 s window to 2 s and 1 s, so
    # probing the task config alone would miss the shapes the run will use.
    rows = check_model(AdaptiveFm(), "eeg", "cvep", dataset="all")
    assert set(rows["n_temporal_samples"]) == {480, 240, 120}
    assert set(rows["status"]) == {"ok"}
