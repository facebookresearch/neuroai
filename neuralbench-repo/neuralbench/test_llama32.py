"""Checkpoint behavior for models with a separately loaded frozen Llama."""

import logging

import torch
from torch import nn

from neuralbench.experiment_config import build_experiment_configs
from neuralbench.pl_module import BrainModule
from neuralbench.plots._constants import MEEG_CLASSIC_DISPLAY, MODEL_DISPLAY_NAMES
from neuralbench.registry import (
    CLASSIC_MODELS,
    DEVICE_CLASSIC_MODELS,
    TASKS,
    _expand_models,
)
from neuralbench.utils import load_checkpoint
from neuraltrain.models.llama32_eeg import _parse_multilabel_json


class _AdapterModel(nn.Module):
    omit_frozen_backbone_from_checkpoint = True

    def __init__(self):
        super().__init__()
        self.llama = nn.Linear(3, 3)
        self.llama.requires_grad_(False)
        self.eeg_projector = nn.Linear(3, 3)

    def forward(self, x):
        return self.eeg_projector(x)


def test_llama_checkpoint_saves_adapter_and_strictly_restores_it():
    module = BrainModule(
        model=_AdapterModel(),
        loss=nn.CrossEntropyLoss(),
        metrics={},
        lightning_optimizer_config=None,
    )
    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    assert not any(k.startswith("model.llama.") for k in checkpoint["state_dict"])
    assert any(k.startswith("model.eeg_projector.") for k in checkpoint["state_dict"])
    restored = BrainModule(
        model=_AdapterModel(),
        loss=nn.CrossEntropyLoss(),
        metrics={},
        lightning_optimizer_config=None,
    )
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    assert torch.equal(
        restored.model.eeg_projector.weight, module.model.eeg_projector.weight
    )


def test_llama_adapter_checkpoint_loads_through_standard_model_path(tmp_path):
    module = BrainModule(
        model=_AdapterModel(),
        loss=nn.CrossEntropyLoss(),
        metrics={},
        lightning_optimizer_config=None,
    )
    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    path = tmp_path / "adapter.ckpt"
    torch.save(checkpoint, path)

    restored = load_checkpoint(_AdapterModel(), path, logging.getLogger(__name__))
    assert torch.equal(restored.eeg_projector.weight, module.model.eeg_projector.weight)


def test_llama32_config_assembles_for_every_eeg_task():
    configs = build_experiment_configs("eeg", "all", model="llama32", debug=True)
    assert {config["task_name"] for config in configs} == set(TASKS["eeg"])
    assert all(config["brain_model_config"]["name"] == "Llama32EEG" for config in configs)
    assert all(config["data"]["batch_size"] == 2 for config in configs)
    assert all(config["trainer_config"]["limit_test_batches"] == 5 for config in configs)
    variant = build_experiment_configs(
        "eeg", "mental_arithmetic", model="llama32", dataset="shin2017b", debug=True
    )
    assert variant


def test_llama32_is_in_eeg_baseline_selection_and_result_display():
    assert "llama32" in CLASSIC_MODELS
    assert "llama32" in _expand_models("all_classic", device="eeg")
    assert "llama32" in _expand_models("all", device="eeg")
    assert "llama32" not in DEVICE_CLASSIC_MODELS["meg"]
    assert "llama32" not in _expand_models("all", device="meg")
    display = MODEL_DISPLAY_NAMES["Llama32EEG"]
    assert display == "Llama 3.2 3B + EEG"
    assert display in MEEG_CLASSIC_DISPLAY


def test_generative_tuev_config_uses_existing_multilabel_evaluation():
    config = build_experiment_configs(
        "eeg", "clinical_event", model="llama32_generate", debug=True
    )[0]
    assert config["brain_model_config"]["name"] == "Llama32EEGGenerate"
    assert config["loss"]["name"] == "BCEWithLogitsLoss"
    assert config["brain_model_output_size"] == 6
    assert config["trainer_config"]["limit_test_batches"] == 5
    assert config["save_test_predictions"] is True
    assert "llama32_generate" not in _expand_models("all", device="eeg")
    assert MODEL_DISPLAY_NAMES["Llama32EEGGenerate"] != MODEL_DISPLAY_NAMES["Llama32EEG"]


def test_generated_tuev_event_list_is_strictly_multilabel():
    labels = ["artf", "bckg", "eyem", "gped", "pled", "spsw"]
    assert _parse_multilabel_json('["spsw", "artf"]', labels) == (
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        True,
    )
    assert _parse_multilabel_json("[]", labels) == ([0.0] * 6, True)
    for invalid in ('["artf", "artf"]', '["unknown"]', "artf", '["artf"] extra'):
        assert _parse_multilabel_json(invalid, labels) == ([0.0] * 6, False)
