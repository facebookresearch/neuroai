"""Unit tests for the dependency-light NeuroLM NeuralBench adapter."""

import torch

from neuralbench.instruction_models import load_backend
from neuralbench.instruction_models.backends.neurolm import (
    NeuroLMInputAdapter,
    NeuroLMTaskSpec,
    align_task_spec_to_dataset,
    builtin_task_spec,
)
from neuralbench.instruction_models.backends.neurolm.adapter import (
    build_generation_inputs,
)
from neuralbench.instruction_models.backends.neurolm.distributed import (
    DistributedEvalSampler,
)
from neuralbench.instruction_models.backends.neurolm.runner import _target_labels
from neuralbench.instruction_models.backends.neurolm.multitask import (
    select_task_configs,
    task_neuro_overrides,
)


def test_tuab_answer_parser() -> None:
    spec = NeuroLMTaskSpec("tuab", "Question", ("Yes", "No"))
    assert spec.parse_answer(" Yes") == 0
    assert spec.parse_answer("No") == 1
    assert spec.parse_answer("unknown") is None


def test_official_multiple_choice_prompt_continuation_and_parser() -> None:
    spec = builtin_task_spec("clinical_event")
    assert spec.prompt.endswith("Answer: (")
    assert spec.answer_continuation(0) == "A)"
    assert spec.parse_answer("A) <|endoftext|>") == 0


def test_official_tuab_uses_yes_no_answers() -> None:
    spec = builtin_task_spec("pathology")
    assert spec.prompt == "Question: Is this EEG segment abnormal? Answer:"
    assert spec.answers == ("No", "Yes")
    assert spec.answer_continuation(1) == " Yes"


def test_input_adapter_reorders_into_200_sample_tokens() -> None:
    neuro = torch.randn(2, 2, 400)
    adapter = NeuroLMInputAdapter(amplitude_scale=1.0)
    eeg, chans, times, eeg_mask, attention = adapter(neuro, ["Fp1", "Cz"])
    assert eeg.shape == (2, 4, 200)
    assert chans.shape == (2, 4)
    assert times.tolist()[0] == [0, 0, 1, 1]
    assert eeg_mask.shape == (2, 4)
    assert attention.shape == (1, 4, 4)


def test_generation_mask_has_head_dimension_for_official_generate() -> None:
    tensors = [
        torch.randn(1, 2, 200),
        torch.zeros(1, 2, dtype=torch.long),
        torch.zeros(1, 2, dtype=torch.long),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, 2, dtype=torch.bool),
        torch.tensor([50257, 100], dtype=torch.long),
    ]
    result = build_generation_inputs(*tensors)
    assert result["eeg_text_mask"].shape == (1, 1, 4, 4)


def test_distributed_eval_sampler_has_no_padding_duplicates() -> None:
    dataset = list(range(5))
    rank0 = list(DistributedEvalSampler(dataset, rank=0, world_size=2))
    rank1 = list(DistributedEvalSampler(dataset, rank=1, world_size=2))
    assert rank0 == [0, 2, 4]
    assert rank1 == [1, 3]
    assert sorted(rank0 + rank1) == list(range(5))


def test_neurolm_task_specs_cover_the_seven_tasks() -> None:
    cases = [
        ("motor_imagery", "schalk2004bci2000", 4),
        ("pathology", None, 2),
        ("clinical_event", None, 6),
        ("sleep_stage", "alvarez2022haaglanden", 5),
        ("emotion", None, 9),
        ("mental_arithmetic", None, 2),
        ("mental_workload", None, 3),
    ]
    for task, dataset, n_classes in cases:
        assert len(builtin_task_spec(task, dataset).answers) == n_classes


def test_neurolm_control_task_selection_keeps_canonical_order() -> None:
    selected = select_task_configs(("hmc", "tuab"))
    assert [task.name for task in selected] == ["tuab", "hmc"]


def test_choice_parser_uses_prepared_label_order() -> None:
    spec = NeuroLMTaskSpec(
        name="hmc",
        prompt="Question: Answer: (",
        answers=("(B)", "(C)", "(D)", "(E)", "(A)"),
    )
    assert spec.parse_answer("B) <|endoftext|>") == 0
    assert spec.parse_answer("A) <|endoftext|>") == 4


def test_tuev_empty_multihot_maps_to_background() -> None:
    target = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
    assert _target_labels(target, background_index=5).tolist() == [5, 2]


def test_answer_order_follows_neuralbench_label_encoder() -> None:
    class Extractor:
        _label_to_ind = {"mental_arithmetic": 0, "rest": 1}

    class Dataset:
        extractors = {"target": Extractor()}

    spec = align_task_spec_to_dataset(
        builtin_task_spec("mental_arithmetic"), Dataset(), "mental_arithmetic"
    )
    assert spec.answers == ("Yes", "No")


def test_neurolm_is_discovered_as_instruction_backend() -> None:
    backend = load_backend("neurolm")
    assert backend is not None
    assert backend.name == "neurolm"
    assert load_backend("not_a_real_backend") is None


def test_neurolm_null_preprocessing_overrides_are_not_dropped(monkeypatch) -> None:
    import neuralbench
    from neuralbench.instruction_models.backends.neurolm.cli import _prepare_loaders

    captured = {}

    def fake_loader(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(neuralbench, "get_default_dataloaders", fake_loader)
    _prepare_loaders(
        "pathology",
        None,
        batch_size=2,
        workers=0,
        multi_task=False,
        debug=True,
    )
    assert captured["neuro.scaler"] is None
    assert captured["neuro.baseline"] is None
    assert captured["neuro.clamp"] is None


def test_non_eeg_hmc_channels_are_excluded() -> None:
    overrides = task_neuro_overrides(
        "sleep_stage", "alvarez2022haaglanden"
    )
    assert overrides["neuro.picks"] == ["F4", "C4", "O2", "C3"]
    assert "EMG" not in overrides["neuro.picks"]
    assert "ECG" not in overrides["neuro.picks"]
