"""NeuralBench-native seven-task NeuroLM data orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from .presets import builtin_task_spec


@dataclass(frozen=True)
class NeuroLMTaskConfig:
    """Mapping from a BLPM task name to its NeuralBench task/variant."""

    name: str
    neuralbench_task: str
    dataset: str | None = None


BLPM_SEVEN_TASKS = (
    NeuroLMTaskConfig("physionet", "motor_imagery", "schalk2004bci2000"),
    NeuroLMTaskConfig("tuab", "pathology"),
    NeuroLMTaskConfig("tuev", "clinical_event"),
    NeuroLMTaskConfig("hmc", "sleep_stage", "alvarez2022haaglanden"),
    NeuroLMTaskConfig("faced", "emotion"),
    NeuroLMTaskConfig("mentalarithmetic", "mental_arithmetic"),
    NeuroLMTaskConfig("cog_bci_workload", "mental_workload"),
)


def select_task_configs(task_names: tuple[str, ...] = ()) -> tuple[NeuroLMTaskConfig, ...]:
    """Return the requested subset of the fixed NeuralBench NeuroLM suite."""
    if not task_names:
        return BLPM_SEVEN_TASKS
    by_name = {task.name: task for task in BLPM_SEVEN_TASKS}
    unknown = [name for name in task_names if name not in by_name]
    if unknown:
        choices = ", ".join(by_name)
        raise ValueError(f"Unknown NeuroLM train task(s) {unknown}; choose from: {choices}")
    # Preserve the suite's canonical order even when names are supplied in a
    # different order, so task scheduling remains reproducible.
    requested = set(task_names)
    return tuple(task for task in BLPM_SEVEN_TASKS if task.name in requested)

TASK_NEURO_OVERRIDES: dict[tuple[str, str | None], dict[str, Any]] = {
    # Match official NeuroLM's four HMC EEG derivations; NeuralFetch also
    # exposes EMG/ECG, which are not valid NeuroLM channel tokens.
    ("sleep_stage", "alvarez2022haaglanden"): {
        "neuro.picks": ["F4", "C4", "O2", "C3"],
    },
    # Official NeuroLM Workload preprocessing drops the A2-A1 reference lead.
    ("mental_arithmetic", None): {
        "neuro.picks": [
            "Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4", "C3",
            "C4", "T5", "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz",
        ],
    },
}


def task_neuro_overrides(task: str, dataset: str | None) -> dict[str, Any]:
    return dict(TASK_NEURO_OVERRIDES.get((task, dataset), {}))


def build_seven_task_loaders(
    *,
    batch_size: int,
    num_workers: int,
    debug: bool = False,
    task_names: tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    """Prepare the seven NeuralBench loaders with NeuroLM preprocessing."""
    from neuralbench import get_default_dataloaders
    from .cli import _load_preprocessing_config

    preprocessing = _load_preprocessing_config()["neuro"]
    loaders: dict[str, dict[str, Any]] = {}
    for task in select_task_configs(task_names):
        effective_batch = min(batch_size, 8) if debug else batch_size
        effective_workers = 0 if debug else num_workers
        overrides = {
            f"neuro.{key}": value
            for key, value in preprocessing.items()
        }
        overrides.update(task_neuro_overrides(task.neuralbench_task, task.dataset))
        loaders[task.name] = get_default_dataloaders(
            "eeg",
            task.neuralbench_task,
            dataset=task.dataset,
            batch_size=effective_batch,
            num_workers=effective_workers,
            **overrides,
        )
    return loaders


def iter_multitask_train_batches(
    loaders: dict[str, dict[str, Any]], *, seed: int = 1337
) -> Iterator[tuple[str, Any]]:
    """Randomly interleave homogeneous task batches and exhaust each once.

    This preserves official NeuroLM concat-dataset exposure (larger datasets
    contribute more samples) while avoiding mixed-shape batches.
    """
    import random

    rng = random.Random(seed)
    iterators = {name: iter(parts["train"]) for name, parts in loaders.items()}
    active = list(iterators)
    while active:
        name = rng.choice(active)
        try:
            yield name, next(iterators[name])
        except StopIteration:
            active.remove(name)


def task_spec_for_name(name: str):
    config = next(task for task in BLPM_SEVEN_TASKS if task.name == name)
    return builtin_task_spec(config.neuralbench_task, config.dataset)
