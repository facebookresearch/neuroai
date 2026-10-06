"""Built-in, label-safe NeuroLM instructions for the seven EEG tasks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .task_spec import NeuroLMTaskSpec


def _choice_spec(name: str, prompt: str, n_classes: int) -> NeuroLMTaskSpec:
    labels = tuple(f"({chr(ord('A') + i)})" for i in range(n_classes))
    if not prompt.rstrip().endswith("Answer:"):
        raise ValueError(f"Multiple-choice prompt must end with 'Answer:': {prompt}")
    # Match official TUEV/TUSL/HMC evaluation: the opening parenthesis is part
    # of the prompt and generation supplies only the choice continuation.
    prompt = f"{prompt.rstrip()} ("
    return NeuroLMTaskSpec(name=name, prompt=prompt, answers=labels)


def tuab_pathology() -> NeuroLMTaskSpec:
    # Exact question/answer convention from official NeuroLM TUAB.
    return NeuroLMTaskSpec(
        name="eeg/pathology/Lopez2017Tuab",
        prompt="Question: Is this EEG segment abnormal? Answer:",
        answers=("No", "Yes"),
    )


def cog_bci_workload() -> NeuroLMTaskSpec:
    return _choice_spec(
        "eeg/mental_workload/Hinss2023Open",
        "Question: What is the mental workload level of this EEG segment? "
        "Options: (A) low, (B) medium, (C) high. Answer:",
        3,
    )


def physionet_motor_imagery() -> NeuroLMTaskSpec:
    return _choice_spec(
        "eeg/motor_imagery/Schalk2004Bci2000",
        "Question: Which motor imagery task is the subject performing? "
        "Options: (A) left fist, (B) right fist, (C) both fists, "
        "(D) both feet. Answer:",
        4,
    )


def tuev_clinical_event() -> NeuroLMTaskSpec:
    return _choice_spec(
        "eeg/clinical_event/Harati2015Tuev",
        "Question: Which event type does this EEG segment belong to? Options: "
        "(A) spike and slow wave. (B) generalized periodic epileptiform "
        "discharge. (C) periodic lateralized epileptiform discharge. (D) eye "
        "movement. (E) artifact. (F) background. Answer:",
        6,
    )


def hmc_sleep_stage() -> NeuroLMTaskSpec:
    return _choice_spec(
        "eeg/sleep_stage/Alvarez2022Haaglanden",
        "Question: Which sleep type does this EEG segment belong to? "
        "Options: (A) Wake. (B) NREM-1. (C) NREM-2. (D) NREM-3. "
        "(E) REM. Answer:",
        5,
    )


def faced_emotion() -> NeuroLMTaskSpec:
    return _choice_spec(
        "eeg/emotion/Chen2023Large",
        "Question: Which emotion does this EEG recording represent? Options: "
        "(A) disgust, (B) fear, (C) sadness, (D) neutral, (E) amusement, "
        "(F) inspiration, (G) joy, (H) tenderness, (I) anger. Answer:",
        9,
    )


def mental_arithmetic() -> NeuroLMTaskSpec:
    # Extend official NeuroLM's natural-language binary convention to this
    # NeuralBench-only task.
    return NeuroLMTaskSpec(
        name="eeg/mental_arithmetic/Zyma2019Electroencephalograms",
        prompt=(
            "Question: Is the subject performing a mental arithmetic task in "
            "this EEG recording? Answer:"
        ),
        answers=("No", "Yes"),
    )


def builtin_task_spec(task: str, dataset: str | None = None) -> NeuroLMTaskSpec:
    key = task.lower()
    if key == "pathology":
        return tuab_pathology()
    if key == "clinical_event":
        return tuev_clinical_event()
    if key == "sleep_stage":
        return hmc_sleep_stage()
    if key == "emotion":
        return faced_emotion()
    if key == "mental_arithmetic":
        return mental_arithmetic()
    if key == "mental_workload":
        return cog_bci_workload()
    if key == "motor_imagery" and (dataset or "").lower() in {
        "schalk2004bci2000", "physionet", "physionetmi"
    }:
        return physionet_motor_imagery()
    raise ValueError(
        f"No built-in NeuroLM task spec for task={task!r}, dataset={dataset!r}."
    )


_RAW_LABEL_ANSWERS: dict[str, dict[str, str]] = {
    "pathology": {"normal": "No", "abnormal": "Yes"},
    "clinical_event": {
        "spsw": "(A)", "spike and slow wave": "(A)",
        "gped": "(B)", "generalized periodic epileptiform discharge": "(B)",
        "pled": "(C)", "periodic lateralized epileptiform discharge": "(C)",
        "eyem": "(D)", "eye movement": "(D)",
        "artf": "(E)", "art": "(E)", "artifact": "(E)",
        "bckg": "(F)", "background": "(F)", "background activity": "(F)",
    },
    "sleep_stage": {
        "w": "(A)", "wake": "(A)", "n1": "(B)", "n2": "(C)",
        "n3": "(D)", "r": "(E)", "rem": "(E)",
    },
    "emotion": {
        "disgust": "(A)", "fear": "(B)", "sadness": "(C)", "neutral": "(D)",
        "amusement": "(E)", "inspiration": "(F)", "joy": "(G)",
        "tenderness": "(H)", "anger": "(I)",
    },
    "mental_arithmetic": {
        "rest": "No", "resting state": "No", "mental arithmetic": "Yes",
    },
    "mental_workload": {
        "matbeasy": "(A)", "matb easy": "(A)",
        "matbmed": "(B)", "matb med": "(B)",
        "matbdiff": "(C)", "matb diff": "(C)",
    },
    "motor_imagery": {
        "imagery left fist": "(A)", "imagery right fist": "(B)",
        "imagery bilateral fist": "(C)", "imagery bilateral feet": "(D)",
    },
}


def align_task_spec_to_dataset(
    spec: NeuroLMTaskSpec, dataset: Any, task: str
) -> NeuroLMTaskSpec:
    """Order answer tokens by the prepared NeuralBench LabelEncoder indices."""
    extractor = getattr(dataset, "extractors", {}).get("target")
    mapping = getattr(extractor, "_label_to_ind", None)
    if not mapping:
        return spec
    lookup = _RAW_LABEL_ANSWERS[task]
    answers: list[str | None] = [None] * len(mapping)
    background_label = None
    for raw_label, index in mapping.items():
        key = " ".join(
            str(raw_label).strip().lower().replace("_", " ").replace("-", " ").split()
        )
        answer = lookup.get(key)
        if answer is None:
            raise ValueError(
                f"Unknown {task} label {raw_label!r}; cannot safely map it to "
                "the NeuroLM instruction choices."
            )
        answers[int(index)] = answer
        if answer == "(F)" and task == "clinical_event":
            background_label = int(index)
    if any(answer is None for answer in answers):
        raise ValueError(f"Non-contiguous NeuralBench label mapping for {task}: {mapping}")
    return replace(
        spec,
        answers=tuple(answer for answer in answers if answer is not None),
        background_label=background_label,
    )
