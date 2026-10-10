# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared constants for benchmark plotting and analysis.

Display-name mappings, metric metadata, model metadata, and ordered
model-group lists used across multiple plotting modules.

Per-model metadata (parameter count, publication year, pretraining overlap,
display ordering, ...) is sourced from
:mod:`neuralbench.plots._models` (``MODELS: list[ModelEntry]``), which is the
single source of truth shared with the parameter-counting and table-generation
scripts under ``brainai-repo/brainai/bench/scripts/``. Backbone parameter
counts in :data:`MODEL_PARAMS` are loaded from
``neuralbench/plots/model_params.json``; re-run
``count_model_params.py`` after any model-kwargs change to refresh the cache.
"""

from __future__ import annotations

import re
import typing as tp

from neuralbench.plots._models import MODELS, load_param_counts

# ---------------------------------------------------------------------------
# Display-name mappings
#
# Entries that correspond to a registry model are derived from
# :data:`neuralbench.plots._models.MODELS` (mapping each entry's
# ``config_name`` to its display ``name``).  Aliases and non-registry
# baselines (Dummy, Chance, sklearn pipelines, secondary configs like
# ``NtLuna`` -> "LUNA (base)") are listed explicitly here.
# ---------------------------------------------------------------------------

_NON_REGISTRY_DISPLAY_NAMES: dict[str, str] = {
    "DummyPredictor": "Dummy",
    "Chance": "Chance",
    "SimpleConv": "SimpleConv",
    "NtLuna": "LUNA (base)",
    # Classical sklearn / pyriemann baselines (see neuralbench.baselines).
    # ``feature_based`` is a synthetic label assigned by
    # :func:`neuralbench.plots.tables._collapse_feature_based_baselines` which
    # keeps only the task-appropriate pipeline (per
    # :data:`neuralbench.registry.FEATURE_BASED_BY_TASK`) and relabels it to
    # collapse the three per-pipeline rows into a single headline ``Covariance-
    # based`` bar.  The per-pipeline names below are still used by deeper
    # diagnostic plots that opt out of the collapse.
    "feature_based": "Handcrafted",
    "xdawn_ts_lr": "Xdawn+TS+LR",
    "cov_ts_lr": "TS+LR",
    "cov_ts_ridge": "TS+Ridge",
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    **{m.config_name: m.name for m in MODELS if m.config_name is not None},
    **_NON_REGISTRY_DISPLAY_NAMES,
}

METRIC_DISPLAY_NAMES: dict[str, str] = {
    "test/bal_acc": "Balanced accuracy",
    "test/f1_score_macro": "F1 score (macro)",
    "test/pearsonr": "Pearson R",
    "test/rmse": "RMSE",
    "test/mae": "MAE",
    "test/bmae": "Binned MAE (s)",
    "test/CER": "Character error rate (%)",
    "test/full_retrieval/inv_norm_rank_median": "Inv. norm. rank",
    "test/full_retrieval/inv_norm_rank_mean": "Retrieval AUC",
    "test/full_retrieval/inv_norm_rank_median_subject-agg": "Inv. norm. rank (per-subject)",
    "test/full_retrieval/inv_norm_rank_mean_subject-agg": "Retrieval AUC (per-subject)",
    "test/full_retrieval/top5_acc_subject-agg": "Top-5 accuracy (per-subject)",
}

METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "test/bal_acc": True,
    "test/f1_score_macro": True,
    "test/pearsonr": True,
    "test/rmse": False,
    "test/mae": False,
    "test/bmae": False,
    "test/CER": False,
    "test/full_retrieval/inv_norm_rank_median": True,
    "test/full_retrieval/inv_norm_rank_mean": True,
    "test/full_retrieval/inv_norm_rank_median_subject-agg": True,
    "test/full_retrieval/inv_norm_rank_mean_subject-agg": True,
    "test/full_retrieval/top5_acc_subject-agg": True,
}

METRIC_PERFECT_SCORE: dict[str, float] = {
    "test/bal_acc": 100.0,
    "test/f1_score_macro": 1.0,
    "test/pearsonr": 1.0,
    "test/rmse": 0.0,
    "test/mae": 0.0,
    "test/bmae": 0.0,
    "test/CER": 0.0,
    # inv_norm_rank: [0, 1], 1.0 = true item always first
    "test/full_retrieval/inv_norm_rank_median": 1.0,
    "test/full_retrieval/inv_norm_rank_mean": 1.0,
    "test/full_retrieval/inv_norm_rank_median_subject-agg": 1.0,
    "test/full_retrieval/inv_norm_rank_mean_subject-agg": 1.0,
    "test/full_retrieval/top5_acc_subject-agg": 100.0,
}

TASK_DISPLAY_NAMES: dict[str, str] = {
    "motor_imagery": "Motor imagery",
    "_motor_imagery_stream": "Motor imagery (stream)",
    "motor_execution": "Motor execution",
    "motor_preparation": "Motor prep.",
    "p3": "P300",
    "cvep": "c-VEP",
    "ssvep": "SSVEP",
    "ern": "ERN",
    "lrp": "LRP",
    "n170": "N170",
    "n2pc": "N2pc",
    "n400": "N400",
    "mismatch_negativity": "MMN",
    "acoustic_change": "Acoustic change",
    "stimulus_congruency": "Congruency",
    "sustain_proc_negativity": "SPN",
    "audiovisual_stimulus": "Audiovisual",
    "action_recognition": "Action recog.",
    "asd_diagnosis": "ASD",
    "dementia_diagnosis": "Dementia",
    "depression_diagnosis": "Depression",
    "parkinsons_diagnosis": "Parkinson's",
    "schizophrenia_diagnosis": "Schizophrenia",
    "clinical_event": "Clinical event",
    "mental_arithmetic": "Mental arith.",
    "mental_imagery": "Mental imagery",
    "mental_workload": "Workload",
    "sleep_stage": "Sleep stage",
    "sleep_arousal": "Sleep arousal",
    "sleep_onset": "Sleep onset",
    "_sleep_onset_stream": "Sleep onset (stream)",
    "reaction_time": "Reaction time",
}

# ---------------------------------------------------------------------------
# Pretraining data overlap (dataset-class keys)
# ---------------------------------------------------------------------------
#
# Each entry lists the `neuralhub` study class names the foundation model was
# pretrained on, as validated against the model's source paper (see
# ``neuralbench-repo/refs/eeg/papers/``). Hatching a downstream bar requires an
# exact match between the core dataset's class name and one of the names
# below. Datasets that are not yet wired into any NeuralBench task are still
# included on purpose -- when such a task is added in the future, hatching
# will already be correct.
#
# Sourced from :data:`neuralbench.plots._models.MODELS`; see that module for
# the per-model frozenset definitions.
PRETRAINING_OVERLAP: dict[str, set[str]] = {
    m.name: set(m.pretraining_overlap)
    for m in MODELS
    if m.pretraining_overlap is not None
}

# ---------------------------------------------------------------------------
# Model-group ordering (used for colours and legend layout)
#
# Derived from the registry by filtering on family/device.  Order within each
# group matches the registry's iteration order, which is curated to be
# year-sorted within (family, device).
# ---------------------------------------------------------------------------

MEEG_CLASSIC_DISPLAY: list[str] = [
    m.name for m in MODELS if m.family == "classic" and m.device == "eeg"
]
FMRI_CLASSIC_DISPLAY: list[str] = [
    m.name for m in MODELS if m.family == "classic" and m.device == "fmri"
]
EMG_CLASSIC_DISPLAY: list[str] = [
    m.name for m in MODELS if m.family == "classic" and m.device == "emg"
]
CLASSIC_DISPLAY: list[str] = (
    MEEG_CLASSIC_DISPLAY + FMRI_CLASSIC_DISPLAY + EMG_CLASSIC_DISPLAY
)
MEEG_FM_DISPLAY: list[str] = [
    m.name for m in MODELS if m.family == "foundation" and m.device == "eeg"
]
FMRI_FM_DISPLAY: list[str] = [
    m.name for m in MODELS if m.family == "foundation" and m.device == "fmri"
]
FM_DISPLAY: list[str] = MEEG_FM_DISPLAY + FMRI_FM_DISPLAY
# Constant-predictor baselines plus the collapsed "Handcrafted" bar.
# Order controls (a) left-to-right order inside the "Baselines" legend column
# and (b) the leftmost bars in the benchmark bar chart.  The individual
# sklearn / pyriemann pipeline names are intentionally absent: they are
# collapsed into ``Handcrafted`` at plot time (see
# :func:`neuralbench.plots.tables._collapse_feature_based_baselines`).
DUMMY_DISPLAY = [
    "Chance",
    "Dummy",
    "Handcrafted",
]

# Light sage green for the Handcrafted baseline.  Picked to be distinct
# from the Greys used for the constant-predictor baselines (Chance /
# Dummy).
FEATURE_BASED_COLOR = "#4ea64e"

MODEL_YEAR: dict[str, int] = {m.name: m.year for m in MODELS if m.year is not None}

# Coarse model group per display name, used to pick which models define the
# ``task_max`` ceiling.  "baseline" covers the constant predictors
# (Chance/Dummy) and the collapsed Handcrafted bar.
MODEL_GROUP: dict[str, str] = {
    **{name: "foundation" for name in FM_DISPLAY},
    **{name: "classic" for name in CLASSIC_DISPLAY},
    **{name: "baseline" for name in DUMMY_DISPLAY},
}


# The adaptation-strategy suffix ``tables.eval_mode_suffix`` appends to
# foundation-model names (``"REVE (LoRA r32 flatten)"``, ``"LaBraM (FT mean)"``).
_EVAL_MODE_SUFFIX_RE = re.compile(r" \((?:LP-FT|LP|FT|AP|LoRA r\d+)(?: \w+)?\)$")


def strip_eval_mode_suffix(name: str) -> str:
    """Display name without its adaptation-strategy suffix, if it has one."""
    return _EVAL_MODE_SUFFIX_RE.sub("", name)


def model_group(name: str) -> str:
    """Coarse group ("foundation" / "classic" / "baseline" / "other") for a
    model display name.  Unknown names (e.g. disambiguated variants like
    ``"LUNA (base) [cfg=...]"`` or strategy-suffixed ones like
    ``"REVE (LP flatten)"``) fall back to their base name, then to
    ``"other"``."""
    if name in MODEL_GROUP:
        return MODEL_GROUP[name]
    base = strip_eval_mode_suffix(name.split(" [", 1)[0])
    return MODEL_GROUP.get(base, "other")


# ---------------------------------------------------------------------------
# Adaptation ``eval_mode`` tags
#
# :class:`AdaptationMode` is the single home for the grammar: ``aggregator``
# renders tags through it and the plotting modules parse them back.
# ---------------------------------------------------------------------------

_LORA_STRATEGY = "lora"
_LORA_PREFIX = f"{_LORA_STRATEGY}_r"

# by trainable-parameter budget; unknown strategies sort last
_STRATEGY_ORDER: dict[str, int] = {
    "linear_probe": 0,
    "attentive_probe": 1,
    _LORA_STRATEGY: 2,
    "finetune": 3,
    "lpft": 4,
}
_UNKNOWN_ORDER = 5
_PLAIN_STRATEGIES = tuple(s for s in _STRATEGY_ORDER if s != _LORA_STRATEGY)


# no -w preset: each foundation model runs with its own default wrapper
DEFAULT_EVAL_MODE = "default"


class AdaptationMode(tp.NamedTuple):
    """An adaptation strategy, as tagged on a result row.

    Renders as ``{strategy}[_{aggregation}]``, with LoRA using ``lora_r{rank}``
    as its stem. The aggregation is part of the tag because presets differing
    only by it would otherwise merge everywhere downstream.
    """

    strategy: str
    aggregation: str = ""
    lora_rank: int | None = None

    @classmethod
    def parse(cls, tag: str) -> AdaptationMode:
        """Read a tag back. Unrecognised tags become a strategy of their own."""
        if tag.startswith(_LORA_PREFIX):
            rank, _, aggregation = tag[len(_LORA_PREFIX) :].partition("_")
            if rank.isdigit():
                return cls(_LORA_STRATEGY, aggregation, int(rank))
        for strategy in _PLAIN_STRATEGIES:
            if tag == strategy:
                return cls(strategy)
            if tag.startswith(f"{strategy}_"):
                return cls(strategy, tag[len(strategy) + 1 :])
        return cls(tag)

    @property
    def is_lora(self) -> bool:
        return self.lora_rank is not None

    @property
    def tag(self) -> str:
        stem = (
            self.strategy if self.lora_rank is None else f"{_LORA_PREFIX}{self.lora_rank}"
        )
        return f"{stem}_{self.aggregation}" if self.aggregation else stem

    @property
    def sort_key(self) -> tuple[int, int, str]:
        """Canonical order, by trainable budget: probes -> LoRA -> finetune -> LP-FT."""
        order = _STRATEGY_ORDER.get(self.strategy, _UNKNOWN_ORDER)
        return (order, self.lora_rank or 0, self.aggregation)


_STRATEGY_LABEL: dict[str, str] = {
    "linear_probe": "Linear Probe",
    "attentive_probe": "Attentive Probe",
    "finetune": "Full FT",
    "lpft": "LP-FT",
}


def eval_mode_label(tag: str) -> str:
    """Human-readable label for an ``eval_mode`` tag (``"lora_r8"`` -> ``"LoRA r8"``)."""
    mode = AdaptationMode.parse(tag)
    if mode.is_lora:
        label = f"LoRA r{mode.lora_rank}"
    elif mode.strategy in _STRATEGY_LABEL:
        label = _STRATEGY_LABEL[mode.strategy]
    else:
        return tag
    return f"{label} ({mode.aggregation})" if mode.aggregation else label


# ---------------------------------------------------------------------------
# Task categories and ordering (used for category-grouped layouts)
# ---------------------------------------------------------------------------

TASK_CATEGORIES: dict[str, list[str]] = {
    "Cognitive": [
        "action_recognition",
        "image",
        "sentence",
        "speech",
        "typing",
        "video",
        "word",
    ],
    "BCI": [
        "cvep",
        "mental_imagery",
        "motor_execution",
        "motor_imagery",
        "_motor_imagery_stream",
        "motor_preparation",
        "p3",
        "pose",
        "ssvep",
    ],
    "Evoked Responses": [
        "acoustic_change",
        "audiovisual_stimulus",
        "ern",
        "mismatch_negativity",
        "n170",
        "n2pc",
        "n400",
        "lrp",
        "stimulus_congruency",
        "sustain_proc_negativity",
    ],
    "Clinical": [
        "asd_diagnosis",
        "clinical_event",
        "dementia_diagnosis",
        "depression_diagnosis",
        "parkinsons_diagnosis",
        "pathology",
        "schizophrenia_diagnosis",
        "seizure",
    ],
    "Internal State": [
        "emotion",
        "mental_arithmetic",
        "mental_workload",
    ],
    "Sleep": [
        "sleep_arousal",
        "sleep_onset",
        "_sleep_onset_stream",
        "sleep_stage",
    ],
    "Phenotyping": [
        "age",
        "psychopathology",
        "sex",
    ],
    "Misc": [
        "artifact",
        "reaction_time",
    ],
}

TASK_TO_CATEGORY: dict[str, str] = {
    task: cat for cat, tasks in TASK_CATEGORIES.items() for task in tasks
}

CATEGORY_ROW_GROUPS: list[list[str]] = [
    ["Cognitive"],
    ["BCI"],
    ["Evoked Responses"],
    ["Clinical"],
    ["Internal State"],
    ["Phenotyping"],
    ["Sleep"],
    ["Misc"],
]

CATEGORY_ORDER: list[str] = [
    "Cognitive",
    "BCI",
    "Evoked Responses",
    "Clinical",
    "Internal State",
    "Sleep",
    "Phenotyping",
    "Misc",
]

CATEGORY_COLORS: dict[str, str] = {
    "Cognitive": "#0072B2",
    "BCI": "#E69F00",
    "Evoked Responses": "#009E73",
    "Clinical": "#D55E00",
    "Internal State": "#CC79A7",
    "Sleep": "#F0E442",
    "Phenotyping": "#56B4E9",
    "Misc": "#999999",
}

# ---------------------------------------------------------------------------
# Model parameter counts
#
# Backbone parameter counts (excluding the task-specific output projection),
# loaded from the JSON cache produced by
# ``brainai-repo/brainai/bench/scripts/count_model_params.py``. Re-run that
# script to regenerate ``neuralbench/plots/model_params.json`` after model
# kwargs change. Classic braindecode counts use representative n_chans=22,
# n_times=256 and vary slightly across tasks.
# ---------------------------------------------------------------------------

MODEL_PARAMS: dict[str, int] = load_param_counts()
