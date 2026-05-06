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
    # Classical sklearn / pyriemann baselines (see neuralbench.sklearn_baseline).
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
    "test/full_retrieval/top5_acc_subject-agg": "Top-5 accuracy (per-subject)",
}

METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    "test/bal_acc": True,
    "test/f1_score_macro": True,
    "test/pearsonr": True,
    "test/rmse": False,
    "test/full_retrieval/top5_acc_subject-agg": True,
}

METRIC_PERFECT_SCORE: dict[str, float] = {
    "test/bal_acc": 100.0,
    "test/f1_score_macro": 1.0,
    "test/pearsonr": 1.0,
    "test/rmse": 0.0,
    "test/full_retrieval/top5_acc_subject-agg": 100.0,
}

TASK_DISPLAY_NAMES: dict[str, str] = {
    "motor_imagery": "Motor imagery",
    "motor_execution": "Motor execution",
    "p3": "P300",
    "cvep": "c-VEP",
    "ssvep": "SSVEP",
    "ern": "ERN",
    "lrp": "LRP",
    "n170": "N170",
    "n2pc": "N2pc",
    "n400": "N400",
    "mismatch_negativity": "MMN",
    "audiovisual_stimulus": "Audiovisual",
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
CLASSIC_DISPLAY: list[str] = MEEG_CLASSIC_DISPLAY + FMRI_CLASSIC_DISPLAY
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
# Dummy).  fMRI classic models previously used a cm.Greens palette; they
# have been moved to a custom purple colormap (see ``_FMRI_PURPLES`` in
# :mod:`neuralbench.plots.non_eeg`) so this sage green can unambiguously
# signal the Handcrafted baseline across all figures.
FEATURE_BASED_COLOR = "#4ea64e"

MODEL_YEAR: dict[str, int] = {m.name: m.year for m in MODELS if m.year is not None}

# ---------------------------------------------------------------------------
# Task categories and ordering (used for category-grouped layouts)
# ---------------------------------------------------------------------------

TASK_CATEGORIES: dict[str, list[str]] = {
    "Cognitive": [
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
        "p3",
        "ssvep",
    ],
    "Evoked Responses": [
        "audiovisual_stimulus",
        "ern",
        "mismatch_negativity",
        "n170",
        "n2pc",
        "n400",
        "lrp",
    ],
    "Clinical": [
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
    ["Sleep", "Misc"],
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
# Recording-device metadata (used by the non-EEG bar chart)
# ---------------------------------------------------------------------------

DEVICE_DISPLAY_NAMES: dict[str, str] = {
    "eeg": "EEG",
    "meg": "MEG",
    "fmri": "fMRI",
    "emg": "EMG",
    "fnirs": "fNIRS",
}

DEVICE_COLORS: dict[str, str] = {
    "eeg": "#0072B2",
    "meg": "#009E73",
    "fmri": "#D55E00",
    "emg": "#CC79A7",
    "fnirs": "#E69F00",
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
