# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Discovery, scanning, and validation of tasks, models, and datasets.

This module builds the module-level registries (``TASKS``, ``ALL_MODELS``,
``ALL_DATASETS``, etc.) at import time by scanning the ``tasks/`` and
``models/`` directories as well as any entry-point plugins.
"""

import functools
import importlib
import importlib.metadata
import json
import logging
import typing as tp
from pathlib import Path
from warnings import warn

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class _SafeIgnoreLoader(yaml.SafeLoader):
    """YAML loader that silently ignores ``!!python/*`` tags."""


_SafeIgnoreLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/",
    lambda loader, suffix, node: None,
)


def _make_include_loader(base_dir: Path, parent: type = yaml.Loader) -> type:
    """Create a YAML loader subclass that resolves ``!include`` relative to *base_dir*."""

    class IncludeLoader(parent):  # type: ignore[valid-type]
        pass

    def _include_constructor(loader: yaml.Loader, node: yaml.ScalarNode) -> tp.Any:
        rel_path = loader.construct_scalar(node)
        full_path = base_dir / rel_path
        with open(full_path) as f:
            if full_path.suffix == ".json":
                return json.load(f)
            return yaml.load(f, Loader=IncludeLoader)

    IncludeLoader.add_constructor("!include", _include_constructor)
    return IncludeLoader


def load_yaml_config(fname: Path, safe: bool = False) -> dict[str, tp.Any] | None:
    """Load a YAML config file.

    When *safe* is ``True``, ``!!python/*`` tags are silently ignored
    instead of being resolved (useful for extracting plain data without
    importing heavy dependencies).
    """
    base_dir = Path(fname).parent
    if safe:
        loader = _make_include_loader(base_dir, parent=_SafeIgnoreLoader)
    else:
        loader = _make_include_loader(base_dir)
    with open(fname, "r") as f:
        return yaml.load(f, Loader=loader)


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _discover_entry_point_dirs(group: str) -> tuple[Path, ...]:
    """Import entry-point modules and return their package directories.

    Importing the module also triggers ``DiscriminatedModel`` class
    registration for any model configs defined there.
    """
    dirs: list[Path] = []
    for ep in importlib.metadata.entry_points(group=group):
        try:
            module = importlib.import_module(ep.value)
        except ImportError:
            logger.warning("Failed to import %s entry point: %s", group, ep.value)
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            dirs.append(Path(module_file).resolve().parent)
    return tuple(dirs)


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DEFAULTS_DIR = BASE_DIR / "defaults"
MODELS_DIR = BASE_DIR / "models"


# ---------------------------------------------------------------------------
# Task scanning
# ---------------------------------------------------------------------------

# Names of subdirectories under ``tasks/`` and ``tasks/<device>/`` that are
# never devices or tasks (cache artifacts, hidden dirs).
_NON_TASK_DIRS = frozenset({"__pycache__"})


def _all_task_roots() -> list[Path]:
    """Return internal + external task root directories."""
    roots = [BASE_DIR / "tasks"]
    for ext_dir in _discover_entry_point_dirs("neuralbench.tasks"):
        candidate = ext_dir / "tasks"
        if candidate.is_dir():
            roots.append(candidate)
    return roots


def _scan_task_dirs(*, unvalidated: bool = False) -> dict[str, list[str]]:
    """Scan ``tasks/<device>/`` for task directories.

    Devices are discovered from the filesystem: every non-hidden, non-cache
    subdirectory of a task root is treated as a device. A device with no
    shipped tasks (e.g. an empty placeholder for an upcoming modality) is
    still included with an empty task list.

    When *unvalidated* is False (default), returns validated tasks (those
    **not** prefixed with ``_``).  When True, returns only unvalidated
    (``_``-prefixed) task directories.
    """
    tasks: dict[str, list[str]] = {}
    for root in _all_task_roots():
        if not root.is_dir():
            continue
        for device_dir in root.iterdir():
            if (
                not device_dir.is_dir()
                or device_dir.name.startswith(".")
                or device_dir.name in _NON_TASK_DIRS
            ):
                continue
            device = device_dir.name
            names = [
                d.stem
                for d in device_dir.iterdir()
                if d.is_dir()
                and d.name not in _NON_TASK_DIRS
                and d.stem.startswith("_") == unvalidated
            ]
            tasks.setdefault(device, []).extend(names)
    return {k: sorted(set(v)) for k, v in tasks.items()}


def _resolve_task_dir(device: str, task_name: str) -> Path:
    """Find the task directory for *task_name* across all task roots."""
    for root in _all_task_roots():
        path = root / device / task_name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"No task directory found for '{device}/{task_name}'")


# ---------------------------------------------------------------------------
# Model scanning
# ---------------------------------------------------------------------------


def _all_model_dirs() -> list[Path]:
    """Return internal + external model directories."""
    dirs = [MODELS_DIR]
    dirs.extend(_discover_entry_point_dirs("neuralbench.models"))
    return dirs


def _build_all_models() -> list[str]:
    """List all model YAML stems from internal + external sources."""
    models: list[str] = []
    for d in _all_model_dirs():
        models.extend(f.stem for f in d.iterdir() if f.suffix == ".yaml")
    return sorted(set(models))


def _resolve_model_config_path(model_name: str) -> Path:
    """Find the YAML config for *model_name* across all model directories."""
    for d in _all_model_dirs():
        path = d / f"{model_name}.yaml"
        if path.exists():
            return path
    raise FileNotFoundError(f"No config found for model '{model_name}'")


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------


def _build_all_datasets(
    tasks: dict[str, list[str]],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict[str, list[str]]]]:
    """Build study-name and file-stem mappings for every task's datasets.

    Returns ``(study_names, file_stems)`` where each is
    ``{device: {task: [...]}}``.
    """
    study_names: dict[str, dict[str, list[str]]] = {}
    file_stems: dict[str, dict[str, list[str]]] = {}
    for device, task_names in tasks.items():
        study_names[device] = {}
        file_stems[device] = {}
        for task_name in task_names:
            task_dir = _resolve_task_dir(device, task_name)
            config = load_yaml_config(task_dir / "config.yaml", safe=True)
            assert config is not None
            default_study = config["data"]["study"]["source"]["name"]
            studies = [default_study]
            stems: list[str] = []
            datasets_dir = task_dir / "datasets"
            if datasets_dir.exists():
                for f in sorted(datasets_dir.glob("*.yaml")):
                    stems.append(f.stem)
                    ds_config = load_yaml_config(f, safe=True)
                    if ds_config is None:
                        continue
                    name = (
                        ds_config.get("data", {})
                        .get("study", {})
                        .get("source", {})
                        .get("name")
                    )
                    if name and name not in studies:
                        studies.append(name)
            study_names[device][task_name] = studies
            file_stems[device][task_name] = stems
    return study_names, file_stems


# ---------------------------------------------------------------------------
# Module-level registries (populated at import time)
# ---------------------------------------------------------------------------

TASKS = _scan_task_dirs(unvalidated=False)
ALL_TASKS: list[str] = sorted(set(sum(TASKS.values(), [])))
UNVALIDATED_TASKS = _scan_task_dirs(unvalidated=True)
ALL_UNVALIDATED_TASKS: list[str] = sorted(set(sum(UNVALIDATED_TASKS.values(), [])))
# ``ALL_DEVICES`` is the union of every device discovered under ``tasks/``,
# whether or not it currently ships any tasks. Empty placeholder dirs (e.g.
# ``tasks/emg/``) keep upcoming modalities visible in ``--help`` so users
# know they are on the roadmap.
ALL_DEVICES: list[str] = sorted(set(TASKS) | set(UNVALIDATED_TASKS))
ALL_MODELS = _build_all_models()
ALL_DOWNSTREAM_WRAPPERS: dict[str, dict[str, tp.Any]] = (
    load_yaml_config(DEFAULTS_DIR / "downstream_wrappers.yaml") or {}
)
DEBUG_STUDY_QUERIES: dict[str, str | None] = (
    load_yaml_config(DEFAULTS_DIR / "debug_study.yaml") or {}
)
ALL_DATASETS, _ALL_DATASET_STEMS = _build_all_datasets(TASKS)


def get_available_datasets(device: str, task: str) -> list[str]:
    """Get list of available dataset file stems for a given task."""
    return _ALL_DATASET_STEMS.get(device, {}).get(task, [])


# ---------------------------------------------------------------------------
# Model categories
#
# The trainable-model lists (``CLASSIC_MODELS``, ``FM_MODELS``, and the
# ``DEVICE_*_MODELS`` per-device dicts) are derived from
# :data:`neuralbench.plots._models.MODELS` so the CLI / launch infrastructure
# stays in sync with the plotting / table-generation code.  ``BASELINE_MODELS``
# and ``SKLEARN_BASELINE_MODELS`` are not in the registry (they're task-
# agnostic predictors / classical-ML pipelines, not trainable architectures)
# and remain enumerated here.
# ---------------------------------------------------------------------------

from neuralbench.plots._models import MODELS as _REGISTRY_MODELS  # noqa: E402


def _cli_names(*, family: str, device: str) -> list[str]:
    return [
        m.cli_name
        for m in _REGISTRY_MODELS
        if m.family == family and m.device == device and m.cli_name is not None
    ]


# EEG/MEG defaults (MEG reuses the EEG model list because no MEG-specific
# trainable models are registered today).
CLASSIC_MODELS: list[str] = _cli_names(family="classic", device="eeg")
FM_MODELS: list[str] = _cli_names(family="foundation", device="eeg")

# Non-DL / classical-ML baselines, run once at model-build time then
# evaluated on the test set with the same preprocessing pipeline as the DL
# models.  ``chance`` and ``dummy`` are task-agnostic constant predictors;
# the remaining pyriemann / sklearn baselines are task-specific and dispatched
# via ``FEATURE_BASED_BY_TASK`` below so each task sees exactly one classical
# baseline (the most task-appropriate one) in the headline plots.
SKLEARN_BASELINE_MODELS: frozenset[str] = frozenset(
    {
        "xdawn_ts_lr",
        "cov_ts_lr",
        "cospectra_log_lr",
        "cov_ts_ridge",
    }
)
BASELINE_MODELS: list[str] = [
    "chance",
    "dummy",
    *sorted(SKLEARN_BASELINE_MODELS),
]

DEVICE_CLASSIC_MODELS: dict[str, list[str]] = {
    "eeg": CLASSIC_MODELS,
    "meg": CLASSIC_MODELS,
    "fmri": _cli_names(family="classic", device="fmri"),
}
DEVICE_FM_MODELS: dict[str, list[str]] = {
    "eeg": FM_MODELS,
    "meg": FM_MODELS,
    "fmri": _cli_names(family="foundation", device="fmri"),
}
# Per-device classical baseline lists.
#
# - EEG uses the full ``BASELINE_MODELS`` list: Xdawn and CoSpectra are tuned to
#   low-channel-count ERP / SSVEP signals, so they stay EEG-specific.
# - MEG reuses the Riemannian TS pipelines (``cov_ts_lr`` / ``cov_ts_ridge``),
#   which have no structural EEG assumption -- they fit per-trial covariances
#   and a linear head regardless of channel count.  On ~272-306 MEG sensors
#   the tangent-space dimension is C(C+1)/2 ≈ 37k-47k features; OAS shrinkage
#   and the default ``max_fit_samples=20_000`` cap keep the linear head
#   feasible.  Xdawn (per-class ERP filter, explodes on 29-way multiclass) and
#   CoSpectra (sfreq-tuned narrowband spectral features) remain excluded.
# - fMRI has no classical pyriemann / covariance analogue that transfers
#   meaningfully from a channel basis to a voxel basis, so only the constant
#   predictors are kept.
DEVICE_BASELINE_MODELS: dict[str, list[str]] = {
    "eeg": BASELINE_MODELS,
    "meg": ["chance", "dummy", "cov_ts_lr", "cov_ts_ridge"],
    "fmri": ["chance", "dummy"],
}

# ---------------------------------------------------------------------------
# Task -> canonical "handcrafted" sklearn baseline
# ---------------------------------------------------------------------------
# A single classical baseline is picked per task based on the dominant signal
# characteristics: Xdawn covariances for event-locked ERPs, Riemannian
# tangent-space + LR for oscillatory / state classification, and tangent-space
# + Ridge for regression and multi-output retrieval.  This mapping drives (a)
# task-aware dispatch in ``_expand_models`` and (b) the plot-time collapse to a
# single "Handcrafted" bar in the headline bar / box charts.
FEATURE_BASED_BY_TASK: dict[str, str] = {
    # ERP / event-locked signals -> Xdawn covariances (moabb P300 convention).
    "p3": "xdawn_ts_lr",
    "ern": "xdawn_ts_lr",
    "lrp": "xdawn_ts_lr",
    "n170": "xdawn_ts_lr",
    "n2pc": "xdawn_ts_lr",
    "n400": "xdawn_ts_lr",
    "mismatch_negativity": "xdawn_ts_lr",
    "audiovisual_stimulus": "xdawn_ts_lr",
    # Oscillatory / BCI classification (moabb TSLR convention).
    "motor_imagery": "cov_ts_lr",
    "motor_execution": "cov_ts_lr",
    "cvep": "cov_ts_lr",
    "ssvep": "cospectra_log_lr",
    "mental_imagery": "cov_ts_lr",
    "mental_arithmetic": "cov_ts_lr",
    "mental_workload": "cov_ts_lr",
    "emotion": "cov_ts_lr",
    # Clinical / state classification -- same Riemann-TS features but the
    # discriminative signal is spectral-power rather than event-locked.
    "dementia_diagnosis": "cov_ts_lr",
    "depression_diagnosis": "cov_ts_lr",
    "parkinsons_diagnosis": "cov_ts_lr",
    "schizophrenia_diagnosis": "cov_ts_lr",
    "pathology": "cov_ts_lr",
    "seizure": "cov_ts_lr",
    "clinical_event": "cov_ts_lr",
    "artifact": "cov_ts_lr",  # multilabel -> wrapped in OneVsRestClassifier
    "sex": "cov_ts_lr",
    "sleep_stage": "cov_ts_lr",
    "sleep_arousal": "cov_ts_lr",
    # Univariate regression.
    "age": "cov_ts_ridge",
    "reaction_time": "cov_ts_ridge",
    "psychopathology": "cov_ts_ridge",
    # Multi-output retrieval (image / text / speech decoding).  TS+Ridge
    # handles multi-output targets natively and captures spatial covariance
    # structure, which a channel-wise mean + Ridge would discard.
    "image": "cov_ts_ridge",
    "sentence": "cov_ts_ridge",
    "speech": "cov_ts_ridge",
    "typing": "cov_ts_ridge",
    "video": "cov_ts_ridge",
    "word": "cov_ts_ridge",
}


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _task_aware_baseline(device: str, task_name: str | None) -> list[str]:
    """Resolve the canonical baseline list for *task_name* on *device*.

    Returns ``chance`` + ``dummy`` plus at most one task-appropriate sklearn
    pipeline from ``FEATURE_BASED_BY_TASK``.  When *task_name* is ``None`` or
    the device has no sklearn baselines (fMRI), returns the full device-level
    list from ``DEVICE_BASELINE_MODELS`` for backward compat.  When the
    task's canonical pipeline is registered for the device (EEG and MEG both
    expose the Riemannian pipelines), that pipeline is appended so
    ``--model all_baseline`` on MEG picks up a Handcrafted baseline.
    """
    if task_name is None:
        return list(DEVICE_BASELINE_MODELS.get(device, BASELINE_MODELS))
    device_baselines = DEVICE_BASELINE_MODELS.get(device, BASELINE_MODELS)
    preferred = FEATURE_BASED_BY_TASK.get(task_name)
    if preferred is None or preferred not in device_baselines:
        return [m for m in device_baselines if m not in SKLEARN_BASELINE_MODELS]
    return [m for m in device_baselines if m not in SKLEARN_BASELINE_MODELS] + [preferred]


def _expand_models(
    model: str | list[str] | None,
    device: str = "eeg",
    task_name: str | None = None,
) -> list[str | None]:
    """Expand model aliases into concrete model names.

    Recognised aliases: ``all``, ``all_classic``, ``all_fm``, ``all_baseline``.
    When *device* is provided, each alias resolves to the device-specific
    list (``DEVICE_CLASSIC_MODELS`` / ``DEVICE_FM_MODELS`` /
    ``DEVICE_BASELINE_MODELS``) rather than the EEG defaults.

    When *task_name* is provided, the baseline alias resolution is task-aware:
    only ``chance``, ``dummy`` and the single sklearn pipeline registered in
    :data:`FEATURE_BASED_BY_TASK` are emitted.  This avoids launching N
    pipelines per task (only one is used in plots anyway).  Passing
    ``task_name=None`` preserves the legacy "all sklearn baselines" behaviour
    for workflows that want to benchmark every pipeline standalone.
    """
    if model is None:
        return [None]
    if isinstance(model, str):
        model = [model]
    models: list[str | None] = list(model)
    classic = DEVICE_CLASSIC_MODELS.get(device, CLASSIC_MODELS)
    fm = DEVICE_FM_MODELS.get(device, FM_MODELS)
    baseline = _task_aware_baseline(device, task_name)
    if "all" in models:
        models.remove("all")
        models += classic + fm + baseline
    if "all_classic" in models:
        models.remove("all_classic")
        models += classic
    if "all_fm" in models:
        models.remove("all_fm")
        models += fm
    if "all_baseline" in models:
        models.remove("all_baseline")
        models += baseline
    models = sorted(set(models), key=lambda x: (x is None, x))
    return models


def _resolve_tasks(device: str, task: str | list[str]) -> list[str]:
    """Resolve ``"all"`` / ``"all_multi_dataset"`` into validated task lists for *device*."""
    if isinstance(task, str):
        task = [task]
    if task == ["all"] or task == ["all_multi_dataset"]:
        unvalidated = UNVALIDATED_TASKS.get(device, [])
        if unvalidated:
            warn(
                f"Skipping unvalidated task(s): {', '.join(unvalidated)}. "
                "To run them, specify them explicitly."
            )
        resolved = TASKS[device]
        if task == ["all_multi_dataset"]:
            resolved = [t for t in resolved if _ALL_DATASET_STEMS.get(device, {}).get(t)]
        return resolved
    return task


def _resolve_datasets(
    device: str, task_name: str, dataset_arg: str | list[str] | None
) -> list[str | None]:
    """Resolve ``"all"`` into the available dataset file stems for a task."""
    if dataset_arg is None:
        return [None]
    if isinstance(dataset_arg, str):
        dataset_arg = [dataset_arg]
    if dataset_arg == ["all"]:
        available = get_available_datasets(device, task_name)
        if not available:
            warn(
                f"No additional datasets found for {task_name}, running with base config"
            )
            return [None]
        return [None] + available  # type: ignore[return-value]
    return dataset_arg  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Validation and formatting
# ---------------------------------------------------------------------------


def _format_datasets_epilog() -> str:
    """Format ``ALL_DATASETS`` for display in ``--help``."""
    lines = ["available datasets per task:"]
    for device, tasks in sorted(ALL_DATASETS.items()):
        if not tasks:
            continue
        n_tasks = len(tasks)
        n_datasets = sum(len(ds) for ds in tasks.values())
        lines.append(f"\n  {device} ({n_tasks} tasks, {n_datasets} datasets):")
        for task, datasets in sorted(tasks.items()):
            lines.append(f"    {task}: {', '.join(datasets)}")
    return "\n".join(lines)


def _validate_inputs(
    device: str,
    task: str | list[str],
    model: str | list[str] | None,
    downstream_wrapper: str | list[str] | None,
) -> None:
    """Raise ``ValueError`` on invalid device / task / model names."""
    if device not in ALL_DEVICES:
        raise ValueError(f"Unknown device {device!r}. Choose from: {ALL_DEVICES}")
    tasks = [task] if isinstance(task, str) else task
    device_tasks = set(TASKS.get(device, [])) | set(UNVALIDATED_TASKS.get(device, []))
    if not device_tasks:
        raise ValueError(
            f"No tasks ship for device {device!r} yet. "
            f"Devices with tasks: {[d for d in ALL_DEVICES if TASKS.get(d) or UNVALIDATED_TASKS.get(d)]}"
        )
    valid_tasks = {"all", "all_multi_dataset"} | device_tasks
    for t in tasks:
        if t not in valid_tasks:
            raise ValueError(
                f"Unknown task {t!r} for device {device!r}. "
                f"Choose from: {sorted(valid_tasks)}"
            )
    if model is not None:
        models = [model] if isinstance(model, str) else model
        valid_models = {
            "all",
            "all_classic",
            "all_fm",
            "all_baseline",
        } | set(ALL_MODELS)
        for m in models:
            if m not in valid_models:
                raise ValueError(
                    f"Unknown model {m!r}. Choose from: {sorted(valid_models)}"
                )
    if downstream_wrapper is not None:
        wrappers = (
            [downstream_wrapper]
            if isinstance(downstream_wrapper, str)
            else downstream_wrapper
        )
        valid_wrappers = {"all"} | set(ALL_DOWNSTREAM_WRAPPERS.keys())
        for w in wrappers:
            if w not in valid_wrappers:
                raise ValueError(
                    f"Unknown downstream wrapper {w!r}. "
                    f"Choose from: {sorted(valid_wrappers)}"
                )
