# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Config assembly: mode overlays, grid expansion, and experiment preparation.

This module transforms base YAML configs into concrete per-experiment
configs ready for ``BenchmarkAggregator``.
"""

import logging
import os
import shutil
import typing as tp
from itertools import product
from pathlib import Path
from warnings import warn

import torch
import yaml
from exca import ConfDict

from neuralbench.config_manager import get_config
from neuralbench.registry import (
    DEBUG_STUDY_QUERIES,
    DEFAULTS_DIR,
    _resolve_model_config_path,
    _resolve_task_dir,
    load_yaml_config,
)

LOGGER = logging.getLogger(__name__)

#: A registered model name, an inline config dict (:mod:`neuralbench.evaluate`),
#: or ``None`` for the default model of ``defaults/config.yaml``.
ModelSpec = str | dict[str, tp.Any] | None

# ---------------------------------------------------------------------------
# Mode overlays
# ---------------------------------------------------------------------------


def apply_cluster(config: ConfDict, cluster: str | None) -> None:
    """Point every infra in *config* at *cluster*.

    ``None`` computes in-process (exca maps it to submitit's debug executor, so
    a job array runs inline and blocks); ``"auto"`` fans out to SLURM when one
    is available and falls back to local otherwise.

    The run and the extractor/target caches are set together: leaving a cache
    on SLURM while the run is local, or the reverse, is never what a caller
    means and fails confusingly on machines without a cluster.
    """
    config["infra.cluster"] = cluster
    if "data" in config:
        config["data.neuro.infra.cluster"] = cluster
    target_cfg = config.get("data", {}).get("target", {})
    if isinstance(target_cfg, dict) and "infra" in target_cfg:
        config["data.target.infra.cluster"] = cluster


def _apply_debug_overlay(config: ConfDict) -> None:
    """Apply debug-mode overrides: disable SLURM/W&B, reduce epochs and batches."""
    LOGGER.info("--- RUNNING IN DEBUG MODE ---")
    config["wandb_config"] = None
    apply_cluster(config, None)
    config["infra.gpus_per_node"] = 1
    config["infra.tasks_per_node"] = 1
    config["infra.slurm_use_srun"] = False
    if "data" in config:
        config["data.batch_size"] = 8
        config["trainer_config"] = {
            "strategy": "auto",
            "n_epochs": 2,
            "limit_train_batches": 5,
            "limit_val_batches": 5,
        }
        config["lightning_optimizer_config.scheduler"] = None
        config["data.num_workers"] = 0
        config["data.study.source.query"] = DEBUG_STUDY_QUERIES.get(
            config["data.study.source.name"], None
        )


def _apply_prepare_overlay(config: ConfDict) -> None:
    """Apply prepare-mode overrides: single run to warm the preprocessing cache."""
    LOGGER.info("--- RUNNING SINGLE EXPERIMENT TO PREPARE CACHE ---")
    apply_cluster(config, get_config().get("CLUSTER", "auto"))
    config["infra.gpus_per_node"] = 1
    config["infra.tasks_per_node"] = 1
    config["infra.slurm_use_srun"] = False
    config["data.neuro.infra.min_samples_per_job"] = 8
    target_cfg = config.get("data", {}).get("target", {})
    if isinstance(target_cfg, dict) and "infra" in target_cfg:
        config["data.target.infra.min_samples_per_job"] = 8
    if "trainer_config" in config:
        config["trainer_config"] = {
            "n_epochs": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
        }
    if config["wandb_config"] is not None:
        config["wandb_config.name"] = "prepare"


def _download_dataset(config: ConfDict) -> None:
    """Download the study dataset without running experiments."""
    LOGGER.info("--- DOWNLOADING DATASET ---")
    import neuralset as ns

    study_dict = dict(config["data"]["study"]["source"])
    study_dict["path"] = Path(study_dict["path"]) / study_dict["name"]
    ns.Study(**study_dict).download()


# ---------------------------------------------------------------------------
# Config merging
# ---------------------------------------------------------------------------


def merge_task_config(
    device: str,
    task_name: str,
    dataset: str | None = None,
    base: ConfDict | None = None,
) -> ConfDict:
    """Layer a task config, and an optional dataset variant, over the base defaults.

    *base* defaults to ``defaults/config.yaml``; callers pass their own copy of
    it when it already carries run-level overrides (checkpoint, W&B).
    """
    config = (
        ConfDict(load_yaml_config(DEFAULTS_DIR / "config.yaml"))
        if base is None
        else base.copy()
    )
    task_config_fname = _resolve_task_dir(device, task_name) / "config.yaml"
    config.update(load_yaml_config(task_config_fname))
    if dataset is not None:
        _merge_dataset_config(config, device, task_name, dataset)
    return config


def _merge_dataset_config(
    config: ConfDict, device: str, task_name: str, dataset: str
) -> None:
    """Layer a task's ``datasets/<dataset>.yaml`` over *config*, in place."""
    datasets_dir = _resolve_task_dir(device, task_name) / "datasets"
    dataset_fname = datasets_dir / f"{dataset}.yaml"
    if not dataset_fname.is_file():
        raise ValueError(
            f"Unknown dataset {dataset!r} for {device}/{task_name}. "
            f"Choose from: {sorted(p.stem for p in datasets_dir.glob('*.yaml'))}"
        )
    source_defaults = dict(config["data.study.source"])
    config.update(load_yaml_config(dataset_fname))
    # =replace= may wipe source; restore default path/infra
    for k, v in source_defaults.items():
        config["data.study.source"].setdefault(k, v)


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


def _expand_grid(
    config: ConfDict,
    grid: ConfDict,
    device: str,
    task_name: str,
    use_task_grid: bool,
    prepare: bool,
    debug: bool = False,
    quiet: bool = False,
) -> list[ConfDict]:
    """Expand the grid into a list of concrete experiment configs."""
    if not quiet:
        LOGGER.info("--- GRID CONFIGURATION ---")
    if use_task_grid and not prepare and not debug:
        if not quiet:
            LOGGER.info("--- USING TASK-SPECIFIC GRID ---")
        task_grid_fname = _resolve_task_dir(device, task_name) / "grid.yaml"
        task_grid = load_yaml_config(task_grid_fname)
        grid.update(task_grid)

    flat_grid = grid.flat()
    if not quiet:
        LOGGER.info("Grid:\n%s", yaml.dump(flat_grid, default_flow_style=None).rstrip())
    grid_product = list(
        dict(zip(flat_grid.keys(), v)) for v in product(*flat_grid.values())
    )

    configs = []
    for params in grid_product:
        updated_config = config.copy()
        # a multi-key preset, not a config key: merge whole, not per-axis
        overlay = params.pop("_adaptation_overlay", None)
        updated_config.update(params)
        if overlay is not None:
            updated_config.update(overlay)
            # re-null: debug's short run is under OneCycleLR warmup (pct_start div-by-zero)
            if debug:
                updated_config["lightning_optimizer_config.scheduler"] = None
        configs.append(updated_config)

    return configs


# ---------------------------------------------------------------------------
# Single-task and multi-task config preparation
# ---------------------------------------------------------------------------


def _prepare_single_task_config(
    config: ConfDict,
    grid: ConfDict,
    device: str,
    task_name: str,
    use_task_grid: bool,
    debug: bool,
    force: bool,
    prepare: bool,
    download: bool,
    dataset_name: str | None = None,
    quiet: bool = False,
    retry: bool = False,
) -> list[ConfDict]:
    """Assemble experiment configs for a single task, applying mode overlays and grid expansion."""
    config["task_name"] = task_name
    if config.get("wandb_config") is not None:
        config["wandb_config.group"] = f"{device}/{task_name}"

    if debug:
        _apply_debug_overlay(config)
    if force:
        if not quiet:
            LOGGER.info("--- USING INFRA.MODE=FORCE ---")
        config["infra.mode"] = "force"
    elif retry:
        if not quiet:
            LOGGER.info("--- USING INFRA.MODE=RETRY ---")
        config["infra.mode"] = "retry"
    if prepare:
        _apply_prepare_overlay(config)
    if download:
        _download_dataset(config)
        return [config]

    return _expand_grid(
        config, grid, device, task_name, use_task_grid, prepare, debug=debug, quiet=quiet
    )


def prepare_task_configs(
    config: ConfDict,
    grid: ConfDict,
    device: str,
    task_name: str,
    use_task_grid: bool,
    debug: bool,
    force: bool,
    prepare: bool,
    download: bool,
    models: tp.Sequence[ModelSpec],
    datasets: list[str | None] | None = None,
    quiet: bool = False,
    retry: bool = False,
) -> list[ConfDict]:
    """Run a specific neuralbench task with given configuration."""
    if not quiet:
        LOGGER.info("=== PREDICTING %s FROM %s ===", task_name, device)

    if datasets is None:
        datasets = [None]

    config = merge_task_config(device, task_name, base=config)

    configs = []
    for model in models:
        exp_config = config.copy()
        if model is not None:
            # A dict is an inline config: out-of-tree models reach the
            # benchmark through neuralbench.evaluate and have no models/*.yaml.
            if isinstance(model, dict):
                model_config = model
                label = model.get("brain_model_name", "inline config")
            else:
                model_config = load_yaml_config(_resolve_model_config_path(model)) or {}
                label = model
            if not quiet:
                LOGGER.info("--- USING MODEL %s ---", label)
            exp_config.update(model_config)

        for dataset_name in datasets:
            dataset_exp_config = exp_config.copy()
            if dataset_name is not None:
                if not quiet:
                    LOGGER.info("~~~ USING DATASET: %s ~~~", dataset_name)
                _merge_dataset_config(dataset_exp_config, device, task_name, dataset_name)

            exp_configs = _prepare_single_task_config(
                dataset_exp_config,
                grid,
                device,
                task_name,
                use_task_grid,
                debug,
                force,
                prepare,
                download,
                dataset_name,
                quiet=quiet,
                retry=retry,
            )
            configs.extend(exp_configs)

    return configs


def build_experiment_configs(
    device: str,
    task: str | list[str],
    *,
    model: str | list[str] | dict[str, tp.Any] | None = None,
    dataset: str | list[str] | None = None,
    checkpoint: str | None = None,
    downstream_wrapper: str | list[str] | None = None,
    grid: bool = False,
    debug: bool = False,
    force: bool = False,
    retry: bool = False,
    prepare: bool = False,
    download: bool = False,
    quiet: bool = False,
) -> list[ConfDict]:
    """Assemble one experiment config per (task, dataset, model, grid point).

    The single home for turning benchmark selections into configs, shared by
    :func:`neuralbench.cli.run_benchmark` and
    :mod:`neuralbench.evaluate`.  *model* is a registered name (or ``"all"``
    and friends) as on the CLI, or an already-loaded config dict for a model
    with no ``models/*.yaml`` -- see :mod:`neuralbench.evaluate`.

    Arguments mirror :func:`neuralbench.cli.run_benchmark`; see its docstring.
    """
    from neuralbench.config_manager import _ensure_initialized
    from neuralbench.registry import (
        ALL_DOWNSTREAM_WRAPPERS,
        DEVICE_FM_MODELS,
        FM_MODELS,
        _expand_models,
        _resolve_datasets,
        _resolve_tasks,
        _validate_inputs,
    )

    _ensure_initialized()
    inline_model = model if isinstance(model, dict) else None
    _validate_inputs(device, task, None if inline_model else model, downstream_wrapper)  # type: ignore[arg-type]
    _warn_slurm_partition(debug, prepare=prepare, download=download)
    _warn_unsupported_gpu()

    config = ConfDict(load_yaml_config(DEFAULTS_DIR / "config.yaml"))
    # A blank W&B host means logging is off for the whole pipeline (mirrors
    # the debug-mode overlay).
    wandb_cfg = config.get("wandb_config")
    host = wandb_cfg.get("host") if isinstance(wandb_cfg, dict) else None
    if not host:
        config["wandb_config"] = None

    grid_conf = ConfDict(load_yaml_config(DEFAULTS_DIR / "grid.yaml"))
    if prepare or debug:
        grid_conf["seed"] = [grid_conf["seed"][0]]

    if checkpoint is not None:
        config["pretrained_weights_fname"] = checkpoint

    overlays: list[dict[str, tp.Any]] | None = None
    if downstream_wrapper is not None:
        wrappers = (
            [downstream_wrapper]
            if isinstance(downstream_wrapper, str)
            else list(downstream_wrapper)
        )
        if wrappers == ["all"]:
            wrappers = list(ALL_DOWNSTREAM_WRAPPERS.keys())
        overlays = [ALL_DOWNSTREAM_WRAPPERS[name] for name in wrappers]

    tasks = _resolve_tasks(device, task)

    configs: list[ConfDict] = []
    task_iter: tp.Iterable[str] = tasks
    if prepare and len(tasks) > 1:
        from tqdm import tqdm

        task_iter = tqdm(tasks, desc="Preparing tasks")

    for task_name in task_iter:
        # Resolve models per-task so the `all` / `all_baseline` aliases pick
        # only the task-appropriate sklearn baseline (via FEATURE_BASED_BY_TASK)
        # instead of launching every pipeline on every task.
        models: list[ModelSpec] = (
            [inline_model]
            if inline_model is not None
            else list(_expand_models(model, device=device, task_name=task_name))  # type: ignore[arg-type]
        )
        datasets = _resolve_datasets(device, task_name, dataset)
        # adaptation needs a pretrained backbone: non-FMs get an overlay-free grid
        model_groups: list[tuple[ConfDict, list[ModelSpec]]]
        if overlays is not None:
            # An out-of-tree model is a backbone by assumption: it is not in the
            # in-tree FM list, but adaptation is exactly why it is being run.
            fm_names = set(DEVICE_FM_MODELS.get(device, FM_MODELS))
            inline = inline_model is not None
            fm_models = [m for m in models if inline or m in fm_names]
            other_models = [m for m in models if not inline and m not in fm_names]
            if not fm_models:
                LOGGER.warning(
                    "Adaptation wrappers requested (-w) but no foundation model "
                    "selected for task %r; running without adaptation.",
                    task_name,
                )
            model_groups = []
            if fm_models:
                fm_grid = grid_conf.copy()
                fm_grid["_adaptation_overlay"] = overlays
                model_groups.append((fm_grid, fm_models))
            if other_models:
                model_groups.append((grid_conf, other_models))
        else:
            model_groups = [(grid_conf, models)]
        for group_grid, group_models in model_groups:
            configs.extend(
                prepare_task_configs(
                    config.copy(),
                    group_grid,
                    device,
                    task_name,
                    grid,
                    debug,
                    force,
                    prepare,
                    download,
                    group_models,
                    datasets,
                    quiet=quiet,
                    retry=retry,
                )
            )
    return configs


# ---------------------------------------------------------------------------
# SLURM warning
# ---------------------------------------------------------------------------


def _warn_slurm_partition(
    debug: bool, *, prepare: bool = False, download: bool = False
) -> None:
    """Warn when SLURM is detected but no partition is configured.

    Skipped when the run does not actually submit jobs (``debug``,
    ``prepare``, ``download``, or ``CLUSTER`` forcing local execution).  The
    warning surfaces the resolved config path, honoring the
    ``NEURALBENCH_CONFIG`` environment variable.
    """
    from neuralbench.config_manager import get_default_config_path

    if debug or prepare or download:
        return
    config = get_config()
    # CLUSTER=null forces fully local execution, so no SLURM jobs are submitted.
    if config.get("CLUSTER", "auto") is None:
        return
    if not config.get("SLURM_PARTITION") and shutil.which("srun") is not None:
        config_path = os.environ.get("NEURALBENCH_CONFIG") or str(
            get_default_config_path()
        )
        warn(
            "SLURM is available on this machine but SLURM_PARTITION is not set "
            f"in your neuralbench config ({config_path}). Non-debug runs will "
            f"fail when submitting jobs. Either set SLURM_PARTITION in "
            f'{config_path}, set "CLUSTER": null to run locally, or use --debug.'
        )


# ---------------------------------------------------------------------------
# GPU capability warning
# ---------------------------------------------------------------------------


def _warn_unsupported_gpu() -> None:
    """Warn when a visible GPU is absent from the installed torch's arch list."""
    if not torch.cuda.is_available():
        return
    # Release wheels are SASS-only (no PTX JIT), so a device needs a listed
    # arch of its own major with minor <= its own.
    built = {
        (int(sm[:-1]), int(sm[-1]))
        for arch in torch.cuda.get_arch_list()
        if (sm := arch.removeprefix("sm_")).isdigit()
    }
    if not built:
        return
    warned: set[tuple[int, int]] = set()
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        major, minor = capability
        if capability in warned or any(
            major == bmajor and minor >= bminor for bmajor, bminor in built
        ):
            continue
        warned.add(capability)
        # sm_50-sm_60 survive only in the CUDA 12.6 wheels; PyPI's default build
        # tracks the newest CUDA, which drops them.
        if capability < min(built):
            fix = (
                "reinstall from a CUDA 12.6 build, which still ships sm_50-sm_90: "
                "pip install --force-reinstall torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu126"
            )
        else:
            fix = "upgrade torch: pip install --upgrade torch torchvision torchaudio"
        warn(
            f"{torch.cuda.get_device_name(index)} has CUDA capability "
            f"sm_{major}{minor}, which torch {torch.__version__} was not built for "
            f"(built for {', '.join(f'sm_{a}{b}' for a, b in sorted(built))}). "
            f"GPU runs will fail; to fix, {fix}"
        )
