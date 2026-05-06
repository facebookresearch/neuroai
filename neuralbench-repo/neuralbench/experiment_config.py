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
from itertools import product
from pathlib import Path
from warnings import warn

import yaml
from exca import ConfDict

from neuralbench.registry import (
    DEBUG_STUDY_QUERIES,
    _resolve_model_config_path,
    _resolve_task_dir,
    load_yaml_config,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mode overlays
# ---------------------------------------------------------------------------


def _apply_debug_overlay(config: ConfDict) -> None:
    """Apply debug-mode overrides: disable SLURM/W&B, reduce epochs and batches."""
    LOGGER.info("--- RUNNING IN DEBUG MODE ---")
    config["wandb_config"] = None
    config["infra.cluster"] = None
    config["infra.gpus_per_node"] = 1
    config["infra.tasks_per_node"] = 1
    config["infra.slurm_use_srun"] = False
    if "data" in config:
        config["data.neuro.infra.cluster"] = None
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
    config["infra.cluster"] = "auto"
    config["infra.gpus_per_node"] = 1
    config["infra.tasks_per_node"] = 1
    config["infra.slurm_use_srun"] = False
    # Parallel SLURM-based caching for the extractors (use "slurm" rather than
    # "auto" so that prepare_extractors launches them concurrently)
    config["data.neuro.infra.cluster"] = "slurm"
    config["data.neuro.infra.min_samples_per_job"] = 8
    target_cfg = config.get("data", {}).get("target", {})
    if isinstance(target_cfg, dict) and "infra" in target_cfg:
        config["data.target.infra.cluster"] = "slurm"
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
        updated_config.update(params)
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
    models: list[str | None],
    datasets: list[str | None] | None = None,
    quiet: bool = False,
    retry: bool = False,
) -> list[ConfDict]:
    """Run a specific neuralbench task with given configuration."""
    if not quiet:
        LOGGER.info("=== PREDICTING %s FROM %s ===", task_name, device)

    if datasets is None:
        datasets = [None]

    task_dir = _resolve_task_dir(device, task_name)
    task_config_fname = task_dir / "config.yaml"
    task_config = load_yaml_config(task_config_fname)
    config.update(task_config)

    configs = []
    for model_name in models:
        exp_config = config.copy()
        if model_name is not None:
            if not quiet:
                LOGGER.info("--- USING MODEL %s ---", model_name)
            model_config_fname = _resolve_model_config_path(model_name)
            model_config = load_yaml_config(model_config_fname)
            exp_config.update(model_config)

        for dataset_name in datasets:
            dataset_exp_config = exp_config.copy()
            if dataset_name is not None:
                if not quiet:
                    LOGGER.info("~~~ USING DATASET: %s ~~~", dataset_name)
                dataset_config_fname = task_dir / "datasets" / f"{dataset_name}.yaml"
                dataset_config = load_yaml_config(dataset_config_fname)
                dataset_exp_config.update(dataset_config)
                # =replace= may wipe source; restore default path/infra
                for k, v in config["data.study.source"].items():
                    dataset_exp_config["data.study.source"].setdefault(k, v)

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


# ---------------------------------------------------------------------------
# SLURM warning
# ---------------------------------------------------------------------------


def _warn_slurm_partition(
    debug: bool, *, prepare: bool = False, download: bool = False
) -> None:
    """Warn when SLURM is detected but no partition is configured.

    Skipped when the run does not actually submit jobs (``debug``,
    ``prepare``, ``download``).  The warning surfaces the resolved config
    path, honoring the ``NEURALBENCH_CONFIG`` environment variable.
    """
    import os
    import shutil

    from neuralbench.config_manager import get_config, get_default_config_path

    if debug or prepare or download:
        return
    config = get_config()
    if not config.get("SLURM_PARTITION") and shutil.which("srun") is not None:
        config_path = os.environ.get("NEURALBENCH_CONFIG") or str(
            get_default_config_path()
        )
        warn(
            "SLURM is available on this machine but SLURM_PARTITION is not set "
            f"in your neuralbench config ({config_path}). Non-debug runs will "
            f"fail when submitting jobs. Either set SLURM_PARTITION in "
            f"{config_path}, or use --debug to run locally."
        )
