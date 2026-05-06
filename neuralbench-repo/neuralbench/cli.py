# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CLI entry point and programmatic API for NeuralBench.

Parse command-line arguments (``run_benchmark_cli``) or call the Python
API directly (``run_benchmark``).  Task/model discovery, validation, and
config assembly are delegated to :mod:`neuralbench.registry` and
:mod:`neuralbench.experiment_config`.
"""

import argparse
import logging
import sys
import traceback
import typing as tp

from exca import ConfDict

from neuralbench.experiment_config import (
    _warn_slurm_partition,
    prepare_task_configs,
)
from neuralbench.registry import (
    ALL_DEVICES,
    ALL_DOWNSTREAM_WRAPPERS,
    ALL_MODELS,
    ALL_TASKS,
    ALL_UNVALIDATED_TASKS,
    DEFAULTS_DIR,
    _expand_models,
    _format_datasets_epilog,
    _resolve_datasets,
    _resolve_tasks,
    _validate_inputs,
    load_yaml_config,
)

logger = logging.getLogger(__name__)


def run_benchmark(
    device: str,
    task: str | list[str],
    *,
    model: str | list[str] | None = None,
    dataset: str | list[str] | None = None,
    checkpoint: str | None = None,
    downstream_wrapper: str | list[str] | None = None,
    grid: bool = False,
    debug: bool = False,
    force: bool = False,
    retry: bool = False,
    prepare: bool = False,
    download: bool = False,
    plot_cached: bool = False,
) -> list[dict[str, tp.Any]]:
    """Run one or more NeuralBench experiments from Python.

    This is the programmatic equivalent of the ``neuralbench`` CLI.
    It assembles experiment configs from the same YAML files and returns
    test-metric dictionaries when running in debug mode.

    Parameters
    ----------
    device : str
        Brain recording device (``"eeg"``, ``"meg"``, ``"fmri"``, ...).
    task : str or list of str
        Task name(s), ``"all"``, or ``"all_multi_dataset"``.
    model : str or list of str or None
        Predefined model name(s), ``"all"``, ``"all_classic"``, ``"all_fm"``,
        ``"all_baseline"`` (chance / dummy / classical sklearn pipelines),
        or ``None`` (uses default model from ``config.yaml``).
    dataset : str or list of str or None
        Dataset variant(s) or ``"all"``. ``None`` uses the base config.
    checkpoint : str or None
        Path to a model checkpoint to reload.
    downstream_wrapper : str or list of str or None
        Downstream wrapper name(s) or ``"all"``.
    grid : bool
        Expand the task-specific hyperparameter grid.
    debug : bool
        Run locally with a reduced config (2 epochs, 5 batches).
    force : bool
        Force re-running experiments.
    retry : bool
        Retry failed experiments while keeping completed results.
    prepare : bool
        Run a single experiment to warm the preprocessing cache.
    download : bool
        Only download the dataset; do not run experiments.
    plot_cached : bool
        Generate plots and tables from cached results only, without
        running any new experiments.

    Returns
    -------
    list of dict
        One result dict per experiment (empty when experiments are
        submitted asynchronously via Slurm).
    """
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    # fontTools.subset emits very chatty INFO logs while matplotlib embeds
    # font subsets into vector outputs (e.g. PDFs); mute them.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    logging.getLogger("fontTools.ttLib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    # exca and neuralset attach their own StreamHandlers; disable propagation
    # to the root logger to avoid duplicate log lines.
    logging.getLogger("exca").propagate = False
    logging.getLogger("neuralset").propagate = False

    if plot_cached and (force or retry or prepare):
        raise ValueError(
            "Cannot use force, retry, or prepare flags when plotting cached results."
        )

    from neuralbench.config_manager import _ensure_initialized

    _ensure_initialized()
    _validate_inputs(device, task, model, downstream_wrapper)
    _warn_slurm_partition(debug, prepare=prepare, download=download)

    # --- base config & grid ---
    default_config = load_yaml_config(DEFAULTS_DIR / "config.yaml")
    config = ConfDict(default_config)

    default_grid = load_yaml_config(DEFAULTS_DIR / "grid.yaml")
    grid_conf = ConfDict(default_grid)

    if prepare or debug:
        grid_conf["seed"] = [grid_conf["seed"][0]]

    if checkpoint is not None:
        config["pretrained_weights_fname"] = checkpoint

    # --- downstream wrappers ---
    if downstream_wrapper is not None:
        wrappers = (
            [downstream_wrapper]
            if isinstance(downstream_wrapper, str)
            else list(downstream_wrapper)
        )
        if wrappers == ["all"]:
            wrappers = list(ALL_DOWNSTREAM_WRAPPERS.keys())
        wrapper_configs = [ALL_DOWNSTREAM_WRAPPERS[name] for name in wrappers]
        grid_conf["downstream_model_wrapper"] = wrapper_configs

    # --- tasks ---
    tasks = _resolve_tasks(device, task)

    # --- assemble experiment configs ---
    configs: list[tp.Any] = []
    task_iter: tp.Iterable[str] = tasks
    if prepare and len(tasks) > 1:
        from tqdm import tqdm

        task_iter = tqdm(tasks, desc="Preparing tasks")

    for task_name in task_iter:
        # Resolve models per-task so the `all` / `all_baseline` aliases pick
        # only the task-appropriate sklearn baseline (via FEATURE_BASED_BY_TASK)
        # instead of launching every pipeline on every task.
        models = _expand_models(model, device=device, task_name=task_name)
        datasets = _resolve_datasets(device, task_name, dataset)
        task_configs = prepare_task_configs(
            config.copy(),
            grid_conf,
            device,
            task_name,
            grid,
            debug,
            force,
            prepare,
            download,
            models,
            datasets,
            quiet=plot_cached,
            retry=retry,
        )
        configs.extend(task_configs)

    if download:
        return []

    if plot_cached:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from neuralbench.main import BenchmarkAggregator

    agg = BenchmarkAggregator(
        experiments=configs,
        debug=debug,
    )

    if not plot_cached:
        agg.prepare()

    results = []
    if plot_cached:
        logger.info("--- PREPARING GLOBAL PLOTS AND TABLES ---")
        results = agg.run(cached_only=True)

    return results


def run_benchmark_cli() -> None:
    """CLI entry point for ``neuralbench``.

    Parses command-line arguments and delegates to :func:`run_benchmark`.
    """
    parser = argparse.ArgumentParser(
        description="Run neuralbench.",
        epilog=_format_datasets_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "device",
        type=str,
        choices=ALL_DEVICES,
        help="Brain device on which the desired task relies.",
    )
    parser.add_argument(
        "task",
        type=str,
        nargs="+",
        choices=["all", "all_multi_dataset"] + ALL_TASKS + ALL_UNVALIDATED_TASKS,
        help=(
            "Task(s) to run. Use 'all' to run all tasks, "
            "'all_multi_dataset' to run only tasks with multiple dataset variants, "
            "or specify one or more task names."
        ),
    )
    parser.add_argument(
        "-g", "--grid", action="store_true", help="Run task-specific grid."
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Run in debug mode (locally and smaller config, with infra.mode='force').",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-f", "--force", action="store_true", help="Force rerunning of experiment."
    )
    mode_group.add_argument(
        "-r",
        "--retry",
        action="store_true",
        help="Retry failed experiments (keep completed results).",
    )
    parser.add_argument(
        "-p",
        "--prepare",
        action="store_true",
        help="Run single experiment to prepare cache.",
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="*",
        choices=["all", "all_classic", "all_fm", "all_baseline"] + ALL_MODELS,
        help="Override config to use one or more predefined models. Multiple models will be run in the grid.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        help=(
            "Path to a model checkpoint to reload. If this follows the format "
            "`wandb:entity/project/grid`, it will be used to find all available checkpoint paths "
            "for a specific wandb grid."
        ),
    )
    parser.add_argument(
        "-w",
        "--downstream-wrapper",
        nargs="*",
        choices=["all"] + list(ALL_DOWNSTREAM_WRAPPERS.keys()),
        help="Override/add a model wrapper for the downstream tasks.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the study. The experiment(s) will not be run.",
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        help="Launch pdb on exception.",
    )
    parser.add_argument(
        "--plot-cached",
        action="store_true",
        help="Plot from cached results only, without running any experiments.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Specify a dataset variant for the task. "
            "Use 'all' to run on all available datasets. "
            "If provided, will load dataset-specific overrides from datasets/{dataset}.yaml "
            "and merge them with the base config.yaml. "
            "Example: --dataset steyrl2016 or --dataset all"
        ),
    )
    args = parser.parse_args()

    try:
        run_benchmark(
            device=args.device,
            task=args.task,
            model=args.model,
            dataset=args.dataset,
            checkpoint=args.checkpoint,
            downstream_wrapper=args.downstream_wrapper,
            grid=args.grid,
            debug=args.debug,
            force=args.force,
            retry=args.retry,
            prepare=args.prepare,
            download=args.download,
            plot_cached=args.plot_cached,
        )
    except Exception:
        if not args.pdb:
            raise
        import pdb

        tb = sys.exc_info()[2]
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    run_benchmark_cli()
