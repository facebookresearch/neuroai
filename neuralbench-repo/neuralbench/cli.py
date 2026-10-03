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
import os
import sys
import traceback
import typing as tp

from neuralbench.experiment_config import build_experiment_configs
from neuralbench.registry import (
    ALL_DEVICES,
    ALL_DOWNSTREAM_WRAPPERS,
    ALL_MODELS,
    ALL_TASKS,
    ALL_UNVALIDATED_TASKS,
    _format_datasets_epilog,
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
    seed: int | list[int] | None = None,
    wandb_paper_summary: bool = False,
    multi_task: bool = False,
    gpus: str | None = None,
    official_source_root: str | None = None,
    text_data_dir: str | None = None,
    output_dir: str = "outputs",
    workers: int = 4,
    batch_size: int = 8,
    text_batch_size: int = 16,
    epochs: int = 5,
    gradient_accumulation_steps: int = 1,
    log_interval: int = 10,
    learning_rate: float = 5e-4,
    no_text_loss: bool = False,
) -> list[dict[str, tp.Any]]:
    """Run one or more NeuralBench experiments from Python.

    This is the programmatic equivalent of the ``neuralbench`` CLI: it
    assembles experiment configs from the same YAML files and launches them.
    For a model built outside this repo, and for the results of the runs in
    hand, see :func:`neuralbench.evaluate_model`.

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
        Adaptation-strategy preset name(s) from
        ``defaults/downstream_wrappers.yaml``, or ``"all"``.  Swept over
        foundation models only.
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
        One result dict per experiment, with ``plot_cached=True`` only.  Every
        other mode launches experiments and returns an empty list, the results
        being written to the results folder.
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

    selected_models = [model] if isinstance(model, str) else list(model or [])
    from neuralbench.instruction_models import load_backend

    instruction_backend = (
        load_backend(selected_models[0]) if len(selected_models) == 1 else None
    )
    if instruction_backend is not None:
        if grid or plot_cached or downstream_wrapper is not None:
            raise ValueError(
                "Instruction models use their native adaptation protocol and do not "
                "use the encoder grid, cached plots, or downstream wrappers."
            )
        selected_seed = seed[0] if isinstance(seed, list) else (seed or 1337)
        return instruction_backend.run_from_neuralbench(
            device=device,
            task=task,
            checkpoint=checkpoint,
            dataset=dataset,
            debug=debug,
            force=force,
            retry=retry,
            prepare=prepare,
            download=download,
            multi_task=multi_task,
            gpus=gpus,
            official_source_root=official_source_root,
            text_data_dir=text_data_dir,
            output_dir=output_dir,
            workers=workers,
            batch_size=batch_size,
            text_batch_size=text_batch_size,
            epochs=epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_interval=log_interval,
            learning_rate=learning_rate,
            seed=selected_seed,
            no_text_loss=no_text_loss,
        )

    configs = build_experiment_configs(
        device,
        task,
        model=model,
        dataset=dataset,
        checkpoint=checkpoint,
        downstream_wrapper=downstream_wrapper,
        grid=grid,
        debug=debug,
        force=force,
        retry=retry,
        prepare=prepare,
        download=download,
        quiet=plot_cached,
    )

    if download:
        return []

    if plot_cached:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from neuralbench.main import BenchmarkAggregator

    agg = BenchmarkAggregator(
        # ConfDicts, which the pydantic model coerces into Experiment instances.
        experiments=configs,  # type: ignore[arg-type]
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
        help=(
            "Adaptation strategy preset(s) to sweep over; applied to foundation "
            "models only."
        ),
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
        "--seed",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Override the seed grid. Use one value to run a single seed "
            "(e.g. --seed 33), or multiple values to run a smaller seed list."
        ),
    )
    parser.add_argument(
        "--wandb-paper-summary",
        action="store_true",
        help=(
            "After all selected seeds finish, upload a quiet W&B summary run "
            "with mean/std test metrics for manuscript tables."
        ),
    )
    parser.add_argument(
        "--multi-task",
        action="store_true",
        help="For instruction models, jointly tune the configured multi-task suite.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for instruction-model DDP, e.g. 0,1,2,3.",
    )
    parser.add_argument("--official-source-root", type=str, default=None)
    parser.add_argument("--text-data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--no-text-loss",
        action="store_true",
        help="Disable the auxiliary text loss for instruction models.",
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
            seed=args.seed,
            wandb_paper_summary=args.wandb_paper_summary,
            multi_task=args.multi_task,
            gpus=args.gpus,
            official_source_root=args.official_source_root,
            text_data_dir=args.text_data_dir,
            output_dir=args.output_dir,
            workers=args.workers,
            batch_size=args.batch_size,
            text_batch_size=args.text_batch_size,
            epochs=args.epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_interval=args.log_interval,
            learning_rate=args.learning_rate,
            no_text_loss=args.no_text_loss,
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
