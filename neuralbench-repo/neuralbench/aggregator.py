# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Orchestrate multiple benchmark experiments and collect/plot results."""

from __future__ import annotations

import json
import logging
import typing as tp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import Field
from tqdm import tqdm

import neuralset as ns

from .plots.benchmark import plot_all_results
from .plots.tables import print_skip_table

if tp.TYPE_CHECKING:
    from .main import Experiment

LOGGER = logging.getLogger(__name__)


def _default_output_dir() -> str:
    """Resolve the default output directory under the user-configured ``SAVE_DIR``.

    Imported lazily so that ``BenchmarkAggregator`` can be imported before
    ``neuralbench.config_manager`` has been initialised (avoids triggering
    interactive setup at import time).
    """
    from neuralbench.config_manager import get_config

    return str(Path(get_config()["SAVE_DIR"]) / "outputs")


class BenchmarkAggregator(ns.BaseModel):
    """Orchestrate multiple :class:`Experiment` runs and visualise results.

    Experiments are submitted (possibly via Slurm) with :meth:`prepare`,
    collected with :meth:`_collect_results`, and plotted/tabled via the
    functions in :mod:`neuralbench.plots.benchmark`.
    """

    experiments: list["Experiment"]
    max_workers: int = 256
    collect_max_workers: int = 32
    debug: bool = False

    output_dir: str = Field(default_factory=_default_output_dir)

    loss_to_metric_mapping: dict[str, str] = {
        "CrossEntropyLoss": "test/bal_acc",
        "BCEWithLogitsLoss": "test/f1_score_macro",
        "MSELoss": "test/pearsonr",
        "ClipLoss": "test/full_retrieval/top5_acc_subject-agg",
    }

    def prepare(self) -> None:
        n_total = len(self.experiments)
        statuses = [exp.infra.status() for exp in self.experiments]
        n_completed = statuses.count("completed")
        n_running = statuses.count("running")
        n_cached = n_completed + n_running
        is_force = self.experiments[0].infra.mode == "force"
        if n_cached == n_total:
            parts = []
            if n_completed:
                parts.append(f"{n_completed} completed")
            if n_running:
                parts.append(f"{n_running} still running")
            if is_force:
                LOGGER.info(
                    "All %d experiment(s) already cached/running (%s). "
                    "Re-running with --force.",
                    n_total,
                    ", ".join(parts),
                )
            else:
                LOGGER.info(
                    "All %d experiment(s) already cached/running (%s). "
                    "Nothing to launch. Use --force to re-run.",
                    n_total,
                    ", ".join(parts),
                )
                return

        if self.debug:
            for experiment in self.experiments:
                experiment.run()
        else:
            tmp = self.experiments[0].infra.clone_obj()
            with tmp.infra.job_array(max_workers=self.max_workers) as tasks:
                tasks.extend(self.experiments)

    @staticmethod
    def _process_one_experiment(
        experiment: "Experiment", cached_only: bool
    ) -> tuple[dict[str, tp.Any] | None, str, str]:
        """Process one experiment, returning ``(result_or_None, task, model)``."""
        task = experiment.task_name
        model = experiment.brain_model_name
        if cached_only and experiment.infra.status() != "completed":
            return None, task, model
        out = experiment.run()
        out["task_name"] = experiment.task_name
        study = experiment.data.study
        if isinstance(study, ns.Chain):
            out["dataset_name"] = type(study[0]).__name__
        else:
            out["dataset_name"] = type(study).__name__
        out["brain_model_name"] = experiment.brain_model_name
        out["loss"] = {"name": type(experiment.loss).__name__}
        out["seed"] = experiment.seed
        return out, task, model

    def _collect_results(self, cached_only: bool = False) -> list[dict[str, tp.Any]]:
        """Gather experiment results, optionally skipping uncached ones.

        When *cached_only* is ``True`` the work is purely I/O-bound (pickle
        loads), so experiments are processed in parallel using threads.
        """
        if cached_only:
            return self._collect_results_parallel()
        return self._collect_results_sequential(cached_only=False)

    def _collect_results_sequential(
        self, cached_only: bool = False
    ) -> list[dict[str, tp.Any]]:
        results: list[dict[str, tp.Any]] = []
        total: dict[tuple[str, str], int] = {}
        skipped: dict[tuple[str, str], int] = {}
        for experiment in self.experiments:
            out, task, model = self._process_one_experiment(experiment, cached_only)
            key = (task, model)
            total[key] = total.get(key, 0) + 1
            if out is None:
                skipped[key] = skipped.get(key, 0) + 1
            else:
                results.append(out)
        if skipped:
            print_skip_table(total, skipped)
        return results

    def _collect_results_parallel(self) -> list[dict[str, tp.Any]]:
        results: list[dict[str, tp.Any]] = []
        total: dict[tuple[str, str], int] = {}
        skipped: dict[tuple[str, str], int] = {}
        n_workers = min(self.collect_max_workers, len(self.experiments))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(self._process_one_experiment, exp, True): exp
                for exp in self.experiments
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Collecting cached results",
            ):
                out, task, model = future.result()
                key = (task, model)
                total[key] = total.get(key, 0) + 1
                if out is None:
                    skipped[key] = skipped.get(key, 0) + 1
                else:
                    results.append(out)
        if skipped:
            print_skip_table(total, skipped)
        return results

    def _save_computational_stats(self, results: list[dict[str, tp.Any]]) -> Path:
        """Write per-experiment computational stats to JSON for later analysis."""
        _COMP_KEYS = (
            "training_time_s",
            "peak_gpu_memory_mb",
            "peak_cpu_memory_mb",
            "n_total_params",
            "n_trainable_params",
        )
        stats = []
        for r in results:
            entry: dict[str, tp.Any] = {
                "task_name": r.get("task_name"),
                "dataset_name": r.get("dataset_name"),
                "model": r.get("brain_model_name", "unknown"),
                "seed": r.get("seed"),
            }
            entry.update({k: r.get(k) for k in _COMP_KEYS})
            stats.append(entry)
        out_path = Path(self.output_dir) / "other" / "computational_stats.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        LOGGER.info("Saved computational stats to %s", out_path)
        return out_path

    def run(self, cached_only: bool = False) -> list[dict[str, tp.Any]]:
        results = self._collect_results(cached_only=cached_only)
        if not results:
            if cached_only:
                # --plot-cached only finds canonical (non-debug) runs: --debug
                # uses a reduced config (fewer epochs, smaller batch, subset
                # query) and is therefore cached under a different key.
                LOGGER.info(
                    "No cached results found. Nothing to plot.\n"
                    "  --plot-cached looks for completed canonical runs only; "
                    "--debug runs are cached separately and are not picked up.\n"
                    "  To populate the cache, run the canonical command first "
                    "(drop --debug), e.g.:\n"
                    "      neuralbench <device> <task> -m <model>\n"
                    "  then re-run with --plot-cached:\n"
                    "      neuralbench <device> <task> -m <model> --plot-cached"
                )
            else:
                LOGGER.info("No results found. Nothing to plot.")
            return []
        self._save_computational_stats(results)
        plot_all_results(results, self.loss_to_metric_mapping, self.output_dir)
        return results
