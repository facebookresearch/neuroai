# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate ``core_non_eeg_bar_chart.png/pdf`` from cached MEG + fMRI results.

The main ``neuralbench`` CLI processes one device per invocation.  This
script assembles cached configs for each non-EEG device, collects their
results via :class:`neuralbench.main.BenchmarkAggregator`, concatenates
them, and drives :func:`neuralbench.plots.benchmark.plot_non_eeg_bar_chart`
directly -- without touching the CLI or running the full plotting
pipeline.

Usage
-----
.. code-block:: bash

    python -m neuralbench.plots.plot_non_eeg
"""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import matplotlib
import seaborn as sns
from exca import ConfDict

from neuralbench.experiment_config import prepare_task_configs
from neuralbench.main import BenchmarkAggregator
from neuralbench.plots._filters import filter_core_dataset
from neuralbench.plots.benchmark import (
    build_results_df,
    plot_non_eeg_bar_chart,
)
from neuralbench.registry import (
    DEFAULTS_DIR,
    _expand_models,
    _resolve_datasets,
    _resolve_tasks,
    load_yaml_config,
)

LOGGER = logging.getLogger(__name__)

NON_EEG_DEVICES: tuple[str, ...] = ("meg", "fmri")


def _task_configs_valid(task_configs: list[tp.Any]) -> bool:
    """Return True when *task_configs* deserialize cleanly into experiments.

    Used to skip tasks whose required study isn't installed in the current
    environment (e.g. private datasets like ``Levy2025BrainMeg`` that
    aren't shipped with the public ``neuralfetch``). Constructs a
    throwaway :class:`BenchmarkAggregator` purely to trigger pydantic /
    exca discriminator resolution; the instance is discarded.
    """
    try:
        BenchmarkAggregator(experiments=task_configs)
    except ImportError:
        return False
    return True


def collect_cached_results(
    devices: tuple[str, ...] = NON_EEG_DEVICES,
) -> tuple[list[dict[str, tp.Any]], BenchmarkAggregator | None]:
    """Assemble experiment configs per device and collect cached results.

    Returns the concatenated list of result dicts along with the last
    :class:`BenchmarkAggregator` instance (used downstream to read field
    defaults such as ``loss_to_metric_mapping`` and ``output_dir``).

    Tasks whose required study can't be resolved in the current
    environment (typically private datasets absent from the public
    ``neuralfetch``) are skipped with a warning so the rest of the
    benchmark can still be plotted.
    """
    base_config = ConfDict(load_yaml_config(DEFAULTS_DIR / "config.yaml"))
    grid_conf = ConfDict(load_yaml_config(DEFAULTS_DIR / "grid.yaml"))

    all_results: list[dict[str, tp.Any]] = []
    last_agg: BenchmarkAggregator | None = None
    for device in devices:
        tasks = _resolve_tasks(device, "all")
        configs: list[tp.Any] = []
        for task_name in tasks:
            models = _expand_models("all", device=device, task_name=task_name)
            datasets = _resolve_datasets(device, task_name, None)
            task_configs = prepare_task_configs(
                base_config.copy(),
                grid_conf,
                device,
                task_name,
                use_task_grid=False,
                debug=False,
                force=False,
                prepare=False,
                download=False,
                models=models,
                datasets=datasets,
                quiet=True,
            )
            if not _task_configs_valid(task_configs):
                LOGGER.warning(
                    "Skipping %s/%s: required study not available in this environment.",
                    device,
                    task_name,
                )
                continue
            configs.extend(task_configs)
        if not configs:
            LOGGER.warning("No configs assembled for %s; skipping.", device)
            continue
        agg = BenchmarkAggregator(experiments=configs)
        device_results = agg._collect_results(cached_only=True)
        LOGGER.info("Collected %d cached result(s) for %s.", len(device_results), device)
        all_results.extend(device_results)
        last_agg = agg
    return all_results, last_agg


def main(
    devices: tuple[str, ...] = NON_EEG_DEVICES,
    output_dir: Path | None = None,
) -> Path | None:
    """Entry point: collect cached results and write the non-EEG bar chart."""
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("exca").propagate = False
    logging.getLogger("neuralset").propagate = False

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
        }
    )
    sns.set_theme(context="paper", style="white")

    results, agg = collect_cached_results(devices)
    if not results or agg is None:
        raise SystemExit(
            "No cached results found for any non-EEG device. "
            "Run `neuralbench meg -t all -m all --plot-cached` and "
            "`neuralbench fmri -t all -m all --plot-cached` first."
        )

    df = build_results_df(results, agg.loss_to_metric_mapping)
    core = filter_core_dataset(df)

    # Default matches BenchmarkAggregator's layout: core_non_eeg_bar_chart is a
    # NeuralBench-Core single-dataset plot and lives under outputs/core/.
    # Caller can still override via ``output_dir`` (used e.g. by tests).
    out_dir = output_dir if output_dir is not None else Path(agg.output_dir) / "core"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = plot_non_eeg_bar_chart(core, out_dir, devices=devices)
    if written is None:
        LOGGER.warning(
            "plot_non_eeg_bar_chart produced no output "
            "(no rows matched non-EEG devices in the collected results)."
        )
    else:
        LOGGER.info("Wrote %s", written)
    return written


if __name__ == "__main__":
    main()
