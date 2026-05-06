# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NeuralBench plot orchestrator.

This module is the entry point for building every plot and table that
``neuralbench`` ships.  Public plot functions live in cohesive sibling
modules (``bar_charts``, ``non_eeg``, ``rank_chart``, ``ranking``,
``tables``, ``variability``, ``normalized_summary``); this file is the
thin orchestrator on top.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import seaborn as sns
from tqdm import tqdm

from neuralbench.plots._filters import filter_core_dataset
from neuralbench.plots.bar_charts import (  # noqa: F401  (re-exports)
    LETTER_STYLE,
    _FullGridStyle,
    plot_bar_chart,
    plot_full_bar_chart,
)
from neuralbench.plots.non_eeg import plot_non_eeg_bar_chart
from neuralbench.plots.normalized_summary import plot_normalized_lines_summary
from neuralbench.plots.rank_chart import plot_core_rank_boxplot
from neuralbench.plots.ranking import (  # noqa: F401  (re-exports)
    _prepare_rank_data,
    compute_task_ranks,
)
from neuralbench.plots.tables import (
    build_results_df,
    make_full_table,
    make_rank_table,
    make_results_table,
)
from neuralbench.plots.variability import plot_full_variability_panel

OUTPUT_SUBFOLDERS: tuple[str, str, str] = ("core", "full", "other")
"""Three output groups written by :func:`plot_all_results`.

* ``core``  -- single-dataset (core-study) plots and tables for the
  NeuralBench-Core variant (one dataset per task).
* ``full``  -- per-dataset breakdowns and variability analyses for the
  NeuralBench-Full variant (all datasets per task).
* ``other`` -- everything else (data scaling, computational stats,
  LaTeX tables produced by external scripts).
"""


@dataclass(frozen=True)
class _Step:
    """One artefact produced by :func:`plot_all_results`.

    The *group* must be one of :data:`OUTPUT_SUBFOLDERS`.  Steps are
    executed in order and short-circuit gracefully when the underlying
    plot/table function returns ``None`` (e.g. no multi-dataset task).
    """

    label: str
    group: tp.Literal["core", "full", "other"]
    fn: tp.Callable[[], tp.Any]


def _build_steps(
    df,
    core_df,
    full_df,
    core_dir: Path,
    full_dir: Path,
    other_dir: Path,
    optional_scaling: tuple[
        tp.Callable[..., tp.Any] | None, tp.Callable[..., tp.Any] | None
    ],
) -> list[_Step]:
    """Assemble the full ordered registry of plotting / table steps."""
    plot_data_scaling, make_data_scaling_table = optional_scaling

    steps: list[_Step] = [
        # ---------------- NeuralBench-Core outputs ----------------
        _Step("core_bar_chart", "core", lambda: plot_bar_chart(core_df, core_dir)),
        _Step(
            "core_non_eeg_bar_chart",
            "core",
            lambda: plot_non_eeg_bar_chart(core_df, core_dir),
        ),
        _Step(
            "core_rank_boxplot",
            "core",
            lambda: plot_core_rank_boxplot(core_df, core_dir),
        ),
        _Step(
            "core_results_table", "core", lambda: make_results_table(core_df, core_dir)
        ),
        _Step("core_rank_table", "core", lambda: make_rank_table(core_df, core_dir)),
        _Step(
            "core_normalized_summary",
            "core",
            lambda: plot_normalized_lines_summary(
                core_df, core_dir, group_by_category=True
            ),
        ),
        _Step(
            "core_normalized_summary_task_max",
            "core",
            lambda: plot_normalized_lines_summary(
                core_df, core_dir, group_by_category=True, ceiling="task_max"
            ),
        ),
        # ---------- NeuralBench-Full breakdowns + variability ----------
        _Step(
            "full_bar_chart_letter",
            "full",
            lambda: plot_full_bar_chart(full_df, full_dir),
        ),
        _Step("full_table", "full", lambda: make_full_table(full_df, full_dir)),
        _Step(
            "full_variability_panel",
            "full",
            lambda: plot_full_variability_panel(core_df, full_df, full_dir),
        ),
    ]

    if plot_data_scaling is not None:
        steps.append(
            _Step("data_scaling", "other", lambda: plot_data_scaling(df, other_dir))
        )
    if make_data_scaling_table is not None:
        steps.append(
            _Step(
                "data_scaling_table",
                "other",
                lambda: make_data_scaling_table(df, other_dir),
            )
        )

    return steps


def _load_optional_scaling() -> tuple[
    tp.Callable[..., tp.Any] | None, tp.Callable[..., tp.Any] | None
]:
    """Return the optional brainai data-scaling plot/table, if importable."""
    try:
        from brainai.bench.plots.scaling import (  # type: ignore[import-not-found]
            make_data_scaling_table,
            plot_data_scaling,
        )
    except ImportError:
        return None, None
    return plot_data_scaling, make_data_scaling_table


def plot_all_results(
    results: list[dict[str, tp.Any]],
    loss_to_metric_mapping: dict[str, str],
    output_dir: str | Path,
) -> None:
    """Build a results DataFrame and produce all plots and tables.

    Artefacts are written into three subfolders of *output_dir*:
    ``core/``, ``full/``, and ``other/``.  See
    :data:`OUTPUT_SUBFOLDERS`.
    """
    # Use matplotlib's default font (``font.family = ["sans-serif"]`` ->
    # DejaVu Sans on most systems) so every text element -- including
    # legend, mathtext, anchor labels, and tick labels -- shares a
    # single consistent font family.  ``pdf.fonttype = 42`` keeps text
    # selectable / searchable in the rendered PDF.
    sns.set_theme(context="paper", style="white", font="sans-serif")
    matplotlib.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    df = build_results_df(results, loss_to_metric_mapping)
    out_dir = Path(output_dir)
    core_dir = out_dir / "core"
    full_dir = out_dir / "full"
    other_dir = out_dir / "other"
    for d in (core_dir, full_dir, other_dir):
        d.mkdir(parents=True, exist_ok=True)

    full_df = df
    core_df = filter_core_dataset(full_df)

    steps = _build_steps(
        df,
        core_df,
        full_df,
        core_dir,
        full_dir,
        other_dir,
        optional_scaling=_load_optional_scaling(),
    )

    for step in tqdm(steps, desc="Generating plots"):
        step.fn()

    print(f"\nOutputs saved under {out_dir.resolve()} in subfolders {OUTPUT_SUBFOLDERS}")
