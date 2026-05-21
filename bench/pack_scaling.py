# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Local scaling benchmark for ``neuralbench.packing.PackedExperiment``.

Times how long a single packed scheduler job takes to run a fixed batch of
synthetic experiments as the in-job worker count varies. Produces a log-scale
plot of wall-clock time and achieved speedup, alongside the ideal
``T(1) / N`` reference curve.

Two regimes are measured to make the spawn-overhead trade-off explicit:

* ``short`` — 2 s per experiment. ``ProcessPoolExecutor`` with ``spawn``
  context pays a one-time per-worker boot cost (re-importing the full
  ``neuralbench`` / ``neuraltrain`` / ``exca`` / ``numpy`` stack). When the
  per-experiment work is similar in magnitude to that boot cost, the
  achieved speedup is well below ideal.
* ``long``  — 10 s per experiment. Spawn cost amortizes; observed speedup
  tracks ideal closely. This is the realistic regime for benchmark
  experiments, which typically run for minutes to hours.

Run from ``bench/``:

    cd bench && python pack_scaling.py

Outputs (in the current directory):

    pack_scaling.png
    pack_scaling.pdf
    pack_scaling.csv
"""

# ruff: noqa: E402   (env vars below must be set before numpy is imported)

# IMPORTANT: cap BLAS threads BEFORE numpy is imported, so each pool worker
# uses exactly one BLAS thread. Otherwise n_jobs=8 might fan out to 8*8=64
# threads and saturate the machine on its own.
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import csv
import logging
import statistics
import sys
import tempfile
import time
import typing as tp
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Workload classes live in a sibling module so the spawn worker can import
# them by qualname (``pack_workloads.SleepExperiment``). Defining them in
# ``__main__`` here would make pickle.dump fail across the spawn boundary.
from pack_workloads import SleepExperiment

from neuralbench.packing import pack_experiments_for_submission
from neuraltrain.utils import BaseExperiment

# Use a non-interactive backend; we just save files.
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)


WorkloadCls = tp.Type[BaseExperiment]


def time_pack(
    experiment_cls: WorkloadCls,
    n_experiments: int,
    n_jobs: int,
    work_seconds: float,
    repeats: int = 2,
) -> tuple[float, float]:
    """Return ``(median, stdev)`` wall-clock for one configuration.

    Each repeat uses a fresh temp folder so exca's cache never short-circuits.
    """
    measurements: list[float] = []
    for _r in range(repeats):
        with tempfile.TemporaryDirectory(prefix="pack_bench_") as td:
            tdp = Path(td)
            experiments = [
                experiment_cls(
                    seed=i,
                    work_seconds=work_seconds,
                    infra={
                        "cluster": "auto",
                        "folder": tdp,
                        "mode": "force",
                    },
                )
                for i in range(n_experiments)
            ]
            packed = pack_experiments_for_submission(
                experiments, experiments_per_job="all", n_jobs=n_jobs
            )
            assert len(packed) == 1, "expected one packed job"
            cfg = packed[0].infra.model_dump(mode="python")
            cfg["cluster"] = None
            Pkg = type(packed[0])
            local = Pkg(
                experiments=packed[0].experiments,
                infra=cfg,
                n_jobs=n_jobs,
            )

            t0 = time.perf_counter()
            local.run()
            measurements.append(time.perf_counter() - t0)
    return statistics.median(measurements), (
        statistics.stdev(measurements) if len(measurements) > 1 else 0.0
    )


def sweep(
    regime_name: str,
    work_seconds: float,
    n_experiments: int,
    n_jobs_grid: list[int],
    repeats: int,
) -> list[dict[str, float]]:
    print(
        f"\n=== regime: {regime_name} "
        f"({n_experiments} exp × {work_seconds:.0f} s, "
        f"{repeats} repeats) ==="
    )
    print(f"{'n_jobs':>6} {'median_s':>10} {'stdev_s':>9} {'speedup':>9}")
    print("-" * 40)

    rows: list[dict[str, float]] = []
    baseline: float | None = None
    for n_jobs in n_jobs_grid:
        med, sd = time_pack(
            SleepExperiment,
            n_experiments=n_experiments,
            n_jobs=n_jobs,
            work_seconds=work_seconds,
            repeats=repeats,
        )
        if baseline is None:
            baseline = med
        speedup = baseline / med if med > 0 else float("nan")
        print(f"{n_jobs:>6} {med:>10.3f} {sd:>9.3f} {speedup:>8.2f}×")
        rows.append(
            {
                "regime": regime_name,
                "work_seconds": work_seconds,
                "n_experiments": n_experiments,
                "n_jobs": n_jobs,
                "median_s": med,
                "stdev_s": sd,
                "speedup": speedup,
            }
        )
    return rows


def plot(
    regimes: dict[str, list[dict[str, float]]],
    n_jobs_grid: list[int],
    out_png: Path,
    out_pdf: Path,
) -> None:
    # Okabe-Ito palette (colorblind-safe).
    C_SHORT = "#0072B2"
    C_LONG = "#D55E00"
    C_IDEAL = "#888888"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(11, 4.4))

    xs = np.array(n_jobs_grid)
    t_short = np.array([r["median_s"] for r in regimes["short"]])
    t_long = np.array([r["median_s"] for r in regimes["long"]])
    short_work = regimes["short"][0]["work_seconds"]
    long_work = regimes["long"][0]["work_seconds"]
    n_exp_short = int(regimes["short"][0]["n_experiments"])
    n_exp_long = int(regimes["long"][0]["n_experiments"])

    # Panel 1: wall-clock time (log-log)
    ideal_long = t_long[0] / xs

    ax_t.plot(xs, ideal_long, "--", color=C_IDEAL, lw=1.2, label="ideal $T(1) / N$")
    ax_t.plot(
        xs,
        t_long,
        "s-",
        color=C_LONG,
        lw=2.2,
        ms=7,
        label=f"long: {n_exp_long}×{long_work:.0f}s per exp.",
    )
    ax_t.plot(
        xs,
        t_short,
        "o-",
        color=C_SHORT,
        lw=2.2,
        ms=7,
        label=f"short: {n_exp_short}×{short_work:.0f}s per exp.",
    )

    # Annotate endpoints on each curve.
    for ys, color, dy in ((t_long, C_LONG, 8), (t_short, C_SHORT, -14)):
        ax_t.annotate(
            f"{ys[0]:.1f}s",
            xy=(xs[0], ys[0]),
            xytext=(8, dy),
            textcoords="offset points",
            color=color,
            fontweight="bold",
        )
        ax_t.annotate(
            f"{ys[-1]:.1f}s",
            xy=(xs[-1], ys[-1]),
            xytext=(8, dy),
            textcoords="offset points",
            color=color,
            fontweight="bold",
        )

    ax_t.set_xscale("log", base=2)
    ax_t.set_yscale("log")
    ax_t.set_xticks(xs)
    ax_t.set_xticklabels([str(x) for x in xs])
    ax_t.set_xlabel("local_workers_per_job (n_jobs inside the packed job)")
    ax_t.set_ylabel("wall-clock seconds (median, log scale)")
    ax_t.set_title("Packed-job runtime vs. worker count")
    ax_t.grid(True, which="both", alpha=0.3)
    ax_t.legend(loc="upper right")

    # Panel 2: speedup
    s_short = np.array([r["speedup"] for r in regimes["short"]])
    s_long = np.array([r["speedup"] for r in regimes["long"]])

    ax_s.plot(xs, xs, "--", color=C_IDEAL, lw=1.2, label="ideal (linear)")
    ax_s.plot(
        xs,
        s_long,
        "s-",
        color=C_LONG,
        lw=2.2,
        ms=7,
        label=f"long: {long_work:.0f}s per exp.",
    )
    ax_s.plot(
        xs,
        s_short,
        "o-",
        color=C_SHORT,
        lw=2.2,
        ms=7,
        label=f"short: {short_work:.0f}s per exp.",
    )

    for ys, color in ((s_long, C_LONG), (s_short, C_SHORT)):
        ax_s.annotate(
            f"{ys[-1]:.1f}×",
            xy=(xs[-1], ys[-1]),
            xytext=(8, -3),
            textcoords="offset points",
            color=color,
            fontweight="bold",
        )

    ax_s.set_xscale("log", base=2)
    ax_s.set_xticks(xs)
    ax_s.set_xticklabels([str(x) for x in xs])
    ax_s.set_xlabel("local_workers_per_job")
    ax_s.set_ylabel("speedup vs. n_jobs=1")
    ax_s.set_title("Achieved speedup (closer to ideal = better)")
    ax_s.grid(True, which="both", alpha=0.3)
    ax_s.legend(loc="upper left")

    fig.suptitle(
        "PackedExperiment local parallelization "
        "— spawn-pool overhead vs. amortized regime\n"
        "(neuralbench.packing on 14-core macOS, BLAS pinned to 1 thread)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")


def main() -> int:
    logging.getLogger("exca").setLevel(logging.WARNING)

    n_jobs_grid = [1, 2, 4, 8]
    repeats = 2

    regimes: dict[str, list[dict[str, float]]] = {}
    regimes["short"] = sweep(
        "short",
        work_seconds=2.0,
        n_experiments=8,
        n_jobs_grid=n_jobs_grid,
        repeats=repeats,
    )
    regimes["long"] = sweep(
        "long",
        work_seconds=10.0,
        n_experiments=8,
        n_jobs_grid=n_jobs_grid,
        repeats=repeats,
    )

    # --- csv ---
    out_csv = Path("pack_scaling.csv")
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "regime",
                "work_seconds",
                "n_experiments",
                "n_jobs",
                "median_s",
                "stdev_s",
                "speedup",
            ]
        )
        for rows in regimes.values():
            for r in rows:
                w.writerow(
                    [
                        r["regime"],
                        r["work_seconds"],
                        r["n_experiments"],
                        r["n_jobs"],
                        f"{r['median_s']:.4f}",
                        f"{r['stdev_s']:.4f}",
                        f"{r['speedup']:.3f}",
                    ]
                )

    out_png = Path("pack_scaling.png")
    out_pdf = Path("pack_scaling.pdf")
    plot(regimes, n_jobs_grid, out_png, out_pdf)

    print(f"\nwrote {out_csv}\nwrote {out_png}\nwrote {out_pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
