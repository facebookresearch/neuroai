from __future__ import annotations

import argparse
import statistics
import time
import typing as tp
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import neuralset as ns
from neuralset.events.utils import extract_events, query_with_index

plt.switch_backend("Agg")


ROOT = Path(__file__).resolve().parent
SPACES = ("fsaverage5", "MNI152NLin2009cAsym", "T1w", "fsaverage")
RUNS = ("1", "2", "3", "4")
SUBJECTS = tuple(f"sub-{idx:03d}" for idx in range(64))


def make_fmri_events(n_events: int) -> pd.DataFrame:
    rows: list[dict[str, tp.Any]] = []
    for idx in range(n_events):
        subject = SUBJECTS[idx % len(SUBJECTS)]
        timeline = f"{subject}_run-{RUNS[idx % len(RUNS)]}"
        rows.append(
            {
                "type": "Fmri",
                "start": float(idx % 100),
                "duration": 10.0,
                "timeline": timeline,
                "filepath": f"method:load_fmri?timeline={timeline}",
                "frequency": 1.0,
                "subject": subject,
                "run": RUNS[idx % len(RUNS)],
                "space": SPACES[idx % len(SPACES)],
                "preproc": "fmriprep",
            }
        )
    return pd.DataFrame(rows)


def dataframe_query(df: pd.DataFrame, query: str) -> list[ns.events.Event]:
    return extract_events(query_with_index(df, query), types="Fmri")


def event_list_query(events: list[ns.events.Event], query: str) -> list[ns.events.Event]:
    events_df = pd.DataFrame([event.to_dict() for event in events])
    events_df.index = pd.RangeIndex(len(events))
    selected = query_with_index(events_df, query)
    return [events[int(index)] for index in selected.index]


def iter_single(events: list[ns.events.Event]) -> list[ns.events.Event]:
    return [
        event for event in events if event._get_field_or_extra("space") == "fsaverage5"
    ]


def iter_multi(events: list[ns.events.Event]) -> list[ns.events.Event]:
    selected = []
    subject_to_index: dict[str, int] = {}
    for event in events:
        subject = event._get_field_or_extra("subject")
        if subject not in subject_to_index:
            subject_to_index[subject] = len(subject_to_index)
        if (
            event._get_field_or_extra("space") == "fsaverage5"
            and event._get_field_or_extra("run") == "1"
            and subject_to_index[subject] < 16
        ):
            selected.append(event)
    return selected


def time_call(
    fn: tp.Callable[[], list[ns.events.Event]], repeats: int
) -> tuple[float, int]:
    elapsed = []
    n_selected = 0
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        elapsed.append(time.perf_counter() - start)
        n_selected = len(out)
    return statistics.median(elapsed), n_selected


def run_benchmark(sizes: list[int], repeats: int) -> pd.DataFrame:
    queries = {
        "single_column": "space == 'fsaverage5'",
        "multi_column": "space == 'fsaverage5' and run == '1' and subject_index < 16",
    }
    rows = []
    for n_events in sizes:
        df = make_fmri_events(n_events)
        events = extract_events(df, types="Fmri")
        methods: dict[str, dict[str, tp.Callable[[], list[ns.events.Event]]]] = {
            "dataframe_query": {
                name: lambda q=query: dataframe_query(df, q)
                for name, query in queries.items()
            },
            "event_list_query": {
                name: lambda q=query: event_list_query(events, q)
                for name, query in queries.items()
            },
            "python_iteration": {
                "single_column": lambda: iter_single(events),
                "multi_column": lambda: iter_multi(events),
            },
        }
        for method, method_queries in methods.items():
            for query_name, fn in method_queries.items():
                median_s, n_selected = time_call(fn, repeats)
                rows.append(
                    {
                        "n_events": n_events,
                        "query": query_name,
                        "method": method,
                        "median_s": median_s,
                        "n_selected": n_selected,
                    }
                )
                print(
                    f"{n_events:>7} {query_name:<14} {method:<18} "
                    f"{median_s * 1e3:>8.1f} ms selected={n_selected}"
                )
    return pd.DataFrame(rows)


def save_plot(results: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, (query_name, data) in zip(axes, results.groupby("query", sort=False)):
        for method, method_data in data.groupby("method", sort=False):
            method_data = method_data.sort_values("n_events")
            axis.plot(
                method_data["n_events"],
                method_data["median_s"] * 1e3,
                marker="o",
                label=method,
            )
        axis.set_title(query_name.replace("_", " "))
        axis.set_xlabel("events")
        axis.set_xscale("log")
        axis.grid(True, which="both", axis="both", alpha=0.3)
    axes[0].set_ylabel("median runtime (ms)")
    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[500, 5_000, 50_000, 100_000],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "query_benchmark.png",
    )
    args = parser.parse_args()

    results = run_benchmark(args.sizes, args.repeats)
    csv_path = args.output.with_suffix(".csv")
    results.to_csv(csv_path, index=False)
    save_plot(results, args.output)
    print(f"\nSaved {args.output}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
