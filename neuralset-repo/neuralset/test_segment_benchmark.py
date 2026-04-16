# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Segment benchmark: list_segments, __getitem__, pickle, epoch.

Two scenarios modeled after actual studies:

  nastase: 18 timelines, ~1k words each (duration=0.01, ~4k-char context)
    + ~4k phonemes + ~70 sentences + 1 Stimulus. Trigger: Word.
    18k segments total.

  allen: 50k timelines x 2 events (Stimulus + Image). Trigger: Image.
    50k segments total.

Run with ``pytest -v -s`` to see timing output.
Set ``BENCH_MAX_SEGS`` to control how many segments are used (default 500).

Reference results (view design vs previous eager design)
--------------------------------------------------------

End-to-end (Pulse extractors, fork workers):

  ==================  =============  ============  ===========  ==========
  Metric              Nastase eager  Nastase view  Allen eager  Allen view
  ==================  =============  ============  ===========  ==========
  list_segments       2.70 s         1.08 s        1.72 s       1.82 s
  epoch workers=0     1.01 s         1.02 s        1.80 s       1.80 s
  epoch workers=8     403 / 619 ms   211 / 214 ms  582 / 577ms  409/406ms
  ==================  =============  ============  ===========  ==========

Internal diagnostics:

  ====================  =============  ============  ===========  ==========
  Metric                Nastase eager  Nastase view  Allen eager  Allen view
  ====================  =============  ============  ===========  ==========
  __getitem__           56 us          56 us         37 us        36 us
  pickle/batch (32)     180 kB/0.67ms  1.6 kB/0.04ms 12kB/0.27ms 1.8kB/0.04ms
  overlap lookup        --             4.9 us        --           2.0 us
  ====================  =============  ============  ===========  ==========

Design assumption validation (degraded implementation):

  searchsorted matters:  Nastase 10->62 us (6x), Allen no effect
  Event cache matters:   Nastase 10->5142 us (500x), Allen 5->326 us (70x)

Production (Gifford2025AlgonautsBold, 3 epochs, 1 GPU):

  workers=0:  eager 1:39/1:30/1:29, view 1:39/1:29/1:27 (identical)
  workers=20: eager 0:45/0:49/0:49, view 0:42/0:45/0:45 (view ~8% faster)
  Correctness: val/loss and val/pearson identical.
"""

from __future__ import annotations

import gc
import os
import pickle
import time
import typing as tp

import numpy as np
import pandas as pd
import pytest

import neuralset as ns
from neuralset import dataloader as dl
from neuralset import segments as seg
from neuralset.events.utils import standardize_events

BATCH_SIZE = 32
MAX_SEGS = int(os.environ.get("BENCH_MAX_SEGS", "500"))

_NEURO_EXT = ns.extractors.Pulse(
    frequency=100.0,
    event_types="Stimulus",
    aggregation="sum",
)


# ---------------------------------------------------------------------------
# Data builders (cached)
# ---------------------------------------------------------------------------


def _build_nastase() -> tuple[pd.DataFrame, pd.Series, float, tp.Any]:
    rng = np.random.default_rng(42)
    rows: list[dict[str, tp.Any]] = []
    for t in range(18):
        tl = f"story_{t}"
        rows.append(dict(type="Stimulus", start=0.0, duration=600.0, timeline=tl))
        cursor = 0.0
        sent_id = 0
        sent_start = 0.0
        for i in range(1000):
            gap = rng.uniform(0.3, 0.7)
            ctx = f"ctx_s{t}_w{i}_" + "x" * (4000 - len(f"ctx_s{t}_w{i}_"))
            rows.append(
                dict(
                    type="Word",
                    start=cursor,
                    duration=0.01,
                    text=f"w{i}",
                    timeline=tl,
                    context=ctx,
                    sequence_id=sent_id,
                )
            )
            n_ph = rng.integers(2, 7)
            ph_dur = gap / n_ph
            for p in range(n_ph):
                rows.append(
                    dict(
                        type="Phoneme",
                        start=cursor + p * ph_dur,
                        duration=float(ph_dur),
                        timeline=tl,
                        text=f"p{p}",
                    )
                )
            cursor += gap
            if (i + 1) % 15 == 0:
                rows.append(
                    dict(
                        type="Sentence",
                        start=sent_start,
                        duration=cursor - sent_start,
                        timeline=tl,
                        text=f"sent{sent_id}",
                    )
                )
                sent_id += 1
                sent_start = cursor
    df = standardize_events(pd.DataFrame(rows))
    triggers = df.type == "Word"
    ext = ns.extractors.Pulse(frequency=100.0, event_types="Word", aggregation="sum")
    return df, triggers, 2.0, ext


def _build_allen() -> tuple[pd.DataFrame, pd.Series, float, tp.Any]:
    n_tl = 50_000
    tls = np.array([f"trial_{t}" for t in range(n_tl)])
    fmri = pd.DataFrame(
        dict(
            type="Stimulus",
            start=np.zeros(n_tl),
            duration=np.ones(n_tl),
            timeline=tls,
        )
    )
    image = pd.DataFrame(
        dict(
            type="Image",
            start=np.full(n_tl, 0.5),
            duration=np.full(n_tl, 0.5),
            timeline=tls,
            filepath="img.png",
        )
    )
    df = standardize_events(pd.concat([fmri, image], ignore_index=True))
    triggers = df.type == "Image"
    ext = ns.extractors.Pulse(frequency=100.0, event_types="Image", aggregation="sum")
    return df, triggers, 1.0, ext


_DATA: dict[str, tuple] = {}
_SEGS: dict[str, list] = {}


def _data(scenario: str) -> tuple[pd.DataFrame, pd.Series, float, tp.Any]:
    if scenario not in _DATA:
        _DATA[scenario] = {"nastase": _build_nastase, "allen": _build_allen}[scenario]()
    return _DATA[scenario]


def _segments(scenario: str) -> list:
    if scenario not in _SEGS:
        df, triggers, dur, _ = _data(scenario)
        _SEGS[scenario] = seg.list_segments(df, triggers=triggers, duration=dur)
    return _SEGS[scenario]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


@pytest.fixture(params=["nastase", "allen"])
def scenario(request: pytest.FixtureRequest) -> str:
    return request.param


def test_list_segments(scenario: str) -> None:
    df, triggers, dur, _ = _data(scenario)
    gc.collect()
    t0 = time.perf_counter()
    segs = seg.list_segments(df, triggers=triggers, duration=dur)
    elapsed = time.perf_counter() - t0
    print(f"\n[{scenario}] list_segments: {len(segs)} segs in {elapsed:.3f}s")


def test_getitem(scenario: str) -> None:
    _, _, dur, stim_ext = _data(scenario)
    segs = _segments(scenario)
    extractors = {"neuro": _NEURO_EXT, "stim": stim_ext}
    n = min(MAX_SEGS, len(segs))
    ds = dl.SegmentDataset(extractors, segs[:n], pad_duration=dur)
    _ = ds[0]
    gc.collect()
    t0 = time.perf_counter()
    for idx in range(n):
        _ = ds[idx]
    per_seg = (time.perf_counter() - t0) / n
    print(f"\n[{scenario}] __getitem__: {per_seg * 1e6:.0f} us/seg")


def test_pickle(scenario: str) -> None:
    segs = _segments(scenario)
    batch = segs[:BATCH_SIZE]
    pkl_bytes = len(pickle.dumps(batch))
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(200):
        pickle.loads(pickle.dumps(batch))
    per_round = (time.perf_counter() - t0) / 200
    print(f"\n[{scenario}] pickle/batch: {pkl_bytes:,} B, {per_round * 1000:.3f} ms")


def _run_epoch(loader: tp.Any) -> tuple[float, float]:
    """Run one full epoch, return (first_batch_ms, rest_ms)."""
    it = iter(loader)
    t0 = time.perf_counter()
    next(it)
    first = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    for _ in it:
        pass
    rest = (time.perf_counter() - t0) * 1000
    return first, rest


@pytest.mark.parametrize("num_workers", [0, 2, 4, 8])
def test_epoch(scenario: str, num_workers: int) -> None:
    """Two full epochs through DataLoader."""
    import torch.utils.data

    _, _, dur, stim_ext = _data(scenario)
    segs = _segments(scenario)[:MAX_SEGS]
    extractors = {"neuro": _NEURO_EXT, "stim": stim_ext}
    ds = dl.SegmentDataset(extractors, segs, pad_duration=dur)
    kwargs: dict[str, tp.Any] = dict(
        collate_fn=ds.collate_fn,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    if num_workers > 0:
        kwargs.update(num_workers=num_workers, multiprocessing_context="fork")
    try:
        loader = torch.utils.data.DataLoader(ds, **kwargs)
        gc.collect()
        first1, rest1 = _run_epoch(loader)
    except Exception as exc:
        pytest.skip(f"DataLoader workers={num_workers} unavailable: {exc}")
    first2, rest2 = _run_epoch(loader)
    n_batches = (len(segs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(
        f"\n[{scenario}] epoch workers={num_workers} ({n_batches} batches, {len(segs)} segs):"
        f"\n  epoch1: {first1 + rest1:.0f}ms  epoch2: {first2 + rest2:.0f}ms"
        f"  (1st batch: {first1:.1f}ms)"
    )
