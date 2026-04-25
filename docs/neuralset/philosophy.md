---
orphan: true
---

# Philosophy

Design principles behind NeuralSet.

## Pydantic Everywhere

NeuralSet uses [pydantic](https://docs.pydantic.dev/) as the backbone for
every configurable object: Events, Studies, Extractors, Segmenters, and
Transforms are all `BaseModel` subclasses.

This gives you:

- **Typed configs** — every parameter is annotated; your IDE auto-completes
  and type-checks.
- **Validation** — pydantic validates inputs at construction time. Pass a
  string where an `int` is expected and you get a clear error, not a
  silent bug downstream.
- **Serialization** — any config can round-trip to/from a dictionary, JSON,
  or YAML. This makes experiment configs easy to log, reproduce, and share.

Because Events are pydantic models, you can create them from dicts
(`Event.from_dict(d)`), convert back (`event.to_dict()`), and store them
in a pandas DataFrame where each row is a validated event.

## exca — Caching and Infrastructure

NeuralSet relies on [`exca`](https://facebookresearch.github.io/exca/) for
caching expensive computation and for submitting work to compute clusters.

**What caching does.** When an extractor's `prepare(events)` is called, it
runs `_get_data()` for every event and stores the result on disk. Subsequent
calls read from cache instead of recomputing. This is critical for large
datasets where feature extraction (e.g. running a HuggingFace model on
every image) would otherwise dominate wall-clock time.

**What `prepare()` triggers.** The `Segmenter.apply()` method calls
`prepare()` on all extractors automatically. You can also call it manually
on individual extractors.

**Infrastructure configs.** The `infra` parameter on extractors, studies,
and chain steps controls *where* and *how* computation runs:

- `folder` — where cached results are stored on disk.
- `keep_in_ram` — whether to also keep results in memory for fast access.
- `cluster` — the compute backend: `None` (in-process), `"local"`
  (subprocess), or a Slurm configuration for cluster submission.

See [Caching & Cluster Execution](caching_and_cluster.rst) for
configuration details and examples, or the
[exca documentation](https://facebookresearch.github.io/exca/) for the
full API reference.

## Lazy Loading

NeuralSet keeps everything **lightweight** until you actually need the data:

- A Study's `run()` returns a pandas DataFrame of event metadata — no
  raw signals are loaded.
- Transforms modify this DataFrame (adding columns, filtering rows) without
  touching raw data.
- The Segmenter defines time windows and assigns extractors, but still
  doesn't load data.
- Actual data loading and feature extraction happen lazily in
  `dataset.__getitem__()` — i.e. when you iterate through a DataLoader or
  call `dataset[i]`.

This means you can configure a full pipeline, inspect events, filter
segments, and validate shapes without waiting for heavy I/O or GPU
computation. When you're ready, `prepare()` precomputes everything in one
pass.

## Modularity

The NeuralSet pipeline is built from four composable building blocks:

| Component | Role |
|-----------|------|
| **Study** | Interface to an external dataset: download, iterate timelines, produce events |
| **Transform** | Modifies the events DataFrame (filter, split, enrich) |
| **Extractor** | Converts events in a time window into a tensor |
| **Segmenter** | Creates time-locked segments and wires extractors to a Dataset |

Each component can be used independently — you can call an extractor on its
own without a Segmenter, or apply a transform without a Chain. When
composed, they form a reproducible pipeline where each step is
independently cacheable and configurable.
