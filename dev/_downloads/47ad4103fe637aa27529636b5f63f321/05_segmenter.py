"""
Segmenter & Dataset
===================

The :class:`~neuralset.dataloader.Segmenter` creates time-locked
segments from events, pairs them with :term:`extractors <Extractor>`,
and produces a :class:`~neuralset.dataloader.SegmentDataset` — a
standard PyTorch Dataset ready for a DataLoader.
"""

# %%
# What is a Segment?
# ------------------
#
# A :class:`~neuralset.segments.Segment` is a time window defined by
# ``start`` and ``duration``.  It holds references to all events that
# overlap with it, plus an optional **trigger** event that anchored
# the window.

import pandas as pd

import neuralset as ns

tl = "sub-01_run-01"
events = ns.events.standardize_events(
    pd.DataFrame(
        [
            dict(type="Stimulus", start=0, duration=60, timeline=tl, code=0),
            dict(type="Word", start=5.0, duration=0.3, text="hello", timeline=tl),
            dict(type="Word", start=10.0, duration=0.4, text="world", timeline=tl),
            dict(type="Word", start=15.0, duration=0.3, text="foo", timeline=tl),
        ]
    )
)

# %%
# Trigger-based Segmentation
# --------------------------
#
# Use :func:`~neuralset.segments.list_segments` with a ``triggers``
# mask to create one segment per matching event.  ``start`` is relative
# to the trigger:

raw_segments = ns.segments.list_segments(
    events,
    triggers=events.type == "Word",
    start=-0.5,
    duration=2.0,
)
print(f"{len(raw_segments)} segments created")
for seg in raw_segments:
    print(f"  {repr(seg)}")

# %%
# The Segmenter
# -------------
#
# :class:`~neuralset.dataloader.Segmenter` brings together extractors
# and segmentation parameters into a single config.
# ``apply(events)`` creates segments and returns a
# :class:`~neuralset.dataloader.SegmentDataset`:

segmenter = ns.dataloader.Segmenter(
    extractors={
        "pulse": {"name": "Pulse", "event_types": "Word", "aggregation": "trigger"}
    },
    trigger_query="type == 'Word'",
    start=-0.5,
    duration=2.0,
)
dataset = segmenter.apply(events)
print(f"Dataset: {len(dataset)} segments")

# %%
# Key parameters:
#
# .. list-table::
#    :header-rows: 1
#
#    * - Parameter
#      - Description
#    * - ``extractors``
#      - dict of name → extractor
#    * - ``trigger_query``
#      - pandas query selecting trigger events
#    * - ``start``
#      - offset relative to trigger (negative = before)
#    * - ``duration``
#      - segment length; ``None`` = use trigger event duration
#    * - ``drop_incomplete``
#      - remove segments missing events for any extractor

# %%
# Accessing Data
# --------------
#
# Each ``dataset[i]`` returns a
# :class:`~neuralset.dataloader.Batch` with two fields:
#
# - ``.data`` — a dict mapping extractor names to tensors
# - ``.segments`` — the list of segments for this item

item = dataset[0]
print(f"data keys: {list(item.data.keys())}")
print(f"pulse shape: {item.data['pulse'].shape}")

# %%
# Batching with DataLoader
# -------------------------
#
# Use a standard PyTorch ``DataLoader`` with the dataset's
# ``collate_fn`` to properly merge items into batches:

from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
for batch in loader:
    print(f"Batch pulse shape: {batch.data['pulse'].shape}")
    print(f"Batch segments: {len(batch.segments)}")
    break

# %%
# Subselection
# ------------
#
# ``.select()`` accepts integer indices or boolean masks to create
# a subset of the dataset:

sub = dataset.select([0, 2])
print(f"Selected: {len(sub)} segments")

sub = dataset.select(dataset.triggers.text == "hello")
print(f"Segments for 'hello': {len(sub)}")

# %%
# Strided Segments
# ----------------
#
# Use ``stride`` to create regularly-spaced segments across the
# recording, independent of trigger events.  Some windows may land
# where no Word events exist, so ``drop_incomplete=True`` silently
# discards those segments:

strided_segmenter = ns.dataloader.Segmenter(
    extractors={"pulse": {"name": "Pulse", "event_types": "Word", "aggregation": "sum"}},
    trigger_query="type == 'Stimulus'",
    start=0.0,
    duration=5.0,
    stride=2.5,
    drop_incomplete=True,
)
strided_dataset = strided_segmenter.apply(events)
print(f"Strided dataset: {len(strided_dataset)} segments")

# %%
# Validation
# ----------
#
# The Segmenter validates that:
#
# - ``trigger_query`` matches at least one event
# - All extractors have matching event types in the DataFrame
# - ``drop_incomplete`` removes segments where an extractor has no
#   matching events
#
# Use :func:`~neuralset.segments.find_incomplete_segments` to inspect
# which segments would be dropped before building the dataset.

# %%
# Memory Considerations
# ---------------------
#
# Data is loaded lazily in ``__getitem__`` — only the requested
# segment's events are passed through the extractors at access time.
# For large datasets:
#
# - Call ``extractor.prepare(events)`` to precompute and cache heavy
#   operations before iteration
# - Set ``infra.keep_in_ram=True`` on extractors for frequently
#   accessed data
# - Use ``drop_unused_events=True`` (default in Segmenter) to reduce
#   the events DataFrame size
# - Use ``drop_incomplete=True`` to skip segments that would fail due
#   to missing event types

# %%
# Next Steps
# ----------
#
# - Compose studies and transforms with chains: :doc:`06_chains`
