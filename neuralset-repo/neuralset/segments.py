# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Segments: lightweight time-window views into a shared event store.

Create segments with :func:`list_segments`, then pass them to
:class:`~neuralset.dataloader.SegmentDataset` to feed extractors via a
PyTorch ``DataLoader``.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import logging
import math
import typing as tp
import uuid
import weakref

import numpy as np
import pandas as pd

from neuralset.events.etypes import Event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _EventStore
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _EventBucket:
    """Sorted arrays for one event-type group within a timeline."""

    starts: np.ndarray  # sorted event start times
    stops: np.ndarray  # start + duration (parallel to starts)
    events: list[Event]  # parallel to starts/stops
    max_dur: float  # max(duration) in this bucket

    @classmethod
    def from_events(
        cls, events: list[Event], min_bucket_size: int = 50
    ) -> list[_EventBucket]:
        """Build sorted buckets from events.

        Each bucket tracks its own ``max_dur`` so searchsorted windows
        stay tight (e.g. 0.3 s for Words vs 500 s for Stimulus). Types
        with fewer than ``min_bucket_size`` events are merged into one
        bucket since their scan cost is negligible regardless of window.
        """
        by_type: dict[str, list[Event]] = collections.defaultdict(list)
        for e in events:
            by_type[e.type].append(e)

        groups: dict[str, list[Event]] = {}
        merged: list[Event] = []
        for typ, typed in by_type.items():
            if len(typed) >= min_bucket_size:
                groups[typ] = typed
            else:
                merged.extend(typed)
        if merged:
            groups[""] = merged

        buckets: list[_EventBucket] = []
        for typed in groups.values():
            typed.sort(key=lambda e: e.start)
            starts = np.array([e.start for e in typed])
            durations = np.array([e.duration for e in typed])
            buckets.append(
                cls(
                    starts=starts,
                    stops=starts + durations,
                    events=typed,
                    max_dur=float(durations.max()) if len(durations) else 0.0,
                )
            )
        return buckets

    def overlapping(self, start: float, stop: float) -> list[Event]:
        """Return events overlapping the ``[start, stop)`` window."""
        events = self.events
        # O(N) linear scan — faster for small buckets due to no Python overhead
        if len(self.starts) <= 150:
            mask = (self.starts < stop) & (self.stops > start)
            return list(itertools.compress(events, mask))
        # O(log N + k) searchsorted — widen lo by max_dur so we don't
        # miss long events that start early but extend into the window
        lo = int(np.searchsorted(self.starts, start - self.max_dur))
        hi = int(np.searchsorted(self.starts, stop))
        if lo >= hi:
            return []
        # Filter candidates whose actual end time overlaps the query start
        mask = self.stops[lo:hi] > start
        return list(itertools.compress(events[lo:hi], mask))


class _EventStore:
    """Shared, read-only container for events and a fast overlap index.

    Built once from a DataFrame via :meth:`from_dataframe`. Segments hold
    a reference (strong in-process, UUID across pickle) and resolve
    overlapping events through :meth:`overlapping`.
    """

    # process-local registry; segments look up stores by UUID after unpickling
    _REGISTRY: tp.ClassVar[weakref.WeakValueDictionary[uuid.UUID, _EventStore]] = (
        weakref.WeakValueDictionary()
    )

    def __init__(
        self,
        events: list[Event],
        timeline_index: dict[str, list[_EventBucket]],
    ) -> None:
        self._id = uuid.uuid4()
        self._events = events  # flat list for trigger lookup (positional)
        self._timeline_index = timeline_index
        _EventStore._REGISTRY[self._id] = self

    def __setstate__(self, state: dict[str, tp.Any]) -> None:
        self.__dict__.update(state)
        _EventStore._REGISTRY[self._id] = self

    # --- construction -------------------------------------------------------

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> _EventStore:
        """Build a store from a validated events DataFrame."""
        from neuralset.events import utils as ev_utils

        events = ev_utils.extract_events(df)

        by_timeline: dict[str, list[Event]] = collections.defaultdict(list)
        for event in events:
            by_timeline[event.timeline].append(event)

        timeline_index: dict[str, list[_EventBucket]] = {}
        for tl, tl_events in by_timeline.items():
            timeline_index[tl] = _EventBucket.from_events(tl_events)

        return cls(events, timeline_index)

    # --- public lookup ------------------------------------------------------

    def overlapping(self, start: float, duration: float, timeline: str) -> list[Event]:
        """Events overlapping [start, start + duration) on *timeline*."""
        buckets = self._timeline_index.get(timeline)
        if buckets is None:
            return []
        stop = start + duration
        result: list[Event] = []
        for bucket in buckets:
            result.extend(bucket.overlapping(start, stop))
        return result

    def __getitem__(self, pos: int) -> Event:
        return self._events[pos]

    @property
    def timelines(self) -> list[str]:
        return list(self._timeline_index)


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Segment:
    """A time window on a single :term:`timeline`, backed by a shared event store.

    Created by :func:`list_segments` (or via
    :class:`~neuralset.dataloader.Segmenter`) — not meant to be
    instantiated directly.  Each segment references an internal event
    store that holds all events; overlapping events are resolved on
    access via :attr:`ns_events`.

    Segments are **mutable**: changing :attr:`start` or :attr:`duration`
    changes which events :attr:`ns_events` returns (useful for jittering).

    The underlying event data is snapshotted at creation time — later
    modifications to the source DataFrame have no effect.
    """

    start: float
    duration: float
    timeline: str
    # position in the _EventStore flat event list (for trigger lookup)
    _trigger_idx: int | None = None
    # UUID for cross-process store lookup (pickled)
    _store_id: uuid.UUID = dataclasses.field(repr=False, default_factory=uuid.uuid4)
    # strong in-process ref, not pickled
    _store_ref: _EventStore | None = dataclasses.field(default=None, repr=False)
    # memoized ns_events: invalidated when (start, duration, timeline) changes
    _cache_key: tuple[float, float, str] | None = dataclasses.field(
        default=None,
        repr=False,
        compare=False,
    )
    _cached_events: list[Event] = dataclasses.field(
        default_factory=list,
        repr=False,
        compare=False,
    )

    # Segments don't own the store — SegmentDataset holds the strong
    # references that keep stores alive across pickle (see _event_stores).
    @property
    def _store(self) -> _EventStore:
        if self._store_ref is not None:
            return self._store_ref
        try:
            store = _EventStore._REGISTRY[self._store_id]
        except KeyError:
            raise RuntimeError(
                "_EventStore not found in this process. This segment was "
                "likely unpickled outside of a SegmentDataset/DataLoader."
            ) from None
        self._store_ref = store  # keep strong ref from now on
        return store

    @property
    def stop(self) -> float:
        """End time of the segment.

        Returns
        -------
        float
            The stop time, calculated as ``start + duration``.
        """
        return self.start + self.duration

    @property
    def trigger(self) -> Event | None:
        """The event that triggered this segment, or ``None``."""
        if self._trigger_idx is None:
            return None
        return self._store[self._trigger_idx]

    @property
    def ns_events(self) -> list[Event]:
        """Events overlapping this segment's ``[start, start + duration)`` window.

        The result is cached and automatically invalidated when
        :attr:`start`, :attr:`duration`, or :attr:`timeline` changes.
        """
        key = (self.start, self.duration, self.timeline)
        if self._cache_key != key:
            self._cached_events = self._store.overlapping(
                self.start, self.duration, self.timeline
            )
            self._cache_key = key
        return self._cached_events

    @property
    def events(self) -> pd.DataFrame:
        """Events occurring within the segment.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all events in this segment, with the
            original indices preserved.
        """
        ns_events = self.ns_events
        evts = [e.to_dict() for e in ns_events]
        for ev in evts:
            ev.pop("Index", None)
        indices = [e._index for e in ns_events]
        df = pd.DataFrame(data=evts, index=indices)
        return df.sort_index()

    def _to_extractor(self) -> dict[str, tp.Any]:
        """Dict that can be passed as ``**kwargs`` to extractors.

        Returns
        -------
        dict
            Keys: ``start``, ``duration``, ``events``, ``trigger``.
        """
        return {
            "start": self.start,
            "duration": self.duration,
            "events": self.ns_events,
            "trigger": self.trigger,
        }

    def copy(self, offset: float = 0.0, duration: float | None = None) -> Segment:
        """Create a copy of the current segment with optional offset and duration."""
        return Segment(
            start=self.start + offset,
            duration=duration if duration is not None else self.duration,
            timeline=self.timeline,
            _trigger_idx=self._trigger_idx,
            _store_id=self._store_id,
            _store_ref=self._store_ref,
        )

    def __getstate__(self) -> dict[str, tp.Any]:
        state = self.__dict__.copy()
        # _store_ref is large (all events); segments pickle only the
        # UUID and look up the store via _EventStore._REGISTRY.
        state["_store_ref"] = None
        return state

    def __setstate__(self, state: dict[str, tp.Any]) -> None:
        self.__dict__.update(state)
        # Re-establish strong ref if the store is already in this process.
        if self._store_ref is None:
            self._store_ref = _EventStore._REGISTRY.get(self._store_id)


# ---------------------------------------------------------------------------
# Segment creation
# ---------------------------------------------------------------------------


def list_segments(
    events: pd.DataFrame,
    triggers: pd.Series,
    *,
    start: float = 0.0,
    duration: float | None = None,
    stride: float | None = None,
    stride_drop_incomplete: bool = True,
) -> list[Segment]:
    """Create a list of segments based on events, sliding windows, or both.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing events, must be normalized first using :func:`~neuralset.events.standardize_events`.
    triggers : pd.Series
        Boolean mask of events to use for defining segments.
    start : float, optional
        Start offset (in seconds) of segments relative to reference events or stride.
        Use negative values to start before the reference. Default is 0.0.
    duration : float, optional
        Duration (in seconds) of each segment. If None, defaults to the duration of each event.
        Required when using strided windows. Default is None.
    stride : float, optional
        Step size (in seconds) for sliding window segmentation. Default is None.
    stride_drop_incomplete : bool, optional
        If True and stride is not None, drop segments not fully contained within
        the valid time range. Default is True.

    Returns
    -------
    list of Segment
        List of segments with populated :attr:`ns_events` field and :attr:`trigger` containing the event
        that triggered the segment.

    Notes
    -----
    Two segmentation modes are supported:

    1. Single window: each event specified by `triggers` yields a single segment.
    2. Sliding window: for each event specified by `triggers`, create strided
       windows of duration `duration` and step size `stride`.

    Examples
    --------
    Single window segmentation:
    >>> seg = list_segments(events, triggers=events.type == "Image", start=-0.5, duration=2.0)

    Sliding window segmentation:
    >>> seg = list_segments(events, triggers=events.type == "Meg", stride=1, duration=3)
    """
    start = float(start)
    if duration is not None:
        if isinstance(duration, np.ndarray):
            duration = duration.item()
        duration = float(duration)
    if stride is not None:
        stride = float(stride)
    if stride is not None and duration is None:
        raise RuntimeError("duration must be provided for strided windows")

    if not hasattr(events, "stop"):
        raise ValueError("Run standardize_events on the DataFrame first")
    if not isinstance(triggers, pd.Series):
        raise TypeError(
            f"triggers must be a boolean pd.Series, got {type(triggers).__name__}"
        )

    store = _EventStore.from_dataframe(events)

    trigger_df = events.loc[triggers]
    if not len(trigger_df):
        raise ValueError("Empty trigger events")

    df_to_pos = events.index.get_indexer(trigger_df.index)

    seg_starts: list[tp.Any] = []
    seg_durations: list[tp.Any] = []
    trigger_positions: list[tp.Any] = []
    trigger_timelines: list[tp.Any] = []

    if stride is None:
        seg_starts = (trigger_df["start"] + start).tolist()
        seg_durations = (
            trigger_df["duration"].tolist()
            if duration is None
            else [duration] * len(trigger_df)
        )
        trigger_positions = df_to_pos.tolist()
        trigger_timelines = trigger_df["timeline"].tolist()
    else:
        assert duration is not None
        for i, trig in enumerate(trigger_df.itertuples()):
            s, d = _prepare_strided_windows(
                float(trig.start) + start,  # type: ignore
                float(trig.stop),  # type: ignore
                stride,
                duration,
                drop_incomplete=stride_drop_incomplete,
            )
            seg_starts.extend(s)
            seg_durations.extend(d)
            trigger_positions.extend([df_to_pos[i]] * len(s))
            trigger_timelines.extend([str(trig.timeline)] * len(s))

    segments: list[Segment] = []
    for s, d, trig_idx, tl in zip(
        seg_starts, seg_durations, trigger_positions, trigger_timelines
    ):
        segments.append(
            Segment(
                start=s,
                duration=d,
                timeline=tl,
                _trigger_idx=trig_idx,
                _store_id=store._id,
                _store_ref=store,
            )
        )
    return segments


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _prepare_strided_windows(
    start: float,
    stop: float,
    stride: float,
    duration: float,
    drop_incomplete: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare parameters for strided sliding windows.

    Parameters
    ----------
    start : float
        Start time of the overall window.
    stop : float
        Stop time of the overall window.
    stride : float
        Step size between consecutive windows.
    duration : float
        Duration of each window.
    drop_incomplete : bool, optional
        If True, drop windows that are not fully contained within (start, stop).
        This matches the behavior of :code:`mne.events_from_annotations` with
        :code:`chunk_duration`. Default is True.

    Returns
    -------
    starts : np.ndarray
        Array of start times for each window.
    durations : np.ndarray
        Array of durations for each window (all equal to the input duration).

    Notes
    -----
    When :code:`drop_incomplete=True`, the effective stop time is adjusted to
    :code:`stop - duration` to ensure the last window fits completely.
    """
    effective_stop = (stop - duration) if drop_incomplete else stop
    span = effective_stop - start
    if span < -1e-10:
        msg = f"Dropping all windows as asking window duration {duration} on "
        msg += f"shorter overall window duration {stop - start} and {drop_incomplete=}"
        raise RuntimeError(msg)
    ratio = span / stride
    if drop_incomplete:
        n = math.floor(ratio + 1e-9) + 1
    else:
        n = math.ceil(ratio - 1e-9)
    n = max(0, n)
    starts = start + np.arange(n) * stride
    durations = np.full_like(starts, fill_value=duration)
    return starts, durations


def find_enclosed(df: pd.DataFrame, start: float, duration: float) -> pd.Series:
    """Find events fully enclosed within a time window.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing events with 'start' and 'duration' columns.
    start : float
        Start time of the enclosing window.
    duration : float
        Duration of the enclosing window.

    Returns
    -------
    pd.Series
        Series containing indices of events that are fully enclosed within
        the specified time window.

    Notes
    -----
    An event is considered enclosed if both its start and end times fall within
    the window: :code:`start <= event_start` and :code:`event_end <= start + duration`.

    Examples
    --------
    >>> enclosed = find_enclosed(events, start=5.0, duration=10.0)
    >>> events.loc[enclosed]  # Get fully enclosed events
    """
    estart = np.array(df.start)
    estop = estart + np.array(df.duration)
    is_enclosed = np.logical_and(estart >= start, estop <= start + duration)
    return pd.Series(df.index[is_enclosed])


def find_overlap(
    events: pd.DataFrame,
    triggers: pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | None = None,
) -> pd.Series:
    """Find events that overlap with a reference time window or events.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing events.
    triggers : pd.Series, optional
        If provided, reference events to check overlap against. If None, uses
        the absolute time window defined by start and duration. Default is None.
    start : float, optional
        If triggers is None, absolute start time of the reference window. If triggers is
        provided, offset relative to reference events. Default is 0.0.
    duration : float, optional
        Duration of the reference window. Required if triggers is None, otherwise
        can be None to use reference event durations. Default is None.

    Returns
    -------
    pd.Series
        Series containing indices of events that overlap with the reference.

    Raises
    ------
    AssertionError
        If triggers is None but duration is not provided, or if events contains multiple
        timelines when triggers is None.

    Notes
    -----
    Overlap is detected if any of these conditions are met:

    - Event starts within the reference window
    - Event ends within the reference window
    - Event fully encloses the reference window

    Examples
    --------
    Find overlap with absolute time window:

    >>> overlapping = find_overlap(events, start=5.0, duration=10.0)

    Find overlap with reference events:

    >>> ref_events = events.type == "Stimulus"
    >>> overlapping = find_overlap(events, triggers=ref_events, start=-1.0, duration=3.0)
    """
    if triggers is None:
        assert duration is not None
        assert events.timeline.nunique() == 1
        has_overlap = (events.start >= start) & (events.start < start + duration)
        has_overlap |= (events.start + events.duration > start) & (
            events.start + events.duration <= start + duration
        )
        has_overlap |= (events.start <= start) & (
            events.start + events.duration >= start + duration
        )
        return pd.Series(events.index[has_overlap])
    else:
        sel: list[int] = []
        for segment in list_segments(
            events,
            triggers=triggers,
            start=start,
            duration=duration,
        ):
            sel.extend(e._index for e in segment.ns_events)  # type: ignore[misc]
        return pd.Series(sel)


def find_incomplete_segments(
    segments: tp.Sequence[Segment], event_types: tp.Sequence[tp.Type[Event]]
) -> list[int]:
    """Find segments missing required event types.

    Returns indices of segments that do not contain at least one event of each
    specified event type (or their subclasses).

    Parameters
    ----------
    segments : sequence of Segment
        Segments to check for completeness.
    event_types : sequence of Event classes
        Event types that must be present in each segment.

    Returns
    -------
    list of int
        Sorted list of indices of segments that are missing at least one
        required event type.

    Warnings
    --------
    Logs a warning for each event type indicating how many segments are incomplete.

    Examples
    --------
    >>> from neuralset.events.etypes import Image, Audio
    >>> incomplete = find_incomplete_segments(segments, [Image, Audio])
    >>> complete_segments = [s for i, s in enumerate(segments) if i not in incomplete]

    Notes
    -----
    An event type's subclasses also count as valid. For example, if checking for
    :class:`~neuralset.events.Image`, segments containing :class:`~neuralset.events.NaturalImage`
    would also be considered complete.
    """
    all_invalid_indices = set()
    for event_type in event_types:
        invalid_indices = set()
        subclasses = [
            name for name, cls in Event._CLASSES.items() if issubclass(cls, event_type)
        ]
        for i, segment in enumerate(segments):
            if not any(e.type in subclasses for e in segment.ns_events):
                invalid_indices.add(i)
        if invalid_indices:
            msg = f"{len(invalid_indices)} segments out of {len(segments)} did not contain valid events for event type {event_type}"
            logger.warning(msg)
        all_invalid_indices.update(invalid_indices)
    return sorted(list(all_invalid_indices))
