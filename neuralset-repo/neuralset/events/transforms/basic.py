# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import logging
import typing as tp
import urllib.parse

import numpy as np
import pandas as pd

from neuralset import base
from neuralset import segments as _segs

from .. import etypes as ev
from ..study import EventsTransform
from ..utils import query_with_index

logger = logging.getLogger(__name__)


class RemoveMissing(EventsTransform):
    """
    Remove events of specified type(s) that have missing or empty values in a given field.

    By default, this transform removes Word-type events that do not have any
    context, but it can be configured to remove other event types or check
    different fields.

    Parameters
    ----------
    event_types : str or sequence of str, default="Word"
        Type(s) of events to check for missing values.
    field : str, default="context"
        Column name in the events DataFrame to check for missing or empty values.
    """

    event_types: str | tp.Sequence[str] = "Word"
    field: str = "context"

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if self.field not in events.columns:
            msg = f"Field {self.field} not found in events dataframe, skipping RemoveMissing"
            logger.warning(msg)
            return events
        names = ev.EventTypesHelper(self.event_types).names
        data = events.loc[:, self.field]
        missing = data.fillna("").eq("")
        return events.loc[np.logical_or(~events.type.isin(names), ~missing)]


class AlignEvents(EventsTransform):
    """Creates timelines where events (eg: Meg, FMRI) are aligned to a trigger (eg: Image / Word)

    Parameters
    ----------
    trigger_type: str
        event type that serves as trigger for aligning other events
    trigger_field: str or tuple of str
        field that serves as hash for matching identical events
        (if tuple, the tuple of the fields will be used)
        Eg: Image: "filepath", Word: "text", chunked Video: ("filepath", "offset", "duration")
    types_to_align: str or tuple of str
        event types that must be aligned based on the trigger


    .. note::

        - columns ``origin_index`` and ``origin_timeline`` will be added to the dataframe
        - there will be 1 trigger per created timeline, starting at ``start=0``; other events will be shifted to match this timing
        - this transform can be used to perform average experiments, for example with a MEG extractor using ``aggregation="mean"``

    Example::

        Timeline-1
        MEG   m1-raw.fif start=0
        Word  blublu     start=1
        Word  bla        start=3
        Timeline-2
        MEG   m2-raw.fif start=0
        Word  blublu     start=2

    would produce the following events dataframe::

        from the transform AlignEvents(trigger_type="Word", trigger_field="text", types_to_align="MEG"):

        AlignEvents:blublu
        Word  blublu     start=0
        MEG   m1-raw.fif start=-1
        MEG   m2-raw.fif start=-2
        AlignEvents:bla
        Word  bla        start=0
        MEG   m1-raw.fif start=-3
    """

    trigger_type: str
    trigger_field: str | tuple[str, ...]
    types_to_align: str | tuple[str, ...]

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        tfields = self.trigger_field
        if isinstance(tfields, str):
            tfields = (tfields,)
        cls = ev.Event._CLASSES[self.trigger_type]
        missing = set(tfields) - set(cls.model_fields)
        if missing:
            logger.warning(
                "Event type %r has no fields %s — will look in event.extra at runtime",
                cls.__name__,
                missing,
            )

    def _trigger_hash(self, event: ev.Event) -> tuple[tp.Any, ...]:
        tfields = self.trigger_field
        if isinstance(tfields, str):
            tfields = (tfields,)
        return tuple(event._get_field_or_extra(f) for f in tfields)

    @staticmethod
    def _gen_timeline_name(h: tuple[tp.Any, ...]) -> str:
        parts = []
        suffix: list[str] = []
        for f in h:
            parts.append(str(f))
            if len(parts[-1]) > 18:
                parts[-1] = parts[-1][:8] + ".." + parts[-1][-8:]
                if not suffix:
                    suffix = [hashlib.sha256(str(h).encode()).hexdigest()[:8]]
        return ",".join(parts + suffix)

    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        store = _segs._EventStore.from_dataframe(df)
        trigger_cls = ev.EventTypesHelper(self.trigger_type).classes
        align_cls = ev.EventTypesHelper(self.types_to_align).classes
        triglist = [e for e in store._events if isinstance(e, trigger_cls)]
        triggers: dict[tuple[str, ...], list[ev.Event]] = {}
        for event in triglist:
            triggers.setdefault(self._trigger_hash(event), []).append(event)
        new_events = []
        for hash_, utriggers in triggers.items():
            uid = self._gen_timeline_name(hash_)
            utriggers.sort(key=lambda e: e.timeline)
            for k, ref in enumerate(utriggers):
                events = store.overlapping(ref.start, ref.duration, ref.timeline)
                if not k:
                    events = [ref] + [e for e in events if not isinstance(e, trigger_cls)]
                else:
                    events = [e for e in events if isinstance(e, align_cls)]
                for e in events:
                    if "origin_timeline" in e.extra:
                        msg = "Seemingly applying AlignEvents a 2nd time. "
                        msg += "This is not currently supported, please discuss your needs with maintainers."
                        raise RuntimeError(msg)
                    e = ev.Event.from_dict(e.to_dict())
                    e.start -= ref.start
                    e.extra["origin_timeline"] = e.timeline
                    e.timeline = f"{type(self).__name__}:{uid}"
                    new_events.append(e)
        out = pd.DataFrame([e.to_dict() for e in new_events])
        out = out.rename(columns={"Index": "origin_index"})
        return out


class QueryEvents(EventsTransform):
    """Filter events based on a pandas query, with auto-generated index columns.

    See :func:`query_with_index` for details on index column resolution.

    Example::

        # Filter by subject name
        QueryEvents(query='subject == "Subject1"')

        # Keep only the first 2 subjects
        QueryEvents(query="subject_index < 2")

        # Keep only the first timeline per subject
        QueryEvents(query="subject_timeline_index < 1")

    Parameters
    ----------
    query : Query | None
        A pandas query string (see :data:`base.Query`) that may reference
        ``*_index`` columns. If ``None``, returns events unchanged.
    """

    query: base.Query | None = None

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if self.query is None:
            return events
        return query_with_index(events, self.query)


class CreateColumn(EventsTransform):
    """Create a new column with a default value, and optionally update selected rows via a query.

    Parameters
    ----------
    column : str
        Name of the column to create.
    query_row : Query
        A pandas query string to select rows to update
        (see :data:`base.Query`).
    default_value : Any
        Default value to assign to all rows. If None, the column will not be initialized and only
        the rows selected by the query will be assigned the values from the query_column_name
        column.
    query_value : Any
        Value to assign to rows selected by the query.
    query_column_name : str | None
        If provided instead of `query_value`, the rows selected by the query will be assigned the
        values from this column.
    on_column_exists : Literal["raise", "warn", "ignore"]
        Behavior if the column already exists.
    """

    column: str
    query_row: base.Query
    default_value: tp.Any = None
    query_value: tp.Any = None
    query_column_name: str | None = None
    on_column_exists: tp.Literal["raise", "warn", "ignore"] = "raise"

    def model_post_init(self, context: tp.Any) -> None:
        super().model_post_init(context)
        if not (self.query_value is not None) ^ (self.query_column_name is not None):
            raise ValueError("Either query_value or query_column_name must be provided.")

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if self.column in events:
            if self.on_column_exists == "raise":
                raise ValueError(f"Column {self.column} already exists in events.")
            if self.on_column_exists == "warn":
                logger.warning(f"Overwriting existing column {self.column}.")
            elif self.on_column_exists == "ignore":
                pass

        test_inds = events.query(self.query_row, engine="python").index
        if self.default_value is not None:
            events[self.column] = self.default_value
        if self.query_value is not None:
            events.loc[test_inds, self.column] = self.query_value
        else:
            events.loc[test_inds, self.column] = events.loc[
                test_inds, self.query_column_name
            ]

        return events


class SelectIdx(EventsTransform):
    """Select a subset of events based on unique values in a column.
    e.g. SelectIdx(column='timeline', idx=3)

    Parameters
    ----------
    column : str
        Name of the column to count and select.
    idx: int | list
        Number of unique values from which to select.
    """

    column: str
    idx: int | list

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        codes, _ = pd.factorize(events[self.column])
        sel = [self.idx] if isinstance(self.idx, int) else self.idx
        return events.loc[np.isin(codes, sel)]


class ConfigureEventLoader(EventsTransform):
    """
    Modifies loading parameters for events with dynamic filepaths, i.e.,
    URIs and JSON SpecialLoader jsons. Leaves plain filepaths unchanged.

    Examples
    --------
    # Select FMRI space:
    ConfigureEventLoader(event_types="Fmri", params={"space": "MNI"})

    # Multiple event types:
    ConfigureEventLoader(event_types=("Fmri", "Meg"), params={"preprocessing": "hp"})
    """

    event_types: str | tuple[str, ...]
    params: dict[str, str] = {}

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if not self.params:
            logger.debug("ConfigureEventLoader called with empty params")
            return events

        events = events.copy()
        names = ev.EventTypesHelper(self.event_types).names
        mask = events["type"].isin(names)
        events.loc[mask, "filepath"] = events.loc[mask, "filepath"].map(
            self._update_filepath
        )
        return events

    def _update_filepath(self, fp: str) -> str:
        fp = str(fp)
        if fp.startswith("method:"):
            return self._update_method_uri(fp)
        if fp.startswith("{"):
            # Avoid circular dependency
            from ..study import SpecialLoader

            loader = SpecialLoader.from_json(fp)
            loader.kwargs.update(self.params)
            return loader.to_json()
        return fp

    def _update_method_uri(self, fp: str) -> str:
        parsed = urllib.parse.urlparse(fp)
        existing = dict(urllib.parse.parse_qsl(parsed.query))
        existing.update(self.params)
        new_query = urllib.parse.urlencode(existing)
        return urllib.parse.urlunparse(parsed._replace(query=new_query))
