# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Event utilities: extraction, validation, and BIDS entity constants."""

import inspect
import logging
import re
import typing as tp
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import ujson
from exca.cachedict import DumpContext
from exca.cachedict.handlers import ParquetPandasDataFrame

from neuralset import utils

from .etypes import Event, EventTypesHelper

logger = logging.getLogger(__name__)
TypesParam = str | tp.Sequence[str] | tp.Type[Event] | EventTypesHelper

# Standard BIDS entity columns enforced on all events DataFrames
# (not fields on Event classes -- managed at the DataFrame level)
BIDS_ENTITIES = ("subject", "session", "task", "run")
BIDS_ENTITY_DEFAULT = ""


def extract_events(obj: tp.Any, types: TypesParam | None = None) -> list[Event]:
    """Returns a list of neuralset events extracted from the input parameter.

    Parameters
    ----------
    obj: dataframe, event, or list of events/segments
        the object to extract the list of events from, generally a dataframe or a list of
        segments
    types: optional str/list of str/ Event type/EventTypesHelper
        filters only the provided type(s)

    Returns
    -------
    list of Event
        the list of events extracted from the input object, possibly filtered by the
        provided type
    """
    from neuralset import segments  # lazy to avoid circular imports

    helper: EventTypesHelper | None = None
    if isinstance(types, EventTypesHelper):
        helper = types
    elif types is not None:
        helper = EventTypesHelper(types)
    # fast track for lists as it's the most common
    if isinstance(obj, (list, tuple)):
        if not obj:
            return []
        if isinstance(obj[0], Event):
            if helper is not None:
                obj = [e for e in obj if isinstance(e, helper.classes)]
            return obj
    if isinstance(obj, pd.DataFrame):
        if helper is not None:
            obj = obj.loc[obj.type.isin(helper.names), :]
        unknown = set(obj.type) - set(Event._CLASSES)
        if unknown:
            logger.warning("Ignoring unknown event types: %s", unknown)
            obj = obj.loc[~obj.type.isin(unknown), :]
        # skip itertuple if only one/two event :) (pandas is slooow)
        num = len(obj)
        iterable = (obj.iloc[k, :] for k in range(num)) if num <= 2 else obj.itertuples()
        out = [Event.from_dict(r) for r in iterable]
        for i, e in zip(obj.index, out):
            e._index = i  # noqa
        return out
    if isinstance(obj, Event):
        obj = [obj]
    elif isinstance(obj, (dict, pd.Series)):
        obj = [Event.from_dict(obj)]
    if isinstance(obj, segments.Segment):
        obj = [obj]
    if not isinstance(obj, (list, tuple)):
        raise NotImplementedError(f"Conversion of {type(obj)} is not supported")
    if not obj:
        return []
    if isinstance(obj[0], segments.Segment):
        event_dict = {}
        for segment in obj:
            event_dict.update({id(e): e for e in segment.ns_events})
            if segment.trigger is not None:
                event_dict[id(segment.trigger)] = segment.trigger
        obj = list(event_dict.values())
    if not isinstance(obj[0], Event):
        raise NotImplementedError(f"Unexpected list of {type(obj[0])} is not supported")
    return extract_events(obj, types=helper)


def expand_bids_fmri(
    pattern: str, preproc: str, **common: tp.Any
) -> list[dict[str, tp.Any]]:
    """Resolve a BIDS glob pattern into one Fmri event dict per space.

    Parameters
    ----------
    pattern
        Glob pattern ending with ``*`` (e.g.
        ``"derivatives/deepprep/.../sub-01_ses-01_task-rest_run-1_*"``).
    preproc
        Preprocessing pipeline name (``"deepprep"``, ``"fmriprep"``).
    **common
        Fields shared across all output dicts (``start``, ``frequency``, …).

    Returns
    -------
    list[dict]
        One dict per discovered space, ready for ``pd.DataFrame``.
    """
    from bids.layout import parse_file_entities  # type: ignore[import-untyped]

    fp_pattern = Path(pattern)
    fps: defaultdict[str, dict[str, str]] = defaultdict(dict)
    for ext in (".nii.gz", ".gii"):
        for fp in fp_pattern.parent.glob(fp_pattern.name + ext):
            ents = parse_file_entities(str(fp))
            space = ents.get("space")
            if space is None:
                continue
            if "hemi" in ents:
                part = "left" if ents["hemi"] == "L" else "right"
            elif ents.get("suffix") == "mask":
                part = "mask"
            elif ents.get("suffix") == "bold" and ents.get("desc") == "preproc":
                part = "data"
            else:
                continue
            fps[space][part] = str(fp)
    typed: list[dict[str, tp.Any]] = []
    for space, parts in fps.items():
        kw: dict[str, tp.Any] = {
            **common,
            "space": space,
            "preproc": preproc,
            "type": "Fmri",
        }
        if "left" in parts:
            # Only the left-hemi path is stored; Fmri._read() derives
            # the right hemisphere by replacing hemi-L with hemi-R.
            kw.update(filepath=parts["left"])
        elif "data" in parts:
            kw.update(filepath=parts["data"], mask_filepath=parts.get("mask"))
        else:
            continue
        typed.append(kw)
    return typed


def query_with_index(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Execute a pandas query with auto-generated index columns.

    Recognises ``<col>_index`` and ``<col1>_<col2>_index`` tokens in the
    query string and materialises them as temporary columns before executing
    the query. Temporary columns are dropped from the result.

    Example::

        # Filter by subject name
        query_with_index(df, 'subject == "Subject1"')

        # Keep only the first 2 subjects
        query_with_index(df, "subject_index < 2")

        # Keep only the first timeline per subject
        query_with_index(df, "subject_timeline_index < 1")
    """
    out = df.copy()
    temp_cols: list[str] = []
    columns = set(df.columns)

    for token in re.findall(r"\b(\w+_index)\b", query):
        if token in temp_cols:
            continue
        if token in columns:
            raise ValueError(
                f"Ambiguous index token {token!r}: a column with that name "
                "already exists in the dataframe."
            )
        prefix = token.rsplit("_index", 1)[0]
        if prefix in columns:
            out.loc[:, token] = out.groupby(prefix, sort=False).ngroup()
            temp_cols.append(token)
        else:
            matched = False
            for col in sorted(columns, key=len, reverse=True):
                if not prefix.startswith(col + "_"):
                    continue
                remainder = prefix[len(col) + 1 :]
                if remainder in columns:
                    out.loc[:, token] = out.groupby(col, sort=False)[remainder].transform(
                        lambda x: pd.factorize(x)[0]
                    )
                    temp_cols.append(token)
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"Cannot resolve index token {token!r}: "
                    "no matching columns found in the dataframe"
                )

    result = out.query(query, engine="python")
    if temp_cols:
        result = result.drop(columns=temp_cols)
    return result


def standardize_events(
    events: pd.DataFrame,
    auto_fill: bool = True,
) -> pd.DataFrame:
    """Normalize an events DataFrame into canonical form.

    Sorts by (timeline, start ASC, duration DESC) preserving first-seen
    timeline order, fills BIDS entity columns, reorders columns, and adds
    a computed ``stop`` column. Always returns a new DataFrame.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing events. Must have at least a ``type`` column
        with string values.
    auto_fill : bool
        If True (default), round-trip each row through its pydantic Event
        model (``from_dict`` → ``to_dict``). This fills missing fields
        with defaults, validates types, and coerces values. This is
        **significantly slower** than the rest of the normalization
        (~10 s on 100k rows vs ~60 ms) and is only needed the first time
        raw events are ingested. Pass ``False`` when the data has already
        been through ``auto_fill`` (e.g. post-concat re-normalization,
        transform wrapper, cache load).

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame. The ``stop`` column (``start + duration``)
        signals that normalization has been applied.
    """
    if events.empty:
        raise ValueError("Cannot normalize an empty events DataFrame.")
    for name in ["index", "Index"]:
        if name in events.columns:
            msg = f"The events dataframe contains an `{name}` column. This is "
            msg += "dangerous, please add drop=True in calls to df.reset_index(). "
            msg += "Dropping it automatically."
            warnings.warn(msg)
            events = events.drop(columns=[name])
    msg = 'events DataFrame must have a "type" column with strings'
    if "type" not in events.keys():
        raise ValueError(msg)
    types = events["type"].unique()
    if not all(isinstance(typ, str) for typ in types):
        raise ValueError(msg)
    if auto_fill:
        validated = events.apply(_coerce_event, axis=1)
        df = pd.DataFrame(validated.tolist(), index=events.index)
        null = df.loc[df.duration <= 0, :]
        if not null.empty:
            types = null["type"].unique()
            msg = f"Found {len(null)} event(s) with null duration (types: {types})"
            warnings.warn(msg)
    else:
        df = events
    for col in ("start", "duration"):
        if col not in df.columns:
            raise ValueError(f"Events DataFrame is missing required column '{col}'.")
        if df[col].isna().any():
            n = df[col].isna().sum()
            raise ValueError(
                f"Events DataFrame has {n} row(s) with NaN {col}. "
                f"Use auto_fill=True if events were built without {col}."
            )
    if "timeline" in df.columns and df["timeline"].isna().any():
        n = df["timeline"].isna().sum()
        raise ValueError(
            f"Events DataFrame contains {n} row(s) with NaN timeline. "
            "All events must have a valid 'timeline' value."
        )
    # Sort by (timeline, start ASC, duration DESC) preserving first-seen timeline order
    tl_order = pd.Categorical(
        df["timeline"], categories=df["timeline"].unique(), ordered=True
    )
    df = df.assign(_tl_order=tl_order.codes)
    df = df.sort_values(
        ["_tl_order", "start", "duration"],
        ascending=[True, True, False],
        ignore_index=True,
    )
    df = df.drop(columns=["_tl_order"])
    for entity in BIDS_ENTITIES:
        if entity not in df.columns:
            df[entity] = BIDS_ENTITY_DEFAULT
        df[entity] = df[entity].fillna(BIDS_ENTITY_DEFAULT).astype(str)
    important = ["type", "start", "duration", "timeline"] + list(BIDS_ENTITIES)
    columns = important + [c for c in df.columns if c not in important]
    df = df.loc[:, columns]
    df = df.assign(stop=lambda x: x.start + x.duration)
    return df


def _coerce_event(event: pd.Series) -> dict[str, tp.Any]:
    """Coerce a single event row through its pydantic Event model.

    Instantiates the Event subclass for this row's ``type``, which validates
    fields and fills defaults, then converts back to a dict.
    """
    from . import etypes

    event_type = event["type"]
    lower = {x.lower() for x in etypes.Event._CLASSES}
    if event_type in etypes.Event._CLASSES:
        event_class = etypes.Event._CLASSES[event_type]
        event_obj = event_class.from_dict(event).to_dict()
        event_dict = {**event, **event_obj}
    elif event_type in lower:
        raise ValueError(
            f"Unknown event type {event_type!r} (did you mean {event_type.title()!r}?)"
        )
    else:
        utils.warn_once(
            f'Unexpected type "{event["type"]}". Support for new event '
            "types can be added by creating new `Event` classes in "
            "`neuralset.events`."
        )
        event_dict = {**event}

    return event_dict


@DumpContext.register
class ValidatedParquet(ParquetPandasDataFrame):
    """Parquet cache type that coerces events before dumping and
    normalizes on load.

    Dump: coerces each event through its pydantic model (safety net),
    strips the derived ``stop`` column before writing.
    Load: normalizes (re-sorts, fills BIDS columns, recomputes ``stop``).
    """

    @classmethod
    def __dump_info__(cls, ctx: DumpContext, value: tp.Any) -> dict[str, tp.Any]:
        value = standardize_events(value)
        value = value.drop(columns=["stop"], errors="ignore")
        # Cast object-dtype columns with mixed types (e.g. int + str) to str
        # so pyarrow can serialize them without ArrowInvalid errors.
        for col in value.columns:
            if value[col].dtype == object:
                types = {type(v) for v in value[col].dropna()}
                if len(types) > 1:
                    value[col] = value[col].astype(str)
        return super().__dump_info__(ctx, value)

    @classmethod
    def __load_from_info__(cls, ctx: DumpContext, filename: str) -> tp.Any:
        df = pd.read_parquet(ctx.folder / filename)
        return standardize_events(df, auto_fill=False)


class SpecialLoader:
    """Loader for special methods that need to be serialized and called later.

    Parameters
    ----------
    method: method
        method to use for loading
    timeline: dict[str, Any]
        parameters defining the timeline (subject, task etc)
    **kwargs: Any
        any additional (json-able) parameters used to call the method

    Example
    -------
    In ``_load_timeline_events``::

        loader = SpecialLoader(method=self._my_method, timeline=timeline, additional_param=...)

    In the method::

        def _my_method(self, timeline: dict[str, tp.Any], additional_param):
            ...
    """

    def __init__(
        self,
        method: tp.Callable[..., tp.Any],
        timeline: dict[str, tp.Any],
        **kwargs: tp.Any,
    ) -> None:
        self.method = method
        self.timeline = timeline
        self.kwargs = kwargs

    @classmethod
    def from_json(cls, string: str) -> "SpecialLoader":
        from .study import STUDIES, STUDY_PATHS  # deferred: study imports utils

        data = ujson.loads(string)
        name = data["cls"]
        if name not in STUDIES or name not in STUDY_PATHS:
            raise RuntimeError(
                f"Study class {name!r} is not available in this process "
                f"(registered: {list(STUDIES)}, with path: {list(STUDY_PATHS)}).\n"
                "A Study instance must be constructed before SpecialLoader.from_json "
                "can be used — in a child job / subprocess, make sure the study "
                "module is imported and the Study is instantiated."
            )
        scls = STUDIES[name]
        study = scls(path=STUDY_PATHS[name], **data.get("cls_kwargs", {}))
        method = getattr(study, data["method"])
        kwargs = data.get("kwargs", {})
        return cls(method=method, timeline=data["timeline"], **kwargs)

    def load(self) -> tp.Any:
        return self.method(timeline=self.timeline, **self.kwargs)  # type: ignore

    def to_json(self) -> str:
        inst = self.method.__self__  # type: ignore
        data = {
            "cls": inst.__class__.__name__,
            "timeline": self.timeline,
            "method": self.method.__name__,
        }
        if self.kwargs:
            data["kwargs"] = self.kwargs
        cls_kwargs = inst._cls_kwargs()
        if cls_kwargs:
            data["cls_kwargs"] = cls_kwargs
        return ujson.dumps(data, sort_keys=True)

    def specs(self) -> dict[str, list[str]]:
        """Extract configurable parameters from the method's Literal type hints.

        Returns a dict mapping parameter names to their allowed values,
        e.g. ``{"registration": ["msmall", "none", "mni"], ...}``.
        """
        hints = tp.get_type_hints(self.method)
        sig = inspect.signature(self.method)
        out: dict[str, list[str]] = {}
        for name, _param in sig.parameters.items():
            if name in ("self", "timeline"):
                continue
            hint = hints.get(name)
            if hint is None:
                continue
            args = tp.get_args(hint)
            if args and all(isinstance(a, str) for a in args):
                out[name] = list(args)
        return out

    def specs_json(self) -> str:
        """JSON string of :meth:`specs`, suitable for a DataFrame column."""
        return ujson.dumps(self.specs(), sort_keys=True)
