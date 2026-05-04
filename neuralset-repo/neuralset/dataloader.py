# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=super-init-not-called

import collections
import concurrent.futures
import dataclasses
import logging
import typing as tp
import warnings

import numpy as np
import pandas as pd
import pydantic
import torch
from pydantic import PositiveFloat

import neuralset as ns

from . import base
from .events import EventTypesHelper
from .extractors import BaseExtractor
from .extractors import BaseExtractor as Feat
from .segments import find_incomplete_segments

logger = logging.getLogger(__name__)

T = tp.TypeVar("T", bound="SegmentsMixin")


@dataclasses.dataclass(kw_only=True)
class SegmentsMixin:
    """Mixin class for segment-structured classes.

    Subclasses must implement ``_subselect`` so that ``.select()`` works.
    """

    segments: tp.Sequence[ns.segments.Segment]

    def _duration_str(self) -> str:
        durs = {s.duration for s in self.segments}
        if len(durs) == 1:
            return f"{next(iter(durs)):.2f}s"
        return f"{min(durs):.2f}-{max(durs):.2f}s"

    def __post_init__(self):
        if not isinstance(self.segments, list):
            raise TypeError("segments must be a list of Segment instances")

    def _subselect(self: T, idx: list[int]) -> T:
        """To be defined by subclass: how to subselect from a list of indices"""
        raise NotImplementedError

    def select(
        self: T,
        idx: int | list[int] | list[bool] | np.ndarray | pd.Series,
    ) -> T:
        """Subselect segments by integer indices or boolean mask.

        Parameters
        ----------
        idx
            One of:
            - ``int``: single segment index
            - ``list[int]`` or ``np.ndarray[int]``: positional indices
            - ``list[bool]``, ``np.ndarray[bool]``, or ``pd.Series[bool]``:
              boolean mask (must have length equal to the number of segments)
        """
        if isinstance(idx, pd.Series):
            idx = idx.to_numpy()  # type: ignore[assignment]
        indices = np.atleast_1d(idx)
        if indices.ndim != 1:
            raise ValueError(f"select indices must be 1-d, got shape {indices.shape}")
        if indices.dtype == bool:
            if len(indices) != len(self):
                raise ValueError(
                    f"Boolean mask length {len(indices)} != "
                    f"number of segments {len(self)}"
                )
            indices = np.where(indices)[0]
        elif indices.dtype.kind not in ("i", "u"):
            raise ValueError(f"select indices must be int or bool, got {indices.dtype}")
        if not len(indices):
            raise ValueError("Empty subselection")
        return self._subselect(indices.tolist())

    @property
    def triggers(self) -> pd.DataFrame:
        triggers = [s.trigger for s in self.segments]
        if any(t is None for t in triggers):
            raise RuntimeError(f"Segments did not all have a trigger: {triggers}")
        df = pd.DataFrame([t.to_dict() for t in triggers])  # type: ignore
        df.index = df.pop("Index")  # type: ignore
        return df

    @property
    def events(self) -> pd.DataFrame:
        """All of the events spanned by all the segments"""
        # TODO FIXME why are segments events with both an index AND and Index column
        # TODO discuss should the trigger event be in events? Probably not
        return pd.concat([seg.events for seg in self.segments]).drop_duplicates()

    def __len__(self) -> int:
        return len(self.segments)


@dataclasses.dataclass(kw_only=True)
class Batch(SegmentsMixin):
    """A collection of extracted features for a list of segments.

    Returned by the :class:`SegmentDataset` or ``DataLoader``, this object holds both
    the underlying segments and the computed tensor data ready for machine learning models.

    Attributes
    ----------
    data : dict of str to torch.Tensor
        The extracted feature tensors. Keys match the names of the extractors
        passed to the Segmenter. Tensors are always batched along the first dimension.
    segments : list of Segment
        The :class:`~neuralset.segments.Segment` instances corresponding to this batch. Useful for debugging or custom logic.
    """

    data: dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.data, dict):
            raise TypeError(f"'data' needs to be a dict, got: {type(self.data)}")
        if not self.data:
            raise ValueError(f"No data in {self}")
        # check batch dimension
        batch_size = next(iter(self.data.values())).shape[0]
        if len(self.segments) != batch_size:
            raise RuntimeError(
                f"Incoherent batch size {batch_size} for {len(self.segments)} segments in {self}"
            )

    def to(self, device: str) -> "Batch":
        """Creates a new instance on the appropriate device"""
        out = {name: d.to(device) for name, d in self.data.items()}
        return Batch(data=out, segments=self.segments)

    # pylint: disable=unused-argument
    def __getitem__(self, key: str) -> None:
        raise RuntimeError("Batch is not a dict, use batch.data instead")

    def __repr__(self) -> str:
        shapes = "; ".join(
            f"{k}: {', '.join(str(d) for d in v.shape)}" for k, v in self.data.items()
        )
        n = len(self.segments)
        return f"Batch({n} segments, {self._duration_str()}, shapes: {shapes})"

    def _subselect(self, idx: list[int]):
        """subselect the dataset through index or query"""
        segments = [self.segments[i] for i in idx]
        data = {key: d[idx] for key, d in self.data.items()}
        return self.__class__(data=data, segments=segments)


def validate_extractors(extractors: tp.Mapping[str, Feat]) -> tp.Mapping[str, Feat]:
    """Validate the extractor container provided as input
    and map all cases to the more general a dict of list of sequences of extractors
    """
    if not extractors:
        return {}
    # use extractor names for list
    if not isinstance(extractors, collections.abc.Mapping):
        raise ValueError(f"Only dict of extractors are supported, got {type(extractors)}")
    # single extractors are mapped to list to unify all cases
    return extractors


def prepare_extractors(
    extractors: tp.Sequence[BaseExtractor] | tp.Mapping[str, BaseExtractor],
    events: tp.Any,
) -> None:
    """Prepare the extractors using slurm in parallel, and the others sequentially.

    Parameters
    ----------
    extractors: list/dict of extractors
        the extractors to prepare
    events: DataFrame or list of events/segments
        The structure containing all the events, be it as a dataframe or list of events or list
        of segments.
    """
    from .events.utils import extract_events

    events = extract_events(events)
    extractor_list = list(
        extractors.values() if isinstance(extractors, dict) else extractors
    )
    extractors_using_slurm = [
        extractor
        for extractor in extractor_list
        if hasattr(extractor, "infra")
        and getattr(extractor.infra, "cluster", None) == "slurm"
    ]
    other_extractors = [
        extractor
        for extractor in extractor_list
        if extractor not in extractors_using_slurm
    ]
    slurm_names = ", ".join(
        extractor.__class__.__name__ for extractor in extractors_using_slurm
    )
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for extractor in extractors_using_slurm:
            futures.append(executor.submit(extractor.prepare, events))
            # for debugging:
            futures[-1].__dict__["_name"] = extractor.__class__.__name__
        if extractors_using_slurm:
            msg = f"Started parallel preparation of extractors {slurm_names} on slurm"
            logger.info(msg)
        for extractor in other_extractors:
            logger.info(f"Preparing extractor: {extractor.__class__.__name__}")
            extractor.prepare(events)
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Raise any exception from the task
            except Exception as e:
                name = future.__dict__.get("_name", "UNKNOWN")
                logger.warning("Error occurred while preparing extractor %s: %s", name, e)
                raise


def _get_pad_lengths(
    extractors: tp.Mapping[str, Feat],
    pad_duration: float | None,  # in seconds
) -> dict[str, int]:
    """Precompute pad length in samples for each extractor if applicable
    extractors: mapping of Extractors
        the extractors
    pad_duration: float or None
        padding duration in seconds (if any)
    """
    pad_lengths: dict[str, int] = {}
    if pad_duration is None:
        return pad_lengths
    for name, f in extractors.items():
        if isinstance(f, Feat):
            freq = base.Frequency(f.frequency)
            pad_lengths[name] = freq.to_ind(pad_duration)
    return pad_lengths


def _pad_to(tensor: torch.Tensor, pad_len: int | None):
    """Pad last dimension to a given length"""
    if pad_len is None:
        return tensor
    if pad_len < tensor.shape[-1]:
        msg = "Pad duration is shorter than segment duration, cropping."
        warnings.warn(msg, UserWarning)
        return tensor[:, :pad_len]
    else:
        return torch.nn.functional.pad(tensor, (0, pad_len - tensor.shape[-1]))


@dataclasses.dataclass
class SegmentDataset(torch.utils.data.Dataset[Batch], SegmentsMixin):
    """Dataset defined through :class:`~neuralset.segments.Segment` instances
    and :class:`~neuralset.extractors.BaseExtractor` instances.

    Parameters
    ----------
    extractors: dict of :class:`~neuralset.extractors.BaseExtractor`
        extractors to be computed, returned in the Batch.data dictionary items
    segments: list of :class:`~neuralset.segments.Segment`
        the list of segment instances defining the dataset
    pad_duration: float | tp.Literal["auto"] | None
        pad the segments to the maximum duration or to a specific duration
            None: no padding. Will throw error if segment durations vary.
            "auto": will pad with the max(segments.duration)
    remove_incomplete_segments: bool
        remove segments which do not contain events for one of the extractors
    transforms: dict, optional
        Map of extractor names to transforms (callables transforming the extractor tensor).
        If an extractor name is not present, no transform is applied. Keys must be a subset
        of the extractor names.

    Usage
    -----
    .. code-block:: python

        extractors = {"whatever": ns.extractors.Pulse()}
        ds = ns.SegmentDataset(extractors, segments)
        # one data item
        item = ds[0]
        assert item.data["whatever"].shape[0] == 1  # batch dimension is always added
        # through dataloader:
        dataloader = torch.utils.data.DataLoader(ds, collate_fn=ds.collate_fn, batch_size=2)
        batch = next(iter(dataloader))
        print(batch.data["whatever"])
        # batch.segments holds the corresponding segments

    """

    def __init__(
        self,
        extractors: tp.Mapping[str, Feat],
        segments: tp.Sequence[ns.segments.Segment],
        *,
        remove_incomplete_segments: bool = False,
        pad_duration: float | tp.Literal["auto"] | None = None,
        transforms: dict[str, tp.Callable] | None = None,
    ) -> None:
        super().__init__(segments=segments)
        self.extractors = validate_extractors(extractors)
        self.pad_duration = pad_duration
        self.remove_incomplete_segments = remove_incomplete_segments
        self.segments = _remove_incomplete_segments(
            list(segments), extractors, remove_incomplete_segments
        )
        self._pad_lengths: None | dict[str, int] = None
        transforms = transforms or {}
        additional = set(transforms) - set(extractors)
        if additional:
            raise ValueError(f"Keys in transforms are not present in data: {additional}")
        self.transforms = transforms
        # Hold strong refs to _EventStore instances so they survive pickling
        # to spawn workers (store.__setstate__ self-registers in the worker).
        estores = {seg._store_id: seg._store_ref for seg in self.segments}
        self._event_stores = list(estores.values())

    def __getstate__(self) -> dict[str, tp.Any]:
        state = self.__dict__.copy()
        # Lightweight segments for spawn-worker pickle: copy() produces
        # cache-free clones so we don't serialize per-segment event lists.
        state["segments"] = [seg.copy() for seg in self.segments]
        return state

    def prepare(self) -> None:
        prepare_extractors(self.extractors, self.segments)

    def collate_fn(self, batches: list[Batch]) -> Batch:
        """Creates a new instance from several by stacking in a new first dimension
        for all attributes
        """
        if not batches:
            return Batch(data={}, segments=[])
        if len(batches) == 1:
            return batches[0]
        if not batches[0].data:
            raise ValueError(f"No extractor in first batch: {batches[0]}")
        # move everything to pytorch if first one is numpy
        extractors = {}
        for name in batches[0].data:
            data = [b.data[name] for b in batches]
            try:
                extractors[name] = torch.cat(data, axis=0)  # type: ignore
            except Exception:
                string = f"Failed to collate data with shapes {[d.shape for d in data]}\n"
                logger.warning(string)
                raise
        segments = [s for b in batches for s in b.segments]
        return Batch(data=extractors, segments=segments)

    def _check_padding(self) -> None:  # check if padding is needed
        if self._pad_lengths is not None:
            return
        if self.pad_duration is None:
            if len(set([s.duration for s in self.segments])) > 1:
                msg = "Segments have different durations, so they cannot be collated into batches."
                msg += " Set `pad_duration` to `auto` to pad the segments to the maximum duration."
                raise ValueError(msg)
            pad_duration = self.pad_duration
        elif self.pad_duration == "auto":
            pad_duration = max([s.duration for s in self.segments])
        else:
            pad_duration = self.pad_duration
        self._pad_lengths = _get_pad_lengths(self.extractors, pad_duration)

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int | slice) -> Batch:
        if not isinstance(idx, (int, slice)):
            raise ValueError(f"idx must be int or slice, got {type(idx)}")

        self._check_padding()
        if isinstance(idx, slice):
            indices = list(range(len(self))[idx])
            if not indices:
                return self.collate_fn([])
            return self._subselect(indices).load_all()

        assert isinstance(self._pad_lengths, dict)  # for mpy

        seg = self.segments[idx]
        events = seg.ns_events
        out: dict[str, torch.Tensor] = {}
        for name, extractors in self.extractors.items():
            data = extractors(
                events,
                start=seg.start,
                duration=seg.duration,
                trigger=seg.trigger,
            )
            # pad if need be
            data = _pad_to(data, self._pad_lengths.get(name, None))
            # append to specific extractor list
            out[name] = data[None, ...]  # add back dimension and set
        segment_data = Batch(data=out, segments=[seg])
        for key in segment_data.data:
            if key in self.transforms:
                segment_data.data[key] = self.transforms[key](segment_data.data[key])
        return segment_data

    def build_dataloader(self, **kwargs: tp.Any) -> torch.utils.data.DataLoader:
        """Returns a dataloader for this dataset"""
        self._check_padding()
        return torch.utils.data.DataLoader(self, collate_fn=self.collate_fn, **kwargs)

    def load_all(self, num_workers: int = 0) -> Batch:
        """Returns a single batch with all the dataset data, un-shuffled"""
        num_workers = min(num_workers, len(self))
        batch_size = len(self)
        if num_workers > 1:
            batch_size = max(1, len(self) // (3 * num_workers))
        if num_workers == 1:
            num_workers = 0  # simplifies debugging
        loader = self.build_dataloader(
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
        )
        return self.collate_fn(list(loader))

    def as_one_batch(self, num_workers: int = 0) -> Batch:
        """Deprecated: use :meth:`load_all` instead."""
        warnings.warn(
            "as_one_batch has been renamed to load_all",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.load_all(num_workers=num_workers)

    def __repr__(self) -> str:
        ext = ", ".join(self.extractors) or "none"
        n = len(self.segments)
        return f"SegmentDataset({n} segments, {self._duration_str()}, extractors: {ext})"

    def _subselect(self, idx: list[int]):
        """Subselect the dataset through trigger index or query"""
        segments = [self.segments[i] for i in idx]
        return self.__class__(
            extractors=self.extractors,
            segments=segments,
            pad_duration=self.pad_duration,
            # TODO this should be done in the dataset builder
            remove_incomplete_segments=self.remove_incomplete_segments,
            transforms=self.transforms,
        )


class Segmenter(base.BaseModel):
    """Build a :class:`SegmentDataset` from an events DataFrame and extractors.

    Parameters
    ----------
    extractors: dict of :class:`~neuralset.extractors.BaseExtractor`
        extractors to be computed, returned in the Batch.data dictionary items
    start: float
        Start time (in seconds) of the segment, with respect to the
        :term:`trigger` event (or stride). E.g. use -1.0 if you want the
        segment to start 1s before the event.
    duration: optional float
        Duration (in seconds) of the segment (defaults to event duration if
        ``trigger_query`` is used to extract segments based on specific events).
    trigger_query: optional Query
        Dataframe query selecting which events act as :term:`triggers <trigger>`
        — segments are time-locked to the matching events
        (see :data:`base.Query`).
        At least one of ``trigger_query`` or ``stride`` must be provided.
    stride: optional float
        Stride (in seconds) to use to define sliding window segments.
    stride_drop_incomplete: optional bool
        If True and stride is not None, drop segments that are not fully contained within the
        (start, stop) block.
    padding: optional float | tp.Literal["auto"] | None
        pad the segments to the maximum duration or to a specific duration.
            None: no padding. Will throw error if segment durations vary.
            "auto": will pad with the max(segments.duration)
    drop_incomplete: bool
        remove segments which do not contain events for one of the extractors
    drop_unused_events: bool
        remove events not used by the extractors before creating the segments.

    Usage
    -----
    .. code-block:: python

        extractors = {"whatever": ns.extractors.Pulse()}
        segmenter = ns.Segmenter(extractors=extractors)
        dset = segmenter.apply(events)
        # one data item
        item = dset[0]
        assert item.data["whatever"].shape[0] == 1  # batch dimension is always added
        # through dataloader:
        dataloader = torch.utils.data.DataLoader(dset, collate_fn=dset.collate_fn, batch_size=2)
        batch = next(iter(dataloader))
        print(batch.data["whatever"])
        # batch.segments holds the corresponding segments

    """

    # segments
    start: float = 0.0
    duration: PositiveFloat | None
    trigger_query: base.Query | None = None
    stride: PositiveFloat | None = None
    stride_drop_incomplete: bool = True
    # extractors
    extractors: dict[str, BaseExtractor]
    # dataset
    padding: float | None = None
    drop_incomplete: bool = False
    drop_unused_events: bool = True
    #
    model_config = pydantic.ConfigDict(extra="forbid")

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.trigger_query is None and self.stride is None:
            raise ValueError("At least one of trigger_query or stride must be provided.")

    def apply(self, events: pd.DataFrame) -> SegmentDataset:
        # Segment the events based on stride and/or triggers
        if self.trigger_query is not None:
            trigger_idx = pd.Series(events.query(self.trigger_query).index)
            if not len(trigger_idx):
                raise RuntimeError(
                    f"the trigger query: {self.trigger_query} led to an empty set."
                )
        else:
            trigger_idx = pd.Series(dtype=int)

        # TODO fixme class typing: self.duration cannot be trigger
        duration = None if self.duration == "trigger" else self.duration

        # Drop events not used by extractors or triggers
        if self.drop_unused_events:
            event_types = []
            for extractor in self.extractors.values():
                event_types.extend(EventTypesHelper(extractor.event_types).names)
            event_types.extend(events.loc[trigger_idx].type.unique())
            events = events.loc[events.type.isin(event_types)]

        segments = ns.segments.list_segments(
            events=events,
            triggers=trigger_idx,
            start=self.start,
            duration=duration,
            stride=self.stride,
            stride_drop_incomplete=self.stride_drop_incomplete,
        )

        segments = _remove_incomplete_segments(
            segments, self.extractors, self.drop_incomplete
        )

        if not segments:
            raise RuntimeError(f"empty segments with {self!r}")

        ds = SegmentDataset(
            extractors=self.extractors,
            segments=segments,
            pad_duration=self.padding,
            remove_incomplete_segments=False,
        )
        return ds


def _remove_incomplete_segments(
    segments: list[ns.segments.Segment],
    extractors: tp.Mapping[str, Feat],
    drop_incomplete: bool,
) -> list[ns.segments.Segment]:
    """Check that each segment has a least the events that correspond to each extractor"""

    # List event types required across extractors
    event_types = collections.defaultdict(list)
    for extractor in extractors.values():
        if extractor.aggregation == "trigger":
            continue
        for event_type in extractor._event_types_helper.classes:
            event_types[event_type].append(extractor)

    # Identify problematic segments
    invalid_indices: set[int] = set()
    for event_type, extracts in event_types.items():
        # Find segments that don't have this event type
        invalids = find_incomplete_segments(segments, [event_type])

        if invalids:
            # Checks whether the extractor authorize missing event
            required = any([not f.allow_missing for f in extracts])

            # Raise error if extractor requires this event type
            msg = f"{len(invalids)} segments are missing events of type {event_type.__name__}."
            if not drop_incomplete and required:
                msg += " Use `drop_incomplete=True` to remove them,"
                msg += f" or set {extracts} with `allow_missing=True`."
                raise ValueError(msg)
            if not drop_incomplete and set(range(len(segments))):
                msg = f" . {extracts} allow_missing, however there are"
                msg += f"no segments with {event_type.__name__}. Missing values cannot be guessed."
            else:
                msg += (
                    " They will be populated with default missing values through prepare."
                )
                logger.info(msg)

            invalid_indices.update(invalids)

    # Drop invalid segments
    if drop_incomplete and invalid_indices:
        msg = f"Removing {len(invalid_indices)} segments out of {len(segments)}"
        logger.info(msg)
        segments = [s for i, s in enumerate(segments) if i not in sorted(invalid_indices)]
    return segments
