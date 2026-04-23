# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic
import torch
from exca.base import DEFAULT_CHECK_SKIPS

import neuralset as ns
from neuralset import base
from neuralset.base import Frequency as Frequency
from neuralset.base import TimedArray as TimedArray
from neuralset.events import Event, EventTypesHelper, etypes
from neuralset.segments import Segment

T = tp.TypeVar("T", bound=torch.Tensor | np.ndarray)
logger = logging.getLogger(__name__)


class BaseExtractor(base._Module, base.NamedModel):
    """Base class for extracting features from :term:`events <event>` within a
    :class:`~neuralset.segments.Segment`.

    Subclasses define **what** data to extract (e.g. audio embeddings, EEG
    signals) while ``BaseExtractor`` handles event selection, temporal
    alignment, and multi-event aggregation.

    To create a custom extractor, subclass ``BaseExtractor`` and implement:

    * ``_get_data(events)`` — expensive per-event computation (typically
      cached via ``exca.MapInfra``).
    * ``_get_timed_arrays(events, start, duration)`` — return an iterable of
      :class:`~neuralset.base.TimedArray`, one per event.

    For extractors that produce a single static value per event (no time
    dimension), subclass :class:`BaseStatic` instead and override
    :meth:`~BaseStatic.get_static`.

    Parameters
    ----------
    event_types : str or tuple of str
        Event type name(s) this extractor operates on (e.g. ``"Audio"`` or
        ``("Image", "Text")``).  Must be set as a class-level default in
        every concrete subclass.
    aggregation : str
        Strategy for combining values when multiple matching events fall
        inside the same segment:

        * ``"single"`` — exactly one event expected (raises otherwise).
        * ``"sum"`` / ``"mean"`` — element-wise sum or mean.
        * ``"first"`` / ``"middle"`` / ``"last"`` — pick one event.
        * ``"cat"`` — concatenate along the first dimension.
        * ``"stack"`` — stack along a new first dimension.
        * ``"trigger"`` — use only the trigger event passed to ``__call__``.
    allow_missing : bool
        If True, return a zero tensor when no matching event is found in the
        segment instead of raising.  Requires that
        :meth:`prepare` has been called first so the output shape is known.
    frequency : float or ``"native"``
        Output sampling rate in Hz.  Use ``"native"`` to keep the original
        sampling rate of the input data.  ``0`` is reserved for static
        extractors (:class:`BaseStatic`).
    """

    event_types: str | tuple[str, ...] = ""
    # eg: event_types: str | tuple[str, ...] = ("Image", "Text")

    aggregation: tp.Literal[
        "single",
        "sum",
        "mean",
        "first",
        "middle",
        "last",
        "cat",
        "stack",
        "trigger",
    ] = "single"
    # builds output even when no corresponding event is provided
    allow_missing: bool = False
    frequency: float | tp.Literal["native"] = 0.0

    # internal
    _CLASSES: tp.ClassVar[dict[str, tp.Type["BaseExtractor"]]] = {}
    _effective_frequency: float | None = pydantic.PrivateAttr(None)
    _event_types_helper: EventTypesHelper = pydantic.PrivateAttr()
    _missing_default: torch.Tensor | None = pydantic.PrivateAttr(None)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        # check params
        super().__init_subclass__()
        # add event requirements to the extractor requirements
        if not cls._can_be_instantiated():
            return
        name = cls.__name__
        BaseExtractor._CLASSES[name] = cls
        model_fields: dict[str, pydantic.FieldInfo] = cls.model_fields  # type: ignore
        event_types: tp.Any = model_fields["event_types"].default  # type:ignore
        if not event_types:
            msg = f"Default event_types must be specified for {cls.__name__}"
            raise RuntimeError(msg)
        # security checks for new _get_data
        legafuncs = [
            "_get_latents",
            "_get_latent",
            "_get_preprocessed_data",
            "_events_to_data",
            "_get_channel_positions",
        ]
        for func in legafuncs:
            if hasattr(cls, func):
                msg = f'In {name!r}, found function {func!r} which should be renamed to "_get_data"'
                raise RuntimeError(msg)
        infrafield = cls.model_fields.get("infra", None)
        if infrafield is not None:
            funcname = infrafield.default._infra_method.method.__name__
            if funcname != "_get_data":
                msg = f'In {name!r}, found infra decorating {funcname!r} it should be "_get_data" by convention'
                raise RuntimeError(msg)
        # security checks for new event_types
        if not isinstance(event_types, str):
            is_tuple = isinstance(event_types, tuple)
            if not (is_tuple and all(isinstance(d, str) for d in event_types)):
                msg = f"In {name!r}, event_types attribute must be a string "
                msg += f"or tuple of string, got {event_types}"
                raise TypeError(msg)
        type_helper = EventTypesHelper(event_types)
        for etype in type_helper.classes:
            cls.requirements = cls.requirements + etype.requirements

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self._event_types_helper = EventTypesHelper(self.event_types)
        name = self.__class__.__name__
        if self.frequency != "native" and self.frequency < 0.0:
            msg = f"{name}.frequency is neither 'native' nor >= 0 (got {self.frequency})."
            raise ValueError(msg)
        if not (self.frequency or isinstance(self, BaseStatic)):
            msg = f"{name}.frequency=0 is only allowed for static extractors (did you mean 'native'?)"
            raise ValueError(msg)

    def _exclude_from_cache_uid(self) -> list[str]:
        # extractor convention from inheriting cache uid exclusion list
        return ["aggregation", "allow_missing"]

    def prepare(
        self, obj: pd.DataFrame | tp.Sequence[Event] | tp.Sequence[Segment]
    ) -> None:
        """Pre-compute and cache extractor data for a collection of events.

        This method triggers ``_get_data`` on every matching event so that
        expensive computation (e.g. model inference) is done once and cached.
        It then calls the extractor on a single event to populate the output
        shape, which is needed when ``allow_missing=True``.

        Call ``prepare`` before using the extractor in a dataloader.

        Parameters
        ----------
        obj : DataFrame or sequence of Event or sequence of Segment
            The structure containing the events.  When calling ``prepare`` on
            several objects, prefer passing a list of events or segments over
            a DataFrame to avoid redundant conversion overhead.
        """
        events = self._event_types_helper.extract(obj)
        if self.frequency == "native" and events and hasattr(events[0], "frequency"):
            freqs = set(e.frequency for e in events)  # type: ignore
            cls = self.__class__.__name__
            if len(freqs) > 1:
                msg = f"frequency='native' in {cls} with several different frequencies: {freqs}"
                msg += "\n(all data will not be processing at the same frequency, "
                msg += "should you set the extractor frequency?"
                logger.warning(msg)
            elif len(freqs) == 1:
                cls = self.__class__.__name__
                freq = list(freqs)[0]
                msg = f"Processing to native frequency in {cls}.prepare: {freq}Hz"
                logger.info(msg)
        self._get_data(events)
        if events:  # run extractor on 1 event to populate shape
            self(
                events[0],
                start=events[0].start,
                duration=0.001,
                trigger=events[0],
            )

    def _get_data(self, events: list[Event]) -> tp.Iterable[tp.Any]:
        """Put heavy computation steps here, and cache the result using exca.MapInfra"""
        for _ in events:
            yield None
        raise NotImplementedError

    def _get_timed_arrays(
        self, events: list[Event], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        raise NotImplementedError

    def __call__(
        self,
        events: tp.Any,  # too complex: pd.DataFrame | list | dict | ns.events.Event,
        start: float,
        duration: float,
        trigger: Event | pd.Series | dict | None = None,
    ) -> torch.Tensor:
        """events: the single event (dict | ns.events.Event) or the series
        of events (list of Events | pd.DataFrame) describing the events, each
        containing start and duration.
        start: the start of the segment in the same timeline as the event.
        duration: the duration of the segment.
        """
        from neuralset.events.utils import extract_events

        # Check input
        start, duration = float(start), float(duration)
        if duration < 0.0:
            raise ValueError(f"duration must be >= 0, got {duration}")
        if trigger is not None and not isinstance(trigger, Event):
            triggers = extract_events(trigger)
            if len(triggers) != 1:
                msg = f"trigger must be a single event, got {len(triggers)} from {trigger!r}"
                raise TypeError(msg)
            trigger = triggers[0]

        ns_events = self._get_relevant_events(
            events, trigger, start=start, duration=duration
        )

        # If there is no valid event, return default values or raise error.
        if not ns_events:
            if self.allow_missing and self._missing_default is not None:
                # Return tensor with default values if you can.
                if self._effective_frequency is not None:
                    default = self._missing_default
                    ef = Frequency(self._effective_frequency)
                    if ef:
                        n_times = max(1, ef.to_ind(duration))
                        reps = [1 for _ in range(default.ndim)] + [n_times]
                        default = default.unsqueeze(-1).repeat(reps)
                    return default
                else:
                    msg = f"_missing_default was set for {self.name} but _effective_frequency is missing"
                    raise RuntimeError(msg)

            # Raise if you can't return default values.
            event_types = self._event_types_helper.classes
            found_types = {type(e) for e in extract_events(events)}
            msg = f"No {event_types} found in segment for extractor {self.name} "
            msg += f"(types found: {found_types} in {events!r}) "
            if not self.allow_missing:
                msg += (
                    f"(filter invalid segments or set allow_missing=True to {self.name})"
                )
            else:
                msg += "and extractor shape not populated "
                msg += '(you may need to call "prepare" on the extractor).'
            raise ValueError(msg)

        # Main part: get data from each event
        tarrays = list(
            self._get_timed_arrays(events=ns_events, start=start, duration=duration)
        )

        # Store frequency (TimedArray frequency for native, else declared frequency)
        freq = float(
            tarrays[0].frequency if self.frequency == "native" else self.frequency
        )
        if self._effective_frequency is None:
            self._effective_frequency = freq

        # Align and combine the data of all events in the segment
        tensor = self._tarrays_to_tensor(
            tarrays,
            start,
            duration,
            freq,
            self.aggregation,
        )

        # Store default value
        if self._missing_default is None:
            # last dimension is time if frequency is not 0
            shape = tuple(tensor.shape[: -1 if freq else None])
            self._missing_default = torch.zeros(*shape, dtype=tensor.dtype)

        return tensor

    def _get_relevant_events(
        self,
        events: tp.Any,
        trigger: Event | None,
        *,
        start: float,
        duration: float,
    ) -> list[Event]:
        """Select the extractor-relevant events for the segment window and aggregation."""
        # if trigger-aggregation, we only extract data from the trigger event
        if self.aggregation == "trigger":
            event_types = self._event_types_helper.classes
            if not isinstance(trigger, event_types):
                aggregation = self.aggregation
                msg = f"Extractor {self.name} has {aggregation=} but trigger is {trigger!r} (not {event_types})"
                raise ValueError(msg)
            events = [trigger]

        # make sure events is list[Event]
        ns_events = self._event_types_helper.extract(events)
        if ns_events and len(timelines := {e.timeline for e in ns_events}) > 1:
            msg = f"Multiple timelines {timelines} in events passed to extractor {self!r}: {ns_events}"
            raise ValueError(msg)

        if self.aggregation != "trigger":
            stop = start + duration
            in_window = [
                e for e in ns_events if e.start < stop and e.start + e.duration > start
            ]
            # fall back to full list when nothing overlaps, so raw-data events
            # (Meg, Audio, Fmri) called outside their nominal range still reach
            # the zero-fill in _tarrays_to_tensor (see test_first_samp).
            ns_events = in_window or ns_events

        if self.aggregation in ("first", "trigger", "single"):
            if self.aggregation == "single" and len(ns_events) > 1:
                msg = (
                    f"Found {len(ns_events)} events in the segment but expected only one "
                )
                msg += f"since {self.name}.aggregation='single'. "
                msg += "Update it to sum/average/first/trigger/... ?\n"
                msg += f"{ns_events=}"
                raise ValueError(msg)
            ns_events = ns_events[:1]
        elif self.aggregation == "last":
            ns_events = ns_events[-1:]
        elif self.aggregation == "middle":
            ns_events = [ns_events[len(ns_events) // 2]]

        return ns_events

    @staticmethod
    def _tarrays_to_tensor(
        tarrays: list[TimedArray],
        start: float,
        duration: float,
        frequency: float,
        aggregation: (
            tp.Literal["mean", "sum", "trigger", "cat", "stack"]
            | tp.Literal[
                "single", "first", "middle", "last"
            ]  # equivalent to sum, expect a single tarrays
        ),
    ) -> torch.Tensor:
        """Combine a list of time array into a torch Tensor."""

        # aggregate time arrays
        err_msg = "Something went wrong. There should be only 1 trigger, got %s"
        if aggregation == "trigger" and not frequency:
            if not len(tarrays) == 1:
                raise RuntimeError(err_msg % tarrays)
            tarrays[0].start = start  # fake an overlap as start time does not matter

        match aggregation:
            case "trigger" | "single" | "first" | "middle" | "last":
                aggregation = "sum"
                # expect a single Event
                if not len(tarrays) == 1:
                    raise RuntimeError(err_msg % tarrays)
            case "mean" | "sum" | "cat" | "stack":
                pass
            case _:
                raise RuntimeError(f"unknown aggregation: {aggregation}")

        segment_info: dict[str, tp.Any] = {
            "start": start,
            "frequency": frequency,
            "duration": duration,
        }

        if aggregation not in ("cat", "stack"):
            # Sum event data in that segment
            tarray = TimedArray(aggregation=aggregation, **segment_info)  # type: ignore
            for ta in tarrays:
                tarray += ta
            tensor = torch.from_numpy(tarray.data)
        else:
            # Gather all event data
            arrays = []
            for ta in tarrays:
                # Define the time window to populate the data tensor
                tarray = TimedArray(**segment_info)
                # Actually add the data
                tarray += ta
                # Concatenate the tensors
                arrays.append(tarray.data)
            if aggregation == "cat":
                data = np.concatenate(arrays, axis=0)
            else:
                assert aggregation == "stack"
                data = np.stack(arrays, axis=0)
            tensor = torch.from_numpy(data)

        if not tensor.ndim:
            tensor = tensor.unsqueeze(0)
        return tensor


class BaseStatic(BaseExtractor):
    """Base class for extractors that produce one feature vector per event.

    Subclasses implement :meth:`get_static` instead of the full
    dynamic pipeline.  ``frequency`` defaults to 0 (no time axis).
    """

    frequency: float = 0.0  # FIXME should be pydantic.PositiveFloat

    def get_static(self, event: etypes.Event) -> torch.Tensor:
        """Return a single feature vector for the given event.

        Override this method in subclasses to produce a static (non-temporal)
        embedding for one event.  The returned tensor should have no time
        dimension — temporal wrapping is handled by ``BaseStatic``
        automatically.

        Parameters
        ----------
        event : Event
            The event to extract a feature from.

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(*feature_shape,)`` (no time axis).
        """
        raise NotImplementedError

    def _get_timed_arrays(
        self, events: list[Event], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        for event in events:
            embedding = self.get_static(event)
            ta = TimedArray(
                frequency=0,
                duration=event.duration,
                start=event.start,
                data=embedding.numpy(),
            )
            yield ta


# pylint: disable=unused-argument
def _skip_new_event_types(key, val, prev):
    if "event_types" in key and not prev:
        return True
    return False


DEFAULT_CHECK_SKIPS.append(_skip_new_event_types)


class HuggingFaceMixin(base.BaseModel):
    """Mixin for extractors that use a HuggingFace model.
    These extractors all return a tensor of shape (n_layers, n_tokens, *embedding_shape).
    This mixin determines how to aggregate the layers and tokens.

    Parameters
    ----------
    model_name: str
        Name of the model to use.
    device: str
        Device to use for the model:
        - cpu: for cpu computation
        - cuda: for using gpu0
        - auto: to use gpu if available else cpu
        - accelerate: to use huggingface accelerate (maps to multiple-gpus + use float16)
    layers: float | list[float] | "all"
        Specifies the layers to keep.
        - "all": keep all layers
        - a float between 0 and 1 (or list of): the relative depth of the layer(s) to use,
        where 0 stands for the first layer and 1 for the last layer.
    cache_n_layers: None, int
        if provided, n equidistributed layers will be cached
        If None, only cache the output of the layers specified by `layers`.
    layer_aggregation: str
        How to aggregate the layers (first dimension of the tensor of activations).
        Can be "mean", "sum", "group_mean" or None (in which case we keep the original dimension).
        "group_mean" will average the layers by groups defined by the layers parameter,
        e.g. if layers=[0, 0.5, 1] and there are 10 layers, the layers will be grouped as [0:5], [5:10].
    token_aggregation: str
        How to aggregate the tokens (second dimension of the tensor of activations).
        Can be "first", "last", "mean", "sum", "max", or None (in which case we keep the original dimension).

    Requires `transformers` and `huggingface_hub`. You can install them manually
    or via `pip install "neuralset[all]"`.
    """

    requirements: tp.ClassVar[tuple[str, ...]] = (
        "transformers>=4.29.2",
        "huggingface_hub>=0.27.0",
    )
    model_name: str
    device: tp.Literal["auto", "cpu", "cuda", "accelerate"] = "auto"
    layers: float | list[float] | tp.Literal["all"] = 2 / 3
    cache_n_layers: int | None = None
    layer_aggregation: tp.Literal["mean", "sum", "group_mean"] | None = "mean"
    token_aggregation: tp.Literal["first", "last", "mean", "sum", "max"] | None = "mean"
    _REPOS: tp.ClassVar[list[str]] = []
    _skip_repo_check: bool = False  # for simpler hacking (eg: custom dinov2 checkpoints)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        name = self.__class__.__name__
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.layers != "all":
            layers = self.layers if isinstance(self.layers, list) else [self.layers]
            if not all(isinstance(layer, float) and 0 <= layer <= 1 for layer in layers):
                raise ValueError(f"The layers must be floats between 0 and 1 for {name}")
        if self.cache_n_layers == 1:
            msg = f"Set {name}.cache_n_layers=None instead of 1"
            raise ValueError(msg)
        if not self._skip_repo_check and not self.repo_exists():
            raise ValueError(f"The model {self.model_name} does not exist")

    def repo_exists(self) -> bool:
        fp = Path(__file__).with_name("data") / "huggingface-repos.txt"
        name = self.model_name
        if not self._REPOS:  # load offline huggingface white list
            if not fp.exists():
                raise RuntimeError(f"Please reinstall neuralset: missing file {fp}")
            self._REPOS.extend(fp.read_text("utf8").splitlines())
        if name in self._REPOS:
            return True
        else:
            from huggingface_hub import repo_info

            try:  # if not in whitelist, check through API and save it if it exists
                repo_info(name)
                self._REPOS.append(name)
                self._REPOS.sort()
                try:
                    fp.write_text("\n".join(self._REPOS))
                except Exception:
                    pass  # nevermind if there is no write permission
                return True
            except Exception:
                return False

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["device"]

    def _exclude_from_cache_uid(self) -> list[str]:
        excluded = ["device"]
        if self.cache_n_layers is not None:
            excluded.extend(["layers", "layer_aggregation"])
        return excluded

    def _aggregate_layers(self, latents: np.ndarray) -> np.ndarray:
        """
        Input:
            a tensor of activations of shape (n_model_layers, *embedding_shape)
        Output:
            a tensor of activations of following shape:
            - (len(self.layers), *embedding_shape) if layer_aggregation is None,
            - (len(self.layers)-1, *embedding_shape) if layer_aggregation is "group_mean",
            - (*embedding_shape) if layer_aggregation is "mean" or "sum".
        This function should be called before caching to reduce the size of the cached tensors,
        except if cache_n_layers > 1, in which case it should be called after caching.
        """
        n_model_layers = latents.shape[0]
        if self.layers == "all":
            layer_indices = list(range(n_model_layers))
        else:
            layers = self.layers if isinstance(self.layers, list) else [self.layers]
            layer_indices = np.unique(
                [int(i * (n_model_layers - 1)) for i in layers]
            ).tolist()  # type: ignore
        if len(layer_indices) == 1:
            if self.layer_aggregation is None:
                return latents[layer_indices[0]][None, :]
            else:
                return latents[layer_indices[0]]
        else:  # aggregate
            latents = np.asarray(latents)  # ContiguousMemmap must be loaded first
            if self.layer_aggregation == "mean":
                return latents[layer_indices].mean(0)  # type: ignore
            elif self.layer_aggregation == "sum":
                return latents[layer_indices].sum(0)  # type: ignore
            elif self.layer_aggregation == "group_mean":
                groups = []
                layer_indices[-1] += 1
                for l1, l2 in zip(layer_indices[:-1], layer_indices[1:]):
                    groups.append(latents[l1:l2].mean(0))
                return np.stack(groups)
            elif self.layer_aggregation is None:
                return latents[layer_indices]
            else:
                raise ValueError(f"Unknown layer aggregation: {self.layer_aggregation}")

    def _layer_subselection(self, latents: T) -> T:
        n_layers = latents.shape[0]
        if self.cache_n_layers is None or self.cache_n_layers >= n_layers:
            return latents
        selected = [
            int(round(x)) for x in np.linspace(0, n_layers - 1, self.cache_n_layers)
        ]
        return latents[selected, ...]  # type: ignore

    def _aggregate_tokens(self, latents: T) -> T:
        """Aggregate tokens and subselects the layers

        Input:
            a tensor of activations of shape (layer_idx, token_idx, *embedding_shape)
        Output:
            a tensor of activations with possibly downsampled number of layers, and token
            dimension removed through the aggregation method (if not None)
        This function should always be called before caching, to reduce the size
        of the cached tensors.
        """
        latents = self._layer_subselection(latents)
        match self.token_aggregation:
            case "mean":
                out = latents.mean(axis=1)  # type: ignore
            case "sum":
                out = latents.sum(axis=1)  # type: ignore
            case "max":
                out = latents.max(axis=1)  # type: ignore
                if isinstance(latents, torch.Tensor):
                    out = out.values
            case "first":
                # np.take in numpy, torch.select in torch
                out = latents[:, 0, ...]
            case "last":
                out = latents[:, -1, ...]
            case None:
                out = latents
        if isinstance(out, torch.Tensor):
            out = out.float()  # recast bf16
        return out  # type: ignore


class Pulse(BaseStatic):
    """Constant-one extractor — returns a single 1.0 scalar for every event.

    Useful as a bias term or baseline feature in models that expect at
    least one extractor per event type.
    """

    event_types: str | tuple[str, ...] = "Event"

    def get_static(self, event: etypes.Event) -> torch.Tensor:
        return torch.ones(1, dtype=torch.float32)


class EventField(BaseStatic):
    """Extractor which extracts an int or float attribute from an event.

    `event_field` can be either an attribute of the event or a key in the event.extra dictionary.

    Parameters
    ----------
    event_types : str or tuple of str
        Type of event(s) to apply this extractor to.
    event_field : str
        Field to extract from the event.
    """

    event_types: str | tuple[str, ...] = "Event"
    event_field: str

    _dtype: torch.dtype | None = pydantic.PrivateAttr(None)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.event_types == "Event":
            warnings.warn(
                f"{self.name}: event_types has not been set, are you sure you want to apply this extractor to all events?"
            )

    def _get_field_values(
        self,
        obj: pd.DataFrame | tp.Sequence[Event] | tp.Sequence[Segment],
    ) -> set[tp.Any]:
        events = self._event_types_helper.extract(obj)
        if not events:
            raise ValueError(f"No events found for {self.name}")
        return set(e._get_field_or_extra(self.event_field) for e in events)

    def prepare(
        self, obj: pd.DataFrame | tp.Sequence[Event] | tp.Sequence[Segment]
    ) -> None:
        values = self._get_field_values(obj)

        types = [type(v) for v in values]
        if len(set(types)) > 1:
            raise ValueError(
                f"Field {self.event_field} has different types in events for {self.__class__.__name__}: {types}"
            )

        # Enforce int or float types
        int_types = (int, np.int32, np.int64)
        float_types = (float, np.float32, np.float64)
        if types[0] not in (int_types + float_types):
            msg = f"Field {self.event_field} has type {types[0]}, which is not supported."
            raise ValueError(msg)
        self._dtype = torch.float32 if types[0] in float_types else torch.long

    def get_static(self, event: etypes.Event) -> torch.Tensor:
        if self._dtype is None:
            msg = f"{self.name}: Must call extractor.prepare(events) before using the extractor."
            raise ValueError(msg)
        return torch.tensor(
            [event._get_field_or_extra(self.event_field)],
            dtype=self._dtype,
        )


class LabelEncoder(EventField):
    """Encode a given field from an event, e.g. to be used as a label.

    Parameters
    ----------
    event_types : str or tuple of str
        Type of event(s) to apply this extractor to.
    event_field : str
        Field to encode from the event.
    allow_missing : bool
        If True, allow missing events without raising errors.
    treat_missing_as_separate_class: bool
        If True, treat missing events as a separate class with index -1, or one-hot vector with
        last index set to 1. This is only relevant if allow_missing is True.
        Note: If using LabelEncoder for a multilabel classification task, set this to False for
        missing labels to be represented by a vector of all zeros.
    return_one_hot : bool
        If True, return one-hot representation of the index. Otherwise, return an int in
        [0, n_unique_values - 1] (or the corresponding values provided in ``predefined_mapping``,
        and ``-1`` for missing events if ``treat_missing_as_separate_class=True``).
    predefined_mapping : dict, optional
        If provided, use this mapping from label to index instead of computing it from data. Values
        must be >= 0. If ``return_one_hot=True``, these indices MUST be contiguous and start from 0.
    """

    treat_missing_as_separate_class: bool = False
    return_one_hot: bool = False
    predefined_mapping: dict[str, int] | None = None

    _label_to_ind: dict[str, int] = {}
    _n_classes: int = 0

    def model_post_init(self, log__):
        super().model_post_init(log__)

        if (
            self.allow_missing
            and not self.return_one_hot
            and not self.treat_missing_as_separate_class
        ):
            msg = (
                "Missing events will be encoded using the default all-zero value "
                "(for example, 0 or a zero vector/tensor), which may be indistinguishable "
                "from a valid class if that class is also mapped to zeros. "
                "Set treat_missing_as_separate_class=True to avoid this."
            )
            logger.warning(msg)

        if self.predefined_mapping is not None:
            indices = set(self.predefined_mapping.values())
            if min(indices) < 0:
                msg = f"predefined_mapping values must be >= 0, got {min(indices)}."
                raise ValueError(msg)
            expected_indices = set(range(min(indices), min(indices) + len(indices)))
            if indices != expected_indices:
                msg = f"Label indices are not contiguous. Expected indices: {expected_indices}, "
                msg += f"but got: {indices}. This may cause issues with one-hot encoding "
                msg += "or class-based operations."
                logger.warning(msg)

            if (
                self.treat_missing_as_separate_class
                and "__missing__" in self.predefined_mapping
            ):
                msg = "Key '__missing__' is reserved when treat_missing_as_separate_class is True."
                raise ValueError(msg)

    def prepare(
        self, obj: pd.DataFrame | tp.Sequence[Event] | tp.Sequence[Segment]
    ) -> None:
        labels = self._get_field_values(obj)
        # Build mapping on first call; subsequent calls are no-ops if labels are known
        mapping = self._label_to_ind or self.predefined_mapping
        if mapping:
            unknown = labels - mapping.keys()
            if unknown:
                msg = (
                    f"Labels {unknown} are not in the existing mapping "
                    f"{sorted(mapping)}. If prepare() is called per-subject "
                    "after an initial full-data prepare(), this means new "
                    "labels appeared. Use predefined_mapping to fix the "
                    "mapping upfront."
                )
                raise ValueError(msg)
            if self._label_to_ind:
                return
            self._label_to_ind = dict(self.predefined_mapping)  # type: ignore[arg-type]
        else:
            if len(labels) < 2:
                logger.warning(
                    f"LabelEncoder has only found one label: {labels}. "
                    "This was probably not intended."
                )
            self._label_to_ind = {label: i for i, label in enumerate(sorted(labels))}

        self._n_classes = len(set(self._label_to_ind.values()))
        if self.treat_missing_as_separate_class:
            self._label_to_ind["__missing__"] = -1  # Register in dict for transparency
            self._n_classes += 1

            if self.return_one_hot:
                self._missing_default = torch.nn.functional.one_hot(
                    torch.tensor([self._n_classes - 1], dtype=torch.long),
                    num_classes=self._n_classes,
                )[0]
            else:
                self._missing_default = torch.tensor([-1], dtype=torch.long)

        events = self._event_types_helper.extract(obj)
        self(
            events[0],
            events[0].start,
            duration=0.001,
            trigger=events[0],
        )

    def get_static(self, event: etypes.Event) -> torch.Tensor:
        if not self._label_to_ind:
            msg = f"{self.name}: Must call extractor.prepare(events) before using the extractor."
            raise ValueError(msg)
        inds = [self._label_to_ind[event._get_field_or_extra(self.event_field)]]
        label = torch.tensor(inds, dtype=torch.long)
        if self.return_one_hot:
            # Remove the batch dim
            label = torch.nn.functional.one_hot(label, num_classes=self._n_classes)[0]
        return label


class EventDetector(BaseExtractor):
    """Extracts time-aligned extractors from event annotations.

    Parameters
    ----------
    event_types: str
        the event type to detect (e.g., "Keystroke").
    frequency: float
        sampling frequency in Hz.
    mode: str
        mode of labeling ("dense", "start", "center", "duration").
    allow_missing: bool
        if True, missing events are allowed without raising errors.
    """

    event_types: str = "Event"
    frequency: float = 100.0
    mode: tp.Literal["dense", "start", "duration", "center"] = "dense"
    allow_missing: bool = True
    # freeze aggregation as it is bypassed
    aggregation: tp.Literal["sum"] = "sum"

    def _get_timed_arrays(self, events, start: float, duration: float):
        freq = Frequency(self.frequency)
        n_samples = max(1, freq.to_ind(duration))
        data = np.zeros((1, n_samples), dtype=np.float32)

        for event in events:
            idx_start = freq.to_ind(event.start - start)
            idx_end = freq.to_ind(event.start + event.duration - start)
            idx_center = freq.to_ind(event.start + event.duration / 2 - start)

            value = 1.0
            if self.mode == "dense":
                data[0, max(0, idx_start) : min(n_samples, idx_end)] = value
                continue
            elif self.mode == "start":
                idx = idx_start
            elif self.mode == "center":
                idx = idx_center
            elif self.mode == "duration":
                idx = idx_center
                value = min(event.duration / duration, 1.0)
            else:
                raise NotImplementedError(f"Mode {self.mode} is not implemented")

            if 0 <= idx < n_samples:
                data[0, idx] = value

        yield ns.base.TimedArray(
            data=data,
            start=start,
            duration=duration,
            frequency=self.frequency,
        )
