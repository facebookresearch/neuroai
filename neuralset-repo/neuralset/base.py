# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
import collections
import importlib.util
import logging
import pprint
import typing as tp
from pathlib import Path

import exca
import exca.steps
import numpy as np
import pydantic
import yaml
from exca.cachedict import DumpContext

PathLike = str | Path


# # # # # CONFIGURE LOGGER # # # # #
logger = logging.getLogger("neuralset")
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s", "%Y-%m-%d %H:%M:%S"
)
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
# # # # # CONFIGURED LOGGER # # # # #


def _int_cast(v: tp.Any) -> tp.Any:
    """casts integers to string"""
    if isinstance(v, int):
        return str(v)
    return v


# type hint for casting integers to string
# this is useful for subject field which can be automatically converted from
# str to int by pandas
StrCast = tp.Annotated[str, pydantic.BeforeValidator(_int_cast)]


def _coerce_spec(v: tp.Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for key, val in v.items():
            if not isinstance(key, str) or not isinstance(val, str):
                raise TypeError(
                    f"spec keys and values must be strings, got {key!r}: {val!r}"
                )
        return "&".join(f"{k}={v[k]}" for k in sorted(v))
    raise TypeError(f"spec must be a dict[str, str], got {type(v).__name__}")


FmriSpec = tp.Annotated[str, pydantic.BeforeValidator(_coerce_spec)]
"""Variant parameters encoded as sorted ``key=value&...`` string.

Accepts a ``dict[str, str]`` on input and auto-encodes it::

    spec={"registration": "msmall", "resolution": "1.6mm"}
    # stored as "registration=msmall&resolution=1.6mm"
"""


def validate_query(query: str) -> str:
    """Ensure a pandas query string is a valid Python expression.

    Examples
    --------
    >>> validate_query("index >= 0")
    'index >= 0'
    >>> validate_query("type in ['Word', 'Audio']")
    "type in ['Word', 'Audio']"

    Raises
    ------
    ValueError
        If the query is not valid Python expression syntax, or if it
        contains ``@`` references to local variables (fragile when the
        query is defined far from its execution site).
    """
    try:
        ast.parse(query, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Query '{query}' is not valid: {exc}") from exc
    if "@" in query:
        raise ValueError(
            f"Query '{query}' uses '@' to reference local variables. "
            "This is not supported because the query may be evaluated in a "
            "different scope from where it was defined."
        )
    return query


Query = tp.Annotated[str, pydantic.AfterValidator(validate_query)]
"""Validated pandas query string.

Use as a Pydantic field type to automatically check syntax on construction::

    class MyModel(pydantic.BaseModel):
        query: Query               # required valid query
        query: Query | None = None # optional query, None means "no filter"
"""

CACHE_FOLDER = Path.home() / ".cache/neuralset/"
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)


class NamedModel(exca.helpers.DiscriminatedModel, discriminator_key="name"):
    @property
    def name(self) -> str:  # for compatibility
        return self.__class__.__name__


class BaseModel(pydantic.BaseModel):
    """Base pydantic model with extra=forbid and nicer print"""

    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")

    def __repr__(self) -> str:
        data = self.model_dump()
        return f"{self.__class__.__name__}(**\n{pprint.pformat(data, indent=2)}\n)"


class _Module(BaseModel):
    requirements: tp.ClassVar[tuple[str, ...]] = ()

    @classmethod
    def _check_requirements(cls) -> None:
        """Verify that all packages listed in ``requirements`` are importable.

        Raises ``ModuleNotFoundError`` with a copy-pasteable ``pip install``
        command when one or more packages are missing.
        """
        # pip name → importable name, for packages where they differ
        import_names = {
            "pillow": "PIL",
            "scikit-image": "skimage",
            "opencv-python": "cv2",
            "datalad-installer": "datalad_installer",
            "openneuro-py": "openneuro",
            "sonar-space": "sonar",
        }
        missing = []
        for req in cls.requirements:
            name = req.split(">")[0].split("<")[0].split("=")[0].split("[")[0]
            spec_name = import_names.get(name, name.replace("-", "_"))
            if importlib.util.find_spec(spec_name) is None:
                missing.append(name)
        if missing:
            raise ModuleNotFoundError(
                f"{cls.__name__} requires packages that are not installed.\n"
                f"  pip install {' '.join(missing)}"
            )

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return []

    @tp.final  # make sure nobody gets it wrong and override it
    def __post_init__(self) -> None:
        """This should not exist in subclasses, as we use pydantic's model_post_init"""

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        # get requirements from superclasses as well
        reqs = tuple(x.strip() for x in cls.requirements)
        for base in cls.__bases__:
            breqs = getattr(base, "requirements", ())
            if breqs is not cls.requirements:
                reqs = breqs + reqs
        cls.requirements = reqs
        # check for _exclude_from_cls_uid override as attribute as it gets ignored
        exc_tag = "_exclude_from_cls_uid"
        if exc_tag in getattr(cls, "__private_attributes__", {}):
            msg = f"Class {cls.__name__!r} cannot have a private attr {exc_tag!r} "
            msg += f"use a method `def {exc_tag}(cls) -> list[str]` instead."
            msg += f"(defined in module: {cls.__module__})"
            raise TypeError(msg)

    @classmethod
    def _can_be_instantiated(cls) -> bool:
        return not cls.__name__.startswith(("Base", "_"))


class Frequency(float):
    """Sampling rate in Hz, with helpers for second/sample conversion.

    A ``float`` subclass that provides ``to_ind`` (seconds → sample index)
    and ``to_sec`` (sample index → seconds).

    Examples
    --------
    >>> freq = Frequency(100.0)
    >>> freq.to_ind(0.5)   # 0.5 s at 100 Hz → sample 50
    50
    >>> freq.to_sec(50)    # sample 50 at 100 Hz → 0.5 s
    0.5

    .. admonition:: Design rationale — ``to_ind`` uses ``round()``

       ``to_ind`` uses a single rule — ``round(seconds * freq)`` — for
       both start times and durations, rather than mixing floor/ceil
       depending on context. This minimizes worst-case alignment error
       to ±0.5 samples and keeps the conversion trivially predictable.
    """

    @tp.overload
    def to_ind(self, seconds: float) -> int: ...

    @tp.overload  # noqa
    def to_ind(self, seconds: np.ndarray) -> np.ndarray:  # noqa
        ...

    def to_ind(self, seconds: tp.Any) -> tp.Any:  # noqa
        """Convert time in seconds to a sample index.

        Uses ``round(seconds * frequency)`` to produce a deterministic
        sample count for any given duration.
        """
        if isinstance(seconds, np.ndarray):
            return np.round(seconds * self).astype(int)
        return int(round(seconds * self))

    @tp.overload
    def to_sec(self, index: int) -> float: ...

    @tp.overload  # noqa
    def to_sec(self, index: np.ndarray) -> np.ndarray:  # noqa
        ...

    def to_sec(self, index: tp.Any) -> tp.Any:  # noqa
        """Converts a sample index to a time in seconds"""
        return index / self

    @staticmethod
    def _yaml_representer(dumper, data):
        "Represents Frequency instances as floats in yamls"
        return dumper.represent_scalar("tag:yaml.org,2002:float", str(float(data)))


_UNSET_START = float("inf")

_TA = tp.TypeVar("_TA", bound="TimedArray")


@DumpContext.register
class TimedArray:
    """Numpy array annotated with time metadata.

    Carries ``frequency``, ``start``, ``duration``, and an optional
    ``header`` dict for domain-specific attributes (channel names,
    electrode positions, space info). Time is always the last dimension
    (when ``frequency > 0``).

    .. admonition:: Design rationale

       Attaching ``frequency`` and ``start`` to the array lets extractors
       handle time slicing uniformly — whether the data is a time series
       (``frequency > 0``, slicing by sample indices) or a single static
       representation (``frequency == 0``, no time dimension).
       Slicing and ``+=`` handle time alignment automatically (resampling
       and shifting to a common grid). Data can be backed by memmap for
       fast access without loading full arrays into memory.
    """

    def __init__(
        self,
        *,  # forbid positional
        frequency: float,
        start: float,
        data: np.ndarray | None = None,
        duration: float | None = None,
        aggregation: tp.Literal["sum", "mean"] = "sum",
        header: dict[str, tp.Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        frequency: float
            Sampling frequency of the data. If >0, the last dimension
            of the data is time; if 0 the data has no time dimension.
        start: float
            Start time of the data in seconds.
        data: optional array
            If provided, the data with time as last dimension (when
            frequency > 0).
        duration: optional float
            Duration of the data. If ``data`` is also provided and
            ``frequency > 0``, the last dimension is checked for
            consistency. If ``data`` is not provided, shape is inferred
            from the first data added.
        aggregation: "sum" or "mean"
            Aggregation mode on the time domain when adding to the
            timed array.
        header: optional dict
            Domain-specific attributes (channel names, electrode
            positions, space info, etc.) persisted alongside the data.
        """
        self.frequency = Frequency(frequency)
        self.start = start
        self.aggregation = aggregation
        self.header = header
        exp_size = 0
        if duration is not None and duration < 0:
            raise ValueError(f"duration should be None or >=0, got {duration}")

        if data is None:
            if duration is None:
                raise ValueError("Missing data or duration")
            # post-poned initialization of data through __iadd__
            # initialize with data.size == 0
            if not frequency:
                data = np.zeros((0,))
            else:
                exp_size = max(1, self.frequency.to_ind(duration))
                data = np.zeros((0, exp_size))
        self.data = data
        if frequency and duration is not None:
            exp_size = 0 if not duration else max(1, self.frequency.to_ind(duration))
            if duration and not self.data.shape[-1]:
                msg = "Last dimension is empty but frequency and duration are not null "
                msg += f"(shape={self.data.shape})"
                raise ValueError(msg)
            if abs(data.shape[-1] - exp_size) > 1:
                msg = f"Data has incorrect (last) dimension {data.shape} for duration "
                msg += f"{duration} and frequency {frequency} (expected {exp_size})"
                raise ValueError(msg)
        if frequency:
            self.duration = self.frequency.to_sec(data.shape[-1])
        elif duration is None:
            raise ValueError(f"duration must be provided if {frequency=}")
        else:
            self.duration = duration
        # averaging
        self._overlapping_data_count: None | np.ndarray = None
        if aggregation == "mean":
            num = self.data.shape[-1] if self.frequency else 1
            self._overlapping_data_count = np.zeros(num, dtype=int)
        elif aggregation != "sum":
            raise ValueError(f"Unknown {aggregation=}")

    def __repr__(self) -> str:
        # Show shape/dtype only — stringifying data triggers numpy arrayprint,
        # which reads the underlying memmap and is catastrophic on network storage.
        cls = self.__class__.__name__
        fields = "frequency,start,duration,aggregation".split(",")
        string = ",".join(f"{f}={getattr(self, f)}" for f in fields)
        return f"{cls}({string},data=<{self.data.shape} {self.data.dtype}>)"

    def __iadd__(self, other: "TimedArray") -> "TimedArray":
        if other.frequency and self.frequency != other.frequency:
            diff = abs(self.frequency - other.frequency)
            if diff * max(self.duration, other.duration) >= 0.5:  # half sample diff
                msg = f"Cannot add with different (non-0) frequencies ({other.frequency} and {self.frequency})"
                raise ValueError(msg)
        if not self.data.size:
            # post-poned initialization of data, recover shape from other.data
            last = -1 if other.frequency else None
            shape = other.data.shape[:last]
            if self.frequency:
                shape += (self.data.shape[-1],)
            self.data = np.zeros(shape, dtype=other.data.dtype)
        if self.frequency:
            slices = [
                sa1._overlap_slice(sa2.start, sa2.duration)
                for sa1, sa2 in [(self, other), (other, self)]
            ]
            if slices[0] is None or slices[1] is None:
                return self  # no overlap
            # slices
            self_slice = slices[0][-1]
            other_slice = slices[1][-1]
        else:
            if self._overlap_slice(other.start, other.duration) is None:
                return self  # no overlap
            self_slice = None
            other_slice = None
        # materialize ContiguousMemmap (file-IO proxy) before arithmetic
        self.data = np.asarray(self.data)
        other_data = np.asarray(other.data[..., other_slice])
        if self._overlapping_data_count is None:  # sum
            self.data[..., self_slice] += other_data
        else:  # average
            counts = self._overlapping_data_count[..., self_slice]
            upd = counts / (1.0 + counts)
            self.data[..., self_slice] *= upd
            self.data[..., self_slice] += (1 - upd) * other_data
            counts += 1
        return self

    def _overlap_slice(
        self, start: float, duration: float
    ) -> tuple[float, float, slice | None] | None:
        if self.start == _UNSET_START or start == _UNSET_START:
            raise RuntimeError(
                "Cannot compute overlap on a TimedArray with unset start time. "
                "Call with_start() first."
            )
        if duration < 0:
            raise ValueError(f"duration should be >=0, got {duration=}")
        overlap_start = max(start, self.start)
        overlap_stop = min(start + duration, self.start + self.duration)
        if overlap_stop < overlap_start:
            return None  # no overlap
        if overlap_stop == overlap_start and self.duration and duration:
            return None  # 2 timed arrays with durations with one starting when the other ends
        if not self.frequency:
            return overlap_start, overlap_stop - overlap_start, None
        if not self.duration:
            return None  # frequency but no duration -> empty
        start_ind = self.frequency.to_ind(overlap_start - self.start)
        duration_ind = self.frequency.to_ind(overlap_stop - overlap_start)
        # # # right edge border case # # #
        if duration_ind <= 0:  # faster than max
            duration_ind = 1
        # then make sure we move the start according to the number of selected samples
        tps = self.data.shape[-1]
        if start_ind > tps - duration_ind:
            start_ind = tps - duration_ind
        if start_ind < 0:
            raise RuntimeError(f"Fail for {start=} {duration=} on {self}")
        start = self.frequency.to_sec(start_ind) + self.start
        duration = self.frequency.to_sec(duration_ind)
        # # # build # # #
        out = start, duration, slice(start_ind, start_ind + duration_ind)
        return out

    def with_start(self: _TA, start: float) -> _TA:
        """Return a lightweight copy sharing the data array with a new start time."""
        cls = type(self)
        return cls(
            data=self.data,
            frequency=self.frequency,
            start=start,
            header=self.header,
        )

    def overlap(self: _TA, start: float, duration: float) -> _TA:
        """Returns the sub TimedArray overlapping with the provided start
        and duration
        In case of lack of overlap, a timed array with 0 duration and empty
        data on the time dimension will be returned.
        """
        if not self.frequency:
            msg = "Cannot call overlap with no time dimension (TimedArray.frequency = 0)"
            raise RuntimeError(msg)
        out = self._overlap_slice(start, duration)
        if out is not None:
            ostart, oduration, sl = out
        else:
            ostart, oduration, sl = min(start, self.start), 0, slice(0, 0)
        cls = type(self)
        return cls(
            frequency=self.frequency,
            start=ostart,
            duration=oduration,
            data=self.data[..., sl],
            header=self.header,
        )

    # -- DumpContext serialization protocol --

    def __dump_info__(self, ctx: tp.Any) -> dict[str, tp.Any]:
        # Time-first on disk: time-slice reads become contiguous (~9s → sequential on NFS)
        data = np.moveaxis(self.data, -1, 0) if self.frequency else self.data
        info: dict[str, tp.Any] = {
            "data": ctx.dump(data),
            "frequency": float(self.frequency),
            "start": self.start,
            "duration": self.duration,
        }
        if self.header:
            info["header"] = ctx.dump(self.header)
        return info

    @classmethod
    def __load_from_info__(cls, ctx: tp.Any, **content: tp.Any) -> "TimedArray":
        # file-IO proxy: reads into freeable heap buffers instead of faulting memmap pages
        ctx.options.replace.setdefault("MemmapArray", "ContiguousMemmapArray")
        data = ctx.load(content.pop("data"))
        if content.get("frequency"):
            data = np.moveaxis(data, 0, -1)  # disk is time-first, memory is time-last
        if "header" in content:
            content["header"] = ctx.load(content["header"])
        return cls(data=data, **content)


yaml.representer.SafeRepresenter.add_representer(Frequency, Frequency._yaml_representer)


class Step(exca.steps.Step, _Module, discriminator_key="name"):
    """Base class for composable pipeline nodes.

    A ``Step`` represents a single operation in a data processing pipeline,
    such as loading a study (:class:`~neuralset.events.Study`) or transforming
    events (:class:`~neuralset.events.EventsTransform`). Steps can be executed
    individually via ``.run()`` or grouped into a :class:`Chain`.

    Inherits caching and execution infrastructure from ``exca.steps.Step``
    (configured via the ``infra`` parameter).

    Parameters
    ----------
    infra : exca.steps.Backend, optional
        Caching and execution infrastructure. Supports caching to disk
        (e.g. ``Cached`` backend) and executing remotely on a cluster
        (e.g. ``Slurm``). Pydantic will coerce a dict, e.g.
        ``{"backend": "Cached", "folder": "~/.cache/neuralset"}``.
    """

    # Cache format; read by exca (>=0.5.23) or written to infra below.
    _CACHE_TYPE: tp.ClassVar[str | None] = None

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        # exca >=0.5.23 cascades _CACHE_TYPE itself; older needs manual propagation.
        if hasattr(exca.steps.Step, "_CACHE_TYPE"):
            return
        if (
            self._CACHE_TYPE is not None
            and self.infra is not None
            and self.infra.cache_type is None
        ):
            self.infra.cache_type = self._CACHE_TYPE

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _discover_study(
        cls,
        value: tp.Any,
        handler: pydantic.ValidatorFunctionWrapHandler,
    ) -> "Step":
        """Trigger lazy study-package scanning for unknown discriminator names."""
        if isinstance(value, dict):
            name = value.get(cls._exca_discriminator_key)
            if name and name not in cls._get_discriminated_subclasses():
                from .events import study as _study

                _study._resolve_study(name)
        return handler(value)


class Chain(exca.steps.Chain, Step):
    """A sequence of processing steps executed in order.

    A ``Chain`` groups multiple ``Step`` objects (such as :class:`~neuralset.events.Study`
    and :class:`~neuralset.events.EventsTransform`) into a single cohesive pipeline.
    Because a ``Chain`` is itself a ``Step``, it can be nested inside other chains or
    used anywhere a ``Step`` is expected. When you call ``.run()``, it passes the
    output of each step as the input to the next.

    Parameters
    ----------
    steps : list of Step or dict of str to Step
        The ordered sequence of steps to execute. Pydantic will coerce each entry
        from a dict (strongly recommended over instantiated objects for simplicity).
    infra : exca.steps.Backend, optional
        Caching and execution infrastructure inherited from ``Step``.
        If provided, it determines how the final output of the chain is cached
        (e.g. using the ``Cached`` backend) or executed remotely (e.g. via ``Slurm``).

    Examples
    --------
    .. code-block:: python

        chain = ns.Chain(steps=[
            {"name": "MyStudy", "path": "/data",
             "infra": {"backend": "Cached", "folder": "/cache"}},
            {"name": "QueryEvents", "query": "timeline_index < 5"},
        ])
        events = chain.run()
    """

    steps: list[Step] | collections.OrderedDict[str, Step]  # type: ignore

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if hasattr(exca.steps.Step, "_CACHE_TYPE"):
            return  # cache propagation handled by exca itself (>=0.5.23)
        # Chain output matches last step: use its cache format (instance > class default).
        if self.infra is not None and self.infra.cache_type is None:
            seq = (
                list(self.steps.values()) if isinstance(self.steps, dict) else self.steps
            )
            if seq:
                last = seq[-1]
                if last.infra is not None and last.infra.cache_type is not None:
                    self.infra.cache_type = last.infra.cache_type
                else:
                    self.infra.cache_type = last._CACHE_TYPE
