# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Event handling classes and functions.

The class `Event` and its children (e.g. `Audio`, `Word`, etc.) define the
expected fields for each event type.
"""

import inspect
import itertools
import logging
import typing as tp
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic

from neuralset import utils
from neuralset.base import FmriSpec, Frequency, StrCast, _Module

E = tp.TypeVar("E", bound="Event")
logger = logging.getLogger(__name__)


class Event(_Module):
    """Base class for all event types.

    Every event has a time span (``start``, ``duration``) and belongs to a
    timeline. Concrete subclasses (``Meg``, ``Image``, ``Word``,
    ``Fmri``, …) add type-specific fields. Events are collected into a
    pandas DataFrame for bulk processing; the ``stop`` column
    (``start + duration``) is added by :func:`standardize_events`.

    When instantiated via :meth:`from_dict`, extra fields not in the
    schema are silently grouped under ``extra`` rather than raising.

    Parameters
    ----------
    start : float
        Start time of the event in seconds.
    timeline : str
        Timeline identifier to which this event belongs.
    duration : float, optional
        Duration of the event in seconds (default: 0.0).
    extra : dict, optional
        Dictionary for storing additional arbitrary fields (default: {}).

    .. admonition:: Design rationale — ``start``/``duration``, not ``onset``/``offset``

       Events use ``start``/``duration`` rather than ``onset``/``offset``
       because ``offset`` is ambiguous — it can mean "end time" or
       "time shift from a reference" (e.g. ``Video.offset``, fMRI HRF
       delays, MEG ``first_samp``). ``duration`` is unambiguous and is
       used directly to compute sample counts, rather than
       ``to_ind(stop) - to_ind(start)`` which can vary with alignment.
    """

    start: float
    timeline: str
    duration: pydantic.NonNegativeFloat = 0.0
    extra: dict[str, tp.Any] = {}
    type: tp.ClassVar[str] = "Event"
    _CLASSES: tp.ClassVar[dict[str, tp.Type["Event"]]] = {}
    _index: int | None = None  # records index in dataframe for debugging

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        # register all events
        cls.type = cls.__name__
        Event._CLASSES[cls.__name__] = cls

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if pd.isna(self.start):
            raise ValueError(f"Start time needs to be provided for {self!r}")

    @classmethod
    def from_dict(cls: tp.Type[E], row: tp.Any) -> E:
        """Create event from dictionary/row/named-tuple while grouping non-specified fields.

        Fields not defined in the event class schema are automatically grouped under the
        :code:`extra` dict attribute.

        Parameters
        ----------
        row : dict, NamedTuple, or pandas.Series
            Input data containing event fields. Must include a 'type' field indicating
            the event class name.

        Returns
        -------
        Event
            Instance of the appropriate Event subclass based on the 'type' field

        Notes
        -----
        - NaN values in the input are ignored
        - if the input has an Index attribute, it's stored in :code:`_index` for debugging

        Examples
        --------
        .. code-block:: python

            data = {'type': 'Word', 'start': 1.0, 'timeline': 'exp1', 'text': 'hello'}
            event = Event.from_dict(data)  # returns a Word instance
        """
        index: int | None = None
        if hasattr(row, "_asdict"):
            index = getattr(row, "Index", None)
            row = row._asdict()  # supports named tuples
        cls_ = cls._CLASSES[row["type"]]
        if not issubclass(cls_, cls):
            raise TypeError(f"{cls_} is not a subclass of {cls}")
        fs = set(cls_.model_fields)  # type: ignore
        kwargs: dict[str, tp.Any] = {}
        extra = {}
        for k, v in row.items():
            if pd.isna(v):
                continue
            if k in fs:
                kwargs[k] = v
            elif k != "type":
                extra[k] = v
        kwargs.setdefault("extra", {}).update(extra)  # can bug if extra is a column
        try:
            out = cls_(**kwargs)
        except Exception as e:
            logger.warning(
                "Event.from_dict parsing failed for input %s\nmapped to %s\n with error: %s)",
                row.to_string() if hasattr(row, "to_string") else row,
                kwargs,
                e,
            )
            raise
        out._index = index
        return out

    def to_dict(self) -> dict[str, tp.Any]:
        """Export the event as a dictionary usable for CSV dump.

        The :code:`extra` dictionary field is flattened into separate fields to simplify
        queries through pandas DataFrames.

        Returns
        -------
        dict
            Flattened dictionary representation of the event with 'type' field included

        Examples
        --------
        .. code-block:: python

            event = Word(start=1.0, timeline="exp", text="hello")
            event_dict = event.to_dict()
            df = pd.DataFrame([event_dict])
        """
        out = dict(self.extra)
        out["type"] = self.type
        # avoid Path in exports
        tag = "extra"
        fields = {x: str(y) if isinstance(y, Path) else y for x, y in self if x != tag}
        out.update(fields)
        return out

    @property
    def stop(self) -> float:
        """Compute the end time of the event.

        Returns
        -------
        float
            The stop time calculated as start + duration
        """
        return self.start + self.duration

    def _get_field_or_extra(self, field: str) -> tp.Any:
        """Get a value from model fields or :attr:`extra`."""
        if field in type(self).model_fields:
            return getattr(self, field)
        if field in self.extra:
            return self.extra[field]
        raise ValueError(f"Field {field!r} not found in event: {self!r}")

    def __str__(self) -> str:
        core_fields = {k: v for k, v in self if k != "extra"}
        return ", ".join([f"{k}={v}" for k, v in core_fields.items()])


Event._CLASSES["Event"] = Event


class EventTypesHelper:
    """Computes and stores information about event types.

    Provides a unified interface for accessing event type information whether the input
    is an actual type, a type name, or a tuple of type names. Particularly useful for
    filtering dataframes by event types including their subclasses.

    Parameters
    ----------
    event_types : type, str, or tuple of str
        Event type specification - can be a class, a string name, or tuple of names

    Attributes
    ----------
    classes : tuple of type
        The Event classes specified (always as a tuple, even if only 1 type)
    names : list of str
        List of all event type names including subclasses, useful for filtering:
        :code:`events[events.type.isin(helper.names)]`
    specified : type, str, or tuple of str
        The original input specification

    Raises
    ------
    ValueError
        If an invalid event name is provided (not in Event._CLASSES)

    Examples
    --------
    .. code-block:: python

        helper = EventTypesHelper("Word")
        filtered_events = events[events.type.isin(helper.names)]

        # Multiple types
        helper = EventTypesHelper(("Word", "Sentence"))
        print(helper.classes)  # (Word, Sentence)
    """

    def __init__(self, event_types: str | tp.Type[Event] | tp.Sequence[str]) -> None:
        self.specified = event_types
        if inspect.isclass(event_types):
            self.classes: tuple[tp.Type[Event], ...] = (event_types,)
        else:
            if isinstance(event_types, str):
                event_types = (event_types,)
            try:
                self.classes = tuple(Event._CLASSES[x] for x in event_types)  # type: ignore
            except KeyError as e:
                avail = list(Event._CLASSES)
                msg = f"{event_types} is an invalid event name, use one of {avail}"
                raise ValueError(msg) from e
        items = Event._CLASSES.items()
        self.names = [x for x, y in items if issubclass(y, self.classes)]

    def extract(self, obj: tp.Any) -> list[Event]:
        from .utils import extract_events  # avoid circular imports

        return extract_events(obj, self)


class BaseDataEvent(Event):
    """Base class for events whose data needs to be read from a file.

    This class handles file path validation and provides an abstract interface
    for reading data from files. Supports both regular file paths and method URIs
    for dynamic data loading.

    Parameters
    ----------
    filepath : Path or str
        Path to the data file or a method URI (format: :code:`method:<name>?timeline=<timeline>`)
    frequency : float, optional
        Sampling frequency in Hz (default: 0, auto-detected when possible)

    Raises
    ------
    ValueError
        If filepath is empty or if method URI is missing the timeline parameter

    Notes
    -----
    - File existence is checked unless the path contains a colon (e.g., method URIs)
    - Method URIs allow loading data through registered timeline methods
    - Subclasses must implement :code:`_read()` for actual file reading logic

    Examples
    --------
    .. code-block:: python

        # Regular file path
        event = Audio(start=0, timeline="t1", filepath="/path/to/audio.wav")

        # Method URI
        event = Audio(start=0, timeline="t1",
                     filepath="method:load_audio?timeline=experiment")
    """

    filepath: Path | str = ""
    frequency: float = 0
    _special_loader: tp.Any = None

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.filepath:
            raise ValueError("A filepath must be provided")
        self._resolve_special_loader()
        self._check_filepath()

    def _resolve_special_loader(self) -> None:
        """Resolve SpecialLoader eagerly so the Study reference survives pickling.

        See "Pickle chain for subprocess registry population" in studies.md.
        """
        if self._special_loader is not None:
            return
        fp = str(self.filepath)
        if fp.startswith("{") and "cls" in fp:
            from . import study

            try:
                self._special_loader = study.SpecialLoader.from_json(fp)
            except KeyError:
                pass  # JSON filepath but not a SpecialLoader (e.g. finder pattern)

    def _check_filepath(self, error: bool = False) -> None:
        fp = str(self.filepath)
        self.filepath = fp
        if not any(c in str(fp) for c in ":{*"):  # deactivate check
            # make sure to store file as a string in dataframe
            if not Path(fp).exists():
                cls = self.__class__.__name__
                msg = f"File missing for {cls} event: {fp}"
                if error:
                    raise ValueError(msg)
                utils.warn_once(msg)

    def read(self) -> tp.Any:
        """Read and return the data from the file or method URI.

        Returns
        -------
        Any
            The loaded data (type depends on the specific Event subclass)

        Examples
        --------
        .. code-block:: python

            audio_event = Audio(start=0, timeline="t1", filepath="audio.wav")
            audio_tensor = audio_event.read()  # Returns torch.Tensor
        """
        self._resolve_special_loader()
        if self._special_loader is not None:
            return self._special_loader.load()
        return self._read()

    @abstractmethod
    def _read(self) -> tp.Any:
        """Implement file reading logic in subclasses.

        Returns
        -------
        Any
            The loaded data
        """
        return

    def _missing_duration_or_frequency(self) -> bool:
        return any(not x or pd.isna(x) for x in [self.duration, self.frequency])

    def study_relative_path(self) -> Path:
        """Returns the filepath if 'study' is not in event.extra
        else returns the part of the filepath after the study directory if it is
        identified in the path
        """
        study: str | None = self.extra.get("study", None)
        if study is None or ":" in str(self.filepath):
            return Path(self.filepath)
        study = study.lower()
        parts = Path(self.filepath).parts
        remaining = list(itertools.dropwhile(lambda x: x.lower() != study, parts))
        if not remaining:
            return Path(self.filepath)
        return Path(*remaining)


class BaseSplittableEvent(BaseDataEvent):
    """Base class for dynamic events (audio and video) which can be read in parts.

    Supports reading only a specific section [offset, offset + duration] of the file,
    enabling efficient partial loading of large media files.

    Parameters
    ----------
    offset : float, optional
        Start position within the file in seconds (relative to file start, not timeline)
        Default: 0.0

    Notes
    -----
    - offset is relative to the event file, not the absolute timeline
    - The :code:`_split()` method can divide an event into multiple sub-events

    Examples
    --------
    .. code-block:: python

        # Load only seconds 5-10 of an audio file
        audio = Audio(start=100, timeline="t1", filepath="long_audio.wav",
                     offset=5.0, duration=5.0)
        audio_data = audio.read()  # Only loads 5 seconds
    """

    offset: pydantic.NonNegativeFloat = 0.0

    def _splittable_event_uid(self) -> str:
        """Cache UID incorporating file path, offset and duration.

        Used as ``item_uid`` by extractors that process splittable events
        (audio, video, MNE, fMRI). The format ensures that chunked and
        non-chunked events for the same file get distinct cache entries.

        Precision is millisecond (``:.3f``), which is sufficient
        for all current chunking granularities.
        """
        return f"{self.study_relative_path()}_{self.offset:.3f}_{self.duration:.3f}"

    def _split(
        self, timepoints: list[float], min_duration: float | None = None
    ) -> tp.Sequence["BaseSplittableEvent"]:
        """Split the event at specified timepoints into multiple sub-events.

        Given n ordered timepoints, returns n + 1 corresponding events for each section.
        Timepoints are relative to the event start, not the absolute timeline time.

        Parameters
        ----------
        timepoints : list of float
            Ordered list of split points in seconds (relative to event start)
        min_duration : float, optional
            Minimum duration in seconds for resulting segments. Timepoints that would
            create segments shorter than this are filtered out.

        Returns
        -------
        list of BaseSplittableEvent
            List of new event instances representing each segment

        Raises
        ------
        ValueError
            If timepoints are not strictly increasing

        Examples
        --------
        .. code-block:: python

            audio = Audio(start=0, timeline="t1", filepath="audio.wav", duration=10.0)
            # Split at 3s and 7s with minimum 2s segments
            segments = audio._split([3.0, 7.0], min_duration=2.0)
            # Returns 3 sounds: [0-3s], [3-7s], [7-10s]
        """
        # keep only timepoints that are within the sound duration
        timepoints = [t for t in timepoints if 0 < t < self.duration]
        timepoints = sorted(set(timepoints))
        if min_duration:
            round_decimals = max(0, int(-np.floor(np.log10(1 / self.frequency))) + 1)
            delta_before = np.round(np.diff(timepoints, prepend=0), round_decimals)
            delta_after = np.round(
                np.diff(timepoints, append=self.duration), round_decimals
            )
            timepoints = [
                t
                for t, db, da in zip(timepoints, delta_before, delta_after)
                if db >= min_duration and da >= min_duration
            ]
        timepoints.append(self.duration)

        start = 0.0
        data = dict(self)
        cls = self.__class__
        events = []
        for stop in list(timepoints):
            if start >= stop:
                raise ValueError(
                    f"Timepoints should be strictly increasing (got {start} and {stop})"
                )
            data.update(
                start=self.start + start,
                duration=stop - start,
                offset=self.offset + start,
            )
            events.append(cls(**data))
            start = stop
        return events


class Image(BaseDataEvent):
    """Image event with optional caption.

    Requires :code:`pillow>=9.2.0` to be installed.

    Parameters
    ----------
    caption : str, optional
        Caption or description of the image. Multiple captions should be
        newline-separated. Default: ""

    Returns
    -------
    PIL.Image
        RGB PIL Image object when :code:`read()` is called

    Warnings
    --------
    Image events with duration <= 0 will be logged as ignored

    Examples
    --------
    .. code-block:: python

        img_event = Image(start=1.0, timeline="exp", filepath="img.jpg",
                         duration=2.0, caption="A cat")
        pil_image = img_event.read()
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("pillow>=9.2.0",)
    # Multiple captions for the same image should be '\n'-separated
    caption: str = ""

    def _read(self) -> tp.Any:
        # pylint: disable=import-outside-toplevel
        import PIL.Image  # noqa

        return PIL.Image.open(self.filepath).convert("RGB")

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.duration <= 0:
            logger.info("Image event has null duration and will be ignored.")


class Audio(BaseSplittableEvent):
    """Audio event corresponding to a WAV file.

    Requires :code:`soundfile` to be installed. Duration and frequency are auto-detected
    from the file if not provided.

    Returns
    -------
    torch.Tensor
        Audio tensor of shape (num_samples, num_channels) when :code:`read()` is called

    Notes
    -----
    - Automatically detects frequency and duration from file metadata
    - Supports partial loading via offset and duration
    - Mono audio is automatically reshaped to (N, 1)

    Examples
    --------
    .. code-block:: python

        audio = Audio(start=0, timeline="audio_exp", filepath="speech.wav")
        audio_tensor = audio.read()  # Returns torch.Tensor
        print(audio_tensor.shape)  # (num_samples, num_channels)
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("soundfile",)
    # soundfile (or PySoundFile?) or sox or sox_io may be needed as a backend
    # https://pytorch.org/audio/0.7.0/backend.html

    def model_post_init(self, log__: tp.Any) -> None:
        if self._missing_duration_or_frequency():
            import soundfile

            self._check_filepath(error=True)
            info = soundfile.info(str(self.filepath))
            self.frequency = Frequency(info.samplerate)
            self.duration = info.duration
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        import soundfile
        import torch

        self._check_filepath(error=True)
        sr = Frequency(self.frequency)
        offset = sr.to_ind(self.offset)
        num = sr.to_ind(self.duration)
        fp = str(self.filepath)
        wav = soundfile.read(fp, start=offset, frames=num)[0]
        out = torch.Tensor(wav)
        if out.ndim == 1:
            out = out[:, None]
        return out  # (Nsamples, Nchannels) as noted in scipy.io.wavfile.write


class Video(BaseSplittableEvent):
    """Video event with support for partial loading.

    Requires :code:`moviepy>=2.1.2` to be installed. Duration and FPS are auto-detected
    from the file if not provided.

    Notes
    -----
    - Automatically detects FPS (as frequency) and duration from file
    - Supports partial loading via offset and duration
    - Returns a moviepy VideoFileClip that can be further processed

    Examples
    --------
    .. code-block:: python

        video = Video(start=0, timeline="video_exp", filepath="video.mp4",
                     offset=5.0, duration=10.0)
        clip = video.read()  # Returns 10-second clip starting at 5s
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("moviepy>=2.1.2",)

    def model_post_init(self, log__: tp.Any) -> None:
        if self._missing_duration_or_frequency():
            from moviepy import VideoFileClip  # noqa

            self._check_filepath(error=True)
            video = VideoFileClip(str(self.filepath))
            self.frequency = Frequency(video.fps)
            self.duration = video.duration
            video.close()
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        from moviepy import VideoFileClip  # noqa

        self._check_filepath(error=True)
        with utils.ignore_all():
            clip = VideoFileClip(str(self.filepath))
            start, end = self.offset, self.offset + self.duration
            if end > clip.duration:
                raise ValueError(
                    f"Requested end {end} exceeds video duration {clip.duration}"
                )
            clip = clip.subclipped(start, end)
        return clip


class BaseText(Event):
    """Base class for text-based events.

    Parameters
    ----------
    language : str, optional
        ISO language code (e.g., 'en', 'fr'). Default: ""
    text : str
        The text content (must be non-empty)
    context : str, optional
        Contextual information (e.g., preceding words). Default: ""
    modality : Literal["heard", "read", "imagined", "typed", "written", "caption"], optional
        Modality of the text event. Default: None

    Raises
    ------
    ValidationError
        If text is empty (min_length=1)
    """

    text: str = pydantic.Field(..., min_length=1)
    language: str = ""
    context: str = ""
    modality: (
        tp.Literal["heard", "read", "imagined", "typed", "written", "caption"] | None
    ) = None


class Text(BaseText):
    """General text event, possibly containing multiple sentences."""


class Sentence(BaseText):
    """Single sentence text event with optional context information."""


class Word(BaseText):
    """Single word event with optional sentence information.

    Parameters
    ----------
    sentence : str, optional
        Full sentence containing this word. Default: ""
    sentence_char : int, optional
        Character position of the word in the sentence. Default: None

    """

    sentence: str = ""
    sentence_char: int | None = None


class Phoneme(BaseText):
    """Single phoneme event."""

    sentence: str = ""


class Keystroke(BaseText):
    """Keystroke event with associated text."""

    sentence: str = ""


class CategoricalEvent(Event):
    """Base class for categorical events.

    Categorical events represent discrete labels or categories (e.g., stimulus
    codes, sleep stages, artifact types) rather than pointing to data files.
    """


class Action(CategoricalEvent):
    """Event indicating the participant performed (or imagined) an action.

    This includes motor executions (e.g., finger tapping, arm movement) and imaginations. Unlike
    stimulus events which are presented to the participant, actions are performed by the
    participant.

    Parameters
    ----------
    code : int, optional
        Action code or trigger value. Default: -100 (ignored by CrossEntropyLoss)
    description : str, optional
        Human-readable description of the action. Default: ""
    """

    code: int = -100
    description: str = ""


class Stimulus(CategoricalEvent):
    """General stimulus presentation event identified by a code.

    Unlike specific stimulus events (:code:`Image`, :code:`Audio`) which point to actual
    stimulus files, this event only registers a code/trigger value mapped to an event.

    Parameters
    ----------
    code : int, optional
        Stimulus code or trigger value. Default: -100 (ignored by CrossEntropyLoss)
    description : str, optional
        Human-readable description of the stimulus. Default: ""

    See Also
    --------
    neuralset.events.testing.mne2013sample : Example usage

    Examples
    --------
    .. code-block:: python

        stim = Stimulus(start=1.0, timeline="exp", code=1,
                       description="Visual checkerboard")
    """

    code: int = -100  # Default value ignored by CrossEntropyLoss (`ignore_index`)
    modality: tp.Literal["audio", "visual", "tactile"] | None = None
    description: str = ""


class EyeState(CategoricalEvent):
    """Eye state event indicating whether eyes are open or closed.

    Parameters
    ----------
    state : {'closed', 'open'}
        Current eye state

    See Also
    --------
    Babayan2019 : Example usage
    """

    state: tp.Literal["closed", "open"]


class Artifact(CategoricalEvent):
    """Artifact or noise event in neural recordings.

    Parameters
    ----------
    state : {'eyem', 'musc', 'chew', 'shiv', 'elpp', 'artf'}
        Type of artifact:

        - 'eyem': Eye movement
        - 'musc': Muscle artifact
        - 'chew': Chewing
        - 'shiv': Shivering
        - 'elpp': Electrode artifact (pop, static, lead artifacts)
        - 'artf': Catch-all for other artifacts

    See Also
    --------
    Hamid2020Tuar : Example usage
    """

    state: tp.Literal["eyem", "musc", "chew", "shiv", "elpp", "artf"]


class Seizure(CategoricalEvent):
    """Seizure event with specific seizure type classification.

    Parameters
    ----------
    state : str
        Seizure type:

        - 'bckg': Background (no seizure)
        - 'seiz': General seizure
        - 'gnsz': Generalized periodic epileptiform discharges
        - 'fnsz': Focal non-specific seizure
        - 'spsz': Simple partial seizure
        - 'cpsz': Complex partial seizure
        - 'absz': Absence seizure
        - 'tnsz': Tonic seizure
        - 'cnsz': Clonic seizure
        - 'tcsz': Tonic-clonic seizure
        - 'atsz': Atonic seizure
        - 'mysz': Myoclonic seizure

    See Also
    --------
    Harati2015Tuev : Example usage
    """

    state: tp.Literal[
        "bckg",  # Background
        "seiz",  # Seizure
        "gnsz",  # Generalized periodic epileptiform discharges
        "fnsz",  # Focal non-specific seizure
        "spsz",  # Simple partial seizure
        "cpsz",  # Complex partial seizure
        "absz",  # Absence seizure
        "tnsz",  # Tonic seizure
        "cnsz",  # Clonic seizure
        "tcsz",  # Tonic clonic seizure
        "atsz",  # Atonic seizure
        "mysz",  # Myoclonic Seizure
        "nesz",  # Non-Epileptic Seizure
    ]


class Background(CategoricalEvent):
    """'Background' activity, i.e. no specific epilepsy or arousal event happening during this time."""


class SleepStage(CategoricalEvent):
    """Sleep stage event following AASM manual classification [sleep1]_.

    Parameters
    ----------
    stage : {'W', 'N1', 'N2', 'N3', 'R'}
        Sleep stage:

        - 'W': Waking state
        - 'N1': Stage 1 of Non-Rapid Eye Movement (N-REM)
        - 'N2': Stage 2 of Non-Rapid Eye Movement (N-REM)
        - 'N3': Stage 3/4 of Non-Rapid Eye Movement (N-REM)
        - 'R': Rapid Eye Movement (REM)

    References
    ----------
    .. [sleep1] Berry, R., Quan, S. and Abreu, A. (2020) The AASM Manual for the
                Scoring of Sleep and Associated Events: Rules, Terminology and
                Technical Specifications, Version 2.6. American Academy of Sleep
                Medicine, Darien.
    """

    stage: tp.Literal[
        "W",  # Waking state
        "N1",  # Stage 1 of Non-Rapid Eye Movement (N-REM)
        "N2",  # Stage 2 of Non-Rapid Eye Movement (N-REM)
        "N3",  # Stage 3/4 of Non-Rapid Eye Movement (N-REM)
        "R",  # Rapid Eye Movement (REM)
    ]


class MneRaw(BaseSplittableEvent):
    """Brain recording saved as MNE Raw object.

    Base class for neurophysiological recordings (MEG, EEG, etc.) that can be
    loaded using MNE-Python.

    Parameters
    ----------
    subject : str
        Subject identifier (required, cannot be empty)

    Notes
    -----
    - Automatically detects frequency and duration from file metadata
    - If ``start`` is ``"auto"``, auto-resolves from ``raw.first_samp / sfreq``
      (use for FIF files with nonzero first_samp)
    - Guards against ``start=0`` when ``raw.first_samp > 0``
    - Subject ID is cast to string to handle numeric IDs from dataframes

    Examples
    --------
    .. code-block:: python

        meg = Meg(start="auto", timeline="scan1",
                 filepath="data_raw.fif", subject="sub-01")
        raw = meg.read()  # Returns mne.io.Raw object
    """

    subject: StrCast = ""

    @pydantic.field_validator("start", mode="before")
    @classmethod
    def _auto_start(cls, v: tp.Any) -> float:
        # Convert "auto" to nan; model_post_init resolves it from first_samp
        return float("nan") if v == "auto" else v

    def model_post_init(self, log__: tp.Any) -> None:
        if self.offset:
            raise ValueError(
                "offset is not supported for MneRaw events (chunking not yet implemented)"
            )
        self.subject = self.subject
        if self._missing_duration_or_frequency() or pd.isna(self.start):
            raw = self.read()
            self.frequency = Frequency(raw.info["sfreq"])

            if hasattr(raw, "duration"):
                self.duration = raw.duration
            else:
                msg = "Raw MEG has no duration, using raw.times to compute duration."
                logger.warning(msg)
                self.duration = raw.times[-1] - raw.times[0] + (1.0 / self.frequency)
            # nan → auto-resolve from first_samp
            if pd.isna(self.start):
                self.start = raw.first_samp / raw.info["sfreq"]
            elif raw.first_samp > 0 and not self.start:
                raise ValueError(
                    f"start=0 but raw.first_samp={raw.first_samp} > 0 "
                    f"for timeline {self.timeline}. Use start='auto' to auto-resolve."
                )

        if not self.subject:
            raise ValueError("Missing 'subject' field")
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        import mne

        kwargs: dict[str, tp.Any] = {}
        suffixes = Path(self.filepath).suffixes
        if ".fif" in suffixes:
            # sets raw.info["maxshield"]; checked by extractors.MneRaw._preprocess_raw
            kwargs["allow_maxshield"] = True
        if ".ds" in suffixes:
            # strip CTF system-specific serial-number suffixes (e.g. MLT31-4408 → MLT31)
            kwargs["clean_names"] = True
        with utils.ignore_all():
            return mne.io.read_raw(self.filepath, **kwargs)


class Meg(MneRaw):
    """Magnetoencephalography (MEG) recording event."""


class Eeg(MneRaw):
    """Electroencephalography (EEG) recording event."""


class Emg(MneRaw):
    """Electromyography (EMG) recording event."""


class Ieeg(MneRaw):
    """Intracranial EEG (iEEG) event.

    Encompasses stereoelectroencephalography (sEEG) and electrocorticography (ECoG).
    """


class Spikes(BaseDataEvent):
    """Spikes recording event saved as HDF5 object.

    Base class for spikes recordings that can be loaded using HDF5.

    Parameters
    ----------
    subject : str
        Subject identifier (required, cannot be empty)
    filepath: str
        Path to the HDF5 file (required, cannot be empty)

    Notes
    -----
    - Subject ID is cast to string to handle numeric IDs from dataframes

    Examples
    --------
    .. code-block:: python

        spikes = Spikes(start=0, timeline="scan1", filepath="data_raw.nwb",
                 subject="sub-01")
        h5py_file = spikes.read()  # Returns h5py.File object
    """

    subject: StrCast = ""

    def model_post_init(self, log__: tp.Any) -> None:
        self.subject = self.subject
        if self._missing_duration_or_frequency():
            msg = "Spikes requires `frequency` and `duration` to be non-zero (Hz) "
            msg += f"to support time-to-sample cropping. Got {self.frequency} and {self.duration}."
            raise ValueError(msg)

        if not self.subject:
            raise ValueError("Missing 'subject' field")
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        import h5py  # type: ignore[import-untyped]

        with utils.ignore_all():
            return h5py.File(self.filepath, "r")


class Fnirs(MneRaw):
    """Functional Near-Infrared Spectroscopy (fNIRS) recording event.

    Supports multiple file formats through MNE-Python:

    - .snirf: Shared Near-Infrared Format
    - .hdr: NIRX format
    - .csv: Hitachi format
    - .txt: Boxy format

    """

    def _read(self) -> tp.Any:
        import mne

        ext = Path(self.filepath).suffix
        with utils.ignore_all():
            return {
                ".snirf": mne.io.read_raw_snirf,
                ".hdr": mne.io.read_raw_nirx,
                ".csv": mne.io.read_raw_hitachi,
                ".txt": mne.io.read_raw_boxy,
            }[ext](self.filepath)


class Fmri(BaseSplittableEvent):
    """Functional MRI (fMRI) recording event.

    Handles both volumetric NIfTI files and FreeSurfer surface data
    (left + right hemisphere).  For surface data, ``filepath`` must be
    the **left hemisphere** (containing ``hemi-L``); the right hemisphere
    is derived automatically by replacing ``hemi-L`` with ``hemi-R``.

    Parameters
    ----------
    subject : str
        Subject identifier (required).
    space : str
        Coordinate space, e.g. ``"MNI152NLin2009cAsym"``, ``"T1w"``,
        ``"fsaverage"``, ``"custom"``.
    preproc : str
        Preprocessing pipeline: ``"fmriprep"``, ``"deepprep"``, ``"custom"``.
    mask_filepath : str or None
        Path to a brain mask NIfTI file (volumetric only).
    spec : dict[str, str] or None
        Variant parameters for DataFrame filtering (e.g.
        ``{"registration": "msmall", "resolution": "1.6mm"}``).
        Auto-encoded to sorted ``"key=value&..."`` strings.
    frequency : float
        Sampling frequency in Hz (required).
    """

    subject: StrCast
    space: str = "custom"
    preproc: str = "custom"
    mask_filepath: tp.Optional[str] = None
    spec: FmriSpec | None = None

    def model_post_init(self, log__: tp.Any) -> None:
        if self.offset:
            raise ValueError(
                "offset is not supported for Fmri events (chunking not yet implemented)"
            )
        if not self.frequency or pd.isna(self.frequency):
            raise ValueError(
                "Frequency must be provided for Fmri event: "
                "Don't rely on get_zooms as the header is sometimes unreliable.\n"
                f"Got: {self}"
            )
        if not self.duration or pd.isna(self.duration):
            self.duration = self.read().shape[-1] / self.frequency
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        import nibabel

        fp = str(self.filepath)
        if "hemi-L" in fp:
            right_fp = fp.replace("hemi-L", "hemi-R")
            if not Path(right_fp).exists():
                raise FileNotFoundError(
                    f"Right-hemisphere file derived from filepath not found: {right_fp}"
                )
            hemi_nps = []
            for p in (fp, right_fp):
                nii = nibabel.load(p, mmap=True)
                if hasattr(nii, "darrays"):
                    hemi_nps.append(np.stack([da.data for da in nii.darrays], axis=-1))
                else:
                    hemi_nps.append(nii.get_fdata().squeeze())  # type: ignore[attr-defined]
            return nibabel.Nifti2Image(np.concatenate(hemi_nps, axis=0), affine=np.eye(4))
        nii_img = nibabel.load(fp, mmap=True)
        if fp.endswith(".dtseries.nii"):
            # CIFTI: native shape is (time, nodes) -> transpose to (nodes, time)
            data = nii_img.get_fdata().transpose((1, 0))  # type: ignore[attr-defined]
            return nibabel.Nifti2Image(data, np.eye(4))
        if nii_img.ndim not in (4, 2):  # type: ignore
            raise ValueError(f"{fp} should be 2D or 4D with time the last dim.")
        return nii_img

    def read_mask(self) -> tp.Any:
        """Read brain mask. Returns None if no mask is available."""
        import nibabel

        if self.mask_filepath is None:
            return None
        return nibabel.load(self.mask_filepath, mmap=True)
