# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import typing as tp
from collections import defaultdict
from itertools import compress

import mne
import numpy as np
import pandas as pd
import pydantic
import sklearn.preprocessing
import torch
from exca import MapInfra
from exca.cachedict import DumpContext
from exca.helpers import DiscriminatedModel
from mne._fiff.pick import _VALID_CHANNEL_TYPES  # type: ignore
from tqdm import tqdm

import neuralset as ns
from neuralset import utils
from neuralset.base import TimedArray
from neuralset.events import etypes

from .base import BaseExtractor, BaseStatic

logger = logging.getLogger(__name__)
DataframeOrEventsOrSegments = (
    pd.DataFrame | tp.Sequence[etypes.Event] | tp.Sequence[ns.segments.Segment]
)

FSAVERAGE_SIZES = {
    "fsaverage3": 642,
    "fsaverage4": 2562,
    "fsaverage5": 10242,
    "fsaverage6": 40962,
    "fsaverage7": 163842,
}


def _overlap(
    start1: float,
    duration1: float,
    start2: float,
    duration2: float,
) -> tuple[float, float]:
    """
    Computes the overlap times between two windows
    """
    starts = (start1, start2)
    stops = tuple(s + d for s, d in zip(starts, (duration1, duration2)))
    start = max(starts)
    stop = min(stops)
    return start, max(0, stop - start)


@DumpContext.register
class MneTimedArray(TimedArray):
    """TimedArray subclass for MNE data with convenience accessors for channel info.

    Replaces MneRawFif: avoids 1-3s FIF open overhead per recording (seek/decompress).
    """

    header: dict[str, tp.Any]  # always set by from_native / __load_from_info__

    @classmethod
    def from_native(cls, raw: mne.io.Raw, start: float | None = None) -> "MneTimedArray":
        """Build from an mne.io.Raw, extracting channel header.

        Parameters
        ----------
        raw: mne.io.Raw
        start: optional float
            Timeline start. Defaults to ``raw.first_samp / sfreq``.
        """
        data = raw.get_data().astype(np.float32)
        if any("," in name for name in raw.ch_names):
            raise ValueError("Channel names must not contain commas")
        ch_locs = np.array([ch["loc"] for ch in raw.info["chs"]])
        header: dict[str, tp.Any] = {
            # comma-sep: 0.18ms init vs 0.73ms for JSON list
            "ch_names": ",".join(raw.ch_names),
            "ch_types": ",".join(raw.get_channel_types()),
            "ch_locs": ch_locs,
            "highpass": raw.info["highpass"],
            "lowpass": raw.info["lowpass"],
        }
        if start is None:
            start = raw.first_samp / raw.info["sfreq"]
        return cls(
            data=data,
            frequency=raw.info["sfreq"],
            start=start,
            header=header,
        )

    @property
    def ch_names(self) -> list[str]:
        # 0.31ms vs 5.41ms via to_info().ch_names (mne.create_info is ~3.3ms/call)
        return self.header["ch_names"].split(",")

    @property
    def ch_types(self) -> list[str]:
        return self.header["ch_types"].split(",")

    def to_info(self) -> mne.Info:
        """Reconstruct an ``mne.Info`` from the stored header.

        Includes ch_names, ch_types, sfreq, channel locations, highpass/lowpass.
        Rich fields (bads, projs, filter history, …) are not preserved.
        """
        info = mne.create_info(
            self.ch_names, sfreq=float(self.frequency), ch_types=self.ch_types
        )
        ch_locs: np.ndarray = self.header["ch_locs"]
        for i, ch in enumerate(info["chs"]):
            ch["loc"][:] = ch_locs[i]
        with info._unlock():
            info["highpass"] = self.header["highpass"]
            info["lowpass"] = self.header["lowpass"]
        return info

    def to_native(self) -> mne.io.RawArray:
        """Reconstruct a minimal ``mne.io.RawArray`` from the stored data and header."""
        first_samp = int(round(self.start * float(self.frequency)))
        return mne.io.RawArray(
            self.data, self.to_info(), first_samp=first_samp, verbose=False
        )


@DumpContext.register
class FmriTimedArray(TimedArray):
    """TimedArray subclass for fMRI with spatial metadata for plotting.

    Header always contains ``space`` (output coordinate system, e.g.
    ``"fsaverage5"`` after surface projection, or the event's original
    space when unprojected) and ``preproc``.

    Optional extra key: ``affine`` (4×4 ndarray) — present only for
    unprojected volumetric data, enables ``to_native()``.
    """

    header: dict[str, tp.Any]  # narrow down typing (not optional)

    def to_native(self, time_index: int | None = None) -> tp.Any:
        """Reconstruct a NIfTI image. Only valid for unprojected volumetric data."""
        affine = self.header.get("affine")
        if affine is None:
            raise ValueError(
                f"No affine in header={self.header}"
                " — data was projected or is surface input"
            )
        import nibabel

        data = np.asarray(self.data)
        if time_index is not None:
            data = data[..., time_index]
        return nibabel.Nifti1Image(data, affine)


class MneRaw(BaseExtractor):
    """
    Feature extractor for raw MNE data files.

    This class handles loading, preprocessing, and caching of continuous MNE
    raw recordings. Extractor preparation and caching are executed with extractor.prepare()

    The steps of preprocessing, if specified, are ordered as follows:
    1. Channel selection
    2. Drop bad channels
    3. Bipolar referencing
    4. Notch filtering
    5. Band-pass filtering
    6. Hilbert transform
    7. Resampling
    8. Scaling
    9. Applying projectors
    10. Baseline correction (applied on segments)
    11. Clamp (applied on segments)

    Parameters
    ----------
    baseline : tuple of float, optional
        If provided as a tuple ``(start, end)``, defines the start and end
        times (in seconds) relative to the **beginning of the analysis window**
        — note this differs from MNE's convention, which defines baseline
        relative to the epoch onset. Used for baseline correction.
    picks : str or tuple of str
        Channels to pick from the raw data. Can be channel types (e.g. `'meg'`,
        `'eeg'`), channel names (e.g. `'MEG 0111'`), `"all"` for all channels,
        or `"data"` for data channels. Regular expressions are supported for
        selecting channel names automatically (e.g. `'MEG ...1'`).
    frequency : "native" or float, default="native"
        Target sampling frequency. If `"native"`, uses the frequency of the
        input recording.
    offset : float, default=0.0
        Time offset (in seconds) to apply to the event (typically for aligning with the response)
    apply_proj : bool, default=False
        Whether to apply projectors stored in the MNE raw object.
    filter : tuple of (float or None, float or None), optional
        Band-pass filter limits as ``(l_freq, h_freq)``. If None, no band-pass
        filtering is applied.
    apply_hilbert : bool, default=False
        If True, applies the Hilbert transform to extract the signal envelope.
    notch_filter : float or list of float, optional
        Frequencies (in Hz) to apply a notch filter at. For a single frequency (as
        float) or a list of frequencies (as list of float), all harmonics of specified
        frequencies up to 300 Hz will be filtered out.
    drop_bads : bool, default=False
        Whether to drop channels marked as bad in the MNE info structure.
    mne_cpus : int, default=-1
        Number of CPUs to use for multiprocessing in MNE operations.
    scaler : {"RobustScaler", "StandardScaler"}, optional
        Optional scaling strategy to normalize channel data using scikit-learn
        scalers.
    scale_factor : float, optional
        Optional multiplicative factor applied to the data after scaling, but before clamping. E.g,
        can be used to convert from V to mV or uV.
    clamp : float, optional
        Maximum absolute value for clamping the data after preprocessing.
    fill_non_finite : float or None, optional
        If a float, any non-finite values (NaN / +inf / -inf) found after
        preprocessing are replaced with this value and a warning is logged.
        If None (the default), no replacement is performed.
    bipolar_ref : tuple of (list of str, list of str), optional
        Explicit anode/cathode channel name lists for bipolar referencing via
        ``mne.set_bipolar_reference``. The first list contains anode names and
        the second list contains cathode names; they must have the same length.
        Applied after channel selection and dropping bad channels but before
        filtering. The original monopolar channels consumed by the pairs are
        removed and replaced with the new bipolar channels.
    channel_order: ["unique", "original"]
        `if self.channel_order=="original"`
        Assigns channel indices for each raw file (doesn't match channel names across files).
        Allows use of a subject layer of fixed dimension across subjects.
        Prevents building a too large channel dimension when many subjects.

        `else`: is default behavior (unique)
        channels are numbered based on unique names across all subjects.
    allow_maxshield : bool, default=False
        If True, allow processing of Elekta/MEGIN MEG data recorded with
        Internal Active Shielding (MaxShield). Such recordings contain
        compensation signals that should normally be removed via SSS/tSSS
        (MaxFilter) before analysis. Does not affect the cache uid.

    """

    event_types: tp.Literal["Meg", "Eeg", "Emg", "Fnirs", "Ieeg"] = "Meg"

    frequency: tp.Literal["native"] | float = "native"
    offset: float = 0.0
    baseline: tuple[float, float] | None = None
    picks: str | tuple[str, ...] = pydantic.Field(("data",), min_length=1)
    apply_proj: bool = False
    filter: tuple[float | None, float | None] | None = None
    apply_hilbert: bool = False
    notch_filter: float | list[float] | None = None
    drop_bads: bool = False
    mne_cpus: int = -1
    infra: MapInfra = MapInfra(
        timeout_min=120,
        cpus_per_task=10,
        version="1",
    )
    scaler: None | tp.Literal["RobustScaler", "StandardScaler"] = None
    scale_factor: float | None = None
    clamp: float | None = None
    fill_non_finite: float | None = None
    bipolar_ref: tuple[list[str], list[str]] | None = None
    channel_order: tp.Literal["unique", "original"] = "unique"
    allow_maxshield: bool = False

    _channels: dict[str, int] = {}

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        prev = super()._exclude_from_cls_uid()
        return prev + ["mne_cpus", "allow_maxshield"]

    def _exclude_from_cache_uid(self) -> list[str]:
        prev = super()._exclude_from_cache_uid()
        return prev + ["baseline", "offset", "scale_factor", "clamp"]

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # check baseline
        if self.baseline is not None:
            issue = len(self.baseline) != 2
            issue |= not all(isinstance(b, float) for b in self.baseline)
            issue |= self.baseline[1] <= self.baseline[0]
            if issue:
                msg = f"baseline must be None or 2 floats, got {self.baseline}"
                raise ValueError(msg)
        if self.bipolar_ref is not None:
            anodes, cathodes = self.bipolar_ref
            if len(anodes) != len(cathodes):
                raise ValueError(
                    f"bipolar_ref anodes and cathodes must have equal length, "
                    f"got {len(anodes)} and {len(cathodes)}"
                )

    def prepare(self, obj: DataframeOrEventsOrSegments) -> None:
        """Specify how to load and preprocess the event.
        Can be overridden by user.
        """
        events: list[etypes.MneRaw]
        events = self._event_types_helper.extract(obj)  # type: ignore
        for ta in self._get_data(events):
            self._update_channels(ta.ch_names)
        if events:
            self(events[0], start=events[0].start, duration=0.001, trigger=events[0])

    @staticmethod
    def _pick_channels(
        raw: mne.io.Raw, picks_or_regexp: str | tuple[str, ...]
    ) -> mne.io.Raw:
        if isinstance(picks_or_regexp, str) and picks_or_regexp not in [
            "all",
            "data",
            "meg",
            "ref_meg",
        ] + list(_VALID_CHANNEL_TYPES):
            sel = mne.pick_channels_regexp(raw.ch_names, picks_or_regexp)
            picks = [raw.ch_names[i] for i in sel]
        else:
            picks = picks_or_regexp  # type: ignore
        return raw.pick(picks, verbose=False)

    def _preprocess_raw(self, raw: mne.io.Raw, event: etypes.MneRaw) -> MneTimedArray:
        if raw.info.get("maxshield", False) and not self.allow_maxshield:
            raise ValueError(
                f"Data for {event!r} was recorded with Elekta MaxShield "
                "(Internal Active Shielding). These recordings contain "
                "compensation signals that corrupt brain data unless removed "
                "by SSS/tSSS (MaxFilter). Set allow_maxshield=True on the "
                "extractor to proceed anyway."
            )
        raw = self._pick_channels(raw, self.picks)

        if self.drop_bads:
            raw.load_data()
            raw = raw.drop_channels(raw.info["bads"])

        if self.bipolar_ref is not None:
            raw.load_data()
            anodes, cathodes = self.bipolar_ref
            raw = mne.set_bipolar_reference(raw, anodes, cathodes, verbose="WARNING")

        if self.notch_filter is not None:
            raw.load_data()
            raw = self._notch_filter(raw, self.notch_filter, self.mne_cpus)

        if self.filter is not None:
            raw.load_data()
            l_freq, h_freq = self.filter
            # Ignore lowpass filter if cutoff is higher than Nyquist frequency
            if h_freq is not None and h_freq >= raw.info["sfreq"] / 2:
                logger.warning(
                    "Lowpass filter cutoff frequency is higher than or equal to the Nyquist frequency. "
                    "Setting it to None."
                )
                h_freq = None
            raw.filter(l_freq, h_freq, n_jobs=self.mne_cpus, verbose=False)

        if self.apply_hilbert:
            raw.load_data()
            raw = raw.apply_hilbert(envelope=True)

        if self.frequency not in ("native", event.frequency):
            raw.load_data()
            raw = raw.resample(float(self.frequency), n_jobs=self.mne_cpus, verbose=False)

        if self.scaler is not None:
            raw.load_data()
            scaler = getattr(sklearn.preprocessing, self.scaler)()
            raw._data = scaler.fit_transform(raw._data.T).T

        if self.apply_proj:
            raw.apply_proj()

        if self.fill_non_finite is not None:
            raw.load_data()
            if not np.all(np.isfinite(raw._data)):
                n_nonfinite = np.count_nonzero(~np.isfinite(raw._data))
                val = self.fill_non_finite
                logger.warning(
                    "Non-finite values (NaN/inf) detected in %s data for "
                    "event %s (%d values). Replacing with %s before caching.",
                    self.event_types,
                    event,
                    n_nonfinite,
                    val,
                )
                raw._data = np.nan_to_num(raw._data, nan=val, posinf=val, neginf=val)

        return MneTimedArray.from_native(raw)

    @infra.apply(
        item_uid=lambda e: e._splittable_event_uid(),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_data(self, events: tp.Sequence[etypes.MneRaw]) -> tp.Iterator[MneTimedArray]:
        for event in events:
            yield self._preprocess_raw(event.read(), event)

    def _get_timed_arrays(
        self, events: list[etypes.MneRaw], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        for event in events:
            yield self._get_timed_array(event, start, duration)

    def _get_timed_array(
        self, event: etypes.MneRaw, start: float, duration: float
    ) -> TimedArray:
        start += self.offset

        # Extend window in case of disjoint baseline
        window_start, window_stop = start, start + duration
        if self.baseline is not None:
            if self.baseline[0] >= self.baseline[1]:
                msg = f"unexpected baseline:{self.baseline}"
                raise RuntimeError(msg)
            window_start = min(window_start, start + self.baseline[0])
            window_stop = max(window_stop, start + self.baseline[1])

        ta = next(self._get_data([event]))
        freq = ta.frequency
        ch_names = ta.ch_names

        # Relocate cached data to the event's timeline position and extract window
        ta = ta.with_start(event.start)
        tdata = ta.overlap(start=window_start, duration=window_stop - window_start)
        tdata.data = np.asarray(
            tdata.data
        )  # materialize ContiguousMemmap before arithmetic
        if self.scale_factor is not None:
            tdata.data = tdata.data * self.scale_factor

        # Apply baseline to the data
        if self.baseline is not None:
            baseline_duration = self.baseline[1] - self.baseline[0]
            base = tdata.overlap(start + self.baseline[0], baseline_duration).data
            if base.size:
                tdata.data = tdata.data - base.mean(1, keepdims=True)
        tdata = tdata.overlap(start=start, duration=duration)

        # initialize output
        channel_idx = self._get_channels(ch_names)
        timed_out = TimedArray(frequency=freq, start=start, duration=duration)
        out_shape = (max(self._channels.values()) + 1, timed_out.data.shape[-1])
        out = np.zeros(out_shape, dtype=np.float32)
        if tdata.start == start and tdata.duration == duration:
            timed_out = tdata  # bypass copy for efficiency
        else:
            timed_out += tdata
        if self.clamp is not None:
            timed_out.data = np.clip(timed_out.data, a_min=-self.clamp, a_max=self.clamp)
        out[channel_idx, :] = timed_out.data
        timed_out.start -= self.offset
        timed_out.data = out
        return timed_out

    def _update_channels(self, ch_names: list[str]) -> None:
        """
        Update the indices assigned to channels based on unique indices across the dataset
        or based on original (per raw file) channel order.

        Example::

            `self.channel_order == "original"`

            Stack based on original channel order only:
            subject1: [a, b, c,]; subject2: [a, d, e]
            self._channel: {a: 0, b: 1, c: 2}
            self._channel: {a: 0, d:1, e:2}
            Allows use of a subject layer of fixed dimension across subjects
            Prevents building a too large channel dimension when many subjects

        Example::

            `self.channel_order = "unique"` (default behavior)

            Unique channel stacking: we loop across all mne channels.
            if this channel is not known, we create a new dimension for it:
            dimension = len(self._channels)
            subject1: [a, b, c,]; subject2: [a, d, e]
            self._channel: {a: 0, b: 1, c: 2}
            self._channel: {a: 0, b: 1, c: 2, d:3, e:3}
        """
        match self.channel_order:
            case "original":
                for i, ch in enumerate(ch_names):
                    self._channels[ch] = i

            case "unique":
                for ch in ch_names:
                    if ch not in self._channels:
                        self._channels[ch] = len(self._channels)

    def _get_channels(self, ch_names: list[str]) -> list[int]:
        if not self._channels:
            self._update_channels(ch_names)
        try:
            channel_idx = [self._channels[ch] for ch in ch_names]
        except KeyError as e:
            msg = f"Channel {e} not found in the channel mapping, likely because "
            msg += "this dataset contains recordings with different sets of channel "
            msg += "names. Try calling self.prepare on the whole events dataframe."
            raise KeyError(msg) from e
        return channel_idx

    @staticmethod
    def _notch_filter(
        raw: mne.io.Raw, notch_filter: float | list[float], mne_cpus: int
    ) -> mne.io.Raw:
        notch_filter = [notch_filter] if isinstance(notch_filter, float) else notch_filter
        notch_freqs: list[float] = []
        for freq in notch_filter:
            notch_freqs.extend(
                np.arange(freq, min(raw.info["sfreq"] / 2, 301), freq).tolist()  # type: ignore
            )

        if len(notch_freqs) == 0:
            logger.info("Not applying notch filter as no valid frequencies were found.")
        else:
            logger.info("Applying notch filter with notch_freqs=%s", sorted(notch_freqs))
            raw = raw.notch_filter(
                notch_freqs, phase="zero", filter_length="auto", n_jobs=mne_cpus
            )
        return raw


class MegExtractor(MneRaw):
    """
    MEG feature extractor.

    Parameters
    ----------
    picks: default = ("meg",)
        pick "meg" channels by default.
    """

    event_types: tp.Literal["Meg"] = "Meg"
    picks: str | tuple[str, ...] = pydantic.Field(("meg",), min_length=1)


class EegExtractor(MneRaw):
    """
    EEG feature extractor.

    Parameters
    ----------
    picks: default = ("eeg",)
        pick "eeg" channels by default.
    """

    event_types: tp.Literal["Eeg"] = "Eeg"
    picks: tuple[str, ...] = pydantic.Field(("eeg",), min_length=1)


class EmgExtractor(MneRaw):
    """
    EMG feature extractor.

    Parameters
    ----------
    picks: default = ("emg",)
        pick "emg" channels by default.
    """

    event_types: tp.Literal["Emg"] = "Emg"
    picks: tuple[str, ...] = pydantic.Field(("emg",), min_length=1)


class IeegExtractor(MneRaw):
    """
    Intracranial EEG feature extractor.

    Parameters
    ----------
    picks: default = ("seeg", "ecog", )
        pick "seeg" and "ecog" channels by default.
    reference: "bipolar" or None, default=None
        If "bipolar", applies a bipolar reference to the data, i.e., uses neighboring electrode as reference.
        Uses mne.set_bipolar_reference under the hood. [ieeg1]_

    Notes
    ----------
    Bipolar reference currently can only be applied to sEEG. It expects
    that the channels in raw.ch_names are ordered by probe,
    and with ascending order for each probe, and the names consists of
    the probe name followed by the position on the probe.
    eg: ['OF1', 'OF2', 'OF3', ... , 'OF12', 'OF13', 'OF14', 'H1', 'H2',
    'H3', ... , 'H13', 'H14', 'H15', ...]

    WATCH-OUT: this will take the closest electrode on the probe,
    meaning that if the neighboring electrode is missing for some
    reason (eg: rejected before applying the reference) then the next
    electrode will be used for referencing.


    References
    ----------
    .. [ieeg1] https://mne.tools/stable/generated/mne.set_bipolar_reference.html

    """

    event_types: tp.Literal["Ieeg"] = "Ieeg"
    picks: tuple[str, ...] = pydantic.Field(
        (
            "seeg",
            "ecog",
        ),
        min_length=1,
    )
    reference: tp.Literal["bipolar"] | None = None

    def model_post_init(self, log__):
        super().model_post_init(log__)

        if self.reference == "bipolar" and self.picks != ("seeg",):
            raise ValueError(
                f"Bipolar reference can only be applied on seeg signals, got picks {self.picks} instead"
            )
        if self.reference == "bipolar" and self.bipolar_ref is not None:
            raise ValueError(
                "Cannot use both reference='bipolar' (auto-derived from "
                "neighboring electrodes) and bipolar_ref (explicit "
                "anode/cathode lists) at the same time."
            )

    def _preprocess_raw(self, raw: mne.io.Raw, event: etypes.MneRaw) -> MneTimedArray:
        raw = self._pick_channels(raw, self.picks)

        if self.drop_bads:
            raw.load_data()
            raw = raw.drop_channels(raw.info["bads"])

        if self.reference == "bipolar":
            raw.load_data()
            raw = self._apply_bipolar_ref(raw)

        return super()._preprocess_raw(raw, event)

    def _apply_bipolar_ref(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Apply bipolar reference for EEG, i.e., uses neighboring electrode as reference.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw instance that will be referenced.

        Returns
        -------
        raw : mne.io.Raw
            Referenced Raw object

        """
        logger.info("Applying bipolar reference")
        logger.warning(
            "Assumes raw.ch_names are ordered by probe, with ascending order for each probe, and the names consists of the probe name followed by the probe position."
        )
        logger.warning(
            "WATCH-OUT: Taking the closest electrode on the probe... If the neighboring electrode is missing (e.g., rejected before applying the reference) then the next electrode will be used as the reference."
        )

        reference_ch = list(raw.ch_names)

        anodes = reference_ch[0:-1]
        cathodes = reference_ch[1:]
        to_del = []
        # allow constructions that are on the same probe
        #  (e.g., HA 5-HA 6 or HA 5-HA 7 if contact HA 6 is turned off)
        # don't allow constructions across probes
        #  (e.g., HA 5 with FA 6)
        for i, (a, c) in enumerate(zip(anodes, cathodes)):
            if [j for j in a if not j.isdigit()] != [j for j in c if not j.isdigit()]:
                to_del.append(i)
        for idx in to_del[::-1]:
            del anodes[idx]
            del cathodes[idx]
        bipol = mne.set_bipolar_reference(raw, anodes, cathodes, verbose="WARNING")
        return bipol


class SpikesExtractor(BaseExtractor):
    """Feature extractor for spike data stored in HDF5/NWB files.

    Reads spike times from HDF5 files and creates a dense binned array
    of shape (n_units, n_time_bins) at the specified frequency.

    The preprocessing steps, if specified, are ordered as follows:
    1. Spike binning at target frequency
    2. Scaling
    3. Baseline correction (applied on segments)
    4. Clamp (applied on segments)

    Parameters
    ----------
    frequency : "native" or float, default="native"
        Target sampling frequency for spike binning. If ``"native"``, uses the
        frequency declared in the Spikes event.
    offset : float, default=0.0
        Time offset (in seconds) applied to the segment window.
    baseline : tuple of float, optional
        If provided as ``(start, end)``, defines the baseline correction
        window in seconds relative to the segment start.
    scaler : {"RobustScaler", "StandardScaler"}, optional
        Scaling strategy to normalize channel data using scikit-learn scalers.
    scale_factor : float, optional
        Multiplicative factor applied to the data after scaling but before clamping.
    clamp : float, optional
        Maximum absolute value for clamping after preprocessing.
    channel_order : {"unique", "original"}, default="unique"
        ``"unique"``: channels are numbered based on unique names across all
        recordings. ``"original"``: channel indices follow per-recording order,
        enabling a fixed-size channel dimension across subjects.
    """

    event_types: tp.Literal["Spikes"] = "Spikes"
    frequency: tp.Literal["native"] | float = "native"
    offset: float = 0.0
    baseline: tuple[float, float] | None = None
    scaler: None | tp.Literal["RobustScaler", "StandardScaler"] = None
    scale_factor: float | None = None
    clamp: float | None = None
    channel_order: tp.Literal["unique", "original"] = "unique"

    requirements: tp.ClassVar[tp.Any] = ("h5py",)

    infra: MapInfra = MapInfra(
        timeout_min=120,
        cpus_per_task=10,
        version="1",
    )

    _channels: dict[str, int] = {}

    def _exclude_from_cache_uid(self) -> list[str]:
        prev = super()._exclude_from_cache_uid()
        return prev + ["baseline", "offset", "scale_factor", "clamp"]

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.baseline is not None:
            issue = len(self.baseline) != 2
            issue |= not all(isinstance(b, float) for b in self.baseline)
            issue |= self.baseline[1] <= self.baseline[0]
            if issue:
                msg = f"baseline must be None or 2 floats, got {self.baseline}"
                raise ValueError(msg)

    def prepare(self, obj: DataframeOrEventsOrSegments) -> None:
        events: list[etypes.Spikes]
        events = self._event_types_helper.extract(obj)  # type: ignore
        for ta in self._get_data(events):
            assert ta.header is not None
            self._update_channels(ta.header["ch_names"].split(","))
        if events:
            self(events[0], start=events[0].start, duration=0.001, trigger=events[0])

    @staticmethod
    def _bin_spikes(nwb_file: tp.Any, sfreq: float) -> tuple[np.ndarray, list[str]]:
        """Bin spike times from an open HDF5/NWB file into a dense array.

        Parameters
        ----------
        nwb_file : h5py.File
            Open HDF5 file following the NWB spikes convention
            (``units/id``, ``units/spike_times``, ``units/spike_times_index``).
        sfreq : float
            Sampling frequency for binning (Hz).

        Returns
        -------
        data : np.ndarray of shape (n_units, n_bins)
            Dense spike-count array.
        ch_names : list of str
            One name per unit.
        """
        units_id = nwb_file["units/id"][:]
        spikes = nwb_file["units/spike_times"][:]
        spike_times_index = nwb_file["units/spike_times_index"][:]

        max_time = float(np.max(spikes))
        n_bins = int(np.ceil(max_time * sfreq))
        data = np.zeros((len(units_id), n_bins), dtype=np.float32)

        for row_idx in range(len(units_id)):
            start_idx = 0 if row_idx == 0 else int(spike_times_index[row_idx - 1])
            stop_idx = int(spike_times_index[row_idx])
            unit_spikes = spikes[start_idx:stop_idx]
            bin_indices = np.floor(unit_spikes * sfreq).astype(int)
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            np.add.at(data[row_idx, :], bin_indices, 1)

        if (
            "units/electrodes" in nwb_file
            and "general/extracellular_ephys/electrodes" in nwb_file
        ):
            unit_electrodes = nwb_file["units/electrodes"][:]
            electrode_ids = nwb_file["general/extracellular_ephys/electrodes/id"][:]
            ch_names = [
                f"unit_{uid}_{electrode_ids[unit_electrodes[i]]}"
                for i, uid in enumerate(units_id)
            ]
        else:
            ch_names = [f"unit_{uid}" for uid in units_id]

        return data, [str(n) for n in ch_names]

    def _preprocess_spikes(self, event: etypes.Spikes) -> TimedArray:
        import h5py  # type: ignore[import-untyped]

        sfreq = (
            float(event.frequency)
            if self.frequency == "native"
            else float(self.frequency)
        )

        nwb_file = event.read()
        try:
            data, ch_names = self._bin_spikes(nwb_file, sfreq)
        finally:
            if isinstance(nwb_file, h5py.File):
                nwb_file.close()

        if self.scaler is not None:
            scaler_cls = getattr(sklearn.preprocessing, self.scaler)()
            data = scaler_cls.fit_transform(data.T).T

        header: dict[str, tp.Any] = {"ch_names": ",".join(ch_names)}
        return TimedArray(data=data, frequency=sfreq, start=event.start, header=header)

    @infra.apply(
        item_uid=lambda e: str(e.study_relative_path()),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_data(self, events: tp.Sequence[etypes.Spikes]) -> tp.Iterator[TimedArray]:
        for event in events:
            yield self._preprocess_spikes(event)

    def _get_timed_arrays(
        self, events: list[etypes.Spikes], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        for event in events:
            yield self._get_timed_array(event, start, duration)

    def _get_timed_array(
        self, event: etypes.Spikes, start: float, duration: float
    ) -> TimedArray:
        start += self.offset

        window_start, window_stop = start, start + duration
        if self.baseline is not None:
            if self.baseline[0] >= self.baseline[1]:
                raise RuntimeError(f"unexpected baseline:{self.baseline}")
            window_start = min(window_start, start + self.baseline[0])
            window_stop = max(window_stop, start + self.baseline[1])

        ta = next(self._get_data([event]))
        freq = ta.frequency
        assert ta.header is not None
        ch_names = ta.header["ch_names"].split(",")

        ta = ta.with_start(event.start)
        tdata = ta.overlap(start=window_start, duration=window_stop - window_start)
        tdata.data = np.asarray(tdata.data)
        if self.scale_factor is not None:
            tdata.data = tdata.data * self.scale_factor

        if self.baseline is not None:
            baseline_duration = self.baseline[1] - self.baseline[0]
            base = tdata.overlap(start + self.baseline[0], baseline_duration).data
            if base.size:
                tdata.data = tdata.data - base.mean(1, keepdims=True)
        tdata = tdata.overlap(start=start, duration=duration)

        channel_idx = self._get_channels(ch_names)
        timed_out = TimedArray(frequency=freq, start=start, duration=duration)
        out_shape = (max(self._channels.values()) + 1, timed_out.data.shape[-1])
        out = np.zeros(out_shape, dtype=np.float32)
        if tdata.start == start and tdata.duration == duration:
            timed_out = tdata
        else:
            timed_out += tdata
        if self.clamp is not None:
            timed_out.data = np.clip(timed_out.data, a_min=-self.clamp, a_max=self.clamp)
        out[channel_idx, :] = timed_out.data
        timed_out.start -= self.offset
        timed_out.data = out
        return timed_out

    def _update_channels(self, ch_names: list[str]) -> None:
        match self.channel_order:
            case "original":
                for i, ch in enumerate(ch_names):
                    self._channels[ch] = i
            case "unique":
                for ch in ch_names:
                    if ch not in self._channels:
                        self._channels[ch] = len(self._channels)

    def _get_channels(self, ch_names: list[str]) -> list[int]:
        if not self._channels:
            self._update_channels(ch_names)
        try:
            channel_idx = [self._channels[ch] for ch in ch_names]
        except KeyError as e:
            msg = f"Channel {e} not found in the channel mapping, likely because "
            msg += "this dataset contains recordings with different sets of channel "
            msg += "names. Try calling self.prepare on the whole events dataframe."
            raise KeyError(msg) from e
        return channel_idx


class FnirsExtractor(MneRaw):
    """
    Functional Near-Infrared Spectroscopy (fNIRS) feature extractor.

    Provides preprocessing and handling of fNIRS data using functions from mne.preprocessing.nirs,
    including channel selection, optical density conversion, haemodynamic response computation,
    and signal enhancement.

    The order of preprocessing, if specified, is as follows:
    1. Channel selection
    2. Drop bad channels
    3. Exclude channels based on source-detector distance
    4. Convert raw intensity data to optical density
    5. Exclude channels based on scalp coupling index
    6. Apply Temporal Derivative Distribution Repair (TDDR) to reduce motion artifacts
    7. Compute haemodynamic responses from optical density
    8. Band-pass filtering
    9. Enhance negatively correlated signals
    10. Resampling
    11. Scaling
    All steps executed with extractor.prepare()

    Parameters
    -----
    picks: default=("fnirs",)
        Channels to select for analysis, picks "fnirs" channels by default.
    distance_threshold: float | None, default=None
        Minimum source-detector distance for channel selection. Channels with
        distances below this threshold will be excluded.
    compute_optical_density: bool, default=False
        Whether to convert raw intensity data to optical density.
    scalp_coupling_index_threshold: float | None, default=None
        Minimum scalp coupling index for channel inclusion. Requires
        `compute_optical_density=True`.
    apply_tddr: bool, default=False
        Whether to apply Temporal Derivative Distribution Repair to reduce motion artifacts.
    compute_heamo_response: bool, default=False
        Whether to compute haemodynamic responses from optical density. Requires
        `compute_optical_density=True`.
    partial_pathlength_factor: float, default=0.1
        Partial pathlength factor used in the Beer-Lambert law for haemodynamic response calculation.
    enhance_negative_correlation: bool, default=False
        Whether to enhance negatively correlated signals. Requires
        `compute_heamo_response=True`.

    Notes
    -----
    - Filtering, resampling, scaling are applied if `filter` or `frequency` attributes are set (see parent class).
    """

    event_types: tp.Literal["Fnirs"] = "Fnirs"
    picks: tuple[str, ...] = pydantic.Field(("fnirs",), min_length=1)

    distance_threshold: float | None = None
    compute_optical_density: bool = False
    scalp_coupling_index_threshold: float | None = None
    apply_tddr: bool = False  # Apply temporal derivative distribution repair
    compute_heamo_response: bool = False
    partial_pathlength_factor: float = 0.1
    enhance_negative_correlation: bool = False

    requirements: tp.ClassVar[tp.Any] = ("mne-nirs",)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)

        # Ensure preprocessing steps are consistent with one another
        if self.compute_heamo_response and not self.compute_optical_density:
            msg = "Computing haemodynamic response requires computing optical density first."
            raise ValueError(msg)
        if self.scalp_coupling_index_threshold is not None:
            if not self.compute_optical_density:
                raise ValueError(
                    "Thresholding with the SCI requires computing optical density first."
                )

        if self.enhance_negative_correlation and not self.compute_heamo_response:
            msg = "Applying negative correlation enhancement requires haemodynamic responses."
            raise ValueError(msg)

    def _preprocess_raw(self, raw: mne.io.Raw, event: etypes.MneRaw) -> MneTimedArray:
        raw = self._pick_channels(raw, self.picks)

        if self.drop_bads:
            raw.load_data()
            raw = raw.drop_channels(raw.info["bads"])

        if self.distance_threshold is not None:
            dists = mne.preprocessing.nirs.source_detector_distances(raw.info)
            if np.isnan(dists).any():
                msg = "Some or all distances are nan, please fix montage information."
                raise ValueError(msg)
            picks = compress(raw.ch_names, dists > self.distance_threshold)
            raw = raw.pick(list(picks))

        if self.compute_optical_density:
            raw = mne.preprocessing.nirs.optical_density(raw)

        if self.scalp_coupling_index_threshold is not None:
            sci = mne.preprocessing.nirs.scalp_coupling_index(raw)
            picks = compress(raw.ch_names, sci > self.scalp_coupling_index_threshold)
            raw = raw.pick(list(picks))

        if self.apply_tddr:
            raw = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw)

        if self.compute_heamo_response:
            raw = mne.preprocessing.nirs.beer_lambert_law(
                raw, ppf=self.partial_pathlength_factor
            )

        if self.filter is not None:
            raw.load_data()
            raw.filter(
                self.filter[0], self.filter[1], n_jobs=self.mne_cpus, verbose=False
            )

        if self.enhance_negative_correlation:
            import mne_nirs

            raw = mne_nirs.signal_enhancement.enhance_negative_correlation(raw)

        if self.frequency not in ("native", event.frequency):
            raw.load_data()
            raw = raw.resample(float(self.frequency), n_jobs=self.mne_cpus, verbose=False)

        if self.scaler is not None:
            raw.load_data()
            scaler = getattr(sklearn.preprocessing, self.scaler)()
            raw._data = scaler.fit_transform(raw._data.T).T

        return MneTimedArray.from_native(raw)


# ---------------------------------------------------------------------------
# FmriExtractor sub-configs
# ---------------------------------------------------------------------------


class FmriCleaner(pydantic.BaseModel):
    """Configuration for fMRI signal cleaning.


    Parameters
    ----------
    standardize : str | bool
        Standardization method forwarded to ``nilearn.signal.clean``.
    detrend : bool
        Whether to remove linear trends.
    low_pass : float | None
        Low-pass filter cutoff in Hz.
    high_pass : float | None
        High-pass filter cutoff in Hz.
    filter: tp.Literal["butterworth", "cosine"] | None
        Filter to use: "butterworth" or "cosine".
    ensure_finite: bool
        Whether to set nans to 0.
    """

    model_config = pydantic.ConfigDict(extra="forbid")
    standardize: tp.Literal["zscore_sample", "zscore", "psc"] | bool = "zscore_sample"
    detrend: bool = True
    high_pass: float | None = None
    low_pass: float | None = None
    filter: tp.Literal["butterworth", "cosine"] | None = None
    ensure_finite: bool = True

    def clean(self, data: np.ndarray, t_r: float) -> np.ndarray:
        if (
            self.detrend
            or self.standardize
            or (self.high_pass is not None)
            or (self.low_pass is not None)
        ):
            import nilearn.signal

            data = data.T  # set time as first dim
            shape = data.shape
            data = nilearn.signal.clean(
                data.reshape(shape[0], -1),
                t_r=t_r,
                standardize=self.standardize,
                high_pass=self.high_pass,
                low_pass=self.low_pass,
                filter=self.filter,
                detrend=self.detrend,
                ensure_finite=self.ensure_finite,
            )
            data = data.reshape(shape).T
        return data


class BaseFmriProjector(DiscriminatedModel, discriminator_key="name"):
    """Base class for spatial projection sub-configs.

    Uses ``name`` as the pydantic discriminator key, following the
    ``NamedModel`` convention used throughout the codebase.

    Subclasses must implement :meth:`apply`.
    """

    def apply(self, rec: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        """Project a recording to the target space.

        Parameters
        ----------
        rec
            NIfTI-like recording (4-D volumetric or 2-D surface).
        **kwargs
            Subclass-specific options (e.g. ``standardize`` for atlas maskers).

        Returns
        -------
        np.ndarray
            Projected data with shape ``(n_features, time)``.
        """
        raise NotImplementedError


class SurfaceProjector(BaseFmriProjector):
    """Project data to an fsaverage surface mesh.
    For volumetric data, this uses ``nilearn.surface.vol_to_surf`` to project the data to the surface.
    For surface data, this simply downsamples the data to the target mesh resolution.

    Fields beyond ``mesh`` mirror the keyword arguments of
    ``nilearn.surface.vol_to_surf`` and are forwarded to it.

    Examples
    --------
    >>> SurfaceProjector(mesh="fsaverage5")
    >>> SurfaceProjector(mesh="fsaverage6", radius=5.0, interpolation="nearest")
    """

    mesh: str
    radius: float = 3.0
    interpolation: tp.Literal["linear", "nearest"] = "linear"
    kind: tp.Literal["auto", "line", "ball"] = "auto"
    n_samples: int | None = None
    mask_img: tp.Any | None = None
    depth: list[float] | None = None

    _mesh: tp.Any | None = pydantic.PrivateAttr(default=None)

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        if self.mesh not in FSAVERAGE_SIZES:
            raise ValueError(f"mesh must be an fsaverage mesh (got {self.mesh!r})")

    def get_mesh(self) -> tp.Any:
        if self._mesh is None:
            from nilearn import datasets

            fsaverage = datasets.fetch_surf_fsaverage(self.mesh)
            self._mesh = fsaverage
        return self._mesh

    def apply(self, rec: tp.Any) -> np.ndarray:
        if len(rec.shape) == 4:
            # 4-D volume data → use nilearn.surface.vol_to_surf
            mesh = self.get_mesh()
            from nilearn.surface import vol_to_surf

            hemis = [
                vol_to_surf(
                    rec,
                    surf_mesh=mesh[f"pial_{hemi}"],
                    inner_mesh=mesh[f"white_{hemi}"],
                    radius=self.radius,
                    interpolation=self.interpolation,
                    kind=self.kind,
                    n_samples=self.n_samples,
                    mask_img=self.mask_img,
                    depth=self.depth,
                )
                for hemi in ("left", "right")
            ]
            return np.vstack(hemis)

        elif len(rec.shape) == 2:
            # 2-D surface data → downsample to target mesh resolution
            n_vertices = rec.shape[0] // 2
            if n_vertices not in list(FSAVERAGE_SIZES.values()) or rec.shape[0] % 2:
                msg = f"The detected number of vertices ({rec.shape[0]}) is not in {list(FSAVERAGE_SIZES.values())}"
                raise ValueError(msg)
            n_vertices_resampled = FSAVERAGE_SIZES.get(self.mesh)
            data = rec.get_fdata()
            if n_vertices < n_vertices_resampled:
                raise NotImplementedError(
                    f"Cannot upsample from {n_vertices} vertices to {n_vertices_resampled} vertices"
                )
            if n_vertices > n_vertices_resampled:
                left = data[:n_vertices_resampled, :]
                right = data[n_vertices : n_vertices + n_vertices_resampled, :]
                data = np.concatenate([left, right], axis=0)
            return data
        else:
            raise ValueError(
                f"Unexpected shape {rec.shape} (should have 2 or 4 dimensions)"
            )


class MaskProjector(BaseFmriProjector):
    """Apply a mask to volumetric images.
    This converts the volumetric data of shape [x, y, z, time] to [n_voxels, time].

    Parameters
    ----------
    mask: str
        Mask to load from nilearn.datasets.
    resolution: int
        Resolution of the mask in millimeters. Must be 1 or 2.
    """

    mask: tp.Literal["mni152_brain", "mni152_gm", "mni152_wm", "subcortical"] = (
        "mni152_gm"
    )
    resolution: tp.Literal[1, 2] = 2
    _mask: tp.Any | None = pydantic.PrivateAttr(default=None)

    def get_mask(self) -> tp.Any:
        import nibabel as nib
        from nilearn import datasets

        if self._mask is None:
            if self.mask == "subcortical":
                atlas = datasets.fetch_atlas_harvard_oxford(
                    f"sub-maxprob-thr50-{self.resolution}mm"
                )
                excluded = ["Cortex", "White", "Stem", "Background"]
                selected_indices = [
                    i
                    for i, label in enumerate(atlas.labels)
                    if any([exc.lower() in label.lower() for exc in excluded])
                ]
                mask_data = atlas.maps.get_fdata()
                mask_data[np.isin(mask_data, selected_indices)] = 0
                mask = nib.Nifti1Image(mask_data, atlas.maps.affine, atlas.maps.header)
            else:
                mask_func = getattr(datasets, f"load_{self.mask}_mask")
                mask = mask_func(resolution=self.resolution)
            self._mask = mask
        return self._mask

    def apply(self, rec: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        if len(rec.shape) != 4:
            raise ValueError(f"Mask projection requires 4D data, got {rec.shape}")
        from nilearn.image import resample_to_img

        mask = self.get_mask()
        rec = resample_to_img(rec, mask, copy_header=True)
        return rec.get_fdata()[mask.get_fdata() > 0, :]


class AtlasProjector(BaseFmriProjector):
    """Project to atlas parcels via a nilearn masker.
    This converts volumetric data of shape [x, y, z, time] to [n_parcels, time],
    where each parcel is a weighted sum of voxels from the volumetric data.

    Parameters
    ----------
    atlas : str
        Atlas name; must match a ``nilearn.datasets.fetch_atlas_{atlas}`` function.
    atlas_kwargs : dict | None
        Keyword arguments forwarded to the nilearn fetch function.
        Validated against the function signature at init time.

    Examples
    --------
    >>> AtlasProjector(atlas="schaefer_2018", atlas_kwargs={"n_rois": 400})
    >>> AtlasProjector(atlas="difumo", atlas_kwargs={"dimension": 64})
    """

    atlas: str
    atlas_kwargs: dict[str, tp.Any] | None = None
    _masker: tp.Any = pydantic.PrivateAttr(default=None)

    def model_post_init(self, __context: tp.Any) -> None:
        super().model_post_init(__context)
        import nilearn.datasets as nds

        func_name = f"fetch_atlas_{self.atlas}"
        if not hasattr(nds, func_name):
            raise ValueError(f"Atlas {self.atlas!r} not found in nilearn.datasets")
        if self.atlas_kwargs is not None:
            func = getattr(nds, func_name)
            params = inspect.signature(func).parameters
            for k in self.atlas_kwargs:
                if k not in params:
                    raise ValueError(f"Invalid {func_name} argument: {k!r}")

    def get_atlas(self) -> tp.Any:
        from nilearn import datasets

        atlas_func = getattr(datasets, f"fetch_atlas_{self.atlas}")
        atlas = atlas_func(**(self.atlas_kwargs or {}))
        return atlas

    def _get_masker(
        self,
    ) -> tp.Any:
        """Build (and cache) the nilearn atlas masker."""
        if self._masker is None:
            from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker

            maps = self.get_atlas().maps
            if isinstance(maps, str):
                import nibabel as nib

                maps = nib.load(maps)
            if maps.get_fdata().ndim == 3:  # deterministic atlas
                masker = NiftiLabelsMasker(
                    labels_img=maps,
                )
            else:  # probabilistic atlas
                masker = NiftiMapsMasker(maps_img=maps)
            self._masker = masker
        return self._masker

    def apply(self, rec: tp.Any) -> np.ndarray:
        if len(rec.shape) != 4:
            raise ValueError(f"Atlas projection requires 4D data, got {rec.shape}")
        masker = self._get_masker()
        return masker.fit_transform(rec).T


class FoscoProjector(BaseFmriProjector):
    """Subset HCP grayordinates to FOSCO visual ROIs.

    Applies a spatial mask using ``hcp_utils`` to keep only vertices
    belonging to the FOSCO ROI set (41 visual areas from the HCP MMP1.0
    parcellation).  Expects 2-D input of shape ``(nodes, time)``.

    Examples
    --------
    >>> FoscoProjector()

    Config usage::

        "neuro": {
            "name": "FmriExtractor",
            "projection": {"name": "FoscoProjector"},
            "from_space": "atlas_msmall",
        }
    """

    ROIS: tp.ClassVar[list[str]] = [
        "V1",
        "MST",
        "V6",
        "V2",
        "V3",
        "V4",
        "MT",
        "V8",
        "V3A",
        "RSC",
        "POS2",
        "V7",
        "IPS1",
        "FFC",
        "V3B",
        "LO1",
        "LO2",
        "PIT",
        "PCV",
        "STV",
        "7m",
        "POS1",
        "23d",
        "v23ab",
        "d23ab",
        "31pv",
        "LIPv",
        "VIP",
        "MIP",
        "PH",
        "TPOJ1",
        "TPOJ2",
        "TPOJ3",
        "IP2",
        "IP1",
        "IP0",
        "VMV1",
        "VMV3",
        "LO3",
        "VMV2",
        "VVC",
    ]

    _mask: np.ndarray | None = pydantic.PrivateAttr(default=None)

    def _get_mask(self) -> np.ndarray:
        if self._mask is None:
            import hcp_utils as hcp  # type: ignore[import-not-found]

            self._mask = np.isin(
                hcp.mmp.map_all,
                [
                    k
                    for k, v in hcp.mmp.labels.items()
                    if (v.startswith("R_") or v.startswith("L_")) and v[2:] in self.ROIS
                ],
            )
        return self._mask

    def apply(self, rec: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        data = rec.get_fdata()
        if data.ndim != 2:
            raise ValueError(
                f"FoscoProjector expects 2D (nodes, time) data, got {data.ndim}D"
            )
        return data[self._get_mask(), :]


# ---------------------------------------------------------------------------
# FmriExtractor
# ---------------------------------------------------------------------------


class FmriExtractor(BaseExtractor):
    """fMRI feature extraction with optional projection, signal cleaning,
    and caching to a NumPy memmap.

    Input: a volumetric image of shape [x, y, z, time] or a surface image of shape [n_vertices, time].

    Preprocessing pipeline (each step is optional):

    1. **Spatial smoothing** (``fwhm``):
       If active, smooths the image with an isotropic Gaussian kernel.

    2. **Spatial projection** (``projection``):
       If active, projects the data to an array [n_features, time].
       There are three projection types: SurfaceProjector (-> [n_vertices, time]),
       MaskProjector (-> [n_voxels, time]), and AtlasProjector (-> [n_parcels, time]).

    3. **Signal cleaning** (``cleaning``):
       If active, cleans the data using ``nilearn.signal.clean``.

    4. **Temporal resampling** (``frequency``):
       If active, resamples the data to the target frequency, using np.interp.

    Parameters
    ----------
    offset : float
        Seconds to shift TRs forward to align delayed BOLD response.
    projection : BaseFmriProjector | None
        Spatial projection config — one of:

        * ``SurfaceProjector(mesh="fsaverage5")``
        * ``MaskProjector(mask="mni152_gm", resolution=2)``
        * ``AtlasProjector(atlas="schaefer_2018", atlas_kwargs={"n_rois": 400})``
        * ``None`` — keep raw volumetric / surface data
    cleaning : FmriCleaner | None
        Signal cleaning config, passed to ``nilearn.signal.clean``.
        ``None`` skips all cleaning.
    frequency : ``"native"`` | float
        Target sampling frequency.
    padding : int | ``"auto"`` | None
        Pad 1-D+T data to a uniform voxel count across subjects.
    from_space : str | ``"auto"`` | None
        Input space to load.  ``None`` (default) passes through when only
        one space is present, raises when multiple are found.
        ``"auto"`` uses a projection-aware heuristic.
        An explicit string selects that space.
    from_preproc : str | tuple[str, ...] | None
        Filter events by preprocessing pipeline.
    fwhm : float | None
        Full width at half maximum (in mm) for isotropic spatial smoothing
        via ``nilearn.image.smooth_img``. Applied after masking and before
        projection. ``None`` skips smoothing.
    """

    requirements: tp.ClassVar[tp.Any] = ("nilearn",)
    offset: float = 0.0
    event_types: tp.Literal["Fmri"] = "Fmri"
    projection: BaseFmriProjector | None = None
    cleaning: FmriCleaner | None = FmriCleaner()
    frequency: tp.Literal["native"] | float = "native"
    padding: int | tp.Literal["auto"] | None = None
    from_space: str | None = None
    from_preproc: str | tuple[str, ...] | None = None
    fwhm: float | None = None
    infra: MapInfra = MapInfra(
        timeout_min=120,
        cpus_per_task=10,
        version="2",
    )
    _padding: int | None = None

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if isinstance(self.padding, int):
            self._padding = self.padding
        if not self.projection and self.infra.folder is not None:
            logger.warning(
                f"{self.name}: volumetric fMRI (with projection=None) should not be cached: preprocessing should be fast, and caching will use too much disk space. Please set infra.folder=None"
            )

    def _exclude_from_cache_uid(self) -> list[str]:
        return super()._exclude_from_cache_uid() + ["offset", "padding"]

    def _auto_filter_fmri_events(
        self, fmri_events: list[etypes.Fmri]
    ) -> list[etypes.Fmri]:
        """Filter fMRI events by ``from_preproc``, ``from_space``, and ``projection``.

        1. **Preproc filter** -- drop events not matching ``self.from_preproc``
           (skipped when ``None``).
        2. **Space selection**:

           - Explicit string (e.g. ``"MNI152NLin2009cAsym"``) -- keep only
             that space; raises if no events match.
           - ``"auto"`` -- projection-aware heuristic: ``SurfaceProjector``
             prefers fsaverage meshes, other projectors prefer MNI spaces.
             Requires ``self.projection`` to be set.
           - ``None`` (default) -- pass through if one space, raise if
             multiple.
        """
        from_preproc = self.from_preproc
        from_space = self.from_space
        if isinstance(from_preproc, str):
            from_preproc = (from_preproc,)
        if from_preproc is not None:
            available_preprocs = {e.preproc for e in fmri_events}
            missing = set(from_preproc) - available_preprocs
            if missing:
                raise ValueError(
                    f"from_preproc={missing} not found. Available: {available_preprocs}"
                )
            fmri_events = [e for e in fmri_events if e.preproc in from_preproc]
        preproc_ctx = (
            f" (after from_preproc={self.from_preproc!r} filter)"
            if from_preproc is not None
            else ""
        )
        spaces = {e.space for e in fmri_events}
        if len(spaces) > 1 and from_space is None:
            logger.warning(
                "Multiple fMRI spaces found: %s. If you trigger list_segments "
                "on Fmri events, each space produces a separate segment. "
                "Filter with from_space or a QueryEvents transform "
                "before segmentation to avoid duplicates.",
                spaces,
            )
        if isinstance(from_space, str) and from_space != "auto":
            filtered = [e for e in fmri_events if e.space == from_space]
            if not filtered:
                raise ValueError(
                    f"from_space={from_space!r} matched no events. "
                    f"Available spaces: {spaces}{preproc_ctx}"
                )
            return filtered
        if len(spaces) <= 1:
            return fmri_events
        if from_space is None:
            raise ValueError(
                f"Multiple spaces found ({spaces}), set from_space explicitly "
                "or use from_space='auto'"
            )
        # from_space == "auto": projection-aware heuristic
        if self.projection is None:
            raise ValueError(
                f"from_space='auto' requires a projection to choose among "
                f"{spaces}. Set from_space to an explicit space name, or "
                "configure a projection."
            )
        fs = "fsaverage"
        if isinstance(
            self.projection, SurfaceProjector
        ) and self.projection.mesh.startswith(fs):
            candidates = [f"{fs}{n}" for n in range(3, 8)] + [fs]
        else:
            candidates = ["MNI152NLin2009cAsym"] + [s for s in spaces if "MNI" in s]
        best = next((c for c in candidates if c in spaces), None)
        if best is None:
            raise ValueError(
                f"No matching space for projection={self.projection!r} among "
                f"{spaces}. Set from_space to an explicit space name."
            )
        return [e for e in fmri_events if e.space == best]

    def prepare(self, obj: DataframeOrEventsOrSegments) -> None:
        all_events: list[etypes.Fmri] = self._event_types_helper.extract(obj)  # type: ignore[assignment]
        events = self._auto_filter_fmri_events(all_events)
        if self.padding == "auto":
            # for padding, we first need everything to be preprocessed
            # but we need on an object without padding since missing_default filling
            # will apply the extractor on 1 event, and we'll need the padding length for that
            self.infra.clone_obj(padding=None).prepare(events)
            self._padding = max(ta.data.shape[0] for ta in self._get_data(events))  # type: ignore
            # (recompute prepare to just fill the missing default value)
        super().prepare(events)

    @staticmethod
    def _resample_trs(
        data: np.ndarray, orig_tr: float, new_tr: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample TRs (i.e. in the time dimension) using np.interp.

        Since np.interp does not extrapolate, we cannot resample beyond the original time range.
        This means we likely lose one TR (or a few) at the very end of the timeline. These
        "invalid" TRs are filled with NaNs to keep the output data shape consistent with the
        original duration.
        '"""
        voxel_dims, n_trs = data.shape[:-1], data.shape[-1]

        trs = np.arange(n_trs) * orig_tr
        new_trs = np.arange(0, trs[-1] + orig_tr - new_tr, step=new_tr)

        flat_data = data.reshape(-1, n_trs)
        resamp_data = np.stack(
            [np.interp(new_trs, trs, voxel, right=voxel[-1]) for voxel in flat_data],
            axis=0,
        )
        resamp_data = resamp_data.reshape(*voxel_dims, -1)

        # Evaluate how many TRs were dropped at the end because of no extrapolation
        n_new_trs_without_extrapolation = int(trs[-1] / new_tr) + 1
        dropped_trs = len(new_trs) - n_new_trs_without_extrapolation
        if dropped_trs > 0:
            msg = (
                f"{dropped_trs} invalid TR(s) at the end of the timeline due to resampling without"
                f" interpolation from T={orig_tr:0.2f}s to T={new_tr:0.2f}s. These invalid TRs "
                "have been filled with NaNs."
            )
            logger.warning(msg)

        return resamp_data, new_trs

    def _get_relevant_events(
        self, events: tp.Any, trigger: etypes.Event | None
    ) -> list[etypes.Event]:
        """Filter Fmri events by preproc/space, then delegate aggregation to base."""
        fmri_events = self._auto_filter_fmri_events(
            self._event_types_helper.extract(events),  # type: ignore[arg-type]
        )
        return super()._get_relevant_events(fmri_events, trigger)

    def _preprocess_event(self, event: etypes.Fmri) -> FmriTimedArray:
        rec = event.read()
        space_dims = 3 if len(rec.shape) == 4 else 1

        if space_dims == 3 and event.mask_filepath is not None:
            rec = utils.get_masked_bold_image(rec, event.read_mask())

        # --- spatial smoothing ---
        if self.fwhm is not None:
            if space_dims != 3:
                raise ValueError(
                    f"Spatial smoothing (fwhm={self.fwhm}) requires volumetric "
                    f"(4-D) data, but got {len(rec.shape)}-D input."
                )
            from nilearn.image import smooth_img

            rec = smooth_img(rec, self.fwhm)

        # --- spatial projection ---
        header: dict[str, tp.Any] = {"preproc": event.preproc, "space": event.space}
        if self.projection is not None:
            data = self.projection.apply(rec)
            if isinstance(self.projection, SurfaceProjector):
                header["space"] = self.projection.mesh
        else:
            data = rec.get_fdata()
            if space_dims == 3:
                header["affine"] = np.array(rec.affine, dtype=np.float64)

        # --- signal cleaning ---
        if self.cleaning is not None:
            data = self.cleaning.clean(data, t_r=1 / event.frequency)

        # --- temporal resampling ---
        if self.frequency not in ("native", event.frequency):
            orig_tr = 1 / event.frequency
            new_tr = 1 / float(self.frequency)
            data, _ = self._resample_trs(data, orig_tr, new_tr)

        freq = event.frequency if self.frequency == "native" else self.frequency
        return FmriTimedArray(
            data=data.astype(np.float32),
            frequency=freq,
            start=event.start,
            duration=event.duration,
            header=header,
        )

    @infra.apply(
        item_uid=lambda e: e._splittable_event_uid(),
        exclude_from_cache_uid=_exclude_from_cache_uid,
    )
    def _get_data(self, events: list[etypes.Fmri]) -> tp.Iterable[TimedArray]:
        for event in tqdm(events, disable=len(events) < 2, desc="Computing fmri data"):
            yield self._preprocess_event(event)

    def _get_timed_arrays(
        self, events: list[etypes.Fmri], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        if self.padding == "auto" and self._padding is None:
            raise RuntimeError("Fmri.prepare needs to be called to compute auto padding")
        for _event, ta in zip(events, self._get_data(events)):
            data = ta.data
            if self._padding is not None:
                if data.ndim != 2:
                    raise ValueError(f"Only 1D+T FMRI can be padded, got {data.shape=}")
                padding = self._padding - data.shape[0]
                if padding < 0:
                    raise ValueError(
                        f"Padding to length {self._padding} but got {data.shape=}"
                    )
                data = np.pad(data, [(0, self._padding - data.shape[0]), (0, 0)])
            yield FmriTimedArray(
                data=data,
                frequency=ta.frequency,
                start=ta.start - self.offset,
                duration=ta.duration,
                header=ta.header,
            )


class ChannelPositions(BaseStatic):
    """Channel positions in 2D or 3D, extracted from a Raw object's mne.Info.

    3D positions (``n_spatial_dims=3``) are always returned in MNE's **head coordinate frame**,
    which is defined by the LPA, RPA, and nasion fiducial landmarks (origin at the midpoint of
    LPA–RPA, x-axis toward RPA, y-axis toward nasion, z-axis upward). This holds regardless of
    whether positions come from a named standard montage or from the raw data's channel locations.
    See https://mne.tools/stable/documentation/implementation.html#coordinate-systems for details.

    Parameters
    ----------
    neuro:
        Extractor that defines the preprocessing steps applied to the Raw objects.
        This can either be specified in the config, or built with the `build` method.
    n_spatial_dims: int
        Number of spatial dimensions (i.e. coordinates) to extract for each channel. For
        `n_spatial_dims=2`, the 2D projection of the channel positions as obtained through
        `mne.Layout` will be used. For `n_spatial_dims=3`, the 3D positions are extracted from
        `mne.Montage` in head coordinate frame.
    layout_or_montage_name:
        Name of the Layout or Montage to use. See `mne.channels.read_layout()` for a list of valid
        layouts and `mne.channels.get_builtin_montages()` for standard montages. If not provided,
        the function will look for a layout in the `Raw.info` object or for a montage in the `Raw`
        object.
        **Note**: MNE's standard montages are only for EEG systems; MEG montages must be loaded from
        the raw data.
    include_ref_eeg:
        If True, additionally try to extract the position of the anode of bipolar EEG channel (e.g.
        for the channel name "P3-Cz", return position of both "P3" and "Cz"), yielding and output
        of shape (n_channels, n_spatial_dims * 2). If True, `event_types` must be one of
        Eeg or Ieeg.
    normalize:
        If True, min-max normalize channel positions between 0 and 1 across each dimension. If
        False, 2D positions are in arbitrary units given by the mne.Layout projection, while 3D
        positions will be in head coordinate frame (approximately in the range [-0.1, 0.1] meters).
    factor:
        Factor to scale the channel positions by. E.g. set it to 10.0 to get 3D coordinates in
        decimeters, which yields values approximately in the range [-1, 1].
    """

    event_types: tp.Literal["MneRaw", "Meg", "Eeg", "Ieeg"] = "MneRaw"
    neuro: MneRaw | None = None

    n_spatial_dims: tp.Literal[2, 3] = 2
    layout_or_montage_name: str | None = None
    include_ref_eeg: bool = False
    normalize: bool = True
    factor: float = 1.0

    _neuro: MneRaw = pydantic.PrivateAttr()

    # Value to use for channels that are not found in the layout
    INVALID_VALUE: tp.ClassVar[float] = -0.1

    infra: MapInfra = MapInfra()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)

        if self.neuro is not None:
            if self.event_types not in {"MneRaw", self.neuro.event_types}:
                msg = f"event_types={self.event_types} must match "
                msg += f"neuro.event_types={self.neuro.event_types}."
                raise ValueError(msg)

            eegs = {"Eeg", "Ieeg"}
            if self.include_ref_eeg and self.neuro.event_types not in eegs:
                msg = "include_ref_eeg=True is only supported for events_types "
                msg += f"Eeg and Ieeg, got {self.event_types}."
                raise ValueError(msg)
            if self.n_spatial_dims == 2 and isinstance(self.neuro, MegExtractor):
                raise NotImplementedError(
                    "n_spatial_dims=2 is not supported for MEG data. "
                )
            self._neuro = self.neuro

    def build(self, neuro: MneRaw) -> "ChannelPositions":
        config = self.model_dump()
        config["neuro"] = neuro
        if config.get("include_ref_eeg") and neuro.event_types not in {"Eeg", "Ieeg"}:
            logger.warning(
                "include_ref_eeg=True is not supported for event_types=%s; "
                "overriding to False.",
                neuro.event_types,
            )
            config["include_ref_eeg"] = False
        return self.__class__(**config)

    def prepare(self, obj: DataframeOrEventsOrSegments) -> None:
        events = self._event_types_helper.extract(obj)
        if not hasattr(self, "_neuro"):
            raise ValueError(
                "The neuro extractor is not set. Either set it in the config or call build."
            )
        self._neuro.prepare(events)  # Ensure the Raw objects have been precomputed
        super().prepare(events)

    def _get_layout_positions(self, ta: MneTimedArray) -> dict[str, list[float]]:
        if self.layout_or_montage_name is not None:
            layout = mne.channels.read_layout(self.layout_or_montage_name)
        else:
            try:
                layout = mne.find_layout(ta.to_info())
            except RuntimeError as err:
                msg = "No valid layout found. Please specify a layout to load with argument "
                msg += "`layout_name` or explicitly set a montage in the study class (e.g. with "
                msg += "`raw.set_montage()`)."
                raise ValueError(msg) from err

        mapping = {name: pos[:2].tolist() for name, pos in zip(layout.names, layout.pos)}
        return mapping

    def _get_positions_from_ch_locs(self, ta: MneTimedArray) -> dict[str, list[float]]:
        """Extract 3D channel positions from cached ch_locs, filtering out zero positions."""
        ch_locs: np.ndarray = ta.header["ch_locs"]
        mapping: dict[str, list[float]] = {}
        for i, name in enumerate(ta.ch_names):
            loc = np.asarray(ch_locs[i, :3])
            # [0, 0, 0] means the position is unknown. This shouldn't happen for
            # MEG sensors or when ch_locs are properly set, but we filter
            # defensively for robustness.
            if not np.all(loc == 0):
                mapping[name] = loc.tolist()
        if not mapping:
            raise RuntimeError(
                "No valid channel positions found in the cached TimedArray header. "
                "Please specify a layout_or_montage_name."
            )
        return mapping

    def _get_montage_positions(self, ta: MneTimedArray) -> dict[str, list[float]]:
        if self.layout_or_montage_name is not None:
            montage = mne.channels.make_standard_montage(self.layout_or_montage_name)
            native_head_t = mne.channels.compute_native_head_t(montage)
            ch_pos = montage.get_positions()["ch_pos"]
            # Standard montages shouldn't contain [0, 0, 0] positions, but we
            # filter defensively for consistency with _get_positions_from_ch_locs.
            mapping = {
                name: mne.transforms.apply_trans(native_head_t["trans"], pos).tolist()
                for name, pos in ch_pos.items()
                if not np.all(pos == 0)
            }
            if not mapping:
                raise RuntimeError(
                    "No valid montage positions found (all positions were zero). "
                    "Please check the montage name."
                )
            return mapping
        return self._get_positions_from_ch_locs(ta)

    def _compute_positions(self, ta: MneTimedArray) -> torch.Tensor:
        """Get scaled channel positions from a cached MneTimedArray.

        Returns
        -------
        torch.Tensor :
            Positions for each channel, of shape (n_channels, n_spatial_dims). When including
            reference channel (self.include_ref_eeg is True), output shape is
            (n_channels, n_spatial_dims * 2) where each row contains the coordinates of the cathode
            channel followed by the coordinates of the anode.
        """
        pos_mapping: dict[str, list[float]] = {}
        is_meg = isinstance(self._neuro, MegExtractor)
        if self.n_spatial_dims == 2:
            pos_mapping = self._get_layout_positions(ta)
        elif self.n_spatial_dims == 3:
            if is_meg:
                pos_mapping = self._get_positions_from_ch_locs(ta)
            else:
                pos_mapping = self._get_montage_positions(ta)

        ta_ch_names = ta.ch_names
        ch_names: list[str] = []
        for ch_name in ta_ch_names:
            if self.include_ref_eeg:
                parts = ch_name.split("-", 1)
                ch_names.append(parts[0])
                ch_names.append(parts[1] if len(parts) == 2 else "")
            else:
                ch_names.append(ch_name.split("-")[0])
        valid_inds = [i for i, n in enumerate(ch_names) if n in pos_mapping]
        invalid_names = [n for n in ch_names if n and n not in pos_mapping]

        if not valid_inds:
            raise ValueError(f"No channel has valid positions: {ta_ch_names}.")

        if len(valid_inds) < 0.1 * len(ch_names):
            unique_invalid_names = set(invalid_names)
            msg = f"Fewer than 10% of the channels have valid positions: {unique_invalid_names}."
            logger.warning(msg)

        positions = np.array(
            [
                (
                    pos_mapping[name]
                    if name in pos_mapping
                    else [np.nan] * self.n_spatial_dims
                )
                for name in ch_names
            ]
        )

        if self.normalize:
            ptp = np.nanmax(positions, axis=0, keepdims=True) - np.nanmin(
                positions, axis=0, keepdims=True
            )
            if (ptp == 0.0).any():
                ptp[ptp == 0.0] = 1.0
            positions = (positions - np.nanmin(positions, axis=0, keepdims=True)) / ptp

        positions *= self.factor
        positions = np.nan_to_num(positions, nan=self.INVALID_VALUE)

        n_spatial_dims = self.n_spatial_dims
        if self.include_ref_eeg:
            n_spatial_dims *= 2  # type: ignore
            positions = positions.reshape(len(ta_ch_names), n_spatial_dims)

        channel_idx = self._neuro._get_channels(ta_ch_names)
        out = torch.full((len(self._neuro._channels), n_spatial_dims), self.INVALID_VALUE)
        out[channel_idx, :] = torch.from_numpy(positions).float()

        return out

    def _exclude_from_cache_uid(self) -> list[str]:
        ex = super()._exclude_from_cache_uid()
        if not hasattr(self, "_neuro"):
            raise RuntimeError("Should not happen")
        neuro_ex = self._neuro._exclude_from_cache_uid()
        return ex + [f"neuro.{n}" for n in neuro_ex]

    @infra.apply(
        item_uid=lambda e: str(e.study_relative_path()),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_data(self, events: list[etypes.MneRaw]) -> tp.Iterator[torch.Tensor]:
        if not hasattr(self, "_neuro"):
            raise ValueError(
                "The neuro extractor is not set. Either set it in the config or call build."
            )
        for ta in self._neuro._get_data(events):
            yield self._compute_positions(ta)

    def get_static(self, event: etypes.MneRaw) -> torch.Tensor:
        return next(self._get_data([event]))


class HrfConvolve(BaseExtractor):
    """
    Convolve the output of an extractor by the Hemodynamic Response Function.

    Note that this extractor does not support timelines with events.start < 0.
    Note that this is stored by timeline. If the events change in a timeline,
    e.g. by using different transform with a similar, then there will be a silent
    bug.

    Parameters
    ----------
    extractor: BaseExtractor
        the extractor used for feature extraction
    frequency: Literal["native"]
        The frequency of the cropped extractor. Must be "native". Never used
    """

    event_types: str | tuple[str, ...] = "Event"
    extractor: BaseExtractor
    offset: float = 0
    duration: pydantic.PositiveFloat | None = None
    frequency: pydantic.PositiveFloat
    infra: MapInfra = MapInfra(keep_in_ram=True)
    aggregation: tp.Literal["mean"] = "mean"
    _device: tp.Literal["auto", "cuda", "cpu"] = "auto"

    requirements: tp.ClassVar[tuple[str, ...]] = ("nilearn",)

    def model_post_init(self, log__: tp.Any) -> None:
        if self.infra.keep_in_ram is False and self.infra.folder is None:
            msg = "HrfConvolve requires a cache (folder=my/cache or keep_in_ram=True)."
            raise ValueError(msg)

        if isinstance(self.extractor, BaseStatic) and self.extractor.frequency == 0.0:
            raise ValueError(
                "HrfConvolve cannot crop a static extractor as it is timeless."
            )
        self.event_types = self.extractor.event_types
        super().model_post_init(log__)

    def prepare(self, obj: tp.Any) -> None:
        from neuralset.events.utils import extract_events

        # Prepare low-level extractors
        self.extractor.prepare(obj)

        # Group events by timeline
        events = extract_events(obj, types=self._event_types_helper)
        timelines_dict = defaultdict(list)
        for event in events:
            timelines_dict[event.timeline].append(event)
        timelines = list(timelines_dict.values())

        # Cache prepare each hrf-convolved timeline
        self._get_data(timelines)

        # Populate shape
        self(timelines[0], 0, duration=0.001, trigger=timelines[0][0])

        # TODO store somewhere a hash of the timeline events to avoid
        # silent bugs, e.g. on subset of the timeline events?

    @infra.apply(item_uid=lambda tl_events: tl_events[0].timeline)
    def _get_data(self, timelines: list[list[etypes.Event]]) -> tp.Iterator[TimedArray]:
        for events in timelines:
            if any([ev.start < 0 for ev in events]):
                raise ValueError("HrfConvolve does not support events with start<0")
            if len(set([ev.timeline for ev in events])) > 1:
                raise ValueError("events should be grouped by timeline.")

        if self.extractor._effective_frequency is None:
            raise ValueError("hrf.extractor must be prepared.")

        for events in timelines:
            from nilearn.glm.first_level.hemodynamic_models import spm_hrf

            # Get timeline duration
            duration = 0.0
            for event in events:
                event_stop = event.start + event.duration
                if event_stop > duration:
                    duration = event_stop

            # get all dynamic extractors
            TIME_LENGTH = 32.0
            feature_data = self.extractor(
                events, 0, duration + TIME_LENGTH, trigger=None
            ).float()

            # shapes
            *batch_dims, n_times = feature_data.shape
            feature_flat = feature_data.reshape(-1, n_times)  # (N, n_times)

            # compute HRF (numpy array)
            dt = 1.0 / (self.extractor._effective_frequency)
            hrf = spm_hrf(dt, oversampling=1, time_length=TIME_LENGTH)

            # hrf amplitude
            kernel = torch.from_numpy(hrf).float().view(1, 1, len(hrf))
            kernel = torch.flip(kernel, dims=[2])  # pytorch compute corr not conv

            device = self._device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            # full convolution: pad on left by len(hrf) - 1
            conv = torch.nn.functional.conv1d(
                feature_flat.unsqueeze(1).to(device),  # (N, 1, n_times)
                kernel.to(device),
                padding=len(hrf) - 1,
            )  # (N, 1, n_times + len(hrf) - 1)

            # crop back to original time
            data = conv.squeeze(1)  # (N, n_times)

            # restore original batch dims
            yield TimedArray(
                frequency=self.frequency,
                start=0,
                data=data.reshape(*batch_dims, -1).cpu().numpy(),
            )

    def _get_timed_arrays(
        self, events: list[etypes.Event], start: float, duration: float
    ) -> tp.Iterator[TimedArray]:
        if self.extractor._effective_frequency is None:
            raise ValueError("You must first hrf.prepare(events).")
        yield from self._get_data([events])
