# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import logging
import tempfile
import typing as tp

import exca
import numpy as np
import pandas as pd
import pydantic
import torch

from neuralset import events as _ev

from ..events import Event
from ..events.utils import extract_events
from . import base
from .image import HuggingFaceImage
from .text import HuggingFaceText

logger = logging.getLogger(__name__)


class TimeAggregatedExtractor(base.BaseStatic):
    """Remove the time dimension of a dynamic extractor, either by summing/averaging or by
    selecting the first, middle or last time point.

    NOTE: This is not exactly a static extractor because its output depends on the start and duration
    of the window (whereas static extractors only depend on the event). Hence, the get_static method
    is not implemented.

    Parameters
    ----------
    time_aggregation: str
        How to aggregate the time dimension.
        Can be "sum", "mean", "first", "middle", "last" or an integer.
    n_groups_concat: int | None
        If provided, the time dimension is divided into `n_groups` equal parts and the aggregation
        is carried out within each group, before being concatenated.
    extractor: BaseExtractor
        The extractor to aggregate.
    """

    time_aggregation: tp.Literal["sum", "mean", "first", "last"] = "mean"
    n_groups_concat: pydantic.PositiveInt | None = None
    event_types: str | tuple[str, ...] = "Event"
    extractor: base.BaseExtractor

    def model_post_init(self, log__: tp.Any) -> None:
        self.event_types = self.extractor.event_types
        if self.frequency != 0:
            name = self.__class__.__name__
            raise ValueError(f"{name}.frequency must be 0")
        return super().model_post_init(log__)

    def prepare(self, events: pd.DataFrame) -> None:
        self.extractor.prepare(events)

    def _aggregate(self, out: torch.Tensor) -> torch.Tensor:
        match self.time_aggregation:
            case "sum":
                return out.sum(-1)
            case "mean":
                return out.mean(-1)
            case "first":
                return out[..., 0]
            case "last":
                return out[..., -1]
            case other:
                raise ValueError(f"Unknown time_aggregation: {other}")

    def __call__(
        self,
        events: tp.Any,  # too complex: pd.DataFrame | list | dict | Event,
        start: float,
        duration: float,
        trigger: Event | pd.Series | dict | None = None,
    ) -> torch.Tensor:
        out = self.extractor(events, start, duration, trigger)
        if self.n_groups_concat is None:
            return self._aggregate(out)
        if self.n_groups_concat > out.shape[-1]:
            raise ValueError(
                f"n_groups_concat ({self.n_groups_concat}) cannot be greater than "
                f"the number of time points ({out.shape[-1]})."
            )
        return torch.cat(
            [
                self._aggregate(sub)
                for sub in torch.tensor_split(out, self.n_groups_concat, dim=-1)
            ]
        )

    def get_static(self, *args: tp.Any, **kwargs: tp.Any) -> torch.Tensor:
        msg = f"{type(self).__name__}.get_static should not be called as the extractor is dynamic"
        raise RuntimeError(msg)


class AggregatedExtractor(base.BaseExtractor):
    """Aggregate multiple extractors along the specified dimension.
    Note that self.extractor_aggregation determines how the extractors are aggregated for a given event,
    whereas self.aggregation determines how different events are aggregated
    (after the extractors have been aggregated).
    """

    event_types: str | tuple[str, ...] = "Event"
    extractors: list[base.BaseExtractor]
    extractor_aggregation: tp.Literal["cat", "stack", "mean", "sum"] = "cat"
    frequency: tp.Literal["native"] = "native"  # deferred to sub-extractors

    def model_post_init(self, log__: tp.Any) -> None:
        """Check that extractors are all static or all dynamic."""
        fts = self.extractors
        static_count = sum(isinstance(f, base.BaseStatic) for f in fts)
        if static_count not in [0, len(fts)]:
            raise ValueError("Extractors must be either all static or all dynamic.")
        if not static_count:  # dynamic
            frequencies = set(f.frequency for f in self.extractors)
            if len(frequencies) > 1:
                raise ValueError("All extractors must have the same frequency.")
        all_event_types = set(c for f in fts for c in f._event_types_helper.classes)
        self.event_types = tuple(c.__name__ for c in all_event_types)
        super().model_post_init(log__)

    def prepare(self, events: pd.DataFrame) -> None:
        for extractor in self.extractors:
            extractor.prepare(events)

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> torch.Tensor:
        out = [feat(*args, **kwargs) for feat in self.extractors]
        aggreg = self.extractor_aggregation
        if aggreg == "sum":
            return sum(out)  # type: ignore
        if aggreg == "mean":
            return sum(out) / len(out)  # type: ignore
        return getattr(torch, aggreg)(out, dim=0)  # type: ignore


class ExtractorPCA(base.BaseStatic):
    """Applies a PCA to another extractor's data
    The underlying extractor is first computed through the prepare method, and then
    the current extractor applies the PCA on it.
    Both caches are stored.

    Parameters
    ----------
    extractor: Extractor
        the underlying extractor on which the PCA must be applied
    n_components: int
        the number of components of the PCA
    whiten: bool
        whether the whiten post PCA
    """

    extractor: base.BaseExtractor
    n_components: int
    whiten: bool = True
    event_types: str | tuple[str, ...] = "Event"
    infra: exca.MapInfra = exca.MapInfra()
    _uid: str = pydantic.PrivateAttr("")

    def model_post_init(self, log__: tp.Any) -> None:
        self.event_types = self.extractor.event_types
        self.aggregation = self.extractor.aggregation
        self.frequency = self.extractor.frequency  # type: ignore
        super().model_post_init(log__)
        if self.infra.cluster is not None:
            raise ValueError(f"Cannot use a cluster on {self!r}")
        if not isinstance(self.extractor, base.BaseStatic):
            raise NotImplementedError("Cannot handle non-static extractors for now")
        if not hasattr(self.extractor, "infra"):
            raise NotImplementedError("Cannot handle extractor with no infra")
        if self.infra.folder is None:
            if self.extractor.infra.folder is not None:  # type: ignore
                self.infra.folder = self.extractor.infra.folder  # type: ignore
                msg = "Setting ExtractorPCA infra folder to the underlying extractor's"
                logger.warning(msg)

    def _prepare_pca_uid(self, obj: tp.Any) -> dict[str, _ev.Event]:
        pca_events = {
            self.extractor.infra.item_uid(e): e  # type: ignore
            for e in self._event_types_helper.extract(obj)
        }
        # compute a unique hash for this set of data
        m = hashlib.sha256()
        for uid in sorted(pca_events):
            m.update(uid.encode("utf8"))
        self._uid = f"{m.hexdigest()[:8]},{len(pca_events)}"
        logger.debug("%r.prepare called for uid=%r", self, self._uid)
        # use this to have a specific folder for caching
        cache = self.infra.cache_dict
        pca_events = {self._to_uid(e): e for e in pca_events.values()}
        if cache:  # if cache is already filled, we should be done
            missing = set(pca_events) - set(cache)
            if not missing:
                return {}
            if len(missing) != len(pca_events):
                msg = "Cache exists but with missing items, something went wrong"
                msg += f"\n(eg: missing {list(missing)[0]} in {cache.folder})"
                raise RuntimeError(msg)
        return pca_events

    def prepare(self, obj: tp.Any) -> None:
        pca_events = self._prepare_pca_uid(obj)
        if not pca_events:
            logger.debug("In %r, all events for uid=%r are cached", self, self._uid)
            return  # all done
        # compute a unique hash for this set of data
        from sklearn.decomposition import PCA

        # Note: all the PCA has to be done on prepare (in the main thread)
        # because it cannot be split and distributed onto many machines
        events = list(pca_events.values())
        self.extractor.prepare(events)
        start = min(e.start for e in events)
        duration = max(e.start + e.duration for e in events) - start
        tas = list(
            self.extractor._get_timed_arrays(events, start=start, duration=duration)
        )
        if any(ta.frequency for ta in tas):
            raise RuntimeError("Dynamic extractors are not currently supported")
        data = [ta.data.ravel() for ta in tas]
        pca = PCA(n_components=self.n_components, whiten=self.whiten)
        pca_data = pca.fit_transform(np.array(data))
        if len(events) != len(pca_data):
            raise RuntimeError("Something went wrong")
        # write to cache
        done = set()
        with self.infra.cache_dict.writer() as w:
            for i_uid, d in zip(pca_events, pca_data):
                if i_uid not in done:
                    w[i_uid] = d
                    done.add(i_uid)
        logger.debug("%r.prepare finished with uid=%r", self, self._uid)

    def _to_uid(self, event: _ev.Event) -> str:
        if not self._uid:
            msg = f"prepare method must be called first for extractor {self!r}"
            raise RuntimeError(msg)
        uid = f"{self._uid},{self.extractor.infra.item_uid(event)}"  # type: ignore
        # apply item_uid because of automatic uid shortening
        return self.infra.item_uid(uid)

    @infra.apply(item_uid=str, exclude_from_cache_uid="method:_exclude_from_cache_uid")
    def _get_data(self, uids: tp.Iterable[str]) -> np.ndarray:
        uids = list(uids)
        msg = "Events should have been prepared first, trying to recover "
        msg += f"{uids[0]} (prepare uid: {self._uid!r})"
        raise RuntimeError(msg)

    def _get_timed_arrays(
        self, events: list[_ev.Event], start: float, duration: float
    ) -> tp.Iterable[base.TimedArray]:
        uids = [self._to_uid(e) for e in events]
        for event, d in zip(events, self._get_data(uids)):
            yield base.TimedArray(
                data=d, frequency=0.0, start=event.start, duration=event.duration
            )


class HuggingFacePCA(ExtractorPCA):
    """Applies a PCA to the underlying HuggingFace extractor.
    The underlying extractor is first computed through the prepare method, and then
    the current extractor applies the PCA on it.
    Compared to the ExtractorPCA extractor, HuggingFacePCA handles caching of multiple
    layers at once in the cache.
    By default, the hugging face extractor cache is deleted afterwards.

    Parameters
    ----------
    extractor: HuggingFace Extractor
        the underlying extractor on which the PCA must be applied
    n_components: int
        the number of components of the PCA
    whiten: bool
        whether the whiten post PCA
    use_tmp_cache: bool
        whether to use a temporary cache folder for the underlying
        extractor that gets deleted afterwards
    """

    extractor: base.BaseExtractor
    # uses a temporary cache for the extractor which gets deleted afterwards
    use_tmp_cache: bool = True

    def model_post_init(self, log__: tp.Any) -> None:
        if not isinstance(self.extractor, (HuggingFaceText, HuggingFaceImage)):
            raise TypeError("Only HuggingFaceText and HuggingFaceImage are supported")
        if self.extractor.infra.folder is None:
            raise RuntimeError("extractor's infra folder should be provided")
        super().model_post_init(log__)

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["use_tmp_cache"]

    def _exclude_from_cache_uid(self) -> list[str]:
        excl = [f"extractor.{x}" for x in self.extractor._exclude_from_cache_uid()]
        if "extractor.frequency" in excl:
            excl.append("frequency")
        return super()._exclude_from_cache_uid() + excl

    def prepare(self, obj: tp.Any) -> None:
        pca_events = self._prepare_pca_uid(obj)
        if not pca_events:
            logger.debug("In %r, all events for uid=%r are cached", self, self._uid)
            return  # all done

        from sklearn.decomposition import PCA

        # Note: all the PCA has to be done on prepare (in the main thread)
        # because it cannot be split and distributed onto many machines
        if not isinstance(self.extractor, base.HuggingFaceMixin):  # for typing
            raise TypeError("Only HuggingFaceText and HuggingFaceImage are supported")
        events = list(pca_events.values())
        basefolder = self.extractor.infra.folder
        with tempfile.TemporaryDirectory(prefix="HF-PCA-tmp", dir=basefolder) as tmp:
            # change to a temporary folder folder that will be deleted
            feat_folder = tmp if self.use_tmp_cache else basefolder
            feat = self.extractor.infra.clone_obj(**{"infra.folder": feat_folder})
            feat.prepare(events)
            embds = list(feat._get_data(events))  # N x Layer x *Embd
            # note: this may be too big to keep in memory, so we dont turn it to array
            # (keep as list of memmaps)
            # compute PCA per layer
            layers = []
            for k in range(embds[0].shape[0]):
                layer = np.array([e[k].ravel() for e in embds])  # N * prod(*Embd)
                pca = PCA(n_components=self.n_components, whiten=self.whiten)
                layers.append(pca.fit_transform(layer))
                del layer  # free memory if need be
            embds.clear()
            # avoid keeping files open, explicitly delete cache
            del feat.infra._state.cache_dict
            del feat
        # back to N * Layer * prod(*Embd)
        pca_embds = np.array(layers).transpose((1, 0, 2))
        if len(events) != len(pca_embds):
            raise RuntimeError("Something went wrong")
        # write to cache
        done = set()
        with self.infra.cache_dict.writer() as w:
            for i_uid, d in zip(pca_events, pca_embds):
                if i_uid not in done:
                    w[i_uid] = d
                    done.add(i_uid)
        logger.debug("%r.prepare finished with uid=%r", self, self._uid)

    def _get_timed_arrays(
        self, events: list[_ev.Event], start: float, duration: float
    ) -> tp.Iterable[base.TimedArray]:
        uids = [self._to_uid(e) for e in events]
        feat = self.extractor
        if not isinstance(feat, base.HuggingFaceMixin):  # for typing
            raise TypeError("Only HuggingFaceText and HuggingFaceImage are supported")
        for event, d in zip(events, self._get_data(uids)):
            if feat.cache_n_layers is not None:
                d = feat._aggregate_layers(d)
            yield base.TimedArray(
                data=d, frequency=0, start=event.start, duration=event.duration
            )


class CroppedExtractor(base.BaseStatic):  # can be static or not
    """Crop a extractor to a given offset and duration.

    Parameters
    ----------
    extractor: BaseExtractor
        The extractor to crop.
    offset: float
        The offset (in seconds) from the start of the event to begin the crop.
    duration: PositiveFloat | None
        The duration (in seconds) of the crop. If None, the crop extends to the end of the event.
    frequency: Literal["native"]
        The frequency of the cropped extractor. Must be "native". Never used
    """

    event_types: str | tuple[str, ...] = "Event"
    extractor: base.BaseExtractor
    offset: float = 0
    duration: pydantic.PositiveFloat | None = None
    frequency: tp.Literal["native"] = "native"  # type: ignore

    def model_post_init(self, log__: tp.Any) -> None:
        self.event_types = self.extractor.event_types
        self.frequency = self.extractor.frequency  # type: ignore
        super().model_post_init(log__)

    def __call__(
        self,
        events: tp.Any,
        start: float,
        duration: float,
        trigger: Event | pd.Series | dict | None = None,
    ) -> torch.Tensor:
        if self.duration is not None and (self.duration > duration - self.offset):
            msg = f"Crop duration ({self.duration}) cannot be greater than "
            msg += "segment duration minus offset ({duration} - {self.offset})"
            raise ValueError(msg)
        elif self.offset >= duration:
            msg = f"Crop offset ({self.offset}) must be less than event duration ({duration})"
            raise ValueError(msg)

        duration_ = duration - self.offset if self.duration is None else self.duration
        return self.extractor(events, start + self.offset, duration_, trigger)

    def prepare(self, obj: tp.Any) -> None:
        self.extractor.prepare(obj)


class ToStatic(base.BaseStatic):
    """
    Crop a extractor by a given offset and duration.

    Parameters
    ----------
    extractor: BaseExtractor
        The extractor to crop.
    """

    extractor: base.BaseExtractor
    event_types: str | tuple[str, ...] = "Event"
    frequency: pydantic.PositiveFloat = 0.0
    aggregation: tp.Literal["trigger"] = "trigger"

    def model_post_init(self, context: tp.Any) -> None:
        if isinstance(self.extractor, base.BaseStatic):
            raise ValueError("ToStatic cannot crop a static extractor as it is timeless.")
        if self.extractor.aggregation != "single":
            raise NotImplementedError(
                f"ToStatic only accept extractor with `single` aggregation, got {self.extractor.aggregation}"
            )
        self.event_types = self.extractor.event_types
        super().model_post_init(context)

    def __call__(
        self,
        events: tp.Any,
        start: float,
        duration: float,
        trigger: Event | pd.Series | dict,
    ) -> torch.Tensor:
        if not isinstance(trigger, Event):
            triggers = extract_events(trigger)
            if len(triggers) != 1:
                msg = f"trigger must be a single event, got {len(triggers)} from {trigger!r}"
                raise RuntimeError(msg)
            trigger = triggers[0]

        if trigger.start < start or trigger.start > start + duration:
            msg = (
                f"trigger {trigger.start} outside segment [{start}, {start + duration}]."
            )
            raise RuntimeError(msg)

        out = self.extractor(events, trigger.start, 0, None)
        if out.shape[-1] != 1:
            msg = "something went wrong, the extractor output should have 1 time sample"
            raise RuntimeError(msg)
        return out[..., 0]

    def prepare(self, obj: tp.Any) -> None:
        self.extractor.prepare(obj)
