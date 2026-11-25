# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import json
import logging
import typing as tp
from pathlib import Path

import numpy as np
import pydantic
import torch
from exca import MapInfra
from torch.nn import functional
from tqdm import tqdm

from neuralset import base as nsbase
from neuralset.events import etypes as evts

from . import base as extractor_base
from . import hf
from . import image as image_extractors

logger = logging.getLogger(__name__)
# activate with:
# logging.getLogger("neuralset").setLevel(logging.DEBUG)

_VideoImage = image_extractors._VideoImage


def _huggingface_video_event_uid(event: evts.Image | evts.Video) -> str:
    if event.type == "Video":
        return event._splittable_event_uid()  # type: ignore[union-attr]
    elif event.type == "Image":
        return str(event.study_relative_path())
    else:
        raise ValueError(f"Incorrect event type for video extractor: {event.type}")


def resamp_first_dim(data: torch.Tensor, new_first_dim: int) -> torch.Tensor:
    if data.shape[0] == new_first_dim:
        return data
    import julius

    logger.debug(
        "Resampling video embedding from %s samples to %s", data.shape[0], new_first_dim
    )
    resample = julius.resample.ResampleFrac(
        old_sr=data.shape[0],
        new_sr=new_first_dim,
    ).to(data.device)
    dims = []
    for dim in tqdm(data.reshape(data.shape[0], -1).T):
        dims.append(resample(dim.float()))
    # TODO: stack an extra frame here?
    output = torch.stack(dims).reshape(-1, *data.shape[1:])
    return output


class HuggingFaceVideoConfig(hf.HuggingFaceConfig):
    processor_kwargs: dict[str, tp.Any] | None = {"do_rescale": True}
    HF_CLASS_DEFAULTS: tp.ClassVar[dict[str, dict[str, str]]] = {
        "vjepa2": {"processor_cls_name": "AutoVideoProcessor"},
        "google/vivit": {
            "model_cls_name": "VivitModel",
            "processor_cls_name": "VivitImageProcessor",
        },
    }


class HuggingFaceVideo(
    extractor_base.BaseExtractor, hf.HuggingFaceMixin, extractor_base._ContextMixin
):
    """Temporally resolved embeddings from native HuggingFace video transformers
    (VideoMAE, V-JEPA 2, ViViT, TimeSformer).

    Videos are encoded in windows of ``num_frames`` frames spanning
    ``context.duration`` seconds. The tokens are aggregated over space only
    (``token_aggregation``), and the temporal tokens (one per frame or tubelet) of
    all windows are concatenated along time, then averaged to ``frequency`` before
    caching. Image events are embedded as static clips (the image repeated
    ``num_frames`` times), averaged over their temporal tokens.

    Parameters
    ----------
    model_name : str, default="MCG-NJU/videomae-base"
        HuggingFace video model identifier. Image models are not accepted here:
        use :class:`~neuralset.extractors.image.HuggingFaceImage` for frame-by-frame
        video embeddings.
    num_frames : int | None, default=None
        Number of frames sampled evenly within each window (or repeated image).
        None: the model's own, read from its config (``num_frames``, fixed by the
        position embeddings of VideoMAE, ViViT and TimeSformer, or V-JEPA 2's
        ``frames_per_clip``, its pretraining clip length). When set, it must match
        the config's ``num_frames`` if any (V-JEPA 2 takes any number).
    context : ContextWindow, default=ContextWindow(duration=8.0)
        Windows of video (see :class:`~neuralset.extractors.base.ContextWindow`),
        with their overlap rounded to whole temporal tokens: 8 s windows sharing
        half of their frames by default. ``context.duration`` sets the native
        frequency of the temporal tokens (temporal tokens per window /
        ``context.duration``), and must be set for Video events.
    frequency : float
        Output frequency, at most the native one, to which the temporal tokens are
        averaged before caching. 0 (static outputs) is only valid for Image events.
    max_imsize : int | None, default=None
        Maximum image dimension, beyond which frames are downsampled.

    Note
    ----
    In neuralset 0.4, this extractor encoded one clip of ``clip_duration``
    (default ``1 / frequency``) seconds per output sample and aggregated all its
    tokens: ``context={"duration": 1 / frequency, "overlap": 0}`` gives the closest
    features.
    """

    event_types: (
        tp.Literal["Image", "Video"] | tuple[tp.Literal["Image", "Video"], ...]
    ) = "Video"
    # HuggingFace model types (config.model_type), also matched in model_name
    SUPPORTED_MODELS: tp.ClassVar[tuple[str, ...]] = (
        "videomae",
        "vjepa2",
        "vivit",
        "timesformer",
    )
    requirements: tp.ClassVar[tuple[str, ...]] = (
        "torchvision>=0.15.2",
        "pillow>=9.2.0",
    )
    model_name: str = "MCG-NJU/videomae-base"
    hf_config: HuggingFaceVideoConfig = HuggingFaceVideoConfig()
    num_frames: pydantic.PositiveInt | None = None
    context: extractor_base.ContextWindow = extractor_base.ContextWindow(duration=8.0)
    frequency: pydantic.NonNegativeFloat
    max_imsize: int | None = None
    token_aggregation: tp.Literal["mean", "sum", "max"] = "mean"
    infra: MapInfra = MapInfra(
        timeout_min=120,
        gpus_per_node=1,
        cpus_per_task=8,
        min_samples_per_job=128,
        version="v6",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _reject_previous_api(cls, data: tp.Any) -> tp.Any:
        if not isinstance(data, dict):
            return data
        if "image" in data:
            msg = (
                "HuggingFaceVideo no longer accepts the previous API "
                "`image=HuggingFaceImage(...)`. For frame-by-frame video "
                "embeddings, instantiate HuggingFaceImage with event_types='Video'. "
                "For native video models, pass the model name directly as "
                "HuggingFaceVideo(model_name=..., frequency=...)."
            )
            raise ValueError(msg)
        if "clip_duration" in data:
            msg = (
                "HuggingFaceVideo.clip_duration was removed after neuralset 0.4: videos "
                "are now encoded in windows of context.duration seconds, whose "
                "temporal tokens are averaged to frequency. One clip of T seconds "
                "per output sample becomes context={'duration': T, 'overlap': 0} "
                f"with frequency=1/T (got {data!r})."
            )
            raise ValueError(msg)
        return data

    @pydantic.field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, model_name: str) -> str:
        # early check on the name; _temporal_tokens checks the loaded model's type
        if any(z in model_name.lower() for z in cls.SUPPORTED_MODELS):
            return model_name
        msg = (
            "HuggingFaceVideo only supports native video models "
            f"({', '.join(cls.SUPPORTED_MODELS)}). For frame-by-frame video "
            "embeddings from an image model, instantiate HuggingFaceImage "
            f"with event_types='Video' instead of using model_name={model_name!r}."
        )
        raise ValueError(msg)

    def _event_types(self) -> set[str]:
        types = self.event_types
        return {types} if isinstance(types, str) else set(types)

    def _is_static(self) -> bool:
        # images are static clips
        return "Video" not in self._event_types()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)  # downloads the model snapshot
        config = self._local_model_config()
        fixed = config.get("num_frames")  # fixed by the position embeddings
        if self.num_frames is None:
            native = fixed if fixed is not None else config.get("frames_per_clip")
            if not isinstance(native, int):
                msg = f"{self.name}.num_frames must be set: the config of "
                msg += f"{self.model_name!r} has no num_frames or frames_per_clip"
                raise ValueError(msg)
            # resolved here, so that the cache uid is that of the explicit value
            self.num_frames = native
        elif isinstance(fixed, int) and fixed != self.num_frames:
            msg = f"{self.name}.num_frames={self.num_frames}, but {self.model_name!r} "
            msg += f"takes {fixed} frames per clip (config.num_frames, fixed by its "
            raise ValueError(msg + "position embeddings): use num_frames=None")
        if "Video" in self._event_types() and self.context.duration is None:
            msg = f"{self.name}.context.duration must be set for Video events: each "
            msg += f"window spans it with num_frames={self.num_frames} frames"
            raise ValueError(msg)

    @property
    def _num_frames(self) -> int:
        """``num_frames``, resolved in ``model_post_init``."""
        assert self.num_frames is not None
        return self.num_frames

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return hf.HuggingFaceMixin._exclude_from_cls_uid()

    def _exclude_from_cache_uid(self) -> list[str]:
        return [
            *extractor_base.BaseExtractor._exclude_from_cache_uid(self),
            *hf.HuggingFaceMixin._exclude_from_cache_uid(self),
        ]

    def _predict_hidden_states(self, images: np.ndarray) -> torch.Tensor:
        """(batch, layers, tokens, dim) hidden states of a clip of frames."""
        parameters = inspect.signature(self.processor.__call__).parameters
        field = "videos" if "videos" in parameters else "images"
        inputs = self.processor(**{field: list(images)}, return_tensors="pt")
        image_extractors._fix_pixel_values(inputs)  # uniform frames give nans
        inputs = inputs.to(self.model_device)
        kwargs = {"output_hidden_states": True}
        if "skip_predictor" in inspect.signature(self.model.forward).parameters:
            kwargs["skip_predictor"] = True  # V-JEPA: the encoder states suffice
        with torch.inference_mode():
            pred = self.model(**inputs, **kwargs)
        return torch.stack(pred.hidden_states, dim=1)

    def _n_temporal_tokens(self) -> int:
        # TimeSformer has one token per frame, ViViT tubelets are (time, height, width)
        tubelet = getattr(self.model.config, "tubelet_size", 1)
        if isinstance(tubelet, (list, tuple)):
            tubelet = tubelet[0]
        return self._num_frames // tubelet

    def _temporal_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Aggregate (layers, tokens, dim) states over space into (layers, dim, time)."""
        config: tp.Any = self.model.config
        if config.model_type not in self.SUPPORTED_MODELS:
            msg = f"Unsupported video model type {config.model_type!r} for "
            raise ValueError(
                msg + f"{self.model_name!r}, use one of {self.SUPPORTED_MODELS}"
            )
        if config.model_type in ("vivit", "timesformer"):
            hidden_states = hidden_states[:, 1:]  # CLS token
        n_layers, _, dim = hidden_states.shape
        n_times = self._n_temporal_tokens()
        if config.model_type == "timesformer":  # tokens are ordered space-major
            grid = hidden_states.reshape(n_layers, -1, n_times, dim)
        else:
            grid = hidden_states.reshape(n_layers, n_times, -1, dim).transpose(1, 2)
        return self._aggregate_tokens(grid).transpose(1, 2)

    def _resize(self, images: list[tp.Any]) -> np.ndarray:
        """Stacked frames, downsampled beyond ``max_imsize``."""
        if self.max_imsize is not None and max(images[0].size) > self.max_imsize:
            factor = max(images[0].size) / self.max_imsize
            size = tuple(int(s / factor) for s in images[0].size)
            images = [image.resize(size) for image in images]
        return np.stack([np.asarray(image) for image in images])

    def _window_frames(self, video: tp.Any, start: float, step: float) -> np.ndarray:
        times = start + (np.arange(self._num_frames) + 0.5) * step
        # events shorter than the context are padded with their last frame
        times = np.minimum(times, np.nextafter(video.duration, 0))
        return self._resize([_VideoImage(video=video, time=t).read() for t in times])

    def _get_timed_arrays(
        self,
        events: list[evts.Image | evts.Video],
        start: float,
        duration: float,
    ) -> tp.Iterable[nsbase.TimedArray]:
        for event, ta in zip(events, self._get_data(events)):
            if isinstance(event, evts.Image):
                sub = ta.copy(start=event.start, duration=event.duration)
            else:
                sub = ta.copy(start=event.start).overlap(start=start, duration=duration)
            if self.cache_n_layers is not None:
                sub.data = self._aggregate_layers(sub.data)
            yield sub

    @infra.apply(
        item_uid=_huggingface_video_event_uid,
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_data(
        self, events: tp.Sequence[evts.Image | evts.Video]
    ) -> tp.Iterator[nsbase.TimedArray]:
        for event in events:
            if isinstance(event, evts.Image):
                yield self._embed_image(event)
            elif isinstance(event, evts.Video):
                yield self._embed_video(event)
            else:
                msg = f"Incorrect event type for video extractor: {event.type}"
                raise ValueError(msg)

    def _embed_image(self, event: evts.Image) -> nsbase.TimedArray:
        """Static clip of the image repeated ``num_frames`` times."""
        image = event.read().convert("RGB")
        frames = self._resize([image] * self._num_frames)
        tokens = self._temporal_tokens(self._predict_hidden_states(frames)[0])
        data = tokens.mean(dim=-1).cpu().numpy()
        if self.cache_n_layers is None:
            data = self._aggregate_layers(data)
        return nsbase.TimedArray(
            data=data, frequency=0, start=nsbase._UNSET_START, duration=event.duration
        )

    def _embed_video(self, event: evts.Video) -> nsbase.TimedArray:
        duration = self.context.duration
        assert duration is not None  # model_post_init
        native = self._n_temporal_tokens() / duration
        if self.frequency > native:
            raise ValueError(
                f"{self.name}.frequency={self.frequency} exceeds the native frequency "
                f"{native} of the temporal tokens (tokens per window / context.duration)."
            )
        n_native = max(1, nsbase.Frequency(native).to_ind(event.duration))
        n_times = max(1, nsbase.Frequency(self.frequency).to_ind(event.duration))
        windows = self.context.windows(n_native, native)
        video = event.read()
        parts = []
        for start, _, keep_start, keep_stop in tqdm(windows, desc="Encoding video"):
            frames = self._window_frames(
                video, start / native, duration / self._num_frames
            )
            tokens = self._temporal_tokens(self._predict_hidden_states(frames)[0])
            parts.append(tokens[..., keep_start - start : keep_stop - start].cpu())
        video.close()
        latents = torch.cat(parts, dim=-1)
        data = functional.interpolate(latents, size=n_times, mode="area").numpy()
        if self.cache_n_layers is None:
            data = self._aggregate_layers(data)
        return nsbase.TimedArray(
            data=data,
            frequency=self.frequency,
            start=nsbase._UNSET_START,
            duration=event.duration,
        )

    def _local_model_config(self) -> dict[str, tp.Any]:
        """Model config read from the local snapshot (its ``vision_config`` if
        any), without loading the model; empty if not available locally."""
        from huggingface_hub import hf_hub_download

        revision = (self.hf_config.model_kwargs or {}).get("revision")
        try:
            path = hf_hub_download(
                self.model_name, "config.json", revision=revision, local_files_only=True
            )
        except (OSError, ValueError):
            return {}
        config: dict[str, tp.Any] = json.loads(Path(path).read_text(encoding="utf8"))
        return config.get("vision_config", config)
