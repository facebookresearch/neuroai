# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import logging
import typing as tp
import warnings

import numpy as np
import pydantic
import torch
from exca import MapInfra
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
        return tp.cast(evts.Video, event)._splittable_event_uid()
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


class HuggingFaceVideo(extractor_base.BaseExtractor, hf.HuggingFaceMixin):
    """Extract video embeddings using a native HuggingFace video model.

    Videos are divided into clips of `clip_duration` seconds at the specified
    frequency. Each clip is processed by the video model, and features are
    aggregated over layers/tokens using the HuggingFace extractor options.
    Image events are repeated into static `num_frames`-frame clips and
    returned as static embeddings.

    Parameters
    ----------
    model_name : str, default="MCG-NJU/videomae-base"
        HuggingFace video model identifier.
        Image models are not accepted here; use `HuggingFaceImage` for
        frame-by-frame video embeddings.
    clip_duration : float | None, default=None
        Duration (in seconds) of video sub-clips to process. If None, defaults to
        one timestep (1 / frequency).
    max_imsize : int | None, default=None
        Maximum image dimension for downsampling before processing.
    num_frames : int
        Number of frames to pass to the video model per clip.
    """

    event_types: (
        tp.Literal["Image", "Video"] | tuple[tp.Literal["Image", "Video"], ...]
    ) = "Video"
    SUPPORTED_MODELS: tp.ClassVar[tuple[str, ...]] = (
        "vjepa2",
        "videomae",
        "google/vivit",
        "facebook/timesformer",
    )
    # class attributes
    requirements: tp.ClassVar[tuple[str, ...]] = (
        "torchvision>=0.15.2",
        "julius>=0.2.7",
        "pillow>=9.2.0",
    )
    model_name: str = "MCG-NJU/videomae-base"
    hf_config: HuggingFaceVideoConfig = HuggingFaceVideoConfig()
    frequency: float = 1.0
    clip_duration: float | None = None
    max_imsize: int | None = None
    num_frames: int
    infra: MapInfra = MapInfra(
        timeout_min=120,
        gpus_per_node=1,
        cpus_per_task=8,
        min_samples_per_job=128,
        version="v5",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _reject_previous_api(cls, data: tp.Any) -> tp.Any:
        if isinstance(data, dict) and "image" in data:
            msg = (
                "HuggingFaceVideo no longer accepts the previous API "
                "`image=HuggingFaceImage(...)`. For frame-by-frame video "
                "embeddings, instantiate HuggingFaceImage with event_types='Video'. "
                "For native video models, pass the model name directly as "
                "HuggingFaceVideo(model_name=..., num_frames=...)."
            )
            raise ValueError(msg)
        return data

    @pydantic.field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, model_name: str) -> str:
        if any(z in model_name for z in cls.SUPPORTED_MODELS):
            return model_name
        msg = (
            "The HuggingFaceVideo API now only supports native video models. "
            "For the previous frame-by-frame API, instantiate HuggingFaceImage "
            f"with event_types='Video' instead of using model_name={model_name!r}."
        )
        raise ValueError(msg)

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return hf.HuggingFaceMixin._exclude_from_cls_uid()

    def _exclude_from_cache_uid(self) -> list[str]:
        return extractor_base.BaseExtractor._exclude_from_cache_uid(
            self
        ) + hf.HuggingFaceMixin._exclude_from_cache_uid(self)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.event_types == "Image":
            warnings.warn(
                "HuggingFaceVideo with event_types='Image' applies a video model "
                "to repeated static image frames. Consider HuggingFaceImage for "
                "image-only features.",
                stacklevel=2,
            )

    def _get_timed_arrays(
        self,
        events: list[evts.Image | evts.Video],
        start: float,
        duration: float,
    ) -> tp.Iterable[nsbase.TimedArray]:
        for event, data in zip(events, self._get_data(events)):
            if event.type == "Image":
                image_event = tp.cast(evts.Image, event)
                data = np.asarray(data)
                if self.cache_n_layers is not None:
                    data = self._aggregate_layers(data)
                yield nsbase.TimedArray(
                    frequency=0,
                    duration=image_event.duration,
                    start=image_event.start,
                    data=data,
                )
                continue
            video_event = tp.cast(evts.Video, event)
            ta = tp.cast(nsbase.TimedArray, data)
            sub = ta.with_start(video_event.start).overlap(start=start, duration=duration)
            if self.cache_n_layers is not None:
                sub.data = self._aggregate_layers(sub.data)
            yield sub

    @infra.apply(
        item_uid=_huggingface_video_event_uid,
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_data(
        self, events: tp.Sequence[evts.Image | evts.Video]
    ) -> tp.Iterator[np.ndarray | nsbase.TimedArray]:
        # read all media events
        logging.getLogger("neuralset").setLevel(logging.DEBUG)
        self._warn_if_config_num_frames_mismatch()
        video_events = [event for event in events if event.type == "Video"]
        subtimes: list[float] = []
        if video_events:
            freq = self.frequency
            T = 1 / freq if self.clip_duration is None else self.clip_duration
            subtimes = [k / self.num_frames * T for k in reversed(range(self.num_frames))]
        for event in events:
            if event.type == "Image":
                yield self._get_image_data(tp.cast(evts.Image, event))
                continue
            event = tp.cast(evts.Video, event)
            video = event.read()

            freq = self.frequency
            expect_frames = nsbase.Frequency(freq).to_ind(event.duration)
            logger.debug(
                "Loaded Video (duration %ss at %sfps, shape %s):\n%s",
                video.duration,
                video.fps,
                tuple(video.size),
                event.filepath,
            )
            # time at end of sample:
            times = np.linspace(0, video.duration, expect_frames + 1)[1:]
            # samples the frames in-between the main frequency
            output = np.array([])
            # pylint: disable=protected-access
            for k, t in tqdm(enumerate(times), total=len(times), desc="Encoding video"):
                ims = [_VideoImage(video=video, time=max(0, t - t2)) for t2 in subtimes]
                pil_imgs = [i.read() for i in ims]
                pil_imgs = self._resize_pil_images(pil_imgs)
                data = np.array([np.array(pi) for pi in pil_imgs])
                embd = self._embed_clip(data)
                if not output.size:
                    output = np.zeros((len(times),) + embd.shape)
                    logger.debug("Created Tensor with size %s", output.shape)
                output[k] = embd
            video.close()
            # set first (time) dim to last
            output = output.transpose(list(range(1, output.ndim)) + [0])
            yield nsbase.TimedArray(
                data=output.astype(np.float32),
                frequency=freq,
                start=nsbase._UNSET_START,
                duration=event.duration,
            )

    def _get_image_data(self, event: evts.Image) -> np.ndarray:
        pil_img = event.read()
        pil_imgs = [pil_img.copy() for _ in range(self.num_frames)]
        pil_imgs = self._resize_pil_images(pil_imgs)
        data = np.array([np.array(pi) for pi in pil_imgs])
        return self._embed_clip(data)

    def _resize_pil_images(self, pil_imgs: list[tp.Any]) -> list[tp.Any]:
        # resize if images are too big
        if pil_imgs and self.max_imsize is not None:
            factor = max(pil_imgs[0].size) / self.max_imsize
            if factor > 1:
                size = tuple(int(s / factor) for s in pil_imgs[0].size)
                pil_imgs = [pi.resize(size) for pi in pil_imgs]
        return pil_imgs

    def _embed_clip(self, data: np.ndarray) -> np.ndarray:
        t_embd = self._predict_hidden_states(data)
        if t_embd.shape[0] != 1:
            raise RuntimeError(f"Found several batches: {t_embd.shape}")
        t_embd = t_embd[0]  # aggregate_tokens works on non-batched-data
        embd = self._aggregate_tokens(t_embd).cpu().numpy()
        if self.cache_n_layers is None:
            embd = self._aggregate_layers(embd)
        return embd.astype(np.float32)

    def _predict_hidden_states(self, images: np.ndarray) -> torch.Tensor:
        kwargs: dict[str, tp.Any] = {
            self._processor_input_field(): list(images),
            "return_tensors": "pt",
        }
        inputs = self.processor(**kwargs)
        # prevent nans (happening for uniform images)
        image_extractors._fix_pixel_values(inputs)
        inputs = inputs.to(self.model_device)
        with torch.inference_mode():
            pred = self.model(**inputs, output_hidden_states=True)
        states = pred.hidden_states
        out = torch.cat([x.unsqueeze(1) for x in states], axis=1)  # type: ignore
        return out  # B x L x ...

    def _processor_input_field(self) -> tp.Literal["images", "videos"]:
        parameters = inspect.signature(self.processor.__call__).parameters
        return "videos" if "videos" in parameters else "images"

    def _config_num_frames(self) -> int | None:
        config = getattr(self.model, "config", None)
        if config is None:
            return None
        config = getattr(config, "vision_config", config)
        num_frames = getattr(config, "num_frames", None)
        return num_frames if isinstance(num_frames, int) else None

    def _warn_if_config_num_frames_mismatch(self) -> None:
        config_num_frames = self._config_num_frames()
        if config_num_frames is None or config_num_frames == self.num_frames:
            return
        warnings.warn(
            f"Model {self.model_name!r} config expects {config_num_frames} frames, "
            f"but HuggingFaceVideo.num_frames={self.num_frames}.",
            stacklevel=2,
        )
