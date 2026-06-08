# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import numpy as np
import pydantic
import torch
from exca import MapInfra
from tqdm import tqdm

from neuralset import base as nsbase
from neuralset.events import etypes as evts

from . import base as extractor_base
from . import image as image_extractors

logger = logging.getLogger(__name__)
# activate with:
# logging.getLogger("neuralset").setLevel(logging.DEBUG)

_VideoImage = image_extractors._VideoImage


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


class HuggingFaceVideo(extractor_base.BaseExtractor, extractor_base.HuggingFaceMixin):
    """Extract video embeddings using a native HuggingFace video model.

    Videos are divided into clips of `clip_duration` seconds at the specified
    frequency. Each clip is processed by the video model, and features are
    aggregated over layers/tokens using the HuggingFace extractor options.

    Parameters
    ----------
    model_name : str, default="MCG-NJU/videomae-base"
        HuggingFace video model identifier.
        Image models are not accepted here; use `HuggingFaceImage` for
        frame-by-frame video embeddings.
    pretrained : bool, default=True
        Whether to load pretrained weights from model. If False, initializes
        the model with random weights from the model configuration.
    use_audio : bool, default=True
        Whether to include audio alongside video frames during feature extraction.
        Only applicable for models that support multimodal inputs (e.g., LLaVA-Video).
    clip_duration : float | None, default=None
        Duration (in seconds) of video sub-clips to process. If None, defaults to
        one timestep (1 / frequency).
    max_imsize : int | None, default=None
        Maximum image dimension for downsampling before processing. Useful for
        memory-constrained scenarios. For example, Phi-4 downsizes to 448×448
        before tokenization.
    layer_type : str, default=""
        Specific layer extraction mode for certain models.
        For XClip: Use "mit" to extract from Multi-frame Integration Transformer
        layers instead of vision backbone layers.
        For LLaVA models: Must be a prompt string containing the ``<video>`` token
        (e.g., ``"<|user|><video><|end|><|assistant|>"``).

        .. note:: The pipe characters in the example are literal LLaVA tokens.
    num_frames : int | None, default=None
        Number of frames to pass to the video model per clip. If None, uses the
        model's default frame count (e.g., 16 for VideoMAE, 8 for XClip, 64 for VJepa2).
    """

    event_types: tp.Literal["Video"] = "Video"
    # class attributes
    requirements: tp.ClassVar[tuple[str, ...]] = (
        "torchvision>=0.15.2",
        "julius>=0.2.7",
    )
    model_name: str = "MCG-NJU/videomae-base"
    pretrained: bool = True
    use_audio: bool = True
    clip_duration: float | None = None
    max_imsize: int | None = None
    layer_type: str = ""
    num_frames: int | None = None
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
                "HuggingFaceVideo(model_name=...)."
            )
            raise ValueError(msg)
        return data

    @pydantic.field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, model_name: str) -> str:
        if any(z in model_name for z in _HFVideoModel.MODELS):
            return model_name
        msg = (
            "The HuggingFaceVideo API now only supports native video models. "
            "For the previous frame-by-frame API, instantiate HuggingFaceImage "
            f"with event_types='Video' instead of using model_name={model_name!r}."
        )
        raise ValueError(msg)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        _HFVideoModel.check_layer_type(
            layer_type=self.layer_type, model_name=self.model_name
        )

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return extractor_base.HuggingFaceMixin._exclude_from_cls_uid()

    def _exclude_from_cache_uid(self) -> list[str]:
        return extractor_base.BaseExtractor._exclude_from_cache_uid(
            self
        ) + extractor_base.HuggingFaceMixin._exclude_from_cache_uid(self)

    def _get_timed_arrays(
        self, events: list[evts.Video], start: float, duration: float
    ) -> tp.Iterable[nsbase.TimedArray]:
        for event, ta in zip(events, self._get_data(events)):
            sub = ta.with_start(event.start).overlap(start=start, duration=duration)
            if self.cache_n_layers is not None:
                sub.data = self._aggregate_layers(sub.data)
            yield sub

    @infra.apply(
        item_uid=lambda e: e._splittable_event_uid(),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_data(self, events: list[evts.Video]) -> tp.Iterator[nsbase.TimedArray]:
        # read all videos of the events
        logging.getLogger("neuralset").setLevel(logging.DEBUG)
        model = _HFVideoModel(
            model_name=self.model_name,
            pretrained=self.pretrained,
            layer_type=self.layer_type,
            num_frames=self.num_frames,
        )
        if model.model.device.type == "cpu":
            # may already be dispatched (with "accelerate")
            model.model.to(self.device)
        # videomae = 16 frames
        # xclip = 8 or 16 frames (unclear)
        freq = events[0].frequency if self.frequency == "native" else self.frequency
        T = 1 / freq if self.clip_duration is None else self.clip_duration
        subtimes = list(
            k / model.num_frames * T for k in reversed(range(model.num_frames))
        )  # type: ignore
        for event in events:
            video = event.read()
            audio = video.audio if self.use_audio else None

            freq = self.frequency if self.frequency != "native" else event.frequency
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
                audio_clip = (
                    audio.subclipped(max(0, t - T), t) if audio is not None else None
                )
                pil_imgs = [i.read() for i in ims]
                # resize if images are too big
                if pil_imgs and self.max_imsize is not None:
                    factor = max(pil_imgs[0].size) / self.max_imsize
                    if factor > 1:
                        size = tuple(int(s / factor) for s in pil_imgs[0].size)
                        pil_imgs = [pi.resize(size) for pi in pil_imgs]
                data = np.array([np.array(pi) for pi in pil_imgs])
                t_embd = model.predict_hidden_states(data, audio_clip)
                if t_embd.shape[0] != 1:
                    raise RuntimeError(f"Found several batches: {t_embd.shape}")
                t_embd = t_embd[0]  # aggregate_tokens works on non-batched-data
                embd = self._aggregate_tokens(t_embd).cpu().numpy()
                if self.cache_n_layers is None:
                    embd = self._aggregate_layers(embd)
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


class _HFVideoModel:
    """Wrapper that provides a unified interface for loading and using various HuggingFace
    video models

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
        The model will be loaded from the HuggingFace Hub. Please note that you may have to install additional dependencies to load it correctly.
    pretrained : bool, default=True
        Whether to load pretrained weights. If False, initializes the model with
        random weights from the model configuration.
    layer_type: str, default=""
        Specific layer extraction mode for certain models:
        - For XClip: Use "mit" to extract from Multi-frame Integration Transformer
          layers instead of vision backbone layers.
        - For LLaVA models: Must be a prompt string containing the "<video>" token
          (e.g., "<|user|><video><|end|><|assistant|>").
    num_frames : int | None, default=None
        Number of frames to pass to the video model per clip. If None, uses the
        model's default frame count (e.g., 16 for VideoMAE, 8 for XClip, 64 for VJepa2).
    """

    MODELS = (
        "vjepa2",
        "videomae",
        "microsoft/xclip",
        "google/vivit",
        "facebook/timesformer",
        "LLaVA-NeXT-Video",
        "LLaVA-Video",
        "Phi-4",
    )
    # language + video models https://arxiv.org/pdf/2405.21075

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        layer_type: str = "",
        num_frames: int | None = None,
    ) -> None:
        super().__init__()
        if not any(z in model_name for z in self.MODELS):
            raise ValueError(f"Model {model_name!r} is not supported")
        Model: tp.Any  # ignore typing as we'll override the imports
        Processor: tp.Any
        from transformers import AutoModel as Model
        from transformers import AutoProcessor as Processor

        extra: dict[str, tp.Any] = {}
        processor_extra: dict[str, tp.Any] = {"do_rescale": True}
        if "google/vivit" in model_name:
            from transformers import VivitImageProcessor as Processor
            from transformers import VivitModel as Model  # type: ignore
        if "LLaVA" in model_name:
            from transformers import LlavaNextVideoForConditionalGeneration as Model
            from transformers import LlavaNextVideoProcessor as Processor

            extra = {"torch_dtype": torch.float16}
            if "34B" in model_name:
                extra["device_map"] = "auto"  # uses accelerate
        if "Phi-4" in model_name:
            from transformers import AutoModelForCausalLM as Model

            extra = {"_attn_implementation": "eager", "trust_remote_code": True}
            processor_extra["trust_remote_code"] = True
        if "vjepa2" in model_name:
            from transformers import AutoVideoProcessor as Processor

        self.model = Model.from_pretrained(model_name, output_hidden_states=True, **extra)
        if not pretrained:
            self.model = Model.from_config(self.model.config)
        self.model.eval()
        # use do_rescale=True -> don't use totensor
        self.processor = Processor.from_pretrained(model_name, **processor_extra)
        self.model_name = model_name
        self.layer_type = layer_type
        if "llava" in model_name.lower():
            max_frames = 16  # any number works
        elif "vjepa2" in model_name.lower():
            max_frames = 64
        elif "Phi-4" in model_name:
            max_frames = 4  # TODO: make this flexible?
        else:
            config = self.model.config
            config = getattr(config, "vision_config", config)  # xclip
            max_frames = config.num_frames
        if num_frames is None:
            self.num_frames = max_frames
        else:
            self.num_frames = num_frames
        if self.num_frames > max_frames:
            raise ValueError(
                f"{model_name} only seems to supports {max_frames} frames, got {self.num_frames}"
            )
        self.check_layer_type(layer_type, model_name)

    @staticmethod
    def check_layer_type(layer_type: str, model_name: str) -> None:
        if "xclip" in model_name and layer_type == "mit":
            return  # is ok
        if "llava" in model_name.lower():
            if "<video>" not in layer_type:
                msg = f"For {model_name!r}, layer_type must be a prompt with the <video> token\n"
                # note: best aggregation was: mean
                raise ValueError(msg)
            return  # all good
        if layer_type:
            raise ValueError(f"No layer type available for {model_name!r}")

    def predict(self, images: np.ndarray, audio: tp.Any | None = None) -> tp.Any:
        kwargs: dict[str, tp.Any] = {"text": "", "return_tensors": "pt"}
        field = "images"
        if "xclip" in self.model_name:
            field = "videos"
        elif "llava" in self.model_name.lower():
            field = "videos"
            kwargs["text"] = self.layer_type
        elif "vjepa2" in self.model_name:
            field = "videos"
            del kwargs["text"]
        elif "Phi-4" in self.model_name:
            import PIL

            images = [PIL.Image.fromarray(img) for img in images]  # type: ignore
            field = "images"
            prompt = "<|user|>"
            for i in range(1, len(images) + 1):
                prompt += f"<|image_{i}|>"
            if audio is not None:
                kwargs["audios"] = [(audio.to_soundarray(), audio.fps)]  # type: ignore
                prompt += "<|audio_1|>"
            prompt += "<|end|><|assistant|>"
            kwargs["text"] = prompt
        kwargs[field] = list(images)
        inputs = self.processor(**kwargs)
        # prevent nans (happening for uniform images)
        image_extractors._fix_pixel_values(inputs)
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            pred = self.model(**inputs)
        return pred

    def predict_hidden_states(
        self, images: np.ndarray, audio: np.ndarray | None = None
    ) -> torch.Tensor:
        pred = self.predict(images, audio)
        if "xclip" in self.model_name:
            # MIT: Multi-frame Integration Transformer
            is_mit = self.layer_type == "mit"
            pred = pred.mit_output if is_mit else pred.vision_model_output
            # [8, 13, 197, 768] for vision model, [1, 2, 8, 512] for mit model
        states = pred.hidden_states
        out = torch.cat([x.unsqueeze(1) for x in states], axis=1)  # type: ignore
        if "xclip" in self.model_name and not self.layer_type:
            out = out[[-1], ...]  # last batch/timepoint only
        return out  # B x L x ...
