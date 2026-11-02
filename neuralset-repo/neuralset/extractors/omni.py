# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import typing as tp

import numpy as np
import pydantic
import torch
from exca import MapInfra
from torch.nn import functional
from tqdm import tqdm

from neuralset import base as nsbase
from neuralset import utils
from neuralset.events import etypes as evts

from . import hf
from .base import BaseExtractor, ContextWindow, _ContextMixin
from .image import _VideoImage

_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text "
    "and speech."
)


def _round(x: float) -> int:
    """Round half up (``round`` rounds half to even, which drops or duplicates
    samples at boundaries falling exactly between two of them)."""
    return math.floor(x + 0.5)


def _windows(
    duration: float, context: float, stride: float, token_rate: float, causal: bool
) -> list[tuple[float, float, float, float]]:
    """(window_start, window_stop, keep_start, keep_stop) in seconds:
    ``strided_windows`` with windows starting on a grid of ``gcd(context, stride)``
    seconds, so that they start on the token grids (both durations are multiples
    of the token durations, e.g. ``1 / token_rate``). The last window is cut at
    the end of the event."""
    context_tokens, stride_tokens = (_round(d * token_rate) for d in (context, stride))
    unit_tokens = math.gcd(context_tokens, stride_tokens)
    # windows of whole units, but bidirectional splits in the middle of overlaps
    half_unit = unit_tokens / token_rate / 2
    n_units = max(1, math.ceil(duration / (2 * half_unit)))
    context_halves = 2 * context_tokens // unit_tokens
    windows = utils.strided_windows(
        2 * n_units, context_halves, 2 * stride_tokens // unit_tokens, causal=causal
    )
    return [
        (
            start * half_unit,
            min((start + context_halves) * half_unit, duration),
            keep_start * half_unit,
            min(keep_stop * half_unit, duration),
        )
        for start, keep_start, keep_stop in windows
    ]


def _media_tokens(
    states: torch.Tensor, input_ids: torch.Tensor, token_id: int, n_times: int
) -> torch.Tensor:
    """(layers, time, tokens per time step, dim) states of the tokens of one
    modality, from the (layers, sequence, dim) states of the whole sequence.
    Qwen2.5-Omni interleaves video and audio by chunks of time, but keeps each
    modality in temporal order, and video tokens in (time, height, width) order."""
    tokens = states[:, input_ids == token_id]
    return tokens.reshape(len(tokens), n_times, -1, tokens.shape[-1])


def _event_uid(event: evts.Video | evts.Audio | evts.Image) -> str:
    """Cache key of an event: images are keyed by file, as they are static."""
    if isinstance(event, evts.Image):
        return str(event.study_relative_path())
    return event._splittable_event_uid()


def _resample(latents: np.ndarray, n_times: int) -> np.ndarray:
    """Resample the last (time) axis: average below the native rate, repeat above."""
    if latents.shape[-1] == n_times:
        return latents
    mode = "area" if n_times < latents.shape[-1] else "nearest"
    flat = torch.from_numpy(latents).reshape(1, -1, latents.shape[-1])
    out = functional.interpolate(flat, size=n_times, mode=mode)
    return out.reshape(*latents.shape[:-1], n_times).numpy()


class QwenOmni(BaseExtractor, hf.HuggingFaceMixin, _ContextMixin):
    """Time-resolved audio-visual embeddings from the Qwen2.5-Omni or
    Qwen3-Omni thinker, for Video, Audio and Image events.

    The frames and soundtrack of each video are fed jointly to the model, which
    interleaves video and audio tokens in time with time-aligned rotary
    embeddings. The hidden states of the video tokens (averaged over space with
    ``token_aggregation``, one step every ``2 / fps`` seconds) and of the audio
    tokens (25 per second for Qwen2.5-Omni, 13 for Qwen3-Omni) are resampled to
    ``frequency`` and cached as a (2, dim, time) array (video, audio).

    Audio events are fed alone and cached as a (1, dim, time) array (audio), so
    that a single extractor covers studies with and without videos. Likewise,
    the frames of videos without soundtrack are fed alone, and their audio
    latents cached as NaNs. Image events are fed as a static video step (the
    image repeated over two frames, as in ``HuggingFaceVideo``), and their video
    latents cached as a (1, dim) array, returned as static over the image
    duration.

    Outputs always have the shape of those of videos with sound: zeros stand
    for a missing modality, so that its contribution is the same with or
    without the other one (e.g. ``modality="sum"`` returns the audio latents
    of Audio events). On studies with videos, don't
    also extract their soundtrack as Audio events (e.g. with
    ``ExtractAudioFromVideo``): segments holding both event types raise an
    error.

    Long events are embedded in windows (``context``, see ``ContextWindow``). As
    the thinker is causal, each window keeps by default the latents following the
    previous window, so that every latent sees at least ``context.min_context``
    seconds of past (less at the start of the event). Layer 0 holds the outputs of
    the separate audio and vision encoders, deeper layers the fused representations.

    Parameters
    ----------
    fps :
        Frames per second fed to the model (2 in Qwen's reference pipeline).
        Two consecutive frames make one video time step.
    max_pixels :
        Maximum number of pixels per frame or image after resizing (one token per
        28 x 28 pixels for Qwen2.5-Omni, 32 x 32 for Qwen3-Omni).
    context :
        Windows embedded at once. ``context.duration`` must be a multiple of the
        video time step (``2 / fps``) and of the audio one. The stride between
        windows is rounded to whole video time steps, and must also be a multiple
        of the audio time step. The default gives 30 s windows every 15 s, so every
        latent sees at least 15 s of past.
    modality :
        How the cached video and audio latents are returned: summed
        (``"sum"``, default), concatenated along the feature axis (``"cat"``,
        video first) or only one of them. Not part of the cache uid.
    """

    event_types: str | tuple[str, ...] = ("Video", "Audio", "Image")
    requirements: tp.ClassVar[tuple[str, ...]] = (
        "transformers>=5.20",
        "julius>=0.2.7",
        "pillow>=9.2.0",
    )
    model_name: str = "Qwen/Qwen2.5-Omni-7B"
    hf_config: hf.HuggingFaceConfig = hf.HuggingFaceConfig(
        model_cls_name="Qwen2_5OmniThinkerForConditionalGeneration",
    )
    dtype: tp.Literal["auto", "float16", "float32", "float64", "bfloat16"] | None = (
        "bfloat16"
    )
    token_aggregation: tp.Literal["mean", "sum", "max"] = "mean"
    frequency: pydantic.PositiveFloat = 2.0
    fps: pydantic.PositiveFloat = 2.0
    max_pixels: pydantic.PositiveInt = 256 * 28 * 28
    context: ContextWindow = ContextWindow(duration=30.0, causal=True)
    modality: tp.Literal["sum", "cat", "audio", "video"] = "sum"
    infra: MapInfra = MapInfra(
        timeout_min=120,
        gpus_per_node=1,
        cpus_per_task=8,
        min_samples_per_job=1,
        version="v2",
    )

    _SAMPLING_RATE: tp.ClassVar[int] = 16_000
    # thinker class and audio tokens per second of each model family
    _FAMILIES: tp.ClassVar[dict[str, tuple[str, float]]] = {
        "Qwen/Qwen2.5-Omni-": ("Qwen2_5OmniThinkerForConditionalGeneration", 25.0),
        "Qwen/Qwen3-Omni-": ("Qwen3OmniMoeThinkerForConditionalGeneration", 13.0),
    }
    _TEMPORAL_PATCH_SIZE: tp.ClassVar[int] = 2  # frames per video time step

    def model_post_init(self, log__: tp.Any) -> None:
        if "hf_config" not in self.model_fields_set:
            self.hf_config = hf.HuggingFaceConfig(model_cls_name=self._family[0])
        super().model_post_init(log__)
        event_types = (
            {self.event_types}
            if isinstance(self.event_types, str)
            else set(self.event_types)
        )
        if not event_types <= {"Video", "Audio", "Image"}:
            msg = f"{self.name}.event_types must be among Video, Audio and Image "
            raise ValueError(msg + f"(got {self.event_types})")
        if self.context.duration is None:
            raise ValueError(f"{self.name}.context.duration must be set")
        video_rate = self.fps / self._TEMPORAL_PATCH_SIZE
        stride = self._stride_duration
        durations = {
            f"context.duration={self.context.duration}": self.context.duration,
            f"context.overlap={self.context.overlap} (stride {stride} s)": stride,
        }
        for desc, value in durations.items():
            for rate in (video_rate, self._family[1]):
                if not math.isclose(value * rate, round(value * rate)):
                    msg = f"{self.name}.{desc} must give a multiple of the "
                    msg += f"token duration 1 / {rate} s"
                    raise ValueError(msg)

    @property
    def _duration(self) -> float:
        duration = self.context.duration
        assert duration is not None  # model_post_init
        return duration

    @property
    def _stride_duration(self) -> float:
        """Duration (s) between window starts, in whole video time steps."""
        step = self._TEMPORAL_PATCH_SIZE / self.fps
        n_steps = _round((1 - self.context.overlap) * self._duration / step)
        return max(1, n_steps) * step

    @property
    def _family(self) -> tuple[str, float]:
        for prefix, family in self._FAMILIES.items():
            if self.model_name.startswith(prefix):
                return family
        msg = f"{self.name}.model_name must start with one of {list(self._FAMILIES)}"
        raise ValueError(msg + f" (got {self.model_name!r})")

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return hf.HuggingFaceMixin._exclude_from_cls_uid()

    def _exclude_from_cache_uid(self) -> list[str]:
        return [
            *BaseExtractor._exclude_from_cache_uid(self),
            *hf.HuggingFaceMixin._exclude_from_cache_uid(self),
            "modality",
        ]

    def load_model(self) -> torch.nn.Module:
        model = super().load_model()
        # logits over the whole sequence would be computed and never used
        model.lm_head = torch.nn.Identity()  # type: ignore[assignment]
        return model

    def _get_timed_arrays(
        self,
        events: list[evts.Video | evts.Audio | evts.Image],
        start: float,
        duration: float,
    ) -> tp.Iterable[nsbase.TimedArray]:
        has_video = any(isinstance(e, evts.Video) for e in events)
        if has_video and any(isinstance(e, evts.Audio) for e in events):
            msg = f"{self.name} got both Video and Audio events in a segment: don't "
            msg += "extract the soundtrack of the videos as Audio events (e.g. with "
            msg += "ExtractAudioFromVideo), or restrict event_types"
            raise ValueError(msg)
        for event, cached in zip(events, self._get_data(events)):
            if isinstance(event, evts.Image):
                data = np.asarray(cached)
                if self.cache_n_layers is not None:
                    data = self._aggregate_layers(data)
                yield nsbase.TimedArray(
                    data=self._select_modality(data[..., 0, :], None, axis=-1),
                    frequency=0,
                    start=event.start,
                    duration=event.duration,
                )
                continue
            assert isinstance(cached, nsbase.TimedArray)
            sub = cached.copy(start=event.start).overlap(start=start, duration=duration)
            data = np.asarray(sub.data)
            if self.cache_n_layers is not None:
                data = self._aggregate_layers(data)
            # Audio events have no video latents, silent videos NaN audio latents
            video = None if data.shape[-3] == 1 else data[..., 0, :, :]
            audio = data[..., -1, :, :]
            silent = video is not None and audio.size and np.isnan(audio.flat[0])
            sub.data = self._select_modality(video, None if silent else audio, axis=-2)
            yield sub

    def _select_modality(
        self, video: np.ndarray | None, audio: np.ndarray | None, axis: int
    ) -> np.ndarray:
        """Latents returned for ``modality``, from those of each modality (None
        if missing, i.e. zeros), whose feature axis is ``axis``."""
        zeros = np.zeros_like(audio if video is None else video)
        video = zeros if video is None else video
        audio = zeros if audio is None else audio
        match self.modality:
            case "cat":
                return np.concatenate([video, audio], axis=axis)
            case "sum":
                return video + audio
            case "video":
                return video
            case "audio":
                return audio

    @infra.apply(
        item_uid=_event_uid,
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_data(
        self, events: list[evts.Video | evts.Audio | evts.Image]
    ) -> tp.Iterator[nsbase.TimedArray | np.ndarray]:
        video_rate = self.fps / self._TEMPORAL_PATCH_SIZE
        for event in events:
            if isinstance(event, evts.Image):
                image = np.asarray(event.read().convert("RGB"))
                static = np.stack([image] * self._TEMPORAL_PATCH_SIZE)
                (states,) = self._embed_window(static, None)  # (layers, dim, 1)
                if self.cache_n_layers is None:
                    states = self._aggregate_layers(states)
                yield states[..., None, :, 0]  # (..., 1, dim)
                continue
            video: tp.Any = None
            wav: np.ndarray | None = None
            if isinstance(event, evts.Audio):
                wav = self._mono_wav(event.read(), event.frequency, event.duration)
            else:
                video = event.read()
                if video.audio is not None:
                    wav = self._read_wav(video)
            rates: tuple[float, ...] = (video_rate,) if video is not None else ()
            rates += (self._family[1],) if wav is not None else ()
            parts: list[list[np.ndarray]] = [[] for _ in rates]
            frames: dict[int, np.ndarray] = {}
            windows = _windows(
                event.duration,
                self._duration,
                self._stride_duration,
                self._family[1],
                self.context.causal,
            )
            desc = f"Encoding {event.__class__.__name__.lower()}"
            for start, stop, keep_start, keep_stop in tqdm(windows, desc=desc):
                samples = slice(*(_round(t * self._SAMPLING_RATE) for t in (start, stop)))
                wav_window = None if wav is None else wav[samples]
                if video is None:
                    latents = self._embed_window(None, wav_window)
                else:
                    # frames of overlapping windows are read once
                    first = _round(start * self.fps)
                    n_steps = math.ceil((stop - start) * video_rate)
                    indices = range(first, first + n_steps * self._TEMPORAL_PATCH_SIZE)
                    frames = {i: f for i, f in frames.items() if i >= first}
                    for i in indices:
                        if i not in frames:
                            frames[i] = self._read_frame(video, (i + 0.5) / self.fps)
                    latents = self._embed_window(
                        np.stack([frames[i] for i in indices]), wav_window
                    )
                for modality_parts, states, rate in zip(parts, latents, rates):
                    crop = slice(
                        *(_round((t - start) * rate) for t in (keep_start, keep_stop))
                    )
                    states = states[..., crop]
                    if self.cache_n_layers is None:
                        states = self._aggregate_layers(states)
                    modality_parts.append(states)
            if video is not None:
                video.close()
            n_times = max(1, nsbase.Frequency(self.frequency).to_ind(event.duration))
            resampled = [_resample(np.concatenate(p, axis=-1), n_times) for p in parts]
            if wav is None:  # silent video
                resampled.append(np.full_like(resampled[0], np.nan))
            data = np.stack(resampled, axis=-3)
            yield nsbase.TimedArray(
                data=data,
                frequency=self.frequency,
                start=nsbase._UNSET_START,
                duration=event.duration,
            )

    @staticmethod
    def _read_frame(video: tp.Any, time: float) -> np.ndarray:
        time = min(time, np.nextafter(video.duration, 0))
        return np.asarray(_VideoImage(video=video, time=time).read())

    def _read_wav(self, video: tp.Any) -> np.ndarray:
        """Mono soundtrack of a video at 16 kHz."""
        wav = torch.tensor(video.audio.to_soundarray(), dtype=torch.float32)
        return self._mono_wav(wav, video.audio.fps, video.duration)

    def _mono_wav(self, wav: torch.Tensor, sfreq: float, duration: float) -> np.ndarray:
        """Mono 16 kHz version of a (time, channels) waveform, padded or cropped
        to ``duration``."""
        import julius

        n_samples = _round(duration * self._SAMPLING_RATE)
        wav = wav.float().reshape(len(wav), -1).mean(dim=1)
        if int(sfreq) != self._SAMPLING_RATE:
            resample = julius.resample.ResampleFrac(
                old_sr=int(sfreq), new_sr=self._SAMPLING_RATE
            )
            wav = resample(wav)
        wav_np = wav.numpy()[:n_samples]
        return np.pad(wav_np, (0, n_samples - len(wav_np)))

    def _prompt(self, media: tp.Literal["video", "audio"]) -> str:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": media}]},
        ]
        return self.processor.apply_chat_template(conversation, tokenize=False)

    def _embed_window(
        self, frames: np.ndarray | None, wav: np.ndarray | None
    ) -> tuple[np.ndarray, ...]:
        """(layers, dim, time) states of the video tokens (if ``frames`` is given)
        and of the audio tokens (if ``wav`` is given) of a window."""
        with_video, with_audio = frames is not None, wav is not None
        media_inputs: dict[str, tp.Any] = {}
        if wav is not None:
            media_inputs["audio"] = [wav]
            # the feature extractor pads to 300 s by default
            media_inputs["audio_kwargs"] = {"max_length": len(wav), "truncation": False}
        if with_video:
            media_inputs["videos"] = [frames]
            media_inputs["videos_kwargs"] = {
                "fps": self.fps,
                "do_sample_frames": False,
                "use_audio_in_video": with_audio,
                "size": {
                    "shortest_edge": min(128 * 28 * 28, self.max_pixels),
                    "longest_edge": self.max_pixels,
                },
                "cap_pixels_per_frame": False,
            }
        inputs = self.processor(
            text=self._prompt("video" if with_video else "audio"),
            return_tensors="pt",
            **media_inputs,
        ).to(self.model_device)
        with torch.inference_mode():
            out = self.model(
                **inputs,
                use_audio_in_video=with_video and with_audio,
                output_hidden_states=True,
                use_cache=False,
            )
        states = self._layer_subselection(torch.stack(out.hidden_states)[:, 0])
        input_ids = inputs["input_ids"][0]
        config: tp.Any = self.model.config
        media: list[tuple[int, int]] = []
        if with_video:
            n_video = int(inputs["video_grid_thw"][0, 0])
            media.append((config.video_token_id, n_video))
        if with_audio:
            n_audio = int((input_ids == config.audio_token_id).sum())
            media.append((config.audio_token_id, n_audio))
        latents = []
        for token_id, n_times in media:
            tokens = _media_tokens(states, input_ids, token_id, n_times)
            # aggregates the tokens of each time step: (layers, time, dim)
            latents.append(self._aggregate_tokens(tokens.transpose(1, 2)))
        return tuple(x.transpose(1, 2).cpu().numpy() for x in latents)  # type: ignore
