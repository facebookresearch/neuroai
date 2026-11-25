# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import types
import typing as tp
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import PIL.Image
import pytest
import torch

import neuralset as ns
from neuralset.events import etypes

from . import video as vid
from .base import ContextWindow
from .video import _VideoImage, resamp_first_dim

logging.getLogger("neuralset").setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def video_event(
    tmp_path_factory: pytest.TempPathFactory,
) -> tp.Iterator[etypes.Video]:
    yield make_video_event(tmp_path_factory.mktemp("video"))


def make_video_event(
    folder: str | Path,
    fps: int = 4,
    width: int = 144,
    height: int = 128,
    duration: float = 6.0,
) -> etypes.Video:
    filepath = Path(folder) / "random_video_6s.mp4"
    filepath.parent.mkdir(exist_ok=True)
    import moviepy as mp

    num_frames = int(duration * fps)
    shape = (num_frames, height, width, 3)
    frames = np.random.randint(0, 256, shape, dtype=np.uint8)

    # Create a MoviePy video clip from the frames
    video_clip = mp.VideoClip(
        lambda t: frames[int(t * fps) % num_frames], duration=duration
    )
    # Plays the note A in stereo (two sine waves of frequencies 440 and 880 Hz)
    frame_function = lambda t: np.array(
        [np.sin(440 * 2 * np.pi * t), np.sin(880 * 2 * np.pi * t)]
    ).T.copy(order="C")
    audio_clip = mp.AudioClip(frame_function, duration=duration, fps=16000)
    video_clip = video_clip.with_audio(audio_clip)

    # Write file
    video_clip.write_videofile(
        str(filepath), fps=fps, codec="libx264", audio=True, audio_fps=16000
    )
    # make event
    event_dict = dict(type="Video", filepath=filepath, start=0, timeline="foo")
    event = etypes.Video.from_dict(event_dict)
    return event


def test_split_video(video_event: etypes.Video) -> None:
    duration = video_event.duration
    chunks = video_event._split([4.0])
    chunk1, chunk2 = chunks
    assert chunk1.offset == 0.0
    assert chunk2.offset == 4.0
    clip1, clip2 = chunk1.read(), chunk2.read()
    assert clip1.duration == 4
    assert clip2.duration == duration - 4


def test_resamp_first_dim() -> None:
    data = torch.rand(12, 7, 5)
    assert resamp_first_dim(data, 8).shape == (8, 7, 5)


def test_video_requirements() -> None:
    reqs = ",".join(ns.extractors.HuggingFaceVideo.requirements)
    assert "transformers" in reqs, "Missing requirement coming from Extractor"
    assert "moviepy" in reqs, "Missing requirement coming from Event"


def test_video_image(video_event: etypes.Video) -> None:
    movie = video_event.read()
    vi = _VideoImage(video=movie, time=12345.12345)
    assert vi.filepath.endswith("random_video_6s.mp4:12345.123")


def test_huggingface_video_static_image_clip(tmp_path: Path) -> None:
    if "IN_GITHUB_ACTION" in os.environ:
        pytest.skip("Run locally; image events use the full video model path.")
    filepath = tmp_path / "image.png"
    frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    PIL.Image.fromarray(frame).save(filepath)
    event = etypes.Image(start=1.0, duration=3.0, filepath=filepath, timeline="foo")
    infra: tp.Any = {"folder": tmp_path / "cache", "cluster": None}
    kwargs: tp.Any = dict(infra=infra, max_imsize=64, num_frames=16, device="cpu")
    extractor = vid.HuggingFaceVideo(
        event_types=("Image", "Video"),
        frequency=1.0,
        context=ContextWindow(duration=2.0),
        **kwargs,
    )
    out = extractor(event, start=1.0, duration=3.0)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (768, 3)
    assert torch.isfinite(out).all()
    # images alone need no context window, and can be static
    static = vid.HuggingFaceVideo(
        event_types="Image", frequency=0, context=ContextWindow(), **kwargs
    )
    torch.testing.assert_close(static(event, start=1.0, duration=3.0), out[:, 0])


def test_video(video_event: etypes.Video, tmp_path: Path) -> None:
    video_event.read()
    infra: tp.Any = {"folder": tmp_path / "cache"}
    image = ns.extractors.HuggingFaceImage(
        event_types="Video",
        frequency=0.5,
        infra=infra,
        device="cpu",
        layers=0.7,
    )
    folder = image.infra.uid_folder()
    assert folder is not None
    out = image(video_event, start=0.0, duration=0.5)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (768, 1)
    # test out
    df = pd.DataFrame([video_event.to_dict()])
    assert isinstance(df.loc[0, "filepath"], str)


def test_video_image_latent(video_event: etypes.Video, tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    name = "facebook/dinov2-small-imagenet1k-1-layer"
    infra: tp.Any = {"folder": cache}
    image = ns.extractors.HuggingFaceImage(
        event_types="Video",
        frequency=0.5,
        infra=infra,
        device="cpu",
        model_name=name,
    )
    out = image(video_event, start=0.0, duration=4)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (384, 2)
    latent = next(iter(image._get_data([video_event])))
    assert latent.shape == (384, 3)


def test_video_huggingface() -> None:
    extractor = vid.HuggingFaceVideo(
        frequency=4.0,
        context=ContextWindow(duration=2.0),
        model_name="MCG-NJU/videomae-base",
        num_frames=16,
        device="cpu",
    )
    config = tp.cast(tp.Any, extractor.model.config)
    assert extractor._local_model_config()["num_frames"] == config.num_frames == 16
    data = np.random.rand(config.num_frames, 3, 64, 64)
    hidden_states = extractor._predict_hidden_states(data)
    assert hidden_states.shape == (1, 13, 1568, 768)
    assert extractor._temporal_tokens(hidden_states[0]).shape == (13, 768, 8)


def _make_video(config: tp.Any, **kwargs: tp.Any) -> vid.HuggingFaceVideo:
    # fake model configs, whose num_frames differ from the checkpoint's
    with (
        mock.patch.object(vid.HuggingFaceVideo, "_download_huggingface_snapshot"),
        mock.patch.object(vid.HuggingFaceVideo, "_local_model_config", return_value={}),
    ):
        extractor = vid.HuggingFaceVideo(
            model_name="MCG-NJU/videomae-base", device="cpu", **kwargs
        )
    model = torch.nn.Identity()
    setattr(model, "config", config)
    extractor._model = model
    return extractor


@pytest.mark.parametrize(
    "model_type,tubelet_size,has_cls,space_major",
    [
        ("videomae", 2, False, False),
        ("vjepa2", 2, False, False),
        ("vivit", [2, 16, 16], True, False),
        ("timesformer", None, True, True),
    ],
)
def test_huggingface_video_token_layout(
    model_type: str, tubelet_size: tp.Any, has_cls: bool, space_major: bool
) -> None:
    config = types.SimpleNamespace(model_type=model_type)
    if tubelet_size is not None:
        config.tubelet_size = tubelet_size
    extractor = _make_video(
        config, num_frames=4, context={"duration": 1.0}, frequency=1.0
    )
    grid = torch.randn(extractor._n_temporal_tokens(), 3)  # time x space
    tokens = (grid.T if space_major else grid).flatten()
    if has_cls:
        tokens = torch.cat([torch.tensor([1e3]), tokens])
    out = extractor._temporal_tokens(tokens.reshape(1, -1, 1))
    torch.testing.assert_close(out[0, 0], grid.mean(dim=1))


class _VideoProcessor:
    def __call__(self, videos: list[np.ndarray], return_tensors: str) -> tp.Any:
        import transformers

        clip = torch.from_numpy(np.stack(videos))[None]  # batch of one clip
        return transformers.BatchFeature({"pixel_values_videos": clip})


class _VJEPA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, bool]] = []

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        skip_predictor: bool = False,
        output_hidden_states: bool = False,
    ) -> tp.Any:
        self.calls.append(
            {
                "skip_predictor": skip_predictor,
                "output_hidden_states": output_hidden_states,
            }
        )
        return types.SimpleNamespace(hidden_states=(pixel_values_videos,) * 2)


class _VideoMAE(torch.nn.Module):
    def forward(
        self, pixel_values_videos: torch.Tensor, output_hidden_states: bool = False
    ) -> tp.Any:
        return types.SimpleNamespace(hidden_states=(pixel_values_videos,) * 3)


def test_huggingface_video_skip_predictor() -> None:
    config = types.SimpleNamespace(model_type="vjepa2", tubelet_size=2)
    extractor = _make_video(
        config, num_frames=4, context={"duration": 1.0}, frequency=1.0
    )
    frames = np.zeros((4, 2, 2, 3))
    extractor._processor = _VideoProcessor()
    vjepa = _VJEPA()
    extractor._model = vjepa
    assert extractor._predict_hidden_states(frames).shape == (1, 2, 4, 2, 2, 3)
    # V-JEPA models run their encoder only
    assert vjepa.calls == [{"skip_predictor": True, "output_hidden_states": True}]
    extractor._model = _VideoMAE()  # no predictor
    assert extractor._predict_hidden_states(frames).shape == (1, 3, 4, 2, 2, 3)


def test_huggingface_video_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = make_video_event(tmp_path, fps=4, duration=5.0)
    n_frames = []

    def predict(self: vid.HuggingFaceVideo, images: np.ndarray) -> torch.Tensor:
        n_frames.append(len(images))
        return torch.randn(1, 2, 8, 3)  # batch, layers, 2 times x 4 patches, dim

    monkeypatch.setattr(vid.HuggingFaceVideo, "_predict_hidden_states", predict)

    def extract(frequency: float, **context: tp.Any) -> torch.Tensor:
        extractor = _make_video(
            types.SimpleNamespace(model_type="videomae", tubelet_size=2),
            num_frames=4,
            context={"duration": 2.0, **context},  # 2 tokens per window: 1 Hz
            frequency=frequency,
            infra={"folder": tmp_path / "cache"},
        )
        return extractor(event, start=0.0, duration=event.duration)

    assert extract(1.0, overlap=0.2).shape == (3, 5)
    # overlap=0.2 rounds to 0 of the 2 tokens: 2s windows starting at 0, 2 and 3s
    # (ending at the event end)
    assert n_frames == [4, 4, 4]
    assert extract(0.4, overlap=0.2).shape == (3, 2)
    assert extract(1.0).shape == (3, 5)
    assert len(n_frames) == 3 + 3 + 4  # default overlap=0.5: windows every second
    assert extract(1.0, min_context_duration=0.5).shape == (3, 5)
    assert len(n_frames) == 3 + 3 + 4  # overlap=0.5: same cache
    assert extract(1.0, causal=True).shape == (3, 5)
    assert len(n_frames) == 3 + 3 + 4 + 4
    with pytest.raises(ValueError, match="native frequency"):
        extract(2.0)


def test_huggingface_video_previous_api() -> None:
    kwargs: tp.Any = dict(model_name="MCG-NJU/videomae-base", num_frames=16)
    with pytest.raises(ValueError, match="overlap"):
        vid.HuggingFaceVideo.model_validate(
            dict(frequency=1.0, clip_duration=1.0, **kwargs)
        )
    with mock.patch.object(vid.HuggingFaceVideo, "_download_huggingface_snapshot"):
        extractor = vid.HuggingFaceVideo(frequency=1.0, **kwargs)
        assert extractor.context == ContextWindow(duration=8.0, overlap=0.5)
        # fields missing from a context input keep the default's
        extractor = vid.HuggingFaceVideo.model_validate(
            dict(frequency=1.0, context={"overlap": 0.2}, **kwargs)
        )
        assert extractor.context == ContextWindow(duration=8.0, overlap=0.2)
        with pytest.raises(ValueError, match="context.duration must be set"):
            vid.HuggingFaceVideo.model_validate(
                dict(frequency=1.0, context={"duration": None}, **kwargs)
            )
        # num_frames must match the config of models with fixed position embeddings
        config = mock.patch.object(vid.HuggingFaceVideo, "_local_model_config")
        with config as read:
            read.return_value = {"num_frames": 16}
            with pytest.raises(ValueError, match="takes 16 frames per clip"):
                vid.HuggingFaceVideo(frequency=1.0, **{**kwargs, "num_frames": 8})
            # None reads the model's own: num_frames, else V-JEPA 2's frames_per_clip
            assert vid.HuggingFaceVideo(frequency=1.0).num_frames == 16
            read.return_value = {"frames_per_clip": 64, "tubelet_size": 2}
            assert vid.HuggingFaceVideo(frequency=1.0).num_frames == 64
            read.return_value = {}
            with pytest.raises(ValueError, match="num_frames must be set"):
                vid.HuggingFaceVideo(frequency=1.0)
        with pytest.raises(ValueError, match="HuggingFaceImage"):
            vid.HuggingFaceVideo(
                frequency=1.0,
                num_frames=16,
                context=ContextWindow(duration=2.0),
                model_name="facebook/dinov2-small",
            )


# for future TEXT + VIDEO models?
# def test_multimodal(tmp_path: Path) -> None:
#     text_events = test_text._make_test_events()
#     video_event = make_video_event(folder=tmp_path, fps=24, duration=6)
#     events = pd.DataFrame(text_events.to_dict(orient="records") + [video_event.to_dict()])
#     events = segs.validate_events(events)
