# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import neuralset as ns
from neuralset.events import etypes

from . import video as vid
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
    assert "julius" in reqs, "Missing requirement coming from Extractor"
    assert "moviepy" in reqs, "Missing requirement coming from Event"


def test_video_image(video_event: etypes.Video) -> None:
    movie = video_event.read()
    vi = _VideoImage(video=movie, time=12345.12345)
    assert vi.filepath.endswith("random_video_6s.mp4:12345.123")


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


@pytest.mark.parametrize(
    "name,embds",
    [
        # shape is layers x tokens x embeddings
        ("MCG-NJU/videomae-base", (13, 1568, 768)),
        # ("facebook/vjepa2-vith-fpc64-256", (33, 8192, 1280)),
        # ("google/vivit-b-16x2-kinetics400", (13, 3137, 768)),
        # ("facebook/timesformer-base-finetuned-k600", (13, 1569, 768)),
    ],
)
def test_video_models(
    video_event: etypes.Video,
    tmp_path: Path,
    name: str,
    embds: tuple[int, ...],
) -> None:
    if "IN_GITHUB_ACTION" in os.environ and "videomae" not in name:
        pytest.skip("Only download video mae for CI tests")
    infra: tp.Any = {"folder": tmp_path / "cache"}
    video = vid.HuggingFaceVideo(
        frequency=0.5,
        max_imsize=120,
        infra=infra,
        num_frames=16,
        device="cpu",
        model_name=name,
        # show the full dimension
        token_aggregation=None,
        layers="all",
        layer_aggregation=None,
    )
    out = video(video_event, start=0.0, duration=4)
    assert isinstance(out, torch.Tensor)
    assert tuple(out.shape) == embds + (2,)


def test_video_huggingface() -> None:
    extractor = vid.HuggingFaceVideo(
        frequency=0.5,
        model_name="MCG-NJU/videomae-base",
        num_frames=16,
        device="cpu",
    )
    config = tp.cast(tp.Any, extractor.model.config)
    data = np.random.rand(config.num_frames, 3, 64, 64)
    out = extractor._predict_hidden_states(data)
    assert out.shape == (1, 13, 1568, 768)


# for future TEXT + VIDEO models?
# def test_multimodal(tmp_path: Path) -> None:
#     text_events = test_text._make_test_events()
#     video_event = make_video_event(folder=tmp_path, fps=24, duration=6)
#     events = pd.DataFrame(text_events.to_dict(orient="records") + [video_event.to_dict()])
#     events = segs.validate_events(events)
