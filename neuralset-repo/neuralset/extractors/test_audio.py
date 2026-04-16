# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import os
import typing as tp
from pathlib import Path

import numpy as np
import pytest
import scipy

from neuralset.events import etypes

from . import audio


def create_wav(
    fp: Path, fs: int = 44100, duration: float = 10, channels: int | None = None
) -> None:
    y = np.random.randn(int(duration * fs))
    if channels is not None:
        y = np.array([y for _ in range(channels)]).T
    scipy.io.wavfile.write(fp, fs, y)


def test_split_audio(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    sr = 44100
    create_wav(fp, fs=sr, duration=10)
    event = etypes.Audio(start=1, timeline="whatever", filepath=fp)
    split1 = event._split([4])
    assert split1[0].start == 1.0 and split1[0].offset == 0.0
    assert split1[1].start == 5.0 and split1[1].offset == 4.0
    split2 = split1[1]._split([2, 3])
    assert len(split2) == 3
    assert split2[1].start == 7.0 and split2[1].offset == 6.0

    chunk1, chunk2 = split1
    assert chunk1.offset == 0.0
    assert chunk2.offset == 4.0
    wav1, wav2 = chunk1.read(), chunk2.read()
    seconds1, seconds2 = wav1.shape[0] // sr, wav2.shape[0] // sr
    expected1, expected2 = 4.0, 6.0
    assert expected1 == seconds1
    assert expected2 == seconds2


@pytest.mark.parametrize("channels", (None, 1, 2))
def test_mel_spectrum(tmp_path: Path, channels: int | None) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10, channels=channels)
    event = etypes.Audio(start=0, timeline="whatever", filepath=fp)
    feat = audio.MelSpectrum(frequency=50)
    out = feat(event, start=8, duration=4)
    assert out.shape == (40, 200)


def test_mel_spectrum_no_freq(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10)
    event = etypes.Audio(start=0, timeline="whatever", filepath=fp)
    feat = audio.MelSpectrum()
    assert feat.frequency == "native"
    out = feat(event, start=8, duration=4)
    assert feat._effective_frequency == 125
    assert out.shape == (40, 500)


@pytest.mark.skipif(
    "IN_GITHUB_ACTION" in os.environ, reason="Models are too big for CI cache"
)
@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers is not installed",
)
@pytest.mark.parametrize(
    "class_name", ["Wav2Vec", "Wav2VecBert", "SeamlessM4T", "Whisper"]
)
def test_audio_models(class_name: str, tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10)
    event = etypes.Audio(start=0, timeline="whatever", filepath=fp)
    model = getattr(audio, class_name)(frequency=50)
    out = model(event, start=8, duration=4)
    assert out.ndim == 2
    assert out.shape[-1] == 200


@pytest.mark.parametrize("layers", [0.0, 1.0, [0.0, 1.0], [0.0, 0.5, 1.0], "all"])
def test_wav2vec_layers(
    tmp_path: Path, layers: float | list[float] | tp.Literal["all"]
) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10)
    event = etypes.Audio(start=0, timeline="whatever", filepath=fp)
    m = "facebook/wav2vec2-base"
    feat = audio.HuggingFaceAudio(model_name=m, layers=layers, device="cpu")
    out = feat(event, start=8, duration=4)

    assert out.shape == (768, 200)


def test_wav2vec_cache_n_layers(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10)
    event = etypes.Audio(start=0, timeline="whatever", filepath=fp)
    infra = {"folder": tmp_path / "cache"}
    cfg: dict[str, tp.Any] = dict(frequency=50, device="cpu", infra=infra)
    cfg["model_name"] = "facebook/wav2vec2-base"
    layers = [0, 0.1, 0.2]
    feat1 = audio.HuggingFaceAudio(layers=layers, cache_n_layers=11, **cfg)
    layers2 = [0.8, 0.9, 1]
    feat2 = audio.HuggingFaceAudio(layers=layers2, cache_n_layers=11, **cfg)
    feat3 = audio.HuggingFaceAudio(layers=layers2, cache_n_layers=None, **cfg)

    out1 = feat1(event, start=8, duration=4)
    out2 = feat2(event, start=8, duration=4)
    out3 = feat3(event, start=8, duration=4)

    assert out1.shape == (768, 200)
    assert not (out1 == out2).all()
    assert (out2 == out3).all()

    assert feat1.infra.uid_folder() == feat2.infra.uid_folder()
    assert feat1.infra.uid_folder() != feat3.infra.uid_folder()


def test_mel_spectrum_size_issue(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=16_000, duration=1.310812)
    event = etypes.Audio(start=0, timeline="whatever", filepath=fp)
    melspec = audio.MelSpectrum(
        n_mels=23,
        n_fft=50,
        hop_length=10,
        frequency=1024,
        infra={"folder": tmp_path / "cache"},  # type: ignore
    )
    # this one is a edge case because of last rounding to a sample that does not exist
    out = melspec(event, start=1.310228937325665, duration=0.05000000000001137)
    assert out.shape == (23, 51)
    # a few more tests
    for _ in range(1000):
        seed = np.random.randint(2**32 - 1)
        print(f"Seeding with {seed} for reproducibility")
        rng = np.random.default_rng(seed)
        event.start = rng.uniform()
        melspec(event, start=1.0 + rng.uniform(), duration=rng.uniform())


@pytest.mark.parametrize("fs", [16000, 44100])
def test_speech_envelope_basic(tmp_path: Path, fs: int) -> None:
    """Test basic envelope extraction from audio."""
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=fs, duration=2.0)
    event = etypes.Audio(start=0, timeline="whatever", filepath=fp)
    feat = audio.SpeechEnvelope(frequency=100)
    out = feat(event, start=0, duration=1.0)

    # Check output shape: 1D envelope at 100 Hz for 1 second = 100 samples
    assert out.shape == (1, 100), f"Expected shape (1, 100), got {out.shape}"

    # Envelope should be mostly positive (filtering can introduce small negative values)
    assert out.mean() > 0, "Mean envelope should be positive"
    assert (out > -0.5).all(), "Envelope should not have large negative values"


@pytest.mark.parametrize("lowpass_freq", [10.0, 30.0, 50.0])
def test_speech_envelope_different_lowpass(tmp_path: Path, lowpass_freq: float) -> None:
    """Test envelope extraction with different lowpass frequencies."""
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=16000, duration=2.0)
    event = etypes.Audio(start=0, timeline="whatever", filepath=fp)

    feat = audio.SpeechEnvelope(lowpass_freq=lowpass_freq, frequency=100)
    out = feat(event, start=0, duration=1.0)

    assert out.shape == (1, 100)
    # Filtering can introduce small negative values, but mean should be positive
    assert out.mean() > 0
