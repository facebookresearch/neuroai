# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""All the supported audio extractors."""

import typing as tp
import warnings
from abc import abstractmethod
from typing import List

import numpy as np
import pydantic
import torch
from exca import MapInfra
from torch import nn
from torch.nn import functional as F

from neuralset import base as nsbase
from neuralset.events import etypes

from .base import BaseExtractor, HuggingFaceMixin

# pylint: disable=import-outside-toplevel


class BaseAudio(BaseExtractor):
    """Audio extractor

    Note
    ----
    Default frequency is derived from event duration and computed latent dimension
    after the first call. Note that this can be slightly off due to sampling, so you
    should provide the frequency yourself if you want consistency.
    """

    event_types: str | tuple[str, ...] = "Audio"
    requirements: tp.ClassVar[tuple[str, ...]] = (
        "julius>=0.2.7",
        "pillow>=9.2.0",
    )
    # frequency derived from sampling rate of the produced extractor
    frequency: tp.Literal["native"] | float = "native"
    norm_audio: bool = True

    infra: MapInfra = MapInfra(
        timeout_min=25,
        gpus_per_node=1,
        cpus_per_task=8,
        min_samples_per_job=4096,
        version="v5",
    )

    @property
    @abstractmethod
    def _input_frequency(self) -> float:
        raise NotImplementedError

    def _exclude_from_cache_uid(self) -> List[str]:
        return super()._exclude_from_cache_uid()

    @abstractmethod
    def _process_wav(self, wav: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _preprocess_wav(self, wav: torch.Tensor) -> torch.Tensor:
        wav = torch.mean(wav, dim=1)  # stereo to mono
        if self.norm_audio:
            wav = (wav - wav.mean()) / (1e-8 + wav.std())
        return wav

    def _resample_wav(
        self, wav: torch.Tensor, old_frequency: float, new_frequency: float
    ) -> torch.Tensor:
        for freq in (old_frequency, new_frequency):
            if not float(freq).is_integer():
                raise ValueError(f"Frequencies need to be integers, got {freq}")
        old_frequency, new_frequency = int(old_frequency), int(new_frequency)
        import julius  # noqa

        wav = julius.resample.ResampleFrac(
            old_sr=old_frequency,
            new_sr=new_frequency,  # type: ignore
        )(wav.T).T
        return wav

    @infra.apply(
        item_uid=lambda e: e._splittable_event_uid(),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_data(self, events: list[etypes.Event]) -> tp.Iterator[nsbase.TimedArray]:
        if len(events) > 1:
            from tqdm import tqdm

            events = tqdm(events, desc="Computing audio embeddings")  # type: ignore
        for event in events:
            if isinstance(event, etypes.Audio):
                wav = event.read()
                sfreq = event.frequency
            elif isinstance(event, etypes.Video):
                audio = event.read().audio
                wav = torch.tensor(audio.to_soundarray(), dtype=torch.float32)
                sfreq = audio.fps
            else:
                raise ValueError(
                    f"Unsupported event type for Audio extractor: {type(event)}"
                )
            wav = self._resample_wav(wav, sfreq, self._input_frequency)
            wav = self._preprocess_wav(wav)
            latents = self._process_wav(wav)
            if self.frequency == "native":
                data = latents.numpy()
                freq: float = data.shape[-1] / event.duration
            else:
                freq = float(self.frequency)
                timepoints = nsbase.Frequency(freq).to_ind(event.duration)
                if abs(timepoints - latents.shape[-1]) > 0:
                    if len(latents.shape) == 2:  # d, t
                        latents = F.interpolate(latents[None], timepoints)[0]
                    else:  # n_layers, d, t
                        latents = F.interpolate(latents, timepoints)
                data = latents.numpy()
            yield nsbase.TimedArray(
                data=data,
                frequency=freq,
                start=nsbase._UNSET_START,
                duration=event.duration,
            )


class MelSpectrum(BaseAudio):
    """
    Compute the Mel spectrogram representation of an audio waveform.

    This feature extracts a Mel-scaled power spectrogram from raw waveform data,
    converting time-domain audio into a frequency-domain representation that
    emphasizes perceptually relevant frequency bands. The resulting tensor can
    optionally be log-scaled for improved numerical stability and interpretability.

    Parameters
    ----------
    n_mels : int, default=40
        Number of Mel filter banks to use when computing the Mel spectrogram.
    n_fft : int, default=512
        Size of the FFT window used to compute the short-time Fourier transform (STFT).
    hop_length : int or None, default=None
        Number of samples between successive frames. Defaults to ``n_fft // 4`` if not set.
    normalized : bool, default=True
        If True, normalize the spectrogram output.
    use_log_scale : bool, default=True
        If True, apply a logarithmic transformation (base 10) to the Mel spectrum.
    log_scale_eps : float, default=1e-5
        Small constant added to the Mel spectrum before taking the logarithm,
        to avoid numerical issues with log(0).
    """

    n_mels: int = 40
    n_fft: int = 512
    hop_length: int | None = None  # defaults to n_fft // 4
    normalized: bool = True
    use_log_scale: bool = True
    log_scale_eps: float = 1e-5
    # internal
    _transform: tp.Any = pydantic.PrivateAttr()
    requirements: tp.ClassVar[tuple[str, ...]] = ("torchaudio",)

    @property
    def _input_frequency(self) -> float:
        return 16_000

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        import torchaudio

        hop_length = self.n_fft // 4 if self.hop_length is None else self.hop_length
        self._transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._input_frequency,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=hop_length,
            normalized=self.normalized,
        )

    def _process_wav(self, wav: torch.Tensor) -> torch.Tensor:
        """Returns the wav at the processing frequency (default wav frequency)"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            melspec = self._transform(wav)[:, :-1]  # remove one extra sample
        if self.use_log_scale:
            melspec = torch.log10(melspec + self.log_scale_eps)
        return melspec

    def _get_timed_arrays(
        self, events: list[etypes.Event], start: float, duration: float
    ) -> tp.Iterable[nsbase.TimedArray]:
        for event, ta in zip(events, self._get_data(events)):
            yield ta.with_start(event.start)


class SpeechEnvelope(BaseAudio):
    """
    Extract the acoustic amplitude envelope from audio waveforms.

    The envelope is computed by taking the absolute value of the Hilbert
    transform of the audio signal, optionally followed by lowpass filtering
    to smooth the envelope.

    Parameters
    ----------
    lowpass_freq : float or None, default=30.0
        Cutoff frequency (Hz) for lowpass filtering the envelope.
        If None, no lowpass filtering is applied.
    filter_order : int, default=4
        Order of the Butterworth lowpass filter.
    """

    lowpass_freq: float | None = 30.0
    filter_order: int = 4

    @property
    def _input_frequency(self) -> float:
        return 16_000.0

    def _process_wav(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Extract the amplitude envelope from the audio waveform.

        Returns
        -------
        torch.Tensor
            1D tensor containing the envelope timeseries with shape [time_points].
        """
        import scipy.signal

        analytic_signal = scipy.signal.hilbert(wav.numpy())
        envelope = np.abs(analytic_signal)

        # Apply optional lowpass filtering to smooth the envelope
        if self.lowpass_freq is not None:
            nyquist = self._input_frequency / 2
            normalized_cutoff = self.lowpass_freq / nyquist
            b, a = scipy.signal.butter(
                self.filter_order, normalized_cutoff, btype="low", analog=False
            )
            envelope = scipy.signal.filtfilt(b, a, envelope)

        return torch.from_numpy(np.ascontiguousarray(envelope)).float().unsqueeze(0)

    def _get_timed_arrays(
        self, events: list[etypes.Event], start: float, duration: float
    ) -> tp.Iterable[nsbase.TimedArray]:
        for event, ta in zip(events, self._get_data(events)):
            yield ta.with_start(event.start)


class SonarAudio(BaseAudio):
    """
    Extract deep audio embeddings from waveforms using the Sonar speech encoder.

    SONAR stands for Sentence-level multimOdal and laNguage-Agnostic Representations

    This extractor leverages the `sonar_speech_encoder_eng` model to produce speech sentence embeddings.

    Parameters
    ----------
    sampling_rate : int, default=16_000
        The input sampling rate expected by the Sonar model.
    layer : float, default=0.5
        The relative layer from which to extract the embedding (0=first layer, 1.= last layer).
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("sonar-space", "fairseq2")

    sampling_rate: int = 16_000
    # use hidden_states for transformer layers and extract_features for convolutional layers
    layer: float = 0.5

    # internal
    _model: nn.Module
    _feature_extractor: nn.Module

    @property
    def _input_frequency(self) -> float:
        return 16_000

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model"):
            self._model = self._get_sound_model()
        return self._model

    def _get_sound_model(self) -> nn.Module:
        from sonar.inference_pipelines.speech import (  # type: ignore
            SpeechToEmbeddingModelPipeline,
        )

        pipeline = SpeechToEmbeddingModelPipeline(encoder="sonar_speech_encoder_eng")
        model = pipeline.model
        n_layers = len(model.encoder.layers)
        layer_idx = int(self.layer * n_layers)
        model.encoder.layers = model.encoder.layers[:layer_idx]
        model.forward = lambda x: model.encoder(
            model.encoder_frontend(x.seqs, None)[0], None
        )
        return pipeline

    def _process_wav(self, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model.predict([wav])  # type: ignore

        return out.squeeze(1).detach().cpu().clone().transpose(-1, -2)  # type: ignore


class HuggingFaceAudio(BaseAudio, HuggingFaceMixin):
    """
    Base class for extracting audio features from Hugging Face models.

    This class provides a unified interface to load and process pretrained
    Hugging Face audio models such as Wav2Vec2, HuBERT, or XLS-R. It supports
    both convolutional and transformer layer outputs and handles feature
    extraction, model management, and layer aggregation automatically.

    Some model types should be used through their subclasses as special
    handling is required (e.g., Whisper, SeamlessM4T, or Wav2VecBert).

    Parameters
    ----------
    model_name : str, default='facebook/wav2vec2-large-xlsr-53'
        Name or path of the pretrained Hugging Face model to load.
    normalized : bool, default=True
        Whether to normalize the input waveform before feature extraction.
    layer_type : {'transformer', 'convolution'}, default='transformer'
        Which internal representation to extract from the model:
        - ``'transformer'`` returns hidden states from transformer layers.
        - ``'convolution'`` returns convolutional feature maps.
    """

    model_name: str = "facebook/wav2vec2-large-xlsr-53"
    requirements: tp.ClassVar[tuple[str, ...]] = ("transformers>=4.29.2",)

    normalized: bool = True
    layer_type: tp.Literal["transformer", "convolution"] = "transformer"
    # internal
    _model: nn.Module
    _feature_extractor: nn.Module

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if "whisper" in self.model_name and not isinstance(self, Whisper):
            raise ValueError(
                "Whisper model is not supported by HuggingFaceAudio. Use Whisper class instead."
            )
        if "m4t" in self.model_name and not isinstance(self, SeamlessM4T):
            raise ValueError(
                "SeamlessM4T model is not supported by HuggingFaceAudio. Use SeamlessM4T class instead."
            )
        if "w2v-bert" in self.model_name and not isinstance(self, Wav2VecBert):
            raise ValueError(
                "Wav2VecBert model is not supported by HuggingFaceAudio. Use Wav2VecBert class instead."
            )

    @property
    def _input_frequency(self) -> float:
        return self.feature_extractor.sampling_rate  # type: ignore

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        base = BaseAudio._exclude_from_cls_uid()
        return base + HuggingFaceMixin._exclude_from_cls_uid()

    def _exclude_from_cache_uid(self) -> list[str]:
        base = BaseAudio._exclude_from_cache_uid(self)
        return base + HuggingFaceMixin._exclude_from_cache_uid(self)

    @property
    def feature_extractor(self) -> nn.Module:
        if not hasattr(self, "_feature_extractor"):
            self._feature_extractor = self._get_feature_extractor(self.model_name)
        return self._feature_extractor

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model"):
            self._model = self._get_sound_model(self.model_name)
        return self._model

    def _get_feature_extractor(self, model_name: str) -> torch.nn.Module:
        from transformers import AutoFeatureExtractor

        return AutoFeatureExtractor.from_pretrained(model_name)

    def _get_sound_model(self, model_name: str) -> torch.nn.Module:
        from transformers import AutoModel

        _model = AutoModel.from_pretrained(model_name)
        _model.to(self.device)
        _model.eval()
        return _model

    def _get_features(self, wav):
        out = self._feature_extractor(
            wav,
            return_tensors="pt",
            sampling_rate=self.feature_extractor.sampling_rate,
            do_normalize=self.normalized,
        )
        try:
            return out["input_features"]
        except KeyError:
            return out["input_values"]

    def _get_timed_arrays(
        self, events: list[etypes.Event], start: float, duration: float
    ) -> tp.Iterable[nsbase.TimedArray]:
        if not events:
            raise RuntimeError("_get_timed_arrays should not be called with no event")
        for ta, event in zip(self._get_data(events), events):
            sub = ta.with_start(event.start).overlap(start=start, duration=duration)
            if self.cache_n_layers is not None:
                sub.data = self._aggregate_layers(sub.data)
            yield sub

    def _process_wav(self, wav: torch.Tensor) -> torch.Tensor:
        features = self._get_features(wav)
        with torch.no_grad():
            outputs = self.model(features.to(self.device), output_hidden_states=True)
        if self.layer_type == "transformer":
            out: tp.Any = outputs.get("hidden_states")
        elif self.layer_type == "convolution":
            out = outputs.get("extract_features")
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")
        if isinstance(out, tuple):
            out = torch.stack(out)

        out = out.squeeze(1).detach().cpu().clone().transpose(-1, -2).numpy()  # type: ignore
        if self.cache_n_layers is None:
            out = self._aggregate_layers(out)

        return torch.Tensor(out)


class Wav2Vec(HuggingFaceAudio):
    """
    Extract speech embeddings using a pretrained Wav2Vec 2.0 model from Hugging Face.

    The Wav2Vec 2.0 architecture learns contextualized speech representations from raw
    audio waveforms using self-supervised pretraining on large multilingual audio corpora,
    and is widely used for tasks such as automatic speech recognition (ASR), speaker verification,
    and speech classification.

    Parameters
    ----------
    model_name : str
        The Hugging Face model identifier to load, defaulting to
        ``"facebook/wav2vec2-large-xlsr-53"``.
    """

    model_name: str = "facebook/wav2vec2-large-xlsr-53"


class Wav2VecBert(HuggingFaceAudio):
    """
    Extract speech embeddings using the pretrained Wav2Vec2-BERT model from Hugging Face.

    Wav2Vec2-BERT is a self-supervised speech representation
    model that integrates Wav2Vec 2.0's contrastive pretraining with a BERT-style
    masked language modeling objective. The model produces deep, contextualized audio embeddings
    suitable for a wide range of downstream speech and audio understanding tasks.

    Parameters
    ----------
    model_name : str
        The Hugging Face model identifier to load. Defaults to ``"facebook/w2v-bert-2.0"``.
    """

    model_name: str = "facebook/w2v-bert-2.0"

    def _get_sound_model(self, model_name: str) -> torch.nn.Module:
        from transformers import Wav2Vec2BertModel

        _model = Wav2Vec2BertModel.from_pretrained(model_name)
        _model.to(self.device)
        _model.eval()
        return _model


class SeamlessM4T(HuggingFaceAudio):
    """
    Extract speech embeddings using the pretrained Seamless M4T model from Hugging Face.

    Seamless M4T is a multilingual, multimodal transformer that includes a dedicated speech encoder.
    It converts raw audio waveforms into high-level embeddings suitable for speech
    understanding, translation, and other downstream tasks.

    Attributes
    ----------
    model_name : str
        The Hugging Face model identifier to load. Defaults to
        ``"facebook/hf-seamless-m4t-medium"``.
    """

    model_name: str = "facebook/hf-seamless-m4t-medium"

    def _get_sound_model(self, model_name: str) -> torch.nn.Module:
        from transformers import SeamlessM4TModel

        _model = SeamlessM4TModel.from_pretrained(model_name).speech_encoder.to(
            self.device
        )
        _model.to(self.device)
        _model.eval()
        return _model


class Whisper(HuggingFaceAudio):
    """
    Extract speech embeddings using the pretrained Whisper model from Hugging Face.

    Whisper is a multilingual speech recognition and translation model that includes
    a dedicated encoder for audio processing. This class provides an interface to
    convert raw audio waveforms into high-level embeddings suitable for automatic
    speech recognition (ASR), speech translation, and other downstream tasks.

    Attributes
    ----------
    model_name : str
        The Hugging Face model identifier to load. Defaults to
        ``"openai/whisper-large-v3-turbo"``.
    """

    model_name: str = "openai/whisper-large-v3-turbo"

    def _get_sound_model(self, model_name: str) -> torch.nn.Module:
        from transformers import WhisperModel

        _model = WhisperModel.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).encoder
        _model.to(self.device)
        _model.eval()
        return _model
