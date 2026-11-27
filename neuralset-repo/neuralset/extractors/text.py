# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import itertools
import logging
import typing as tp
from abc import abstractmethod

import numpy as np
import pandas as pd
import pydantic
import torch
from exca import MapInfra
from exca.utils import environment_variables
from torch import nn
from tqdm import tqdm

import neuralset as ns
from neuralset import events as _ev  # avoid circular import
from neuralset import utils
from neuralset.base import TimedArray
from neuralset.extractors.base import (
    BaseExtractor,
    BaseStatic,
    ContextWindow,
    _ContextMixin,
)
from neuralset.extractors.hf import HuggingFaceConfig, HuggingFaceMixin

logger = logging.getLogger(__name__)

# pylint: disable=attribute-defined-outside-init
# pylint: disable=unused-variable
DataframeOrEventsOrSegments = (
    pd.DataFrame | tp.Sequence[_ev.Event] | tp.Sequence[ns.segments.Segment]
)


class BaseText(BaseStatic):
    """
    Base class for text extractors.
    """

    language: str = "english"
    requirements: tp.ClassVar[tuple[str, ...]] = ("rapidfuzz",)

    infra: MapInfra = MapInfra()

    def _exclude_from_cache_uid(self) -> list[str]:
        return super()._exclude_from_cache_uid() + ["duration", "frequency"]

    @infra.apply(
        item_uid=lambda event: f"{event.language}:{event.text}",
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArray",
    )
    def _get_data(self, events: list[_ev.etypes.Text]) -> tp.Iterator[np.ndarray]:
        if len(events) > 1:
            events = tqdm(events, desc="Computing word embeddings")  # type: ignore
        for event in events:
            yield self.get_embedding(event.text, language=event.language)

    def get_static(self, event: _ev.etypes.Text) -> torch.Tensor:
        return torch.tensor(next(self._get_data([event])))  # copy: memmap is read-only

    @abstractmethod
    def get_embedding(self, text: str, language: str = "") -> np.ndarray:
        raise NotImplementedError


class WordLength(BaseText):
    """
    Get word length.
    """

    event_types: tp.Literal["Word"] = "Word"

    # pylint: disable=unused-argument
    def get_embedding(self, text: str, language: str = "") -> np.ndarray:
        # return array of float for aggregation=averaging case in TimedArray
        return np.array([len(text)], dtype=float)


class WordFrequency(BaseText):
    """
    Get word frequency from wordfreq package.
    """

    event_types: tp.Literal["Word"] = "Word"
    requirements: tp.ClassVar[tuple[str, ...]] = ("wordfreq",)
    LANGUAGES: tp.ClassVar[dict[str, str]] = dict(
        english="en",
        french="fr",
        spanish="es",
        dutch="nl",
        chinese="zh",
        japanese="ja",
    )

    # pylint: disable=unused-argument
    def get_embedding(self, text: str, language: str = "") -> np.ndarray:
        from wordfreq import zipf_frequency  # noqa

        # The per-event language wins (it is the ground truth for the data),
        # falling back to the extractor's `language` only when the event carries
        # none -- same precedence as SpacyEmbedding. Lowercase so both names and
        # 2-letter codes resolve (wordfreq needs lowercase).
        lang = (language or self.language or "").lower()
        if not lang:
            raise ValueError(
                "No language specified: set language on the extractor or "
                "populate language on events."
            )
        value = zipf_frequency(text, self.LANGUAGES.get(lang, lang))
        return np.array([value])


class TfidfEmbedding(BaseText):
    """
    Get TF-IDF embeddings for Sentence events.
    """

    event_types: str | tuple[str, ...] = "Sentence"
    max_features: int = 5000
    _vectorizer: None = pydantic.PrivateAttr(None)

    @property
    def vectorizer(self) -> tp.Any:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features, stop_words=self.language
            )
        return self._vectorizer

    def prepare(self, obj: DataframeOrEventsOrSegments) -> None:
        events: list[_ev.etypes.Word]
        events = self._event_types_helper.extract(obj)  # type: ignore
        texts = [event.text for event in events]
        self.vectorizer.fit_transform(texts)
        super().prepare(events)

    # pylint: disable=unused-argument
    def get_embedding(self, text: str, language: str = "") -> np.ndarray:
        if self._vectorizer is None:
            msg = "The vectorizer is not fitted. Please call the prepare method before."
            raise ValueError(msg)
        vector = self.vectorizer.transform([text]).toarray()
        return vector.squeeze(0)


class SpacyEmbedding(BaseText):
    """Get word embedding from spacy.

    Parameters
    ----------
    model_name : str
        Explicit spacy model name (e.g. ``"en_core_web_sm"``).
        Mutually exclusive with ``language``.
    language : str
        Language name or ISO code (e.g. ``"english"``, ``"fr"``).
        Resolved to a default spacy model via :func:`~neuralset.utils.get_spacy_model`.
        Mutually exclusive with ``model_name``.
        If neither is set, language is read from events.
    """

    event_types: tp.Literal["Word"] = "Word"
    requirements: tp.ClassVar[tuple[str, ...]] = ("spacy>=3.8.2",)
    model_name: str = ""
    language: str = ""

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.model_name and self.language:
            raise ValueError(
                "model_name and language are mutually exclusive; "
                f"got model_name={self.model_name!r} and language={self.language!r}"
            )

    def prepare(self, obj: DataframeOrEventsOrSegments) -> None:
        # Install every model the *uncached* events need once, here in the
        # submitting process, so cluster workers sharing this environment find
        # them instead of each pip-installing the same model concurrently.  A
        # fully cached run installs nothing, as before.  An unsupported language
        # also fails here rather than in every worker.
        events: list[_ev.etypes.Word]
        events = self._event_types_helper.extract(obj)  # type: ignore
        to_compute = self._uncached(events)
        if not to_compute:
            models: set[str] = set()
        elif self.model_name:
            models = {self.model_name}
        else:
            languages = {event.language or self.language for event in to_compute}
            models = {utils.spacy_model_name(language=lang) for lang in languages if lang}
        for model in sorted(models):
            utils.install_spacy_model(model)
        super().prepare(events)

    def _uncached(self, events: list[_ev.etypes.Word]) -> list[_ev.etypes.Word]:
        """Events whose embedding ``_get_data`` will actually compute.

        A read-only check mirroring ``MapInfra``'s own cache lookup: every event
        without a cache folder or under ``mode="force"``, none under
        ``mode="read-only"`` (which never computes), else the cache misses.
        """
        infra = self.infra
        if infra.mode == "force":
            return list(events)
        if infra.mode == "read-only":
            return []
        try:
            cache = infra.cache_dict
        except ValueError:  # no cache folder: everything is computed
            return list(events)
        item_uid = infra.item_uid
        with cache.frozen_cache_folder():
            return [event for event in events if item_uid(event) not in cache]

    def get_embedding(self, text: str, language: str = "") -> np.ndarray:
        if self.model_name:
            nlp = utils.get_spacy_model(model=self.model_name)
        else:
            lang = language or self.language
            if not lang:
                raise ValueError(
                    "No language: set model_name or language on the extractor, "
                    "or populate language on events"
                )
            nlp = utils.get_spacy_model(language=lang)
        return nlp(text).vector


class SonarEmbedding(BaseText):
    """
    Get embeddings from sonar: https://arxiv.org/abs/2308.11466
    """

    event_types: tp.Literal["Sentence"] = "Sentence"

    requirements: tp.ClassVar[tuple[str, ...]] = (
        "fairseq2",
        "sonar-space",
    )
    LANGUAGES: tp.ClassVar[dict[str, str]] = dict(en="eng_Latn", english="eng_Latn")

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model"):
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

            self._model = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
            )
            self._model.eval()
        return self._model

    # pylint: disable=unused-argument
    @torch.no_grad()
    def get_embedding(self, text: str) -> np.ndarray:
        vector = self.model.predict(
            [text], source_lang=self.LANGUAGES.get(self.language, self.language)
        )  # type: ignore
        return vector.squeeze(0).cpu().numpy()


class SentenceTransformer(BaseText):
    """
    Get embeddings from SentenceTransformers: https://huggingface.co/sentence-transformers.
    """

    event_types: tp.Literal["Sentence"] = "Sentence"
    model_name: str = "all-mpnet-base-v2"

    requirements: tp.ClassVar[tuple[str, ...]] = ("sentence_transformers",)

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model"):
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
        return self._model

    # pylint: disable=unused-argument
    @torch.no_grad()
    def get_embedding(self, text: str, language: str = "") -> np.ndarray:
        vector = self.model.encode([text])  # type: ignore
        return vector.squeeze(0)


class HuggingFaceTextConfig(HuggingFaceConfig):
    processor_cls_name: str = "AutoTokenizer"
    processor_kwargs: dict[str, tp.Any] | None = {
        "truncation_side": "left",
        "padding_side": "right",
    }
    HF_CLASS_DEFAULTS: tp.ClassVar[dict[str, dict[str, str]]] = {
        "t5": {"model_cls_name": "AutoModelForTextEncoding"},
        "facebook/opt": {"model_cls_name": "OPTModel"},
        "facebook/bart": {"model_cls_name": "BartModel"},
        "gpt2": {"model_cls_name": "GPT2Model"},
        "phi-4": {"model_cls_name": "AutoModelForCausalLM"},
    }


class _HuggingFaceLanguageModel(HuggingFaceMixin):
    """Tokenizer and token limit of the HuggingFace text extractors."""

    hf_config: HuggingFaceTextConfig = HuggingFaceTextConfig()

    @property
    def tokenizer(self) -> tp.Any:
        tokenizer = self.processor
        if tokenizer.pad_token is None:
            # an existing token, so that model weights stay unchanged
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _max_length(self) -> int | None:
        """Token truncation limit, falling back to the model's positional capacity."""
        limit = self.tokenizer.model_max_length
        # HF "infinite" sentinel (~1e30) would disable truncation (overflows OPT)
        if isinstance(limit, int) and limit < int(1e29):
            return limit
        # learned-position tables are sized capacity + offset, so the offset cancels
        config = self.model.config
        for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        return None


class _HuggingFaceStaticText(BaseStatic, _HuggingFaceLanguageModel):
    """One vector per text event, which the language model reads alone (without
    context), aggregating its tokens with ``token_aggregation``.

    The text is read with the special tokens the tokenizer adds (e.g. Llama's BOS,
    without which the first token of the text becomes the attention sink and its
    states barely depend on it), but only the tokens of the text are aggregated."""

    model_name: str = "openai-community/gpt2"
    infra: MapInfra = MapInfra(
        timeout_min=25,
        gpus_per_node=1,
        cpus_per_task=10,
        min_samples_per_job=4096,
        version="2",  # special tokens added (no BOS before)
    )
    batch_size: int = 32

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return (
            ["batch_size"]
            + BaseStatic._exclude_from_cls_uid()
            + HuggingFaceMixin._exclude_from_cls_uid()
        )

    def _exclude_from_cache_uid(self) -> list[str]:
        return (
            ["frequency", "batch_size"]
            + BaseStatic._exclude_from_cache_uid(self)
            + HuggingFaceMixin._exclude_from_cache_uid(self)
        )

    def get_static(self, event: _ev.etypes.BaseText) -> torch.Tensor:  # type: ignore[override]
        (ta,) = self._get_timed_arrays([event], start=event.start, duration=0)
        return torch.from_numpy(np.asarray(ta.data))

    def _get_timed_arrays(
        self,
        events: list[_ev.etypes.BaseText],  # type: ignore[override]
        start: float,
        duration: float,
    ) -> tp.Iterable[TimedArray]:
        # one fetch for all the events, instead of a get_static call per event
        for event, latent in zip(events, self._get_data(events)):
            if self.cache_n_layers is not None:
                latent = self._aggregate_layers(latent)
            yield TimedArray(
                frequency=0, duration=event.duration, start=event.start, data=latent
            )

    @infra.apply(
        item_uid=lambda event: event.text,
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArray",
    )
    def _get_data(self, events: list[_ev.etypes.BaseText]) -> tp.Iterator[np.ndarray]:
        batches = list(itertools.batched(events, self.batch_size))
        if len(batches) > 1:
            batches = tqdm(batches, desc="Computing text embeddings")  # type: ignore
        device = self.model_device
        with torch.no_grad():
            for batch in batches:
                texts = [event.text for event in batch]
                with environment_variables(TOKENIZERS_PARALLELISM="false"):
                    inputs = self.tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        # guards tokenizers reporting no real limit (e.g. OPT)
                        max_length=self._max_length(),
                        return_special_tokens_mask=True,
                    )
                # the text's own tokens: neither special tokens nor padding
                special = inputs.pop("special_tokens_mask").bool()
                text_tokens = inputs["attention_mask"].bool() & ~special
                inputs = inputs.to(device)
                outputs = self.model(**inputs, output_hidden_states=True)
                if "hidden_states" in outputs:
                    states = outputs.hidden_states
                else:  # bart (encoder/decoder)
                    # decoder position j + 1 reads token j (shift_tokens_right): realign
                    decoder = [s.roll(-1, dims=1) for s in outputs.decoder_hidden_states]
                    states = (*outputs.encoder_hidden_states, *decoder)
                hidden_states = torch.stack([layer.cpu() for layer in states])
                for i, text in enumerate(texts):
                    # (n_layers, n_tokens, n_dims), the text's tokens only
                    state = hidden_states[:, i, text_tokens[i]]
                    out = self._aggregate_tokens(state).cpu().numpy()
                    if self.cache_n_layers is None:
                        out = self._aggregate_layers(out)
                    if np.isnan(out).any():
                        raise ValueError(f"NaN in the embedding of {text!r}")
                    yield out
                # free memory
                del hidden_states, states, outputs, inputs
                if device.type == "cuda":
                    torch.cuda.empty_cache()


class HuggingFaceWord(_HuggingFaceStaticText):
    """Embeddings of Word events, each read alone by a HuggingFace language model:
    one vector per word, aggregating its tokens (``token_aggregation``).

    For the embeddings of words in their context, use :class:`HuggingFaceText`.

    Parameters
    ----------
    batch_size: int
        Number of words per forward pass.
    """

    event_types: tp.Literal["Word"] = "Word"


class HuggingFaceSentence(_HuggingFaceStaticText):
    """Embeddings of Sentence events, each read alone by a HuggingFace language
    model: one vector per sentence, aggregating its tokens (``token_aggregation``).

    Parameters
    ----------
    batch_size: int
        Number of sentences per forward pass.

    Note
    ----
    The tokenizer truncates sentences beyond the token limit of the model (their
    first tokens, with the default ``truncation_side="left"``).
    """

    event_types: tp.Literal["Sentence"] = "Sentence"


@dataclasses.dataclass(frozen=True)
class _AlignedText:
    """An AlignedText to embed, identified by its text and word spans, and also by
    its word onsets when they set its windows (``timed``, for ``context.duration``)."""

    event: _ev.etypes.AlignedText
    timed: bool

    def uid(self) -> str:
        uid = self.event.content_uid()
        return f"{uid}_{self.event.onsets_uid()}" if self.timed else uid


class WordContextWindow(ContextWindow):
    """``ContextWindow`` over the words of a text, spanning either ``words`` words
    or ``duration`` seconds (the words starting within it, whatever their number).
    ``min_context`` is then in words or in seconds.

    Parameters
    ----------
    words :
        Number of words per window, exclusive with ``duration``.
    min_context_words :
        Input only, exclusive with ``overlap``: ``min_context`` in words, converted
        to ``overlap`` (with ``words``).
    """

    words: pydantic.PositiveInt | None = None
    _MIN_CONTEXT: tp.ClassVar[dict[str, str]] = {
        **ContextWindow._MIN_CONTEXT,
        "min_context_words": "words",
    }

    @pydantic.model_validator(mode="after")
    def _one_size(self) -> tp.Self:
        if (self.words is None) == (self.duration is None):
            msg = f"Set exactly one of words and duration, got words={self.words}, "
            msg += f"duration={self.duration} (words=None to unset a preset's)"
            raise ValueError(msg)
        return self

    @property
    def _size(self) -> float | None:
        return self.duration if self.words is None else self.words


@dataclasses.dataclass(frozen=True)
class _TextWindow:
    """Words [first, stop) of a text, read in one forward pass to embed
    words [embed_from, embed_stop)."""

    event: _ev.etypes.AlignedText
    first: int
    stop: int
    embed_from: int
    embed_stop: int

    @classmethod
    def strided(
        cls, event: _ev.etypes.AlignedText, context: WordContextWindow
    ) -> list["_TextWindow"]:
        """Windows of ``context.words`` words embedding each word once."""
        n_words = len(event.word_starts)
        return [cls(event, *bounds) for bounds in context.windows(n_words, 1)]

    @classmethod
    def timed(
        cls, event: _ev.etypes.AlignedText, context: WordContextWindow
    ) -> list["_TextWindow"]:
        """As ``strided``, on the word onsets: each window holds the words starting
        within ``context.duration`` seconds. Windows embedding no word (within
        pauses) are skipped."""
        ticks = event.onset_ticks()
        windows = []
        for bounds in context.windows(int(ticks[-1]) + 1, 1 / event.ONSET_TICK):
            first, stop, embed_from, embed_stop = np.searchsorted(ticks, bounds).tolist()
            if embed_from < embed_stop:
                windows.append(cls(event, first, stop, embed_from, embed_stop))
        return windows

    @property
    def is_last(self) -> bool:
        return self.embed_stop == len(self.event.word_starts)

    @property
    def text(self) -> str:
        end = self.event.word_char_stops[self.first : self.stop].max()
        return self.event.text[self._offset : end]

    @property
    def embedded_spans(self) -> np.ndarray:
        """(n_embedded_words, 2) character spans in the window text."""
        words = slice(self.embed_from, self.embed_stop)
        chars = [self.event.word_char_starts[words], self.event.word_char_stops[words]]
        return np.stack(chars, axis=1) - self._offset

    @property
    def _offset(self) -> int:
        return int(self.event.word_char_starts[self.first])


class HuggingFaceText(BaseExtractor, _HuggingFaceLanguageModel, _ContextMixin):
    """Contextual word embeddings of AlignedText events from a language model.

    Each text is read with overlapping windows of words (``context``), each
    embedding only part of its words, so that every word is embedded once and sees
    at least ``context.min_context`` words of context (fewer at the edges of the
    text): in the past with ``context.causal`` (default), else on each side. With
    ``context.duration``, windows span a duration instead of a number of words:
    every word then sees the words starting up to ``context.min_context`` seconds
    before it (and after it, if not causal). A word's tokens are the tokens
    overlapping its characters. One vector per word is cached, and each word fills
    the segment samples over its own time span, as Word events do.

    The AlignedText events, which hold the words of each Text, are built by
    :class:`~neuralset.events.transforms.text.AddAlignedTexts`. Context stops at
    their boundaries: with ``AddAlignedTexts(text_type="Sentence")``, every word
    sees only its sentence. For the embeddings of Word or Sentence events read
    alone, use :class:`HuggingFaceWord` or :class:`HuggingFaceSentence`.

    Parameters
    ----------
    frequency: float
        Output sampling rate in Hz (required). With 0, the output is static: the
        vectors of the words overlapping the segment are aggregated into one
        (``aggregation``).
    context: WordContextWindow
        Windows of words, 1024 words sharing half of them by default. The first
        window of each text must fit in the model's token limit. With
        ``context.duration``, windows, hence features and their cache, depend on
        the word onsets, not only on the text.
    batch_size: int
        Number of windows per forward pass.

    Note
    ----
    Requires a fast tokenizer, for character offsets. With ``context.causal`` and a
    bidirectional model, a word would also see the following words of its
    window, up to ``(1 - context.overlap) * context.words - 1``. Windows longer
    than the model's token limit lose their first words (``truncation_side``),
    so their words see less context than requested (a warning is logged).
    """

    model_name: str = "Qwen/Qwen3-0.6B"
    event_types: tp.Literal["AlignedText"] = "AlignedText"
    aggregation: tp.Literal["single", "sum", "mean"] = "sum"
    frequency: pydantic.NonNegativeFloat
    # one vector per word, so the token dimension must be reduced
    token_aggregation: tp.Literal["first", "last", "mean", "sum", "max"] = "mean"
    context: WordContextWindow = WordContextWindow(words=1024, causal=True)
    batch_size: pydantic.PositiveInt = 4
    infra: MapInfra = MapInfra(
        timeout_min=60,
        gpus_per_node=1,
        cpus_per_task=10,
        min_samples_per_job=16,
        version="v8",
    )
    _warned_truncation: bool = False

    @pydantic.model_validator(mode="before")
    @classmethod
    def _reject_previous_api(cls, data: tp.Any) -> tp.Any:
        # in neuralset 0.4, HuggingFaceText embedded Word and Sentence events
        if not isinstance(data, dict):
            return data
        if "contextualized" in data or data.get("event_types", "AlignedText") not in (
            "AlignedText",
            ("AlignedText",),
        ):
            msg = (
                "HuggingFaceText now embeds the words of AlignedText events in their "
                "context (changed after neuralset 0.4). For the embeddings of Word "
                "or Sentence events read alone (previously contextualized=False), "
                "use HuggingFaceWord or HuggingFaceSentence. For contextual word "
                "embeddings (previously contextualized=True, with AddContextToWords), "
                "add AddAlignedTexts to the events transforms and set the windows of "
                "words with HuggingFaceText.context (e.g. {'words': 1024}), instead "
                "of the context of each Word; for sentence_only=True, use "
                f"AddAlignedTexts(text_type='Sentence') (got {data!r})."
            )
            raise ValueError(msg)
        return data

    def _is_static(self) -> bool:
        # words are static vectors: the output has no time axis with frequency=0
        return not self.frequency

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return (
            ["batch_size"]
            + BaseExtractor._exclude_from_cls_uid()
            + HuggingFaceMixin._exclude_from_cls_uid()
        )

    def _exclude_from_cache_uid(self) -> list[str]:
        return (
            ["frequency", "batch_size"]
            + BaseExtractor._exclude_from_cache_uid(self)
            + HuggingFaceMixin._exclude_from_cache_uid(self)
        )

    def prepare(self, obj: tp.Any) -> None:
        # the extracted events are of event_types, i.e. AlignedText
        events = tp.cast(
            list[_ev.etypes.AlignedText], self._event_types_helper.extract(obj)
        )
        # a previous call (e.g. on another split) may have set the output shape
        if not events and self._missing_default is None:
            # Words left unaligned: AddAlignedTexts is missing from the events chain
            if _ev.etypes.EventTypesHelper("Word").extract(obj):
                raise ValueError(
                    f"{self.name} found Word events but no AlignedText events to "
                    "prepare on: add AddAlignedTexts to the events chain (or use "
                    "HuggingFaceWord for the embeddings of Word events read alone)."
                )
            return  # no text to embed (e.g. image or resting-state data)
        # as BaseExtractor.prepare, on the items of _get_data
        self._get_data(self._texts(events))
        if events:  # populates the output shape
            event = events[0]
            self(event, start=event.start, duration=0.001, trigger=event)

    def _texts(self, events: tp.Sequence[_ev.etypes.AlignedText]) -> list[_AlignedText]:
        timed = self.context.duration is not None
        return [_AlignedText(event, timed) for event in events]

    def _windows(self, event: _ev.etypes.AlignedText) -> list[_TextWindow]:
        if self.context.duration is None:
            return _TextWindow.strided(event, self.context)
        return _TextWindow.timed(event, self.context)

    @property
    def tokenizer(self) -> tp.Any:
        tokenizer = super().tokenizer
        if not tokenizer.is_fast:
            msg = f"{self.name} needs a fast tokenizer (for character offsets), "
            msg += f"got {type(tokenizer).__name__}"
            raise ValueError(msg)
        return tokenizer

    def _aggregate_word_layers(self, latents: np.ndarray) -> np.ndarray:
        """Layer aggregation of (n_words, n_layers, ...) latents."""
        out = self._aggregate_layers(np.moveaxis(np.asarray(latents), 1, 0))
        return np.moveaxis(out, 0, 1) if out.ndim == latents.ndim else out

    def _read_words(self, latents: np.ndarray, indices: np.ndarray) -> np.ndarray:
        out = np.asarray(latents[indices])
        if self.cache_n_layers is not None:
            out = self._aggregate_word_layers(out)
        return out

    def _get_timed_arrays(
        self,
        events: list[_ev.etypes.AlignedText],  # type: ignore[override]
        start: float,
        duration: float,
    ) -> tp.Iterable[TimedArray]:
        if not events:  # the zero placeholder below needs the latents of one
            raise RuntimeError("_get_timed_arrays should not be called with no event")
        found = False
        for event, latents in zip(events, self._get_data(self._texts(events))):
            indices = event.overlapping_words(start, start + duration)
            if not len(indices):
                continue
            found = True
            word_starts = event.start + event.word_starts[indices]
            word_durations = event.word_durations[indices]
            word_latents = self._read_words(latents, indices)
            for word_start, word_duration, latent in zip(
                word_starts, word_durations, word_latents
            ):
                yield TimedArray(
                    frequency=0,
                    start=float(word_start),
                    duration=float(word_duration),
                    data=latent,
                )
        if not found:  # zero placeholder carrying the feature shape
            latent = self._read_words(latents, np.arange(1))[0]
            yield TimedArray(
                frequency=0, start=start, duration=duration, data=np.zeros_like(latent)
            )

    @infra.apply(
        item_uid=_AlignedText.uid,
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArray",
    )
    def _get_data(self, texts: list[_AlignedText]) -> tp.Iterator[np.ndarray]:
        windows = [window for item in texts for window in self._windows(item.event)]
        batches = list(itertools.batched(windows, self.batch_size))
        if len(batches) > 1:
            batches = tqdm(batches, desc="Computing contextual word embeddings")  # type: ignore
        parts: list[np.ndarray] = []
        for batch in batches:
            for window, latents in zip(batch, self._embed_windows(batch)):
                parts.append(latents)
                if not window.is_last:
                    continue
                out = np.concatenate(parts)
                parts = []
                if self.cache_n_layers is None:
                    out = self._aggregate_word_layers(out)
                if np.isnan(out).any():
                    text = window.event.text[:100]
                    raise ValueError(f"NaN in embeddings of text {text!r}")
                yield out

    @torch.no_grad()
    def _embed_windows(self, windows: tp.Sequence[_TextWindow]) -> list[np.ndarray]:
        """(n_embedded_words, n_layers, *embedding_shape) latents per window."""
        texts = [window.text for window in windows]
        max_length = self._max_length()
        with environment_variables(TOKENIZERS_PARALLELISM="false"):
            inputs = self.tokenizer(
                texts,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=True,  # truncation_side="left" in hf_config: trims context
                max_length=max_length,
                return_offsets_mapping=True,
            )
        offsets = inputs.pop("offset_mapping").numpy()
        real = inputs["attention_mask"].numpy().astype(bool)
        self._warn_truncation(texts, offsets, real, max_length)
        outputs = self.model(**inputs.to(self.model_device), output_hidden_states=True)
        hidden = self._layer_subselection(torch.stack(outputs.hidden_states))
        out = []
        for b, (text, window) in enumerate(zip(texts, windows)):
            word_spans = window.embedded_spans
            tok_starts, tok_stops = offsets[b, :, 0], offsets[b, :, 1]
            overlaps = (
                real[b]
                & (tok_starts < word_spans[:, 1:])
                & (tok_stops > word_spans[:, :1])
            )
            latents = []
            for (char_start, char_stop), overlap in zip(word_spans, overlaps):
                tokens = np.flatnonzero(overlap)
                if not len(tokens):
                    msg = f"No token for word {text[char_start:char_stop]!r}: its window "
                    msg += f"was probably truncated to the {max_length}-token limit "
                    msg += f"of {self.model_name}, lower {self._context_field}"
                    raise ValueError(msg)
                word_state = hidden[:, b, tokens[0] : tokens[-1] + 1]
                latents.append(self._aggregate_tokens(word_state))
            out.append(torch.stack(latents).cpu().numpy())
        return out

    @property
    def _context_field(self) -> str:
        return "context.words" if self.context.duration is None else "context.duration"

    def _warn_truncation(
        self,
        texts: list[str],
        offsets: np.ndarray,
        real: np.ndarray,
        max_length: int | None,
    ) -> None:
        """Warns once if windows were truncated to the token limit, as their
        words then see less context than requested."""
        if self._warned_truncation or max_length is None:
            return
        n_truncated = 0
        for text, offset, mask in zip(texts, offsets, real):
            tokens = np.flatnonzero(mask)
            cut = offset[tokens[0], 0] > 0 or offset[tokens[-1], 1] < len(text)
            n_truncated += bool(len(tokens) >= max_length and cut)
        if n_truncated:
            self._warned_truncation = True
            logger.warning(
                "%s: %d of %d windows were truncated to the %d-token limit of %s, "
                "so their words see less context than set by %s (warned once)",
                self.name,
                n_truncated,
                len(texts),
                max_length,
                self.model_name,
                self._context_field,
            )
