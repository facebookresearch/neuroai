# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from abc import abstractmethod

import numpy as np
import pandas as pd
import pydantic
import torch
from exca import MapInfra
from exca.utils import environment_variables
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import neuralset as ns
from neuralset import events as _ev  # avoid circular import
from neuralset import utils
from neuralset.base import TimedArray
from neuralset.extractors.base import BaseStatic
from neuralset.extractors.hf import HuggingFaceConfig, HuggingFaceMixin

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
        cache_type="MemmapArrayFile",
    )
    def _get_data(self, events: list[_ev.etypes.Text]) -> tp.Iterator[np.ndarray]:
        if len(events) > 1:
            events = tqdm(events, desc="Computing word embeddings")  # type: ignore
        for event in events:
            yield self.get_embedding(event.text, language=event.language)

    def get_static(self, event: _ev.etypes.Text) -> torch.Tensor:
        latent = torch.from_numpy(next(self._get_data([event])))
        return latent

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


class TextDataset(Dataset):
    """
    Dataset for contextual embeddings.
    """

    def __init__(self, events: list[_ev.etypes.Word | _ev.etypes.Sentence]) -> None:
        self.events = events

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx) -> tuple[str, str]:
        sel = self.events[idx]
        return sel.text, getattr(sel, "context", "")


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


class HuggingFaceText(BaseStatic, HuggingFaceMixin):
    """
    Get embeddings from HuggingFace language models.
    This extractor can be applied to any kind of event which has a text attribute: Word, Sentence, etc.

    Parameters
    ----------
    batch_size: int
        Batch size for the language model.
    contextualized: bool
        True by default, the context of the event is used to compute the embeddings.

    Note
    ----
    The tokenizer truncates the input to the maximum size specified by the model.
    An empty context will raise an error to the default HuggingFaceText
    since contextualized is True by default.
    To get non-contextualized embeddings, set contextualized to False.
    """

    model_name: str = "openai-community/gpt2"

    # class attributes
    event_types: tp.Literal["Word", "Sentence"] = "Word"
    requirements: tp.ClassVar[tuple[str, ...]] = ("transformers>=4.29.2",)
    infra: MapInfra = MapInfra(
        timeout_min=25,
        gpus_per_node=1,
        cpus_per_task=10,
        min_samples_per_job=4096,
        version="v7",
    )

    # extractor attributes
    batch_size: int = 32
    contextualized: bool = True

    _max_length: int | None = pydantic.PrivateAttr(None)
    hf_config: HuggingFaceTextConfig = HuggingFaceTextConfig()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.event_types == "Sentence" and self.contextualized:
            msg = "Contextualized embeddings are not supported for Sentence events."
            raise ValueError(msg)

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return (
            ["batch_size"]
            + BaseStatic._exclude_from_cls_uid()
            + HuggingFaceMixin._exclude_from_cls_uid()
        )

    def _exclude_from_cache_uid(self) -> list[str]:
        excluded = ["frequency", "duration", "batch_size"]
        return (
            excluded
            + BaseStatic._exclude_from_cache_uid(self)
            + HuggingFaceMixin._exclude_from_cache_uid(self)
        )

    @property
    def tokenizer(self) -> tp.Any:
        tokenizer = self.processor
        if tokenizer.pad_token is None:
            # Use an existing token so model weights stay unchanged.
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_max_length(self) -> int | None:
        """Token truncation limit, falling back to the model's positional capacity."""
        if self._max_length is not None:
            return self._max_length
        tok_max = self.tokenizer.model_max_length
        # HF "infinite" sentinel (~1e30) would disable truncation (overflows OPT)
        if isinstance(tok_max, int) and tok_max < int(1e29):
            self._max_length = tok_max
            return self._max_length
        # learned-position table is sized capacity + offset, so the offset cancels
        config = self.model.config
        for attr in ("max_position_embeddings", "n_positions", "n_ctx"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                self._max_length = value
                return self._max_length
        return None

    def _get_timed_arrays(
        self,
        events: list[_ev.etypes.Word | _ev.etypes.Sentence],
        start: float,
        duration: float,
    ) -> tp.Iterable[TimedArray]:
        if self.contextualized and any(
            not getattr(event, "context", None) for event in events
        ):
            msg = (
                "Contextualized embeddings require non-empty context for all events. "
                "Set contextualized=False for context-free text."
            )
            raise ValueError(msg)
        # optimized fetch of multiple events compared to individual get_static calls:
        for event, latent in zip(events, self._get_data(events)):
            if self.cache_n_layers is not None:
                latent = self._aggregate_layers(latent)
            ta = TimedArray(
                frequency=0,
                duration=event.duration,
                start=event.start,
                data=latent,
            )
            yield ta

    @infra.apply(
        item_uid=lambda event: f"{event.text}_{getattr(event, 'context', '')}",
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArrayFile",
    )
    def _get_data(
        self, events: list[_ev.etypes.Word | _ev.etypes.Sentence]
    ) -> tp.Iterator[np.ndarray]:
        dataset = TextDataset(events)
        dloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Processing the data in batches
        if len(dloader) > 1:
            dloader = tqdm(dloader, desc="Computing word embeddings")  # type: ignore
        device = self.model_device
        with torch.no_grad():
            for target_words, context in dloader:
                # tokenize context
                with environment_variables(TOKENIZERS_PARALLELISM="false"):
                    text = context if self.contextualized else target_words
                    if isinstance(text, tuple):
                        # temporary fix for tokenizers==0.20.2
                        # https://github.com/huggingface/tokenizers/issues/1672
                        text = list(text)
                    if not all(text):
                        msg = f"Empty text or context for target_words {target_words!r}"
                        raise ValueError(msg)
                    inputs = self.tokenizer(
                        text,
                        add_special_tokens=False,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,  # beware to have set truncation_side="left" in init
                        max_length=self._get_max_length(),  # guard tokenizers reporting no real limit (e.g. OPT)
                    ).to(device)
                outputs = self.model(**inputs, output_hidden_states=True)
                if "hidden_states" in outputs:
                    states = outputs.hidden_states
                else:  # bart (encoder/decoder)
                    states = outputs.encoder_hidden_states + outputs.decoder_hidden_states
                hidden_states = torch.stack([layer.cpu() for layer in states])
                n_layers, n_batch, n_tokens, n_dims = hidden_states.shape  # noqa

                # attention_mask is the model's truth, hoisted to one device sync per batch
                n_pads_per_row: list[int] = (
                    (inputs["attention_mask"] == 0).sum(dim=1).tolist()
                )

                # -- for each target word, remove padding, and select target tokens
                for i, target_word in enumerate(target_words):
                    # select batch element
                    hidden_state = hidden_states[:, i]  # n_layers x tokens x embd

                    n_pads = n_pads_per_row[i]
                    if n_pads > 0:
                        hidden_state = hidden_state[:, :-n_pads]

                    # select tokens that belong to the target word
                    if self.contextualized:
                        # inputs already has the tokenized context; subtract
                        # the prefix token count to get the target word's tokens
                        prefix = context[i][: -len(target_word)].rstrip()
                        n_prefix = (
                            len(self.tokenizer.encode(prefix, add_special_tokens=False))
                            if prefix
                            else 0
                        )
                        n_target = hidden_state.shape[1] - n_prefix
                        word_state = hidden_state[:, -max(1, n_target) :]
                    else:
                        word_state = hidden_state
                    # aggregate in cuda for smaller data transfer:
                    word_state = self._aggregate_tokens(word_state)
                    out = word_state.cpu().numpy()
                    if self.cache_n_layers is None:
                        out = self._aggregate_layers(out)
                    if np.isnan(out).any():
                        msg = f"NaN in output for target_word {target_word} with context {context}"
                        raise ValueError(msg)
                    yield out
                # erase variables / free memory
                del hidden_states, hidden_state, word_state, states, outputs, inputs
                if device.type == "cuda":
                    torch.cuda.empty_cache()
