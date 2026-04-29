# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
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
from neuralset.extractors.base import BaseStatic, HuggingFaceMixin

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
        english="en", french="fr", spanish="es", dutch="nl", chinese="zh"
    )

    # pylint: disable=unused-argument
    def get_embedding(self, text: str, language: str = "") -> np.ndarray:
        from wordfreq import zipf_frequency  # noqa

        value = zipf_frequency(text, self.LANGUAGES.get(self.language, self.language))
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
    pretrained: bool or "part-reversal"
        use pretrained model if True, untrained intial model if False, or custom
        scrambling of the model pretrained weights if "part-reveral"

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
    pretrained: bool | tp.Literal["part-reversal"] = True

    # initialized later
    _model: nn.Module = pydantic.PrivateAttr()
    _tokenizer: tp.Any = pydantic.PrivateAttr()

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
    def model(self) -> nn.Module:
        if not hasattr(self, "_model"):
            from transformers import AutoTokenizer

            kwargs: dict[str, tp.Any] = {}
            if self.model_name.lower().startswith("microsoft/phi"):
                kwargs["trust_remote_code"] = True
            # pinned for `_get_data`'s slicing: target at tail, pads trailing
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                truncation_side="left",
                padding_side="right",
                **kwargs,
            )
            # tokens
            if self._tokenizer.pad_token is None:
                # previously:
                # self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                # self._model.resize_token_embeddings(len(self._tokenizer))
                # simpler to use existing EOS token:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            try:
                self._model = self._load_model()
            except Exception as e:
                # some exceptions (AttributeError in particular)
                # are hidden by pydantic
                raise RuntimeError("Model loading went wrong") from e
        return self._model

    def _load_model(self, **kwargs: tp.Any) -> nn.Module:
        # do all the loading in a function without modifying
        # self, so that if an intermediate bug is caught, the
        # model is never in cache
        # (pydantic seems to catch some of the exceptions in the
        # model property, which creates weird silent bugs)
        Model: tp.Any  # ignore typing as we'll override the imports
        from transformers import AutoModel as Model

        if "t5" in self.model_name or "bert" in self.model_name:
            from transformers import AutoModelForTextEncoding as Model
        elif "Phi-3" in self.model_name:
            from transformers import AutoModelForCausalLM as Model
        elif "Llama-3.2-11B-Vision" in self.model_name:
            from transformers import MllamaForConditionalGeneration as Model
        # instantiate
        if self.device == "accelerate":
            kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
        model = Model.from_pretrained(self.model_name, **kwargs)
        if not self.pretrained:
            rawmodel = Model.from_config(model.config)
            with torch.no_grad():
                for p1, p2 in itertools.zip_longest(
                    model.parameters(), rawmodel.parameters()
                ):
                    p1.data = p2.to(p1)
        elif self.pretrained == "part-reversal":
            with torch.no_grad():
                for p in model.parameters():
                    part_reversal(p)
        if self.device != "accelerate":
            model.to(self.device)
        model.eval()
        return model

    @property
    def tokenizer(self) -> tp.Any:
        self.model
        return self._tokenizer

    def _get_timed_arrays(
        self,
        events: list[_ev.etypes.Word | _ev.etypes.Sentence],
        start: float,
        duration: float,
    ) -> tp.Iterable[TimedArray]:
        # Backward compatibility check for contextualized for people that used it as a default without context
        # : raise if contextualized is True but no context is provided
        if self.contextualized and any(
            not getattr(event, "context", None) for event in events
        ):
            msg = "Contextualized embeddings require a context. Please provide a context for all events or set contextualized to False. This might happen since contextualized is now True by default. "
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
        device = "auto" if self.device == "accelerate" else self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
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
                if self.device == "accelerate" and torch.cuda.is_available():
                    # in case of multi-GPU models, explicitely empty cache "just in case"
                    torch.cuda.empty_cache()


def part_reversal(tensor: torch.Tensor) -> None:
    """reverse coefficients by parts, to scramble the weights of a network
    without changing its statistics and with low enough memory footprint
    """
    x = tensor.view(-1)
    # use around 14 splits, and not an integer to avoid alignment with data
    sq200 = 10 * np.sqrt(2)
    num = max(len(x) / sq200, sq200)
    first = 0
    last = num
    while first < len(x):
        x[first : int(last)] = reversed(x[first : int(last)])  # type: ignore
        last += num
        first = int(last)
