# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from collections import deque
from functools import lru_cache

import numpy as np
import pandas as pd
import pydantic
import torch
from tqdm import tqdm

from neuralset import segments as _segs

from .. import etypes as ev
from ..study import EventsTransform
from .utils import MISSING_SENTENCE, TextWordMatcher, _extract_sentences, parse_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EnsureTexts — canonical way to guarantee Text events exist
# ---------------------------------------------------------------------------


@lru_cache
def _get_punct_model() -> tp.Any:
    from deepmultilingualpunctuation import PunctuationModel

    return PunctuationModel()


class EnsureTexts(EventsTransform):
    """Create Text events from Words if not already present.

    Parameters
    ----------
    punctuation : str or None
        ``"spacy"`` — capitalize first letter of each sentence and add
        language-appropriate sentence-ending punctuation via spaCy
        sentence segmentation.
        ``"fullstop"`` — ``oliverguhr/fullstop-punctuation-multilang-large``
        DL punctuation restoration (requires
        ``deepmultilingualpunctuation``).  May need a GPU.
        ``None`` — plain space-join (no punctuation added).
    """

    punctuation: tp.Literal["spacy", "fullstop"] | None = "spacy"

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        text_names = ev.EventTypesHelper("Text").names
        if events.type.isin(text_names).any():
            return events
        word_names = ev.EventTypesHelper("Word").names
        words = events.loc[events.type.isin(word_names)].sort_values(
            ["timeline", "start"]
        )
        if words.empty:
            return events
        text_rows: list[dict[str, tp.Any]] = []
        for timeline, word_group in words.groupby("timeline"):
            raw = " ".join(word_group.text.values)
            text_str = self._punctuate(raw, word_group)
            start = float(word_group.start.min())
            stop = float((word_group.start + word_group.duration).max())
            text_rows.append(
                dict(
                    type="Text",
                    start=start,
                    duration=stop - start,
                    timeline=timeline,
                    text=text_str,
                    language=(
                        word_group.language.iloc[0]
                        if "language" in word_group.columns
                        else ""
                    ),
                )
            )
        return pd.concat([events, pd.DataFrame(text_rows)], ignore_index=True)

    def _punctuate(self, raw: str, words_df: pd.DataFrame) -> str:
        if self.punctuation is None:
            return raw
        if self.punctuation == "fullstop":
            return _add_punctuation(raw)
        lang = words_df.language.iloc[0] if "language" in words_df.columns else ""
        doc = parse_text(raw, lang)
        periods = {
            "chinese": "\u3002",
            "zh": "\u3002",
            "japanese": "\u3002",
            "ja": "\u3002",
        }
        period = periods.get(lang, ".")
        sep = period + " "
        parts: list[str] = []
        for sent in doc.sents:
            s = sent.text.strip()
            if s:
                s = s[0].upper() + s[1:]
            parts.append(s)
        text = sep.join(parts)
        if text and not text.endswith(period):
            text += period
        return text


@lru_cache
def _add_punctuation(text: str) -> str:
    """Cached DL punctuation restoration (shared across subjects with same text)."""
    model = _get_punct_model()
    return model.restore_punctuation(text)


class AddSentenceToWords(EventsTransform):
    """
    Adds sentence-level information to word events based on Text rows.

    This transform processes a DataFrame containing word-level (Word) and text-level (Text) events.
    For each sentence found in the Text rows, it:

    1. Creates a new Sentence row for each sentence.
    2. Assigns `sentence` and `sentence_char` annotations to Word rows to indicate
       which sentence each word belongs to, and which character the word starts at in the sentence.

    Parameters
    ----------
    max_unmatched_ratio : float
        Maximum allowed ratio of word rows that do not match any sentence.
        Raises an error if this ratio is exceeded.
    override_sentences : bool, default=False
        Whether to replace existing Sentence rows if they are already present.
    """

    max_unmatched_ratio: float = 0.0  # raises if did not match enough words
    override_sentences: bool = False

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["max_unmatched_ratio"]

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.max_unmatched_ratio < 0 or self.max_unmatched_ratio >= 1:
            raise ValueError(f"{self.max_unmatched_ratio=}, must be in [0, 1)")

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        """Add sentence information to each word event by parsing the
        corresponding full text"""
        if "Sentence" in events.type.unique():
            if not self.override_sentences:
                msg = "Sentence already present in events dataframe"
                logger.debug(msg)
                return events
            events = events[events.type != "Sentence"]
        if "timeline" in events.columns and len(events.timeline.unique()) > 1:
            timelines = []
            desc = "Add sentence to Word based on Text"
            # 1 timeline at a time
            tl_dfs = events.groupby("timeline", sort=False)
            for _, subevents in tqdm(tl_dfs, desc=desc, mininterval=10):
                timelines.append(self._run(subevents))
            return pd.concat(timelines, ignore_index=True)
        if "Word" not in events.type.unique():
            logger.info("No Word events found, skipping")
            return events
        contexts = events.loc[events.type == "Text"]
        if contexts.empty:
            msg = "No Text event in dataframe, add it in the study "
            msg += "or use 'EnsureTexts' transform to create Text events from Words"
            raise RuntimeError(msg)
        events = events.copy(deep=True)  # avoid border effect
        wtypes = ev.EventTypesHelper("Word")
        words = events[events.type.isin(wtypes.names)]
        events.loc[:, "sentence_char"] = np.nan
        events["sentence"] = ""

        sentences = []
        for context in contexts.itertuples():
            # find words that are enclosed in this context (requires unique timeline)
            encl = _segs.find_enclosed(
                events,
                start=float(context.start),  # type: ignore[arg-type]
                duration=float(context.duration),  # type: ignore[arg-type]
            )
            sub = events.loc[encl]
            sel = sub[sub.type.isin(wtypes.names)].index
            if not len(sel):
                raise ValueError("No word overlapping with context")
            wordseq = words.loc[sel].text.tolist()
            lang = getattr(context, "language", None)
            if not isinstance(lang, str):
                raise ValueError(f"Need language for Text field {context}")
            context_text = getattr(context, "text", None)
            if not isinstance(context_text, str):
                raise ValueError(f"Need text for Text field {context}")
            info = pd.DataFrame(
                TextWordMatcher(context_text, language=lang).match(wordseq),
                index=sel,
            )
            events.loc[sel, info.columns] = info
            # create sentence events
            context_sentences = [s.to_dict() for s in _extract_sentences(events)]
            subject = getattr(context, "subject", None)
            if subject is not None:
                for s in context_sentences:
                    s["subject"] = subject
            sentences.extend(context_sentences)
        sentences = [s for s in sentences if s["text"] != MISSING_SENTENCE]
        sentence_df = pd.DataFrame(sentences)
        events = pd.concat([events, sentence_df], ignore_index=True)
        events = events.sort_values("start")
        events = events.reset_index(drop=True)

        words = events[events.type.isin(wtypes.names)]
        if len(words) == 0:
            return events
        ratio = sum(not s or not isinstance(s, str) for s in words.sentence) / len(words)
        if ratio > self.max_unmatched_ratio:
            max_unmatched_ratio = self.max_unmatched_ratio
            cls = self.__class__.__name__
            msg = f"Ratio of unmatched words is {ratio:.4f} on {len(words)} words "
            msg += f"while {cls}.{max_unmatched_ratio=}"
            raise RuntimeError(msg)
        return events


class AddContextToWords(EventsTransform):
    """Add a context field to the events dataframe, for each word event, by concatenating
    the sentence fields.

    .. warning::
       **Unstable API** — the context representation (per-word concatenated strings)
       will be replaced with compact indices in a future release.

    Parameters
    ----------
    sentence_only: bool
        only use current sentence as context
    max_context_len: None or int
        if not None, caps the context len to a given number of words (counted through whitespaces)
    split_field: str
        field on which to reset contexts. If empty, context is only reset for new timelines.
    """

    sentence_only: bool = True  # only use context from current sentence
    max_context_len: int | None = None  # cut the context after given number of words
    split_field: str = "split"

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if hasattr(events, "context"):  # make sure it is typed as str
            events["context"] = events["context"].fillna("").astype(str)
        wtypes = ev.EventTypesHelper("Word")
        words = events.loc[events.type.isin(wtypes.names), :]
        last_word: tp.Any = None
        contexts = []
        desc = "Add context to words"
        worditer: tp.Iterator[ev.Word] = words.itertuples(index=False)  # type: ignore
        sfield = self.split_field
        if sfield and sfield not in words.columns:
            raise ValueError(f"split_field {sfield!r} is not part of dataframe columns")
        for word in tqdm(worditer, total=len(words), desc=desc, mininterval=10):
            if last_word is not None:
                # check order as a security
                same_timeline = last_word.timeline == word.timeline
                if same_timeline and (word.start < last_word.start):
                    msg = "Words are not in increasing order "
                    msg += f"({word} follows {last_word})"
                    raise ValueError(msg)
                # reset if split differs or timeline differs
                splits = [getattr(w, sfield, "") for w in (word, last_word)]
                if splits[0] != splits[1] or not same_timeline:
                    last_word = None  # restart
            # word is not correctly match, let's not add a context
            has_sent = isinstance(word.sentence, str) and word.sentence
            if word.sentence_char is None or np.isnan(word.sentence_char) or not has_sent:
                contexts.append("")
                continue
            # first word, restart parts
            if last_word is None:
                past_parts: deque[str] = deque(maxlen=self.max_context_len)
                start_char = 0  # assumes not splitting within sentences
            # not first word from now on
            if last_word is not None:
                non_increasing_char = word.sentence_char <= last_word.sentence_char
                if word.sentence != last_word.sentence or non_increasing_char:
                    # new sentence
                    if self.sentence_only:
                        past_parts.clear()
                    else:  # add end of sentence:
                        past_parts[-1] += last_word.sentence[start_char:]
                    start_char = 0
                elif past_parts:  # same sentence
                    # append up to current character to the last context part
                    last_char = int(word.sentence_char)
                    past_parts[-1] += word.sentence[start_char:last_char]
                    start_char = last_char
            # reset context with timeline + check ordering for safety
            last_char = int(word.sentence_char) + len(word.text)
            new = word.sentence[start_char:last_char]
            contexts.append("".join(past_parts) + new)
            past_parts.append(new)
            start_char = last_char
            last_word = word
        # set new context column
        events.loc[words.index, "context"] = contexts
        return events


class AddConcatenationContext(EventsTransform):
    """
    Adds contextual information to events by concatenating previous events of the same type.

    .. warning::
       **Unstable API** — the context representation will be replaced with compact
       indices in a future release.

    This transform iterates over events of a specified type (default "Word") and
    creates a `context` column in the DataFrame. For each event, the context
    consists of the concatenated texts of all previous events in the same chunk,
    where chunks are determined by timeline changes, split changes, or sentence
    boundaries (if `sentence_only=True`). Optionally, the context length can be
    limited by `max_context_len`.

    .. note::

        if an event is missing (eg. a previous word event) it will be missing in the context.
        Use AddContextToWords for a more careful consideration of context, and the addition of punctuation.

    Parameters
    ----------
    event_type : str, default="Word"
        Type of event to use for building context.
    sentence_only : bool, default=False
        If True, chunks are defined by sentence boundaries; otherwise, by
        timeline and split changes.
    max_context_len : int | None, default=None
        Maximum number of previous events to include in the context. If None,
        all previous events in the chunk are used.
    split_field : str, default="split"
        Column name used to detect split boundaries when creating chunks.
    """

    event_type: str = "Word"
    sentence_only: bool = False
    max_context_len: int | None = None
    split_field: str = "split"

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.event_type not in ev.Event._CLASSES:
            raise TypeError(f"Event type {self.event_type} not found in events")

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        """In place: adds concatenation of previous words to context."""
        from collections import deque

        words = events.loc[events.type == self.event_type].copy()

        # identify chunks
        previous = words.copy().shift(1)
        timeline_change = words.timeline.astype(str) != previous.timeline.astype(str)
        chunk_change = timeline_change
        if self.split_field in words.columns:
            prev_split = previous[self.split_field].astype(str)
            split_change = words[self.split_field].astype(str) != prev_split
            chunk_change = chunk_change | split_change
        if self.sentence_only:
            sentence_change = words.sequence_id != previous.sequence_id
            chunk_change = chunk_change | sentence_change
        words.loc[words.index, "chunk"] = np.cumsum(chunk_change)

        for _, df in words.groupby("chunk", sort=False):
            if (df.start.diff() < 0).any():  # type: ignore[operator]
                raise ValueError("Events should be ordered by start time")
            texts = df["text"].tolist()
            chunk_contexts: list[str] = []
            # Build cumulative context strings, optionally capped to a sliding window.
            window: deque[str] | list[str] = (
                deque(maxlen=self.max_context_len)
                if self.max_context_len is not None
                else []
            )
            for word in texts:
                window.append(word)
                chunk_contexts.append(" ".join(window))

            events.loc[df.index, "context"] = chunk_contexts

        return events


class AddSummary(EventsTransform):
    """
    Generate concise summaries for Text events using a pretrained language model.

    This transform processes events of type "Text" and generates a summary for
    each using a large language model (LLM). Summaries are added as new events
    of type "Summary" in the DataFrame. The summarization prompt ensures that
    the model returns exactly the requested number of sentences with a precise
    description of the content.

    Parameters
    ----------
    model_name : str, default="meta-llama/Llama-3.2-3B-Instruct"
        Hugging Face model identifier used for text summarization.
    n_sentences_requested : int, default=3
        Number of sentences to include in each summary.
    """

    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    n_sentences_requested: int = 3
    _pipeline: tp.Any = pydantic.PrivateAttr()

    @property
    def pipeline(self):
        from transformers import pipeline

        if not hasattr(self, "_pipeline"):
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        return self._pipeline

    def summarize(self, text):
        prompt = f"""Summarize the following text in {self.n_sentences_requested} sentences.
        Be as precise about what is going on.
        Simply return the summary, without any introduction.
        """
        messages = [
            {"role": "system", "content": "You are a professional text summarizer."},
            {"role": "user", "content": prompt + text},
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=256,
        )
        return outputs[0]["generated_text"][-1]["content"]

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        texts = events.loc[events.type == "Text"]
        summaries = []
        desc = "Add summary to Text events"
        for text in tqdm(texts.itertuples(), total=len(texts), desc=desc):
            text_str = getattr(text, "text", None)
            if not isinstance(text_str, str):
                continue
            summarized = self.summarize(text_str)
            summary = text._replace(type="Summary", text=summarized)  # type: ignore
            summaries.append(summary)
        events = pd.concat([events, pd.DataFrame(summaries)], ignore_index=True)
        return events


class AddPhonemes(EventsTransform):
    """
    Add phoneme information to events.
    """

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class AddPartOfSpeech(EventsTransform):
    """
    Add Part-Of-Speech (POS) tags to events.
    """

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
