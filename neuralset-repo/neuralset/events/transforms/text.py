# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp
from functools import lru_cache

import numpy as np
import pandas as pd
import pydantic
import torch
from tqdm import tqdm

from neuralset import segments as _segs

from .. import etypes as ev
from ..study import EventsTransform
from .utils import (
    MISSING_SENTENCE,
    TextWordMatcher,
    _extract_sentences,
    parse_text,
    validate_event_types,
    validate_split_labels,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EnsureTexts — canonical way to guarantee Text events exist
# ---------------------------------------------------------------------------


_PUNCT_CHECKPOINT = "oliverguhr/fullstop-punctuation-multilang-large"


@lru_cache
def _get_punct_model() -> tp.Any:
    """Punctuation restoration model, reusing only the package's decoding."""
    from deepmultilingualpunctuation import PunctuationModel
    from transformers import pipeline

    class _Pinned(PunctuationModel):  # type: ignore[misc]
        # upstream builds its pipeline with grouped_entities, dropped in transformers 5
        def __init__(self) -> None:
            self.pipe = pipeline(
                "token-classification",
                _PUNCT_CHECKPOINT,
                aggregation_strategy="none",
                device=0 if torch.cuda.is_available() else -1,
            )

    return _Pinned()


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
    2. Assigns `sentence`, `sentence_char`, and `text_char` annotations to Word rows.

    Parameters
    ----------
    max_unmatched_ratio : float
        Maximum ratio of words without a character-positioned match.
    tolerance : float
        Tolerance (in seconds) for Text event start and end times.
    override_sentences : bool, default=False
        Whether to replace existing Sentence rows if they are already present.
    """

    max_unmatched_ratio: float = 0.0  # raises if did not match enough words
    tolerance: float = 0.0
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
        events["text_char"] = np.nan

        for context in contexts.itertuples():
            # find words that are enclosed in this context (requires unique timeline)
            encl = _segs.find_enclosed(
                events,
                start=float(context.start),  # type: ignore[arg-type]
                duration=float(context.duration),  # type: ignore[arg-type]
                tolerance=self.tolerance,
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
        # create sentence events; standardize_events backfills BIDS/study
        # from sibling rows in the same timeline.
        sentences = [s.to_dict() for s in _extract_sentences(events)]
        sentences = [s for s in sentences if s["text"] != MISSING_SENTENCE]
        sentence_df = pd.DataFrame(sentences)
        events = pd.concat([events, sentence_df], ignore_index=True)
        events = events.sort_values("start")
        events = events.reset_index(drop=True)

        words = events[events.type.isin(wtypes.names)]
        if len(words) == 0:
            return events
        ratio = int(words["text_char"].isna().sum()) / len(words)
        if ratio > self.max_unmatched_ratio:
            max_unmatched_ratio = self.max_unmatched_ratio
            cls = self.__class__.__name__
            msg = (
                f"Ratio of words without text_char is {ratio:.4f} on {len(words)} words "
            )
            msg += f"while {cls}.{max_unmatched_ratio=}"
            raise RuntimeError(msg)
        return events


# ---------------------------------------------------------------------------
# Removed: per-word context strings, replaced by AddAlignedTexts + HuggingFaceText
# ---------------------------------------------------------------------------


def _removed_context_message(name: str, data: tp.Any, splits: bool = False) -> str:
    """Error of the removed transforms, given their config ``data``."""
    params = data if isinstance(data, dict) else {}
    msg = f"{name} was removed: events no longer get a context string. "
    if splits:
        msg += "For its sentences and splits, use AddSentenceToWords and "
        msg += "AssignSentenceSplit. "
    event_type = params.get("event_type", "Word")
    if event_type != "Word":
        msg += f"{event_type} events can be embedded alone (e.g. HuggingFaceSentence); "
        msg += "only words are embedded in context. "
    msg += (
        "For contextual word embeddings, add AddAlignedTexts to the events "
        "transforms (text_type='Sentence' for sentence_only=True) and embed its "
        "AlignedText events with HuggingFaceText, whose context sets the windows of "
        "words: context={'words': N, 'min_context_words': N - 1} reads every word "
        "with the N words of max_context_len=N (one forward pass per word), "
        "context={'words': N} with N/2 to N words (two passes per N words)."
    )
    if params.get("split_field", "split"):
        msg += (
            " This context stops at the boundaries of the Text events, not where "
            "the split changes: to keep the words of a split from seeing those of "
            "another, split the Texts at split boundaries before AddAlignedTexts."
        )
    return msg


class AddContextToWords(EventsTransform):
    """Removed: raises with the replacement, :class:`AddAlignedTexts` (one
    ``AlignedText`` per Text, or per sentence with ``text_type="Sentence"``)
    embedded by :class:`~neuralset.extractors.text.HuggingFaceText` in windows of
    words (its ``context``)."""

    @pydantic.model_validator(mode="before")
    @classmethod
    def _removed(cls, data: tp.Any) -> tp.Any:
        raise ValueError(_removed_context_message(cls.__name__, data))


class AddConcatenationContext(EventsTransform):
    """Removed: raises with the replacement, :class:`AddAlignedTexts` (see
    :class:`AddContextToWords`)."""

    @pydantic.model_validator(mode="before")
    @classmethod
    def _removed(cls, data: tp.Any) -> tp.Any:
        raise ValueError(_removed_context_message(cls.__name__, data))


class AddAlignedTexts(EventsTransform):
    """Build :class:`~neuralset.events.etypes.AlignedText` events for contextual
    word embeddings with :class:`~neuralset.extractors.text.HuggingFaceText`.

    Words are matched to their last enclosing Text; unmatched Words are omitted
    with a warning. Missing Texts are built from Words using spaCy punctuation.
    Text rows are kept, and existing Sentence rows are ignored.

    Parameters
    ----------
    drop_words : bool
        Remove source Word rows when True. Default: False.
    text_type : str
        ``"Text"`` (default) or ``"Sentence"``: group words by Text or by sentences
        derived from its content, before applying any time-based split boundaries.
    event_type_to_split_by : str or None
        Registered event type carrying ``split`` labels, e.g. ``"Ieeg"``. Label
        changes define time boundaries at gap midpoints, as in
        :class:`~neuralset.events.transforms.chunking.ChunkEvents`, so text can
        share recording/audio split times without needing Sentence events or their
        split labels, even within a sentence. Custom registered event types and
        arbitrary labels are supported; selected events need a ``split`` column
        with non-missing labels.

        Words belong to the section where they start; a boundary-starting word
        goes to the following section. Output event spans and word durations are
        clipped to the section, without changing source Word timings. Context
        windows contain only that section's words. Leading/trailing words stay
        in the first/last section. ``None`` (default), no selected events or no
        label changes leave the timeline unsplit.
    """

    drop_words: bool = False
    text_type: tp.Literal["Text", "Sentence"] = "Text"
    event_type_to_split_by: str | None = None

    _check_event_types = pydantic.field_validator("event_type_to_split_by")(
        validate_event_types
    )

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        word_names = ev.EventTypesHelper("Word").names
        if not events.type.isin(word_names).any():
            logger.info("No Word events found, skipping")  # as AddSentenceToWords
            return events
        events = self._ensure_texts(events)
        text_names = ev.EventTypesHelper("Text").names
        new_rows: list[dict[str, tp.Any]] = []
        n_words = n_unmatched = n_empty_timelines = 0
        for _, tl_events in events.groupby("timeline", sort=False):
            words = tl_events.loc[tl_events.type.isin(word_names)]
            if words.empty:
                continue
            texts = tl_events.loc[tl_events.type.isin(text_names)]
            texts = texts.sort_values("start", kind="stable")
            owners, text_chars, sentence_spans = self._match(words, texts)
            matched = text_chars >= 0
            n_words += len(words)
            n_unmatched += int((~matched).sum())
            n_empty_timelines += not matched.any()
            # split section of each word (a single one without event_type_to_split_by)
            bounds = [-np.inf, *self._split_boundaries(tl_events), np.inf]
            sections = np.searchsorted(bounds[1:-1], words.start.to_numpy(), side="right")
            for k, (_, text) in enumerate(texts.iterrows()):
                for c in range(len(bounds) - 1):
                    mine = matched & (owners == k) & (sections == c)
                    if not mine.any():
                        continue
                    section = (bounds[c], bounds[c + 1])
                    order = np.argsort(text_chars[mine], kind="stable")
                    sel = words.loc[mine].iloc[order]
                    char_starts = text_chars[mine][order]
                    if self.text_type == "Text":
                        row = self._aligned_text(text, sel, char_starts, section=section)
                        new_rows.append(row)
                        continue
                    spans = sentence_spans[mine][order]
                    for sent_start, sent_stop in np.unique(spans, axis=0):
                        in_sentence = spans[:, 0] == sent_start
                        sentence = (int(sent_start), int(sent_stop))
                        row = self._aligned_text(
                            text,
                            sel.loc[in_sentence],
                            char_starts[in_sentence],
                            sentence,
                            section,
                        )
                        new_rows.append(row)
        if n_unmatched:
            # with allow_missing, the extractor returns zeros for them
            logger.warning(
                "AddAlignedTexts: %d of %d Words (%.1f%%) match no character of a "
                "Text and get no AlignedText features; %d timelines have no "
                "matched Word",
                n_unmatched,
                n_words,
                100 * n_unmatched / n_words,
                n_empty_timelines,
            )
        if self.drop_words:
            events = events.loc[~events.type.isin(word_names)]
        if new_rows:
            events = pd.concat([events, pd.DataFrame(new_rows)], ignore_index=True)
        return events

    def _split_boundaries(self, events: pd.DataFrame) -> list[float]:
        """Times where the ``split`` of ``event_type_to_split_by`` changes in one
        timeline's ``events``, at the middle of the gap (as ``ChunkEvents``)."""
        if self.event_type_to_split_by is None:
            return []
        rows = events.loc[events.type == self.event_type_to_split_by]
        if rows.empty:
            return []
        rows = rows.sort_values("start")
        splits = validate_split_labels(rows).astype(str).to_numpy()
        stops = (rows.start + rows.duration).to_numpy()
        starts = rows.start.to_numpy()
        change = np.flatnonzero(splits[1:] != splits[:-1])
        return [float((stops[i] + starts[i + 1]) / 2) for i in change]

    @staticmethod
    def _aligned_text(
        text: pd.Series,
        words: pd.DataFrame,
        char_starts: np.ndarray,
        sentence: tuple[int, int] | None = None,
        section: tuple[float, float] = (-np.inf, np.inf),
    ) -> dict[str, tp.Any]:
        """AlignedText row of the Words of a Text (sorted by character), or of
        those of its sentence spanning characters ``sentence``, within the split
        ``section`` (start, stop) of the timeline."""
        content: str = text.text
        word_stops = (words.start + words.duration).to_numpy()
        start, stop = float(words.start.min()), float(word_stops.max())
        if sentence is None:  # spans the Text too
            start = min(float(text.start), start)
            stop = max(float(text.start + text.duration), stop)
        else:
            content = content[sentence[0] : sentence[1]].rstrip()
            char_starts = char_starts - sentence[0]
        start, stop = max(start, section[0]), min(stop, section[1])
        # the word text is assumed to be in the Text, from its first character
        char_stops = char_starts + words.text.str.len().to_numpy()
        row = text.to_dict()
        row.update(
            type="AlignedText",
            text=content,
            start=start,
            duration=stop - start,
            word_starts=words.start.to_numpy() - start,
            word_durations=np.minimum(
                words.duration.to_numpy(), section[1] - words.start.to_numpy()
            ),
            word_char_starts=char_starts,
            word_char_stops=np.minimum(char_stops, len(content)),
        )
        return ev.AlignedText.from_dict(row).to_dict()

    @staticmethod
    def _ensure_texts(events: pd.DataFrame) -> pd.DataFrame:
        """Add Texts built from the Words to the timelines without any Text."""
        text_names = ev.EventTypesHelper("Text").names
        is_text = events.type.isin(text_names)
        has_text = events.timeline.isin(events.loc[is_text, "timeline"].unique())
        if has_text.all():
            return events
        texts = EnsureTexts()(events.loc[~has_text])
        texts = texts.loc[texts.type.isin(text_names)]
        return pd.concat([events, texts], ignore_index=True)

    @staticmethod
    def _match(
        words: pd.DataFrame, texts: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Index of the last Text enclosing each word (-1 if none), character
        offset of the word in that Text (-1 if unmatched), and (start, stop)
        characters of its sentence in the Text (spaCy, as ``AddSentenceToWords``)."""
        word_starts = words.start.to_numpy()
        word_stops = word_starts + words.duration.to_numpy()
        text_starts = texts.start.to_numpy()
        text_stops = text_starts + texts.duration.to_numpy()
        languages = texts.get("language", pd.Series("", index=texts.index))
        owners = np.full(len(words), -1)
        text_chars = np.full(len(words), -1)
        sentence_spans = np.full((len(words), 2), -1)
        for k, (start, stop, content, language) in enumerate(
            zip(text_starts, text_stops, texts.text, languages)
        ):
            enclosed = (word_starts >= start) & (word_stops <= stop)
            if not enclosed.any():
                continue
            matcher = TextWordMatcher(
                content, language=language if isinstance(language, str) else ""
            )
            matches = matcher.match(words.text.to_numpy()[enclosed].tolist())
            owners[enclosed] = k
            text_chars[enclosed] = [
                -1 if m.get("text_char") is None else m["text_char"] for m in matches
            ]
            sentence_spans[enclosed] = [
                (-1, -1)
                if m.get("text_char") is None
                else (
                    m["text_char"] - m["sentence_char"],
                    m["text_char"] - m["sentence_char"] + len(m["sentence"]),
                )
                for m in matches
            ]
        return owners, text_chars, sentence_spans


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
