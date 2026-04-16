# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import hashlib
import logging
import random
import re
import typing as tp
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from neuralset import segments as _segs
from neuralset import utils as _ns_utils

from .. import etypes as ev

logger = logging.getLogger(__name__)


MISSING_SENTENCE = "# MISSING SENTENCE #"


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------


@lru_cache
def parse_text(text: str, language: str = "") -> tp.Any:
    nlp = _ns_utils.get_spacy_model(language=language)
    return nlp(text)


def _extract_sentences(events) -> list[ev.Sentence]:
    """Extract sentence events from the words with sentence annotations"""
    wtypes = ev.EventTypesHelper("Word")
    words_df = events.loc[events.type.isin(wtypes.names), :]
    sentences = []
    words: list[tp.Any] = []
    eps = 1e-6
    for k, word in enumerate(words_df.itertuples(index=False)):
        if words and words[-1].timeline == word.timeline:
            if word.start < words[-1].start:
                raise ValueError(
                    f"Words are not sorted within a timeline ({words!r} and then {word!r}"
                )
        sentence_end = False
        if k == len(words_df) - 1:  # last word event
            sentence_end = True
            words.append(word)
        if words:
            sentence_end |= words[-1].timeline != word.timeline
            sentence_end |= word.sentence != words[-1].sentence
            sentence_end |= word.sentence_char <= words[-1].sentence_char
            if sentence_end:
                w0 = words[0]
                text = w0.sentence
                if not (isinstance(text, str) and text):
                    text = MISSING_SENTENCE
                sentences.append(
                    ev.Sentence(
                        start=w0.start - eps,
                        duration=words[-1].start
                        + words[-1].duration
                        - w0.start
                        + 2 * eps,
                        timeline=w0.timeline,
                        text=text,
                    )
                )
                words = []
        words.append(word)
    return sentences


_LIGATURES = str.maketrans({"œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE"})


def _normalize_with_positions(text: str) -> tuple[str, list[int]]:
    """Normalize *text* (lowercase, decompose ligatures, strip accents)
    and return a mapping from each normalized character index to its
    original position in *text*.
    """
    out: list[str] = []
    orig: list[int] = []
    for i, ch in enumerate(text.translate(_LIGATURES)):
        for c in unicodedata.normalize("NFKD", ch):
            if not unicodedata.combining(c):
                out.append(c)
                orig.append(i)
    return "".join(out), orig


class TextWordMatcher:
    """Match annotated words to character positions in a spaCy-parsed text.

    Two-phase strategy:
    1. Token-level Levenshtein alignment (spaCy tokens vs input words).
    2. Character-level fallback for unmatched words, using the raw text
       spans between surrounding matched tokens.

    After matching, each word gets ``sentence``, ``sentence_char``, and
    ``text_char`` when a reliable match is found.  Words sandwiched
    between neighbours in the same sentence inherit the sentence label.
    """

    _PUNCT_RE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)

    def __init__(self, text: str, language: str = "") -> None:
        self.doc = parse_text(text, language=language)
        self.text = text
        self.tokens: list[tp.Any] = [tok for sent in self.doc.sents for tok in sent]

    @staticmethod
    def normalize(word: str) -> str:
        """Lowercase, strip accents/ligatures, and strip leading/trailing punctuation."""
        text, _ = _normalize_with_positions(word.lower())
        return TextWordMatcher._PUNCT_RE.sub("", text)

    def match(self, words: tp.Sequence[str]) -> list[dict[str, tp.Any]]:
        token_strs = [self.normalize(t.text) for t in self.tokens]
        tok_matched, word_matched = _ns_utils.match_list(
            token_strs, [self.normalize(w) for w in words]
        )
        info: list[dict[str, tp.Any]] = [{"_word": w} for w in words]
        for ti, wi in zip(tok_matched, word_matched):
            info[wi]["_tok"] = ti

        self._resolve_gaps(info)
        self._finalize(info)
        return info

    # -- char-level fallback ------------------------------------------------

    def _resolve_gaps(self, info: list[dict[str, tp.Any]]) -> None:
        """Find contiguous unmatched runs and resolve each via char-level matching."""
        gaps: list[tuple[int, int, int | None, int | None]] = []
        prev_tok: int | None = None
        gap_start: int | None = None
        for k, i in enumerate(info):
            if "_tok" in i:
                if gap_start is not None:
                    gaps.append((gap_start, k, prev_tok, i["_tok"]))
                    gap_start = None
                prev_tok = i["_tok"]
            elif gap_start is None:
                gap_start = k
        if gap_start is not None:
            gaps.append((gap_start, len(info), prev_tok, None))

        for gs, ge, left_tok, right_tok in gaps:
            self._resolve_one_gap(info[gs:ge], left_tok, right_tok)

    def _resolve_one_gap(
        self,
        gap: list[dict[str, tp.Any]],
        left_tok: int | None,
        right_tok: int | None,
    ) -> None:
        """Character-level Levenshtein fallback for one gap of unmatched words."""
        char_start = 0
        if left_tok is not None:
            t = self.tokens[left_tok]
            char_start = t.idx + len(t)
        char_end = len(self.text)
        if right_tok is not None:
            char_end = self.tokens[right_tok].idx

        raw = self.text[char_start:char_end].lower()
        subtext, orig_pos = _normalize_with_positions(raw)
        concat = " ".join(self.normalize(w["_word"]) for w in gap)
        if not subtext or not concat:
            return
        sub_match, concat_match = _ns_utils.match_list(subtext, concat)

        norm_lens = [len(self.normalize(w["_word"])) for w in gap]
        char_to_word = [
            (wi, ci) for wi, nlen in enumerate(norm_lens) for ci in range(nlen + 1)
        ]
        for ti, ci in zip(sub_match, concat_match):
            wi, charnum = char_to_word[ci]
            # character position in original text
            gap[wi].setdefault("_votes", []).append(char_start + orig_pos[ti] - charnum)

        tok_slice = self.tokens[left_tok:right_tok]
        for w, nlen in zip(gap, norm_lens):
            votes: list[int] | None = w.pop("_votes", None)
            if not votes:
                continue
            best = max(votes, key=votes.count)
            if votes.count(best) / max(nlen, 1) <= 0.5:
                logger.warning(
                    "Ignoring unreliable matching for '%s' in '%s'",
                    w["_word"],
                    subtext,
                )
                continue
            found = self.text[best : best + len(w["_word"])]
            if self.normalize(w["_word"]) != self.normalize(found):
                logger.warning(
                    "Approximately matched annotated %r with %r in text",
                    w["_word"],
                    found,
                )
            if not tok_slice:
                continue
            ind = bisect.bisect_right(tok_slice, best, key=lambda t: t.idx)
            ind = max(ind - 1, 0)
            nearest = tok_slice[ind]
            w["text_char"] = best
            w["sentence"] = nearest.sent.text_with_ws
            w["sentence_char"] = best - nearest.sent[0].idx

    # -- finalization -------------------------------------------------------

    def _finalize(self, info: list[dict[str, tp.Any]]) -> None:
        """Convert internal keys to output keys and fill sentence gaps."""
        for i in info:
            i.pop("_word")
            tok_idx = i.pop("_tok", None)
            if tok_idx is not None:
                tok = self.tokens[tok_idx]
                i["text_char"] = tok.idx
                i["sentence"] = tok.sent.text_with_ws
                i["sentence_char"] = tok.idx - tok.sent[0].idx

        prev_sent: str | None = None
        pending: list[dict[str, tp.Any]] = []
        for i in info:
            sent = i.get("sentence")
            if sent is None:
                pending.append(i)
                continue
            if prev_sent == sent:
                for p in pending:
                    p["sentence"] = sent
            pending = []
            prev_sent = sent


def _merge_sentences(
    sentences: list[ev.Sentence],
    min_duration: float | None = None,
    min_words: int | None = None,
) -> list[list[ev.Sentence]]:
    """Merge consecutive sequences into groups so that there is a span of
    at least min_duration between the start of each group
    """
    out: list[list[ev.Sentence]] = []
    for s in sentences:
        new = True
        if out:
            if min_duration is not None:
                new &= s.start - out[-1][0].start >= min_duration
            if min_words is not None:
                new &= sum(len(s.text.split()) for s in out[-1]) >= min_words
        if not new:
            new |= out[-1][-1].timeline != s.timeline
        if new:
            out.append([s])
        else:
            out[-1].append(s)
    return out


# ---------------------------------------------------------------------------
# chunking helpers
# ---------------------------------------------------------------------------


def chunk_events(
    events: pd.DataFrame,
    event_type_to_chunk: tp.Literal["Audio", "Video"],
    event_type_to_use: str | None = None,
    min_duration: float | None = None,
    max_duration: float = np.inf,
):
    """
    Split events into smaller chunks.
    If event_type_to_use is None, the events are chunked into chunks of max_duration.
    If event_type_to_use is not None, the events are chunked based on the train/val/test splits of the event_type_to_use, ensuring that each chunk has duration between min_duration and max_duration.
    """

    added_events: list[dict] = []
    dropped_rows: list[int] = []
    ns_event_type_to_chunk = ev.Event._CLASSES.get(event_type_to_chunk)
    if ns_event_type_to_chunk is None or not hasattr(ns_event_type_to_chunk, "_split"):
        raise ValueError(f"Event type {event_type_to_chunk} is not splittable")
    if event_type_to_use is not None:
        if "split" not in events.columns:
            raise RuntimeError("Events must have a split column")

    for _, df in events.groupby("timeline"):
        to_chunk = df.type == event_type_to_chunk
        if not any(to_chunk):
            continue
        if event_type_to_use is None:  # chunk based on max_duration
            segments = _segs.list_segments(
                df,
                triggers=to_chunk,
                duration=max_duration,
                stride=max_duration,
                stride_drop_incomplete=False,
            )
            timepoints = [segment.start for segment in segments]
        else:  # chunk based on train/test split of the event_type_to_use
            timepoints = []
            events_to_use = df.loc[events.type == event_type_to_use].copy()
            previous = events_to_use.copy().shift(1)
            split_change = events_to_use.split.astype(str) != previous.split.astype(str)
            events_to_use["section"] = np.cumsum(split_change.values)  # type: ignore
            for _, section in events_to_use.groupby("section"):
                start, end = (
                    section.iloc[0].start,
                    section.iloc[-1].start + section.iloc[-1].duration,
                )
                timepoints.extend(np.arange(start, end, max_duration))

        events_to_chunk = df.loc[to_chunk]
        dropped_rows.extend(events_to_chunk.index)
        for row in events_to_chunk.itertuples(index=False):
            event_to_chunk = ns_event_type_to_chunk.from_dict(row)
            new_events = event_to_chunk._split(
                [t - event_to_chunk.start for t in timepoints], min_duration
            )  # type: ignore
            for new_event in new_events:
                new_event_dict = new_event.to_dict()
                # add the columns which were removed by event.from_dict() except index
                for k, v in row._asdict().items():  # type: ignore
                    if k not in new_event_dict:
                        new_event_dict[k] = v
                added_events.append(new_event_dict)

    out_events = events.copy()
    out_events.drop(dropped_rows, inplace=True)
    out_events = pd.concat([out_events, pd.DataFrame(added_events)])
    out_events.reset_index(drop=True, inplace=True)
    return out_events


# ---------------------------------------------------------------------------
# splitting helpers
# ---------------------------------------------------------------------------


@dataclass
class DeterministicSplitter:
    """Hash-based splitter that assigns a deterministic train/val/test split.

    Hashes each sample's unique ID to a float in [0, 1) and maps it to
    a split name according to cumulative ``ratios``.  The assignment is
    stable across runs and independent of dataset order.

    Parameters
    ----------
    ratios :
        Mapping from split name to proportion (must sum to 1).
    seed :
        Added to the hash to produce different splits.
    """

    ratios: dict[str, float]
    seed: float = 0.0

    def __post_init__(self) -> None:
        if not all(ratio > 0 for ratio in self.ratios.values()):
            raise ValueError(f"All ratios must be positive, got {self.ratios}")
        if not np.allclose(sum(self.ratios.values()), 1.0):
            raise ValueError(f"The sum of ratios must be equal to 1. Got {self.ratios}")

    def __call__(self, uid: str) -> str:
        hashed = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
        rng = random.Random(hashed + self.seed)
        score = rng.random()
        cdf = np.cumsum(list(self.ratios.values()))
        names = list(self.ratios.keys())
        for idx, cdf_val in enumerate(cdf):
            if score < cdf_val:
                return names[idx]
        raise ValueError
