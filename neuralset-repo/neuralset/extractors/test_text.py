# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp

import numpy as np
import pandas as pd
import pydantic
import pytest
import torch

from neuralset.base import TimedArray
from neuralset.events import etypes
from neuralset.events.transforms import (
    AddAlignedTexts,
    AddConcatenationContext,
    AddContextToWords,
    AddSentenceToWords,
)
from neuralset.events.utils import extract_events, standardize_events

from . import text


def _make_test_events() -> pd.DataFrame:
    sentence = "This is a sentence for the unit tests"
    sentence_words = sentence.split(" ")
    events_list = [
        dict(
            type="Sentence",
            text=sentence,
            start=i * len(sentence_words),
            duration=len(sentence_words),
            language="english",
            timeline="foo",
            split="train",
        )
        for i in range(3)
    ]
    events_list += [
        dict(
            type="Word",
            text=word,
            start=i,
            duration=i + 1,
            language="english",
            timeline="foo",
            split="train",
            sequence_id=i // len(sentence_words),
        )
        for i, word in enumerate(3 * sentence_words)
    ]
    events = pd.DataFrame(events_list)
    events = standardize_events(events)
    add_context = AddConcatenationContext()
    add_context(events)
    return events


@pytest.mark.parametrize(
    "extractor_cls",
    [
        text.WordLength,
        text.WordFrequency,
        text.SpacyEmbedding,
    ],
)
def test_word_embedding(extractor_cls: tp.Type[text.BaseText]) -> None:
    events = _make_test_events()
    extractor = extractor_cls(aggregation="sum")
    extractor.prepare(events)
    events_list = extract_events(events, types=extractor.event_types)
    out = extractor.get_static(events_list[0])  # type: ignore
    assert isinstance(out, torch.Tensor)


def test_spacy_empty() -> None:
    events = _make_test_events()
    extractor = text.SpacyEmbedding(aggregation="sum", allow_missing=True)
    extractor.prepare(events)
    out = extractor([], 0, 1)
    assert out.shape == (300,)


def test_spacy_embedding_api() -> None:
    """model_name and language are mutually exclusive; language resolves from events."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        text.SpacyEmbedding(model_name="en_core_web_lg", language="english")
    # model_name bypasses language resolution
    ext = text.SpacyEmbedding(model_name="en_core_web_lg", aggregation="sum")
    events = _make_test_events()
    ext.prepare(events)
    events_list = extract_events(events, types=ext.event_types)
    out = ext.get_static(events_list[0])  # type: ignore[arg-type]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (300,)
    # no language anywhere → clear error
    ext_bare = text.SpacyEmbedding(aggregation="sum")
    with pytest.raises(ValueError, match="No language"):
        ext_bare.get_embedding("hello")


def test_spacy_prepare_installs_each_model_once_before_dispatch(
    mocker: tp.Any,
) -> None:
    """prepare() installs every needed model up front, in the submitting process.

    Cluster workers sharing the environment then find the models instead of each
    pip-installing the same one concurrently.
    """
    calls: list[str] = []
    mocker.patch.object(
        text.utils, "install_spacy_model", side_effect=lambda m: calls.append(m)
    )
    dispatch = mocker.patch.object(
        text.BaseText,
        "prepare",
        autospec=True,
        side_effect=lambda self, events: calls.append("dispatch"),
    )
    words = [
        dict(type="Word", text=t, start=i, duration=1, language=lang, timeline="t")
        for i, (t, lang) in enumerate(
            [("hello", "en"), ("bonjour", "fr"), ("hi", "en"), ("γεια", "el")]
        )
    ]
    events = standardize_events(pd.DataFrame(words))
    text.SpacyEmbedding(aggregation="sum").prepare(events)
    assert calls == ["el_core_news_lg", "en_core_web_lg", "fr_core_news_lg", "dispatch"]
    dispatch.assert_called_once()
    # An unsupported language fails here, before anything is dispatched.
    calls.clear()
    bad = standardize_events(pd.DataFrame([{**words[0], "language": "klingon"}]))
    with pytest.raises(ValueError, match="not available"):
        text.SpacyEmbedding(aggregation="sum").prepare(bad)
    assert calls == []


def test_spacy_prepare_skips_install_when_cached(
    mocker: tp.Any, tmp_path: tp.Any
) -> None:
    """A fully cached run (or no events) requests no spaCy model install.

    Nothing is downloaded or loaded: ``install_spacy_model`` is mocked to record
    the requested model names and ``get_embedding`` returns a stub vector.
    """
    installs: list[str] = []
    mocker.patch.object(
        text.utils, "install_spacy_model", side_effect=lambda m: installs.append(m)
    )
    mocker.patch.object(
        text.SpacyEmbedding, "get_embedding", return_value=np.zeros(300, dtype=np.float32)
    )
    words = [
        dict(type="Word", text=t, start=i, duration=1, language="el", timeline="t")
        for i, t in enumerate(["γεια", "σου"])
    ]
    events = standardize_events(pd.DataFrame(words))
    ext = text.SpacyEmbedding(aggregation="sum", infra={"folder": tmp_path})  # type: ignore
    ext.prepare(events)  # first run computes, so it requests the model (mocked)
    assert installs == ["el_core_news_lg"]
    installs.clear()
    text.SpacyEmbedding(aggregation="sum", infra={"folder": tmp_path}).prepare(  # type: ignore
        events
    )
    assert installs == []  # every embedding is cached now
    named = text.SpacyEmbedding(
        model_name="xx_custom",
        aggregation="sum",
        infra={"folder": tmp_path},  # type: ignore
    )
    named.prepare([])
    assert installs == []  # no events, nothing to compute


def test_tfidf() -> None:
    mock_events = []
    sentences = [
        "I want to play with the cat",
        "I have a dog",
        "I want to play with the beautiful cat",
        "I am living in Brooklyn",
    ]

    for sentence in sentences:
        mock_events.append(
            {
                "type": "Sentence",
                "start": 0,
                "timeline": "",
                "duration": 0.1,
                "text": sentence,
            }
        )

    # Adding a non-Sentence event
    mock_events.append(
        {
            "type": "Image",
            "start": 0,
            "duration": None,
            "text": "",
        }
    )

    mock_events_df = pd.DataFrame(mock_events)

    extractor = text.TfidfEmbedding()
    extractor.prepare(mock_events_df)
    events_list = extract_events(mock_events_df, types=extractor.event_types)
    out = extractor.get_static(events_list[0])  # type: ignore

    assert isinstance(out, torch.Tensor)


GPT2: dict[str, tp.Any] = dict(model_name="openai-community/gpt2", device="cpu")


@pytest.mark.parametrize("layer", [0, 0.5, 1])
@pytest.mark.parametrize("cache_n_layers", [None, 10])
@pytest.mark.parametrize("model_name", ["openai-community/gpt2", "google-t5/t5-small"])
def test_llm(layer: int, cache_n_layers: int | None, model_name: str) -> None:
    events = _make_test_events()
    extractor = text.HuggingFaceWord(
        aggregation="sum",
        layers=layer,
        model_name=model_name,
        cache_n_layers=cache_n_layers,
    )
    if hasattr(extractor, "model_name"):
        assert "xl" not in extractor.model_name, "Avoid large models as default"
    extractor.prepare(events)
    events_list = extract_events(events, types=extractor.event_types)
    out = list(extractor._get_timed_arrays(events=events_list[:1], start=0, duration=1))  # type: ignore
    assert len(out) == 1
    assert isinstance(out[0], TimedArray)


@pytest.mark.parametrize("layer_aggregation", ["mean", "sum", "group_mean", None])
def test_layer_aggregation(
    layer_aggregation: tp.Literal["mean", "sum", "group_mean", None],
):
    events = _make_test_events()
    extractor = text.HuggingFaceWord(
        model_name="gpt2",
        layer_aggregation=layer_aggregation,
        layers=[0, 0.5, 1],
        aggregation="sum",
        token_aggregation="mean",
    )
    extractor.prepare(events)
    events_list = extract_events(events, types=extractor.event_types)
    out = extractor(events_list[0], 0, 1)
    assert isinstance(out, torch.Tensor)
    if layer_aggregation in ["group_mean", None]:
        assert out.ndim == 2
        if layer_aggregation == "group_mean":
            assert out.shape[0] == 2  # 2 groups of layers
        else:
            assert out.shape[0] == 3  # 3 layers
    else:
        assert out.ndim == 1


def test_aggregate_layers_depth_grid() -> None:
    latents = np.arange(13)[:, None]
    for k, depth in [*enumerate(np.linspace(0, 1, 13)), (7, 0.58)]:
        extractor = text.HuggingFaceText(layers=float(depth))
        assert extractor._aggregate_layers(latents).item() == k, depth


def test_huggingface_sentence() -> None:
    events = _make_test_events()
    extractor = text.HuggingFaceSentence(**GPT2)
    extractor.prepare(events)
    sentence = extract_events(events, types=extractor.event_types)[0]
    assert isinstance(sentence, etypes.Sentence)
    out = extractor(sentence, 0, 1)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (768,)
    # the static vector of the event, as returned to e.g. splitters
    torch.testing.assert_close(extractor.get_static(sentence), out)
    # each event is embedded alone: no context windows
    with pytest.raises(pydantic.ValidationError, match="context"):
        text.HuggingFaceSentence.model_validate({"context": {"duration": 10.0}, **GPT2})


def test_dynamic_text() -> None:
    extractor = text.HuggingFaceWord(frequency=2, aggregation="sum", layers=1, **GPT2)
    word = etypes.Word(text="word", start=1, duration=1, timeline="x")
    out = extractor(word, start=0, duration=3).cpu().numpy()
    np.testing.assert_array_equal(out[0] != 0, [0, 0, 1, 1, 0, 0])


def test_word_tokens() -> None:
    """With token_aggregation=None, the token dimension of the word is kept."""
    extractor = text.HuggingFaceWord(aggregation="sum", token_aggregation=None, **GPT2)
    # GPT-2 tokenizes "internationalization" as 2 sub-tokens
    word = etypes.Word(text="internationalization", start=0, duration=1, timeline="x")
    assert extractor(word, 0, 1).shape == (2, 768)


def _make_sentence(n_words: int) -> etypes.Sentence:
    words = " ".join(str(k) for k in range(n_words))
    return etypes.Sentence(text=words, start=0, duration=1, timeline="foo")


def test_llm_long_sentence() -> None:
    extractor = text.HuggingFaceSentence(aggregation="sum", **GPT2)
    # truncated to GPT-2's 1024 tokens (~1500 here)
    _ = extractor(_make_sentence(1024), 0, 1)


def test_max_length_real_limit() -> None:
    extractor = text.HuggingFaceWord(**GPT2)
    assert extractor.tokenizer.model_max_length == 1024
    assert extractor._max_length() == 1024


@pytest.mark.skipif("IN_GITHUB_ACTION" in os.environ, reason="OPT not in CI cache")
def test_max_length_sentinel_fallback() -> None:
    """OPT's tokenizer reports the HF 'infinite' sentinel, so _max_length must
    fall back to the model's position table and an over-length text must
    truncate instead of overflowing OPT's learned positions."""
    sentence = _make_sentence(4000)  # >2050 tokens
    try:
        extractor = text.HuggingFaceSentence(model_name="facebook/opt-125m", device="cpu")
        tokenizer = extractor.tokenizer
        assert tokenizer.model_max_length >= int(1e29)  # sentinel, not a real limit
        config: tp.Any = extractor.model.config  # type: ignore[union-attr]
        assert extractor._max_length() == config.max_position_embeddings
        _ = extractor(sentence, 0, 1)
    except (OSError, RuntimeError, pydantic.ValidationError):
        pytest.skip("opt-125m unreachable")


@pytest.mark.skipif(
    "IN_GITHUB_ACTION" in os.environ, reason="Models are too big for CI cache"
)
def test_bart() -> None:
    extractor = text.HuggingFaceWord(
        aggregation="sum", model_name="facebook/bart-base", device="cpu"
    )
    word = etypes.Word(text="Hello", start=0, duration=1, timeline="foo")
    _ = extractor(word, 0, 1)


def test_llm_pretrained() -> None:
    word = etypes.Word(text="Hello", start=0, duration=1, timeline="foo")
    device: tp.Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    outputs = [
        text.HuggingFaceWord(aggregation="sum", device=device, pretrained=pretrained)(
            word, 0, 1
        )
        for pretrained in [True, False]
    ]
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(outputs[0], outputs[1])


def test_batched_output_excludes_pads() -> None:
    """A text's output must not depend on what it's batched with."""
    extractor = text.HuggingFaceSentence(token_aggregation="sum", batch_size=2, **GPT2)
    # alone vs batched would otherwise hit the same MapInfra cache entry
    extractor.infra.keep_in_ram = False
    short = etypes.Sentence(text="ce", start=0, duration=1, timeline="x")
    long_text = (
        "Once upon a time in a faraway kingdom there lived a small fox who "
        "loved to wander through the dense forest looking for berries "
    ) * 3
    long = etypes.Sentence(text=long_text, start=1, duration=1, timeline="x")
    short_alone = np.asarray(list(extractor._get_data([short]))[0])
    long_alone = np.asarray(list(extractor._get_data([long]))[0])
    short_batched, long_batched = (
        np.asarray(x) for x in extractor._get_data([short, long])
    )
    np.testing.assert_allclose(short_batched, short_alone, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(long_batched, long_alone, rtol=1e-3, atol=1e-3)


# HuggingFaceText: contextual word embeddings of AlignedText events

TEXT = "The cat sat on the mat. It was very happy, and then it slept. Nobody woke it."


def _text_and_words(long_word: str = "") -> pd.DataFrame:
    """Text + Words, with sentence annotations on Words (``long_word`` lasts 3 s)."""
    words = [w.strip(".,") for w in TEXT.split()]
    rows = [dict(type="Text", text=TEXT, start=0.0, duration=len(words), timeline="tl")]
    rows += [
        dict(
            type="Word",
            text=w,
            start=i + 0.1,
            duration=3.0 if w == long_word else 0.5,
            timeline="tl",
        )
        for i, w in enumerate(words)
    ]
    events = pd.DataFrame(rows).assign(language="english")
    events = standardize_events(events)
    return AddSentenceToWords()(events)


def _aligned_text(word_starts: np.ndarray, start: float = 0.0) -> etypes.AlignedText:
    n_words = len(word_starts)
    return etypes.AlignedText(
        text="x" * n_words,
        start=start,
        duration=float(word_starts.max()) + 1,
        timeline="tl",
        word_starts=word_starts,
        word_durations=np.full(n_words, 0.2),
        word_char_starts=np.arange(n_words),
        word_char_stops=np.arange(1, n_words + 1),
    )


def _contextual_vectors(contexts: list[str], words: list[str]) -> np.ndarray:
    """Reference: the last tokens of each context, which spell its word (the
    word-by-word pipeline, with a context string per word)."""
    ref = text.HuggingFaceWord(**GPT2)
    tokenizer = ref.tokenizer
    out = []
    with torch.no_grad():
        for context, word in zip(contexts, words):
            ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")
            states = torch.stack(
                ref.model(**ids, output_hidden_states=True).hidden_states
            )
            prefix = context[: -len(word)].rstrip()
            n_prefix = len(tokenizer.encode(prefix, add_special_tokens=False))
            latent = ref._aggregate_tokens(states[:, 0, n_prefix:]).numpy()
            out.append(ref._aggregate_layers(latent))
    return np.stack(out)


def test_overlapping_words() -> None:
    rng = np.random.default_rng(0)
    n_words = 300
    word_starts = np.sort(rng.uniform(0, 100, n_words))
    word_durations = rng.uniform(0, 2, n_words)
    event = etypes.AlignedText(
        text="x" * n_words,
        start=0.1,
        duration=102,
        timeline="tl",
        word_starts=word_starts,
        word_durations=word_durations,
        word_char_starts=np.arange(n_words),
        word_char_stops=np.arange(1, n_words + 1),
    )
    starts = 0.1 + word_starts
    # random windows, and windows whose edges fall on word edges
    bounds = np.concatenate([starts, starts + word_durations, rng.uniform(-5, 105, 100)])
    for start in bounds:
        for duration in (0.0, 0.5, 3.0):
            stop = start + duration
            expected = (starts < stop) & (starts + word_durations > start)
            actual = event.overlapping_words(start, stop)
            np.testing.assert_array_equal(actual, np.flatnonzero(expected))


@pytest.mark.parametrize("causal", [True, False])
def test_timed_windows(causal: bool) -> None:
    rng = np.random.default_rng(0)
    gaps = rng.exponential(0.3, 300)
    gaps[::37] += 20  # pauses longer than the windows
    word_starts = np.cumsum(gaps)
    word_starts[50] -= 0.5  # alignment overlap
    event = _aligned_text(word_starts)
    duration, overlap = 10.0, 0.5
    windowing = text.WordContextWindow(duration=duration, overlap=overlap, causal=causal)
    windows = text._TextWindow.timed(event, windowing)
    embedded = [np.arange(w.embed_from, w.embed_stop) for w in windows]
    np.testing.assert_array_equal(np.concatenate(embedded), np.arange(len(word_starts)))
    assert [w.is_last for w in windows] == [False] * (len(windows) - 1) + [True]
    # each word sees the words starting within its guaranteed context (up to ticks)
    onsets = np.maximum.accumulate(word_starts)
    past = overlap * duration if causal else overlap * duration / 2
    future = 0.0 if causal else past - 2e-3
    for w in windows:
        assert onsets[w.stop - 1] - onsets[w.first] < duration + 1e-3
        for i in range(w.embed_from, w.embed_stop):
            lower, upper = onsets[i] - past + 2e-3, onsets[i] + future
            context = np.flatnonzero((onsets >= lower) & (onsets <= upper))
            assert w.first <= context[0] <= i <= context[-1] < w.stop
    # word onsets set the cache uid of timed texts, wherever the text starts
    shifted = _aligned_text(word_starts + 2.0, start=5.0)
    retimed = _aligned_text(word_starts * 1.1)
    assert text._AlignedText(shifted, True).uid() == text._AlignedText(event, True).uid()
    assert text._AlignedText(retimed, True).uid() != text._AlignedText(event, True).uid()
    assert (
        text._AlignedText(retimed, False).uid() == text._AlignedText(event, False).uid()
    )
    assert text._AlignedText(event, False).uid() == event.content_uid()
    with pytest.raises(ValueError, match="exactly one of words and duration"):
        text.WordContextWindow(words=5, duration=1)
    # missing context fields keep the extractor defaults
    config = dict(frequency=1, context={"duration": 30}, **GPT2)
    extractor = text.HuggingFaceText.model_validate(config)
    assert extractor.context == text.WordContextWindow(
        duration=30, overlap=0.5, causal=True
    )
    config = dict(frequency=1, context={"min_context_words": 128}, **GPT2)
    extractor = text.HuggingFaceText.model_validate(config)
    assert extractor.context == text.WordContextWindow(
        words=1024, overlap=0.125, causal=True
    )


def test_single_window_matches_word_pipeline() -> None:
    events = _text_and_words()
    contexts = AddContextToWords(sentence_only=False, split_field="")(events)
    words = contexts.loc[contexts.type == "Word"]
    expected = _contextual_vectors(list(words.context), list(words.text))
    new = text.HuggingFaceText(aggregation="sum", frequency=10, **GPT2)
    (event,) = extract_events(AddAlignedTexts()(events), types="AlignedText")
    assert isinstance(event, etypes.AlignedText)
    actual = next(iter(new._get_data(new._texts([event]))))
    assert actual.shape == expected.shape == (len(words), 768)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    # each word fills its own time span, and frequency=0 sums the words of the segment
    out = new(event, start=0, duration=float(events.stop.max()))
    assert out.shape == (768, int(10 * events.stop.max()))
    static = text.HuggingFaceText(aggregation="sum", frequency=0, **GPT2)
    out = static(event, start=0.0, duration=2.0)  # first 2 words
    np.testing.assert_allclose(out, expected[:2].sum(axis=0), rtol=1e-4, atol=1e-4)


def test_sentence_only_context_matches_word_pipeline() -> None:
    # "the" ends after "mat", the last word of its sentence
    events = _text_and_words(long_word="the")
    contexts = AddContextToWords(sentence_only=True, split_field="")(events.copy())
    words = contexts.loc[contexts.type == "Word"]
    expected = _contextual_vectors(list(words.context), list(words.text))
    # sentences are split from the Text: Sentence events are not needed
    no_sentences = events.loc[events.type != "Sentence"]
    aligned = AddAlignedTexts(text_type="Sentence")(no_sentences)
    sentences = tp.cast(
        list[etypes.AlignedText], extract_events(aligned, types="AlignedText")
    )
    assert [event.text for event in sentences] == [
        "The cat sat on the mat.",
        "It was very happy, and then it slept.",
        "Nobody woke it.",
    ]
    new = text.HuggingFaceText(frequency=10, **GPT2)
    actual = np.concatenate(list(new._get_data(new._texts(sentences))))
    assert actual.shape == expected.shape == (len(words), 768)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("context", [dict(words=5), dict(duration=4.5)])
def test_strided_windows_match_explicit_context(
    causal: bool, context: dict[str, tp.Any]
) -> None:
    events = AddAlignedTexts()(_text_and_words())
    (event,) = extract_events(events, types="AlignedText")
    assert isinstance(event, etypes.AlignedText)
    windowing = text.WordContextWindow(overlap=0.6, causal=causal, **context)
    new = text.HuggingFaceText(frequency=10, batch_size=3, context=windowing, **GPT2)
    actual = next(iter(new._get_data(new._texts([event]))))
    # each word, read with the prefix of the window that embeds it (GPT-2 is causal)
    char_starts, char_stops = event.word_char_starts, event.word_char_stops
    contexts, words = [], []
    for window in new._windows(event):
        for i in range(window.embed_from, window.embed_stop):
            words.append(event.text[char_starts[i] : char_stops[i]])
            contexts.append(event.text[char_starts[window.first] : char_stops[i]])
    expected = _contextual_vectors(contexts, words)
    assert actual.shape == expected.shape == (len(char_starts), 768)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_huggingface_text_previous_api() -> None:
    for kwargs in (dict(contextualized=True), dict(event_types="Word")):
        with pytest.raises(ValueError, match="HuggingFaceWord or HuggingFaceSentence"):
            text.HuggingFaceText(frequency=1, **kwargs, **GPT2)  # type: ignore[arg-type]
    extractor = text.HuggingFaceText(frequency=1, **GPT2)
    with pytest.raises(ValueError, match="add AddAlignedTexts"):
        extractor.prepare(_text_and_words())
