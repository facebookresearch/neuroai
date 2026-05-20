# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp

import numpy as np
import pandas as pd
import pytest
import torch

from neuralset.base import TimedArray
from neuralset.events import etypes
from neuralset.events.transforms import AddConcatenationContext
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


@pytest.mark.parametrize("contextualized", [True, False])
@pytest.mark.parametrize("layer", [0, 0.5, 1])
@pytest.mark.parametrize("cache_n_layers", [None, 10])
@pytest.mark.parametrize("model_name", ["openai-community/gpt2", "google-t5/t5-small"])
def test_llm(
    contextualized: bool, layer: int, cache_n_layers: int | None, model_name: str
) -> None:
    events = _make_test_events()
    extractor = text.HuggingFaceText(
        aggregation="sum",
        layers=layer,
        contextualized=contextualized,
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
    extractor = text.HuggingFaceText(
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


def test_huggingface_text_with_sentence_events():
    events = _make_test_events()
    events["type"] = "Sentence"
    events = standardize_events(events)
    extractor = text.HuggingFaceText(event_types="Sentence", contextualized=False)
    extractor.prepare(events)
    events_list = extract_events(events, types=extractor.event_types)
    out = extractor(events_list[0], 0, 1)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (768,)


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


def test_llm_explicit_error() -> None:
    extractor = text.HuggingFaceText(
        aggregation="sum",
        layers=1,
        contextualized=True,
        model_name="openai-community/gpt2",
    )
    word = etypes.Word(text="word", start=0, duration=1, timeline="x")
    with pytest.raises(ValueError):
        _ = list(extractor._get_timed_arrays([word], 0, 1))


def test_dynamic_text() -> None:
    extractor = text.HuggingFaceText(
        frequency=2,
        aggregation="sum",
        layers=1,
        contextualized=False,
        model_name="openai-community/gpt2",
    )
    word = etypes.Word(text="word", start=1, duration=1, timeline="x")
    out = extractor(word, start=0, duration=3).cpu().numpy()
    np.testing.assert_array_equal(out[0] != 0, [0, 0, 1, 1, 0, 0])


def _make_word() -> etypes.Word:
    return etypes.Word(
        text="Hello",
        start=0,
        duration=1,
        language="english",
        timeline="foo",
        context="Hello from Paris!",
    )


def test_llm_long_context() -> None:
    extractor = text.HuggingFaceText(
        aggregation="sum",
        contextualized=True,
        model_name="openai-community/gpt2",
        device="cpu",
    )
    word = _make_word()
    word.context = " ".join([str(k) for k in range(1024)])
    # should work even though context is larger that 1024=maximum (~1500)
    _ = extractor(word, 0, 1)


@pytest.mark.skipif(
    "IN_GITHUB_ACTION" in os.environ, reason="Models are too big for CI cache"
)
def test_bart() -> None:
    extractor = text.HuggingFaceText(
        aggregation="sum",
        contextualized=True,
        model_name="facebook/bart-base",
        device="cpu",
    )
    word = _make_word()
    _ = extractor(word, 0, 1)


def test_llm_pretrained() -> None:
    word = _make_word()
    device: tp.Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
    outputs = [
        text.HuggingFaceText(
            aggregation="sum",
            contextualized=True,
            device=device,
            pretrained=pretrained,  # type: ignore
        )(word, 0, 1)
        for pretrained in [True, False, "part-reversal"]
    ]
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(outputs[0], outputs[1])
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(outputs[0], outputs[2])


@pytest.mark.parametrize(
    "word_text, n_target_tokens",
    [
        ("internationalization", 2),
        ("need internationalization", 3),
    ],
)
def test_contextualized_token_slicing(word_text: str, n_target_tokens: int) -> None:
    """With token_aggregation=None the token dimension is preserved,
    so we can verify that the slice length equals the tokenizer's
    token count for the target word, not its character count."""
    context = "we urgently need internationalization"
    extractor = text.HuggingFaceText(
        aggregation="sum",
        contextualized=True,
        token_aggregation=None,
        model_name="openai-community/gpt2",
        device="cpu",
    )
    encode: tp.Any = extractor.tokenizer.encode
    # GPT-2 tokenizes " internationalization" as 2 sub-tokens;
    # the old bug used len(word_text) (character count) instead.
    assert len(encode(word_text, add_special_tokens=False)) == n_target_tokens
    word = etypes.Word(text=word_text, start=0, duration=1, timeline="x", context=context)
    out = extractor(word, 0, 1)
    assert out.shape == (n_target_tokens, 768)


def test_batched_target_slice_excludes_pads() -> None:
    """A word's contextualized output must not depend on what it's batched with."""
    extractor = text.HuggingFaceText(
        aggregation="sum",
        contextualized=True,
        token_aggregation="sum",
        model_name="openai-community/gpt2",
        device="cpu",
        batch_size=2,
    )
    # alone vs batched would otherwise hit the same MapInfra cache entry
    extractor.infra.keep_in_ram = False

    short = etypes.Word(text="ce", start=0, duration=1, timeline="x", context="ce")
    long_ctx = (
        "Once upon a time in a faraway kingdom there lived a small fox who "
        "loved to wander through the dense forest looking for berries "
    ) * 3
    long = etypes.Word(
        text="afternoon",
        start=1,
        duration=1,
        timeline="x",
        context=long_ctx + " afternoon",
    )

    short_alone = np.asarray(list(extractor._get_data([short]))[0])
    long_alone = np.asarray(list(extractor._get_data([long]))[0])
    short_batched, long_batched = (
        np.asarray(x) for x in extractor._get_data([short, long])
    )
    np.testing.assert_allclose(short_batched, short_alone, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(long_batched, long_alone, rtol=1e-3, atol=1e-3)


def test_part_reversal() -> None:
    ref = np.random.rand(2, 3, 4)
    x = torch.from_numpy(np.array(ref, copy=True))
    np.testing.assert_almost_equal(x.numpy(), ref)
    text.part_reversal(x)
    assert x.shape == ref.shape
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(x.numpy(), ref)
