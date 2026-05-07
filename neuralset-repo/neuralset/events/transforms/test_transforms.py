# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import logging
import os
import subprocess
import sys
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic
import pytest
import scipy
from exca.cachedict import CacheDict
from PIL import Image

import neuralfetch.utils
import neuralset as ns
from neuralset.events import etypes
from neuralset.events import transforms as _transf
from neuralset.events.transforms import utils as _tutils

logging.getLogger("exca").setLevel(logging.DEBUG)


def _make_test_events() -> tuple[list[str], pd.DataFrame]:
    sentence = 3 * ("This is a sentence for the unit tests.").split(" ")
    events_list = [
        dict(
            type="Word",
            text=sentence[i],
            start=i,
            duration=0.5,
            language="english",
            timeline="foo",
            split="train",
            sequence_id=0,
        )
        for i in range(len(sentence))
    ]
    events = pd.DataFrame(events_list)
    return sentence, events


def test_context() -> None:
    sentence, events = _make_test_events()
    transform = _transf.AddConcatenationContext()
    events = transform(events)
    for i, row in events.iterrows():
        assert isinstance(i, int)
        assert row["context"] == " ".join(sentence[: i + 1])


class FeatEnhancer(_transf.EventsTransform):
    feat: ns.extractors.BaseExtractor


def test_sub_discriminator(test_data_path: Path) -> None:
    steps: tp.Any = [
        {"name": "Test2023Fmri", "path": test_data_path, "query": "timeline_index<1"},
        {"name": "FeatEnhancer", "feat": {"name": "Pulse"}},
    ]
    chain = ns.Chain(steps=steps)
    assert len(chain.steps) == 2


def test_transforms_on_study() -> None:
    try:
        STUDY_FOLDER = neuralfetch.utils.root_study_folder()
    except RuntimeError as e:
        pytest.skip(str(e))
    name = "PallierListen2023"
    name = "PallierRead2023"
    name = "Gwilliams2022Neural"
    name = "Armeni2022Sherlock"
    steps: list[tp.Any] = [
        {"name": name, "path": STUDY_FOLDER, "query": "timeline_index < 1"},
        # {"name": "AddText"},
        # {"name": "AddSentenceToWords", "override_sentences": True},
        # {"name": "AssignSentenceSplit", "min_duration": 3},
        # {"name": "AddContextToWords"},  # , "sentence_only": True},
        {"name": "AddSentenceToWords", "max_unmatched_ratio": 0.05},
        {
            "name": "AddContextToWords",
            "sentence_only": False,
            "max_context_len": 1024,
            "split_field": "",
        },
    ]
    events = ns.Chain(steps=steps).run()
    assert "context" in events.columns
    # assert "split" in events.columns
    events.loc[
        # events.type == "Word", ["text", "sentence_char", "sentence", "split", "context"]
        events.type == "Word",
        ["text", "sentence_char", "sentence", "context"],
    ].to_csv(f"{name}.csv", sep=";")


def _make_test_dataframe(duplicate: int = -1) -> pd.DataFrame:
    text = (
        "Peut-être bien que cet homme est absurde. Cependant il est moins absurde que le roi, que le vaniteux, "
        "que le businessman et que le buveur. Au moins son travail a-t-il un sens. Quand il allume son réverbère, "
        "c'est comme s'il faisait naître une étoile de plus, ou une fleur. Quand il éteint son réverbère ça endort "
        "la fleur ou l'étoile. C'est une occupation très jolie. C'est véritablement utile puisque c'est joli."
    )
    events = []
    for k, w in enumerate(text.split()):
        events.append(
            {
                "text": w.strip(".,"),
                "type": "Word",
                "start": k,
                "duration": 0.2,
                "timeline": "lpp",
            }
        )
        if k == duplicate:
            events.append(events[-1])
    events.append(
        {
            "type": "Text",
            "text": text,
            "start": -0.1,
            "duration": len(events),
            "timeline": "lpp",
            "language": "french",
            "subject": "test-subject",
        }
    )
    return ns.events.standardize_events(pd.DataFrame(events))


def test_standard_text_enhancement(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    df = _make_test_dataframe()
    df = _transf.AddSentenceToWords()(df)
    # in a standard pipeline, this could be cached at some point, which may change some values
    cd: tp.Any = CacheDict(folder=tmp_path)
    with cd.writer() as writer:
        writer["test"] = df
    df = cd["test"]
    df = cd["test"]
    assert set(df.loc[[37, 38]].type.unique()) == {
        "Word"
    }  # just to be sure we set sentence to Words
    df.loc[[37, 38], "sentence"] = [np.nan, ""]
    df = _transf.AssignSentenceSplit(max_unmatched_ratio=0.05)(df)
    df = _transf.AddContextToWords(sentence_only=False)(df)
    assert df.type[2] == "Word"
    assert df.split[2] == "test"  # deterministic split -> should always be test
    assert df.type[70] == "Word"
    assert df.split[70] == "train"  # deterministic split -> should always be train
    assert np.isnan(df.split[0])  # full text -> no split
    last = len(df) - 1
    assert df.context[last].startswith("Quand")  # wshould be a long context
    df = _transf.AddContextToWords(sentence_only=True)(df)
    assert df.context[last] == "C'est véritablement utile puisque c'est joli"
    df = _transf.AddContextToWords(max_context_len=3)(df)
    assert df.context[last] == "utile puisque c'est joli"
    # try full pipeline as one transform:
    df = _make_test_dataframe()
    df.loc[:, "context"] = np.nan
    df = _transf.AssignWordSplitAndContext(max_context_len=2)(df)
    # if recwarn:
    #     # setting an item of incompatible dtype (for context column)
    #     raise AssertionError(f"Got warnings: {[w.message for w in recwarn.list]}")
    #     # new unknown issue:
    #     # AssertionError: Got warnings: [ResourceWarning('unclosed file <_io.BufferedReader name=33>'), ResourceWarning('unclosed file <_io.BufferedReader name=35>')]

    assert df.context[last] == "puisque c'est joli"
    # test sentence
    sentence_row = df.loc[df.type == "Sentence"].iloc[0]
    sentence = ns.events.Event.from_dict(sentence_row)
    assert sentence.type == "Sentence"
    assert sentence.start == pytest.approx(0, abs=1e-4)
    assert sentence.duration == pytest.approx(6.2, abs=1e-4)
    assert sentence_row.subject == "test-subject", "Subject field must be added"


def test_standard_multi_timeline() -> None:
    # try full pipeline as one transform:
    df = _make_test_dataframe()
    df2 = df.copy()
    df2["timeline"] = "tl2"
    df = pd.concat([df, df2], ignore_index=True)
    assert df.type[70] == "Word"
    df.loc[70, "type"] = "Phoneme"
    df = _transf.AssignWordSplitAndContext(max_context_len=2)(df)
    # override + all the same split (fast track)
    df = _transf.AssignWordSplitAndContext(
        max_context_len=2, override_sentences=True, ratios=(1, 0, 0)
    )(df)
    assert all(x == "train" for x in df.loc[df.type == "Word"].split)


def test_assign_k_splits() -> None:
    # try full pipeline as one transform:
    df = _make_test_dataframe()
    df2 = df.copy()
    df2["timeline"] = "tl2"
    df = pd.concat([df, df2], ignore_index=True)
    df = _transf.AssignWordSplitAndContext(max_context_len=2)(df)
    df = _transf.AssignKSplits(k=3)(df)
    expected = {"4ee9472d-ksplit1/2", "4ee9472d-ksplit2/2", "4ee9472d-ksplit1/1"}
    assert set(df.loc[df.type == "Word"].split) == expected
    df = _transf.AssignKSplits(k=2, groupby="timeline")(df)
    # expected = {"lpp_split_1", "lpp_split_2", "tl2_split_1", "tl2_split_2"}
    expected = {"4ee9472d-ksplit1/2", "4ee9472d-ksplit2/2"}  # finds repetitions
    assert set(df.loc[df.type == "Word"].split) == expected


def test_merge_sentences() -> None:
    data = [(1, 4, "x"), (5, 3, "x"), (9, 3, "x"), (10, 3, "y"), (20, 3, "y")]
    seqs = [
        etypes.Sentence(start=s, duration=d, timeline=tl, text="z.") for s, d, tl in data
    ]
    merged = _tutils._merge_sentences(seqs, min_duration=6)
    assert tuple(len(ss) for ss in merged) == (2, 1, 1, 1)
    merged = _tutils._merge_sentences(seqs, min_words=3)
    assert tuple(len(ss) for ss in merged) == (3, 2)


@pytest.mark.parametrize(
    "change_timeline,split_field,expected",
    [
        (False, "split", "b b b. c c c. d d d"),
        (True, "split", "d d d"),
        (False, "", "a a a. b b b. c c c. d d d"),
    ],
)
def test_add_context_to_words(
    change_timeline: bool, split_field: str, expected: str
) -> None:
    words = []
    ind = 0
    timeline = "lpp"
    for letter_idx, word in enumerate("abcd"):
        if word == "d" and change_timeline:
            timeline = "lpp2"
        for i in range(3):
            words.append(
                {
                    "text": word,
                    "type": "Word",
                    "sentence": (f"{word} " * 3).strip() + ". ",
                    "sentence_char": 2 * i,
                    "text_char": letter_idx * 100 + 2 * i,
                    "start": ind,
                    "duration": 0.2,
                    "timeline": timeline,
                    "split": "test" if word == "a" else "train",
                }
            )
            ind += 1
    df = pd.DataFrame(words)
    df = _transf.AddContextToWords(sentence_only=False, split_field=split_field)(df)
    assert list(df.context)[-1] == expected


def _make_sentence_events(
    text: str = "he runs. she eats",
    words: tp.Sequence[str] = tuple("he runs. she eats".split()),
    add_sentence_to_words: bool = True,
) -> pd.DataFrame:
    """make dataset and apply Sentence transform"""
    common = dict(timeline="foo", language="english")
    events = [dict(type="Text", start=-0.1, duration=len(words), text=text, **common)]
    events += [
        dict(type="Word", start=float(i), duration=0.1, text=word, **common)
        for i, word in enumerate(words)
    ]
    df = ns.events.standardize_events(pd.DataFrame(events))
    if add_sentence_to_words:
        df = _transf.AddSentenceToWords()(df)
    return df


def test_sentence_to_word_standard() -> None:
    # two sentences + punctucation
    text = "he runs. she eats"
    words = "he runs she eats".split()
    df = _make_sentence_events(text, words)
    assert df.iloc[2].sentence == "he runs. "
    assert df.iloc[-1].sentence == "she eats"
    assert (df[df.type == "Sentence"].language == "english").all()

    # tokenization
    words = "I don't run".split()
    text = "I don't run"
    df = _make_sentence_events(text, words)
    assert sum(df.type == "Word") == len(words) == 3
    # TODO capitalization
    # TODO paragraphs


def test_sentence_to_word_missing_word(caplog: pytest.LogCaptureFixture) -> None:
    text = "le petit prince"
    words = ("le", "pince")
    caplog.set_level(logging.WARNING)
    df = _make_sentence_events(text, words)
    assert "Approximately matched" in caplog.text
    assert sum(df.type == "Word") == len(words) == 2
    assert df.iloc[2].sentence == df.iloc[3].sentence
    msg = "found 'pince' to start at the wrong character in {text!r}"
    assert df.loc[3].sentence_char == 10, msg


@pytest.mark.parametrize(
    "text,words,exp_char",
    [
        ("le petit prince", ("le", "Prince"), 9),
        ("froid d'hiver", ("froid", "hiver"), 8),
        ("froid d'hiver", ("froid", "d'hiver"), 6),
        ("froid d'hiver continental", ("froid", "hiver", "continental"), 14),
        ("froid d'hiver continental", ("froid", "hixer", "continental"), 14),
    ],
)
def test_sentence_to_word_cases(
    text: str, words: tp.Sequence[str], exp_char: int
) -> None:
    df = _make_sentence_events(text, words)
    assert sum(df.type == "Word") == len(words)
    assert df.iloc[0].text == text  # Text
    assert df.iloc[1].text == text  # Sentence
    assert df.iloc[2].sentence == df.iloc[-1].sentence == text
    msg = f"found {words[-1]!r} to start at wrong char in {text!r}"
    assert df.iloc[-1].sentence_char == exp_char, msg


@pytest.mark.parametrize(
    "text,words,exp_chars",
    [
        ("froid d'hiver continental", ("froid", "hiver", "continental"), [0, 8, 14]),
        ("froid d'hiver continental", ("froid", "hixer", "continental"), [0, 8, 14]),
        ("froid d'hiver continental", ("froid", "dixer", "continental"), [0, 8, 14]),
        ("aaaa bbbb cccc cccc", ("aaax", "bbbx", "cccc", "cccc"), [0, 5, 10, 15]),
        ("aaaa bbbb cccc cccc", ("aaaa", "bbbx", "cccx", "cccc"), [0, 5, 10, 15]),
        ("aaaa bbbb cccc dddd", ("aaaa", "bbbx", "cccx", "dddd"), [0, 5, 10, 15]),
        ("aaaa bbbb cccc dddd", ("aaaa", "bbbb", "cccx", "dddx"), [0, 5, 10, 15]),
        ("It's a me, Mario", ("It s", "me", "Maro"), [0, 7, 11]),
        ("le petit prince", ("le", "Prince"), [0, 9]),
        (
            '"Salut, ça va ?".\n"Oui, tranquille."',
            ["salut", "va", "oui", "tranquille"],
            [1, 11, 19, 24],
        ),
        (
            '"Salut, ça va ?".\n"Oui, tranquille."',
            ["salut", "va", "oui", "tranxxxxxx"],
            [1, 11, 19, -1],
        ),
        (
            "Salut, ça va ? Oui, tranquille.",
            ["salut", "va", "oui", "tranquille"],
            [0, 10, 0, 5],
        ),
        # accent-insensitive matching (French)
        ("tout à fait", ["tout", "a", "fait"], [0, 5, 7]),
        ("il répondit doucement", ["il", "repondit", "doucement"], [0, 3, 12]),
        ("grâce à toi", ["grace", "a", "toi"], [0, 6, 8]),
        # ligature-insensitive matching (French œ)
        ("une œuvre d'art", ["une", "oeuvre", "art"], [0, 4, 12]),
        ("le cœur bat", ["le", "coeur", "bat"], [0, 3, 8]),
        # ligature in phase 2 gap: text_char must not shift
        ("l'œuvre rare", ["zz", "oeuvre", "rare"], [-1, 2, 8]),
        ("cœur rouge", ["coeur", "rogue"], [0, 5]),
        ("coeur rouge", ["cœur", "rogue"], [0, 6]),
    ],
)
def test_match_text_words(
    text: str, words: tp.Sequence[str], exp_chars: tp.Sequence[int]
) -> None:
    info = _tutils.TextWordMatcher(text).match(words)
    out_chars = [i.get("sentence_char", -1) for i in info]
    np.testing.assert_array_equal(out_chars, exp_chars)


def test_failure_case() -> None:
    text = "It's supposed to be cement. Well? Well it's copper “Really?”"  # "?" can make a mess for match list
    words = text.split()
    info = _tutils.TextWordMatcher(text).match(words)
    assert all(i for i in info)


def test_duplicate() -> None:
    text = "It's supposed to be cement."
    words = text.split()
    words = words[:2] + words[1:]
    info = _tutils.TextWordMatcher(text).match(words)
    assert set(info[1]) == {"sentence"}  # should have sentence but not sentence_char


def test_gap_sentence_at_boundary() -> None:
    """Unmatched word at a sentence boundary gets the correct sentence."""
    text = "He ran.She walked slowly."
    info = _tutils.TextWordMatcher(text).match(["He", "ran", "She's", "walked", "slowly"])
    assert info[2]["text_char"] == 7
    assert info[2]["sentence"] == "She walked slowly."


def test_pending_word_between_repeated_sentence_strings() -> None:
    """An unmatched word sandwiched between two same-string sentences is not back-filled."""
    text = "Hi. Yes please. Hi. End now."
    info = _tutils.TextWordMatcher(text).match(["Hi", "QQQQ", "Hi", "End", "now"])
    assert info[1].get("text_char") is None
    assert info[1].get("sentence") != "Hi. ", info[1]


def test_gap_punctuated_word_offset() -> None:
    """Punctuation in a gap word must not shift text_char of subsequent words."""
    text = "aaa hello! world bbb"
    info = _tutils.TextWordMatcher(text).match(["aaa", "hello!", "world", "bbb"])
    assert info[1].get("text_char") == 4
    assert info[2].get("text_char") == 11


def test_duplicate_full_pipeline(recwarn: pytest.WarningsRecorder) -> None:
    # try full pipeline as one transform:
    df = _make_test_dataframe(duplicate=2)
    df = _transf.AssignWordSplitAndContext(max_context_len=6)(df)
    ws = [x.message for x in recwarn if "removed in Pydantic V3" not in str(x.message)]
    assert not ws  # setting an item of incompatible dtype (for context column)
    # no restart after duplicate:
    assert df.loc[6].context == "Peut-être bien que cet"


def test_remove_missing() -> None:
    df = _make_test_dataframe()
    df.loc[:, "context"] = ""
    df.loc[len(df) // 2 :, "context"] = np.nan
    df.loc[:3, "context"] = "blublu"
    out = _transf.RemoveMissing()(df)
    assert tuple(out.type) == ("Text", "Word", "Word", "Word")


@pytest.mark.parametrize("use_string_dtype", [False, True], ids=["object", "string"])
def test_remove_missing_na_treated_as_missing(use_string_dtype: bool) -> None:
    """NA values in the target field should be treated as missing (removed)."""
    df = _make_test_dataframe()
    n_words = (df.type == "Word").sum()
    df.loc[:, "context"] = "valid"
    word_idx = df.index[df.type == "Word"]
    df.loc[word_idx[0], "context"] = np.nan
    df.loc[word_idx[1], "context"] = ""
    df.loc[word_idx[2], "context"] = pd.NA
    if use_string_dtype:
        df["context"] = df["context"].astype("string")
    out = _transf.RemoveMissing()(df)
    remaining_words = (out.type == "Word").sum()
    assert remaining_words == n_words - 3
    assert (out.type == "Text").sum() == 1


def test_ensure_texts() -> None:
    _, df = _make_test_events()
    df = ns.events.standardize_events(df)
    assert "Text" not in df.type.values
    result = _transf.EnsureTexts(punctuation=None)(df)
    assert "Text" in result.type.values
    text_row = result[result.type == "Text"].iloc[0]
    assert text_row.timeline == "foo"
    # already has Text -> no-op
    assert len(_transf.EnsureTexts()(result)) == len(result)
    # spacy adds capitalization + period
    spacy_result = _transf.EnsureTexts(punctuation="spacy")(df)
    spacy_text = spacy_result[spacy_result.type == "Text"].iloc[0].text
    assert spacy_text[0].isupper()
    # multiple timelines
    df2 = df.copy()
    df2["timeline"] = "bar"
    both = ns.events.standardize_events(pd.concat([df, df2], ignore_index=True))
    multi = _transf.EnsureTexts(punctuation=None)(both)
    assert len(multi[multi.type == "Text"]) == 2
    assert set(multi[multi.type == "Text"].timeline) == {"foo", "bar"}


@pytest.mark.skipif("CI" in os.environ, reason="DL punctuation model not in CI")
def test_ensure_texts_fullstop() -> None:
    _, df = _make_test_events()
    df = ns.events.standardize_events(df)
    try:
        result = _transf.EnsureTexts(punctuation="fullstop")(df)
    except TypeError as e:
        if "grouped_entities" in str(e):
            pytest.skip("deepmultilingualpunctuation incompatible with transformers")
        raise
    text_row = result[result.type == "Text"].iloc[0]
    assert text_row.text != " ".join(df[df.type == "Word"].text)


@pytest.mark.parametrize("use_extra", [False, True])
def test_event_aligner(
    tmp_path: Path, use_extra: bool, caplog: pytest.LogCaptureFixture
) -> None:
    fp = tmp_path / "fake.data"
    fp.touch()
    extra: tp.Any = {"extra": {"cond": "A"}} if use_extra else {}
    events = [
        etypes.Meg(
            start=0, duration=10, filepath=fp, timeline="tl1", frequency=10, subject="s1"
        ),
        etypes.Word(start=1, duration=2, text="blublu", timeline="tl1", **extra),
        etypes.Image(start=2, duration=1, filepath=fp, timeline="tl1", **extra),
        etypes.Word(start=4, duration=1, text="other", timeline="tl1", **extra),
        #
        etypes.Meg(
            start=0, duration=10, filepath=fp, timeline="tl2", frequency=10, subject="s1"
        ),
        etypes.Word(start=5, duration=1, text="blublu", timeline="tl2", **extra),
    ]
    df = pd.DataFrame([e.to_dict() for e in events])
    df = ns.events.standardize_events(df)
    trigger_field: tp.Any = ("text", "cond") if use_extra else "text"
    enh = _transf.AlignEvents(
        trigger_type="Word", trigger_field=trigger_field, types_to_align="Meg"
    )
    if use_extra:
        assert "{'cond'}" in caplog.text
    out = enh(df)
    out = ns.events.standardize_events(out)
    assert all(x in ["tl1", "tl2"] for x in out.origin_timeline)
    segments = {
        seg.trigger.text: seg  # type: ignore[union-attr]
        for seg in ns.segments.list_segments(out, out.type == "Word")
    }
    assert len(segments) == 2
    assert len(segments["blublu"].ns_events) == 4
    suffix = ",A" if use_extra else ""
    assert ("AlignEvents:blublu" + suffix,) * 4 == tuple(
        e.timeline for e in segments["blublu"].ns_events
    )
    assert len(segments["other"].ns_events) == 2


def test_short_aligned_timeline_name() -> None:
    out = _transf.AlignEvents._gen_timeline_name(("abc" * 12, 12))
    assert out == "abcabcab..bcabcabc,12,20c1ffef"


def test_query_events():
    _, events = _make_test_events()
    transform = _transf.QueryEvents(query="text == 'a'")
    new_events = transform(events)
    assert (new_events.text == "a").all()


def test_query_events_invalid():
    with pytest.raises(ValueError):
        _transf.QueryEvents(query="text = 'a'")


@pytest.mark.parametrize(
    "query",
    [
        "index >= 0",
        "text == 'a'",
        "timeline_index < 3",
        "subject == 'subject1' and timeline_index < 2",
        "type in ['Word', 'Audio']",
    ],
)
def test_check_query_valid(query: str) -> None:
    assert ns.base.validate_query(query) == query


@pytest.mark.parametrize(
    "query",
    [
        "text = 'a'",
        "index >=",
        "def foo():",
        "import os",
    ],
)
def test_check_query_invalid_syntax(query: str) -> None:
    with pytest.raises(ValueError, match="is not valid"):
        ns.base.validate_query(query)


def test_check_query_rejects_at_variable() -> None:
    with pytest.raises(ValueError, match="@"):
        ns.base.validate_query("index == @my_var")


def _make_index_test_events() -> pd.DataFrame:
    """3 subjects x 2 timelines x 2 words = 12 rows."""
    events_list = []
    for subj_idx, subject in enumerate(["s1", "s2", "s3"]):
        for tl_idx, timeline in enumerate(["tl_a", "tl_b"]):
            for word_idx, word in enumerate(["hello", "world"]):
                events_list.append(
                    dict(
                        type="Word",
                        text=word,
                        start=float(subj_idx * 4 + tl_idx * 2 + word_idx),
                        duration=0.5,
                        subject=subject,
                        timeline=f"{subject}_{timeline}",
                    )
                )
    return pd.DataFrame(events_list)


@pytest.mark.parametrize(
    "query,expected_len,expected_subjects,expected_timelines",
    [
        ("subject_index < 2", 8, {"s1", "s2"}, 4),
        ("subject_timeline_index < 1", 6, {"s1", "s2", "s3"}, 3),
        ("subject_index < 2 and subject_timeline_index == 0", 4, {"s1", "s2"}, 2),
        ("subject == 's1'", 4, {"s1"}, 2),
    ],
)
def test_query_events_index_columns(
    query: str, expected_len: int, expected_subjects: set, expected_timelines: int
) -> None:
    events = _make_index_test_events()
    result = _transf.QueryEvents(query=query)(events)
    assert len(result) == expected_len
    assert set(result.subject) == expected_subjects
    assert result.timeline.nunique() == expected_timelines
    for col in ("subject_index", "subject_timeline_index"):
        assert col not in result.columns


def test_query_events_index_ambiguous():
    events = _make_index_test_events()
    events["subject_index"] = 99
    with pytest.raises(ValueError, match="Ambiguous"):
        _transf.QueryEvents(query="subject_index < 2")(events)


def test_query_events_index_preserves_insertion_order():
    """Index columns reflect row appearance order (sort=False), not sorted key order.

    Subjects are grouped with subj_b first. With sort=False, timeline_index
    numbers follow appearance order, so timeline_index < 2 returns subj_b's
    timelines. With the old sort=True it would alphabetically reorder and
    return subj_a's timelines instead.
    """
    events_list = []
    for subject in ["subj_b", "subj_a"]:
        for tl in ["tl1", "tl2"]:
            events_list.append(
                dict(
                    type="Word",
                    text="hello",
                    start=0.0,
                    duration=0.5,
                    subject=subject,
                    timeline=f"{subject}_{tl}",
                )
            )
    events = pd.DataFrame(events_list)
    # subj_b appears first ⇒ subject_index 0 should be subj_b
    result = _transf.QueryEvents(query="subject_index < 1")(events)
    assert set(result.subject) == {"subj_b"}
    # timeline_index < 2 keeps the first 2 timelines (both from subj_b)
    result2 = _transf.QueryEvents(query="timeline_index < 2")(events)
    assert set(result2.subject) == {"subj_b"}
    assert set(result2.timeline) == {"subj_b_tl1", "subj_b_tl2"}


def test_create_column_query_value():
    _, events = _make_test_events()
    transform = _transf.CreateColumn(
        column="split",
        default_value="train",
        query_row="text == 'a'",
        query_value="test",
        on_column_exists="ignore",
    )
    new_events = transform(events)
    assert (new_events[new_events.split == "train"].text != "a").all()
    assert (new_events[new_events.split == "test"].text == "a").all()


def test_create_column_query_column_name():
    _, events = _make_test_events()
    transform = _transf.CreateColumn(
        column="new_duration",
        default_value=-1.0,
        query_row="text == 'a'",
        query_column_name="duration",
        on_column_exists="ignore",
    )
    new_events = transform(events)
    assert (new_events[new_events.text == "a"].new_duration == 0.5).all()
    assert (new_events[new_events.text != "a"].new_duration == -1.0).all()


def test_create_column_error():
    with pytest.raises(pydantic.ValidationError):
        _ = _transf.CreateColumn(
            column="type",
            default_value="train",
            query_row="text == 'a'",
            query_value="test",
            query_column_name="test",
            on_column_exists="raise",
        )


def test_create_column_default_value_none():
    _, events = _make_test_events()
    transform = _transf.CreateColumn(
        column="new_col",
        query_row="text == 'a'",
        query_column_name="duration",
    )
    new_events = transform(events)
    assert "new_col" in new_events.columns
    matched = new_events[new_events.text == "a"]
    assert (matched.new_col == 0.5).all()
    unmatched = new_events[new_events.text != "a"]
    assert unmatched.new_col.isna().all()


def test_create_column_exists_error():
    _, events = _make_test_events()
    transform = _transf.CreateColumn(
        column="type",
        default_value="train",
        query_row="text == 'a'",
        query_value="test",
        on_column_exists="raise",
    )
    with pytest.raises(ValueError):
        _ = transform(events)


def test_select_idx():
    _, events = _make_test_events()
    events.loc[0, "timeline"] = "bar"
    assert events.timeline.nunique() == 2

    for idx in (0, [0]):
        transform = _transf.SelectIdx(column="timeline", idx=idx)
        events_ = transform(events)
        assert events_.timeline.nunique() == 1


class FakeChapter(_transf.EventsTransform):
    path: Path

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return super()._exclude_from_cls_uid() + ["path"]

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        out = []
        for tl, df in events.groupby("timeline"):
            kw = dict(timeline=tl, subject=df.subject.values[0])
            meg_fp = df.loc[df.type == "Meg", "filepath"].iloc[0]
            sound = self.path / "noise.wav"
            out.extend(
                [
                    dict(start=0, type="Meg", filepath=meg_fp, **kw),
                    dict(start=1, type="Audio", filepath=sound, **kw),
                    dict(start=1.0, duration=0.5, type="Word", text="hello", **kw),
                    dict(start=2.0, duration=0.5, type="Word", text="world", **kw),
                ]
            )
        df = pd.DataFrame(out)
        df = ns.events.standardize_events(df)

        return df


class Nothing(_transf.EventsTransform):
    is_default: int = 12

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        return events


def _custom_study(test_data_path: Path, tmp_path: Path) -> ns.Chain:
    from neuralset.extractors.test_audio import create_wav

    create_wav(tmp_path / "noise.wav", fs=44100, duration=10)
    steps: list[tp.Any] = [
        {"name": "Test2023Meg", "path": test_data_path},
        FakeChapter(path=tmp_path, infra="Cached"),  # type: ignore
        "Nothing",
        _transf.ChunkEvents(event_type_to_chunk="Audio", max_duration=5.0),
    ]
    return ns.Chain(
        steps=steps,
        infra={"backend": "Cached", "folder": tmp_path / "cache"},  # type: ignore[arg-type]
    )


def test_chunk(test_data_path: Path, tmp_path: Path) -> None:
    chain = _custom_study(test_data_path, tmp_path)
    events = chain.run().query('type=="Audio"')
    assert len(events) == 6
    assert events.duration.max() == 5.0


def test_configure_event_loader(test_data_path: Path) -> None:
    """Test type filtering and proper param merging."""
    base_events = pd.DataFrame(
        [
            dict(
                type="Fmri",
                filepath="method:_load?timeline=tl1",
                start=0,
                duration=1.0,
                timeline="tl1",
                space="custom",
                subject="test",
                frequency=1.0,
            ),
            dict(
                type="Meg",
                filepath="method:_load_meg?timeline=tl1",
                start=0,
                duration=1.0,
                timeline="tl1",
            ),
        ]
    )
    # Add a plain filepath row to verify it's left unchanged
    events = pd.concat(
        [
            base_events,
            pd.DataFrame(
                [
                    dict(
                        type="Fmri",
                        filepath="/plain/path.nii",
                        start=1,
                        duration=1.0,
                        timeline="tl1",
                        space="custom",
                        subject="test",
                        frequency=1.0,
                    )
                ]
            ),
        ]
    )
    transform = _transf.ConfigureEventLoader(event_types="Fmri", params={"space": "MNI"})
    result = transform(events)

    fmri_rows = result[result.type == "Fmri"]
    # Method URI should have merged params
    assert fmri_rows.iloc[0].filepath == "method:_load?timeline=tl1&space=MNI"
    # Plain filepath should be unchanged
    assert fmri_rows.iloc[1].filepath == "/plain/path.nii"
    # Meg event should NOT be updated
    assert "space=MNI" not in result[result.type == "Meg"].iloc[0].filepath

    # Multiple event types via tuple
    multi = _transf.ConfigureEventLoader(
        event_types=("Fmri", "Meg"),
        params={"preprocessing": "hp"},
    )
    assert all("preprocessing=hp" in fp for fp in multi(base_events).filepath)

    # SpecialLoader JSON filepath branch
    study_events = ns.Study(
        name="Test2023Fmri",
        path=test_data_path,
        infra_timelines={"cluster": None},  # type: ignore[arg-type]
    ).run()
    sl_transform = _transf.ConfigureEventLoader(
        event_types="Fmri",
        params={"space": "MNI", "resolution": "2mm"},
    )
    sl_result = sl_transform(study_events)
    assert json.loads(sl_result[sl_result.type == "Fmri"].iloc[0].filepath)["kwargs"] == {
        "space": "MNI",
        "resolution": "2mm",
    }


def test_per_step_validation_in_chain() -> None:
    """Validation fires at each EventsTransform._run, even inside a chain."""

    class DropType(_transf.EventsTransform):
        """Bad transform that drops the required 'type' column."""

        def _run(self, events: pd.DataFrame) -> pd.DataFrame:
            return events.drop(columns=["type"])

    _, events = _make_test_events()
    transform = DropType()
    with pytest.raises(ValueError, match='"type" column'):
        transform(events)
    with pytest.raises(ValueError, match='"type" column'):
        transform._run(events)


def create_wav(fp: Path, fs: int = 44100, duration: float = 10) -> None:
    y = np.random.randn(int(duration * fs))
    scipy.io.wavfile.write(fp, fs, y)


def test_deterministic_splitter() -> None:
    with pytest.raises(ValueError):
        splitter = _tutils.DeterministicSplitter(ratios=dict(train=0.5))
    splitter = _tutils.DeterministicSplitter(ratios=dict(train=0.5, test=0.5))
    assert splitter("0") == "train"
    assert splitter("1") == "test"
    assert splitter("10101001010101") == "train"


def test_chunk_events(tmp_path: Path) -> None:
    fp = tmp_path / "noise.wav"
    create_wav(fp, fs=44100, duration=10.1)
    sound = dict(type="Audio", start=0, timeline="foo", filepath=fp)
    words = [
        dict(
            type="Word",
            text="a",
            start=i,
            duration=i + 1,
            language="english",
            timeline="foo",
            split="train" if i % 2 else "test",
        )
        for i in range(11)
    ]
    events_list = [sound] + words
    events = pd.DataFrame(events_list)
    events = ns.events.standardize_events(events)

    events2 = _tutils.chunk_events(
        events, event_type_to_chunk="Audio", event_type_to_use="Word"
    )
    sounds = events2[events2["type"] == "Audio"]
    assert len(sounds) == 11
    assert all(sounds.offset.values == list(range(11)))

    events3 = _tutils.chunk_events(
        events, event_type_to_chunk="Audio", event_type_to_use="Word", min_duration=0.5
    )
    sounds = events3[events3["type"] == "Audio"]
    assert len(sounds) == 10

    events4 = _tutils.chunk_events(events, event_type_to_chunk="Audio", max_duration=2)
    sounds = events4[events4["type"] == "Audio"]
    assert len(sounds) == 6

    events5 = _tutils.chunk_events(
        events,
        event_type_to_chunk="Audio",
        max_duration=2,
        min_duration=0.5,
    )
    sounds = events5[events5["type"] == "Audio"]
    assert len(sounds) == 5


def test_cluster_assignment():
    test_cases = [
        {
            "clusters": [0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
            "expected_splits": [
                "train",
                "train",
                "train",
                "train",
                "val",
                "val",
                "val",
                "test",
                "test",
                "train",
            ],
            "ratios": {"train": 0.5, "val": 0.3, "test": 0.2},
        },
        {
            "clusters": [0, 0, 1, 1, 2, 2, 3, 3, 4, 5],
            "expected_splits": [
                "train",
                "train",
                "train",
                "train",
                "train",
                "train",
                "train",
                "train",
                "test",
                "val",
            ],
            "ratios": {"train": 0.85, "val": 0.10, "test": 0.05},
        },
        {
            "clusters": [0, 0, 1, 1, 1, 2, 3, 4, 4, 5],
            "expected_splits": [
                "train",
                "train",
                "val",
                "val",
                "val",
                "train",
                "test",
                "test",
                "test",
                "test",
            ],
            "ratios": {"train": 0.3, "val": 0.3, "test": 0.4},
        },
    ]

    for case in test_cases:
        splitter = _transf.SimilaritySplitter(
            extractor=ns.extractors.text.TfidfEmbedding(), ratios=case["ratios"]
        )
        clusters = case["clusters"]
        result = splitter._cluster_assignment(clusters)
        expected_splits = case["expected_splits"]
        assert result == expected_splits


def test_sentences_dataset():
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

    mock_events.append(
        {
            "type": "Image",
            "start": 0,
            "duration": None,
            "text": "",
        }
    )

    mock_events = pd.DataFrame(mock_events)
    splitter = _transf.SimilaritySplitter(
        extractor=ns.extractors.text.TfidfEmbedding(),
        threshold=0.5,
    )
    mock_events = splitter(mock_events)

    not_sentence_event = mock_events[mock_events["type"] != "Sentence"]
    events_empty_split = mock_events[mock_events["split"] == ""]
    n_empty = len(events_empty_split)
    n_non_sent = len(not_sentence_event)
    assert n_empty == n_non_sent, "Non-Sentence events should have NaN split"

    splits = mock_events["split"]
    assert splits[0] == splits[2], "Similar sentences should be grouped"
    assert splits[0] != splits[3], "Dissimilar sentences should not be grouped"


def create_image(path: Path, pixel_value: int) -> None:
    """Generates a synthetic image with a uniform pixel value and saves it to the specified path."""
    image_shape = (100, 100)
    img_array = np.ones(image_shape, dtype=np.uint8) * pixel_value
    img = Image.fromarray(img_array)
    img.save(path)


@pytest.fixture
def image_fpaths(tmp_path: Path):
    """Create images with different pixel values and returns their paths."""
    pixel_values = [1, 1, 250, 30]
    filepaths = [tmp_path / f"image_{k + 1}.png" for k in range(len(pixel_values))]

    for fp, value in zip(filepaths, pixel_values):
        create_image(fp, value)

    return filepaths


def test_similarity_splitter(image_fpaths: list[Path]) -> None:
    mock_events = pd.DataFrame(
        {
            "type": ["Image"] * len(image_fpaths) + ["Sentence"],
            "start": [0] * len(image_fpaths) + [0],
            "timeline": [""] * len(image_fpaths) + [""] * 1,
            "duration": [0.1] * len(image_fpaths) + [None],
            "filepath": [str(filepath) for filepath in image_fpaths] + [None],
        }
    )

    feat = ns.extractors.image.HuggingFaceImage(device="cpu", layers=1.0)
    splitter = _transf.SimilaritySplitter(extractor=feat)
    events = splitter(mock_events)

    not_image_event = events[events["type"] != "Image"]
    events_empty_split = events[events["split"] == ""]

    n_empty = len(events_empty_split)
    n_non_img = len(not_image_event)
    assert n_empty == n_non_img, "Non-Image events should have NaN split values"

    splits = events["split"]
    assert splits[0] == splits[1], "Similar images should be grouped"
    assert splits[0] != splits[2], "Dissimilar images should not be grouped"

    splitted_events = splitter.compute_clusters(mock_events)
    assert splitted_events["cluster_id"].to_list() == [0, 0, 1, 2]
    assert (splitted_events.type == "Image").all()
    assert splitted_events.shape[0] == 4


def test_sklearn_split() -> None:
    events = pd.DataFrame(
        {
            "type": ["Word"] * 20,
            "start": [float(i) for i in range(20)],
            "duration": [1.0] * 20,
            "timeline": [f"session_{i // 2}" for i in range(20)],
        }
    )
    kw: tp.Any = dict(
        split_by="timeline",
        valid_split_ratio=0.2,
        valid_random_state=42,
        test_random_state=42,
    )
    # basic split
    output = _transf.SklearnSplit(test_split_ratio=0.1, **kw)(events)
    assert set(output["split"]) == {"train", "val", "test"}
    assert output.split.value_counts().to_dict() == {"train": 14, "val": 4, "test": 2}
    # stratified split
    events["label"] = events.index // 10
    output = _transf.SklearnSplit(test_split_ratio=0.2, stratify_by="label", **kw)(events)
    assert output.split.value_counts().to_dict() == {"train": 12, "val": 4, "test": 4}
    assert output.value_counts(["split", "label"]).to_dict() == {
        ("train", 1): 6,
        ("train", 0): 6,
        ("val", 0): 2,
        ("val", 1): 2,
        ("test", 1): 2,
        ("test", 0): 2,
    }
    # duplicate groups within split_by → error
    events["label"] = events.index % 2
    with pytest.raises(ValueError, match="Stratified splitting requires unique values"):
        _transf.SklearnSplit(test_split_ratio=0.2, stratify_by="label", **kw)(events)


def test_no_circular_import_extractors() -> None:
    """Regression: extractors must load when neuralset.__init__ hasn't
    pre-imported events (the path submitit pickle workers take)."""
    ns_path = str(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, types; "
            f"m = types.ModuleType('neuralset'); m.__path__ = [{ns_path!r}]; "
            "sys.modules['neuralset'] = m; "
            "import neuralset.extractors",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
