# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import importlib.util
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest


def _make_test_events():
    sentences = [f"This is sentence {i}" for i in range(3)]
    events_list = [
        dict(
            type="Word",
            text=word,
            language="english",
            split="train",
            sequence_id=i,
            start=0,
            duration=1,
            timeline="foo",
        )
        for i, sentence in enumerate(sentences)
        for word in sentence.split(" ")
    ]
    events = pd.DataFrame(events_list)
    return events


def test_add_sentences() -> None:
    from neuralfetch.utils import add_sentences

    events = _make_test_events()
    events = add_sentences(events)
    assert "Sentence" in events.type.unique()
    assert len(events.query('type=="Sentence"')) == 3
    words = events.query('type=="Word"')
    assert words.sentence.isna().sum() == 0


def test_download_things_images_missing_password_raises(tmp_path: Path) -> None:
    """download_things_images raises RuntimeError when images absent and no password."""
    from neuralfetch.utils import download_things_images

    old_val = os.environ.pop("NEURALFETCH_THINGS_PASSWORD", None)
    try:
        with pytest.raises(RuntimeError, match="THINGS-images requires"):
            download_things_images(tmp_path)
    finally:
        if old_val is not None:
            os.environ["NEURALFETCH_THINGS_PASSWORD"] = old_val


_QUERY_STUDY_SOURCE = """\
import typing as tp
import pandas as pd
from neuralset.events import study

class DummyQueryTest2099(study.Study):
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(query="timeline_index < 2")
    def iter_timelines(self):
        yield from ({"subject": f"s{i}"} for i in range(3))
    def _load_timeline_events(self, timeline):
        return pd.DataFrame([{"type": "Stimulus", "start": 0, "duration": 1, "code": 1}])
"""


def test_format_study_info_keeps_non_default_query() -> None:
    """A non-default query must survive serialization back to source."""
    from neuralfetch.utils import format_study_info

    actual = dict(
        num_timelines=3,
        num_subjects=3,
        query="timeline_index < 2",
        num_events_in_query=2,
        event_types_in_query={"Stimulus"},
    )
    assert 'query="timeline_index < 2"' in format_study_info(actual)


def test_format_study_info_omits_default_query() -> None:
    """Studies on the default query emit no ``query=``, so their files don't churn."""
    from neuralfetch.utils import format_study_info

    actual = dict(
        num_timelines=1,
        num_subjects=1,
        num_events_in_query=5,
        event_types_in_query={"Stimulus"},
    )
    # ``\\b`` does not match inside ``num_events_in_query=``: ``_`` is a word char.
    assert re.search(r"\bquery=", format_study_info(actual)) is None


def test_update_source_info_preserves_non_default_query(tmp_path: Path) -> None:
    """Rewriting ``_info`` must not reset a custom query while keeping its counts."""
    from neuralfetch.utils import update_source_info
    from neuralset.events import study as study_mod

    study_file = tmp_path / "dummy_query_study.py"
    study_file.write_text(_QUERY_STUDY_SOURCE)
    spec = importlib.util.spec_from_file_location("dummy_query_study", study_file)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    sys.modules["dummy_query_study"] = mod
    try:
        actual = update_source_info("DummyQueryTest2099", folder=tmp_path)
        # The query keeps 2 of the 3 timelines, one event each.
        assert actual["query"] == "timeline_index < 2"
        assert actual["num_events_in_query"] == 2
        assert 'query="timeline_index < 2"' in study_file.read_text("utf8")
    finally:
        study_mod.STUDIES.pop("DummyQueryTest2099", None)
        sys.modules.pop("dummy_query_study", None)
