# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


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
    import os

    from neuralfetch.utils import download_things_images

    old_val = os.environ.pop("NEURALFETCH_THINGS_PASSWORD", None)
    try:
        with pytest.raises(RuntimeError, match="THINGS-images requires"):
            download_things_images(tmp_path)
    finally:
        if old_val is not None:
            os.environ["NEURALFETCH_THINGS_PASSWORD"] = old_val
