# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pandas as pd
import pytest

from neuralset.events import etypes
from neuralset.events.transforms import ExtractAudioFromVideo
from neuralset.extractors.test_video import make_video_event


def test_extract_audio_from_partial_videos(tmp_path: Path) -> None:
    video = make_video_event(tmp_path, duration=6.0)
    spans = [(10.0, 2.0), (20.0, 0.0)]  # (start, offset), first span not at offset 0
    videos = [
        video.model_copy(update=dict(start=s, offset=o, duration=2.0)) for s, o in spans
    ]
    events = pd.DataFrame([v.to_dict() for v in videos])
    events = ExtractAudioFromVideo()(events)
    audios = [
        etypes.Audio.from_dict(r) for r in events.query("type == 'Audio'").itertuples()
    ]
    full = etypes.Audio(start=0, timeline="x", filepath=audios[0].filepath)
    assert full.duration == pytest.approx(6.0, abs=0.1)
    assert [(a.start, a.offset, a.duration) for a in audios] == [
        (s, o, 2.0) for s, o in spans
    ]
    for audio in audios:
        assert audio.read().shape[0] == 2 * audio.frequency
