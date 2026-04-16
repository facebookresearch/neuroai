# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .. import etypes as ev
from ..study import EventsTransform


class ExtractAudioFromVideo(EventsTransform):
    """
    Extract audio tracks from Video events and add them as separate Audio events.

    This transform iterates over events of type "Video", extracts the audio
    track, saves it as a `.wav` file (if it does not already exist), and adds
    a corresponding event of type "Audio" to the DataFrame.
    """

    overwrite: bool = False

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        video_events = events.loc[events.type == "Video"]
        if self.overwrite:
            for filepath in video_events.filepath.unique():
                audio_filepath = Path(filepath).with_suffix(".wav")
                if audio_filepath.exists():
                    audio_filepath.unlink()
        if len(video_events) == 0:
            return events
        events_to_add = []
        for video_event in tqdm(
            video_events.itertuples(),
            total=len(video_events),
            desc="Extract audio from video events",
        ):
            audio_filepath = Path(video_event.filepath).with_suffix(".wav")  # type: ignore
            video_ns_event = ev.Video.from_dict(video_event)
            if not audio_filepath.exists():
                audio = video_ns_event.read().audio
                if not audio:
                    continue
                audio.write_audiofile(audio_filepath)
                audio.close()
            audio_event = video_event._replace(
                type="Audio", filepath=str(audio_filepath), frequency=pd.NA
            )  # type: ignore
            events_to_add.append(audio_event)
        events = pd.concat([events, pd.DataFrame(events_to_add)], ignore_index=True)
        events = events.reset_index(drop=True)
        return events
