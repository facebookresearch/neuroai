# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import exca
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
        frequencies: dict[str, float] = {}
        unique_videos = video_events.drop_duplicates("filepath")
        # write audio file per video file
        for video_event in tqdm(
            unique_videos.itertuples(),
            total=len(unique_videos),
            desc="Extract audio from videos",
        ):
            # timeline is unused but required by events; wavs are shared across timelines
            filepath, timeline = str(video_event.filepath), str(video_event.timeline)
            audio_filepath = Path(filepath).with_suffix(".wav")
            if not audio_filepath.exists():
                # whole file: Video events may only cover part of it
                video = ev.Video(start=0, timeline=timeline, filepath=filepath)
                with video.read() as video_clip:
                    if video_clip.audio is None:
                        continue
                    with exca.utils.temporary_save_path(audio_filepath) as tmp_filepath:
                        video_clip.audio.write_audiofile(tmp_filepath)
            audio = ev.Audio(start=0, timeline=timeline, filepath=str(audio_filepath))
            frequencies[filepath] = audio.frequency
        events_to_add = []
        # add audio events per video event (may be multiple per audio/video file)
        for video_event in video_events.itertuples(index=False):
            filepath = str(video_event.filepath)
            if filepath not in frequencies:
                continue
            # Audio overwrites duration with length of wav file if frequency is missing
            audio_event = video_event._replace(
                type="Audio",
                filepath=str(Path(filepath).with_suffix(".wav")),
                frequency=frequencies[filepath],
            )  # type: ignore
            events_to_add.append(audio_event)
        events = pd.concat([events, pd.DataFrame(events_to_add)], ignore_index=True)
        events = events.reset_index(drop=True)
        return events
