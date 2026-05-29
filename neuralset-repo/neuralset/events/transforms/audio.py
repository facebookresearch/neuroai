# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import subprocess
import tempfile
import typing as tp
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .. import etypes as ev
from ..study import EventsTransform

logger = logging.getLogger(__name__)


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


class ExtractWordsFromAudio(EventsTransform):
    """
    Transcribe Audio events with WhisperX and add the aligned words as Word events.

    Transcripts are cached next to the audio file as ``.tsv`` files. Existing
    Word events are treated as authoritative and make the transform a no-op.
    """

    language: str = "english"
    overwrite: bool = False

    @staticmethod
    def _get_transcript_from_audio(wav_filename: Path, language: str) -> pd.DataFrame:
        language_codes = {
            "english": "en",
            "french": "fr",
            "spanish": "es",
            "dutch": "nl",
            "chinese": "zh",
        }
        if language not in language_codes:
            raise ValueError(f"Language {language} not supported")

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        with tempfile.TemporaryDirectory() as output_dir:
            logger.info("Running whisperx via uvx...")
            cmd = [
                "uvx",
                "whisperx",
                str(wav_filename),
                "--model",
                "large-v3",
                "--language",
                language_codes[language],
                "--device",
                device,
                "--compute_type",
                compute_type,
                "--batch_size",
                "16",
                "--align_model",
                "WAV2VEC2_ASR_LARGE_LV60K_960H" if language == "english" else "",
                "--output_dir",
                output_dir,
                "--output_format",
                "json",
            ]
            cmd = [c for c in cmd if c]
            env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"whisperx failed:\n{result.stderr}")

            json_path = Path(output_dir) / f"{wav_filename.stem}.json"
            transcript = json.loads(json_path.read_text())

        words: list[dict[str, tp.Any]] = []
        for sequence_id, segment in enumerate(transcript["segments"]):
            sentence = segment["text"].replace('"', "")
            for word in segment["words"]:
                if "start" not in word:
                    continue
                words.append(
                    {
                        "text": word["word"].replace('"', ""),
                        "start": word["start"],
                        "duration": word["end"] - word["start"],
                        "sequence_id": sequence_id,
                        "sentence": sentence,
                    }
                )

        return pd.DataFrame(words)

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if "Word" in events.type.unique():
            logger.warning("Words already present in the events dataframe, skipping")
            return events

        audio_events = events.loc[events.type == "Audio"]
        if audio_events.empty:
            return events

        transcripts: dict[str, pd.DataFrame] = {}
        audio_filepaths = [Path(filepath) for filepath in audio_events.filepath.unique()]
        for wav_filename in tqdm(
            audio_filepaths,
            total=len(audio_filepaths),
            desc="Extracting words from audio",
        ):
            transcript_filename = wav_filename.with_suffix(".tsv")
            if transcript_filename.exists() and not self.overwrite:
                try:
                    transcript = pd.read_csv(transcript_filename, sep="\t")
                except pd.errors.EmptyDataError:
                    transcript = pd.DataFrame()
                    logger.warning("Empty transcript file %s", transcript_filename)
            else:
                transcript = self._get_transcript_from_audio(wav_filename, self.language)
                transcript.to_csv(transcript_filename, sep="\t", index=False)
                logger.info("Wrote transcript to %s", transcript_filename)
            transcripts[str(wav_filename)] = transcript

        all_transcripts = []
        ignored_fields = {
            "frequency",
            "filepath",
            "type",
            "start",
            "duration",
            "offset",
            "stop",
        }
        for audio_event in audio_events.itertuples(index=False):
            transcript = transcripts[str(Path(audio_event.filepath))].copy(deep=True)
            if transcript.empty:
                continue
            for key, value in audio_event._asdict().items():
                if key not in ignored_fields:
                    transcript.loc[:, key] = value
            transcript["type"] = "Word"
            transcript["language"] = self.language
            offset = getattr(audio_event, "offset", 0.0)
            transcript["start"] += audio_event.start + offset
            all_transcripts.append(transcript)

        if not all_transcripts:
            logger.warning("No transcripts found, skipping")
            return events

        events = pd.concat([events, pd.concat(all_transcripts)], ignore_index=True)
        return events.reset_index(drop=True)
