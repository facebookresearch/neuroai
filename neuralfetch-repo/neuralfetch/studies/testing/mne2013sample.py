# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path
from warnings import warn

import mne
import numpy as np
import pandas as pd

import neuralset as ns
from neuralset.events import etypes, study
from neuralset.events.utils import extract_events


class Mne2013Sample(study.Study):
    # Study level
    url: tp.ClassVar[str] = "https://mne.tools/stable/overview/datasets_index.html#sample"
    bibtex: tp.ClassVar[str] = """
    @article{gramfort2013meg,
        title = {{{MEG}} and {{EEG}} Data Analysis with {{MNE-Python}}},
        author = {Gramfort, Alexandre and Luessi, Martin and Larson, Eric and Engemann, Denis A. and Strohmeier, Daniel and Brodbeck, Christian and Goj, Roman and Jas, Mainak and Brooks, Teon and Parkkonen, Lauri and H{\"a}m{\"a}l{\"a}inen, Matti},
        year = 2013,
        month = dec,
        journal = {Frontiers in Neuroscience},
        volume = {7},
        publisher = {Frontiers},
        issn = {1662-453X},
        doi = {10.3389/fnins.2013.00267},
        langid = {english}
    }
    """
    description: tp.ClassVar[str] = """MNE-Python sample MEG/EEG dataset"""
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1,
        num_subjects=1,
        num_events_in_query=289,
        event_types_in_query={"Meg", "Stimulus"},
        data_shape=(306, 41700),  # 32-channel EEG
        frequency=150.15,
    )

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self.infra_timelines.version = (
            "v3"  # other option requires overriding _load_events
        )

    def _download(self) -> None:
        """Download the MNE sample dataset.

        The dataset will be automatically downloaded by MNE-Python
        if not already present at the specified path.
        """

        # absolute: mne.datasets.sample.data_path nests relative paths
        path = (self.path / "download").absolute()
        path.mkdir(parents=True, exist_ok=True)
        mne.datasets.sample.data_path(path, verbose=True)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        yield {"subject": "sample"}

    def _get_data_path(self) -> Path:
        if self.path.absolute().is_relative_to(ns.CACHE_FOLDER):
            return mne.datasets.sample.data_path(ns.CACHE_FOLDER) / "MEG" / "sample"
        else:
            msg = f"Standard cache folder for MNE sample not available ({ns.CACHE_FOLDER=}), using {self.path=})"
            warn(msg)
            return self.path / "download" / "MNE-sample-data" / "MEG" / "sample"

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        data_path = self._get_data_path()
        raw_fname = data_path / "sample_audvis_filt-0-40_raw.fif"
        raw = mne.io.read_raw_fif(raw_fname, verbose=False)
        freq = raw.info["sfreq"]

        # Find stimulus events
        events = mne.find_events(raw)
        df = pd.DataFrame(events, columns=["start_idx", "duration_idx", "trigger"])
        df["start"] = df.start_idx / freq
        df["duration"] = 1.0 / freq
        df["type"] = "Stimulus"
        df["code"] = df.trigger.map(lambda x: x - 1)

        # Add extra info
        df["modality"] = None
        df = df.loc[df.trigger <= 4].reset_index(drop=True)
        df.loc[df.trigger <= 2, "modality"] = "audio"
        df.loc[(df.trigger == 3) | (df.trigger == 4), "modality"] = "visual"
        df.loc[df.trigger.isin([1, 3]), "side"] = "left"
        df.loc[df.trigger.isin([2, 4]), "side"] = "right"
        df["description"] = df.side + "_" + df.modality  # type: ignore

        # Meg and Eeg events
        # start="auto": auto-resolve from raw.first_samp
        meg = {"type": "Meg", "filepath": raw_fname, "start": "auto"}
        df = pd.concat([pd.DataFrame([meg]), df])
        return df


class Mne2013SampleEeg(Mne2013Sample):
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1,
        num_subjects=1,
        num_events_in_query=289,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(60, 41700),  # 32-channel EEG
        frequency=150.15,
    )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        events = super()._load_timeline_events(timeline)
        events["type"] = events["type"].replace("Meg", "Eeg")
        return events


class Fake2025Meg(Mne2013Sample):
    """Builds on the Mne2013Sample events and adds:
    - Image events during visual stimuli (an image with red right side
      for right side events, an image with blue left side for left side event)
    - Word events for each stimulus (visual-right = right, visual-left = left
      audio-left = "bottom" and audio-right = top)
    - Text event concatenating all words on sentences of 4 words.
    - Sentence events: one per group of 4 words (matching the Text sentence boundaries).
    - Audio event with a 400Hz tone on the left channel for left-audio stimuli, and
      600 Hz on the right for right-audio stimuli.

    Note
    ----
    - in the initial study, Stimuli events have duration=0.006 with around 0.6 stride
    - new events have duration duration=0.2 for Word and Image, and the total duration
      of the stimuli period for Text and Audio events.
    """

    description: tp.ClassVar[str] = (
        """mne sample MEG dataset with additional fake events"""
    )

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # sequential: no need to spawn processes for test data
        self.infra_timelines.cluster = None

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=2,
        num_subjects=2,
        num_events_in_query=794,
        event_types_in_query={
            "Meg",
            "Stimulus",
            "Image",
            "Audio",
            "Text",
            "Word",
            "Sentence",
        },
        data_shape=(306, 41700),  # 32-channel EEG
        frequency=150.15,
    )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        # download before concurrent timeline processing:
        mne.datasets.sample.data_path(self.path.absolute())
        for k in range(2):
            yield {"subject": f"sample{k}"}

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        df = super()._load_timeline_events(timeline)
        # Make fake word audio and image
        generated: list[ns.events.Event] = []
        df["timeline"] = "tmp"
        stims = extract_events(df, "Stimulus")
        first_start = min(e.start for e in stims)
        tot_duration = max(e.start + e.duration for e in stims) - first_start
        # Generate fake images
        path = Path(self.path).absolute()
        data_path = mne.datasets.sample.data_path(path) / "fake-data"
        data_path.mkdir(exist_ok=True)
        for side in ["right", "left"]:
            fp = data_path / f"fake-{side}-image.png"
            if not fp.exists():
                import PIL.Image

                array = np.zeros((128, 128, 3))
                if side == "right":  # right side is red
                    array[:, -32:, 0] = 255
                else:  # left side is blue
                    array[:, :32, 2] = 255
                im = PIL.Image.fromarray(array.astype(np.uint8))
                im.save(fp)
        # generate events
        fps = 8_000
        rng = np.random.RandomState(seed=12)
        audio = 0.002 * rng.randn(int(tot_duration * fps), 2)
        duration = 0.2  # in the study, duration=0.006 with around 0.6 stride
        for e in stims:
            kwargs: tp.Any = {"timeline": "x", "start": e.start, "duration": duration}
            side = e.extra["side"]
            if e.modality == "visual":  # type: ignore
                generated.append(etypes.Word(**kwargs, text=side, language="en"))
                generated.append(etypes.Image(**kwargs, filepath=fp))
                fp = data_path / f"fake-{side}-image.png"
            else:  # audio
                # make text different from visual words
                text = "top" if side == "left" else "bottom"
                generated.append(etypes.Word(**kwargs, text=text, language="en"))
                # add a frequency on left and different on right
                freq = 400 if side == "left" else 800
                tone = 0.5 * np.sin(
                    2 * np.pi * freq * np.linspace(0, duration, int(fps * duration))
                )
                ind = int((e.start - first_start) * fps)
                audio[ind : ind + len(tone), int(side == "right")] = tone
        # save audio
        fp = data_path / "fake-audio.wav"
        if not fp.exists():
            import scipy

            scipy.io.wavfile.write(fp, fps, audio)
        generated.append(
            etypes.Audio(
                filepath=fp,
                start=first_start,
                duration=tot_duration,
                timeline="x",
            )
        )
        # full text
        parts: list[str] = []
        for k, w in enumerate(e for e in generated if isinstance(e, etypes.Word)):
            text = w.text
            if not parts or parts[-1].endswith("."):
                text = text.capitalize()
            if k % 4 == 3:
                text += "."
            parts.append(text)
        generated.append(
            etypes.Text(
                text=" ".join(parts),
                language="en",
                start=first_start,
                duration=tot_duration,
                timeline="x",
            )
        )
        # sentence events (one per group of 4 words, matching the Text sentence boundaries)
        words_list = [e for e in generated if isinstance(e, etypes.Word)]
        for i in range(0, len(words_list), 4):
            group = words_list[i : i + 4]
            sentence_parts = []
            for j, w in enumerate(group):
                text = w.text
                if j == 0:
                    text = text.capitalize()
                if j == len(group) - 1:
                    text += "."
                sentence_parts.append(text)
            generated.append(
                etypes.Sentence(
                    text=" ".join(sentence_parts),
                    language="en",
                    start=group[0].start,
                    duration=group[-1].start + group[-1].duration - group[0].start,
                    timeline="x",
                )
            )
        # make a dataframe
        new = pd.DataFrame([e.to_dict() for e in generated])
        return pd.concat([df, new]).drop(columns="timeline")
