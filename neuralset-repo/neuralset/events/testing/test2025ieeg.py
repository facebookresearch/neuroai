# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import mne
import numpy as np
import pandas as pd

from neuralset.events import study


class _BaseTest2025Ieeg(study.Study):
    """Clone of Test2023Meg but adapted for testing iEEG extractors."""

    ch_types: tp.ClassVar[list[str]]
    sfreq: tp.ClassVar[float]

    def _download(self) -> None:
        raise NotImplementedError

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for i in range(3):
            yield dict(subject=str(i))

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        tl = timeline
        fname = self.path / f"sub-{tl['subject']}-raw.fif"

        if not fname.exists():
            ch_names = [
                f"electrode-{i}-{tl['subject']}" for i in range(len(self.ch_types) - 1)
            ] + ["INVALID_CHANNEL"]
            n_times = int(self.sfreq * 5 * 60)
            info = mne.create_info(ch_names, sfreq=self.sfreq, ch_types=self.ch_types)
            data = np.ones((len(ch_names), 1)) * np.arange(n_times)[None, :] / self.sfreq
            raw = mne.io.RawArray(data, info, verbose=False)

            fname.parent.mkdir(exist_ok=True)
            raw.save(fname, verbose=False)

        return mne.io.read_raw_fif(fname)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        sentences = "hello world. the quick brown fox. they quit. good bye."

        events = []
        start = 1.0
        splits = ["train", "test", "val"]
        for sid, sentence in enumerate(sentences.split(".")):
            if not sentence:
                continue
            sentence += "."
            for word in sentence.split():
                events.append(
                    dict(
                        start=start,
                        text=word,
                        duration=len(word) / 30,
                        type="Word",
                        language="english",
                        modality="heard",
                        split=splits[sid % 3],
                    )
                )
                start += 0.5
            start += 2.0
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        events.append({"type": "Ieeg", "filepath": info, "start": 0})
        return pd.DataFrame(events)


class Test2025Ieeg(_BaseTest2025Ieeg):
    ch_types: tp.ClassVar[list[str]] = ["seeg"] * 25 + ["ecog"] * 26
    sfreq: tp.ClassVar[float] = 2048.0

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=3,
        num_subjects=3,
        num_events_in_query=11,
        event_types_in_query={"Ieeg", "Word"},
        data_shape=(51, 614400),
        frequency=2048.0,
    )

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        raw = super()._load_raw(timeline)

        # Set bads
        raw.info["bads"] = ["INVALID_CHANNEL"]

        # Add random montage
        xyz_coords = np.random.randn(len(raw.ch_names), 3)  # convert from mm to meters
        ch_pos = dict(zip(raw.ch_names, xyz_coords))
        ch_pos["INVALID_CHANNEL"] = np.array(
            [np.nan, np.nan, np.nan]
        )  # Add dummy channel
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="mni_tal")
        raw = raw.set_montage(montage, on_missing="raise")

        return raw
