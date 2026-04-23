# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from itertools import product

import mne
import numpy as np
import pandas as pd

from neuralset.events import study


class Test2024Fnirs(study.Study):
    """Clone of Test2023Meg but adapted for testing fNIRS extractors."""

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=3,
        num_subjects=3,
        num_events_in_query=11,
        event_types_in_query={"Fnirs", "Word"},
        data_shape=(32, 3000),
        frequency=10.0,
    )

    def _download(self) -> None:
        raise NotImplementedError

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for i in range(3):
            yield dict(subject=str(i))

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        tl = timeline
        fname = self.path / f"sub-{tl['subject']}-raw.fif"

        if not fname.exists():
            # E.g. Artinis Octamon has 8 sources and 2 detectors
            n_sources, n_detectors = 8, 2
            wavelengths = [760, 850]
            ch_names = [
                f"S{source + 1}_D{detector + 1} {wl}"
                for source, detector, wl in product(
                    range(n_sources), range(n_detectors), wavelengths
                )
            ]
            ch_types = ["fnirs_cw_amplitude"] * len(ch_names)
            sfreq = 10.0
            n_times = int(sfreq * 5 * 60)
            info = mne.create_info(ch_names, sfreq=sfreq, ch_types=ch_types)
            np.random.seed(0)
            data = np.ones((len(ch_names), 1)) * np.arange(n_times)[None, :] / sfreq
            raw = mne.io.RawArray(data, info, verbose=False)

            # TODO: Enable the following so we can test distance-based processing
            # montage = mne.channels.make_standard_montage('artinis-octamon')
            # raw.set_montage(montage)

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
        events.append({"type": "Fnirs", "filepath": info, "start": 0})
        return pd.DataFrame(events)
