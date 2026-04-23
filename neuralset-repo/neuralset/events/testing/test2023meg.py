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


class Test2023Meg(study.Study):
    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # sequential: no need to spawn processes for test data
        self.infra_timelines.cluster = None

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=3,
        num_subjects=3,
        num_events_in_query=9,
        event_types_in_query={"Meg", "Word", "Image"},
        data_shape=(19, 10_000),
        frequency=100,
    )

    def _download(self) -> None:
        raise NotImplementedError

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for i in range(3):
            yield dict(subject=str(i))

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        # FIXME should not be required? use spacy?
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        events = pd.DataFrame(
            [
                # use ":" in name to avoid file existence check
                dict(start=1.0, type="Word", text="hello", split="train"),
                dict(start=2.0, type="Word", text="world", split="train"),
                dict(start=11.0, type="Word", text="good", split="test"),
                dict(start=12.0, type="Word", text="bye", split="test"),
                dict(start=1.0, type="Image", filepath="fake:hello.png", split="train"),
                dict(start=2.0, type="Image", filepath="fake:world.png", split="train"),
                dict(start=11.0, type="Image", filepath="fake:good.png", split="test"),
                dict(start=12.0, type="Image", filepath="fake:bye.png", split="test"),
                dict(
                    start=0,
                    type="Meg",
                    filepath=info,
                ),
            ]
        )
        events.loc[events.type == "Image", "duration"] = 0.7
        events.loc[events.type == "Word", "duration"] = 0.2
        return events

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.Raw:
        tl = timeline
        fif = self.path / f"sub-{tl['subject']}-raw.fif"
        if not fif.exists():
            n_chans = 20
            # lets vary number of channels across subjects
            n_chans += int(tl["subject"])
            sfreq = 100.0
            n_times = 10_000
            # Use some actual channel names (and some channel names not found in default layout) so
            # we can test out channel position-related extractors
            layout = mne.channels.read_layout("Vectorview-mag")
            ch_names = layout.names[: n_chans - 1] + ["INVALID_CHANNEL"]
            ch_types = ["mag"] * (n_chans - 2) + ["grad", "misc"]
            info = mne.create_info(ch_names, sfreq=sfreq, ch_types=ch_types)

            # Add some channel positions
            locs = np.random.rand(n_chans, 3)
            for i, loc in enumerate(locs):
                if ch_names[i] != "INVALID_CHANNEL":
                    info["chs"][i]["loc"][:3] = loc

            np.random.seed(0)
            data = np.ones((n_chans, 1)) * np.arange(n_times)[None, :] / sfreq
            raw = mne.io.RawArray(data, info, verbose=False)
            fif.parent.mkdir(exist_ok=True)

            raw.save(fif, verbose=False)
        return mne.io.Raw(fif, verbose=False)
