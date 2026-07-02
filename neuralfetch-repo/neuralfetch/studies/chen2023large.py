# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import mne
import pandas as pd
from pandas.api.types import CategoricalDtype
from mne_bids import BIDSPath, read_raw_bids

from neuralfetch import download
from neuralset.events import study

from .tuh_eeg import _fix_ch_name

# Emotion categories (ordered) used to derive the integer ``code`` target for
# the emotion classification task. Matches the ordering in the original study.
_EMOTION_CATEGORIES = CategoricalDtype(
    categories=[
        "anger",
        "fear",
        "disgust",
        "sadness",
        "neutral",
        "amusement",
        "tenderness",
        "inspiration",
        "joy",
    ],
    ordered=True,
)


class Chen2023Large(study.Study):
    url: tp.ClassVar[str] = "https://www.synapse.org/Synapse:syn50614194/files/"
    """FACED: EEG responses to emotion-eliciting video clips.

    This study provides 32-channel EEG recordings from 123 participants viewing
    28 video clips targeting nine emotion categories (anger, fear, disgust,
    sadness, neutral, amusement, tenderness, inspiration, joy) with three
    valence levels (negative, neutral, positive).

    Experimental Design:
        - EEG recordings (32-channel, 1000 Hz)
        - 123 participants
        - 1 session per participant
        - Paradigm: passive viewing of emotion-eliciting video clips
            * 28 video clips targeting 9 emotion categories
            * 3 valence categories: negative, neutral, positive

    Download Requirements:
        - Synapse dataset: syn50614194 (https://doi.org/10.7303/syn50614194)
        - Requires Synapse account and authentication token

    Notes:
        - The dataset is described as open-access for research purposes in the
          associated paper, but no explicit Creative Commons or OSI license is
          specified on the Synapse project page. Use is governed by Synapse
          platform terms and any project-specific access conditions.
        - Users should obtain the dataset directly from Synapse and cite the
          original publication and dataset DOI.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = (
        "Finer-grained Affective Computing EEG Dataset",
        "FACED",
    )

    licence: tp.ClassVar[str] = "Custom"
    bibtex: tp.ClassVar[str] = """
    @article{chen2023large,
        url = {http://dx.doi.org/10.1038/s41597-023-02650-w},
        title={A Large Finer-grained Affective Computing EEG Dataset},
        volume={10},
        issn={2052-4463},
        url={http://dx.doi.org/10.1038/s41597-023-02650-w},
        doi={10.1038/s41597-023-02650-w},
        number={1},
        journal={Scientific Data},
        publisher={Springer Science and Business Media LLC},
        author={Chen,  Jingjing and Wang,  Xiaobin and Huang,  Chen and Hu,  Xin and Shen,  Xinke and Zhang,  Dan},
        year={2023},
        month=oct
    }

    @misc{tsinghua_emotion_bci2023thu,
        title={THU_EP},
        author={TSINGHUA_EMOTION_BCI},
        year=2023,
        publisher={Synapse},
        doi={10.7303/SYN50614194},
        urldate={2026-02-19},
        url={https://www.synapse.org/Synapse:syn50614194/files/}
    }
    """
    description: tp.ClassVar[str] = """
        EEG recordings for 123 participants while viewing 28 video clips targeting nine categories of emotion
    """
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=123,
        num_subjects=123,
        num_events_in_query=29,  # 1 Eeg event + 28 Stimulus events (query=1st timeline by default)
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(32, 4881000),  # 32-channel EEG
        frequency=1000,
    )

    # NEMAR re-host of FACED/THU_EP (BIDS). The reader discovers the on-disk
    # BIDS layout directly, so the events live in the BIDS ``events.tsv`` sidecar
    # rather than the Synapse ``evt.bdf`` + ``Stimuli_info.xlsx`` scheme.
    _BIDS_ID: tp.ClassVar[str] = "nm000112"

    @property
    def _bids_root(self):
        return self.path / "download" / self._BIDS_ID

    def _download(self) -> None:
        """Data can also be downloaded manually after login at:
        https://www.synapse.org/Synapse:syn50614194

        Repointed to the NEMAR re-host (nm000112) so the BIDS lands at the same
        root the reader discovers.
        """
        try:
            import nemar

            nemar.download(
                dataset=self._BIDS_ID,
                target_dir=self.path / "download",
                downloader="python",
            )
        except Exception:  # pragma: no cover - fallback to original Synapse source
            download.Synapse(
                study="Chen2023Large",
                study_id="syn50614194",
                dset_dir=self.path,
            ).download()

    def _get_bids_path(self, timeline: dict[str, tp.Any]) -> BIDSPath:
        """Returns the BIDS path for a timeline."""
        return BIDSPath(
            subject=timeline["subject"],
            task=timeline["task"],
            root=self._bids_root,
            datatype="eeg",
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        # NEMAR nm000112 BIDS layout is sub-XXX/eeg/sub-XXX_task-watchingVideoClips_*
        # (no session, no run) -- discover timelines from disk instead of assuming
        # the Synapse Data/subNNN/{data,evt}.bdf scheme.
        from mne_bids import find_matching_paths

        for bp in find_matching_paths(
            self._bids_root, datatypes="eeg", suffixes="eeg", extensions=[".bdf"]
        ):
            ev = bp.copy().update(suffix="events", extension=".tsv")
            if ev.fpath.exists():
                yield dict(subject=bp.subject, task=bp.task)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        stim_df = self._load_stimulus_events(timeline)
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = dict(type="Eeg", filepath=info, start=0.0)
        return pd.concat([pd.DataFrame([eeg]), stim_df], ignore_index=True)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.BaseRaw:
        raw = read_raw_bids(self._get_bids_path(timeline), verbose=False)
        raw = raw.rename_channels(_fix_ch_name)
        raw.set_montage("standard_1020", on_missing="ignore")
        if "HEOL" in raw.ch_names:
            raw.set_channel_types({"HEOL": "eog", "HEOR": "eog"})

        # Some files were saved in V, others in uV
        data, _ = raw[:, : int(raw.info["sfreq"] * 60 * 5)]  # First 5-mins
        std = data.std(axis=1).mean()
        if std > 1:  # in uV
            raw.load_data()
            raw._data /= 1e6  # convert to V

        return raw

    def _load_stimulus_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        events_path = (
            self._get_bids_path(timeline)
            .copy()
            .update(suffix="events", extension=".tsv")
        )
        events = pd.read_csv(events_path.fpath, sep="\t", encoding="utf-8-sig")

        # Keep only the video-clip stimulus events (rows carrying emotion
        # metadata). The "Experiment start" trigger rows have video_index == n/a,
        # which pandas parses to NaN -- drop them via notna() rather than a string
        # comparison.
        events = events[events["video_index"].notna()].copy()

        events = events.rename(
            columns={
                "onset": "start",
                "emotion_label": "emotion",
                "binary_label": "valence",
            }
        )
        events["type"] = "Stimulus"
        events["start"] = events["start"].astype(float)
        events["duration"] = events["duration"].astype(float)
        # Neutral clips are encoded with a bare backslash in emotion_label
        # (matches the original study's `.replace("\\", "neutral")`).
        events["emotion"] = (
            events["emotion"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"\\": "neutral"})
        )
        events["valence"] = events["valence"].astype(str).str.lower()
        # video_index is read as float (n/a -> NaN in other rows); normalise to a
        # clean integer-like string ("11" not "11.0").
        events["video_index"] = (
            events["video_index"].astype(float).astype(int).astype(str)
        )
        # Integer target for the emotion classification task.
        events["code"] = events["emotion"].astype(_EMOTION_CATEGORIES).cat.codes

        events = events[
            ["type", "start", "duration", "code", "emotion", "valence", "video_index"]
        ]
        return events.reset_index(drop=True)
