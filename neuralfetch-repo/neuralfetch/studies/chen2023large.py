# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from functools import cached_property
from pathlib import Path

import mne
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy.io import loadmat

from neuralfetch import download
from neuralset.events import study

from .tuh_eeg import _fix_ch_name


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
    requirements: tp.ClassVar[tuple[str, ...]] = ("openxl",)
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=123,
        num_subjects=123,
        num_events_in_query=29,  # 1 Eeg event + 28 Stimulus events (query=1st timeline by default)
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(32, 4881000),  # 32-channel EEG
        frequency=1000,
    )

    def _download(self) -> None:
        """Data can also be downloaded manually after login at:
        https://www.synapse.org/Synapse:syn50614194
        """
        download.Synapse(
            study="Chen2023Large",
            study_id="syn50614194",
            dset_dir=self.path,
        ).download()

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for subj_ind in range(123):
            yield dict(subject=f"sub{subj_ind:03}")

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        duration = self._load_raw(timeline).duration
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg_df = pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    start=0.0,
                    filepath=info,
                    duration=duration,
                ),
            ]
        )
        stim_df = self._load_stimulus_events(timeline)
        return pd.concat([eeg_df, stim_df]).reset_index(drop=True)

    def _get_filename(self, filetype: str, timeline: dict[str, tp.Any]) -> Path:
        file_dir = self.path / "download"
        match filetype:
            case "eeg":
                file_dir = file_dir / "Data" / timeline["subject"]
                file_path = "data.bdf"
            case "events":
                file_dir = file_dir / "Data" / timeline["subject"]
                file_path = "evt.bdf"
            case "stimuli_info":
                file_path = "Stimuli_info.xlsx"
            case "ratings":
                file_dir = file_dir / "Data" / timeline["subject"]
                file_path = "After_remarks.mat"
        return file_dir / file_path

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        eeg_file = self._get_filename("eeg", timeline)
        raw = mne.io.read_raw_bdf(eeg_file)
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
        annots = mne.read_annotations(self._get_filename("events", timeline))
        annots_df = annots.to_data_frame(time_format=None)

        # Select movie events
        movie_inds = [str(x) for x in range(1, 29)]
        inds = annots_df[annots_df["description"].isin(movie_inds)].index
        video_desc = annots_df.loc[
            annots_df["description"].isin(movie_inds), "description"
        ].values
        assert (annots_df.loc[inds + 1, "description"] == "101").all(), (
            "Video clip index should be followed by trigger value 101"
        )
        assert (annots_df.loc[inds + 2, "description"] == "102").all(), (
            "Video clip start should be followed by trigger value 102"
        )
        video_clip_start = annots_df.loc[inds + 1, "onset"]
        video_clip_end = annots_df.loc[inds + 2, "onset"]
        video_clip_duration = video_clip_end.values - video_clip_start.values
        events_df = pd.DataFrame(
            {
                "type": "Stimulus",
                "start": video_clip_start.values,
                "duration": video_clip_duration,
                "video_index": video_desc,
            }
        )
        events_df = events_df.set_index("video_index")
        assert events_df.shape[0] == len(movie_inds), (
            "Expected number of video clips should match number of video clips"
        )

        # Merge
        events_df = events_df.merge(
            self._stimulus_info,
            on="video_index",
            how="left",
            suffixes=("", "_stimulus_info"),
        ).reset_index()
        events_df = events_df.rename(columns={"label": "code"})

        # Add ratings
        # ratings_df = self._load_ratings()
        # events_df = events_df.merge(ratings_df, on="video_index", how="left", suffixes=("", "_ratings"))

        return events_df

    def _load_ratings(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        fname = self._get_filename("ratings", timeline)
        out = loadmat(fname)["After_remark"]

        # Item order from DataStructureOfBehaviouralData.xlsx
        item_names = ["joy", "tenderness", "inspiration", "amusement", "anger", "disgust"]
        item_names += ["fear", "sadness", "arousal", "valence", "familiarity", "liking"]

        ratings = []
        for clip in out:
            scores = clip["score"][0][0]
            trial = clip["trial"][0][0][0]
            video_ind = clip["vid"][0][0][0]
            scores_dict = dict(zip(item_names, scores.tolist()))
            ratings.append({"video_index": str(video_ind), "trial": trial, **scores_dict})
        ratings_df = pd.DataFrame(ratings)
        return ratings_df

    @cached_property
    def _stimulus_info(self) -> pd.DataFrame:
        stim_df = pd.read_excel(
            self._get_filename("stimuli_info", {"": ""}), skipfooter=6
        )
        rename_cols = {
            "Video index": "video_index",
            "Duration（s）": "duration",
            "Source Film": "video_title",
            "Source Database": "video_database",
            "Valence": "valence",
            "Targeted Emotion": "description",
        }
        stim_df = stim_df.rename(columns=rename_cols)
        EmotionCategories = CategoricalDtype(
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
        stim_df["description"] = (
            stim_df["description"].str.lower().replace("\\", "neutral")
        ).astype(EmotionCategories)
        stim_df["valence"] = stim_df["valence"].str.lower()
        stim_df["label"] = stim_df["description"].cat.codes
        stim_df["duration"] = stim_df["duration"].astype(float)
        stim_df["video_index"] = stim_df["video_index"].astype(str)
        stim_df = stim_df.set_index("video_index")
        return stim_df
