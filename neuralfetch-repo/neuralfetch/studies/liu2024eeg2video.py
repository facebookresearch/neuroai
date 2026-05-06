# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures
import logging
import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from neuralset.events import study

logger = logging.getLogger(__name__)


class Liu2024Eeg2video(study.Study):
    url: tp.ClassVar[str] = "https://bcmi.sjtu.edu.cn/home/eeg2video/"
    """SEED-DV: EEG responses to video clips.

    This dataset contains EEG recordings of participants viewing 1,400 two-second
    video clips spanning 40 visual concepts grouped into 9 categories. The study
    was designed to investigate decoding dynamic visual perception from EEG signals.

    Experimental Design:
        - EEG recordings (62-channel, 200 Hz)
        - 20 participants
        - 7 blocks per session; 1 participant has 2 sessions, others have 1
        - Paradigm: passive viewing of 2-second video clips
            * 40 concepts across 9 categories
            * 5 clips per sub-block, 40 sub-blocks per block
            * 1,400 total video clips

    Download Requirements:
        - Requires application and license agreement from BCMI Lab (SJTU):
          1. Download and sign the license: https://cloud.bcmi.sjtu.edu.cn/sharing/o64PBIsIc
          2. Submit the application form: https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/
             (select "SEED-DV" from the dataset checklist; use institutional email)
          3. After approval, download links are provided via email
        - Dataset page: https://bcmi.sjtu.edu.cn/home/eeg2video/

    Notes:
        - Part of the SEED dataset collection from BCMI Lab (SJTU).
        - Project page: https://bcmi.sjtu.edu.cn/home/eeg2video/
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("SJTU Emotion EEG Dataset DV", "SEED-DV")

    licence: tp.ClassVar[str] = "Custom"
    bibtex: tp.ClassVar[str] = """
    @article{liu2024eeg2video,
        title={EEG2video: Towards decoding dynamic visual perception from EEG signals},
        author={Liu, Xuan-Hao and Liu, Yan-Kai and Wang, Yansen and Ren, Kan and Shi, Hanwen and Wang, Zilong and Li, Dongsheng and Lu, Bao-Liang and Zheng, Wei-Long},
        journal={Advances in Neural Information Processing Systems},
        volume={37},
        pages={72245--72273},
        year={2024},
        doi={10.52202/079017-2306},
        url={https://doi.org/10.52202/079017-2306}
    }
    """
    # Mapping based on Fig. 1A of Liu et al., 2024 (see ref)
    # Cluster of each of 40 concepts to 9 courser classes used in benchmark
    CONCEPT_CATEGORIES: tp.ClassVar[dict] = {
        "cat": "land_animal",
        "dog": "land_animal",
        "elephant": "land_animal",
        "horse": "land_animal",
        "panda": "land_animal",
        "rabbit": "land_animal",
        "bird": "land_animal",
        "fish": "water_animal",
        "jellyfish": "water_animal",
        "shark-like": "water_animal",
        "turtle": "water_animal",
        "flower": "plant",
        "mushroom": "plant",
        "tree": "plant",
        "boxing": "exercise",
        "dancing": "exercise",
        "running": "exercise",
        "skiing": "exercise",
        "face": "human",
        "couple": "human",
        "crowd": "human",
        "beach": "natural_scene",
        "building": "natural_scene",
        "mountain": "natural_scene",
        "road": "natural_scene",
        "water": "natural_scene",
        "fireworks": "natural_scene",
        "banana": "food",
        "cake": "food",
        "drink": "food",
        "pizza": "food",
        "watermelon": "food",
        "drum": "musical_instrument",
        "guitar": "musical_instrument",
        "piano": "musical_instrument",
        "motorcycle": "transportation",
        "car": "transportation",
        "hot air balloon": "transportation",
        "airplane": "transportation",
        "ship": "transportation",
    }

    # Manually inferred from BLIP captions for each label
    CONCEPT_LABELS: tp.ClassVar[dict] = {
        1.0: "cat",
        2.0: "dog",
        3.0: "elephant",
        4.0: "horse",
        5.0: "panda",
        6.0: "rabbit",
        7.0: "bird",
        8.0: "fish",
        9.0: "jellyfish",
        10.0: "shark-like",
        11.0: "turtle",
        12.0: "flower",
        13.0: "mushroom",
        14.0: "tree",
        15.0: "boxing",
        16.0: "dancing",
        17.0: "running",
        18.0: "skiing",
        19.0: "face",
        20.0: "couple",
        21.0: "crowd",
        22.0: "beach",
        23.0: "building",
        24.0: "mountain",
        25.0: "road",
        26.0: "water",
        27.0: "fireworks",
        28.0: "banana",
        29.0: "cake",
        30.0: "drink",
        31.0: "pizza",
        32.0: "watermelon",
        33.0: "drum",
        34.0: "guitar",
        35.0: "piano",
        36.0: "motorcycle",
        37.0: "car",
        38.0: "hot air balloon",
        39.0: "airplane",
        40.0: "ship",
    }

    CLIP_DURATION_S: tp.ClassVar[float] = 2.0
    N_CLIPS_PER_SUBBLOCK: tp.ClassVar[int] = 5
    N_CLIPS_PER_CONCEPT: tp.ClassVar[int] = 40
    description: tp.ClassVar[str] = (
        "EEG recordings of 20 participants viewing 1,400 video clips (2s) for 40 concepts"
    )
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("moviepy>=2.1.2",)
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_subjects=20,
        num_timelines=147,
        num_events_in_query=201,
        event_types_in_query={"Eeg", "Video"},
        data_shape=(62, 104000),
        frequency=200.0,
    )

    def _download(self) -> None:
        raise NotImplementedError(
            "SEED-DV requires manual download. Steps:\n"
            "1. Download and sign the license agreement:\n"
            "   https://cloud.bcmi.sjtu.edu.cn/sharing/o64PBIsIc\n"
            "2. Submit the application form (select 'SEED-DV', use institutional email):\n"
            "   https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/\n"
            "3. After approval, download the dataset using the links sent via email\n"
            f"4. Place the data in: {self.path}\n"
            "5. Run prepare_video_clips() to extract individual video clips"
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        subjects = [f"sub{i}" for i in range(1, 21)]
        for subject in subjects:
            sessions = [None, "session2"] if subject == "sub1" else [None]
            for session in sessions:
                for block in range(7):
                    yield dict(subject=subject, session=session, block=block)

    @classmethod
    def _prepare_video_clip_block(
        cls,
        path: Path,
        save_dir: Path,
        block_name: str,
        block_id: int,
    ) -> None:
        from moviepy import VideoFileClip

        fname = path / "SEED-DV" / "Video" / f"{block_name}.mp4"

        # Video clip subblocks start at 3 s, then every 13 s (FPS=24)
        subblock_start_frame_inds = 72 + np.arange(cls.N_CLIPS_PER_CONCEPT) * 312

        with VideoFileClip(fname) as clip:
            block_video_events = []
            for j, frame_ind in enumerate(subblock_start_frame_inds):
                clip_start = (frame_ind + 1) / clip.fps
                for k in range(cls.N_CLIPS_PER_SUBBLOCK):
                    clip_name = (
                        f"block-{block_id + 1}_subblock-{j + 1:02d}_clip-{k + 1}.mp4"
                    )
                    video_clip_filename = save_dir / clip_name
                    video_clip = clip.subclipped(
                        clip_start, clip_start + cls.CLIP_DURATION_S
                    )
                    video_clip.write_videofile(str(video_clip_filename))
                    logger.info(f"Saved {video_clip_filename}")

                    # Create the event for this video clip
                    # Note: duration and frequency will be inferred when creating the event
                    video_event = dict(
                        type="Video",
                        start=clip_start,
                        filepath=video_clip_filename,
                    )
                    block_video_events.append(video_event)

                    clip_start += cls.CLIP_DURATION_S

        # Save the video events for this block
        block_video_events_ = pd.DataFrame(block_video_events)
        block_video_events_.to_csv(
            save_dir / f"block-{block_id + 1:02d}_video_events.csv",
            index=False,
        )

    @classmethod
    def prepare_video_clips(cls, path: Path) -> None:
        """Chunk the original .mp4 files into 2-s video clips."""
        save_dir = path / "prepare" / "video_clips"
        save_dir.mkdir(parents=True, exist_ok=True)

        block_names = [
            "1st_10min",
            "2nd_10min",
            "3rd_10min",
            "4th_10min",
            "5th_10min",
            "6th_10min",
            "7th_10min",
        ]

        jobs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(block_names)) as exc:
            for block_id, block_name in enumerate(block_names):
                jobs.append(
                    exc.submit(
                        cls._prepare_video_clip_block,
                        path,
                        save_dir,
                        block_name,
                        block_id,
                    )
                )

    def _get_video_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        block = timeline["block"]
        fname = (
            self.path
            / "prepare"
            / "video_clips"
            / f"block-{block + 1:02d}_video_events.csv"
        )
        if not fname.exists():
            msg = f"{fname} not found, make sure to run Liu2024Eeg2video().prepare_video_clips(path) first."
            raise FileNotFoundError(msg)
        video_df = pd.read_csv(fname, index_col=0)

        video_clips_dir = self.path / "prepare" / "video_clips"
        video_df["filepath"] = video_df["filepath"].apply(
            lambda fp: str(video_clips_dir / Path(fp).name)
        )

        # Groups of 5 clips for each of 40 concepts = 200 clips
        label_fname = (
            self.path / "SEED-DV" / "Video" / "meta-info" / "All_video_label.npy"
        )
        codes = np.load(label_fname)[block]
        concepts = [
            self.CONCEPT_LABELS[i]
            for i in codes
            for _ in range(self.N_CLIPS_PER_SUBBLOCK)
        ]
        video_df["concept"] = concepts
        video_df["category"] = video_df.concept.map(self.CONCEPT_CATEGORIES)

        return video_df

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg_df = pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    start=0.0,
                    filepath=info,
                ),
            ]
        )
        video_df = self._get_video_events(timeline)
        return pd.concat(
            [
                eeg_df,
                video_df,
            ]
        ).reset_index(drop=True)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        subject = timeline["subject"]
        session = timeline["session"]
        block = timeline["block"]
        folder = self.path / "SEED-DV"

        # Create info and montage
        locs_file = folder / "channel_62_pos.locs"
        locs_df = pd.read_csv(locs_file, sep=r"\s+", header=None, index_col=0)
        ch_names = list(locs_df[3].str.replace(" ", "").fillna(""))  # fix names
        info = mne.create_info(ch_names=ch_names, sfreq=200.0, ch_types="eeg")
        montage = mne.channels.read_custom_montage(locs_file)

        # Load EEG data and create Raw mne object
        if session is None:
            eeg_file = folder / "EEG" / f"{subject}.npy"
        else:
            eeg_file = folder / "EEG" / f"{subject}_{session}.npy"
        eeg_data = np.load(eeg_file)
        raw = mne.io.RawArray(eeg_data[block], info=info)
        raw = raw.set_montage(montage, on_missing="ignore")

        return raw
