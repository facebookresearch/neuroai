# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import typing as tp
from pathlib import Path

import nibabel
import pandas as pd

from neuralfetch import download
from neuralset.events import study
from neuralset.utils import get_bids_filepath, get_masked_bold_image, read_bids_events


class Chang2019Bold5000(study.Study):
    """BOLD5000: 3T fMRI responses to diverse image categories.

    This study provides 3T BOLD fMRI data from 4 participants viewing ~5,000
    images from three major computer vision datasets: scene images (SUN),
    ImageNet, and Microsoft COCO. The dataset is designed to study visual
    representation across diverse stimulus types.

    Experimental Design:
        - 3T fMRI recordings (TR = 2.0 seconds)
        - 4 participants (CSI1-4)
        - 9-15 scanning sessions per participant
        - 9-10 runs per session (varies by session)
        - Paradigm: passive viewing of still images
            * ~37 image presentations per run (2s duration each)
            * Three image categories: Scene (SUN), ImageNet, MS COCO
            * 113 images repeated across sessions for test-retest reliability

    Data Format:
        - BIDS-compliant dataset structure (OpenNeuro ds001499)
        - Preprocessed in MNI152NLin2009aSym space
        - 510 total timelines
        - Event types: Fmri, Image
        - ImageNet class labels and scene categories provided
        - Train/test split based on repeated stimuli

    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("BOLD5000",)
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds001499"
    bibtex: tp.ClassVar[str] = """
    @article{chang2019bold5000,
        title={{{BOLD5000}}, a Public {{fMRI}} Dataset While Viewing 5000 Visual Images},
        author={Chang, Nadine and Pyles, John A. and Marcus, Austin and Gupta, Abhinav and Tarr, Michael J. and Aminoff, Elissa M.},
        year=2019,
        month=may,
        journal={Scientific Data},
        volume={6},
        number={1},
        pages={49},
        publisher={Nature Publishing Group},
        issn={2052-4463},
        doi={10.1038/s41597-019-0052-3},
        copyright={2019 The Author(s)},
        langid={english},
    }

    @misc{chang2020bold5000,
        url = {https://openneuro.org/datasets/ds001499/versions/1.3.1},
        title={{{BOLD5000}}},
        author={Chang, Nadine and Pyles, John A. and Marcus, Austin and {Abhinav Gupta} and Tarr, Michael J. and Aminoff, Elissa M.},
        year=2020,
        publisher={Openneuro},
        doi={10.18112/OPENNEURO.DS001499.V1.3.1},
        url={https://openneuro.org/datasets/ds001499/versions/1.3.1},
    }
    """
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Preprocessed BOLD data (in MNI152NLin2009aSym) for"
        "4 participants watching still images in 3T fMRI"
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("openneuro-py>=2025.2.0",)

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=510,
        num_subjects=4,
        num_events_in_query=38,
        event_types_in_query={"Fmri", "Image"},
        data_shape=(77, 94, 80, 194),
        frequency=0.5,
        fmri_spaces=("custom",),
    )

    STIMULUS_URL: str = (
        "https://www.dropbox.com/s/5ie18t4rjjvsl47/BOLD5000_Stimuli.zip?dl=1"
    )

    SESSIONS_PER_SUBJECT: tp.ClassVar[dict[int, int]] = {1: 15, 2: 15, 3: 15, 4: 9}

    # Each session has 9 or 10 runs
    SESSIONS_WITH_10_RUNS: tp.ClassVar[dict[int, tuple[int, ...]]] = {
        1: (1, 2, 3, 5, 7, 10, 15),
        2: (1, 3, 4, 5, 11, 12, 14),
        3: (1, 3, 5, 6, 7, 11, 15),
        4: (1, 4, 9),
    }

    BIDS_FOLDER: tp.ClassVar[str] = "download"
    DERIVATIVES_FOLDER: tp.ClassVar[str] = "derivatives_in_standard_space"
    BOLD_SPACE: tp.ClassVar[str] = "MNI152NLin2009aSym"
    TASK: tp.ClassVar[str] = "5000scenes"
    TR_FMRI_S: tp.ClassVar[float] = 2.0
    STIMULI_FOLDER: tp.ClassVar[str] = "BOLD5000_Stimuli/"
    SUBJ_PADDING: tp.ClassVar[str] = "01"
    SUBJ_SUFFIX: tp.ClassVar[str] = "CSI"

    def _download(self) -> None:
        with download.success_writer(self.path / "download_all") as already_done:
            if already_done:
                return
            # Download fMRI data from OpenNeuro
            from neuralfetch.download import Openneuro

            Openneuro(study="ds001499", dset_dir=self.path).download()

            # Download stimuli data
            stimuli_zip = self.path / "BOLD5000_Stimuli.zip"
            print(f"Downloading stimuli to {stimuli_zip}...")
            download.download_file(self.STIMULUS_URL, stimuli_zip, show_progress=True)

            # Extract BOLD5000_Stimuli folder
            print(f"Extracting stimuli to {self.path}...")
            download.extract_zip(stimuli_zip, destination=self.path, remove_after=True)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for subject, sessions in self.SESSIONS_PER_SUBJECT.items():
            for session in range(1, sessions + 1):
                n_runs = 10 if session in self.SESSIONS_WITH_10_RUNS[subject] else 9
                for run in range(1, n_runs + 1):
                    yield dict(
                        subject=str(subject), session=session, task=self.TASK, run=run
                    )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        fmri = {
            "filepath": info,
            "type": "Fmri",
            "start": 0.0,
            "frequency": self._get_fmri_frequency(),
            "duration": self._get_bold_image(timeline).shape[-1] * self.TR_FMRI_S,
        }
        bids_events_df_fp = get_bids_filepath(
            root_path=self.path / self.BIDS_FOLDER,
            filetype="events",
            data_type="Fmri",
            subj_padding=self.SUBJ_PADDING,
            subj_suffix=self.SUBJ_SUFFIX,
            **timeline,
        )
        bids_events_df = read_bids_events(bids_events_df_fp)
        repeated_stimuli = self.get_repeated_stimuli()
        image_net_classes = self._get_imagenet_classes()
        ns_events_df = self._get_ns_img_events_df(
            bids_events_df,
            self._get_fmri_frequency(),
            image_net_classes,
            repeated_stimuli,
        )
        return pd.concat([pd.DataFrame([fmri]), ns_events_df], axis=0)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> nibabel.Nifti1Image:
        return get_masked_bold_image(
            self._get_bold_image(timeline), self._get_bold_mask(timeline)
        )

    def _get_ns_img_events_df(
        self,
        bids_events_df: pd.DataFrame,
        frequency: float,
        image_net_classes: dict[str, str],
        repeated_stimuli: list[str],
    ) -> pd.DataFrame:
        bids_events = bids_events_df.to_dict("records")
        ns_events = []
        for bids_event in bids_events:
            stimulus_path = self._get_stimulus_path(
                bids_event["stim_file"], bids_event["ImgType"]
            )
            fp = self.path / self.STIMULI_FOLDER / "Scene_Stimuli" / stimulus_path
            ns_event = dict(
                type="Image",
                start=bids_event["onset"],
                duration=bids_event["duration"],
                frequency=frequency,
                filepath=str(fp),
                split="test" if bids_event["stim_file"] in repeated_stimuli else "train",
                annotation=self._get_image_annotation(
                    bids_event["stim_file"], bids_event["ImgType"], image_net_classes
                ),
            )
            ns_events.append(ns_event)
        return pd.DataFrame(ns_events)

    def _get_stimulus_path(self, stim_file: str, img_type: str) -> Path:
        if img_type.endswith("scenes"):
            foldername = "Scene"
        elif img_type.endswith("coco"):
            foldername = "COCO"
        elif img_type.endswith("imagenet"):
            foldername = "ImageNet"
        else:
            raise ValueError(f"Unknown image type {img_type}")
        return Path("Presented_Stimuli") / foldername / stim_file

    def _get_bold_mask(self, timeline: dict[str, tp.Any]):
        fp = get_bids_filepath(
            root_path=self.path / self.DERIVATIVES_FOLDER,
            filetype="bold_mask",
            data_type="Fmri",
            space=self.BOLD_SPACE,
            subj_padding=self.SUBJ_PADDING,
            subj_suffix=self.SUBJ_SUFFIX,
            **timeline,
        )
        return nibabel.load(fp, mmap=True)

    def _get_bold_image(self, timeline: dict[str, tp.Any]):
        fp = get_bids_filepath(
            root_path=self.path / self.DERIVATIVES_FOLDER,
            filetype="bold",
            data_type="Fmri",
            space=self.BOLD_SPACE,
            subj_padding=self.SUBJ_PADDING,
            subj_suffix=self.SUBJ_SUFFIX,
            **timeline,
        )
        return nibabel.load(fp, mmap=True)

    def _get_fmri_frequency(self) -> float:
        return 1.0 / self.TR_FMRI_S

    def get_repeated_stimuli(self) -> list[str]:
        filepath = (
            self.path
            / self.STIMULI_FOLDER
            / "Scene_Stimuli"
            / "repeated_stimuli_113_list.txt"
        )
        return [line.strip() for line in filepath.read_text("utf8").splitlines()]

    def _get_imagenet_classes(self) -> dict[str, str]:
        image_to_annotations = {}
        filepath = (
            self.path / self.STIMULI_FOLDER / "Image_Labels" / "imagenet_final_labels.txt"
        )
        with filepath.open("r", encoding="utf8") as file:
            for line in file:
                identifier, string_list = line.split(" ", 1)
                string = " ".join([s.strip() for s in string_list.split(",")])
                image_to_annotations[identifier] = string
        return image_to_annotations

    def _get_image_annotation(
        self, stim_file: str, img_type: str, image_net_classes: dict[str, str]
    ) -> str:
        if img_type in ["scenes", "rep_scenes"]:
            match = re.match(r"([a-zA-Z_]+)(\d|[.])", stim_file)
            if match is not None:
                return match.group(1)
            else:
                raise ValueError(f"'{stim_file}' has an unexpected pattern")
        elif img_type in ["imagenet", "rep_imagenet"]:
            return image_net_classes[stim_file.split("_")[0]]
        elif img_type in ["coco", "rep_coco"]:
            return "none"
        else:
            msg = f"{img_type} should be one of 'scenes', 'imagenet', or 'coco'"
            raise ValueError(msg)
