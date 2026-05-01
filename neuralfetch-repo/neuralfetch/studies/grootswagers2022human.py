# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import typing as tp
from pathlib import Path

import pandas as pd

from neuralfetch import download, utils
from neuralset.events import study


class Grootswagers2022Human(study.Study):
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds003825/versions/1.2.0"
    """THINGS-EEG1: EEG responses to RSVP streams of object images from THINGS database.

    This study provides EEG recordings from 50 participants viewing rapid serial
    visual presentation (RSVP) streams of object images spanning all 1,854 concepts
    in the THINGS database. Part of the THINGS initiative for studying object
    recognition in the human visual system.

    Experimental Design:
        - EEG recordings (64-channel, 1000 Hz, BioSemi)
        - 50 participants
        - 12 recording sessions per participant
        - Paradigm: Rapid serial visual presentation (RSVP)
            * ~82 image presentations per block
            * 10 blocks per session
            * Images from THINGS database covering all 1,854 object concepts

    Data Format:
        - BIDS-compliant dataset structure (OpenNeuro ds003825)
        - EEG data in BrainVision format
        - Shared THINGS image database expected in ../THINGS-images/

    Download Requirements:
        - OpenNeuro dataset: ds003825
        - THINGS-images database (shared across THINGS studies): checked for in
          ../THINGS-images/ and downloaded automatically if missing

    Notes:
        - Data was referenced online to Cz, which is not recorded (63 of 64 channels
          available for most participants)
        - Participants 49-50 have 127 channels; all others have 63
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("THINGS-EEG1",)
    licence: tp.ClassVar[str] = "CC0-1.0"
    bibtex: tp.ClassVar[str] = """
    @article{grootswagers2022human,
        title={Human {{EEG}} Recordings for 1,854 Concepts Presented in Rapid Serial Visual Presentation Streams},
        author={Grootswagers, Tijl and Zhou, Ivy and Robinson, Amanda K. and Hebart, Martin N. and Carlson, Thomas A.},
        year=2022,
        month=jan,
        journal={Scientific Data},
        volume={9},
        number={1},
        pages={3},
        publisher={Nature Publishing Group},
        issn={2052-4463},
        doi={10.1038/s41597-021-01102-7},
        copyright={2022 The Author(s)},
        langid={english}
    }

    @misc{grootswagers2022humana,
        title={Human Electroencephalography Recordings from 50 Subjects for 22,248 Images from 1,854 Object Concepts},
        author={Grootswagers, Tijl and Zhou, Ivy and Robinson, Amanda and Hebart, Martin and Carlson, Thomas},
        year=2022,
        publisher={Openneuro},
        doi={10.18112/OPENNEURO.DS003825.V1.2.0},
        url={https://openneuro.org/datasets/ds003825/versions/1.2.0}
    }
    """
    description: tp.ClassVar[str] = (
        "EEG recordings from 50 participants watching still images."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = (
        "pyunpack",
        "boto3",
        "osfclient>=0.0.5",
        "openneuro-py",
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=50,
        num_subjects=50,
        num_events_in_query=22249,
        event_types_in_query={"Eeg", "Image"},
        data_shape=(63, 3035740),
        frequency=1000,
    )

    _test_image_names: list[str] | None = None

    def _download(self) -> None:
        # Download EEG data from OpenNeuro
        download.Openneuro(study="ds003825", dset_dir=self.path).download()
        # Check for / Download THINGS-images database (used across multiple THINGS studies)
        # Note: Images must be preprocessed and placed in prepare/ folder separately
        utils.download_things_images(self.path)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        for k in range(1, 51):
            folder = self.path / "download" / f"sub-{k:02}" / "eeg"
            if not folder.exists():
                raise RuntimeError(f"Folder does not exist: {folder}")
            yield dict(subject=str(k))

    def _format_img_path(self, img_path: str) -> str:
        correct = img_path.endswith(".jpg") & img_path.startswith("stimuli")
        img_path = img_path.replace("\\", "/")
        correct &= img_path.count("/") == 2
        if not correct:
            msg = f"img_path must be .jpg, starting with stiumuli, with 2 '/', got {img_path}"
            raise ValueError(msg)
        filename = Path(img_path).name
        category = Path(img_path).parent.name
        out = self.path / "prepare" / category / filename
        if not out.exists():
            # Fall back to the shared THINGS-images directory (used by sample datasets)
            shared = (self.path / ".." / "THINGS-images" / category / filename).resolve(
                strict=False
            )
            if shared.exists():
                return str(shared)
            raise RuntimeError(f"Output path does not exist: {out}")
        return str(out)

    def _get_test_info(self) -> tuple[list[str], list[str]]:
        test_image_file = self.path / "download" / "test_images.csv"
        if not test_image_file.exists():
            # Some partial OpenNeuro downloads may omit this optional file.
            return [], []
        test_image_names = pd.read_csv(test_image_file, header=None)[0].tolist()
        test_categories = [name.split("/")[0] for name in test_image_names]
        return test_image_names, test_categories

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        subj_str = f"sub-{int(timeline['subject']):02}"
        folder = self.path / "download" / subj_str / "eeg"
        events_file = folder / f"{subj_str}_task-rsvp_events.tsv"
        events = pd.read_csv(events_file, sep="\t")

        events["filepath"] = events.stim.apply(self._format_img_path)
        events["category"] = events.filepath.apply(lambda x: x.split("/")[-2])
        events["type"] = "Image"

        sfreq = 1e3
        events["start"] = events.onset / sfreq
        events["duration"] /= sfreq

        events["caption"] = events.filepath.apply(
            lambda x: " ".join(Path(x).stem.split("_")[:-1])
        ).apply(lambda s: re.sub(r"\d", "", s))

        # Identify test images (called "validation images" in the paper)
        test_image_names, test_categories = self._get_test_info()
        events["split"] = "train"
        image_names = events.filepath.apply(lambda x: "/".join(x.split("/")[-2:]))
        events.loc[image_names.isin(test_image_names), "split"] = "test"
        events["is_test_category"] = events.category.isin(test_categories)
        events["stem"] = events.filepath.apply(lambda x: Path(x).stem)

        # Add shared THINGS filepaths to images
        shared_things_path = (self.path / ".." / "THINGS-images").resolve(strict=False)
        if shared_things_path.exists():
            fp = f"{shared_things_path}/{events.category}/{events.stem}.jpg"
            events["shared_filepath"] = fp

        # Only keep useful columns
        cols = [
            "filepath",
            "type",
            "start",
            "duration",
            "category",
            "caption",
            "split",
            "is_test_category",
            "stem",
            "shared_filepath",
        ]
        events = events[cols]

        # Add raw Eeg event
        raw_fname = folder / f"{subj_str}_task-rsvp_eeg.vhdr"
        # NOTE 1: Data was referenced online to Cz, which has not been recorded
        # NOTE 2: Subjects 1-48 have 63 channels; subject 49-50 have 127
        eeg = {"filepath": str(raw_fname), "type": "Eeg", "start": 0}
        events = pd.concat([pd.DataFrame([eeg]), events])

        return events


class Grootswagers2022HumanSample(Grootswagers2022Human):
    """Sample version of Grootswagers2022Human with selective OpenNeuro download.

    Keeps one EEG timeline (sub-01) and image events to enable fast loading
    checks without downloading the full dataset.

    Still needs to download the full THINGS-images database (shared across THINGS studies)
    to access the image files, but this is a much smaller download than the full EEG dataset.
    """

    description: tp.ClassVar[str] = (
        "Sample Grootswagers2022: OpenNeuro EEG + image events for sub-01."
    )

    _SAMPLE_SUBJECT: tp.ClassVar[str] = "01"

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1,
        num_subjects=1,
        num_events_in_query=22249,
        event_types_in_query={"Eeg", "Image"},
        data_shape=(63, 3035740),
        frequency=1000,
    )

    def _download(self, overwrite=False) -> None:
        sub = self._SAMPLE_SUBJECT
        include_patterns = [
            "dataset_description.json",
            "participants.tsv",
            f"sub-{sub}/eeg/sub-{sub}_task-rsvp_eeg.vhdr",
            f"sub-{sub}/eeg/sub-{sub}_task-rsvp_eeg.eeg",
            f"sub-{sub}/eeg/sub-{sub}_task-rsvp_eeg.vmrk",
            f"sub-{sub}/eeg/sub-{sub}_task-rsvp_events.tsv",
        ]
        download.Openneuro(
            study="ds003825",
            dset_dir=self.path,
            include=include_patterns,
        ).download(overwrite=overwrite)

        utils.download_things_images(self.path)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        sub = self._SAMPLE_SUBJECT
        eeg_file = (
            self.path
            / "download"
            / f"sub-{sub}"
            / "eeg"
            / f"sub-{sub}_task-rsvp_eeg.vhdr"
        )
        if not eeg_file.exists():
            raise FileNotFoundError(
                f"Missing sample EEG file for {self.__class__.__name__}: {eeg_file}. "
                "Please run study.download() first."
            )
        yield dict(subject=str(int(sub)))

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        subj_str = f"sub-{int(timeline['subject']):02}"
        folder = self.path / "download" / subj_str / "eeg"
        events_file = folder / f"{subj_str}_task-rsvp_events.tsv"
        events = pd.read_csv(events_file, sep="\t")

        events["filepath"] = events.stim.apply(self._format_img_path)
        events["category"] = events.filepath.apply(lambda x: Path(x).parent.name)
        events["type"] = "Image"
        sfreq = 1e3
        events["start"] = events.onset / sfreq
        events["duration"] /= sfreq
        events["caption"] = events.filepath.apply(
            lambda x: " ".join(Path(x).stem.split("_")[:-1])
        ).apply(lambda s: re.sub(r"\d", "", s))

        test_image_names, test_categories = self._get_test_info()
        image_names = events.filepath.apply(lambda x: "/".join(x.split("/")[-2:]))
        events["split"] = "train"
        events.loc[image_names.isin(test_image_names), "split"] = "test"
        events["is_test_category"] = events.category.isin(test_categories)
        events["stem"] = events.filepath.apply(lambda x: Path(x).stem)

        cols = [
            "filepath",
            "type",
            "start",
            "duration",
            "category",
            "caption",
            "split",
            "is_test_category",
            "stem",
        ]
        events = events[cols]

        raw_fname = folder / f"{subj_str}_task-rsvp_eeg.vhdr"
        eeg = {"filepath": str(raw_fname), "type": "Eeg", "start": 0}
        events = pd.concat([pd.DataFrame([eeg]), events], ignore_index=True)
        return events
