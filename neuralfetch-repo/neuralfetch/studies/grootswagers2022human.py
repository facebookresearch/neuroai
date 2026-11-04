# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import typing as tp
from pathlib import Path

import pandas as pd

from neuralfetch import download
from neuralset.events import study


class Grootswagers2022Human(study.Study):
    url: tp.ClassVar[str] = "https://nemar.org/dataexplorer/detail?dataset_id=on003825"
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
        - NEMAR on003825 v1.0.2: BIDS import of OpenNeuro ds003825
        - EEG data in BrainVision format
        - The THINGS images are included under stimuli/<concept>/

    Download Requirements:
        - NEMAR dataset on003825 (EEG and images, ~50 GB; no OSF password)

    Notes:
        - Data was referenced online to Cz, which is not recorded (63 of 64 channels
          available for most participants)
        - Participants 49-50 have 127 channels; all others have 63
        - Test split: the 200 validation images (``isteststim`` in the events),
          each shown 12 times to 48 participants. Participants 1 and 6 never saw
          them, so all their events are train.
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

    @misc{nemar.on003825,
        title={Human electroencephalography recordings from 50 subjects for 22,248 images from 1,854 object concepts},
        publisher={NEMAR},
        doi={10.82901/nemar.on003825},
        url={https://doi.org/10.82901/nemar.on003825}
    }
    """
    description: tp.ClassVar[str] = (
        "EEG recordings from 50 participants watching still images."
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=50,
        num_subjects=50,
        num_events_in_query=22249,
        event_types_in_query={"Eeg", "Image"},
        data_shape=(63, 3035740),
        frequency=1000,
    )

    # Participants 1 and 6 never saw the validation images; the other 48 saw
    # the same 200, so any of them lists the test set.
    _TEST_REFERENCE_SUBJECT: tp.ClassVar[str] = "02"

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self.version = "v2"  # NEMAR layout, onsets in s, test split from events

    def _download(self, overwrite: bool = False) -> None:
        download.Nemar(
            study="on003825",
            dset_dir=self.path,
            version="1.0.2",
            exclude=["code/**", "derivatives/**"],
        ).download(overwrite=overwrite)

    def _events_file(self, subject: str) -> Path:
        sub = f"sub-{int(subject):02}"
        return (
            self.path
            / "download"
            / "on003825"
            / sub
            / "eeg"
            / f"{sub}_task-rsvp_events.tsv"
        )

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        for k in range(1, 51):
            folder = self._events_file(str(k)).parent
            if not folder.exists():
                raise RuntimeError(f"Folder does not exist: {folder}")
            yield dict(subject=str(k))

    def _format_img_path(self, img_path: str) -> str:
        out = self.path / "download" / "on003825" / img_path
        if not out.exists():
            raise RuntimeError(f"Output path does not exist: {out}")
        return str(out)

    def _get_test_info(self) -> tuple[set[str], set[str]]:
        ref = pd.read_csv(
            self._events_file(self._TEST_REFERENCE_SUBJECT),
            sep="\t",
            usecols=["stim", "isteststim"],
        )
        test_stims = set(ref.stim[ref.isteststim == 1])
        return test_stims, {Path(stim).parent.name for stim in test_stims}

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        events_file = self._events_file(timeline["subject"])
        events = pd.read_csv(events_file, sep="\t")

        events["filepath"] = events.stim.apply(self._format_img_path)
        events["category"] = events.filepath.apply(lambda x: x.split("/")[-2])
        events["type"] = "Image"
        events["start"] = events.onset  # on003825 >= 1.0.1 gives seconds

        events["caption"] = events.filepath.apply(
            lambda x: " ".join(Path(x).stem.split("_")[:-1])
        ).apply(lambda s: re.sub(r"\d", "", s))

        # Identify test images (called "validation images" in the paper)
        test_stims, test_categories = self._get_test_info()
        events["split"] = "train"
        events.loc[events.stim.isin(test_stims), "split"] = "test"
        events["is_test_category"] = events.category.isin(test_categories)
        events["stem"] = events.filepath.apply(lambda x: Path(x).stem)

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
        ]
        events = events[cols]

        # Add raw Eeg event
        raw_fname = str(events_file).replace("_events.tsv", "_eeg.vhdr")
        # NOTE 1: Data was referenced online to Cz, which has not been recorded
        # NOTE 2: Subjects 1-48 have 63 channels; subject 49-50 have 127
        eeg = {"filepath": raw_fname, "type": "Eeg", "start": 0}
        events = pd.concat([pd.DataFrame([eeg]), events])

        return events


class Grootswagers2022HumanSample(Grootswagers2022Human):
    """Sample version of Grootswagers2022Human with selective NEMAR download.

    Keeps one EEG timeline (sub-01) and the THINGS images (~6 GB in total) to
    enable fast loading checks without downloading the full EEG dataset.
    """

    description: tp.ClassVar[str] = (
        "Sample Grootswagers2022: NEMAR EEG + image events for sub-01."
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

    def _download(self, overwrite: bool = False) -> None:
        ref = self._TEST_REFERENCE_SUBJECT
        download.Nemar(
            study="on003825",
            dset_dir=self.path,
            version="1.0.2",
            include=[
                f"sub-{self._SAMPLE_SUBJECT}/eeg/**",
                f"sub-{ref}/eeg/sub-{ref}_task-rsvp_events.tsv",  # test split
                "stimuli/**",
            ],
        ).download(overwrite=overwrite)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        events_file = self._events_file(self._SAMPLE_SUBJECT)
        if not events_file.exists():
            raise FileNotFoundError(
                f"Missing sample events file for {self.__class__.__name__}: {events_file}. "
                "Please run study.download() first."
            )
        yield dict(subject=str(int(self._SAMPLE_SUBJECT)))
