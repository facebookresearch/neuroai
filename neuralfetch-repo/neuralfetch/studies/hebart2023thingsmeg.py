# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
import shutil
import typing as tp
import warnings
from functools import lru_cache
from itertools import product
from pathlib import Path

import mne
import pandas as pd

from neuralfetch import download, utils
from neuralset.events import study

logger = logging.getLogger(__name__)


def _read_attributes_csv(path: str | Path, subject_id: str) -> pd.DataFrame:
    filename = f"sample_attributes_P{subject_id}.csv"
    csv = Path(path) / "download" / "sourcedata" / filename
    subject_events = pd.read_csv(csv, sep=",")
    return subject_events


def _load_events(
    path: str | Path, subject_id: str, session_id: int, run_id: int
) -> pd.DataFrame:
    subj_events = _read_attributes_csv(path, subject_id)

    # Add category and is_test_category information
    subj_events["stem"] = subj_events.image_path.apply(lambda x: Path(x).stem)
    subj_events["category"] = subj_events.stem.apply(
        lambda x: "_".join(x.split("_")[:-1])
    )
    subj_events["is_test_category"] = subj_events.category.isin(
        subj_events.loc[subj_events.trial_type == "test", "category"]
    )

    sel_session = subj_events.session_nr == session_id
    sel_run = subj_events.run_nr == run_id
    columns = ["trial_type", "image_on", "image_off", "image_path", "image_nr"]
    columns += ["things_image_nr", "stem", "category", "is_test_category"]
    events = subj_events.loc[sel_session & sel_run, columns]
    return events


@lru_cache
def _get_bids(subject: str, session: int, run: int, path: Path) -> mne.io.Raw:
    """mne bids is super slow, so let's cache it"""
    from mne_bids import BIDSPath  # pylint: disable=unused-import

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BIDSPath(
            subject=f"BIGMEG{subject}",
            session=f"{session:02d}",
            task="main",
            run=f"{run:02d}",
            datatype="meg",
            suffix="meg",
            extension=".ds",
            root=path / "download",
        )


class Hebart2023ThingsMeg(study.Study):
    """THINGS-MEG1: MEG responses to object images from THINGS database.

    This study is part of the THINGS initiative, providing MEG recordings from
    participants viewing object images from the THINGS database in rapid succession.

    Experimental Design:
        - 4 participants (BIGMEG subjects 1-4)
        - 12 scanning sessions per subject
        - 10 runs per session
        - Rapid serial visual presentation (RSVP) paradigm
        - Images from THINGS database covering 1,854 object concepts
        - MEG sampling rate: 1200 Hz

    Data Format:
        - BIDS-compliant dataset structure
        - CTF MEG data format
        - Includes event timing and image annotations
        - Multiple train/test split strategies
        - Preprocessed images available in preprocessed/images/
        - Shared THINGS image database expected in ../THINGS-images/

    Download Requirements:
        - OpenNeuro dataset: ds004212
        - Approximately ~several GB dataset size
        - THINGS-images must be manually downloaded from https://osf.io/jum2f/
            * Download images_THINGS.zip (~12GB)
            * Read license terms in password_images.txt
            * Extract with password to agree to terms
            * Place at ../THINGS-images/ (sibling directory to study folder)
        - Note: Preprocessed images extraction requires password (currently disabled)
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("THINGS-MEG1",)

    bibtex: tp.ClassVar[str] = """
    @article{hebart2023things,
        article_type = {journal},
        title = {THINGS-data, a multimodal collection of large-scale datasets for investigating object representations in human brain and behavior},
        author = {Hebart, Martin N and Contier, Oliver and Teichmann, Lina and Rockter, Adam H and Zheng, Charles Y and Kidder, Alexis and Corriveau, Anna and Vaziri-Pashkam, Maryam and Baker, Chris I},
        editor = {Barense, Morgan and de Lange, Floris P and Konkle, Talia and Glerean, Enrico},
        volume = 12,
        year = 2023,
        month = {feb},
        pub_date = {2023-02-27},
        pages = {e82580},
        citation = {eLife 2023;12:e82580},
        doi = {10.7554/eLife.82580},
        url = {https://doi.org/10.7554/eLife.82580},
        journal = {eLife},
        issn = {2050-084X},
        publisher = {eLife Sciences Publications, Ltd},
    }
    """
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds004212/versions/2.0.0"
    licence: tp.ClassVar[str] = "CC-BY C0"
    description: tp.ClassVar[str] = (
        "THINGS-data: MEG recordings from 4 participants viewing 1,854 diverse "
        "object concepts."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("pyunpack", "boto3", "osfclient>=0.0.5")

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=480,
        num_subjects=4,
        num_events_in_query=207,
        event_types_in_query={"Meg", "Image"},
        data_shape=(300, 417600),
        frequency=1200.0,
    )

    def _download(self) -> None:
        # Check for / Download THINGS-images database (used across multiple THINGS studies)
        utils.download_things_images(self.path)
        download.Openneuro("ds004212", self.path).download()  # type: ignore

        # Decompress
        parts = "A-C", "D-K", "L-Q", "R-S", "T-Z"
        prep_dir = self.path / "preprocessed" / "images"
        prep_dir.mkdir(parents=True, exist_ok=True)
        for part in parts:
            success = prep_dir / f"{part}_success.txt"
            if success.exists():
                continue
            from pyunpack import Archive  # noqa

            password = os.environ.get("NEURALHUB_THINGS_PASSWORD")
            if not password:
                raise RuntimeError(
                    "THINGS-images requires accepting a licence agreement.\n"
                    "Please export NEURALHUB_THINGS_PASSWORD=<pwd> where the password can be found "
                    "in password_images.txt at https://osf.io/jum2f/files/52wrx"
                )

            img_dir = (
                self.path / "download" / "stimuli" / "download" / "THINGS" / "Images"
            )
            zip_file = img_dir / f"object_images_{part}.zip"
            print(f"Unzipping {zip_file.name}...")
            Archive(str(zip_file), password=password).extractall(str(prep_dir))

            # fix bad unzipping: the L-Q archive extracts into a nested subdirectory
            if part == "L-Q":
                bad_dir = prep_dir / "object_images_L-Q"
                if bad_dir.exists():
                    for folder in bad_dir.iterdir():
                        if folder.is_file():
                            continue

                        target_dir = prep_dir / folder.name
                        target_dir.mkdir(exist_ok=True)
                        for file in folder.iterdir():
                            print(file)
                            shutil.move(str(file), str(target_dir / file.name))

            with open(success, "w") as f:
                f.write("done")

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        sub_ses_run = range(1, 5), range(1, 13), range(1, 11)
        for subject_id, session_id, run_id in product(*sub_ses_run):
            yield dict(subject=str(subject_id), session=session_id, run=run_id)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        from neuralset import utils as nutils

        tl = timeline
        raw_fname = _get_bids(tl["subject"], tl["session"], tl["run"], self.path)
        event_fname = str(raw_fname).replace("_meg.ds", "_events.tsv")
        raw_events = pd.read_csv(event_fname, sep="\t")

        events = _load_events(self.path, tl["subject"], tl["session"], tl["run"])

        if len(events) != len(raw_events):
            msg = f"Event count mismatch: {len(events)} vs {len(raw_events)}"
            raise RuntimeError(msg)

        # The source CSV (sample_attributes) uses a different image numbering scheme
        # than the BIDS events.tsv `value` column for a small subset of catch/repeated
        # trials. Up to 20 mismatches per run is a known data quirk in ds004212.
        _MAX_IMAGE_NR_MISMATCHES = 20
        mismatch = sum(events.things_image_nr.values != raw_events.value)  # type: ignore
        if mismatch > 0:
            logger.warning(
                "things_image_nr vs events.tsv value: %d mismatches (max allowed: %d)",
                mismatch,
                _MAX_IMAGE_NR_MISMATCHES,
            )
        if mismatch > _MAX_IMAGE_NR_MISMATCHES:
            raise RuntimeError(f"Too many things_image_nr mismatches: {mismatch}")

        # Use raw trigger times from the UPPT001 stim channel as ground truth for
        # onset alignment. The events.tsv image_on column has a ~150ms software-measured
        # delay relative to the hardware triggers, so we prefer the raw trigger times.
        # TODO: check why 150 ms offset between raw events and event.tsv?
        with nutils.ignore_all():
            raw = mne.io.read_raw(str(raw_fname.fpath))
        raw_events = mne.find_events(raw, stim_channel="UPPT001")
        if len(raw_events) != (len(events) + 1):
            msg = f"Raw event count mismatch: {len(raw_events)} vs {len(events) + 1}"
            raise RuntimeError(msg)
        if set(raw_events[:-1, 2]) != {64}:
            raise RuntimeError(f"Unexpected event values: {set(raw_events[:-1, 2])}")
        if raw_events[-1, 2] != 32:
            raise RuntimeError(f"Unexpected last event value: {raw_events[-1, 2]}")

        starts = raw_events[:, 0] / raw.info["sfreq"]
        events["start"] = starts[:-1]
        events["duration"] = events.image_off - events.image_on
        if not all(events.duration > 0.450):
            raise RuntimeError("Some durations are too short")
        if not all(events.duration < 0.550):
            raise RuntimeError("Some durations are too long")

        # specify image path
        img_dir = self.path / "prepare"

        def format_path(img_path):
            if not img_path.endswith(".jpg"):
                raise RuntimeError(f"Expected .jpg image path: {img_path}")
            filename = img_path.split("/")[-1]
            category = "_".join(filename.split("_")[:-1])
            return img_dir / category / filename

        events["filepath"] = events.image_path.apply(format_path)

        test = events.trial_type == "test"
        events["split"] = "train"
        events["valid"] = True
        events.loc[test, "split"] = "test"
        # Filter out catch trials (attention-check images to ensure subject is focused)
        valid = events.image_path.apply(lambda f: not f.startswith("images_catch_meg"))
        events.loc[~valid, "split"] = None
        events.loc[~valid, "valid"] = False
        check = events.filepath.apply(lambda x: x.exists())
        if check.loc[valid].mean() <= 0.95:
            raise RuntimeError("Too many missing image files")
        events.loc[~check, "split"] = None
        events.loc[~check, "valid"] = False

        # Add shared filepaths
        shared_things_path = (self.path / ".." / "THINGS-images").resolve(strict=False)
        if shared_things_path.exists():
            fp = f"{shared_things_path}/{events.category}/{events.stem}.jpg"
            events["shared_filepath"] = fp

        events["type"] = "Image"
        events["filepath"] = events.filepath.apply(str)
        events["caption"] = events.category.str.replace("_", " ").apply(
            lambda s: re.sub(r"\d", "", s)
        )
        invalid = int((~valid).sum())
        if invalid:
            msg = "Removing %s invalid (catch images) events from hebart2023thingsmeg"
            logger.warning(msg, invalid)
            events = events.loc[valid, :]
        meg = {"type": "Meg", "filepath": str(raw_fname.fpath), "start": 0}
        events = pd.concat([pd.DataFrame([meg]), events])
        return events
