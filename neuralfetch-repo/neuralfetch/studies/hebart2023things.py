# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import typing as tp
import warnings
from functools import lru_cache
from itertools import product
from pathlib import Path

import mne
import nibabel
import pandas as pd

import neuralset.utils as nutils
from neuralfetch import download, utils
from neuralset.events import study

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers (MEG)
# ---------------------------------------------------------------------------


def _read_attributes_csv(path: str | Path, subject_id: str) -> pd.DataFrame:
    filename = f"sample_attributes_P{subject_id}.csv"
    csv = Path(path) / "download" / "sourcedata" / filename
    subject_events = pd.read_csv(csv, sep=",")
    return subject_events


def _load_meg_events(
    path: str | Path, subject_id: str, session_id: int, run_id: int
) -> pd.DataFrame:
    subj_events = _read_attributes_csv(path, subject_id)

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


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _Hebart2023Things(study.Study):
    """Private base class for THINGS-data studies (MEG and fMRI)."""

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

    def _things_images_path(self) -> Path:
        """Resolve the shared THINGS-images directory (sibling of study folder)."""
        return (self.path / ".." / "THINGS-images").resolve(strict=False)

    def _add_shared_filepath(
        self, df: pd.DataFrame, category_col: str = "category", stem_col: str = "stem"
    ) -> pd.DataFrame:
        """Add a ``shared_filepath`` column pointing into the shared THINGS-images DB."""
        things_path = self._things_images_path()
        if things_path.exists():
            base = str(things_path)
            df["shared_filepath"] = (
                base + "/" + df[category_col] + "/" + df[stem_col] + ".jpg"
            )
        return df

    def _download(self) -> None:
        """Download the shared THINGS-images database."""
        utils.download_things_images(self.path)


# ---------------------------------------------------------------------------
# MEG study
# ---------------------------------------------------------------------------


class Hebart2023ThingsMeg(_Hebart2023Things):
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
        - Image stimuli are sourced from the shared THINGS image database in
          ../THINGS-images/ (downloaded automatically; see Download Requirements)

    Download Requirements:
        - OpenNeuro dataset: ds004212
        - Approximately ~several GB dataset size
        - THINGS-images must be manually downloaded from https://osf.io/jum2f/
            * Download images_THINGS.zip (~12GB)
            * Read license terms in password_images.txt
            * Extract with password to agree to terms
            * Place at ../THINGS-images/ (sibling directory to study folder)
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("THINGS-MEG1",)

    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds004212/versions/2.0.0"
    licence: tp.ClassVar[str] = "CC-BY C0"
    description: tp.ClassVar[str] = (
        "THINGS-data: MEG recordings from 4 participants viewing 1,854 diverse "
        "object concepts."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("boto3", "osfclient>=0.0.5")

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=480,
        num_subjects=4,
        num_events_in_query=207,
        event_types_in_query={"Meg", "Image"},
        data_shape=(300, 417600),
        frequency=1200.0,
    )

    def _download(self) -> None:
        super()._download()  # Download shared THINGS-images database
        download.Openneuro("ds004212", self.path).download()  # type: ignore

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

        events = _load_meg_events(self.path, tl["subject"], tl["session"], tl["run"])

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

        # specify image path (shared THINGS-images database)
        img_dir = self._things_images_path()

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

        events = self._add_shared_filepath(events)

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


# ---------------------------------------------------------------------------
# fMRI study
# ---------------------------------------------------------------------------


class Hebart2023ThingsBold(_Hebart2023Things):
    """THINGS-fMRI1: 3T fMRI responses to a large-scale object image database.

    This study is part of the THINGS initiative, a multimodal collection
    investigating object representations during 3T fMRI scanning. The large
    number of unique object concepts and multiple split strategies make it
    valuable for both within-domain and zero-shot generalization studies.

    Experimental Design:
        - 3T fMRI recordings (TR = 1.5 seconds)
        - 3 participants
        - 12 scanning sessions per participant
        - Paradigm: passive viewing of object images
            * 10 runs per session (except sub-02/ses-03/run-07 is missing)
            * ~82 image presentations per run
            * Images from THINGS database cover 1,854 object concepts
            * Catch trials included to ensure attention

    Data Format:
        - BIDS-compliant dataset structure (OpenNeuro ds004192)
        - Preprocessed in MNI152NLin2009aSym space
        - 359 total timelines (3 participants x 12 sessions x ~10 runs)
        - Event types: Fmri, Image
        - Multiple train/test split strategies:
            * Original paper split (test trials vs. train trials)
            * Zero-shot split (held-out object categories)
            * Large split (includes held-out categories in test set)

    Download Requirements:
        - datalad (for dataset download from OpenNeuro ds004192)
        - THINGS-images database (shared across THINGS studies): checked for in
          ../THINGS-images/ and downloaded automatically if missing
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("THINGS-fMRI1",)

    bibtex: tp.ClassVar[str] = """
    @article{hebart2023things,
        title={THINGS-data, a multimodal collection of large-scale datasets for
        investigating object representations in human brain and behavior},
        author={Hebart, Martin N and Contier, Oliver and Teichmann, Lina and Rockter,
        Adam H and Zheng, Charles Y and Kidder, Alexis and Corriveau, Anna
        and Vaziri-Pashkam, Maryam and Baker, Chris I},
        journal={Elife},
        volume={12},
        pages={e82580},
        year={2023},
        publisher={eLife Sciences Publications Limited},
        doi={10.7554/eLife.82580},
        url={https://elifesciences.org/articles/82580}
    }

    @misc{hebart2022things,
        url = {https://openneuro.org/datasets/ds004192/versions/1.0.5}
        author={Martin N. Hebart AND Oliver Contier AND Lina Teichmann AND Adam H. Rockter AND Charles Zheng AND Alexis Kidder AND Anna Corriveau AND Maryam Vaziri-Pashkam AND Chris I. Baker},
        title={"THINGS-fMRI"},
        year={2022},
        doi={10.18112/openneuro.ds004192.v1.0.5},
        publisher={OpenNeuro},
        url={https://openneuro.org/datasets/ds004192/versions/1.0.5}
    }
    """
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "THINGS-data: 3T BOLD fMRI from 3 participants viewing 1,854 diverse "
        "object concepts."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("datalad>=0.19.5",)

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=359,
        num_subjects=3,
        num_events_in_query=83,
        event_types_in_query={"Fmri", "Image"},
        data_shape=(77, 94, 80, 284),
        frequency=0.667,
        fmri_spaces=("custom",),
    )

    SUBJECTS: tp.ClassVar[tuple[int, ...]] = (1, 2, 3)
    SESSIONS_PER_SUBJECT: tp.ClassVar[int] = 12
    RUNS_PER_SESSION: tp.ClassVar[int] = 10
    BIDS_FOLDER: tp.ClassVar[str] = "download/ds004192"
    DERIVATIVES_FOLDER: tp.ClassVar[str] = "derivatives"
    BOLD_SPACE: tp.ClassVar[str] = "MNI152NLin2009aSym"
    TASK: tp.ClassVar[str] = "things"
    SESSION_SUFFIX: tp.ClassVar[str] = "things"
    TR_FMRI_S: tp.ClassVar[float] = 1.5

    def _download(self) -> None:
        with download.success_writer(self.path / "download_all") as already_done:
            if already_done:
                return
            super()._download()  # Download shared THINGS-images database
            download.Datalad(
                study="hebart2023bold",
                dset_dir=self.path,
                repo_url="https://github.com/OpenNeuroDatasets/ds004192.git",
                threads=4,
                folders=[download.Wildcard(folder="sub-*")],
            ).download()
            self._write_test_categories()

    def _write_test_categories(self) -> None:
        event_dfs = []
        for tl in self.iter_timelines():
            bids_events_df_fp = nutils.get_bids_filepath(
                root_path=self.path / self.BIDS_FOLDER,
                filetype="events",
                data_type="Fmri",
                ses_suffix=self.SESSION_SUFFIX,
                **tl,
            )
            event_dfs.append(nutils.read_bids_events(bids_events_df_fp))
        event_df = pd.concat(event_dfs, axis=0)
        test_trials = event_df[event_df.trial_type == "test"]
        test_categories = test_trials.file_path.map(self._get_category).unique()
        with open(self.path / "test_categories.txt", mode="w", encoding="utf8") as f:
            f.writelines([f"{category}\n" for category in test_categories])

    def _get_test_categories(self) -> tp.Set[str]:
        with (self.path / "test_categories.txt").open(encoding="utf8") as f:
            return set(line.strip() for line in f)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for subject in self.SUBJECTS:
            for session in range(1, self.SESSIONS_PER_SUBJECT + 1):
                for run in range(1, self.RUNS_PER_SESSION + 1):
                    if subject == 2 and session == 3 and run == 7:
                        continue  # Missing data, see hebart2023bold/download/ds004192/sub-02/ses-things03/func/
                    yield dict(subject=subject, session=session, task=self.TASK, run=run)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        fmri = {
            "filepath": info,
            "type": "Fmri",
            "start": 0.0,
            "frequency": self._get_fmri_frequency(),
            "duration": self._get_bold_image(timeline).shape[-1] * self.TR_FMRI_S,
        }
        bids_events_df_fp = nutils.get_bids_filepath(
            root_path=self.path / self.BIDS_FOLDER,
            filetype="events",
            data_type="Fmri",
            ses_suffix=self.SESSION_SUFFIX,
            **timeline,
        )
        bids_events_df = nutils.read_bids_events(bids_events_df_fp)
        path_to_stimuli = self._things_images_path()
        test_cats = self._get_test_categories()
        ns_events_df = self._get_ns_img_events_df(
            bids_events_df, path_to_stimuli, self._get_fmri_frequency(), test_cats
        )

        ns_events_df["stem"] = ns_events_df.filepath.apply(lambda x: Path(x).stem)
        ns_events_df["is_test_category"] = ns_events_df.category.isin(test_cats)
        # For consistency with other THINGS-derived studies, redefine categories and split
        ns_events_df["category"] = ns_events_df.stem.apply(
            lambda x: "_".join(x.split("_")[:-1])
        )
        ns_events_df["split"] = ns_events_df.hebart2023_paper_split

        ns_events_df = self._add_shared_filepath(ns_events_df)

        return pd.concat([pd.DataFrame([fmri]), ns_events_df], axis=0)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> nibabel.Nifti1Image:
        return nutils.get_masked_bold_image(
            self._get_bold_image(timeline), self._get_bold_mask(timeline)
        )

    @classmethod
    def _get_category(cls, file_path: str) -> str:
        return re.sub(r"\d", "", " ".join(Path(file_path).stem.split("_")[:-1]))

    def _get_ns_img_events_df(
        self,
        bids_events_df: pd.DataFrame,
        stimuli_path: str | Path,
        frequency: float,
        test_cats: tp.Set[str],
    ) -> pd.DataFrame:
        # Leave out 'catch' trials (used for making sure subject is focused)
        bids_events_df = bids_events_df[bids_events_df.trial_type != "catch"]
        bids_events = bids_events_df.to_dict("records")
        ns_events = []
        for bids_event in bids_events:
            parent = "_".join(Path(bids_event["file_path"]).stem.split("_")[:-1])
            fp = Path(stimuli_path) / parent / Path(bids_event["file_path"]).name
            split = "test" if bids_event["trial_type"] == "test" else "train"
            ns_event = dict(
                type="Image",
                start=bids_event["onset"],
                duration=bids_event["duration"],
                frequency=frequency,
                filepath=str(fp),
                hebart2023_paper_split=split,
                category=self._get_category(bids_event["file_path"]),
            )
            ns_events.append(ns_event)

        ns_events_df = pd.DataFrame(ns_events)

        # Add zero-shot split, that is:
        # Remove images from train-set whose category is in test
        ns_events_df["zero_shot_split"] = ns_events_df.hebart2023_paper_split
        sel = (ns_events_df.hebart2023_paper_split == "train") & (
            ns_events_df.category.isin(test_cats)
        )
        ns_events_df.loc[sel, "zero_shot_split"] = "trash"

        # Add large split, that is:
        # from zero-shot split, add 'trash' images to 'test'
        ns_events_df["large_split"] = ns_events_df.zero_shot_split
        sel = ns_events_df.zero_shot_split == "trash"
        ns_events_df.loc[sel, "large_split"] = "test"
        return ns_events_df

    def _get_bold_mask(self, timeline: dict[str, tp.Any]):
        fp = nutils.get_bids_filepath(
            root_path=self.path / self.DERIVATIVES_FOLDER,
            filetype="bold_mask",
            data_type="Fmri",
            space=self.BOLD_SPACE,
            ses_suffix=self.SESSION_SUFFIX,
            **timeline,
        )
        return nibabel.load(fp, mmap=True)

    def _get_bold_image(self, timeline: dict[str, tp.Any]):
        fp = nutils.get_bids_filepath(
            root_path=self.path / self.DERIVATIVES_FOLDER,
            filetype="bold",
            data_type="Fmri",
            space=self.BOLD_SPACE,
            ses_suffix=self.SESSION_SUFFIX,
            **timeline,
        )
        return nibabel.load(fp, mmap=True)

    def _get_fmri_frequency(self) -> float:
        return 1.0 / self.TR_FMRI_S
