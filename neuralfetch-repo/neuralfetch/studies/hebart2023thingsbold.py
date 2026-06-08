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

import neuralset.utils as nutils
from neuralfetch import download, utils
from neuralset.events import study


class Hebart2023ThingsBold(study.Study):
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
            # Gate on the THINGS licence and fetch the shared image database before
            # the large Datalad download (used across multiple THINGS studies).
            utils.download_things_images(self.path)
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
        path_to_stimuli = (self.path / ".." / "THINGS-images").resolve(strict=False)
        test_cats = self._get_test_categories()
        ns_events_df = self._get_ns_img_events_df(
            bids_events_df, path_to_stimuli, self._get_fmri_frequency(), test_cats
        )

        # Add indicator column for events of test categories
        ns_events_df["stem"] = ns_events_df.filepath.apply(lambda x: Path(x).stem)
        ns_events_df["is_test_category"] = ns_events_df.category.isin(test_cats)
        # For consistency with other THINGS-derived studies, redefine categories and split
        ns_events_df["category"] = ns_events_df.stem.apply(
            lambda x: "_".join(x.split("_")[:-1])
        )
        ns_events_df["split"] = ns_events_df.hebart2023_paper_split

        # Add shared filepath
        shared_things_path = (self.path / ".." / "THINGS-images").resolve(strict=False)
        if shared_things_path.exists():
            base = str(shared_things_path)
            ns_events_df["shared_filepath"] = (
                base + "/" + ns_events_df.category + "/" + ns_events_df.stem + ".jpg"
            )
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
