# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import typing as tp
from itertools import product
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from neuralfetch import download
from neuralfetch.utils import download_things_images
from neuralset.events import study


class Gifford2022Large(study.Study):
    url: tp.ClassVar[str] = "https://osf.io/3jk45/"
    """THINGS-EEG2: EEG responses to object images from THINGS database.

    This study provides EEG recordings from 10 participants viewing object images
    from the THINGS database across training and test sessions. Part of the THINGS
    initiative, it offers a large-scale dataset for modeling human visual object
    recognition.

    Experimental Design:
        - EEG recordings (63-channel, 1000 Hz, EASYCAP)
        - 10 participants
        - 4 recording sessions per participant, each with training and test splits
        - Paradigm: Rapid serial visual presentation (RSVP)
            * Training set and test set images from THINGS database
            * 80 total timelines (10 participants x 4 sessions x 2 splits)

    Data Format:
        - Raw EEG data stored as .npy files, converted to MNE .fif format
        - Image annotations and category labels provided
        - Shared THINGS image database expected in ../THINGS-images/

    Download Requirements:
        - Data hosted on Figshare (dataset ID: 18470912)
        - Study-specific images downloaded from OSF (y63gw)
        - THINGS-images database (shared across THINGS studies): checked for in
          ../THINGS-images/ and downloaded automatically if missing
        - pyunpack required for zip extraction
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("THINGS-EEG2",)

    licence: tp.ClassVar[str] = "CC-BY-4.0"
    bibtex: tp.ClassVar[str] = """
    @article{gifford2022large,
        url = {https://pmc.ncbi.nlm.nih.gov/articles/PMC9771828/}
        title={A Large and Rich {{EEG}} Dataset for Modeling Human Visual Object Recognition},
        author={Gifford, Alessandro T. and Dwivedi, Kshitij and Roig, Gemma and Cichy, Radoslaw M.},
        year=2022,
        month=dec,
        journal={Neuroimage},
        volume={264},
        pages={119754},
        issn={1053-8119},
        doi={10.1016/j.neuroimage.2022.119754},
        pmcid={PMC9771828},
        pmid={36400378},
        url={https://pmc.ncbi.nlm.nih.gov/articles/PMC9771828/}
    }

    @misc{gifford2022largea,
        url = {https://plus.figshare.com/articles/dataset/A_large_and_rich_EEG_dataset_for_modeling_human_visual_object_recognition/18470912}
        title={A Large and Rich {{EEG}} Dataset for Modeling Human Visual Object Recognition},
        author={Gifford, Alessandro T.},
        year=2022,
        month=mar,
        publisher={Figshare+},
        doi={10.25452/figshare.plus.18470912.v4},
        langid={english},
        url={https://plus.figshare.com/articles/dataset/A_large_and_rich_EEG_dataset_for_modeling_human_visual_object_recognition/18470912}
    }

    @article{gifford2021large,
        doi={10.17605/OSF.IO/3JK45},
        url={https://osf.io/3jk45/},
        author={Gifford,  Alessandro Thomas},
        keywords={Computational neuroscience,  DNNs,  EEG,  Encoding models,  Human vision,  THINGS database},
        title={A large and rich EEG dataset for modeling human visual object recognition},
        publisher={OSF},
        year={2021},
        copyright={Creative Commons Attribution 4.0 International}
    }
    """
    description: tp.ClassVar[str] = (
        "EEG recordings from 10 participants watching still images."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("pyunpack>=0.3",)

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=80,
        num_subjects=10,
        num_events_in_query=16711,
        event_types_in_query={"Eeg", "Image"},
        data_shape=(63, 5450560),
        frequency=1000.0,
    )

    @staticmethod
    def _create_raw_from_npy(fname: str | Path) -> mne.io.RawArray:
        """Create mne Raw object from custom npy format used by the authors."""
        out = np.load(fname, allow_pickle=True).item()

        ch_names = out["ch_names"]
        ch_names[ch_names.index("stim")] = (
            "STI101"  # Use different channel name from channel type
        )
        info = mne.create_info(ch_names, sfreq=out["sfreq"], ch_types=out["ch_types"])
        with info._unlock():
            info["lowpass"] = out["lowpass"]
            info["highpass"] = out["highpass"]
        info.set_montage("standard_1020")

        raw = mne.io.RawArray(out["raw_eeg_data"], info)
        return raw

    def _download(self) -> None:
        dataset_id = "18470912"
        figshare = download.Figshare(study=dataset_id, dset_dir=self.path)
        figshare.download()

        # Unzip EEG data
        from pyunpack import Archive

        dl_dir = self.path / "download"
        for zip_file in dl_dir.glob("sub-*.zip"):
            with download.success_writer(zip_file) as already_done:
                if not already_done:
                    Archive(str(zip_file)).extractall(str(dl_dir))

        # Load .npy and save as mne.io.Raw to enable memmaping
        for eeg_fname in tqdm(dl_dir.glob("**/**/raw_eeg_*.npy")):
            with download.success_writer(eeg_fname) as already_done:
                if not already_done:
                    raw = self._create_raw_from_npy(eeg_fname)
                    raw.save(str(eeg_fname).replace(".npy", "_raw.fif"), overwrite=True)

        # Download images from OSF
        download.Osf(study="y63gw", dset_dir=self.path, folder="download").download()

        # Check for / Download THINGS-images database (used across multiple THINGS studies)
        download_things_images(self.path)

        # Unzip images
        for image_fname in ["training_images.zip", "test_images.zip"]:
            zip_file = dl_dir / image_fname
            with download.success_writer(zip_file) as already_done:
                if not already_done:
                    Archive(str(zip_file)).extractall(str(dl_dir))

        # Clean up zip files
        for zip_file in dl_dir.glob("*.zip"):
            zip_file.unlink()

    def _get_fname(self, timeline: dict[str, tp.Any]) -> Path:
        tl = timeline
        folder = self.path / "download"
        folder = folder / f"sub-{int(tl['subject']):02}" / f"ses-{int(tl['session']):02}"
        names = {"train": "raw_eeg_training_raw.fif", "test": "raw_eeg_test_raw.fif"}
        return folder / names[timeline["split"]]

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        for subject, session, split in product(
            range(1, 11), range(1, 5), ["train", "test"]
        ):
            tl = dict(subject=str(subject), session=session, split=split)
            fname = self._get_fname(tl)
            if not fname.exists():
                raise RuntimeError(f"File {fname} does not exist.")
            yield tl

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        split = timeline["split"]

        # Load image metadata to get mapping from stim codes to image filenames
        fp = self.path / "download" / "image_metadata.npy"
        image_metadata = np.load(fp, allow_pickle=True).item()
        concept_key = str(split) + "_img_concepts"
        files_key = str(split) + "_img_files"
        event_desc = {
            i + 1: concept + "/" + basename
            for i, (concept, basename) in enumerate(
                zip(image_metadata[concept_key], image_metadata[files_key])
            )
        }

        # Extract annotations from stim channel
        raw_fname = self._get_fname(timeline)
        raw = mne.io.read_raw(raw_fname)

        mne_events = mne.find_events(raw, stim_channel="STI101")
        annot_from_events = mne.annotations_from_events(
            events=mne_events,
            event_desc=event_desc,
            sfreq=raw.info["sfreq"],
            orig_time=raw.info["meas_date"],
        )

        # Build events dataframe.
        # ``pd.DataFrame(annotations)`` produces dict-typed ``extras`` and
        # object-typed ``orig_time`` columns, which ``ValidatedParquet``
        # cannot serialize; drop them here at the source.
        events = pd.DataFrame(annot_from_events)
        events = events.drop(columns=["extras", "orig_time"], errors="ignore")
        events["description"] = events.description.apply(str)  # numpy.str_ -> str
        events["start"] = events.onset
        events["duration"] = 0.1
        events["type"] = "Image"
        image_folder = {"train": "training_images", "test": "test_images"}[split]
        events["filepath"] = (
            str(self.path / "download" / image_folder) + "/" + events.description
        )
        events["stem"] = events.description.apply(lambda x: Path(x).stem)
        events["category"] = events["stem"].apply(lambda x: "_".join(x.split("_")[:-1]))
        events["caption"] = events.category.str.replace("_", " ").apply(
            lambda s: re.sub(r"\d", "", s)
        )
        # For compatibility with other THINGS
        events["is_test_category"] = split == "test"

        # Add shared THINGS filepaths to images
        shared_things_path = (self.path / ".." / "THINGS-images").resolve(strict=False)
        if shared_things_path.exists():
            events["shared_filepath"] = (
                str(shared_things_path)
                + "/"
                + events.category
                + "/"
                + events.stem
                + ".jpg"
            )

        # add raw event from method
        eeg = {"filepath": str(raw_fname), "type": "Eeg", "start": 0}
        events = pd.concat([pd.DataFrame([eeg]), events])
        return events
