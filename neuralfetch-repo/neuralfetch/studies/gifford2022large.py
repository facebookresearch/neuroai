# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import typing as tp
import zipfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd

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
        - NEMAR re-host (nm000232): the original author-format ``.npy`` recordings
          are shipped inside per-subject zip archives (``download/sub-XX.zip``),
          each containing ``sub-XX/ses-YY/raw_eeg_{training,test}.npy``. These are
          the same custom dicts read by :meth:`_create_raw_from_npy` (63 EEG + 1
          ``stim`` channel, 1000 Hz). No Figshare npy->fif conversion is needed.
        - Image annotations are optional: if ``download/image_metadata.npy`` is
          present the stim codes are mapped to THINGS concept/basename filenames,
          otherwise a generic ``image_<code>`` label is used.

    Download Requirements:
        - Data hosted on NEMAR (dataset ID: nm000232), re-host of the Figshare
          release (18470912). Per-subject zips are extracted on first read.
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
        # NEMAR re-host currently provides sub-01 only (4 sessions x 2 splits = 8
        # timelines). iter_timelines discovers whatever subjects are on disk; keep
        # this guard in sync with the available data (was 80 for the full release).
        num_timelines=8,
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

    def _ensure_extracted(self) -> None:
        """Extract any ``download/sub-*.zip`` NEMAR archive not yet unpacked.

        The NEMAR re-host ships per-subject zips holding ``sub-XX/ses-YY/
        raw_eeg_{training,test}.npy``. Extraction is idempotent: a subject whose
        npy files are already present is skipped.
        """
        folder = self.path / "download"
        if not folder.exists():
            return
        for zp in sorted(folder.glob("sub-*.zip")):
            sub_dir = folder / zp.stem  # e.g. sub-01
            if sub_dir.exists() and any(sub_dir.glob("ses-*/raw_eeg_*.npy")):
                continue
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(folder)

    def _download(self) -> None:
        # Repointed to NEMAR nm000232 (THINGS-EEG2): pre-processed BIDS re-host,
        # avoids the OOM-prone Figshare npy->fif conversion + OSF/THINGS downloads.
        import nemar
        nemar.download(
            dataset="nm000232",
            target_dir=self.path / "download",
            downloader="python",
        )
        self._ensure_extracted()

    def _get_fname(self, timeline: dict[str, tp.Any]) -> Path:
        tl = timeline
        folder = self.path / "download"
        folder = folder / f"sub-{int(tl['subject']):02}" / f"ses-{int(tl['session']):02}"
        names = {"train": "raw_eeg_training.npy", "test": "raw_eeg_test.npy"}
        return folder / names[timeline["split"]]

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Discover recordings from the extracted NEMAR npy layout on disk."""
        self._ensure_extracted()
        folder = self.path / "download"
        split_of = {"training": "train", "test": "test"}
        for npy in sorted(folder.glob("sub-*/ses-*/raw_eeg_*.npy")):
            m = re.fullmatch(r"raw_eeg_(training|test)\.npy", npy.name)
            if m is None:
                continue
            sub_dir = npy.parent.parent.name  # sub-XX
            ses_dir = npy.parent.name  # ses-YY
            sub_m = re.fullmatch(r"sub-(\d+)", sub_dir)
            ses_m = re.fullmatch(r"ses-(\d+)", ses_dir)
            if sub_m is None or ses_m is None:
                continue
            yield dict(
                subject=str(int(sub_m.group(1))),
                session=int(ses_m.group(1)),
                split=split_of[m.group(1)],
            )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        split = timeline["split"]

        # Extract annotations from stim channel of the author-format npy recording.
        raw_fname = self._get_fname(timeline)
        raw = self._create_raw_from_npy(raw_fname)
        mne_events = mne.find_events(raw, stim_channel="STI101")

        # Optional image metadata to map stim codes -> THINGS image filenames.
        # The NEMAR re-host ships only raw EEG npy, so fall back to generic
        # per-code labels when ``image_metadata.npy`` is absent.
        fp = self.path / "download" / "image_metadata.npy"
        event_desc: dict[int, str]
        if fp.exists():
            image_metadata = np.load(fp, allow_pickle=True).item()
            concept_key = str(split) + "_img_concepts"
            files_key = str(split) + "_img_files"
            event_desc = {
                i + 1: concept + "/" + basename
                for i, (concept, basename) in enumerate(
                    zip(image_metadata[concept_key], image_metadata[files_key])
                )
            }
        else:
            codes = np.unique(mne_events[:, 2]) if mne_events.size else np.array([])
            event_desc = {int(c): f"image_{int(c):05d}/image_{int(c):05d}" for c in codes}

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
