# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Natural Scenes Dataset (NSD): 7T fMRI responses to natural images.

This study provides high-resolution 7T BOLD fMRI data from 8 participants viewing
natural images from the Microsoft COCO dataset, using NSD's own preprocessed
timeseries (func1pt8mm resolution, nsdgeneral ROI).

Experimental Design:
    - 8 participants
    - 30-40 scanning sessions per subject (varies by subject)
    - 12 runs per session (14 runs in later sessions; runs 1 and 14 are resting state)
    - ~750 image trials per session
    - Total of 73,000 unique natural images from COCO dataset
    - TR = 4/3 seconds (func1pt8mm)
    - Each trial consists of image presentation (3s) followed by blank screen

Data Format:
    - NSD preprocessed timeseries (func1pt8mm)
    - nsdgeneral ROI applied by default
    - Includes image stimuli and experimental design metadata

Download Requirements:
    - AWS CLI must be installed
    - Large dataset size (~several TB)
    - Data stored in AWS S3: s3://natural-scenes-dataset/
"""

import functools
import logging
import os
import subprocess
import sys
import tarfile
import typing as tp
import webbrowser
from pathlib import Path
from urllib.parse import urlencode

import nibabel
import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import trange

from neuralfetch.download import success_writer
from neuralset.events import study

logger = logging.getLogger(__name__)


def get_nsd_tcs_marker() -> Path:
    """Path to the NSD T&C acceptance marker in the user's config directory."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_dir = Path(xdg) if xdg else Path.home() / ".config"
    return config_dir / "neuralfetch" / ".nsd_tcs_accepted"


# Helper to load 'nsd_expdesign.mat' file Copy-pasted,
# from https://github.com/ozcelikfu/brain-diffuser/blob/main/data/prepare_nsddata.py
# Commit 1c07200
def _loadmat(filename):
    """
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_keys(d):
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    import scipy.io as spio

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


_S3_BUCKET = "s3://natural-scenes-dataset"


@functools.lru_cache(maxsize=None)
def _get_roi_mask(path: Path) -> np.ndarray:
    roi_img: nibabel.Nifti1Image = nibabel.load(path)  # type: ignore[assignment]
    return roi_img.get_fdata() > 0


class Allen2022Massive(study.Study):
    """Natural Scenes Dataset (NSD): 7T fMRI responses to natural images.

    High-resolution 7T BOLD fMRI data from 8 participants viewing natural images
    from the Microsoft COCO dataset. Uses NSD's own preprocessed timeseries
    (func1pt8mm) with nsdgeneral ROI masking applied.

    Experimental Design:
        - 7T fMRI recordings (TR = 4/3 s, func1pt8mm resolution)
        - 8 participants
        - 30-40 sessions per participant, 12 runs per session
            * Sessions 21+ have 14 runs (runs 1 and 14 are resting state, skipped)
        - Paradigm: passive viewing of 73,000 natural images from COCO dataset
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("NSD",)
    dataset_name: tp.ClassVar[str] = "Natural Scenes Dataset"
    bibtex: tp.ClassVar[str] = """
    @article{allen2022massive,
        title={A massive 7T fMRI dataset to bridge cognitive neuroscience and
        artificial intelligence},
        author={Allen, Emily J and St-Yves, Ghislain and Wu, Yihan and Breedlove, Jesse L
        and Prince, Jacob S and Dowdle, Logan T and Nau, Matthias and Caron, Brad
        and Pestilli, Franco and Charest, Ian and others},
        journal={Nature neuroscience},
        volume={25},
        number={1},
        pages={116--126},
        year={2022},
        publisher={Nature Publishing Group US New York},
        doi={doi:10.1038/s41593-021-00962-x},
        url={https://www.nature.com/articles/s41593-021-00962-x}
    }
    """
    licence: tp.ClassVar[str] = (
        "custom https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions"
    )
    description: tp.ClassVar[str] = (
        "Natural Scenes Dataset: 7T BOLD fMRI from 8 participants viewing "
        "73,000 natural images from Microsoft COCO dataset"
    )

    requirements: tp.ClassVar[tuple[str, ...]] = (
        "scipy>=1.11.4",
        "h5py>=3.10.0",
        "requests>=2.31.0",
        "awscli",
    )

    SESSIONS_PER_SUBJECT: tp.ClassVar[dict[int, int]] = {
        1: 40,
        2: 40,
        3: 32,
        4: 30,
        5: 40,
        6: 32,
        7: 40,
        8: 30,
    }

    # Inclusive session range where 14-run sessions occur per subject.
    # Verified from S3 at s3://natural-scenes-dataset/nsddata_timeseries/ppdata/.
    # In these sessions runs 1 and 14 are resting state and are skipped.
    SESSIONS_WITH_14_RUNS: tp.ClassVar[dict[int, tuple[int, int]]] = {
        1: (21, 38),
        2: (21, 30),
        3: (21, 30),
        4: (21, 30),
        5: (21, 38),
        6: (21, 30),
        7: (21, 30),
        8: (21, 30),
    }

    N_STIMULI: tp.ClassVar[int] = 73000
    TASK: tp.ClassVar[str] = "nsdcore"
    TR_FMRI_S: tp.ClassVar[float] = 4 / 3  # func1pt8mm

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=3408,
        num_subjects=8,
        num_events_in_query=64,
        event_types_in_query={"Fmri", "Image"},
        data_shape=(15724, 225),
        frequency=0.75,
        fmri_spaces={"custom"},
    )

    # NSD Data Access Agreement Google Form
    NSD_FORM_VIEW_URL: tp.ClassVar[str] = (
        "https://docs.google.com/forms/d/e/"
        "1FAIpQLSduTPeZo54uEMKD-ihXmRhx0hBDdLHNsVyeo_kCb8qbyAkXuQ/viewform"
    )
    NSD_TCS_TEXT: tp.ClassVar[str] = """\
=== NSD Terms and Conditions ===
(https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions)

Before you download the NSD data, please read the data sharing and usage \
agreement below. You must agree to all terms and conditions before accessing \
the data.

Data Sharing and Usage Agreement

Before I download or process the NSD dataset, I agree to the following terms \
and conditions:
1. The Center for Magnetic Resonance Research (CMRR) grants me non-exclusive, \
royalty-free access to download and process this dataset.
2. I will utilize this dataset only for research and educational purposes.
3. I will not distribute this dataset or its components to any other individual \
or entity.
4. I will require anyone on my team who utilizes these data to comply with this \
data use agreement.
5. I will neither sell this dataset or its components nor monetize it.
6. I will comply with any rules and regulations imposed by my institution and \
its institutional review board in requesting these data.
7. The NSD dataset is collected from human subjects and has been de-identified. \
I will not retrieve or try to retrieve protected health information (PHI) of \
the human subjects in this dataset. If I incidentally discover PHI information, \
I will immediately inform the principal investigator.
8. I agree that all presentations and publications resulting from any use of \
this dataset must cite the relevant work using the suggested citation format \
listed below.
9. CMRR specifically disclaims any warranties including, but not limited to, \
the implied warranties of merchantability and fitness for a particular purpose. \
The dataset and the encompassing software provided hereunder is on an "as is" \
basis, and CMRR has no obligation to provide maintenance, support, updates, \
enhancements, or modifications.
10. In no event shall CMRR be liable to any party for direct, indirect, \
special, incidental, or consequential damages arising out of the use of this \
dataset and the accompanying software, even if CMRR has been advised of the \
possibility of such damage.
11. In addition to the above-listed terms and conditions, I will also comply \
with federal, state, local, and institutional policies and regulations.

Citation Format

If you make use of the NSD dataset, please cite the NSD data paper:
  Allen, St-Yves, Wu, Breedlove, Prince, Dowdle, Nau, Caron, Pestilli, \
Charest, Hutchinson, Naselaris*, & Kay*. A massive 7T fMRI dataset to bridge \
cognitive neuroscience and artificial intelligence. Nature Neuroscience (2021).

If you make use of the NSD synthetic data, please also cite the NSD synthetic \
paper:
  Gifford, Cichy, Naselaris, Kay. A 7T fMRI dataset of synthetic images for \
out-of-distribution modeling of vision. Nature Communications (2026).

In addition, please acknowledge the NSD funding sources using wording similar \
to:
  "Collection of the NSD dataset was supported by NSF IIS-1822683 and NSF \
IIS-1822929."
============================"""

    NSD_DATA_COMPONENTS: tp.ClassVar[tuple[str, ...]] = (
        "Task fMRI data (visually evoked responses to natural scenes)",
    )
    NSD_DATA_FORMAT: tp.ClassVar[tuple[str, ...]] = (
        "Prepared data (extensively pre-processed data)",
    )

    NSD_ROLES: tp.ClassVar[tuple[str, ...]] = (
        "Undergraduate student",
        "Graduate student",
        "Postdoc",
        "Faculty",
    )

    # ---- path helpers ----

    def _ts_path(self, subject: int, session: int, run: int) -> Path:
        return (
            self.path
            / f"nsddata_timeseries/ppdata/subj{subject:02}/func1pt8mm"
            / f"timeseries/timeseries_session{session:02}_run{run:02}.nii.gz"
        )

    def _design_path(self, subject: int, session: int, run: int) -> Path:
        return (
            self.path
            / f"nsddata_timeseries/ppdata/subj{subject:02}/func1pt8mm"
            / f"design/design_session{session:02}_run{run:02}.tsv"
        )

    def _roi_path(self, subject: int) -> Path:
        return (
            self.path
            / f"nsddata/ppdata/subj{subject:02}/func1pt8mm/roi/nsdgeneral.nii.gz"
        )

    # ---- T&C ----

    @staticmethod
    def _prompt_required(prompt: str, field_name: str) -> str:
        value = input(prompt).strip()
        if not value:
            raise ValueError(
                f"{field_name} is required for the NSD Data Access Agreement."
            )
        return value

    def _check_nsd_data_access_agreement(self) -> None:
        """Display T&C, collect user info, and submit the NSD Data Access Agreement form.

        Skips if the user has already accepted.
        The marker is stored in ``~/.config/neuralfetch/`` so it persists
        across data paths and covers all NSD studies automatically.
        """
        marker = get_nsd_tcs_marker()
        if marker.exists():
            logger.info("NSD Data Access Agreement already accepted (marker file found).")
            return

        if not sys.stdin or not getattr(sys.stdin, "isatty", lambda: False)():
            raise RuntimeError(
                "NSD Data Access Agreement requires an interactive session. "
                "Please run study.download() from a login node or interactive "
                "terminal first, then resubmit your batch job."
            )

        print("\n" + self.NSD_TCS_TEXT + "\n")
        response = input("Do you agree to the NSD Terms and Conditions? [y/N]: ")
        if response.lower() not in ("y", "yes"):
            raise PermissionError(
                "NSD dataset requires acceptance of the Terms and Conditions. "
                "Please review: https://cvnlab.slite.page/p/IB6BSeW_7o/Terms-and-Conditions"
            )

        print("\nPlease provide your information for the NSD Data Access Agreement:\n")
        name = self._prompt_required("Your name: ", "Name")
        email = self._prompt_required("Email: ", "Email")
        department = self._prompt_required("Department: ", "Department")
        institution = self._prompt_required("Institution: ", "Institution")

        num_options = len(self.NSD_ROLES) + 1
        print("\nRole:")
        for i, r in enumerate(self.NSD_ROLES, 1):
            print(f"  {i}. {r}")
        print(f"  {num_options}. Other")
        role_choice = input(f"Select your role [1-{num_options}]: ").strip()
        try:
            role_idx = int(role_choice) - 1
            if role_idx < 0 or role_idx > len(self.NSD_ROLES):
                raise ValueError()
        except ValueError:
            raise ValueError(
                f"Invalid role selection: {role_choice!r}. "
                f"Please enter a number 1-{num_options}."
            ) from None
        if role_idx == len(self.NSD_ROLES):
            role = self._prompt_required("Please specify your role: ", "Role")
        else:
            role = self.NSD_ROLES[role_idx]

        self._open_nsd_form(name, email, department, institution, role)

        submitted = input(
            "Once you have submitted the form, type 'y' to continue.\n"
            "Did you submit the form? [y/N]: "
        )
        if submitted.lower() not in ("y", "yes"):
            raise PermissionError(
                "Please submit the NSD Data Access Agreement form before downloading. "
                "Re-run download() to restart the consent flow."
            )

        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("NSD Data Access Agreement accepted.\n")
        marker.chmod(0o600)
        logger.info("NSD Data Access Agreement submitted and recorded.")

    def _open_nsd_form(
        self,
        name: str,
        email: str,
        department: str,
        institution: str,
        role: str,
    ) -> None:
        """Open the NSD Data Access Agreement Google Form pre-filled with user info."""
        params: dict[str, str | tuple[str, ...]] = {
            "emailAddress": email,
            "entry.1259301776": (
                "I agree to the Terms and Conditions as described at "
                "https://cvnlab.slite.com/api/s/note/9dgh5HCqgZYhMoAESZBS86/Terms-and-Conditions"
            ),
            "entry.1976545571": name,
            "entry.2017192646": department,
            "entry.1707004834": institution,
            "entry.2030550593": self.NSD_DATA_COMPONENTS,
            "entry.343810193": self.NSD_DATA_FORMAT,
        }
        if role in self.NSD_ROLES:
            params["entry.443953373"] = role
        else:
            params["entry.443953373"] = "__other_option__"
            params["entry.443953373.other_option_response"] = role

        prefill_url = self.NSD_FORM_VIEW_URL + "?" + urlencode(params, doseq=True)
        print(
            "\nAll fields have been pre-filled — please review and click Submit.\n"
            f"Form URL:\n  {prefill_url}\n"
        )
        opened = False
        try:
            opened = webbrowser.open(prefill_url)
        except Exception:
            pass
        if not opened:
            logger.warning(
                "Could not open the browser automatically; use the printed URL manually."
            )

    # ---- download ----

    def _download(self) -> None:
        self._check_nsd_data_access_agreement()
        with success_writer(self.path / "download_all") as already_done:
            if already_done:
                return
            self._download_nsd_timeseries_dataset()
            self._validate_downloaded_dataset()
            self._prepare_dataset()
            self._validate_prepared_dataset()

    def _run_aws(self, command: str) -> None:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            raise RuntimeError(
                f"AWS command failed with return code {result.returncode}:\n"
                f"Command: {command}\n"
                f"Error: {error_msg}\n"
                "Hint: Make sure the AWS CLI is installed and "
                "that the S3 bucket is reachable."
            )

    def _download_nsd_timeseries_dataset(self) -> None:
        self.path.mkdir(exist_ok=True, parents=True)

        for subject in self.SESSIONS_PER_SUBJECT:
            # Timeseries and design files
            self._run_aws(
                f"aws s3 sync --no-sign-request"
                f" {_S3_BUCKET}/nsddata_timeseries/ppdata/subj{subject:02}/func1pt8mm/"
                f" {self.path}/nsddata_timeseries/ppdata/subj{subject:02}/func1pt8mm/"
            )
            # ROI file
            roi_local = self._roi_path(subject)
            roi_local.parent.mkdir(parents=True, exist_ok=True)
            self._run_aws(
                f"aws s3 cp --no-sign-request"
                f" {_S3_BUCKET}/nsddata/ppdata/subj{subject:02}/func1pt8mm/roi/nsdgeneral.nii.gz"
                f" {roi_local}"
            )

        # Experimental design matrix
        self._run_aws(
            f"aws s3 cp --no-sign-request"
            f" {_S3_BUCKET}/nsddata/experiments/nsd/nsd_expdesign.mat"
            f" {self.path}/nsd_expdesign.mat"
        )
        # Stimulus HDF5
        self._run_aws(
            f"aws s3 cp --no-sign-request"
            f" {_S3_BUCKET}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
            f" {self.path}/nsd_stimuli.hdf5"
        )

    def _validate_downloaded_dataset(self) -> None:
        for tl in self._iter_subject_session_run():
            ts = self._ts_path(tl["subject"], tl["session"], tl["run"])
            if not ts.exists():
                raise RuntimeError(f"Missing timeseries file {ts} for {tl}")
            design = self._design_path(tl["subject"], tl["session"], tl["run"])
            if not design.exists():
                raise RuntimeError(f"Missing design file {design} for {tl}")

        for subject in self.SESSIONS_PER_SUBJECT:
            roi = self._roi_path(subject)
            if not roi.exists():
                raise RuntimeError(f"Missing ROI file {roi} for subject {subject}")

        for filename in ["nsd_expdesign.mat", "nsd_stimuli.hdf5"]:
            fp = self.path / filename
            if not fp.exists():
                raise RuntimeError(f"Missing common file {fp}")

    def _prepare_dataset(self) -> None:
        self._extract_stimuli(self.path)
        self._extract_test_images_ids(self.path)

    def _extract_stimuli(self, path: Path) -> None:
        import h5py

        f_stim = h5py.File(path / "nsd_stimuli.hdf5", "r")
        stim = f_stim["imgBrick"][:]
        nsd_stimuli_folder = path / "nsd_stimuli"
        nsd_stimuli_folder.mkdir(exist_ok=True, parents=True)
        for idx in trange(stim.shape[0]):
            Image.fromarray(stim[idx]).save(nsd_stimuli_folder / f"{idx}.png")

    def _extract_test_images_ids(self, path: Path) -> None:
        path_to_expdesign_mat = path / "nsd_expdesign.mat"
        expdesign_mat = _loadmat(path_to_expdesign_mat)
        np.save(path / "test_images_ids.npy", expdesign_mat["sharedix"])

    def _validate_prepared_dataset(self) -> None:
        files = [f for f in (self.path / "nsd_stimuli").iterdir() if f.suffix == ".png"]
        assert len(files) == self.N_STIMULI, (
            f"There should be {self.N_STIMULI} .png files in"
            f" {self.path / 'nsd_stimuli'} but found only {len(files)}"
        )
        test_ids = self.path / "test_images_ids.npy"
        if not test_ids.exists():
            raise RuntimeError(f"Missing file {test_ids} after dataset preparation")
        arr = np.load(test_ids)
        if arr.size == 0:
            raise RuntimeError(f"Empty test image-id array in {test_ids}")

    # ---- timelines ----

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for tl in self._iter_subject_session_run():
            yield tl

    @classmethod
    def _iter_subject_session_run(cls) -> tp.Iterator[dict[str, tp.Any]]:
        for subject, num_sessions in cls.SESSIONS_PER_SUBJECT.items():
            lo, hi = cls.SESSIONS_WITH_14_RUNS[subject]
            for session in range(1, num_sessions + 1):
                if lo <= session <= hi:
                    runs = range(2, 14)
                else:
                    runs = range(1, 13)
                for run in runs:
                    yield dict(subject=subject, session=session, task=cls.TASK, run=run)

    # ---- data loading ----

    def _load_raw(self, timeline: dict[str, tp.Any]) -> nibabel.Nifti2Image:
        subject = timeline["subject"]
        session = timeline["session"]
        run = timeline["run"]
        img: nibabel.Nifti1Image = nibabel.load(  # type: ignore[assignment]
            self._ts_path(subject, session, run)
        )
        data = img.get_fdata()
        num_volumes = data.shape[-1]
        if num_volumes != 226:
            raise RuntimeError(
                f"Unexpected number of fMRI volumes for Allen2022Massive "
                f"subj{subject:02} session{session:02} run{run:02}: "
                f"expected 226, got {num_volumes} (shape={data.shape})."
            )
        data = data[..., :225]
        data = data[_get_roi_mask(self._roi_path(subject))]
        return nibabel.Nifti2Image(data, np.eye(4))

    def _get_test_image_ids(self) -> list[int]:
        return np.load(self.path / "test_images_ids.npy").tolist()

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        subject = timeline["subject"]
        session = timeline["session"]
        run = timeline["run"]

        fmri = {
            "filepath": study.SpecialLoader(
                method=self._load_raw, timeline=timeline
            ).to_json(),
            "type": "Fmri",
            "start": 0.0,
            "frequency": 1.0 / self.TR_FMRI_S,
        }

        design_path = self._design_path(subject, session, run)
        test_ids = self._get_test_image_ids()
        stimuli_dir = self.path / "nsd_stimuli"

        image_events = []
        with open(design_path) as f:
            for timestep, line in enumerate(f):
                image_id = int(line.strip())  # 1-based, 0=no stimulus
                if image_id == 0:
                    continue
                image_events.append(
                    dict(
                        type="Image",
                        start=timestep * self.TR_FMRI_S,
                        duration=3.0,
                        filepath=str(stimuli_dir / f"{image_id - 1}.png"),
                        split="test" if image_id in test_ids else "train",
                    )
                )

        return pd.concat([pd.DataFrame([fmri]), pd.DataFrame(image_events)], axis=0)


# # # # # mini dataset # # # # #


_MINI_DATA_URL = "https://huggingface.co/datasets/ilikeitlikeit/mini-nsd-bold/resolve/main/mini_data.tar.gz"

_S3_TIMESERIES_PREFIX = "nsddata_timeseries/ppdata/subj01/func1pt8mm"
_S3_ROI_PATH = "nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz"


class Allen2022MassiveSample(Allen2022Massive):
    """Lightweight NSD study.

    Uses preprocessed timeseries for subject 1, session 1, runs 1-4.
    """

    description: tp.ClassVar[str] = (
        "Mini NSD: preprocessed 7T BOLD fMRI for subject 1, session 1, "
        "runs 1-4 (4 timelines)"
    )

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=4,
        num_subjects=1,
        num_events_in_query=64,
        event_types_in_query={"Fmri", "Image"},
        data_shape=(15724, 225),
        frequency=0.75,
        fmri_spaces={"custom"},
    )

    def _download(self) -> None:
        self._check_nsd_data_access_agreement()

        s3_files = [_S3_ROI_PATH]
        for run in range(1, 5):
            s3_files.append(
                f"{_S3_TIMESERIES_PREFIX}/timeseries/timeseries_session01_run{run:02d}.nii.gz"
            )
            s3_files.append(
                f"{_S3_TIMESERIES_PREFIX}/design/design_session01_run{run:02d}.tsv"
            )

        for filepath in s3_files:
            local_path = self.path / filepath
            if local_path.exists():
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                "aws",
                "s3",
                "cp",
                f"{_S3_BUCKET}/{filepath}",
                str(local_path),
                "--no-sign-request",
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                raise RuntimeError(
                    f"AWS command failed (rc={result.returncode}):\n"
                    f"Command: {' '.join(command)}\nError: {error_msg}"
                )

        archive_path = self.path / "mini_data.tar.gz"
        if not archive_path.exists():
            resp = requests.get(_MINI_DATA_URL)
            resp.raise_for_status()
            archive_path.write_bytes(resp.content)

        with tarfile.open(archive_path) as tar:
            tar.extractall(self.path, filter="data")

        self._validate_mini_files()

    def _validate_mini_files(self) -> None:
        """Check that all expected files are present after download."""
        missing = []
        for run in range(1, 5):
            if not self._ts_path(1, 1, run).exists():
                missing.append(str(self._ts_path(1, 1, run)))
            if not self._design_path(1, 1, run).exists():
                missing.append(str(self._design_path(1, 1, run)))

        d = self.path / "nsd_stimuli"
        if not d.exists() or not d.is_dir():
            missing.append(str(d))

        roi = self.path / _S3_ROI_PATH
        if not roi.exists():
            missing.append(str(roi))

        if missing:
            raise RuntimeError("Missing files after download:\n  " + "\n  ".join(missing))

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        if not self._ts_path(1, 1, 1).exists() or not self._design_path(1, 1, 1).exists():
            raise FileNotFoundError(
                f"Missing Allen2022MassiveSample files in {self.path}. "
                "Please run study.download() first."
            )
        for run in range(1, 5):
            yield dict(subject=1, session=1, run=run)

    # NSD "shared" image IDs (1-based) that appear in session 1, runs 1-4.
    _TEST_IMAGE_IDS: tp.ClassVar[frozenset[int]] = frozenset(
        {
            4931,
            5302,
            5603,
            6432,
            7660,
            21602,
            25960,
            30374,
            32626,
            36577,
            42172,
            44981,
            46003,
            48618,
            53053,
            57047,
            62303,
            65415,
            70336,
        }
    )

    def _get_test_image_ids(self) -> tp.Collection[int]:  # type: ignore[override]
        return self._TEST_IMAGE_IDS
