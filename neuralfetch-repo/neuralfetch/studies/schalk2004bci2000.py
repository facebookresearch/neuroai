# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import shutil
import subprocess
import typing as tp
from pathlib import Path

import mne
import pandas as pd

from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)
logger.propagate = False


class Schalk2004Bci2000(study.Study):
    url: tp.ClassVar[str] = "https://www.physionet.org/content/eegmmidb/1.0.0/"
    """EEG Motor Movement/Imagery Dataset (EEGMMIDB): EEG responses to motor execution and imagery tasks.

    This dataset contains 1,526 EEG recordings from 109 participants performing
    motor and motor imagery tasks using the BCI2000 system. Participants performed
    baseline (eyes open/closed), motor execution (opening/closing fists and feet),
    and motor imagery tasks across 14 experimental runs per participant.

    Experimental Design:
        - EEG recordings (64-channel, 160 Hz)
        - 109 participants
        - 14 runs per participant (2 baseline + 12 task runs)
        - Paradigm: motor execution and motor imagery
            * Baseline runs: eyes open and eyes closed
            * Motor tasks: open/close left or right fist, both fists, or both feet
            * Imagery tasks: imagine opening/closing left or right fist, both fists, or both feet

    Notes:
        - Includes all 14 runs per participant (baseline + motor + imagery).
        - See also Schalk2004Bci2000Moabb in moabb2025.py, which wraps the same
          dataset via MOABB's PhysionetMI but only exposes the motor imagery
          subset with parsed trial labels.
    """

    aliases: tp.ClassVar[tuple[str, ...]] = (
        "EEG Motor Movement/Imagery Dataset",
        "EEGMMIDB",
    )

    licence: tp.ClassVar[str] = "ODC-By-1.0"
    bibtex: tp.ClassVar[str] = """
    @article{schalk2004bci2000,
        author={Schalk, G. and McFarland, D.J. and Hinterberger, T. and Birbaumer, N. and Wolpaw, J.R.},
        journal={IEEE Transactions on Biomedical Engineering},
        title={BCI2000: a general-purpose brain-computer interface (BCI) system},
        year={2004},
        volume={51},
        number={6},
        pages={1034-1043},
    }

    @misc{schalk2009eeg,
        doi={10.13026/C28G6P},
        url={https://physionet.org/content/eegmmidb/},
        author={Schalk,  Gerwin and McFarland,  Dennis J and Hinterberger,  Thilo and Birbaumer,  Niels and Wolpaw,  Jonathan R},
        title={EEG Motor Movement/Imagery Dataset},
        publisher={physionet.org},
        year={2009}
    }
    """
    description: tp.ClassVar[str] = (
        "Corpus of 1,526 EEG recordings (1-2 mins) from 109 participants performing motor and imagery tasks."
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1526,
        num_subjects=109,
        num_events_in_query=2,
        event_types_in_query={"Eeg", "EyeState"},
        data_shape=(64, 9760),
        frequency=160,
    )

    def _download(self) -> None:
        #  Option to download through S3
        #  aws s3 sync s3://physionet-open/eegmmidb/1.0.0/ DESTINATION
        folder = self.path / "download"
        temp_folder = self.path / "temp"
        download_url = "https://physionet.org/files/eegmmidb/1.0.0/"
        with download.success_writer(folder) as already_done:
            if not already_done:
                temp_folder.mkdir(exist_ok=True, parents=True)
                subprocess.run(
                    (f"wget -r -N -c -np -P {temp_folder} {download_url}"),
                    shell=True,
                    check=False,
                )
                download_path = temp_folder / "physionet.org/files/eegmmidb/1.0.0/"
                download_path.rename(folder)
                shutil.rmtree(temp_folder)  # cleanup empty folder
                logger.info(f"Downloaded files to {folder}.")

    _RUN_CONDITION: tp.ClassVar[dict[str, str]] = {
        "R01": "Rest",
        "R02": "Rest",
        "R03": "Motor",
        "R07": "Motor",
        "R11": "Motor",
        "R05": "Motor",
        "R09": "Motor",
        "R13": "Motor",
        "R04": "Imagery",
        "R08": "Imagery",
        "R12": "Imagery",
        "R06": "Imagery",
        "R10": "Imagery",
        "R14": "Imagery",
    }

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        folder = self.path / "download"
        for sub_dir in sorted(folder.iterdir()):
            for file_path in sorted(sub_dir.rglob("*.edf")):
                subject = file_path.stem[:4]
                run = file_path.stem[4:]
                yield dict(
                    subject=subject, run=run, condition=self._RUN_CONDITION.get(run, "")
                )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg_events = pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    filepath=info,
                    start=0.0,
                ),
            ]
        )
        motor_events = self._get_event_annotations(timeline)
        return pd.concat([eeg_events, motor_events]).reset_index(drop=True)

    def _get_eeg_filename(self, timeline: dict[str, tp.Any]) -> Path:
        return (
            self.path
            / "download"
            / timeline["subject"]
            / f"{timeline['subject']}{timeline['run']}.edf"
        )

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        raw = mne.io.read_raw_edf(self._get_eeg_filename(timeline))
        replacements = {
            ".": "",
            "Af": "AF",
            "Cp": "CP",
            "Fc": "FC",
            "Po": "PO",
            "Tp": "TP",
            "Ft": "FT",
        }
        ch_types, ch_names_mapping = {}, {}
        for ch in raw.ch_names:
            fixed_ch = ch
            for key in replacements.keys():
                fixed_ch = fixed_ch.replace(key, replacements[key])
            ch_names_mapping[ch] = fixed_ch
            ch_types[fixed_ch] = "eeg"
        montage = mne.channels.make_standard_montage("standard_1005")
        raw = raw.rename_channels(ch_names_mapping)
        raw = raw.set_channel_types(ch_types)
        raw = raw.set_montage(montage, on_missing="ignore")
        to_drop = [
            name
            for name in raw.ch_names
            if ch_types[name] == "eeg" and name not in montage.ch_names
        ]
        raw = raw.drop_channels(to_drop)
        if len(to_drop) > 0:
            logger.info("Dropped %s unrecognized EEG channels: %s", len(to_drop), to_drop)
        raw = raw.set_montage(montage, on_missing="ignore")
        return raw

    def _get_event_annotations(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        eeg = mne.io.read_raw_edf(self._get_eeg_filename(timeline))
        motor_df = pd.DataFrame(
            columns=[
                "type",
                "filepath",
                "start",
                "duration",
                "frequency",
                "code",
                "description",
                "state",
            ]
        )

        DESC_CODE_MAPPING = {
            "rest": 0,
            "motor_left_fist": 1,
            "motor_right_fist": 2,
            "motor_bilateral_fist": 3,
            "motor_bilateral_feet": 4,
            "imagery_left_fist": 5,
            "imagery_right_fist": 6,
            "imagery_bilateral_fist": 7,
            "imagery_bilateral_feet": 8,
        }
        annots = eeg.annotations

        run = timeline["run"]
        # Rest runs
        if run in ["R01", "R02"]:
            event_type = "EyeState"
            # Skip annotation for unknown task
            if set(annots.description) != {"T0"}:
                return motor_df
            if run == "R01":
                description_dict = {"T0": "open"}
            else:
                description_dict = {"T0": "closed"}
        # Motor runs
        elif run in ["R03", "R07", "R11", "R05", "R09", "R13"]:
            event_type = "Action"
            if run in ["R03", "R07", "R11"]:
                description_dict = {
                    "T0": "rest",
                    "T1": "motor_left_fist",
                    "T2": "motor_right_fist",
                }
            else:
                description_dict = {
                    "T0": "rest",
                    "T1": "motor_bilateral_fist",
                    "T2": "motor_bilateral_feet",
                }
        # Motor imagery runs
        elif run in ["R04", "R08", "R12", "R06", "R10", "R14"]:
            event_type = "Action"
            if run in ["R04", "R08", "R12"]:
                description_dict = {
                    "T0": "rest",
                    "T1": "imagery_left_fist",
                    "T2": "imagery_right_fist",
                }
            else:
                description_dict = {
                    "T0": "rest",
                    "T1": "imagery_bilateral_fist",
                    "T2": "imagery_bilateral_feet",
                }
        else:
            ValueError(f"Invalid input for variable. Got run={run}, expected R[00-12].")

        motor_df["start"] = annots.onset
        motor_df["duration"] = annots.duration
        motor_df["frequency"] = eeg.info["sfreq"]
        motor_df["type"] = event_type
        if event_type == "EyeState":
            motor_df["state"] = [description_dict[c] for c in annots.description]
        else:
            motor_df["description"] = [description_dict[c] for c in annots.description]
            motor_df["code"] = [DESC_CODE_MAPPING[c] for c in motor_df["description"]]

        return motor_df
