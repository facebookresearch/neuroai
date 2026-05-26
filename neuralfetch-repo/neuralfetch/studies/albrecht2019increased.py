# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import mne
import pandas as pd
from scipy.io import loadmat

from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)


class Albrecht2019Increased(study.Study):
    """EEG responses during a reinforcement learning Simon task.

    EEG recorded at 1000 Hz from 77 participants (46 healthy controls,
    31 participants with schizophrenia) performing a modified Simon task
    that introduces response conflict as an implicit cognitive cost during
    reinforcement learning. One session per participant.

    Experimental Design:
        - EEG recordings (64-channel, 1000 Hz)
        - 77 participants (46 healthy controls, 31 with schizophrenia)
        - 1 session per participant
        - Paradigm: reinforcement learning Simon task
            * Response conflict as implicit cognitive cost

    Notes:
        - Downloaded from Hugging Face (not uploaded by study authors).
    """

    bibtex: tp.ClassVar[str] = """
    @article{albrecht2019increased,
        title={Increased Conflict-Induced Slowing, but No Differences in Conflict-Induced Positive or Negative Prediction Error Learning in Patients with Schizophrenia},
        author={Albrecht, Matthew A. and Waltz, James A. and Cavanagh, James F. and Frank, Michael J. and Gold, James M.},
        year=2019,
        journal={Neuropsychologia},
        volume={123},
        pages={131--140},
        doi={10.1016/j.neuropsychologia.2018.04.031},
    }

    @misc{albrecht2019_data,
        url={https://huggingface.co/datasets/jalauer/Albrecht2019/tree/main}
    }
    """
    licence: tp.ClassVar[str] = "PDDL-1.0"
    description: tp.ClassVar[str] = (
        "EEG recordings (64-channel, 1000 Hz) from 77 participants "
        "(46 healthy controls, 31 with schizophrenia) performing a "
        "reinforcement learning Simon task."
    )
    url: tp.ClassVar[str] = (
        "https://huggingface.co/datasets/jalauer/Albrecht2019/tree/main"
    )
    _SUBJECT_RUNS: dict[str, tp.Iterable[int]] = {
        "P": set(range(101, 149)) - set((106, 118)),
        "N": set(range(149, 181)) - set((175,)),
    }
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        # on HF, the uploader placed `CC_EEG_s118_P.mat` in a `Not_used` folder
        # Created an issue to ask why:
        # https://huggingface.co/datasets/jalauer/Albrecht2019/discussions/1
        num_subjects=77,
        num_timelines=77,
        num_events_in_query=1459,
        event_types_in_query={"Eeg", "Stimulus"},
        data_shape=(62, 1665450),
        frequency=1000,
    )

    def _download(self, overwrite=False) -> None:
        # https://predict.cs.unm.edu/downloads.php d004
        # Unable to use original dataset on PRED+ct.
        # Files shared here rely on Sharepoint and cannot be programmatically downloaded there.
        # Alternate source found on Hugging Face, not uploaded by original authors.
        # https://huggingface.co/jalauer/datasets
        hf_org = "jalauer"
        hf_repo = "Albrecht2019"
        hg = download.Huggingface(org=hf_org, study=hf_repo, dset_dir=self.path)
        if hg.get_success_file().exists() and not overwrite:
            return
        hg.download(overwrite=overwrite)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """Returns a generator of all recordings"""
        mapping = {"N": "Control", "P": "Schizophrenic"}
        task = "Simon"
        for group, subjects in self._SUBJECT_RUNS.items():
            for subject in subjects:
                sub_id = f"CC_EEG_s{subject}_{group}"
                yield dict(subject=sub_id, diagnosis=mapping[group], task=task)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        basename = self.path / "download" / "data" / timeline["subject"]
        # extract annotations
        mat = loadmat(
            basename.with_suffix(".mat"), simplify_cells=True, variable_names=["EEG"]
        )["EEG"]
        events = mat["event"]
        events = pd.DataFrame(events)[["latency", "code", "duration", "type"]]
        # convert to str
        events["code"] = events["code"].astype(str)
        events_df = events.rename(
            columns={"latency": "start", "code": "type", "type": "trigger"}
        )
        # convert from ms to s
        events_df[["start", "duration"]] /= 1000
        # Stimuli presented in different reward functions where a color is associated
        # with button press. Shapes are displayed in one of the two colors. If color
        # correspond the same side as button press, it's considered congruent, else
        # it's considered incongruent.
        events_df = events_df[events_df["type"] == "Stimulus"]

        def trigger_to_description(trigger: str):
            # Triggers were designed as three digit strings with each digit corresponding
            # to a different condition. Descriptions go left to right:
            mapping = {
                "6": "-3 Error!",
                "7": "-3! No Response",
                "8": "Correct +1",
                "9": "No Reward 0",
            }
            if len(trigger) != 3:
                return mapping.get(trigger)
            mapping_digit0 = {"1": "yellow", "2": "blue"}
            mapping_digit1 = {"1": "congruent", "2": "incongruent"}
            mapping_digit2 = {
                "1": "Reinforced-100",
                "2": "Reinforced-100-congruent-only",
                "3": "Reinforced-100-incongruent-only",
                "4": "Reinforced-0",
            }
            hed = (
                f"{mapping_digit0.get(trigger[0])}/"
                f"{mapping_digit1.get(trigger[1])}/"
                f"{mapping_digit2.get(trigger[2])}"
            )

            return hed

        # Convert trigger to str for easier parsing
        events_df["description"] = events_df["trigger"].transform(trigger_to_description)

        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = dict(type="Eeg", filepath=info, start=0)
        events = pd.concat([pd.DataFrame([eeg]), events_df], ignore_index=True)
        return events

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        basename = self.path / "download" / "data" / timeline["subject"]

        chs_df = pd.read_csv(
            basename.parents[1] / "MMN_ElectrodeCoords_MA2.ced", delimiter="\t"
        )
        mat = loadmat(
            basename.with_suffix(".mat"), simplify_cells=True, variable_names=["EEG"]
        )["EEG"]
        data = mat["data"]
        sfreq = mat["srate"]
        info = mne.create_info(
            ch_names=list(chs_df["labels"]), sfreq=sfreq, ch_types="eeg"
        )
        raw = mne.io.RawArray(data=data, info=info)
        eog_channels = [ch for ch in raw.ch_names if ch.startswith("VEOG")]
        if eog_channels:
            raw.set_channel_types({ch: "eog" for ch in eog_channels})
        raw.set_montage("standard_1005", on_missing="ignore")
        return raw
