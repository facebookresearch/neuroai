# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)
logger.propagate = False


BAD_SUBJECTS = [
    # From https://github.com/eeg2025/startkit/pull/10
    "sub-NDARWV769JM7",
    "sub-NDARME789TD2",
    "sub-NDARUA442ZVF",
    "sub-NDARJP304NK1",
    "sub-NDARTY128YLU",
    "sub-NDARDW550GU6",
    "sub-NDARLD243KRE",
    "sub-NDARUJ292JXV",
    "sub-NDARBA381JGH",
]


class Shirazi2024Hbn(study.Study):
    url: tp.ClassVar[str] = (
        "https://openneuro.org/datasets/ds005516/versions/1.0.1/download"
    )
    """HBN-EEG: EEG recordings from young participants during cognitive and video tasks.

    The Healthy Brain Network EEG study provides large-scale EEG recordings from
    participants aged 5-21 across 11 data releases. Participants completed a variety of
    cognitive tasks and watched movie clips during 128-channel EEG recording. The
    population consists of children and adolescents.

    Experimental Design:
        - EEG recordings (129-channel, 500 Hz)
        - 3,146 participants (ages 5-21, after exclusions)
        - Multiple task paradigms:
            * Movie watching: DespicableMe, DiaryOfAWimpyKid, FunwithFractals,
              ThePresent
            * Resting-state: eyes-closed / eyes-open alternation
            * Surround suppression: passive viewing of flashing peripheral disks
            * Contrast change detection: visual attention task with button response
            * Symbol search (WISC-IV): visual processing speed task
            * Sequence learning (6- and 8-target): visuospatial memory task
            * Stimulus language: English (movie clips)

    Data Format:
        - 26,615 total timelines across 11 releases
        - EEGLAB .set files
        - Event types: Eeg, Stimulus, Keystroke
        - Participant metadata: sex, age, ehq_total, p_factor, attention,
          internalizing, externalizing
    """

    aliases: tp.ClassVar[tuple[str, ...]] = ("Healthy Brain Network EEG", "HBN-EEG")

    licence: tp.ClassVar[str] = "CC-BY-SA-4.0"

    bibtex: tp.ClassVar[str] = """
        @article{shirazi2024hbn,
            url = {http://dx.doi.org/10.1101/2024.10.03.615261},
            title={HBN-EEG: The FAIR implementation of the Healthy Brain Network (HBN) electroencephalography dataset},
            url={http://dx.doi.org/10.1101/2024.10.03.615261},
            doi={10.1101/2024.10.03.615261},
            publisher={Cold Spring Harbor Laboratory},
            author={Shirazi,  Seyed Yahya and Franco,  Alexandre and Hoffmann,  Maurício Scopel and Esper,  Nathalia B. and Truong,  Dung and Delorme,  Arnaud and Milham,  Michael P. and Makeig,  Scott},
            year={2024},
            month=oct
        }

        @misc{shirazi2025healthy,
            author={Seyed Yahya Shirazi AND Alexandre Franco AND Maurício Scopel Hoffmann AND Nathalia B. Esper AND Dung Truong AND Arnaud Delorme AND Michael Milham AND Scott Makeig},
            title={"Healthy Brain Network (HBN) EEG - Release 11"},
            year={2025},
            doi={10.18112/openneuro.ds005516.v1.0.1},
            publisher={OpenNeuro},
            url={https://openneuro.org/datasets/ds005516/versions/1.0.1/download}
            }
        """
    description: tp.ClassVar[str] = (
        "Large-scale set of EEG recordings from young participants (ages 5-21) during 6 cognitive tasks"
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=26615,
        num_subjects=3146,
        num_events_in_query=1,
        event_types_in_query={"Eeg"},
        data_shape=(129, 86151),
        frequency=500,
    )
    RELEASE_TO_STUDYID: tp.ClassVar[dict] = {
        "R1": "ds005505",
        "R2": "ds005506",
        "R3": "ds005507",
        "R4": "ds005508",
        "R5": "ds005509",
        "R6": "ds005510",
        "R7": "ds005511",
        "R8": "ds005512",
        "R9": "ds005514",
        "R10": "ds005515",
        "R11": "ds005516",
    }

    COGNITIVE_TASKS: tp.ClassVar[list] = [
        "task-contrastChangeDetection",
        "task-surroundSupp",
        "task-symbolSearch",
        "task-seqLearning6target",
        "task-seqLearning8target",
    ]

    # Timing inferred from Fig. 6 of Shirazi et al., 2024
    DUR_CONTRAST_CHANGE_TARGET: tp.ClassVar[float] = 2.4
    DUR_CONTRAST_CHANGE_FEEDBACK: tp.ClassVar[float] = 0.4

    def _download(self) -> None:
        for release, study_id in self.RELEASE_TO_STUDYID.items():
            n = release.split("-")[-1]
            logger.info(
                f"Downloading Dataset: Healthy Brain Network (HBN) EEG - Release {n}"
            )
            download.Openneuro(study=study_id, dset_dir=self.path / release).download()

    def iter_timelines(self):
        for release in self.RELEASE_TO_STUDYID.keys():
            folder = self.path / release / "download"
            for sub_dir in sorted(folder.glob("sub-*")):
                if sub_dir.stem in BAD_SUBJECTS:
                    logger.info(f"Skipping bad subject: {sub_dir.stem}")
                    continue
                eeg_dir = sub_dir / "eeg"
                for filename in sorted(eeg_dir.glob("*.set")):
                    if "run-" in filename.stem:
                        subject, task, run, _ = filename.stem.split("_")
                    else:
                        subject, task, _ = filename.stem.split("_")
                        run = None
                    yield dict(release=release, subject=subject, task=task, run=run)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        tl = timeline
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = pd.DataFrame([dict(type="Eeg", filepath=info, start=0)])

        # Adds participant characteristics to EEG events
        pt_chars = self._get_participant_characteristics(timeline)
        events_df = pd.concat([eeg, pt_chars], axis=1)
        # Adds task events
        if tl["task"] in self.COGNITIVE_TASKS:
            if tl["task"] == "task-surroundSupp":
                cog_df = self._load_surroundsupp_events(timeline)
            elif tl["task"] == "task-contrastChangeDetection":
                cog_df = self._load_contrast_change_detection_events(timeline)
            elif tl["task"] == "task-symbolSearch":
                cog_df = self._load_symbol_search_events(timeline)
            else:
                cog_df = self._load_seq_learning_events(timeline)
            events_df = pd.concat([events_df, cog_df]).reset_index(drop=True)
        col_types = {
            key: str
            for key in ["text", "button", "correct_answer"]
            if key in events_df.columns
        }
        events_df = events_df.astype(col_types)
        return events_df.dropna(subset=["type"])

    def _get_filename(self, filetype: str, timeline: dict[str, tp.Any]) -> Path:
        tl = timeline
        file_dir = self.path / tl["release"] / "download"
        if filetype in ["eeg", "events", "channels"]:
            file_dir = file_dir / tl["subject"] / "eeg"
            file_format = {"eeg": "set", "events": "tsv", "channels": "tsv"}
            if tl["run"] is None:
                prefix = f"{tl['subject']}_{tl['task']}_{filetype}"
            else:
                prefix = f"{tl['subject']}_{tl['task']}_{tl['run']}_{filetype}"
            file_name = f"{prefix}.{file_format[filetype]}"
        elif filetype == "task_info":
            file_name = f"{tl['task']}_eeg.json"
        elif filetype == "participant_info":
            file_name = "participants.tsv"
        return file_dir / file_name

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.Raw:
        eeg_file = self._get_filename("eeg", timeline)
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
        ref_channel = self._get_task_info(timeline)["EEGReference"]
        if ref_channel in raw.ch_names:
            raw.load_data()
            raw.set_eeg_reference(ref_channels=[ref_channel])
        else:
            logger.warning(f"No reference set: missing reference channel: {ref_channel}")

        # Crop events that overlap with the last timestep of the recording
        # Required to avoid errors when exporting to BrainVision - happened at least for
        # timeline="b46fbe339f_Shirazi2024Hbn_subject-_b46fbe339f_run-run-2_release-R7"
        raw.annotations.crop(raw.times[0], raw.times[-3])

        return raw

    def _get_participant_characteristics(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        particpant_file = self._get_filename("participant_info", timeline)
        df = pd.read_csv(particpant_file, sep="\t")
        participant_mask = df.participant_id == timeline["subject"]
        output = df.loc[
            participant_mask,
            [
                "sex",
                "age",
                "ehq_total",
                "p_factor",
                "attention",
                "internalizing",
                "externalizing",
            ],
        ]
        return output.reset_index(drop=True)

    def _get_task_info(self, timeline: dict[str, tp.Any]) -> dict:
        info_file = self._get_filename("task_info", timeline)
        with info_file.open() as file:
            data = json.load(file)
        return data

    def _load_surroundsupp_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        events_file = self._get_filename("events", timeline)
        events_df = pd.read_csv(events_file, sep="\t")
        events_df = events_df[events_df.value == "stim_ON"]
        events_df = events_df.rename(
            columns={
                "onset": "start",
                "stimulus_cond": "code",
            }
        )
        condition_dict = {
            1: "Stimulus condition 1: peripheral foreground disks with parallel, spatially opposite-phase background",
            2: "Stimulus condition 2: peripheral foreground disks with parallel, spatially in-phase background",
            3: "Stimulus condition 3: peripheral foreground disks with orthogonal background",
        }
        events_df["description"] = events_df["code"].map(condition_dict)
        events_df["type"] = "Stimulus"
        return events_df[
            [
                "type",
                "start",
                "duration",
                "code",
                "description",
                "background",  # 0 = no background; 1 = background
                "foreground_contrast",  # 0 = no contrast; 0.33 = 1/3 contrast; 0.66 = 2/3 contrast; 1 = full contrast
            ]
        ]

    @staticmethod
    def _add_trial_number(
        events_df: pd.DataFrame,
        event_col: str,
        start_events: list,
        ignore_cols: list | None = None,
        output_col: str = "trial_num",
    ) -> pd.DataFrame:
        """Adds trial number to events table."""
        if ignore_cols is None:
            ignore_cols = []
        events_df[output_col] = (
            (events_df[event_col].isin(start_events)).astype(float).cumsum()
        )
        events_df.loc[events_df[event_col].isin(ignore_cols), output_col] = pd.NA
        return events_df

    @staticmethod
    def _get_trial_times(
        events_df: pd.DataFrame,
        event_col: str,
        target_events: list,
        response_events: list,
        timing_events: list,
        trial_type: tp.Literal["target", "buttonPress"],
    ) -> pd.DataFrame:
        """Returns the start, duration, and reaction time given an events table."""
        target_mask = events_df[event_col].isin(target_events)
        response_mask = events_df[event_col].isin(response_events)
        target_time = events_df[target_mask].groupby("trial_num").onset.min()
        response_time = (
            events_df[response_mask].groupby("trial_num").onset.min()
        )  # handles cases of multiple responses
        duration = (
            events_df.loc[events_df[event_col].isin(timing_events), "onset"]
            .diff(1)
            .dropna()
        )
        duration = duration.reset_index(drop=True).iloc[
            1:
        ]  # removes break before first trial begins
        if trial_type == "target":
            output = pd.concat(
                [
                    pd.Series(target_time, name="start"),
                    pd.Series(duration, name="duration"),
                ],
                axis=1,
            )
        else:
            output = pd.concat(
                [
                    pd.Series(response_time, name="start"),
                    pd.Series(response_time - target_time, name="reaction_time"),
                ],
                axis=1,
            )
        return output.dropna(subset="start")

    def _load_contrast_change_detection_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        tl = timeline
        events_file = self._get_filename("events", timeline)
        events_df = pd.read_csv(events_file, sep="\t")
        events_df = self._add_trial_number(
            events_df=events_df,
            event_col="value",
            start_events=["contrastTrial_start"],
            ignore_cols=["break cnt", "9999", "contrastChangeB1_start"],
        )

        # Adds trial events (i.e., grating stimuli appear)
        trial_df = events_df[events_df.value == "contrastTrial_start"].rename(
            columns={"onset": "start", "value": "description"}
        )
        trial_df["code"] = trial_df.description.astype("category").cat.codes
        trial_df = trial_df[
            ["trial_num", "start", "duration", "description", "code"]
        ].reset_index(drop=True)
        trial_df["duration"] = trial_df["start"].diff(-1).abs()
        trial_df.insert(0, "type", "Stimulus")
        trial_df.insert(1, "event_type", "trial")

        # Adds target events (i.e., left or right contrast change)
        target_df = (
            events_df[events_df.value.isin(["left_target", "right_target"])]
            .rename(columns={"onset": "start", "value": "description"})
            .reset_index(drop=True)
        )
        target_type = CategoricalDtype(
            categories=["left_target", "right_target"], ordered=True
        )
        target_df["code"] = target_df.description.astype(target_type).cat.codes
        target_df = target_df[["trial_num", "start", "duration", "description", "code"]]

        target_df["duration"] = self.DUR_CONTRAST_CHANGE_TARGET
        target_df.insert(0, "type", "Stimulus")
        target_df.insert(1, "event_type", "target")

        # Handles instances of no button press events
        if not {"right_buttonPress", "left_buttonPress"} < set(events_df["value"]):
            logger.warning(
                f"No button responses recorded for {tl['subject']} {tl['task']} {tl['run']}"
            )
            return target_df

        # Adds button press events
        button_df = events_df[
            events_df.value.isin(["right_buttonPress", "left_buttonPress"])
        ].rename(columns={"onset": "start", "value": "text"})
        button_type = CategoricalDtype(
            categories=["left_buttonPress", "right_buttonPress"], ordered=True
        )
        button_df["button"] = button_df["text"].astype(button_type).cat.codes
        button_df = button_df[
            ["trial_num", "start", "text", "button", "feedback"]
        ].reset_index(drop=True)
        button_df.insert(0, "type", "Keystroke")
        button_df.insert(1, "event_type", "button_press")
        # Calculates reaction time given time of target onset
        rt = pd.merge(
            left=button_df[["start", "trial_num"]],
            right=target_df[["start", "trial_num"]],
            how="left",
            on="trial_num",
            suffixes=["_button", "_target"],
        )
        rt["reaction_time"] = rt["start_button"] - rt["start_target"]
        button_df["reaction_time"] = rt.reaction_time.values
        # Determines whether response correct given feedback
        button_df.loc[:, "is_correct"] = button_df.feedback.replace(  # type: ignore
            {
                "smiley_face": True,  # Correct response
                "sad_face": False,  # Incorrect response
                # Response precedes target onset:
                "non_target": pd.NA,  # type: ignore
            }
        )

        # Adds reaction time and correctness to target
        target_df = pd.merge(
            left=target_df,
            right=button_df.groupby("trial_num")[["reaction_time", "is_correct"]].max(),
            how="left",
            on="trial_num",
        )
        # Sets non-response (i.e., miss) correctness to 0.0
        target_df["is_correct"] = target_df["is_correct"].fillna(False)

        # Adds feedback events
        feedback_df = (
            button_df.rename(columns={"feedback": "description"})
            .drop(["button", "reaction_time"], axis=1)
            .drop(["start", "text", "is_correct"], axis=1)
        )
        feedback_df["type"] = "Stimulus"
        feedback_df["event_type"] = "feedback"
        feedback_type = CategoricalDtype(
            categories=["sad_face", "smiley_face", "non_target"], ordered=True
        )
        feedback_df["code"] = feedback_df.description.astype(feedback_type).cat.codes
        # Adds start / duration
        feedback_df = pd.merge(
            left=feedback_df,
            right=target_df[["trial_num", "start"]],
            how="left",
            on="trial_num",
        )
        feedback_df["start"] = feedback_df["start"] + self.DUR_CONTRAST_CHANGE_TARGET
        feedback_df["duration"] = self.DUR_CONTRAST_CHANGE_FEEDBACK

        # Drops responses preceding target onset
        feedback_df = feedback_df[feedback_df.description != "non_target"]

        return (
            pd.concat(
                [trial_df, target_df, button_df.dropna(subset="button"), feedback_df],
                axis=0,
            )
            .sort_values(["trial_num", "start"])
            .reset_index(drop=True)
        )

    def _load_symbol_search_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        task_file = self._get_filename("events", timeline)
        task_df = pd.read_csv(task_file, sep="\t")
        task_df = self._add_trial_number(
            events_df=task_df,
            event_col="value",
            start_events=["newPage"],
            output_col="page_num",
        )
        output = task_df[task_df.value == "trialResponse"]  # Keystroke press events
        output["is_correct"] = output["user_answer"] == output["correct_answer"]
        output = output.rename(
            columns={
                "user_answer": "button",
                "onset": "start",
                "value": "description",
            }
        )
        output.insert(0, "type", "Keystroke")
        output["text"] = output["button"].replace(
            {0: "target_absent", 1: "target_present"}
        )
        return (
            output[
                [
                    "type",
                    "start",
                    "text",
                    "button",
                    "correct_answer",
                    "is_correct",
                    "description",
                    "page_num",
                ]
            ]
            .dropna(subset="text")
            .reset_index(drop=True)
        )

    def _load_seq_learning_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        task_file = self._get_filename("events", timeline)
        task_df = pd.read_csv(task_file, sep="\t")
        block_vals = [
            b for b in task_df["value"].dropna().unique() if "learningBlock_" in b
        ]
        n_blocks = len(block_vals)
        if n_blocks <= 0:
            raise ValueError("`n_block` needs to be at least 1. {n_blocks} provided.")
        n_blocks = len(block_vals)
        task_df = self._add_trial_number(
            events_df=task_df,
            event_col="value",
            start_events=block_vals,
            output_col="block",
        )
        on_events = sorted([x for x in task_df.value.dropna().unique() if "_ON" in x])
        stim_events = task_df[task_df["value"].isin(on_events)]
        stim_events = stim_events.rename(
            columns={
                "onset": "start",
                "value": "description",
            }
        )
        stim_events["code"] = stim_events.description.str.replace(
            "dot_no", ""
        ).str.replace("_ON", "")
        stim_events = stim_events[
            [
                "start",
                "duration",
                "block",
                "description",
                "code",
            ]
        ].reset_index(drop=True)
        stim_events.insert(0, "type", "Stimulus")
        return self._add_button_events_seq_learning(stim_events, timeline)

    def _add_button_events_seq_learning(
        self, stim_events: pd.DataFrame, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        tl = timeline
        task_df = pd.read_csv(self._get_filename("events", timeline), sep="\t")
        task_df = self._add_trial_number(
            events_df=task_df,
            event_col="value",
            start_events=[
                "learningBlock_1",
                "learningBlock_2",
                "learningBlock_3",
                "learningBlock_4",
                "learningBlock_5",
            ],
            output_col="block",
        )
        if "user_answer" not in task_df.columns:
            logger.warning(
                f"No button responses recorded for {tl['subject']} {tl['task']}"
            )
            return stim_events

        # Responses as Keystroke events
        button_events = task_df.dropna(subset=["user_answer"])
        button_events.insert(0, "type", "Keystroke")
        button_events = button_events.rename(
            columns={
                "onset": "start",
                "user_answer": "text",
            }
        )
        button_events["button"] = 0
        button_events["is_correct"] = (
            button_events["text"] == button_events["correct_answer"]
        )
        button_events = (
            button_events[
                [
                    "type",
                    "start",
                    "block",
                    "text",
                    "button",
                    "correct_answer",
                    "is_correct",
                ]
            ]
            .dropna(subset="text")
            .reset_index(drop=True)
        )
        return (
            pd.concat([stim_events, button_events], axis=0)
            .sort_values(["start", "block"])
            .reset_index(drop=True)
        )

    def _add_separate_button_events_seq_learning(
        self, stim_events: pd.DataFrame, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        tl = timeline
        task_df = pd.read_csv(self._get_filename("events", timeline), sep="\t")
        task_df = self._add_trial_number(
            events_df=task_df,
            event_col="value",
            start_events=[
                "learningBlock_1",
                "learningBlock_2",
                "learningBlock_3",
                "learningBlock_4",
                "learningBlock_5",
            ],
            output_col="block",
        )

        if "user_answer" not in task_df.columns:
            logger.warning(
                f"No button responses recorded for {tl['subject']} {tl['task']}"
            )
            return stim_events

        # Responses as Keystroke events
        button_events = task_df.dropna(subset=["user_answer"])
        button_events.insert(0, "type", "Keystroke")
        button_events = button_events.rename(
            columns={
                "onset": "start",
                "user_answer": "text",
                "correct_answer": "correct_sequence",
            }
        )
        # Breaks up sequence of button responses
        button_events["response_sequence"] = button_events["text"]
        button_events["text"] = button_events["text"].str.split("-")
        button_events = button_events.explode("text")
        button_events["button"] = button_events["text"]
        # Breaks up sequence for correct answer
        for b in [1, 2, 3, 4, 5]:
            block_mask = button_events.block == b
            button_events.loc[block_mask, "sequence_order"] = np.arange(
                1, (block_mask).sum() + 1
            )
            correct_sequence = (
                button_events.loc[block_mask, "correct_sequence"].str.split("-").values[0]
            )
            button_events.loc[block_mask, "correct_answer"] = correct_sequence
        button_events["is_correct"] = (
            button_events["button"] == button_events["correct_answer"]
        )
        button_events["is_correct_sequence"] = (
            button_events["response_sequence"] == button_events["correct_sequence"]
        )
        button_events["text"] = button_events["text"].map(
            {
                "1": "one",
                "2": "two",
                "3": "three",
                "4": "four",
                "5": "five",
                "6": "six",
                "7": "seven",
                "8": "eight",
            }
        )
        button_events = (
            button_events[
                [
                    "type",
                    "start",
                    "block",
                    "sequence_order",
                    "text",
                    "button",
                    "correct_answer",
                    "response_sequence",
                    "correct_sequence",
                    "is_correct",
                    "is_correct_sequence",
                ]
            ]
            .dropna(subset="text")
            .reset_index(drop=True)
        )
        output = (
            pd.concat([stim_events, button_events], axis=0)
            .sort_values(["start", "block", "sequence_order"])
            .reset_index(drop=True)
        )
        return output
