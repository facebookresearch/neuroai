# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import typing as tp
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)


class Hollenstein2018Zuco(study.Study):
    """ZuCo: EEG responses to natural sentence reading.

    This dataset contains simultaneous EEG and eye-tracking recordings of
    participants reading natural English sentences across three reading tasks.
    The study provides word-level gaze and brain activity data for investigating
    language processing during naturalistic reading.

    Experimental Design:
        - EEG recordings (128-channel, 500 Hz, HydroCel GSN-128)
        - 12 participants
        - 2 sessions per participant (2-3 hours each)
        - Paradigm: natural sentence reading with eye tracking
            * Task 1 (SR): 400 sentiment-annotated sentences
            * Task 2 (NR): 650 relation-extraction sentences
            * Task 3 (TSR): task-specific relation reading
            * Stimulus language: English

    Notes:
        - Several participant-task combinations are excluded due to incomplete
          or mismatched data (see iter_timelines for details).
    """

    aliases: tp.ClassVar[tuple[str, ...]] = (
        "Zurich Cognitive Language Processing Corpus",
        "ZuCo",
    )
    licence: tp.ClassVar[str] = "CC-BY-4.0"

    bibtex: tp.ClassVar[str] = """
    @article{hollenstein2018zuco,
        title={ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading},
        volume={5},
        issn={2052-4463},
        number={1},
        journal={Scientific Data},
        publisher={Springer Science and Business Media LLC},
        author={Hollenstein,  Nora and Rotsztejn,  Jonathan and Troendle,  Marius and Pedroni,  Andreas and Zhang,  Ce and Langer,  Nicolas},
        year={2018},
        month=dec,
        doi={10.1038/sdata.2018.291}
    }

    @misc{hollenstein2022zurich,
        title={Zurich Cognitive Language Processing Corpus: A simultaneous EEG and eye-tracking resource for analyzing the human reading process},
        url={osf.io/q3zws}, DOI={10.17605/OSF.IO/Q3ZWS},
        publisher={OSF},
        author={Hollenstein, Nora and Langer, Nicolas and Pedroni, Andreas and Troendle, Marius and Rotsztejn, Jonathan and Zhang, Ce and Pfeiffer, Christian and Muttenthaler, Lukas},
        year={2022},
        month={Mar},
        doi={10.17605/OSF.IO/Q3ZWS},
        url={https://osf.io/q3zws/overview}
        }
    """
    description: tp.ClassVar[str] = (
        "EEG recordings of 12 participants reading natural text in English."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("osfclient>=0.0.5",)
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=221,
        num_subjects=12,
        num_events_in_query=1135,
        event_types_in_query={"Eeg", "Sentence", "Word"},
        data_shape=(128, 135858),
        frequency=500.0,
    )

    TASKS: tp.ClassVar[list[str]] = ["task1-SR", "task2-NR", "task3-TSR"]
    SENTENCE_ONSET_TRIGGERS: tp.ClassVar[list[int]] = [10, 12]
    SENTENCE_OFFSET_TRIGGERS: tp.ClassVar[list[int]] = [11, 13]
    # 10: sentence onset
    # 11: sentence offset
    # 12: control sentence onset
    # 13: control sentence offset
    url: tp.ClassVar[str] = "https://osf.io/q3zws/"

    def _download(self) -> None:
        with download.success_writer(self.path / "download_all") as already_done:
            if already_done:
                return
            download.Osf(study="q3zws", dset_dir=self.path, folder="download").download()

            # Cleanup directory names
            dl_dir = self.path / "download"
            for subdir in dl_dir.glob("**/*"):
                if subdir.is_dir() and ("'" in subdir.name or " " in subdir.name):
                    new_name = subdir.name.replace("'", "").replace(" ", "")
                    new_path = subdir.parent / new_name
                    subdir.rename(new_path)

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for task in self.TASKS:
            data_dir = self.path / "download" / task / "Rawdata"
            for mat_path in sorted(data_dir.glob("**/**/*_EEG.mat")):
                subject, run, _ = mat_path.stem.split("_")

                # Handle incomplete/invalid timelines
                if task == "task1-SR" and subject in ["ZDN", "ZKW"]:
                    # Multiple missing raw data files
                    continue
                if task == "task2-NR" and subject in ["ZJS", "ZPH"]:
                    # Mismatch in order of sentences between stimuli files and eye tracking events
                    continue
                if task == "task3-TSR" and subject in ["ZGW", "ZKB", "ZPH"]:
                    # Mismatch in order of sentences between stimuli files and eye tracking events
                    continue

                yield dict(
                    task=task,
                    subject=subject,
                    run=run,
                    block=int(run[-1]),
                )

    def _get_filename(
        self, filetype: str, timeline: dict[str, tp.Any]
    ) -> Path | list[Path]:
        task = timeline["task"]
        subject = timeline["subject"]
        run = timeline["run"]
        task_dir = self.path / "download" / task
        run_fixed = run.replace(
            "SNR", "SR"
        )  # Some task 1 files have a different run name for EEG

        fname: Path | list[Path]
        match filetype:
            case "eeg_file":
                fname = task_dir / "Rawdata" / subject / f"{subject}_{run}_EEG.mat"
            case "et_files":
                fname = list((task_dir / "Rawdata" / subject).glob(f"{subject}_*_ET.mat"))
                n_expected_files = {
                    "task1-SR": 8,
                    "task2-NR": 6,
                    "task3-TSR": 9,
                }
                assert len(fname) == n_expected_files[task], (
                    f"Expected {n_expected_files[task]} ET files per subject, "
                    f"but got {len(fname)}."
                )
            case "et_file":
                fname = task_dir / "Rawdata" / subject / f"{subject}_{run_fixed}_ET.mat"
            case "stim_file":
                fname = (
                    task_dir / "Matlabfiles" / f"results{subject}_{run_fixed[:-1]}.mat"
                )
            case _:
                raise ValueError(f"Unknown filetype: {filetype}")

        return fname

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg_events = pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    start=0.0,
                    filepath=info,
                )
            ]
        )

        text_events = self._get_sentence_and_fixated_word_events(timeline)
        text_events = text_events.dropna(subset="text")
        text_events = text_events[text_events.text != ""]
        text_events["language"] = "english"
        cols_to_keep = ["type", "start", "duration", "trial"]
        cols_to_keep += ["text", "sentence", "language"]
        text_events = text_events[cols_to_keep]

        events = pd.concat(
            [
                eeg_events,
                text_events,
            ]
        ).reset_index(drop=True)

        events.loc[events.type.isin(["Word", "Sentence", "Text"]), "modality"] = "read"

        return events

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.RawArray:
        """Load EEG data from .mat file."""
        eeg_file = self._get_filename("eeg_file", timeline)
        eeg = sio.loadmat(eeg_file, simplify_cells=True)["EEG"]
        eeg_data = eeg["data"]
        ch_names = [ch["labels"] for ch in eeg["chanlocs"]]
        info = mne.create_info(ch_names, eeg["srate"], "eeg")
        raw = mne.io.RawArray(eeg_data, info)
        montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
        return raw.set_montage(montage)

    def _load_eeg_sentence_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Get Sentence events from EEG file."""
        eeg_file = self._get_filename("eeg_file", timeline)
        eeg = sio.loadmat(eeg_file, simplify_cells=True, squeeze_me=True)["EEG"]
        events = pd.DataFrame(eeg["event"])
        events = events.rename(columns={"type": "trigger"})
        events["start"] = events["latency"] / eeg["srate"]  # Convert to seconds
        events["trigger"] = events["trigger"].astype(int)
        trial_start = events["trigger"].isin(self.SENTENCE_ONSET_TRIGGERS)
        trial_end = events["trigger"].isin(self.SENTENCE_OFFSET_TRIGGERS)
        output = events.reset_index(drop=True)
        output["type"] = "Sentence"
        output["block"] = int(timeline["run"][-1])
        output["duration"] = pd.NA
        output.loc[trial_start, "duration"] = (
            events.loc[trial_end, "start"].to_numpy()
            - events.loc[trial_start, "start"].to_numpy()
        )
        return (
            output.loc[trial_start, ["type", "start", "duration", "block", "trigger"]]
            .sort_values(by="start")
            .reset_index(drop=True)
        )

    def _load_eyetracking_sentence_events(
        self, fname: Path, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Get Sentence events from one eye tracking file."""
        et = sio.loadmat(fname, simplify_cells=True, squeeze_me=True)
        et_events = pd.DataFrame(et["event"], columns=["latency", "trigger"])
        trial_start = et_events.trigger.isin(self.SENTENCE_ONSET_TRIGGERS)
        trial_end = et_events.trigger.isin(self.SENTENCE_OFFSET_TRIGGERS)
        et_events.loc[trial_start, "duration"] = np.asarray(
            et_events.loc[trial_end, "latency"].to_numpy()
            - et_events.loc[trial_start, "latency"].to_numpy()
        )
        block = int(fname.stem.split("_")[-2][-1])
        et_events["block"] = block
        et_events["type"] = "Sentence"
        et_events = et_events.rename(columns={"latency": "start"})
        return (
            et_events.loc[trial_start, ["type", "start", "duration", "block", "trigger"]]
            .sort_values(by="start")
            .reset_index(drop=True)
        )

    def _load_eyetracking_task_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Load all eyetracking events for all blocks of a task, i.e. for all 8 blocks."""
        et_fnames = self._get_filename("et_files", timeline)
        assert isinstance(et_fnames, list)
        et_events = []
        for et_fname in et_fnames:
            block_events = self._load_eyetracking_sentence_events(et_fname, timeline)
            et_events.append(block_events)
        et_events_df = pd.concat(et_events, ignore_index=True)
        et_events_df = et_events_df.sort_values(by=["block", "start"]).reset_index(
            drop=True
        )
        return et_events_df

    def _load_eyetracking_data(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        """Load eye tracking data (fixations, blinks, etc.) from .mat file."""
        et_file = self._get_filename("et_file", timeline)
        et = sio.loadmat(et_file, simplify_cells=True, squeeze_me=True)

        et_dfs = []
        # Reformat .mat structure into a pandas DataFrame
        for event_type, d in et["eyeevent"].items():
            df = pd.DataFrame(d["data"], columns=d["colheader"])
            df["eye"] = d["eye"]
            df[event_type[:-1]] = True
            et_dfs.append(df)
        et_data = pd.concat(et_dfs, ignore_index=True)
        for event_type in et["eyeevent"].keys():
            et_data[event_type[:-1]] = et_data[event_type[:-1]].fillna(False)
        et_data["duration"] = et_data["endtime"] - et_data["latency"]  # fix duration
        et_data = et_data.rename(columns={"latency": "start"})
        et_data["block"] = int(timeline["run"][-1])
        return et_data

    def _load_stimuli(self, timeline: dict[str, tp.Any]) -> dict[str, list]:
        """Load stimuli information from .mat file."""
        stim_file = self._get_filename("stim_file", timeline)
        data = sio.loadmat(stim_file)["sentenceData"]

        # retrieve fixations
        fixations = []
        for sent_id, sent in enumerate(data["allFixations"][0]):
            try:
                out: dict[str, int | np.ndarray] = dict(sentence_id=sent_id)
                for k in ("duration", "pupilsize", "x", "y"):
                    out[k] = np.squeeze(sent[0][k]).flatten()[0][0]
            except IndexError:
                # In some cases, we do have sentences with empty fields (probably corrupted files)
                # E.g. timeline='Hollenstein2018Zuco_subject-ZDM_task-task2-NR_run-NR3_block-3'
                out = {
                    "sentence_id": sent_id,
                    "duration": np.array([]),
                    "pupilsize": np.array([]),
                    "x": np.array([]),
                    "y": np.array([]),
                }
            fixations.append(out)

        # retrieve sentences
        sentences = [c[0] for c in data["content"][0]]
        regex = re.compile("[^a-zA-Z0-9]")
        words = [
            [regex.sub("", w).lower() for w in sent.split(" ")] for sent in sentences
        ]
        # NOTE: We could also extract the words directly from the file:
        # words = [[w.item() for w in ws] for c in data["word"][0] for ws in c["content"]]

        # retrieve word bounds
        word_bounds = list(data["wordbounds"][0])
        for i, (words_, bounds_) in enumerate(zip(words, word_bounds)):
            if np.isnan(bounds_).all():
                # E.g. timeline='Hollenstein2018Zuco_subject-ZDN_task-task1-SR_run-SNR1_block-1'
                bounds_ = np.full((len(words_), 4), np.nan)
            assert len(words_) == bounds_.shape[0], (
                "Different number of words and bounds for sentence "
                f"{i}: {len(words_)} vs. {bounds_.shape[0]}"
            )

        assert len(fixations) == len(sentences) == len(word_bounds)
        assert len(fixations) == len(words)
        output = dict(
            fixations=fixations,
            word_bounds=word_bounds,
            words=words,
            sentences=sentences,
        )
        return output

    def _get_stimulus_fixations(
        self,
        stimuli: dict[str, tp.Any],
        et_events: pd.DataFrame,
        et_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Match eye fixations with their corresponding word and sentence."""
        fix_mask = et_data.fixation == True
        et_data.index.rename("index", inplace=True)  # To simplify merging below
        # et_words = pd.Series(index=et_data.index, dtype=object)
        for sid in et_events.index:
            # Associate eyetracking data with trial (i.e., sentence ID)
            start = et_events.loc[sid].start
            stop = start + et_events.loc[sid].duration
            sid_mask = et_data.start.between(start, stop)
            # NOTE: The last eye tracking event may go beyond the sentence end.
            et_data.loc[sid_mask, "trial"] = sid
            et_data.loc[sid_mask, "sentence"] = stimuli["sentences"][sid]
            sid_fix_mask = sid_mask & fix_mask
            if sid_fix_mask.sum() == 0:  # Missing fixations
                logger.info(
                    "No fixations found for trial %i, skipping this sentence.", sid
                )
                continue

            # Associate fixations with specific words in sentence
            sid_data = stimuli["fixations"][sid]
            if sid_data["x"].size == 0 or sid_data["y"].size == 0:
                logger.info(
                    "No valid position information for trial %i, skipping this sentence.",
                    sid,
                )
                continue
            x0, y0, x1, y1 = stimuli["word_bounds"][sid].T
            in_box = (
                (sid_data["x"][:, None] >= x0[None])
                * (sid_data["x"][:, None] <= x1[None])
                * (sid_data["y"][:, None] >= y0[None])
                * (sid_data["y"][:, None] <= y1[None])
            )
            fix_words = ["" for _ in range(in_box.shape[0])]
            words = np.array(stimuli["words"][sid])
            for i in range(in_box.shape[0]):
                n_words = in_box[i].sum()
                if n_words == 1:
                    fix_words[i] = words[in_box[i]].item()
                elif n_words > 1:
                    logger.warning(
                        f"{n_words} words found for fixation {i} in trial {sid}, "
                        "but expected 1. Ignoring this fixation."
                    )
            # We will match the fixated words with the eye tracking events based on x-position and
            # pupil size
            fixated_words = pd.DataFrame(
                {
                    "fix_avgpos_x": sid_data["x"],
                    "fix_avgpupilsize": sid_data["pupilsize"],
                    "word": fix_words,
                }
            )

            # Check mismatches in number of fixations
            n_fixations_in_matlab_data = sid_fix_mask.sum()
            n_fixations_in_stimuli_file = len(fixated_words)
            if abs(n_fixations_in_matlab_data - n_fixations_in_stimuli_file) > 1:
                msg = (
                    f"Mismatch in number of fixations for trial {sid}: "
                    f"{n_fixations_in_matlab_data} in eyetracking data vs. "
                    f"{n_fixations_in_stimuli_file} in stimuli file."
                    " Attempting to align fixations despite mismatch."
                )

                stimuli_x = sid_data["x"]
                et_x = et_data.loc[sid_fix_mask, "fix_avgpos_x"].to_numpy()
                msg += (
                    f"\nStimuli-based x-positions:\n{stimuli_x}. "
                    f"\nEyetracking-based x-positions:\n{et_x}."
                )
                logger.warning(msg)

            # Match row by row to avoid duplicates
            for _, row in fixated_words.iterrows():
                ind = (
                    (
                        et_data.loc[sid_fix_mask, ["fix_avgpos_x", "fix_avgpupilsize"]]
                        == row[["fix_avgpos_x", "fix_avgpupilsize"]]
                    )
                    .all(axis=1)
                    .idxmax()
                )
                et_data.at[ind, "text"] = row["word"]

        output = et_data.loc[fix_mask].dropna(subset="trial").dropna(axis=1, how="all")
        output = output.reset_index().rename(columns={"index": "index_et_data"})
        return output

    def _align_eyetracking_timing_to_eeg(
        self,
        eeg_events: pd.DataFrame,
        et_events: pd.DataFrame,
        et_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align eye tracking and EEG events to convert text events to EEG timing."""

        diff = len(eeg_events) - len(et_events.query("trigger>0"))
        assert diff in (0, 1)
        if diff:
            eeg_events = eeg_events.iloc[:-1]
        x = et_events.query("trigger>0").trigger.to_numpy()
        y = eeg_events.trigger.to_numpy()
        assert np.array_equal(x, y)

        # Fit regression to predict EEG timing from eye tracking timing
        x = et_events.query("trigger>0").start.to_numpy()
        y = eeg_events.start.to_numpy()
        regression = make_pipeline(RobustScaler(), HuberRegressor(max_iter=1000))
        scaler_y = RobustScaler()
        y = scaler_y.fit_transform(y[:, None])[:, 0]
        regression.fit(x[:, None], y)

        # Apply correction to eye tracking events
        et_data = et_data.rename(
            columns={
                "start": "start_eyetracking",
                "endtime": "end_eyetracking",
                "duration": "duration_eyetracking",
            }
        )
        et_start = et_data.start_eyetracking.to_numpy()
        start_pred = regression.predict(et_start[:, None])
        start_pred = scaler_y.inverse_transform(start_pred[:, None])[:, 0]
        et_data["start"] = start_pred
        et_end = et_data.end_eyetracking.to_numpy()
        end_pred = regression.predict(et_end[:, None])
        end_pred = scaler_y.inverse_transform(end_pred[:, None])[:, 0]
        et_data["end"] = end_pred
        et_data["duration"] = end_pred - start_pred
        return et_data

    def _get_sentence_and_fixated_word_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        """Get Sentence and Word events with text information obtained by aligning eye tracking events and EEG events."""
        stimuli = self._load_stimuli(timeline)
        et_events = self._load_eyetracking_task_events(timeline)
        assert len(stimuli["sentences"]) == et_events.shape[0], (
            "Different number of sentences in stimuli and eyetracking events: "
            f"{len(stimuli['sentences'])} vs. {et_events.shape[0]}"
        )
        et_events = et_events[et_events.block == timeline["block"]]
        et_data = self._load_eyetracking_data(timeline)

        eeg_events = self._load_eeg_sentence_events(timeline)
        et_data_fixs = self._get_stimulus_fixations(stimuli, et_events, et_data)
        et_data_fixs = self._align_eyetracking_timing_to_eeg(
            eeg_events, et_events, et_data_fixs
        )
        et_data_fixs.insert(0, "type", "Word")

        # Add actual sentence to the Sentence events
        for ind, row in eeg_events.iterrows():
            sentence = et_data_fixs[
                (et_data_fixs.start >= row.start)
                & (et_data_fixs.end <= (row.start + row.duration))
            ].sentence.unique()
            if len(sentence) == 0:
                logger.warning(
                    "No sentence found for EEG event at start %.3f.", row.start
                )
                continue
            assert len(sentence) == 1, (
                f"{len(sentence)} sentences found for EEG event but expected 1."
            )
            eeg_events.at[ind, "text"] = sentence[0]  # type: ignore

        events = pd.concat([eeg_events, et_data_fixs], ignore_index=True)
        events = events.sort_values(by="start").reset_index(drop=True)

        return events
