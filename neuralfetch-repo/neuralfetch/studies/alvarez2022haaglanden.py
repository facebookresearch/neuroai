# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import mne
import pandas as pd

from neuralfetch import download
from neuralset.events import study

logger = logging.getLogger(__name__)


class Alvarez2022Haaglanden(study.Study):
    url: tp.ClassVar[str] = "https://physionet.org/content/hmc-sleep-staging/1.1/"
    """Haaglanden Medisch Centrum Sleep Staging Database (HMC-Sleep-Staging): EEG recordings during whole-night polysomnographic sleep monitoring.

    A collection of 151 whole-night polysomnographic sleep recordings collected at the Haaglanden
    Medisch Centrum sleep center in The Netherlands. Each recording includes 6-channel EEG with
    sleep stage annotations (W, N1, N2, N3, R). The population consists of clinical sleep center
    patients (85 male, 66 female, mean age 53.9 ± 15.4).

    Experimental Design:
        - EEG recordings (6-channel, 256 Hz)
        - 151 participants (clinical sleep center patients)
        - 1 session per participant (whole-night recording)
        - Paradigm: whole-night sleep staging with annotated sleep stages (W, N1, N2, N3, R)
    """

    aliases: tp.ClassVar[tuple[str, ...]] = (
        "Haaglanden Medisch Centrum Sleep Staging Database",
        "HMC-Sleep-Staging",
    )

    bibtex: tp.ClassVar[str] = """
    @misc{alvarez2022haaglanden,
        title={Haaglanden {{Medisch Centrum}} Sleep Staging Database},
        author={{Alvarez-Estevez}, Diego and Rijsman, Roselyne},
        year=2022,
        month=mar,
        publisher={PhysioNet},
        doi={10.13026/T79Q-FR32},
        url={https://physionet.org/content/hmc-sleep-staging/1.1/}
    }
    """
    licence: tp.ClassVar[str] = "CC-BY-4.0"
    description: tp.ClassVar[str] = """
    EEG sleep recordings from 151 whole-night polysomnographic (PSG) sessions (85 male, 66 female, mean age of 53.9 ± 15.4)
    collected during 2018 at the Haaglanden Medisch Centrum (HMC, The Netherlands) sleep center.
    """
    _MISSING: list[int] = [14, 64, 135]
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=151,
        num_subjects=151,
        event_types_in_query={"Eeg", "SleepStage", "CategoricalEvent"},
        num_events_in_query=103,
        data_shape=(6, 6566400),
        frequency=256,
    )

    def _download(self, overwrite=False) -> None:
        self.path.mkdir(exist_ok=True, parents=True)
        physionet = download.Physionet(
            study="hmc-sleep-staging",
            dset_dir=self.path,
            bucket="physionet-open",
            version="1.1",
        )
        if physionet.get_success_file().exists() and not overwrite:
            return
        physionet.download(overwrite=overwrite)

    @staticmethod
    def _fix_channels(raw: mne.io.Raw) -> mne.io.Raw:
        """
        Rename bipolar EEG channels to monopolar names and add EEG montage.
        """
        rename_map = {
            "EEG F4-M1": "F4",
            "EEG C4-M1": "C4",
            "EEG O2-M1": "O2",
            "EEG C3-M2": "C3",
            "EMG chin": "EMG",
            "EOG E1-M2": "EOG1",
            "EOG E2-M2": "EOG2",
        }
        raw.rename_channels(rename_map)
        raw.set_channel_types({"EOG1": "eog", "EOG2": "eog"})

        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="ignore")

        return raw

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        subjects = set(range(1, 155)).difference(self._MISSING)
        for subject in subjects:
            sub_id = f"SN{subject:03d}"
            yield dict(subject=sub_id)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        sub_dir = self.path / "download" / "recordings"
        fname_events = sub_dir / f"{timeline['subject']}_sleepscoring.edf"
        annots_df = mne.read_annotations(fname_events).to_data_frame(time_format=None)
        annots_df.rename(columns={"onset": "start"}, inplace=True)

        annots_df["stage"] = annots_df["description"].replace(
            {
                "Sleep stage W": "W",
                "Sleep stage N1": "N1",
                "Sleep stage N2": "N2",
                "Sleep stage N3": "N3",
                "Sleep stage N4": "N3",
                "Sleep stage R": "R",
                "Sleep stage ?": pd.NA,
                "Movement time": pd.NA,
            }
        )

        annots_df = annots_df.dropna(subset=["stage"], how="all", axis=0)
        annots_df.insert(0, "type", "")
        annots_df.loc[annots_df.stage.isin(["W", "N1", "N2", "N3", "R"]), "type"] = (
            "SleepStage"
        )
        idx = annots_df.stage.str.contains("Lights")
        annots_df.loc[idx, "type"] = "CategoricalEvent"
        annots_df.loc[idx, "duration"] = 1e-3

        grouping = annots_df["stage"].ne(annots_df["stage"].shift()).cumsum()
        grouped = (
            annots_df.groupby(grouping)
            .agg(
                {
                    "start": "first",
                    "duration": "sum",
                    "type": "first",
                    "description": "first",
                    "stage": "first",
                }
            )
            .reset_index(drop=True)
        )

        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        eeg = dict(type="Eeg", filepath=info, start=0)
        eeg_events = pd.concat([pd.DataFrame([eeg]), grouped])

        return eeg_events

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.BaseRaw:
        sub_dir = self.path / "download" / "recordings"
        fname_eeg = sub_dir / f"{timeline['subject']}.edf"

        raw = mne.io.read_raw_edf(fname_eeg)
        # Fix channel names, types, and add montage
        raw = self._fix_channels(raw)

        return raw
