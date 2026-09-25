"""Muse sleep-onset EEG, NEMAR nm000287."""

import typing as tp
from pathlib import Path

import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from neuralfetch import download
from neuralset.events import study


class Interaxon2026Muse(study.Study):
    """At-home Muse S family EEG with first-N2 point annotations.

    Version 1.0.0 contains 540 recordings from 203 participants, totaling
    approximately 157.52 hours. Four channels (TP9, AF7, AF8, TP10) are stored
    at 128 Hz in EEG-BIDS/BrainVision format.

    Notes
    -----
    - Session tables supply 500 train and 40 test recordings. Every test
      participant also appears in training; this is not the sealed competition
      cohort. The loader preserves these labels; benchmark split transforms
      may replace them in memory.
    - Each recording has one zero-duration first-N2 annotation, not stable N2
      or a full hypnogram. Onset is relative to recording start, not necessarily
      lights out. The scoring method is undocumented.
    - Every recording ends 300 seconds after N2. Total length, annotations,
      future EEG and whole-recording quality summaries must stay outside
      model inputs. Sequential batches alone do not make preprocessing causal.
    - MNE-BIDS applies header unit scaling. This loader does not filter,
      resample, clean artifacts or exclude recordings using quality flags.
    - Exact Muse S generation, firmware, reference and prior filters are
      unconfirmed. Downsampling from 256 Hz is a curator assumption, not a
      verified acquisition fact. Demographics and acquisition dates are absent.
    - Shared electrode coordinates have unknown provenance and should not be
      treated as participant-specific measurements. Quality flags describe
      signal screening, not independently confirmed artifacts.
    - The deposit records consent and sharing authorization, an internal Muse
      exemption determination, and destruction of the re-identification key.
      Credit Muse Team under CC-BY-NC-SA-4.0.
    - The NEMAR download is pinned to 1.0.0. An existing BIDS tree directly
      under the study directory is also supported.
    """

    licence: tp.ClassVar[str] = "CC-BY-NC-SA-4.0"
    url: tp.ClassVar[str] = "https://doi.org/10.82901/nemar.nm000287"
    aliases: tp.ClassVar[tuple[str, ...]] = ("muse", "nm000287")
    bibtex: tp.ClassVar[str] = """
    @misc{muse2026sleeponset,
        author = {{Muse Team}},
        title = {Muse Sleep-Onset EEG — EEG/EMG Foundation Challenge 2026, Track 03},
        year = {2026},
        publisher = {NEMAR},
        version = {1.0.0},
        doi = {10.82901/nemar.nm000287},
        url = {https://doi.org/10.82901/nemar.nm000287},
    }
    """
    description: tp.ClassVar[str] = (
        "Muse Team: 540 at-home recordings from 203 participants; Muse S family "
        "EEG, four channels at 128 Hz, with first-N2 point annotations. "
        "500 train / 40 seen-participant test sessions; no full hypnograms."
    )
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=540,
        num_subjects=203,
        num_events_in_query=2,
        event_types_in_query={"Eeg", "SleepStage"},
        data_shape=(4, 122880),
        frequency=128,
    )

    def _download(self, overwrite: bool = False) -> None:
        download.Nemar(
            study="nm000287",
            dset_dir=self.path,
            version="1.0.0",
        ).download(overwrite=overwrite)

    @property
    def bids_root(self) -> Path:
        if any(self.path.glob("sub-*/sub-*_sessions.tsv")):
            return self.path
        return self.path / "download" / "nm000287"

    def iter_timelines(self):
        files = sorted(self.bids_root.glob("sub-*/sub-*_sessions.tsv"))
        if not files:
            raise FileNotFoundError(
                f"No BIDS session tables in {self.bids_root}; run study.download() first"
            )
        for path in files:
            for row in pd.read_csv(path, sep="\t").itertuples():
                if row.split not in {"train", "test"}:
                    raise ValueError(f"Unknown split in {path}: {row.split}")
                yield dict(subject=path.parent.name, session=row.session_id)

    def _bids_path(self, timeline):
        return BIDSPath(
            root=self.bids_root,
            subject=timeline["subject"].removeprefix("sub-"),
            session=timeline["session"].removeprefix("ses-"),
            task="sleeponset",
            datatype="eeg",
            suffix="eeg",
            extension=".vhdr",
        )

    def _load_raw(self, timeline):
        return read_raw_bids(self._bids_path(timeline), verbose="ERROR")

    def _load_timeline_events(self, timeline):
        raw = self._load_raw(timeline)
        onset = raw.annotations.onset[raw.annotations.description == "n2_onset"].item()
        duration = raw.n_times / raw.info["sfreq"]
        if not 0 <= onset <= duration:
            raise ValueError(f"N2 onset outside recording: {timeline}")
        sessions = pd.read_csv(
            self.bids_root / timeline["subject"] / f"{timeline['subject']}_sessions.tsv",
            sep="\t",
        ).set_index("session_id")
        split = sessions.loc[timeline["session"], "split"]
        return pd.DataFrame(
            [
                dict(
                    type="Eeg",
                    start=0.0,
                    duration=duration,
                    filepath=study.SpecialLoader(
                        method=self._load_raw, timeline=timeline
                    ).to_json(),
                ),
                dict(
                    type="SleepStage",
                    start=onset,
                    duration=0.0,
                    stage="N2",
                ),
            ]
        ).assign(split=split)
