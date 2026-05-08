# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NM000104 (CTRL-Labs emg2qwerty) study source for neuralbench."""

from __future__ import annotations

import logging
import re
import typing as tp
from pathlib import Path

import pandas as pd
import pydantic

from neuralset.events import etypes, study

LOGGER = logging.getLogger(__name__)
_BIDS_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class Emg2qwertyRaw(etypes.Emg):
    """NM000104 EMG event — coerces BDF channel types to ``emg`` and rescales
    volts → microvolts to match the upstream HDF5 storage units.

    Why both fixes are needed:

    * BDF headers don't carry channel-type info; only the BIDS
      ``channels.tsv`` sidecar does, and ``mne.io.read_raw`` ignores
      sidecars.  Without coercion, ``EmgExtractor(picks=('emg',))`` finds
      no channels.
    * The pretrained checkpoints' ``_SpectrogramNorm`` BatchNorm was fit on
      microvolt-scale inputs.  Volts (~1e-5) drive the log10-power STFT to
      its ``log_eps=1e-6`` floor (-6.0), giving the model a constant input;
      ×1e6 gives log-spec mean ≈ 0.2 matching the ckpt's running_mean.
    """

    BDF_TO_MICROVOLT_SCALE: tp.ClassVar[float] = 1e6

    def _read(self) -> tp.Any:
        raw = super()._read().copy()
        raw.set_channel_types({ch: "emg" for ch in raw.ch_names}, verbose=False)
        raw.load_data(verbose=False)
        # Explicit picks: after set_channel_types(..."emg"), MNE's default
        # ``picks="data"`` excludes EMG channels and apply_function raises.
        raw.apply_function(
            lambda x: x * self.BDF_TO_MICROVOLT_SCALE,
            picks=raw.ch_names, channel_wise=False, verbose=False,
        )
        return raw


class Emg2qwerty(study.Study):
    """emg2qwerty (CTRL-Labs, NeurIPS 2024) — surface-EMG keystroke decoding."""

    url: tp.ClassVar[str] = (
        "https://proceedings.neurips.cc/paper_files/paper/2024/file/"
        "a64d53074d011e49af1dfc72c332fe4b-Paper-Datasets_and_Benchmarks_Track.pdf"
    )
    bibtex: tp.ClassVar[str] = (
        "@inproceedings{NEURIPS2024_a64d5307,\n"
        "  author={Sivakumar, Viswanath and Seely, Jeffrey and Du, Alan and\n"
        "          Bittner, Sean R and Berenzweig, Adam and Bolarinwa,\n"
        "          Anuoluwapo and Gramfort, Alexandre and Mandel, Michael I},\n"
        "  title={emg2qwerty: A Large Dataset with Baselines for Touch Typing\n"
        "         using Surface Electromyography},\n"
        "  booktitle={Advances in Neural Information Processing Systems},\n"
        "  editor={A. Globerson and L. Mackey and D. Belgrave and A. Fan and\n"
        "          U. Paquet and J. Tomczak and C. Zhang},\n"
        "  pages={91373--91389},\n"
        "  publisher={Curran Associates, Inc.},\n"
        "  doi={10.52202/079017-2899},\n"
        "  volume={37},\n"
        "  year={2024}}"
    )
    description: tp.ClassVar[str] = (
        "108 subjects doing surface typing with an EMG wristband on each arm."
    )
    aliases: tp.ClassVar[tuple[str, ...]] = ("emg2qwerty", "nm000104")

    # NM000104 is mirrored on NEMAR (s3://nemar/nm000104) and indexed in the
    # eegdash database; ``neuralfetch.download.Eegdash`` handles record
    # discovery + parallel S3 transfer.  ~239 GB / 1136 files for the full
    # 108-subject release.
    NEMAR_DATASET_ID: tp.ClassVar[str] = "nm000104"

    _bids_root_cache: Path | None = pydantic.PrivateAttr(default=None)

    def _download(self) -> None:
        from neuralfetch import download

        download.Eegdash(study=self.NEMAR_DATASET_ID, dset_dir=self.path).download()

    def _bids_root(self) -> Path:
        # ``Study.download`` puts files at ``self.path / "download" / ...``;
        # users with a manual BIDS tree placed directly under ``self.path``
        # also work — pick whichever has the BIDS layout.  Result is cached
        # on first call to keep ``_bids_paths`` (called per-timeline) cheap.
        if self._bids_root_cache is not None:
            return self._bids_root_cache
        root = Path(self.path)
        if not any(root.glob("sub-*/ses-*/emg")) and (root / "download").is_dir():
            root = root / "download"
        self._bids_root_cache = root
        return root

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        # Glob the BIDS layout — discovery feeds cache-uid hashing and must
        # stay fast / dependency-light (no mne_bids).
        for emg_dir in sorted(self._bids_root().glob("sub-*/ses-*/emg")):
            subject = emg_dir.parent.parent.name.removeprefix("sub-")
            session = emg_dir.parent.name.removeprefix("ses-")
            bdf = emg_dir / f"sub-{subject}_ses-{session}_task-typing_emg.bdf"
            if bdf.exists():
                yield {"subject": subject, "session": session}

    def _bids_paths(self, subject: str, session: str) -> tuple[Path, Path]:
        if not _BIDS_ID_RE.match(subject) or not _BIDS_ID_RE.match(session):
            raise ValueError(
                f"unsafe BIDS id: subject={subject!r} session={session!r}"
            )
        emg_dir = self._bids_root() / f"sub-{subject}" / f"ses-{session}" / "emg"
        stem = f"sub-{subject}_ses-{session}_task-typing"
        return emg_dir / f"{stem}_emg.bdf", emg_dir / f"{stem}_events.tsv"

    def _load_timeline_events(
        self, timeline: dict[str, tp.Any]
    ) -> pd.DataFrame:
        subject, session = timeline["subject"], timeline["session"]
        bdf_path, events_path = self._bids_paths(subject, session)
        events = pd.read_csv(events_path, sep="\t")
        events["start"] = events["onset"].astype(float)
        value = events["value"].astype(str)

        rows: list[dict[str, tp.Any]] = [{
            "type": "Emg2qwertyRaw", "filepath": str(bdf_path),
            "start": 0.0, "subject": subject,
        }]

        for _, p in events[value == "prompt"].iterrows():
            text = p.get("prompt_text", "")
            if not isinstance(text, str):
                continue
            # NM000104 prompt_text often ends with a literal "\n"
            # (backslash + 'n', two characters) — strip that suffix
            # explicitly.  ``str.rstrip`` would treat "\\n" as a
            # character set and chew off any trailing 'n' or '\'.
            if text.endswith("\\n"):
                text = text[:-2]
            text = text.strip()
            if text:
                rows.append({
                    "type": "Sentence", "start": float(p["start"]),
                    "duration": float(p["duration"]), "text": text,
                    "language": "en",
                })

        keys = events.loc[
            value.str.startswith("keystroke_"), ["start", "duration", "key"]
        ].copy()
        keys["key"] = keys["key"].astype(str).str.strip()
        for _, k in keys[keys["key"] != ""].iterrows():
            rows.append({
                "type": "Keystroke", "start": float(k["start"]),
                "duration": float(k["duration"]) if pd.notna(k["duration"]) else 0.0,
                "text": k["key"], "language": "en",
            })

        return pd.DataFrame(rows).sort_values(by="start").reset_index(drop=True)
