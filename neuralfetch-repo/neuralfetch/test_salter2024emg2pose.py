# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the EMG2Pose BIDS study source."""

import json
import typing as tp
from pathlib import Path
from types import SimpleNamespace

import mne
import mne_bids
import numpy as np
import pytest

from neuralfetch import download
from neuralfetch.studies.salter2024emg2pose import Salter2024Emg2pose


def _make_release(study: Salter2024Emg2pose, split_in_scans: bool = False) -> Path:
    """Write a one-recording BIDS tree plus the upstream metadata table.

    ``split_in_scans`` selects the upstream ``main`` layout, which carries
    ``split``/``generalization`` in ``scans.tsv``; tags up to v1.0.3 do not.
    """
    root = study.path / "download" / study.NEMAR_DATASET_ID
    emg_dir = root / "sub-01/ses-01/emg"
    emg_dir.mkdir(parents=True)
    bdf = emg_dir / "sub-01_ses-01_task-emg2pose_recording-left_emg.bdf"
    bdf.write_bytes(b"BDF")
    (root / "participants.tsv").write_text(
        "participant_id\toriginal_user\nsub-01\tuser-01\n"
    )
    columns, values = (
        ["stage", "side", "source_file"],
        [
            "HandClawGraspFlicks",
            "left",
            "rec-1_left.hdf5",
        ],
    )
    if split_in_scans:
        columns += ["split", "generalization"]
        values += ["train", "none"]
    (emg_dir.parent / "sub-01_ses-01_scans.tsv").write_text(
        "filename\t{}\nemg/{}\t{}\n".format(
            "\t".join(columns), bdf.name, "\t".join(values)
        )
    )
    study.metadata_path.write_text(
        "filename,split,generalization,stage,side\n"
        "rec-1_left,train,none,HandClawGraspFlicks,left\n"
    )
    return bdf


@pytest.mark.parametrize("split_in_scans", [False, True])
def test_emg2pose_timeline_carries_paper_split(
    tmp_path: Path, split_in_scans: bool
) -> None:
    study = Salter2024Emg2pose(path=tmp_path)
    _make_release(study, split_in_scans=split_in_scans)

    timeline = next(study.iter_timelines())
    events = study._load_timeline_events(timeline)

    assert timeline["split"] == "train"
    assert timeline["generalization"] == "none"
    assert timeline["user_stage"] == "user-01/HandClawGraspFlicks"
    assert events["type"].tolist() == ["Emg"]
    assert json.loads(events.iloc[0]["filepath"])["method"] == "_load_raw"

    # Fresh studies below, since the first one memoized what it read.
    study.metadata_path.unlink()
    if split_in_scans:
        assert (
            next(Salter2024Emg2pose(path=tmp_path).iter_timelines())["split"] == "train"
        )
    else:
        with pytest.raises(FileNotFoundError, match="emg2pose_metadata.csv"):
            next(Salter2024Emg2pose(path=tmp_path).iter_timelines())


def test_emg2pose_blanks_bad_ik_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = mne.create_info(["emg", "target"], sfreq=10.0, ch_types=["emg", "misc"])
    raw = mne.io.RawArray(np.ones((2, 10)), info)
    raw.set_annotations(mne.Annotations([0.0, 0.2], [0.8, 0.3], ["stage", "BAD_IK"]))
    monkeypatch.setattr(mne_bids, "read_raw_bids", lambda *args, **kwargs: raw)

    loaded = Salter2024Emg2pose(path=tmp_path)._load_raw(
        {"path": str(tmp_path / "recording_emg.bdf")}
    )

    # Cropped to the 0.8 s stage annotation, with the BAD_IK span blanked.
    np.testing.assert_array_equal(
        loaded.get_data(picks="misc")[0],
        [1.0, 1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1.0],
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        ("subject == 'Salter2024Emg2pose/13'", ["13"]),
        ("subject in ['Salter2024Emg2pose/60', 'Salter2024Emg2pose/166']", ["60", "166"]),
        # Not a subject selector: the whole release is the only safe scope.
        ("timeline_index < 8", None),
        (None, None),
    ],
)
def test_emg2pose_download_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query: str | None,
    expected: list[str] | None,
) -> None:
    captured: dict[str, tp.Any] = {}

    def eegdash(**kwargs: tp.Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(download=lambda overwrite=False: None)

    monkeypatch.setattr(download, "Eegdash", eegdash)
    monkeypatch.setattr(
        download, "download_file", lambda url, _: captured.update(url=url)
    )

    Salter2024Emg2pose(path=tmp_path, query=query)._download()

    assert captured["subject"] == expected
    assert captured["url"] == Salter2024Emg2pose.METADATA_URL
