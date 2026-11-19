# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import pytest

from neuralfetch import download, utils
from neuralfetch.studies.hebart2023things import Hebart2023ThingsMeg


def test_meg_download_builds_the_openneuro_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real Openneuro model, so a call it rejects (e.g. positional) fails here.
    captured: dict[str, tp.Any] = {}
    monkeypatch.setattr(utils, "download_things_images", lambda path: None)
    monkeypatch.setattr(
        download.Openneuro,
        "download",
        lambda self, overwrite=False: captured.update(
            study=self.study, dset_dir=Path(self.dset_dir)
        ),
    )
    study = Hebart2023ThingsMeg(path=tmp_path)
    study._download()
    assert captured == {"study": "ds004212", "dset_dir": Path(study.path)}
