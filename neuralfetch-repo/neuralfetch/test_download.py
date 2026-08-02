# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Offline tests for download backends — no network access required."""

import asyncio
import os
import sys
import types
import typing as tp
from pathlib import Path
from unittest.mock import patch

import pytest

from neuralfetch import download


def _concrete_subclasses() -> list[type]:
    """Return all non-abstract BaseDownload subclasses."""
    subs: list[type] = []
    queue = list(download.BaseDownload.__subclasses__())
    while queue:
        cls = queue.pop()
        if cls._can_be_instantiated():
            subs.append(cls)
        queue.extend(cls.__subclasses__())
    return subs


# Minimal kwargs needed to construct each subclass beyond study / dset_dir.
_EXTRA_KWARGS: dict[str, dict[str, tp.Any]] = {
    "S3": {"bucket": "test-bucket"},
    "Physionet": {"version": "1.0.0"},
    "Donders": {"study_id": "DSC_123"},
    "Datalad": {"repo_url": "https://example.com/repo.git"},
    "Synapse": {"study_id": "syn123"},
    "Zenodo": {"record_id": "12345"},
    "Huggingface": {"org": "abc"},
    "Dryad": {"doi": "12.3456/dryad.abcdefghijk", "token": "fake-token"},
    "Eegdash": {"database": "eegdash"},
    "Globus": {"collection_id": "00000000-0000-0000-0000-000000000000"},
}


_ENV_VARS: dict[str, dict[str, str]] = {
    "Synapse": {"NEURALFETCH_SYNAPSE_TOKEN": "fake-token"},
    "Donders": {
        "NEURALFETCH_DONDERS_USER": "fake-user",
        "NEURALFETCH_DONDERS_PASSWORD": "fake-pass",
    },
    "Globus": {
        "NEURALFETCH_GLOBUS_CLIENT_ID": "fake-client-id",
        "NEURALFETCH_GLOBUS_CLIENT_SECRET": "fake-client-secret",
    },
}


@pytest.mark.parametrize("cls", _concrete_subclasses(), ids=lambda c: c.__name__)
def test_download_backend(cls: type[download.BaseDownload], tmp_path: Path) -> None:
    """Instantiation, success-file skip, and _download dispatch — all offline."""
    extra = _EXTRA_KWARGS.get(cls.__name__, {})
    env = _ENV_VARS.get(cls.__name__, {})
    with patch.dict(os.environ, env):
        obj = cls(study="test-study", dset_dir=tmp_path / "study", **extra)
    assert obj._dl_dir.exists()
    # first call: _download is invoked and success file is written
    with (
        patch.object(obj, "_download") as mock_dl,
        patch.object(cls, "_check_requirements"),
    ):
        obj.download()
    mock_dl.assert_called_once()
    assert obj.get_success_file().exists()
    # second call: skipped because success file exists
    with patch.object(obj, "_download") as mock_dl:
        obj.download()
    mock_dl.assert_not_called()


@pytest.mark.parametrize(
    "suffix,success_msg", [("_success.txt", "done"), ("_done.tmp", "yep")]
)
def test_success_writer(tmp_path: Path, suffix: str, success_msg: str) -> None:
    fname = tmp_path / "test.txt"
    success_fname = tmp_path / ("test" + suffix)

    # Run once
    with download.success_writer(fname, suffix, success_msg) as success:
        assert not success

    assert success_fname.exists()
    with success_fname.open() as f:
        out = f.read()
    assert out == success_msg

    # Run a second time
    with download.success_writer(fname, suffix, success_msg) as success:
        assert success


def test_temp_mne_data_uses_env_only(tmp_path: Path) -> None:
    """temp_mne_data sets/restores MNE_DATA via env vars only."""
    old_val = os.environ.get("MNE_DATA")
    data_dir = tmp_path / "mne_data"
    data_dir.mkdir()

    with download.temp_mne_data(data_dir):
        assert os.environ["MNE_DATA"] == str(data_dir)

    assert os.environ.get("MNE_DATA") == old_val


def test_temp_mne_data_restores_on_exception(tmp_path: Path) -> None:
    """temp_mne_data restores env vars even if the body raises."""
    old_val = os.environ.get("MNE_DATA")
    data_dir = tmp_path / "mne_data"
    data_dir.mkdir()

    with pytest.raises(ValueError):
        with download.temp_mne_data(data_dir):
            raise ValueError("boom")

    assert os.environ.get("MNE_DATA") == old_val


def test_temp_mne_data_missing_path(tmp_path: Path) -> None:
    """temp_mne_data raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        with download.temp_mne_data(tmp_path / "does_not_exist"):
            pass


def test_synapse_missing_token_raises(tmp_path: Path) -> None:
    """Synapse raises RuntimeError at construction when token env var is missing."""
    env = {k: v for k, v in os.environ.items() if k != "NEURALFETCH_SYNAPSE_TOKEN"}
    with (
        patch.dict(os.environ, env, clear=True),
        pytest.raises(RuntimeError, match="auth_token is required"),
    ):
        download.Synapse(study="test", study_id="syn123", dset_dir=tmp_path / "syn")


def test_donders_missing_credentials_raises(tmp_path: Path) -> None:
    """Donders raises RuntimeError at construction when env vars are missing."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("NEURALFETCH_DONDERS_USER", "NEURALFETCH_DONDERS_PASSWORD")
    }
    with (
        patch.dict(os.environ, env, clear=True),
        pytest.raises(RuntimeError, match="Donders requires user and password"),
    ):
        download.Donders(study="test", study_id="DSC_123", dset_dir=tmp_path / "donders")


def test_dryad_missing_token_raises(tmp_path: Path) -> None:
    """Dryad raises RuntimeError at construction when no token is available."""
    env = {k: v for k, v in os.environ.items() if k != "NEURALFETCH_DRYAD_TOKEN"}
    with (
        patch.dict(os.environ, env, clear=True),
        pytest.raises(RuntimeError, match="Dryad API token is required"),
    ):
        download.Dryad(
            study="test",
            doi="12.3456/dryad.abcdefghijk",
            dset_dir=tmp_path / "dryad",
        )


def test_globus_missing_credentials_raises(tmp_path: Path) -> None:
    """Globus raises RuntimeError at construction when credentials env vars are missing."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("NEURALFETCH_GLOBUS_CLIENT_ID", "NEURALFETCH_GLOBUS_CLIENT_SECRET")
    }
    with (
        patch.dict(os.environ, env, clear=True),
        pytest.raises(RuntimeError, match="service-account credentials are required"),
    ):
        download.Globus(
            study="test",
            dset_dir=tmp_path / "globus",
            collection_id="00000000-0000-0000-0000-000000000000",
        )


def test_physionet_preserves_study_version_structure(tmp_path: Path) -> None:
    """Physionet sets prefix/output_dir and writes under download/<study>/<version>/."""
    physionet = download.Physionet(
        study="eegmat",
        version="1.0.0",
        dset_dir=tmp_path / "study",
    )

    def mock_s3_download(self: download.S3) -> None:
        assert self.prefix == "eegmat/1.0.0"
        assert self.output_dir == self._dl_dir / "eegmat" / "1.0.0"
        out_dir = tp.cast(Path, self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "file1.txt").write_text("data1")
        (out_dir / "file2.txt").write_text("data2")

    with patch.object(download.S3, "_download", mock_s3_download):
        physionet._download()

    out_root = tmp_path / "study" / "download" / "eegmat" / "1.0.0"
    for name, expected in [("file1.txt", "data1"), ("file2.txt", "data2")]:
        assert (out_root / name).read_text("utf8") == expected


@pytest.mark.parametrize("study_name", ["Allen2022Massive", "Allen2022MassiveRaw"])
def test_nsd_data_access_agreement(tmp_path: Path, study_name: str) -> None:
    """NSD consent flow: env-var gate, T&C display, user info collection, marker persistence.

    Both Allen2022Massive and Allen2022MassiveRaw share the same consent
    implementation (Raw subclasses Massive), so the parametrized matrix
    exercises both entry points.
    """
    from neuralfetch.studies import allen2022massive

    study_cls = getattr(allen2022massive, study_name)

    study_path = tmp_path / study_name
    study_path.mkdir()
    bold = study_cls(path=study_path)
    marker = tmp_path / "neuralfetch" / ".nsd_tcs_accepted"

    with patch(
        "neuralfetch.studies.allen2022massive.get_nsd_tcs_marker", return_value=marker
    ):
        # -- Without NSD_ACCEPT_LICENCE the gate raises before any prompt --
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NSD_ACCEPT_LICENCE", None)
            with patch("builtins.input") as mock_input:
                with pytest.raises(PermissionError, match="NSD_ACCEPT_LICENCE"):
                    bold._check_nsd_data_access_agreement()
            mock_input.assert_not_called()
        assert not marker.exists()

        with (
            patch.dict(os.environ, {"NSD_ACCEPT_LICENCE": "1"}),
            patch("neuralfetch.studies.allen2022massive.sys") as mock_sys,
        ):
            mock_sys.stdin.isatty.return_value = True

            # -- Declining T&C raises PermissionError --
            with patch("builtins.input", return_value="n"):
                with pytest.raises(PermissionError, match="Terms and Conditions"):
                    bold._check_nsd_data_access_agreement()
            assert not marker.exists()

            # -- Empty required fields raise ValueError --
            for empty_at in range(1, 5):
                # Sequence: agree, name, email, department, institution
                answers = ["y", "John Doe", "john@example.com", "Department", "Institute"]
                answers[empty_at] = ""  # leave one field empty
                with patch("builtins.input", side_effect=answers):
                    with pytest.raises(ValueError, match="required"):
                        bold._check_nsd_data_access_agreement()
            assert not marker.exists()

            # -- Invalid role selection raises ValueError --
            valid_fields = [
                "y",
                "John Doe",
                "john@example.com",
                "Department",
                "Institute",
            ]
            for bad_role in ["0", "-1", "99", "abc", ""]:
                with patch("builtins.input", side_effect=[*valid_fields, bad_role]):
                    with pytest.raises(ValueError, match="Invalid role selection"):
                        bold._check_nsd_data_access_agreement()
            assert not marker.exists()

            # -- Declining form submission raises PermissionError --
            inputs_declined = [*valid_fields, "4", "n"]
            with (
                patch("builtins.input", side_effect=inputs_declined),
                patch("webbrowser.open", return_value=True),
            ):
                with pytest.raises(PermissionError, match="submit"):
                    bold._check_nsd_data_access_agreement()
            assert not marker.exists()

            # -- Successful flow with a standard role --
            inputs_standard = [*valid_fields, "4", "y"]
            with (
                patch("builtins.input", side_effect=inputs_standard),
                patch("webbrowser.open", return_value=True) as mock_browser,
            ):
                bold._check_nsd_data_access_agreement()
            assert marker.exists()
            url = mock_browser.call_args[0][0]
            assert "entry.1976545571=John+Doe" in url
            assert "john%40example.com" in url
            assert "entry.443953373=Faculty" in url
            assert "__other_option__" not in url

            # -- Marker persistence: second call on Bold skips --
            with patch("builtins.input") as mock_input:
                bold._check_nsd_data_access_agreement()
            mock_input.assert_not_called()

            # -- "Other" role uses __other_option__ in URL --
            marker.unlink()
            inputs_other = [*valid_fields, "5", "Researcher", "y"]
            with (
                patch("builtins.input", side_effect=inputs_other),
                patch("webbrowser.open", return_value=True) as mock_browser,
            ):
                bold._check_nsd_data_access_agreement()
            url = mock_browser.call_args[0][0]
            assert "__other_option__" in url
            assert "Researcher" in url


def _fake_openneuro_module() -> types.ModuleType:
    """Stand-in ``openneuro`` module mirroring openneuro-py's event-loop branch.

    Under a running loop it fires the transfer via ``create_task`` and returns
    immediately (the Jupyter footgun); with no loop it blocks via ``asyncio.run``.
    """

    def download(  # noqa: ANN001, ANN201
        *, dataset, target_dir, include=None, max_concurrent_downloads=5
    ):
        target = Path(target_dir)

        async def _coro() -> None:
            target.mkdir(parents=True, exist_ok=True)
            (target / "participants.tsv").write_text("ok")

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_coro())  # running loop: scheduled, never awaited
        except RuntimeError:
            asyncio.run(_coro())  # no running loop: blocks until written

    mod = types.ModuleType("openneuro")
    mod.download = download  # type: ignore[attr-defined]
    return mod


def test_openneuro_download_blocks_under_running_loop(tmp_path: Path) -> None:
    """``Openneuro._download`` completes synchronously even under a running loop.

    In a notebook a loop is already running, so openneuro-py's bare ``download``
    would fire-and-forget and return before any file lands. The worker-thread
    wrapper must force openneuro's blocking branch so the directory is populated
    by the time ``_download`` returns.
    """
    client = download.Openneuro(folder="", study="ds999999", dset_dir=tmp_path / "study")
    marker = client._dl_dir / "participants.tsv"

    async def _in_notebook() -> bool:
        # Mimics a synchronous notebook cell executing under the kernel's loop.
        client._download()
        return marker.exists()

    with patch.dict(sys.modules, {"openneuro": _fake_openneuro_module()}):
        completed = asyncio.run(_in_notebook())

    assert completed, (
        "openneuro download did not finish before _download() returned under a "
        "running event loop — the worker-thread wrapper is not blocking"
    )


def test_openneuro_download_propagates_worker_errors(tmp_path: Path) -> None:
    """An error inside the worker thread is re-raised on the caller."""
    client = download.Openneuro(folder="", study="ds999999", dset_dir=tmp_path / "study")

    def _boom(**kwargs: tp.Any) -> None:
        raise RuntimeError("download failed")

    fake = types.ModuleType("openneuro")
    fake.download = _boom  # type: ignore[attr-defined]
    with (
        patch.dict(sys.modules, {"openneuro": fake}),
        pytest.raises(RuntimeError, match="download failed"),
    ):
        client._download()
