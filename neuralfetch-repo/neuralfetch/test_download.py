# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Offline tests for download backends — no network access required."""

import os
import sys
import types
import typing as tp
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    "Gin": {"repo_url": "https://gin.g-node.org/test/repo.git"},
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


def _make_gin(tmp_path: Path, **kwargs: tp.Any) -> download.Gin:
    """Factory that builds a Gin instance rooted at *tmp_path*."""
    defaults: dict[str, tp.Any] = {
        "study": "test",
        "dset_dir": tmp_path / "study",
        "repo_url": "https://gin.g-node.org/CUBRIC/WAND.git",
    }
    defaults.update(kwargs)
    return download.Gin(**defaults)


def test_gin_https_base_strips_dot_git_and_appends_branch(tmp_path: Path) -> None:
    """`_https_base` derives the GIN raw URL prefix from `repo_url` + `branch`."""
    gin_master = _make_gin(tmp_path)
    assert gin_master._https_base == "https://gin.g-node.org/CUBRIC/WAND/raw/master"
    gin_main = _make_gin(tmp_path, branch="main")
    assert gin_main._https_base == "https://gin.g-node.org/CUBRIC/WAND/raw/main"
    # `.git` suffix is optional in the input
    gin_bare = _make_gin(tmp_path, repo_url="https://gin.g-node.org/CUBRIC/WAND")
    assert gin_bare._https_base == "https://gin.g-node.org/CUBRIC/WAND/raw/master"


def test_gin_read_pointer_key_unlocked_pointer(tmp_path: Path) -> None:
    """An unlocked in-tree pointer file parses to its annex key."""
    key = "MD5-s976320008--eea09d82b05cc6d6edb5b147f8579575"
    pointer = tmp_path / "sub-001.meg4"
    pointer.write_text(f"/annex/objects/{key}\n")
    assert download.Gin._read_pointer_key(pointer) == key


def test_gin_read_pointer_key_locked_symlink(tmp_path: Path) -> None:
    """A locked symlink into ``.git/annex/objects/...`` resolves to its key."""
    key = "SHA256E-s100--abc"
    target = f"../../../.git/annex/objects/Wp/g8/{key}/{key}"
    symlink = tmp_path / "sub-001.res4"
    symlink.symlink_to(target)
    assert download.Gin._read_pointer_key(symlink) == key


def test_gin_read_pointer_key_rejects_non_pointers(tmp_path: Path) -> None:
    """Regular files, oversize ASCII, and non-annex symlinks return None."""
    real = tmp_path / "real.bin"
    real.write_bytes(b"\x00\x01\x02real binary data")
    assert download.Gin._read_pointer_key(real) is None

    big = tmp_path / "oversize.txt"
    big.write_bytes(b"/annex/objects/foo\n" + b"x" * 1024)
    assert download.Gin._read_pointer_key(big) is None

    elsewhere = tmp_path / "elsewhere.lnk"
    elsewhere.symlink_to("../somewhere/else.bin")
    assert download.Gin._read_pointer_key(elsewhere) is None


def test_gin_download_invokes_clone_register_and_get(tmp_path: Path) -> None:
    """End-to-end ``Gin._download`` wires clone -> registerurl -> annex get.

    ``datalad.api.clone`` is stubbed to materialise the WAND-like directory
    structure with one pointer file, and datalad's ``AnnexRepo`` is mocked, so
    the test stays offline and does not require ``datalad`` to be installed.
    """
    gin = _make_gin(
        tmp_path,
        include=["sub-*/ses-01/meg"],
        threads=3,
    )

    pointer_rel = Path("sub-00395/ses-01/meg/sub-00395_ses-01_task-resting.meg4")
    repo_root = gin._dl_dir / gin.repo_name
    pointer_abs = repo_root / pointer_rel
    annex_key = "MD5-s976320008--eea09d82b05cc6d6edb5b147f8579575"

    def fake_clone(source: str, path: tp.Any) -> tp.Any:
        # `datalad clone` populates the working tree; emulate that by writing
        # the pointer file under the expected repo root.
        Path(path).mkdir(parents=True, exist_ok=True)
        pointer_abs.parent.mkdir(parents=True, exist_ok=True)
        pointer_abs.write_text(f"/annex/objects/{annex_key}\n")
        return MagicMock()

    annex = MagicMock()
    clone_mock = MagicMock(side_effect=fake_clone)
    annex_cls = MagicMock(return_value=annex)

    # Inject fake ``datalad`` modules so the test runs whether or not datalad is
    # installed (CI does not install it). ``_download`` does
    # ``import datalad.api as dlad`` (resolved via ``datalad.api``) and
    # ``from datalad.support.annexrepo import AnnexRepo``.
    fake_api = types.ModuleType("datalad.api")
    fake_api.clone = clone_mock  # type: ignore[attr-defined]
    fake_datalad = types.ModuleType("datalad")
    fake_datalad.api = fake_api  # type: ignore[attr-defined]
    fake_annexrepo = types.ModuleType("datalad.support.annexrepo")
    fake_annexrepo.AnnexRepo = annex_cls  # type: ignore[attr-defined]
    fake_support = types.ModuleType("datalad.support")
    fake_support.annexrepo = fake_annexrepo  # type: ignore[attr-defined]
    fake_modules = {
        "datalad": fake_datalad,
        "datalad.api": fake_api,
        "datalad.support": fake_support,
        "datalad.support.annexrepo": fake_annexrepo,
    }

    with patch.dict(sys.modules, fake_modules):
        gin._download()

    # 0. clone was invoked into download/<repo_name>/, AnnexRepo built on it
    clone_mock.assert_called_once()
    assert clone_mock.call_args.kwargs["path"] == repo_root
    annex_cls.assert_called_once_with(str(repo_root))

    # 1. registerurl invoked with key + correct GIN HTTPS URL
    expected_url = (
        "https://gin.g-node.org/CUBRIC/WAND/raw/master/" + pointer_rel.as_posix()
    )
    register_calls = [
        c for c in annex.call_annex.call_args_list if c.args[0][:1] == ["registerurl"]
    ]
    assert len(register_calls) == 1
    assert register_calls[0].args[0] == ["registerurl", annex_key, expected_url]

    # 2. AnnexRepo.get invoked from the web special remote (via --from web,
    #    since datalad's remote= only accepts git remotes) with jobs + the file
    annex.get.assert_called_once()
    get_args, get_kwargs = annex.get.call_args
    assert pointer_rel.as_posix() in get_args[0]
    assert get_kwargs["options"] == ["--from", "web"]
    assert get_kwargs["jobs"] == 3


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

    def mock_s3_download(self: download.S3, overwrite: bool = False) -> None:
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


# ---------------------------------------------------------------------------
# Unified selective-download matcher (include / exclude)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "patterns,relpath,expected",
    [
        (["sub-01"], "sub-01/eeg/x.set", True),  # bare folder selects its subtree
        (["sub-*"], "sub-01/eeg/x.set", True),  # glob folder selects subtree
        (["sub-02"], "sub-01/eeg/x.set", False),
        (["sub-1/**/*run-01*"], "sub-1/ses-a/func/x_run-01_bold.nii", True),
        (["*.set"], "sub-01/eeg/x.set", True),  # '*' spans '/'
        (["derivatives"], "sub-01/eeg/x.set", False),
        (["/sub-01/"], "sub-01/eeg/x.set", True),  # leading/trailing slashes ignored
        ([], "anything/at/all", False),  # empty pattern list never matches
    ],
)
def test_globs_match(patterns: list[str], relpath: str, expected: bool) -> None:
    assert download._globs_match(patterns, relpath) is expected


def _s3(tmp_path: Path, **kwargs: tp.Any) -> download.S3:
    return download.S3(study="s", dset_dir=tmp_path / "s", bucket="b", **kwargs)


def test_selects_empty_include_matches_everything(tmp_path: Path) -> None:
    assert _s3(tmp_path)._selects("sub-01/eeg/x.set") is True


def test_selects_include_only(tmp_path: Path) -> None:
    obj = _s3(tmp_path, include=["sub-01"])
    assert obj._selects("sub-01/eeg/x.set") is True
    assert obj._selects("sub-02/eeg/x.set") is False


def test_selects_exclude_beats_include(tmp_path: Path) -> None:
    obj = _s3(tmp_path, include=["sub-*"], exclude=["*/derivatives/*", "sub-02"])
    assert obj._selects("sub-01/eeg/x.set") is True
    assert obj._selects("sub-02/eeg/x.set") is False  # excluded by name
    assert obj._selects("sub-01/derivatives/y.tsv") is False  # excluded by subpath


def test_reject_selection_filters(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="selective downloading"):
        _s3(tmp_path, include=["sub-01"])._reject_selection_filters()
    with pytest.raises(NotImplementedError, match="selective downloading"):
        _s3(tmp_path, exclude=["sub-01"])._reject_selection_filters()
    # no filters -> no raise
    _s3(tmp_path)._reject_selection_filters()


# ---------------------------------------------------------------------------
# Per-backend selection wiring
# ---------------------------------------------------------------------------


def test_s3_resolve_prefix_filters(tmp_path: Path) -> None:
    """S3._resolve_prefix keeps only keys whose dataset-relative path selects."""
    obj = _s3(tmp_path, prefix="ds/1.0.0", include=["sub-01"])
    keys = [
        "ds/1.0.0/sub-01/eeg/a.set",
        "ds/1.0.0/sub-02/eeg/b.set",
        "ds/1.0.0/README",
    ]
    bucket = MagicMock()
    bucket.objects.filter.return_value = [MagicMock(key=k) for k in keys]

    jobs = obj._resolve_prefix(bucket, skip_existing=False)
    rels = sorted(str(p.relative_to(obj._out).as_posix()) for _, p in jobs)
    assert rels == ["sub-01/eeg/a.set"]


def _datalad(tmp_path: Path, **kwargs: tp.Any) -> download.Datalad:
    return download.Datalad(
        study="s",
        dset_dir=tmp_path / "s",
        repo_url="https://example.com/ds004192.git",
        **kwargs,
    )


def test_datalad_selected_paths_filters(tmp_path: Path) -> None:
    obj = _datalad(tmp_path, include=["sub-01"])
    repo_root = obj._dl_dir / obj.repo_name
    for rel in [
        "sub-01/eeg/a.set",
        "sub-02/eeg/b.set",
        "dataset_description.json",
        ".git/config",
    ]:
        p = repo_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    assert obj._selected_paths(repo_root) == [str(repo_root / "sub-01" / "eeg" / "a.set")]


def test_datalad_selected_paths_no_filter_returns_dot(tmp_path: Path) -> None:
    obj = _datalad(tmp_path)
    assert obj._selected_paths(obj._dl_dir / obj.repo_name) == ["."]


@pytest.mark.parametrize("name", ["Dandi", "Synapse", "Eegdash", "Donders"])
def test_backend_rejects_selection_filters(name: str, tmp_path: Path) -> None:
    """Backends with no native filter raise NotImplementedError when include set."""
    cls = getattr(download, name)
    extra = _EXTRA_KWARGS.get(name, {})
    env = _ENV_VARS.get(name, {})
    with patch.dict(os.environ, env):
        obj = cls(study="s", dset_dir=tmp_path / name, include=["sub-01"], **extra)
    with pytest.raises(NotImplementedError, match="selective downloading"):
        obj._download()


# ---------------------------------------------------------------------------
# #1751 / neuroai#51 regression: never write a success marker on failure
# ---------------------------------------------------------------------------


def test_download_does_not_mark_success_on_failure(tmp_path: Path) -> None:
    obj = _s3(tmp_path)
    with (
        patch.object(type(obj), "_check_requirements"),
        patch.object(obj, "_download", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            obj.download()
    assert not obj.get_success_file().exists()


def test_datalad_missing_binary_raises_no_marker(tmp_path: Path) -> None:
    """#1751: a missing datalad install must error, not warn + mark success.

    With the move to ``datalad.api``, ``_check_requirements`` raises before
    ``_download`` runs, so no success marker is written over an empty folder.
    """
    obj = _datalad(tmp_path)
    with patch.object(
        download.Datalad,
        "_check_requirements",
        side_effect=ModuleNotFoundError("No module named 'datalad'"),
    ):
        with pytest.raises(ModuleNotFoundError):
            obj.download()
    assert not obj.get_success_file().exists()
    assert list(obj._dl_dir.iterdir()) == []
