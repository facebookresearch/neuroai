# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import typing as tp
import urllib.request
import zipfile
from abc import abstractmethod
from glob import glob
from pathlib import Path

import mne
import pydantic
from tqdm import tqdm

from neuralset.base import BaseModel, PathLike, _Module

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def success_writer(
    fname: str | Path, suffix: str = "_success.txt", success_msg: str = "done"
):
    """Look for a file ending with ``suffix`` indicating ``fname`` has
    already been processed and create it after the encapsulated block
    succeeds.

    Examples
    --------

    >>> fname = './test.txt'
    >>> for _ in range(2):
    >>>     with success_writer(fname) as success:
    ...         if not success:
    ...             print(fname)
    ./test.txt
    """
    success_fname = Path(str(Path(fname).with_suffix("")) + suffix)
    file_exists = success_fname.exists()
    yield file_exists
    if not file_exists:
        with open(success_fname, "w") as f:
            f.write(success_msg)


@contextlib.contextmanager
def temp_mne_data(path: Path, *, clear_dataset_configs: bool = False):
    """Context manager that temporarily redirects MNE's data path.

    Uses **only environment variables** so that concurrent processes on an
    NFS cluster never race on MNE's JSON config file.  ``mne.get_config``
    checks env-vars first, so this is sufficient for both MNE and MOABB
    path resolution.

    Parameters
    ----------
    path : Path
        Directory to use as ``MNE_DATA``.  Must already exist.
    clear_dataset_configs : bool
        If True, temporarily clears all ``MNE_DATASETS_*_PATH`` entries
        (via env-vars) so that MNE/MOABB re-derives paths from
        ``MNE_DATA`` instead of using stale cached paths.  Recommended
        during *download*; leave False during *load* so that existing
        per-dataset configs stay intact.
    """
    if not path.exists():
        raise FileNotFoundError(f"MNE data path does not exist: {path}")

    saved_env: dict[str, str | None] = {}
    try:
        saved_env["MNE_DATA"] = os.environ.get("MNE_DATA")
        os.environ["MNE_DATA"] = str(path)

        if clear_dataset_configs:
            all_cfg = mne.get_config() or {}
            ds_keys = [
                k
                for k in all_cfg
                if k.startswith("MNE_DATASETS_") and k.endswith("_PATH")
            ]
            for k in ds_keys:
                saved_env[k] = os.environ.get(k)
                os.environ[k] = str(path)
        yield
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def download_file(
    url: str,
    destination: PathLike,
    headers: dict[str, str] | None = None,
    show_progress: bool = True,
    file_hash: str | None = None,
) -> None:
    """Download file from URL with optional progress reporting and custom headers.

    Parameters
    ----------
    url : str
        URL to download from
    destination : PathLike
        Where to save the downloaded file. Parent directories will be created if needed.
    headers : dict[str, str], optional
        Custom HTTP headers for the request (e.g., authorization tokens).
    show_progress : bool, optional
        Whether to show a tqdm download progress bar (default: True).
    file_hash : str, optional
        Expected checksum of the downloaded file in the format ``"algorithm:hexdigest"``
        (e.g. ``"md5:abc123"`` or ``"sha256:def456"``). When provided, pooch verifies
        the file after download and raises an error on mismatch.
    """
    import pooch

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Downloading {url} to {destination}")

    downloader = pooch.HTTPDownloader(
        headers=headers or {},
        progressbar=show_progress,
    )
    pooch.retrieve(
        url=url,
        known_hash=file_hash,
        fname=destination.name,
        path=destination.parent,
        downloader=downloader,
    )

    logger.debug(f"Download complete: {destination}")


def extract_zip(
    zip_path: PathLike,
    destination: PathLike | None = None,
    password: str | None = None,
    remove_after: bool = True,
    strip_root: bool = False,
) -> Path:
    """Extract zip file with optional password and cleanup.

    Parameters
    ----------
    zip_path : PathLike
        Path to the zip file to extract
    destination : PathLike, optional
        Where to extract files. If None, extracts to zip_path's parent directory.
    password : str, optional
        Password for encrypted zip files
    remove_after : bool, optional
        Whether to delete the zip file after extraction (default: True)
    strip_root : bool, optional
        If True and the zip contains a single top-level directory, strip that
        directory prefix so its contents land directly in ``destination``
        (default: False). Has no effect when the zip has multiple top-level
        entries or only files at the root level.

    Returns
    -------
    Path
        Path to the extraction destination
    """
    zip_path = Path(zip_path)

    if destination is None:
        destination = zip_path.parent
    else:
        destination = Path(destination)

    destination.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Extracting {zip_path.name} to {destination}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        if password:
            zip_ref.setpassword(password.encode())

        if strip_root:
            names = zip_ref.namelist()
            top_entries = {n.split("/")[0] for n in names if n.strip("/")}
            if len(top_entries) == 1:
                root_prefix = top_entries.pop() + "/"
                for info in zip_ref.infolist():
                    stripped = (
                        info.filename[len(root_prefix) :]
                        if info.filename.startswith(root_prefix)
                        else info.filename
                    )
                    if not stripped:
                        continue
                    target = destination / stripped
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(zip_ref.read(info.filename))
            else:
                zip_ref.extractall(destination)
        else:
            zip_ref.extractall(destination)

    logger.debug("Extraction complete")

    if remove_after:
        zip_path.unlink()
        logger.debug(f"Removed {zip_path.name}")

    return destination


def download_and_extract(
    url: str,
    destination: PathLike,
    password: str | None = None,
    headers: dict[str, str] | None = None,
    show_progress: bool = True,
    keep_zip: bool = False,
) -> Path:
    """Download zip file and extract in one operation.

    Convenience function that downloads a zip file and immediately extracts it,
    optionally removing the zip file after extraction.

    Parameters
    ----------
    url : str
        URL to download zip file from
    destination : PathLike
        Where to extract the files
    password : str, optional
        Password for encrypted zip files
    headers : dict[str, str], optional
        Custom HTTP headers for the download request
    show_progress : bool, optional
        Show download progress (default: True)
    keep_zip : bool, optional
        Keep the zip file after extraction (default: False)

    Returns
    -------
    Path
        Path to the extraction destination
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        download_file(url, tmp_path, headers=headers, show_progress=show_progress)

        extract_zip(
            tmp_path,
            destination=destination,
            password=password,
            remove_after=True,
        )

        return destination

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


class BaseDownload(_Module):
    """Abstract base class for all neuralfetch download backends.

    Subclasses must implement :meth:`_download`.  The public :meth:`download`
    method wraps ``_download`` with idempotency checks via a *success file*:
    once a dataset has been downloaded successfully, subsequent calls are
    no-ops unless ``overwrite=True`` is passed.

    Parameters
    ----------
    study : str
        Dataset identifier (e.g. a Dandiset ID, Zenodo record, or study name).
    dset_dir : PathLike
        Root directory for the study.  The parent must already exist; this
        directory and the ``folder/`` sub-directory are created automatically.
    folder : str
        Name of the sub-directory inside *dset_dir* where raw files land.
        Defaults to ``"download"``.
    """

    study: str
    dset_dir: PathLike
    folder: str = "download"

    _dl_dir: Path = pydantic.PrivateAttr()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # check that parent folder exist and create download sub-folder
        dset_dir = Path(self.dset_dir).resolve()
        if not dset_dir.parent.exists():
            raise ValueError(f"Parent folder must exist for {dset_dir}")
        dset_dir.mkdir(exist_ok=True)
        self._dl_dir = dset_dir / self.folder
        self._dl_dir.mkdir(exist_ok=True, parents=True)

    def get_success_file(self) -> Path:
        cls_name = self.__class__.__name__.lower()
        study = self.study
        return self._dl_dir / f"{cls_name}_{study}_success_download.txt"

    @tp.final
    def download(self, overwrite: bool = False) -> None:
        if self.get_success_file().exists() and not overwrite:
            return
        self._check_requirements()
        print(f"Downloading {self.study} to {self._dl_dir}...")

        self._download()
        self.get_success_file().write_text("success")
        print("Done! Consider running giving read/write permissions to everyone:")
        print(f"chmod -R 777 {self._dl_dir}")  # we should do this on FAIR cluster

    @abstractmethod
    def _download(self) -> None:
        raise NotImplementedError


class S3(BaseDownload):
    """Download files from an S3 bucket via ``boto3``.

    Pure-Python implementation using the ``boto3`` SDK with parallel
    transfers via a thread pool.  No external CLI tools required.

    Parameters
    ----------
    bucket : str
        S3 bucket name (e.g. ``"physionet-open"``).
    prefix : str
        Key prefix to filter objects (e.g. ``"eegmat/1.0.0"``).
    anonymous : bool
        If True (default), use unsigned access.  When False, credentials
        are resolved from the standard boto3 chain (env vars, profile,
        instance metadata, etc.).
    profile : str or None
        Named AWS profile from ``~/.aws/config``.  Affects region,
        endpoint, and credential resolution for the boto3 session.
    files : list[str]
        Explicit S3 keys to download (relative to bucket root). When
        provided, only these keys are fetched instead of listing by prefix.
    files_with_destinations : list[tuple[str, str]]
        Pairs of ``(s3_key, local_path)`` for downloads that map S3 keys
        to arbitrary local paths rather than mirroring the S3 prefix
        structure. Local paths are resolved relative to ``output_dir``.
    output_dir : PathLike or None
        Explicit output directory for downloaded files. When ``None``
        (default), files are written to ``_dl_dir``.  Subclasses like
        ``Physionet`` use this to download into a temp directory before
        rearranging the result.
    aws_access_key_id : str or None
        Explicit AWS credentials for authenticated downloads.
    aws_secret_access_key : str or None
        Explicit AWS credentials for authenticated downloads.
    nworkers : int
        Number of parallel download threads.  Set to 1 to disable
        parallelism.
    skip_existing : bool
        If True (default), skip files that already exist locally.
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("boto3", "botocore")

    bucket: str
    prefix: str = ""
    anonymous: bool = True
    profile: str | None = None
    files: list[str] = []
    files_with_destinations: list[tuple[str, str]] = []
    output_dir: PathLike | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    nworkers: int = 8
    skip_existing: bool = True

    @property
    def _out(self) -> Path:
        """Effective output directory (``output_dir`` if set, else ``_dl_dir``)."""
        if self.output_dir is not None:
            return Path(self.output_dir)
        return self._dl_dir

    def _download(self) -> None:
        import boto3

        session_kwargs: dict[str, tp.Any] = {}
        if self.profile:
            session_kwargs["profile_name"] = self.profile
        session = boto3.Session(**session_kwargs)

        config_kwargs: dict[str, tp.Any] = {}
        if self.anonymous:
            from botocore import UNSIGNED

            config_kwargs["signature_version"] = UNSIGNED

        if self.aws_access_key_id and self.aws_secret_access_key:
            s3 = session.resource(
                "s3",
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                config=boto3.session.Config(**config_kwargs),
            )
        else:
            s3 = session.resource("s3", config=boto3.session.Config(**config_kwargs))
        bucket = s3.Bucket(self.bucket)

        if self.files_with_destinations:
            jobs = self._resolve_mapped()
        elif self.files:
            jobs = self._resolve_keys()
        else:
            jobs = self._resolve_prefix(bucket)

        self._download_parallel(bucket, jobs)

    def _resolve_mapped(self) -> list[tuple[str, Path]]:
        jobs: list[tuple[str, Path]] = []
        for s3_key, local_rel in self.files_with_destinations:
            local_path = (
                Path(local_rel)
                if Path(local_rel).is_absolute()
                else self._out / local_rel
            )
            if self.skip_existing and local_path.exists():
                continue
            jobs.append((s3_key, local_path))
        return jobs

    def _resolve_keys(self) -> list[tuple[str, Path]]:
        jobs: list[tuple[str, Path]] = []
        for s3_key in self.files:
            target = self._out / s3_key
            if self.skip_existing and target.exists():
                continue
            jobs.append((s3_key, target))
        return jobs

    def _resolve_prefix(self, bucket: tp.Any) -> list[tuple[str, Path]]:
        jobs: list[tuple[str, Path]] = []
        for obj in bucket.objects.filter(Prefix=self.prefix):
            rel = obj.key
            if self.prefix:
                rel = obj.key[len(self.prefix) :].lstrip("/")
            if not rel:
                continue
            target = self._out / rel
            if self.skip_existing and target.exists():
                continue
            jobs.append((obj.key, target))
        return jobs

    def _download_parallel(self, bucket: tp.Any, jobs: list[tuple[str, Path]]) -> None:
        if not jobs:
            return

        def _fetch(s3_key: str, local_path: Path) -> None:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Downloading s3://%s/%s -> %s", self.bucket, s3_key, local_path)
            bucket.download_file(s3_key, str(local_path))

        if self.nworkers <= 1:
            for s3_key, local_path in jobs:
                _fetch(s3_key, local_path)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=self.nworkers) as pool:
                futures = {
                    pool.submit(_fetch, s3_key, local_path): s3_key
                    for s3_key, local_path in jobs
                }
                for future in as_completed(futures):
                    future.result()


class Dandi(BaseDownload):
    """Download a Dandiset from the DANDI Archive (dandiarchive.org).

    Uses the official ``dandi`` Python client (``pip install dandi``).

    Parameters
    ----------
    study : str
        Dandiset ID (e.g. ``"000026"`` or ``"000049"``).  Use the numeric
        identifier shown in the archive URL.
    version : str
        Dandiset version to download (default: ``"draft"`` for the latest).
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("dandi",)
    version: str = "draft"

    def _download(self) -> None:
        import dandi.download  # type: ignore[import-not-found,import-untyped]

        url = f"https://dandiarchive.org/dandiset/{self.study}/{self.version}"
        dandi.download.download([url], output_dir=str(self._dl_dir), existing="skip")


class Datalad(BaseDownload):
    """Download datasets via DataLad and git-annex.

    Clones a git-annex repository and optionally retrieves a subset of
    directories.  A password-free SSH key is required for git operations.

    Parameters
    ----------
    repo_url : str
        URL of the DataLad repository to clone.
    threads : int
        Number of parallel download threads for ``datalad get`` (default: 1).
    paths : list of str
        paths or glob patterns to retrieve after cloning.  When empty,
        all content is retrieved via ``datalad get "*"``.
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("datalad-installer",)
    # url of the datalad repo to clone
    repo_url: str
    # number of threads used for the datalad operations
    threads: int = 1
    # list of path to actually fetch
    paths: list[str] = []

    @pydantic.computed_field  # type: ignore
    @property
    def repo_name(self) -> str:
        # retrieve name of the repo
        repo_name = Path(self.repo_url).name
        if Path(repo_name).suffix == ".git":
            repo_name = repo_name[:-4]
        return repo_name

    def _download(self) -> None:
        """Downloads data from datalab

        Since this requires a git connection to handle, make sure that
        git ssh-key is password free

        Parameters
            path: Path to store the dataset (will clone repo to that folder)
            url: Url of the datalad repository
            threads: Number of threads to parallize dataset download
            folders: List of folders to clone explicitly (otherwise everything is cloned).
                Contains a tuple of str and bool. Bool defines if str is a glob
        """
        import datalad.api as dlad
        # clone repo
        clone_res = dlad.clone(source=self.repo_url, path=self._dl_dir)

        expanded_paths = []
        # expand paths
        if len(self.paths) < 1:
            expanded_paths = ['.']
        else:
            for path in self.paths:
                if '*' in path:
                    expanded_paths.extend(self._dl_dir.glob(path))
                else:
                    expanded_paths.append(self._dl_dir/path)
        logging.debug("Downloading %s", expanded_paths)
        clone_res.get(expanded_paths, jobs=self.threads)

        logging.info("Downloaded Dataset in %s", self._dl_dir)


class Donders(BaseDownload):
    """Download datasets from the Donders Repository (data.donders.ru.nl).

    Authenticates via WebDAV using credentials stored in environment variables.
    Obtain access at https://data.donders.ru.nl.

    Environment Variables
    ---------------------
    NEURALHUB_DONDERS_USER
        Your Donders account username.
    NEURALHUB_DONDERS_PASSWORD
        Your Donders account password.

    Parameters
    ----------
    study_id : str
        Donders collection identifier (e.g. ``"DSC_3011020.09_236"``).  Found
        in the collection URL on the Donders data portal.
    parent : str
        Parent collection path (default: ``"dccn"``).
    """

    parent: str = "dccn"
    study_id: str

    _user: str = pydantic.PrivateAttr()
    _password: str = pydantic.PrivateAttr()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        user = os.environ.get("NEURALFETCH_DONDERS_USER")
        password = os.environ.get("NEURALFETCH_DONDERS_PASSWORD")
        if not user or not password:
            raise RuntimeError(
                "Donders requires user and password.\n"
                "Get them from https://data.donders.ru.nl/collections/di/dccn/DSC_3011020.09_236?0\n"
                "and export NEURALFETCH_DONDERS_USER and NEURALFETCH_DONDERS_PASSWORD."
            )
        self._user = user
        self._password = password

    def _download(self) -> None:
        command = "wget -r -nH -np --cut-dirs=1"
        command += " --no-check-certificate -U Mozilla"
        command += f" --user={self._user} --password={self._password}"
        command += " https://webdav.data.donders.ru.nl/"
        command += f"{self.parent}/{self.study_id}/ -P {self.dset_dir}"
        command += ' -R "index.html*" -e robots=off'
        print("Running command : ", command)
        result = subprocess.run(command.split(), capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if "Authentication Failed" in result.stderr:
            raise ValueError("Authentication Failed.")
        # donders download in the authorYEAR/study_code/
        # we want the content to be in authorYEAR/download/
        shutil.move(Path(self.dset_dir) / self.study, self._dl_dir)


dryad_msg = """Dryad API authentication required.
https://datadryad.org/stash/sessions/choose_login

How to generate a Dryad API token:
    1. Log in at datadryad.org.
    2. Click your name in the top-right corner and select "Edit profile".
    3. Under "API tokens", create a new token.
    4. Copy the generated Token value.
"""


class Dryad(BaseDownload):
    """Download datasets from Dryad (datadryad.org).

    Uses the Dryad v2 REST API with Bearer-token authentication.
    Automatically falls back to per-file downloads when the bulk endpoint
    is unavailable for large datasets.

    Environment Variables
    ---------------------
    NEURALHUB_DRYAD_TOKEN
        Dryad personal API token (alternative to ``token=`` parameter).

    Parameters
    ----------
    doi : str
        Dataset DOI (e.g. ``"10.5061/dryad.cz8w9gjjk"``).  Found on the
        dataset landing page on datadryad.org.
    token : str or None
        Dryad personal API token.  Falls back to ``NEURALHUB_DRYAD_TOKEN``
        when not provided.
    """

    doi: str  # e.g. "10.5061/dryad.cz8w9gjjk"
    token: str | None = None

    _DRYAD_API = "https://datadryad.org/api/v2"

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.token:
            self.token = os.environ.get("NEURALFETCH_DRYAD_TOKEN")
        if not self.token:
            raise RuntimeError(
                "A Dryad API token is required.\n"
                "Please pass token= or export NEURALFETCH_DRYAD_TOKEN.\n" + dryad_msg
            )

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _dryad_url(self, path: str) -> str:
        """Build a full Dryad URL, avoiding double /api/v2 prefixes."""
        if path.startswith("/api/v2"):
            return f"https://datadryad.org{path}"
        return f"{self._DRYAD_API}{path}"

    def _api_get(self, path: str) -> dict[str, tp.Any]:
        """Authenticated GET against the Dryad API."""
        import requests as req

        r = req.get(self._dryad_url(path), headers=self._auth_headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def _latest_version_files(self) -> list[dict[str, tp.Any]]:
        """Return the file list for the latest version of the dataset."""
        from urllib.parse import quote

        encoded = quote(f"doi:{self.doi}", safe="")
        versions = self._api_get(f"/datasets/{encoded}/versions")
        all_versions = versions.get("_embedded", {}).get("stash:versions", [])
        if not all_versions:
            raise RuntimeError(f"No versions found for DOI {self.doi}")
        latest = all_versions[-1]
        files_href = latest["_links"]["stash:files"]["href"]
        files_resp = self._api_get(files_href)
        return files_resp.get("_embedded", {}).get("stash:files", [])

    def _download(self) -> None:
        import tempfile
        import zipfile
        from urllib.parse import quote

        import requests as req

        encoded = quote(f"doi:{self.doi}", safe="")
        api_url = f"{self._DRYAD_API}/datasets/{encoded}/download"

        # Try the bulk download endpoint first (works for small datasets).
        tmp = Path(tempfile.mktemp(suffix=".bin"))
        try:
            try:
                download_file(
                    api_url, tmp, headers=self._auth_headers(), show_progress=False
                )
            except req.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 405:
                    self._download_individual_files()
                    return
                raise e
            try:
                extract_zip(tmp, self._dl_dir, remove_after=True)
                return
            except zipfile.BadZipFile:
                pass
        finally:
            if tmp.exists():
                tmp.unlink()

        self._download_individual_files()

    def _download_individual_files(self) -> None:
        """Download each file in the dataset individually."""
        files = self._latest_version_files()
        print(f"Downloading {len(files)} files individually from Dryad...")
        for i, f in enumerate(files, 1):
            name = f.get("path", f"file_{i}")
            dl_href = f.get("_links", {}).get("stash:download", {}).get("href")
            if not dl_href:
                logger.warning(f"Skipping {name}: no download link")
                continue
            url = self._dryad_url(dl_href)
            dest = self._dl_dir / name
            is_zip = name.endswith(".zip")
            extracted_dir = self._dl_dir / Path(name).stem if is_zip else None
            if dest.exists() or (extracted_dir is not None and extracted_dir.exists()):
                print(f"  [{i}/{len(files)}] {name} already exists, skipping")
                continue
            print(f"  [{i}/{len(files)}] Downloading {name}...")
            digest_type = f.get("digestType", "")
            digest = f.get("digest", "")
            file_hash = f"{digest_type}:{digest}" if digest and digest_type else None
            download_file(
                url,
                dest,
                headers=self._auth_headers(),
                show_progress=True,
                file_hash=file_hash,
            )
            if is_zip:
                extract_zip(dest, self._dl_dir, remove_after=True)


class Eegdash(BaseDownload):
    """Download datasets from the EEGDash cloud archive.

    Uses the ``eegdash`` library for record discovery and the :class:`S3`
    downloader for the actual file transfer.

    Parameters
    ----------
    study : str
        Dataset identifier (e.g. ``"ds002718"``).
    dset_dir : PathLike
        Root directory for the study.
    database : str
        EEGDash database to query (``"eegdash"``, ``"eegdash_staging"``, …).
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("eegdash",)
    database: str = "eegdash"

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        """Split ``s3://bucket/key/prefix`` into ``("bucket", "key/prefix")``."""
        without_scheme = uri.removeprefix("s3://")
        bucket, _, prefix = without_scheme.partition("/")
        return bucket, prefix

    def _download(self) -> None:
        from eegdash import EEGDash  # type: ignore[import-not-found]

        client = EEGDash(database=self.database)
        records = client.find(dataset=self.study)
        if not records:
            raise RuntimeError(
                f"No records found for dataset '{self.study}' "
                f"in database '{self.database}'"
            )

        # Group files by bucket so we can batch them into S3 downloaders.
        per_bucket: dict[str, list[tuple[str, str]]] = {}
        for rec in records:
            storage = rec.get("storage", {})
            base = storage.get("base", "")
            raw_key = storage.get("raw_key", "")
            dep_keys: list[str] = storage.get("dep_keys", [])
            bids_rel = rec.get("bids_relpath", raw_key)

            if not base or not raw_key:
                logger.warning("Skipping record with incomplete storage info: %s", rec)
                continue

            bucket, prefix = self._parse_s3_uri(base)

            s3_key = f"{prefix}/{raw_key}" if prefix else raw_key
            local_path = str(self._dl_dir / bids_rel)
            per_bucket.setdefault(bucket, []).append((s3_key, local_path))

            for dep in dep_keys:
                dep_s3_key = f"{prefix}/{dep}" if prefix else dep
                per_bucket.setdefault(bucket, []).append(
                    (dep_s3_key, str(self._dl_dir / dep))
                )

        total_files = sum(len(v) for v in per_bucket.values())
        if total_files == 0:
            raise RuntimeError(f"No downloadable files found for '{self.study}'")

        for bucket, file_pairs in per_bucket.items():
            dl = S3(
                study=self.study,
                dset_dir=Path(self.dset_dir),
                bucket=bucket,
                files_with_destinations=file_pairs,
                anonymous=True,
                skip_existing=True,
            )
            dl._download()

        print(f"\nDownloaded {total_files} files for {self.study}")


class Figshare(BaseDownload):
    """Download all files from a Figshare article (figshare.com).

    Fetches file metadata from the Figshare v2 public API and downloads
    each file into ``dset_dir/download/``.

    Parameters
    ----------
    study : str
        Figshare article ID (numeric string, e.g. ``"12345678"``).  Found
        in the article URL on figshare.com.
    """

    def _download(self) -> None:
        import requests

        item_ids = [self.study]
        BASE_URL = "https://api.figshare.com/v2"
        api_call_headers = {"Authorization": "token ENTER-TOKEN"}

        file_info = []

        for i in item_ids:
            # page_size set to arbritary high value to return items
            r = requests.get(f"{BASE_URL}/articles/{i}/files?page_size=1000")
            file_metadata = json.loads(r.text)
            for j in file_metadata:
                j["item_id"] = i
                file_info.append(j)

        for k in tqdm(file_info):
            response = requests.get(
                f"{BASE_URL}/file/download/{k['id']}",
                headers=api_call_headers,
            )
            open(self._dl_dir / k["name"], "wb").write(response.content)


class Huggingface(BaseDownload):
    """Download a dataset repository from Hugging Face Hub.

    Uses :func:`huggingface_hub.snapshot_download` for a full repository
    snapshot.  Set the ``HUGGING_FACE_HUB_TOKEN`` environment variable for
    private or gated repositories.

    Parameters
    ----------
    study : str
        Dataset repository name (e.g. ``"allen-bold"``).  The full repo ID
        used will be ``org/study``.
    org : str
        Hugging Face organisation or user that owns the dataset
        (e.g. ``"BrainAI"`` or ``"openai"``).  Combined with *study* to
        form ``repo_id``.
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("huggingface_hub",)
    org: str

    def _download(self) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=f"{self.org}/{self.study}",
            repo_type="dataset",
            local_dir=self._dl_dir,
        )
        print("\nDownloaded Dataset")


class Openneuro(BaseDownload):
    """Download datasets from OpenNeuro.

    The ``include`` parameter follows the openneuro-py API: files and
    directories to download. Only these files and directories will be
    retrieved. Uses Unix path expansion (``*`` for any number of wildcard
    characters and ``?`` for one wildcard character; e.g.
    ``'sub-1_task-*.fif'``). As an example, if you would like to download
    only subject '1' and run '01' files, you can do so via
    ``'sub-1/**/*run-01*'``. The pattern ``**`` will match any files and
    zero or more directories, subdirectories and symbolic links to
    directories.
    """

    requirements: tp.ClassVar[tuple[str, ...]] = ("openneuro-py>=2025.2.0",)
    excluded_patterns: list[str] = []
    include: list[str] | None = None

    def _download(self) -> None:
        import openneuro as on

        on.download(dataset=self.study, target_dir=self._dl_dir, include=self.include)


class Osf(BaseDownload):
    """Download datasets from OSF (osf.io).

    Iterates over files in a project storage using ``osfclient`` and
    downloads them to ``dset_dir/download/``.  Already-downloaded files
    are skipped automatically.

    Parameters
    ----------
    study : str
        OSF project ID (5-character alphanumeric string, e.g. ``"abcde"``).
        Found in the project URL: ``https://osf.io/<project_id>/``.
    storage_inds : list of int
        Indices of the project storages to download (default: ``[0]``).
        Useful when a project exposes multiple named storages.
    """

    storage_inds: list[int] = [0]  # In case of multiple storages, storages to download
    requirements: tp.ClassVar[tuple[str, ...]] = ("osfclient>=0.0.5",)

    def _download(self) -> None:
        import osfclient  # noqa

        project = osfclient.OSF().project(self.study)
        store = list(project.storages)

        pbar = tqdm()
        for ind in self.storage_inds:
            for source in store[ind].files:
                path = source.path
                if path.startswith("/"):
                    path = path[1:]

                file_ = self._dl_dir / path

                if file_.exists():
                    continue

                pbar.set_description(file_.name)
                file_.parent.mkdir(parents=True, exist_ok=True)
                with file_.open("wb") as fb:
                    source.write_to(fb)


class Physionet(S3):
    """Physionet datasets via anonymous boto3 access.

    Extends ``S3`` with the convention that Physionet datasets live under
    ``<study>/<version>/`` in the ``physionet-open`` bucket. After download,
    the versioned directory is flattened into ``_dl_dir``.
    """

    bucket: str = "physionet-open"
    version: str

    def _download(self, overwrite=False) -> None:
        self.prefix = f"{self.study}/{self.version}"
        temp_folder = self._dl_dir.parent / "temp"
        self.output_dir = temp_folder

        # Download into a temp directory first because Physionet stores
        # objects under <study>/<version>/, which the S3 base class
        # mirrors on disk.  After downloading we flatten by renaming
        # temp/<study>/<version>/ directly into _dl_dir.
        super()._download()

        inner = temp_folder / self.study / self.version
        inner.rename(self._dl_dir)
        inner.parent.rmdir()
        temp_folder.rmdir()


synapse_msg = """Requires creating a Synapse account with 2FA.
https://accounts.synapse.org/register1?appId=synapse.org

How to generate a Synapse auth token:
    1. Register or log in at synapse.org.
    2. Go to "My Profile" -> "Edit Profile" -> "Personal Access Tokens".
    3. Click "Manage Personal Access Tokens".
    4. Select "Create New Token".
    5. Enter a name for the token and enable the "View" and "Download" options.
    6. Save the generated token securely for use.
"""


class Synapse(BaseDownload):
    """Download datasets from Synapse.

    Requires `synapseclient`. Install with `pip install synapseclient`
    or `pip install "neuralset[all]"`.
    """

    study_id: str  # Project SynID
    requirements: tp.ClassVar[tuple[str, ...]] = ("synapseclient",)

    _auth_token: str = pydantic.PrivateAttr()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        token = os.environ.get("NEURALFETCH_SYNAPSE_TOKEN")
        if not token:
            raise RuntimeError(
                "Synapse auth_token is required but was not provided.\n"
                "Please export NEURALFETCH_SYNAPSE_TOKEN.\n" + synapse_msg
            )
        self._auth_token = token

    def _download(self) -> None:
        import synapseclient  # type: ignore[import-not-found]
        import synapseutils  # type: ignore[import-not-found]

        syn = synapseclient.Synapse()
        syn.login(authToken=self._auth_token)
        synapseutils.syncFromSynapse(syn=syn, entity=self.study_id, path=self._dl_dir)


class Zenodo(BaseDownload):
    """Download datasets from Zenodo (zenodo.org).

    Fetches file metadata from the Zenodo REST API, then downloads each
    file.  Zip archives are extracted automatically; other files are saved
    as-is.  Checksums provided by the API are verified after download.

    Parameters
    ----------
    study : str
        Study name used for the success-file path.
    record_id : str
        Zenodo record ID (numeric string, e.g. ``"1234567"``).  Found in
        the record URL: ``https://zenodo.org/records/<record_id>``.
    """

    record_id: str

    def _fetch_file_metadata(self) -> dict[str, str]:
        """Fetch file metadata from Zenodo API to get filenames and checksums automatically.

        Returns:
            Dictionary mapping filename to checksum (MD5 or SHA256)
        """
        api_url = f"https://zenodo.org/api/records/{self.record_id}"

        try:
            with urllib.request.urlopen(api_url) as response:
                data = json.loads(response.read().decode())

            # Build filename -> checksum mapping
            file_checksums = {}
            for file_info in data.get("files", []):
                filename = file_info["key"]
                # Zenodo provides checksums in format "md5:xxxxx" or "sha256:xxxxx"
                checksum_str = file_info["checksum"]
                file_checksums[filename] = checksum_str

            return file_checksums
        except Exception as e:
            logging.warning(f"Failed to fetch Zenodo metadata from {api_url}: {e}")
            raise RuntimeError(
                f"Could not fetch file metadata from Zenodo API for record {self.record_id}. "
                f"Please check the record ID or provide dataset_fname/dataset_hash manually."
            ) from e

    def _download(self) -> None:
        # TODO: Consider making the download async for multiple file downloads to occur
        # cf. openneuro-py implementation

        # Fetch file metadata from API
        file_metadata = self._fetch_file_metadata()

        zipped_files = {
            fname: file_hash
            for fname, file_hash in file_metadata.items()
            if fname.endswith(".zip")
        }
        info_files = {
            fname: file_hash
            for fname, file_hash in file_metadata.items()
            if not fname.endswith(".zip")
        }

        base_url = f"https://zenodo.org/records/{self.record_id}"

        for filename, file_hash in zipped_files.items():
            dest = self._dl_dir / Path(filename).stem
            if dest.exists():
                continue
            tmp = self._dl_dir / f"temp_{filename}"
            download_file(
                f"{base_url}/files/{filename}",
                tmp,
                show_progress=True,
                file_hash=file_hash,
            )
            extract_zip(tmp, destination=dest, remove_after=True, strip_root=True)

        for filename, file_hash in info_files.items():
            dest = self._dl_dir / filename
            if dest.exists():
                continue
            download_file(
                f"{base_url}/files/{filename}",
                dest,
                show_progress=True,
                file_hash=file_hash,
            )
