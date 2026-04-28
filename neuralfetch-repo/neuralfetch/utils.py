# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utilities for neuralfetch study development."""

import ast
import inspect
import logging
import os
import subprocess
import sys
import tempfile
import typing as tp
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import neuralset as ns
import neuralset.events as ev
from neuralfetch.download import download_file, extract_zip
from neuralset.events import study as base

logger = logging.getLogger(__name__)


def root_study_folder(name: str | None = None, test_folder: Path | None = None) -> Path:
    """Return the root folder where study data is stored.

    Example
    -------
    >>> folder = neuralfetch.utils.root_study_folder()
    >>> study = ns.Study(name="Allen2022Massive", path=folder)

    Built-in test/sample studies use ``ns.CACHE_FOLDER`` (or *test_folder*).
    All others require ``NEURALSET_STUDY_FOLDER`` env var.
    """
    if name is not None:
        if name.startswith(("Mne2013Sample", "Fake2025Meg", "Dummy")):
            return ns.CACHE_FOLDER
        if name.startswith(("Test", "Fake")):
            return test_folder if test_folder is not None else ns.CACHE_FOLDER
    env = os.environ.get("NEURALSET_STUDY_FOLDER")
    if env is None:
        raise RuntimeError(
            "NEURALSET_STUDY_FOLDER env var is not set.\n"
            "Export it to the root folder containing your study data, e.g.:\n"
            "  export NEURALSET_STUDY_FOLDER=/path/to/root/studies/folder"
        )
    return Path(env)


def compute_study_info(name: str, folder: str | Path) -> dict[str, tp.Any]:
    """Load study *name* from *folder* and return a dict of actual ``StudyInfo`` values.

    Always computes num_timelines, num_subjects, num_events_in_query, and
    event_types_in_query.  Attempts to read one Fmri/MneRaw event for
    data_shape, frequency, and fmri_spaces (skipped on failure).
    """
    folder = Path(folder)
    default_query = "timeline_index < 1"
    study = ns.Study(name=name, path=folder, query=default_query)
    cls = type(study)
    info = cls._info
    query = info.query if info is not None else default_query
    if query != default_query:
        study = ns.Study(name=name, path=folder, query=query)
    study.infra_timelines.cluster = None  # avoid process pool startup
    cls._info = None  # bypass num_timelines check during loading
    try:
        summary = study.study_summary(apply_query=False)
        events = study.run()
    finally:
        cls._info = info
    actual: dict[str, tp.Any] = dict(
        num_timelines=len(summary),
        num_subjects=summary.subject.nunique(),
        num_events_in_query=len(events),
        event_types_in_query=set(events["type"].unique()),
    )
    # Read first Fmri/MneRaw event for data_shape / frequency.
    types = ev.etypes.EventTypesHelper(["Fmri", "MneRaw"]).names
    matching = events.loc[events.type.isin(types)]
    if matching.empty:
        return actual
    event = ev.Event.from_dict(matching.iloc[0])
    data = event.read()  # type: ignore
    if isinstance(event, ev.etypes.Fmri):
        actual["data_shape"] = data.shape
        fmri_types = ev.etypes.EventTypesHelper(["Fmri"]).names
        actual["fmri_spaces"] = set(
            matching.loc[matching.type.isin(fmri_types), "space"].unique()
        )
    elif isinstance(event, ev.etypes.MneRaw):
        pick_map: dict[type, str | tuple[str, ...]] = {
            ev.etypes.Eeg: "eeg",
            ev.etypes.Emg: "emg",
            ev.etypes.Fnirs: "fnirs",
            ev.etypes.Ieeg: ("seeg", "ecog"),
            ev.etypes.Meg: "meg",
        }
        if isinstance(event, tuple(pick_map)):
            data.pick(pick_map[type(event)])
        actual["data_shape"] = (len(data.ch_names), int(data.n_times))
    actual["frequency"] = event.frequency  # type: ignore[attr-defined]
    return actual


# ---------------------------------------------------------------------------
# Source-file rewriting
# ---------------------------------------------------------------------------


def _find_info_lines(source: str, class_name: str) -> tuple[int, int]:
    """Return 1-indexed (start, end) line range of the ``_info`` assignment.

    Handles both annotated (``_info: ... = ...``) and plain (``_info = ...``)
    assignments.  If ``_info`` is absent, returns an empty range before the
    first method so a splice inserts a new line there.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        fallback = node.body[-1].end_lineno or node.body[-1].lineno
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first_line = (
                    item.decorator_list[0].lineno if item.decorator_list else item.lineno
                )
                fallback = first_line - 1
                break
            target_name = None
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                target_name = item.target.id
            elif isinstance(item, ast.Assign):
                for t in item.targets:
                    if isinstance(t, ast.Name) and t.id == "_info":
                        target_name = t.id
            if target_name == "_info":
                assert item.end_lineno is not None
                return item.lineno, item.end_lineno
        return fallback + 1, fallback  # empty range: insert at fallback
    raise ValueError(f"class {class_name} not found")


def _repr_val(val: tp.Any) -> str:
    """Deterministic repr: sorted sets, floats rounded to 3 decimals."""
    if isinstance(val, set):
        return "{" + ", ".join(repr(x) for x in sorted(val)) + "}"
    if isinstance(val, float):
        return repr(round(val, 3))
    return repr(val)


def format_study_info(actual: dict[str, tp.Any]) -> str:
    """Return a formatted ``StudyInfo(...)`` string from computed values."""
    parts = [
        f"{f}={_repr_val(actual[f])}"
        for f in base.StudyInfo.model_fields
        if f != "query" and f in actual
    ]
    code = f"StudyInfo({', '.join(parts)})"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--line-length=90",
            "--stdin-filename=_.py",
        ],
        input=code,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def update_source_info(name: str, folder: str | Path | None = None) -> dict[str, tp.Any]:
    """Compute actual ``StudyInfo`` values, rewrite the source file, and run ``ruff format``.

    If *folder* is ``None``, uses the default study folder (or cache folder
    for test/fake studies).  Returns the computed values dict.

    Usage::

        python -c "from neuralfetch.utils import update_source_info; update_source_info('StudyName')"
    """
    if folder is None:
        folder = root_study_folder(name)
    actual = compute_study_info(name, folder)
    info_str = format_study_info(actual)
    new_info = f"    _info: tp.ClassVar[study.StudyInfo] = study.{info_str}\n"
    # Rewrite source file.
    cls = type(ns.Study(name=name, path="."))
    source_file = inspect.getsourcefile(cls)
    if source_file is None:
        raise RuntimeError(f"Cannot locate source file for {name}")
    path = Path(source_file)
    source = path.read_text("utf8")
    lines = source.splitlines(keepends=True)
    start, end = _find_info_lines(source, cls.__name__)
    lines[start - 1 : end] = [new_info]
    path.write_text("".join(lines))
    subprocess.run([sys.executable, "-m", "ruff", "format", str(path)], check=True)
    logger.info("Updated _info in %s", path)
    return actual


# ---------------------------------------------------------------------------
# Study data utilities (moved from neuralset.studies.utils)
# ---------------------------------------------------------------------------


def add_sentences(
    events: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    column_to_group: str = "sequence_id",
) -> pd.DataFrame:
    """
    Add sentence-level information to the events DataFrame based on the sequence_id column.
    """

    assert column_to_group in events.columns
    assert "Sentence" not in events.type.unique()

    if "timeline" in events.columns and len(events.timeline.unique()) > 1:
        # apply to each timeline
        timelines = []
        for _, df in tqdm(events.groupby("timeline"), "Adding sentences"):
            df = add_sentences(df, ratios, column_to_group)
            timelines.append(df)
        return pd.concat(timelines)
    events["stop"] = events.start + events.duration

    words = events.query('type=="Word"')
    assert all(words.start.diff().dropna() >= 0)  # type: ignore

    # Add sentence-level information
    words = events.query('type=="Word"')
    sentences = []
    for _, sent in words.groupby(column_to_group, sort=False):
        # Find all events within the sentence
        duration = sent.stop.max() - sent.start.min() + 1e-8

        sentence = " ".join(sent.text)
        events.loc[sent.index, "sentence"] = sentence

        # Add Sentence event
        to_add = sent.iloc[0].to_dict()
        to_add["type"] = "Sentence"
        to_add["text"] = " ".join(sent.text)
        to_add["start"] = sent.start.min() - 1e-8
        to_add["duration"] = duration
        to_add["stop"] = sent.stop.max()
        sentences.append(to_add)

    events = pd.concat([events, pd.DataFrame(sentences)], ignore_index=True)
    events = events.sort_values("start")
    events = events.reset_index(drop=True)

    return events


def scan_files(
    path: str | Path,
    stopping_criterion: tp.Callable[[str], bool] | None = None,
) -> tp.Iterator[str]:
    """Recursively yield file paths from given directory.

    Parameters
    ----------
    path :
        Directory path to scan for files recursively.
    stopping_criterion :
        Callable that returns True if the current node/leaf should be yielded instead of scanned
        further. E.g. useful to catch recordings of neuro data format that are saved as folders,
        e.g. the CTF format (MEG) saves recordings as folders with a .ds extension.

    Note
    ----
    About 2x faster than a combination of Path.iterdir and Path.rglob for walking through
    Obeid2016's directory.
    """
    stopping_criterion = stopping_criterion or (lambda x: False)
    for entry in sorted(os.scandir(path), key=lambda e: e.name):
        if stopping_criterion(entry.name) or not entry.is_dir(follow_symlinks=False):
            yield entry.path
        else:
            yield from scan_files(entry.path, stopping_criterion=stopping_criterion)


def ensure_imagenet22k(
    custom_path: Path | str | None = None,
    fail_on_error: bool = False,
) -> Path:
    """Ensure ImageNet-22k dataset exists, providing download instructions if not.

    Parameters
    ----------
    custom_path : Path | str | None, optional
        Custom path where ImageNet-22k is or should be located. If None,
        reads ``NEURALSET_IMAGENET22K`` env var; raises if unset.
    fail_on_error : bool, optional
        If True, raise an exception if ImageNet-22k is not found. If False (default),
        log a warning and continue.

    Returns
    -------
    Path
        Path to the ImageNet-22k directory (may not exist if not yet downloaded).
    """
    if custom_path is None:
        env = os.environ.get("NEURALSET_IMAGENET22K")
        if env is None:
            raise RuntimeError(
                "ImageNet-22k path not configured.\n"
                "Please export NEURALSET_IMAGENET22K=/path/to/imagenet-22k "
                "or pass custom_path= to ensure_imagenet22k()."
            )
        imagenet_path = Path(env)
    else:
        imagenet_path = Path(custom_path).resolve(strict=False)

    if imagenet_path.exists():
        logger.info(f"ImageNet-22k found at {imagenet_path}")
        return imagenet_path

    logger.warning(f"ImageNet-22k not found at expected location: {imagenet_path}")
    logger.info("\n" + "=" * 80)
    logger.info("ImageNet-22k is required for natural image stimuli.")
    logger.info("=" * 80)
    logger.info("\nImageNet-22k (ImageNet-21k) Access Instructions:")
    logger.info("1. Visit the ImageNet website: https://image-net.org/")
    logger.info("2. Create an account or log in")
    logger.info("3. Review and agree to the ImageNet Terms of Access")
    logger.info("4. Request access to ImageNet-22k (Fall 2011 release or later)")
    logger.info(
        "5. Follow their download instructions (typically via direct download or torrent)"
    )
    logger.info("6. Extract the dataset to maintain synset folder structure:")
    logger.info("   imagenet-22k/")
    logger.info("     n01440764/  (synset folders)")
    logger.info("       n01440764_00001.JPEG")
    logger.info("       ...")
    logger.info("     ...")
    logger.info(f"7. Place the dataset at: {imagenet_path}")
    logger.info("\nDataset Details:")
    logger.info("- Size: ~1.3 TB compressed, ~1.5 TB uncompressed")
    logger.info("- Images: ~14 million")
    logger.info("- Classes: ~21,841 WordNet synsets")
    logger.info("- Format: JPEG images organized by synset folders")
    logger.info("\nAlternative:")
    logger.info("- Check if your institution has a local copy")
    logger.info("- Contact your system administrator for access")
    logger.info("=" * 80)

    if fail_on_error:
        raise RuntimeError(
            f"ImageNet-22k required but not found at {imagenet_path}. "
            "Please obtain access from https://image-net.org/"
        )

    logger.info(
        "\nContinuing without ImageNet-22k. Study may fail if natural image "
        "stimuli are required."
    )
    return imagenet_path


def download_things_images(
    study_path: Path,
    fail_on_error: bool = False,
) -> Path:
    """Download THINGS-images dataset if not already present.

    THINGS-images contains the full image database (~1,854 object concepts)
    organized by category folders. The dataset is shared across multiple
    THINGS-related studies (Hebart2023ThingsFmri, Gifford2022ThingsEeg, Grootswagers2022ThingsEeg2, etc.).

    Full license terms: https://osf.io/jum2f/files/52wrx

    Parameters
    ----------
    study_path : Path
        Path to the study directory. THINGS-images should be in the parent
        directory as a sibling folder.
    fail_on_error : bool, optional
        If True, raise an exception if THINGS-images cannot be obtained.
    """
    things_images_path = (study_path / ".." / "THINGS-images").resolve(strict=False)

    if things_images_path.exists():
        logger.info(f"THINGS-images found at {things_images_path}")
        return things_images_path

    logger.warning(f"THINGS-images not found at expected location: {things_images_path}")

    password = os.environ.get("NEURALFETCH_THINGS_PASSWORD")
    if password is None:
        raise RuntimeError(
            "THINGS-images requires accepting a licence agreement.\n"
            "Please export NEURALFETCH_THINGS_PASSWORD=<pwd> where the password can be found "
            "in password_images.txt at https://osf.io/jum2f/files/52wrx"
        )

    zip_url = "https://files.osf.io/v1/resources/jum2f/providers/osfstorage/670d6f7dbce84f4a7bb9371b"
    print("\nDownloading images_THINGS.zip...")
    logger.info(f"Downloading from {zip_url}")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    tmp_path.unlink()

    try:
        download_file(zip_url, tmp_path, show_progress=True)
        print("Download complete!")

        extract_path = things_images_path.parent
        print(f"\nExtracting images_THINGS.zip to {extract_path}...")
        print("This may take ~1 hour...")
        logger.info(f"Extracting THINGS-images to {extract_path}")
        extract_zip(tmp_path, extract_path, password=password, remove_after=True)
        extracted_folder = extract_path / "object_images"
        extracted_folder.rename(things_images_path)

        print(f"\nTHINGS-images successfully installed at {things_images_path}")
        logger.info(f"THINGS-images installed at {things_images_path}")
        return things_images_path

    except Exception as e:
        logger.error(f"Error downloading or extracting THINGS-images: {e}")
        print(f"\nError: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        if fail_on_error:
            raise RuntimeError(f"Failed to download THINGS-images: {e}")
        return things_images_path
