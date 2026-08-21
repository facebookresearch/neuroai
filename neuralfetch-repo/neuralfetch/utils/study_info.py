# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""StudyInfo computation and source-file rewriting for neuralfetch studies."""

import ast
import inspect
import logging
import os
import subprocess
import sys
import typing as tp
from pathlib import Path

import neuralset as ns
import neuralset.events as ev
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
