# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Task config / dataset metadata derivation.

The four tables that drive the bar-chart badges and the core/full
filtering all originate from the same on-disk structure under
``{task_root}/{device}/{task_name}/``, for every task root the registry
discovers (the shipped tree plus any registered extension):

* ``config.yaml`` -- the task default study and split.
* ``datasets/*.yaml`` -- per-dataset overrides for multi-dataset
  paradigms.

This module centralises the directory walk plus the regex extraction so
each consumer is a small (~5 line) function.

Public helpers
--------------
* :func:`iter_task_yamls` -- the shared filesystem walker.
* :func:`build_task_split_map` -- ``{(task, study): SplitKind}``.
* :func:`build_task_n_examples_map` -- ``{(task, study): n_examples}``
  derived from the sibling ``brainai`` package's
  ``neuralbench_stats.json``.
* :func:`default_studies_per_task` --
  ``{task: [device-default-study, ...]}``.
* :func:`default_study_names` -- a flat ``{task: study}`` view for
  callers that don't need per-device disambiguation.
* :func:`classify_split_yaml` -- raw split-kind classifier (exposed for
  unit tests and ad-hoc tooling).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from neuralbench.plots._style import SplitKind

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Strict pattern: matches ``source: name: <id>`` directly, with no
# intermediate fields.  Used by the default-study lookup to mirror the
# historical ``benchmark.py`` semantics exactly.
_CONFIG_NAME_STRICT: re.Pattern[str] = re.compile(
    r"^data:\s*\n\s+study:\s*\n\s+source:\s*\n\s+name:\s*(\S+)",
    re.MULTILINE,
)

# Tolerant pattern: accepts intermediate fields (``path:``, ``infra:``,
# ...) between the ``source:`` header and the ``name:`` line.  Required
# for per-dataset override yamls that ``=replace=: true`` the entire
# study spec and therefore have to restate ``source.path`` etc.
_CONFIG_NAME_TOLERANT: re.Pattern[str] = re.compile(
    r"^data:\s*\n\s+study:(?:\s*\n\s+.*)*?\n\s+source:(?:\s*\n\s+[^\n]*)*?\n\s+name:\s*(\S+)",
    re.MULTILINE,
)
_DATASET_NAME_TOLERANT: re.Pattern[str] = re.compile(
    r"(?:^|\n)\s+source:(?:\s*\n\s+[^\n]*)*?\n\s+name:\s*(\S+)",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Filesystem walker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TaskYamlSpec:
    """One task ``config.yaml`` or per-dataset override yaml.

    Attributes
    ----------
    device:
        Top-level recording modality (``"eeg"``, ``"meg"``, ``"fmri"``...).
    task_name:
        Folder name of the task (``"motor_imagery"``, ``"image"``, ...).
    yaml_path:
        Full path on disk.
    yaml_text:
        File contents (UTF-8 decoded).
    is_dataset_override:
        ``True`` for files under ``tasks/{device}/{task}/datasets/``,
        ``False`` for the root ``config.yaml``.
    """

    device: str
    task_name: str
    yaml_path: Path
    yaml_text: str
    is_dataset_override: bool


def _task_roots(tasks_dir: Path | None) -> list[Path]:
    """Task trees to walk: an explicit *tasks_dir*, else every registered root.

    Deferring to the registry keeps the badges and device grouping in step with
    task discovery, so extension-provided tasks are not silently unlabelled.
    """
    if tasks_dir is not None:
        return [tasks_dir]
    # deferred: registry imports neuralbench.plots._models at its module bottom
    from neuralbench import registry

    return registry._all_task_roots()


def iter_task_yamls(
    tasks_dir: Path | None = None,
) -> Iterator[_TaskYamlSpec]:
    """Yield one :class:`_TaskYamlSpec` per discovered yaml.

    The walk visits ``{root}/{device}/{task}/config.yaml`` first, then any
    ``{root}/{device}/{task}/datasets/*.yaml`` -- both in sorted order so
    consumers produce deterministic output without needing their own sort
    step.  Roots are walked in registry order; see :func:`_task_roots`.

    Hidden directories (those whose name starts with ``_``) are
    skipped at every level.
    """
    for base in _task_roots(tasks_dir):
        if not base.is_dir():
            continue
        for device_dir in sorted(base.iterdir()):
            if not device_dir.is_dir() or device_dir.name.startswith("_"):
                continue
            device = device_dir.name
            for task_dir in sorted(device_dir.iterdir()):
                if not task_dir.is_dir() or task_dir.name.startswith("_"):
                    continue
                task_name = task_dir.name
                config_path = task_dir / "config.yaml"
                if config_path.exists():
                    yield _TaskYamlSpec(
                        device=device,
                        task_name=task_name,
                        yaml_path=config_path,
                        yaml_text=config_path.read_text("utf8"),
                        is_dataset_override=False,
                    )
                datasets_dir = task_dir / "datasets"
                if datasets_dir.exists():
                    for ds_path in sorted(datasets_dir.glob("*.yaml")):
                        yield _TaskYamlSpec(
                            device=device,
                            task_name=task_name,
                            yaml_path=ds_path,
                            yaml_text=ds_path.read_text("utf8"),
                            is_dataset_override=True,
                        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_task_split_map(
    tasks_dir: Path | None = None,
) -> dict[tuple[str, str], SplitKind]:
    """Return ``{(task_name, study_name): split_kind}``.

    Per-dataset yamls override the task-level classification when they
    carry their own split spec; otherwise the task default is inherited
    so multi-dataset paradigms still receive a badge.
    """
    out: dict[tuple[str, str], SplitKind] = {}
    # key: (device, task_name); a task's config.yaml is yielded before any dataset yaml
    task_kind: dict[tuple[str, str], SplitKind] = {}
    for spec in iter_task_yamls(tasks_dir):
        if not spec.is_dataset_override:
            m = _CONFIG_NAME_TOLERANT.search(spec.yaml_text)
            if m is not None:
                kind = classify_split_yaml(spec.yaml_text)
                out[(spec.task_name, m.group(1))] = kind
                task_kind[(spec.device, spec.task_name)] = kind
            continue
        m = _DATASET_NAME_TOLERANT.search(spec.yaml_text)
        if m is None:
            continue
        ds_study = m.group(1)
        if re.search(r"(split_by|test_split_query|valid_split_by):", spec.yaml_text):
            out[(spec.task_name, ds_study)] = classify_split_yaml(spec.yaml_text)
        elif (spec.device, spec.task_name) in task_kind:
            out[(spec.task_name, ds_study)] = task_kind[(spec.device, spec.task_name)]
    return out


def default_studies_per_task(
    tasks_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Return ``{task_name: [device-default-study, ...]}``.

    Preserves *all* defaults when the same task appears under multiple
    devices (e.g. ``meg/image`` and ``fmri/image``).  Order follows
    sorted device-folder iteration -- the same order
    :func:`iter_task_yamls` yields.
    """
    out: dict[str, list[str]] = {}
    for spec in iter_task_yamls(tasks_dir):
        if spec.is_dataset_override:
            continue
        m = _CONFIG_NAME_STRICT.search(spec.yaml_text)
        if m is not None:
            out.setdefault(spec.task_name, []).append(m.group(1))
    return out


def default_study_names(
    tasks_dir: Path | None = None,
) -> dict[str, str]:
    """Return ``{task_name: default_study_name}`` from task config files.

    Note: the same task name can appear under multiple device folders
    (e.g., ``meg/image`` and ``fmri/image``).  When that happens this
    helper returns one of the defaults arbitrarily; callers that need
    to disambiguate per device should use
    :func:`default_studies_per_task` instead.
    """
    return {
        task: studies[0] for task, studies in default_studies_per_task(tasks_dir).items()
    }


# ---------------------------------------------------------------------------
# Split-kind classifier
# ---------------------------------------------------------------------------


def classify_split_yaml(yaml_text: str) -> SplitKind:
    """Classify a task/dataset yaml into one of three coarse split buckets.

    Applies a cascade over the ``data.study.split`` fields:

    1. ``test_split_query`` (PredefinedSplit) -- string matched against
       ``subject``/``release``/``split == 'eval'`` (cross-subject) vs
       ``session``/``run``/``sequence_id`` (within-subject).
    2. ``split_by`` (SklearnSplit) -- ``subject``/``release`` →
       cross-subject, ``_index`` → random, anything else (timeline,
       session, concept, text, sequence_id) → within-subject.
    3. ``valid_split_by`` as a last-resort proxy for the test split when
       a PredefinedSplit uses an opaque ``col_name`` column.

    The default when no split spec is found is ``within_subject``; this
    default is never actually hit on the current task set, but is safer
    than ``random`` as a generic fallback.
    """
    # 1. test_split_query
    m = re.search(
        r"test_split_query:\s*\"([^\"]+)\"|test_split_query:\s*(null|\S+)",
        yaml_text,
    )
    if m:
        query = m.group(1) or m.group(2)
        if query not in (None, "null"):
            low = query.lower()
            if "subject" in low or "release" in low or "split == 'eval'" in low:
                return "cross_subject"
            if "session" in low or "run " in low or "run=" in low or "sequence_id" in low:
                return "within_subject"

    # 2. split_by (direct SklearnSplit grouping key)
    m = re.search(r"^\s*split_by:\s*(\S+)", yaml_text, re.MULTILINE)
    if m:
        val = m.group(1)
        if val in ("subject", "release"):
            return "cross_subject"
        if val == "_index":
            return "random"
        return "within_subject"

    # 3. valid_split_by as a proxy when PredefinedSplit uses a `col_name`
    #    column without a queryable `test_split_query`.
    m = re.search(r"^\s*valid_split_by:\s*(\S+)", yaml_text, re.MULTILINE)
    if m:
        val = m.group(1)
        if val in ("subject", "release"):
            return "cross_subject"
        if val == "_index":
            return "random"
        return "within_subject"

    return "within_subject"


# ---------------------------------------------------------------------------
# n_examples (JSON, not yaml)
# ---------------------------------------------------------------------------


def _find_neuralbench_stats_path() -> Path | None:
    """Best-effort lookup for ``neuralbench_stats.json``.

    Tries, in order: the ``NEURALBENCH_STATS_JSON`` env var, the
    sibling ``brainai-repo/brainai/bench/plots/neuralbench_stats.json``
    (standard Meta-internal layout), and the importable
    ``brainai.bench.plots`` package location.  Returns ``None`` if
    nothing is found; callers must then gracefully omit the
    example-count badge.
    """
    env = os.environ.get("NEURALBENCH_STATS_JSON")
    if env and Path(env).exists():
        return Path(env)

    here = Path(__file__).resolve()
    sibling = (
        here.parent.parent.parent.parent
        / "brainai-repo"
        / "brainai"
        / "bench"
        / "plots"
        / "neuralbench_stats.json"
    )
    if sibling.exists():
        return sibling

    try:
        import brainai.bench.plots as _bp  # type: ignore[import-not-found]

        pkg_path = Path(_bp.__file__).resolve().parent / "neuralbench_stats.json"
        if pkg_path.exists():
            return pkg_path
    except ImportError:
        pass
    return None


def build_task_n_examples_map() -> dict[tuple[str, str], int]:
    """Return ``{(task_name, study_name): n_examples}`` from the stats JSON.

    Reads the ``variants`` list in ``neuralbench_stats.json``; only the
    core variant (``variant == "_primary"``) is retained per
    ``(task, study)`` pair.  Returns an empty dict when the stats file
    is unavailable -- callers should treat a missing entry as "unknown"
    and omit the example-count badge.
    """
    path = _find_neuralbench_stats_path()
    if path is None:
        return {}
    blob = json.loads(path.read_text("utf8"))
    mapping: dict[tuple[str, str], int] = {}
    for v in blob.get("variants", []):
        task = v.get("task")
        study = v.get("study")
        n = v.get("n_examples")
        if task is None or study is None or n is None:
            continue
        # Keep the core variant for each (task, study) if present;
        # otherwise accept the first seen (max n_subjects entry).
        if (task, study) in mapping and v.get("variant") != "_primary":
            continue
        mapping[(task, study)] = int(n)
    return mapping
