# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Preset extractor and study configs, to share cached features across projects.

Starting from these presets lets every project reuse the same cached features
instead of recomputing them in per-project folders. Values live in
``presets.yaml``, and presets are cached in :func:`shared_cache_folder`.

Any change to a parameter entering the cache uid (e.g. ``frequency``, ``filter``,
``model_name``) creates a new cache entry, so only override those when needed.
Infra settings which do not enter the uid (cluster, partition, number of jobs...)
and fields excluded from the cache uid (e.g. ``offset``, ``baseline``, ``clamp``,
fMRI ``query``) can be freely overridden. Overrides changing the uid must be
cached in a project folder, unless the variant is meant to be shared by several
projects (``allow_new_cache=True``).

Presets are selected by id (see :data:`EXTRACTORS`), e.g. ``meg_200hz`` or
``meg_100hz``, ``fmri_fs5`` or ``fmri_schaefer1000``.

Computations run on slurm when available (``cluster="auto"``), locally otherwise.

Usage::

    from neuralset import presets

    meg = presets.extractor("meg_200hz", allow_missing=True)
    image = presets.extractor(
        "image_dinov2_large", **{"infra.slurm_constraint": "volta32gb"}
    )
    study = presets.study("Hebart2023ThingsMeg", path=..., query="subject_index < 2")

Word embeddings in context are covered by ``text_qwen3_2hz``
(``HuggingFaceText``), which reads the ``AlignedText`` events of
``neuralset.events.transforms.AddAlignedTexts`` (one per Text, the default).
Audio, video and audio-visual extractors (``QwenOmni``) embed whole events in
windows (``context``) instead of relying on ``ChunkEvents``, so they can be
shared.
"""

import copy
import os
import typing as tp
from pathlib import Path

import yaml
from exca import ConfDict

from ..base import CACHE_FOLDER
from ..events import Study
from ..extractors import BaseExtractor

_PRESETS: dict[str, tp.Any] = yaml.safe_load(
    Path(__file__).with_name("presets.yaml").read_text(encoding="utf8")
)
EXTRACTORS = tuple(_PRESETS["extractors"])


def shared_cache_folder() -> Path:
    """Folder caching the presets: ``$NEURALSET_PRESETS_CACHE_FOLDER``, e.g. a
    folder shared by a team, else ``~/.cache/neuralset/presets``."""
    return Path(
        os.environ.get("NEURALSET_PRESETS_CACHE_FOLDER", CACHE_FOLDER / "presets")
    )


def _extractor_uid(config: ConfDict) -> str:
    extractor = BaseExtractor.model_validate(config)
    return extractor.infra.uid()  # type: ignore[attr-defined]


def _study_uid(config: ConfDict) -> str:
    study = Study(**config)
    uid = ConfDict.from_model(study, uid=True, exclude_defaults=True)
    uid.pop("query", None)  # only selects which timelines are loaded
    return uid.to_uid()


def _check_cache_uid(
    preset: str,
    reference: ConfDict,
    config: ConfDict,
    uid: tp.Callable[[ConfDict], str],
    updates: tp.Iterable[str],
    folder_hint: str,
) -> None:
    ref, new = uid(reference), uid(config)
    if ref != new:
        raise ValueError(
            f"Updates {sorted(updates)} change the cache uid of preset {preset!r} "
            f"(from {ref!r} to {new!r}), which would add new entries to the shared "
            "cache folder. Only override fields excluded from the cache uid, or set "
            f"{folder_hint} to a project folder. Pass allow_new_cache=True only if "
            "this variant is meant to be shared by several projects."
        )


def extractor(
    preset: str, /, *, allow_new_cache: bool = False, **updates: tp.Any
) -> ConfDict:
    """Config of the extractor preset with id ``preset`` (one of :data:`EXTRACTORS`).

    ``updates`` are merged recursively into the preset config (see
    :meth:`exca.ConfDict.update`), e.g. ``infra={"slurm_partition": ...}``.
    Updates changing the cache uid raise an error while the cache stays in the
    shared folder: set ``infra={"folder": ...}`` to a project folder instead.
    ``allow_new_cache=True`` adds the new entry to the shared folder anyway, only
    for variants meant to be shared by several projects.
    """
    if preset not in EXTRACTORS:
        raise ValueError(f"Unknown extractor preset {preset!r}, use one of {EXTRACTORS}")
    reference = ConfDict(copy.deepcopy(_PRESETS["extractors"][preset]))
    reference["infra.folder"] = str(shared_cache_folder())
    config = copy.deepcopy(reference)
    config.update(copy.deepcopy(updates))
    folder = config.get("infra.folder")
    in_shared = folder is not None and Path(folder) == shared_cache_folder()
    if updates and in_shared and not allow_new_cache:
        _check_cache_uid(
            preset, reference, config, _extractor_uid, updates, 'infra={"folder": ...}'
        )
    return config


def study(name: str, /, *, allow_new_cache: bool = False, **updates: tp.Any) -> ConfDict:
    """Study config whose per-timeline loading is cached in :func:`shared_cache_folder`.

    ``updates`` are merged recursively (e.g. ``path``, ``query``,
    ``timelines={"infra": {...}}``), except that setting ``timelines.infra.backend``
    replaces the whole preset ``timelines.infra``, as options differ between
    backends. The study-level ``infra`` is left to the caller as it depends on
    ``query``.

    Cached events hold absolute filepaths while ``path`` is excluded from the cache
    uid, so only share the folder between users reading the study from the same
    path. As in :func:`extractor`, updates changing the cache uid (e.g.
    ``version``) raise an error while timelines stay in the shared folder.
    """
    reference = ConfDict(name=name, **copy.deepcopy(_PRESETS["study"]))
    reference["timelines.infra.folder"] = str(shared_cache_folder())
    overrides = ConfDict(copy.deepcopy(updates))
    config = copy.deepcopy(reference)
    infra = overrides.get("timelines", {}).get("infra")
    if isinstance(infra, dict) and "backend" in infra:
        del config["timelines"]["infra"]
    config.update(overrides)
    infra = config["timelines"].get("infra")
    in_shared = (
        isinstance(infra, dict)
        and infra.get("folder") is not None
        and Path(infra["folder"]) == shared_cache_folder()
    )
    # uid-excluded fields skip the check, which imports the study class
    in_uid = set(overrides) - {"path", "query", "timelines", "infra"}
    if in_shared and in_uid and not allow_new_cache:
        reference["path"] = config.get("path", CACHE_FOLDER)
        _check_cache_uid(
            name,
            reference,
            config,
            _study_uid,
            updates,
            'timelines={"infra": {"folder": ...}}',
        )
    return config
