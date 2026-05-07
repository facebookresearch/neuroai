# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Builders for neuralyze SlidingWindow and TRFScoring from validation configs."""

from __future__ import annotations

import typing as tp
from pathlib import Path

from .config import StudyValidation

_INFRA_PROPAGATE_KEYS = (
    "cluster",
    "slurm_partition",
    "timeout_min",
    "folder",
    "mode",
    "cpus_per_task",
    "mem_gb",
)


def _build_sliding_window(
    study_name: str,
    validation: StudyValidation,
    study_folder: Path,
    cache_dir: Path | None = None,
    query: str | None = None,
    infra: dict[str, tp.Any] | None = None,
) -> tp.Any:
    """Construct a :class:`neuralyze.SlidingWindow` from a validation config."""
    from neuralyze import SlidingWindow

    study_config: dict[str, tp.Any] = {
        "name": study_name,
        "path": str(study_folder),
        # Run timeline loading sequentially to avoid a known upstream bug in
        # ``neuralset.events.study.Study.__setstate__`` that crashes processpool
        # workers with ``AttributeError: ... has no attribute 'path'`` (see
        # fairinternal/brainai#2822). Timeline loading is I/O-bound and typically
        # fine sequentially for validation runs.
        "infra_timelines": {"cluster": None},
    }
    if query is not None:
        study_config["query"] = query
    if cache_dir is not None:
        infra = dict(infra) if infra is not None else {}
        infra.setdefault("folder", str(cache_dir))
        retry = infra.pop("retry", False)
        study_infra: dict[str, tp.Any] = {"backend": "Cached", "folder": str(cache_dir)}
        if retry:
            study_infra["mode"] = "retry"
        study_config["infra"] = study_infra

    neuro_dict = dict(validation.neuro)
    extractor_dict = dict(validation.extractor)
    if infra is not None:
        for d in (neuro_dict, extractor_dict):
            sub_infra = d.setdefault("infra", {})
            for key in _INFRA_PROPAGATE_KEYS:
                if key in infra:
                    sub_infra.setdefault(key, infra[key])

    data_config: dict[str, tp.Any] = {
        "study": study_config,
        "start": validation.start,
        "stop": validation.stop,
        "event_type": validation.event_type,
        "neuro": neuro_dict,
        "extractor": extractor_dict,
    }

    sw_kwargs: dict[str, tp.Any] = {
        "data": data_config,
        "mode": validation.mode,
        "model": dict(validation.model),
        "scoring": dict(validation.scoring),
    }
    if validation.cv:
        sw_kwargs["cv"] = dict(validation.cv)
    if validation.train_query is not None:
        sw_kwargs["train_query"] = validation.train_query
    if validation.test_query:
        sw_kwargs["test_query"] = list(validation.test_query)

    if infra is not None:
        sw_kwargs["infra"] = infra

    return SlidingWindow(**sw_kwargs)


def _build_trf_scoring(
    study_name: str,
    validation: StudyValidation,
    study_folder: Path,
    cache_dir: Path | None = None,
    query: str | None = None,
    infra: dict[str, tp.Any] | None = None,
) -> tp.Any:
    """Construct a :class:`neuralyze.trf.TRFScoring` from a validation config.

    Reuses ``validation.neuro``, ``validation.extractor``, ``validation.scoring``,
    and ``validation.cv`` from the parent :class:`StudyValidation`, overriding
    the extractor's ``aggregation`` key with ``validation.trf.aggregation``.  The
    CV config is also patched to include ``group="filepath"`` if not already set,
    which is required for TRF to avoid data leakage across autocorrelated segments.

    ``mode`` is always forced to ``"encod"`` because a TRF is inherently an
    encoding model (features → brain), regardless of the parent
    :class:`StudyValidation`'s ``mode``.

    """
    from neuralyze.trf import TRFScoring

    assert validation.trf is not None, "trf config must be set"
    trf_cfg = validation.trf

    study_config: dict[str, tp.Any] = {
        "name": study_name,
        "path": str(study_folder),
        "infra_timelines": {"cluster": None},
    }
    if query is not None:
        study_config["query"] = query
    if cache_dir is not None:
        infra = dict(infra) if infra is not None else {}
        infra.setdefault("folder", str(cache_dir))
        retry = infra.pop("retry", False)
        study_infra: dict[str, tp.Any] = {"backend": "Cached", "folder": str(cache_dir)}
        if retry:
            study_infra["mode"] = "retry"
            infra["mode"] = "retry"
        study_config["infra"] = study_infra

    neuro_dict = dict(validation.neuro)
    extractor_dict = dict(validation.extractor)
    extractor_dict["aggregation"] = trf_cfg.aggregation
    if "frequency" in neuro_dict:
        extractor_dict.setdefault("frequency", neuro_dict["frequency"])
    if infra is not None:
        for d in (neuro_dict, extractor_dict):
            sub_infra = d.setdefault("infra", {})
            for key in _INFRA_PROPAGATE_KEYS:
                if key in infra:
                    sub_infra.setdefault(key, infra[key])

    data_config: dict[str, tp.Any] = {
        "study": study_config,
        "event_type": validation.event_type,
        "neuro": neuro_dict,
        "extractor": extractor_dict,
    }

    cv_dict = dict(validation.cv)
    if "group" not in cv_dict:
        cv_dict["group"] = "filepath"
        cv_dict["shuffle"] = False

    # Force mean=False so scoring returns per-channel values (needed for topomap)
    trf_scoring = dict(validation.scoring)
    trf_scoring["mean"] = False

    trf_kwargs: dict[str, tp.Any] = {
        "data": data_config,
        "trf": {"tmin": trf_cfg.tmin, "tmax": trf_cfg.tmax},
        "mode": "encod",
        "scoring": trf_scoring,
        "cv": cv_dict,
    }
    if infra is not None:
        trf_kwargs["infra"] = infra

    return TRFScoring(**trf_kwargs)
