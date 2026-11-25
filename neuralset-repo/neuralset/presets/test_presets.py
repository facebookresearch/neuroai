# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest

import neuralset as ns
from neuralset import presets
from neuralset.extractors.hf import HuggingFaceMixin


@pytest.mark.parametrize("preset", presets.EXTRACTORS)
@HuggingFaceMixin.no_snapshot_download()
def test_extractor_presets_instantiate(preset: str) -> None:
    config = presets.extractor(preset)
    extractor = ns.extractors.BaseExtractor.model_validate(config)
    assert extractor.infra.uid()  # type: ignore[attr-defined]


@pytest.mark.parametrize("preset", presets.EXTRACTORS)
@HuggingFaceMixin.no_snapshot_download()
def test_neuro_presets_cache_as_float16(preset: str) -> None:
    extractor = ns.extractors.BaseExtractor.model_validate(presets.extractor(preset))
    if "store_dtype" in type(extractor).model_fields:
        assert extractor.store_dtype == "float16", (  # type: ignore[attr-defined]
            f"Preset {preset!r} must cache as float16: store_dtype enters the "
            "cache uid, so float32 would create a duplicate entry"
        )


def test_study_preset_instantiates(tmp_path: Path) -> None:
    config = presets.study("Test2023Meg", path=tmp_path)
    study = ns.Study(**config)
    assert study.timelines.infra is not None


def test_presets_guard_the_shared_cache(tmp_path: Path) -> None:
    presets.extractor("meg_200hz", clamp=5.0)
    with pytest.raises(ValueError, match="change the cache uid"):
        presets.extractor("meg_200hz", frequency=150.0)
    presets.extractor("meg_200hz", frequency=150.0, infra={"folder": str(tmp_path)})
    with pytest.raises(ValueError, match="change the cache uid"):
        presets.study("Test2023Meg", path=tmp_path, version="other")
