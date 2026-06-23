# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build every classifier-shaped braindecode model via
:func:`neuralbench.model_factory.build_braindecode_model`.

Drives the catalogue from ``braindecode.models.util.models_dict`` (only
``EEGModuleMixin`` subclasses).  ``ALREADY_IMPLEMENTED`` is derived from
the dedicated YAML configs under ``neuralbench/models/`` so the set
stays in sync automatically; ``NON_EEG_CLASSIFIERS`` is a short
hand-maintained list of upstream classes that don't take 22-ch EEG
input (sleep stagers, EMG/MEG, self-supervised heads).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from braindecode.models.util import models_dict
from torch.utils.data import DataLoader

import neuraltrain.models  # noqa: F401  triggers BaseBrainDecodeModel subclass registration
from neuralbench.model_factory import build_braindecode_model
from neuralbench.registry import load_yaml_config
from neuraltrain.models import base as bd_base


def _all_brain_model_configs() -> dict[str, type[bd_base.BaseBrainDecodeModel]]:
    out: dict[str, type[bd_base.BaseBrainDecodeModel]] = {}
    stack = [bd_base.BaseBrainDecodeModel]
    while stack:
        cls = stack.pop()
        out[cls.__name__] = cls
        stack.extend(cls.__subclasses__())
    return out


def _foundation_class_names() -> set[str]:
    """Braindecode class names already wired up via dedicated YAMLs."""
    yaml_dir = Path(__file__).parent / "models"
    configs = _all_brain_model_configs()
    names = set()
    for path in yaml_dir.glob("*.yaml"):
        cfg = load_yaml_config(path) or {}
        bm = cfg.get("brain_model_config") or {}
        if not isinstance(bm, dict) or "from_pretrained_name" not in bm:
            continue
        cfg_cls = configs.get(bm.get("name", ""))
        if cfg_cls is None:
            continue
        cfg_cls._ensure_model_class()
        underlying = getattr(cfg_cls, "_MODEL_CLASS", None)
        if underlying is not None and underlying.__name__ in models_dict:
            names.add(underlying.__name__)
    # The Interpolated* wrappers reuse the same pretrained weights and are
    # implicitly covered by the foundation entries above.
    names.update(n for n in models_dict if n.startswith("Interpolated"))
    return names


ALREADY_IMPLEMENTED = _foundation_class_names()
# Upstream classes that don't fit a 22-ch EEG fixture (sleep models with
# minute-scale inputs, EMG/MEG-specific channel layouts, self-supervised
# pretext nets with no classifier head).
NON_EEG_CLASSIFIERS = {
    "AttnSleep",
    "USleep",
    "EMG2QwertyNet",
    "MetaNeuromotorHand",
    "SignalJEPA",
    "SignalJEPA_Contextual",
}
CLASSIFIERS = sorted(
    name
    for name in models_dict
    if name not in ALREADY_IMPLEMENTED and name not in NON_EEG_CLASSIFIERS
)

N_CHANS, N_TIMES, N_OUTPUTS, SFREQ = 22, 1000, 4, 120.0
CH_NAMES = (
    "Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T7 T8 P7 P8 Fz Cz Pz FC1 FC2 CP1".split()
)


def _loader() -> DataLoader:
    extractor = SimpleNamespace(frequency=SFREQ, _channels=dict.fromkeys(CH_NAMES))
    loader = SimpleNamespace(dataset=SimpleNamespace(extractors={"neuro": extractor}))
    return cast(DataLoader, loader)


def _build(config: bd_base.BaseBrainDecodeModel) -> torch.nn.Module:
    return build_braindecode_model(
        brain_model_config=config,
        downstream_model_wrapper=None,
        train_loader=_loader(),
        n_in_channels=N_CHANS,
        n_times=N_TIMES,
        n_outputs=N_OUTPUTS,
    )


def test_build_raises_when_runtime_kwargs_overlap_config() -> None:
    """If a YAML pre-sets one of the six auto-injected params, the user is
    redefining a data-derived value -- ``BaseBrainDecodeModel.build()`` must
    raise so the mismatch surfaces instead of being silently overridden."""
    config = getattr(bd_base, "EEGNet")(kwargs={"sfreq": 250.0})
    with pytest.raises(ValueError, match="kwargs overlap with config kwargs for keys"):
        _build(config)


@pytest.mark.parametrize("name", CLASSIFIERS)
def test_build_braindecode_classifier(name: str) -> None:
    config_cls = getattr(bd_base, name, None)
    if config_cls is None:
        pytest.skip(f"{name} not registered as a BaseBrainDecodeModel config")
    model = _build(config_cls())
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, N_CHANS, N_TIMES))
    assert isinstance(out, torch.Tensor) and out.shape[0] == 2
