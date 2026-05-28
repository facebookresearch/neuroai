# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build every classifier-shaped braindecode model via
:func:`neuralbench.model_factory.build_braindecode_model`.

Drives the catalogue from ``braindecode.models.util.models_dict`` (which
only includes ``EEGModuleMixin`` subclasses, so naturally excludes TCN
and helper classes), minus families covered elsewhere (foundation
models, sleep stagers, EMG/MEG, signal-JEPA, interpolated wrappers).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from braindecode.models.util import models_dict
from neuraltrain.models import base as bd_base

from neuralbench.model_factory import build_braindecode_model


# Foundation models with neuralbench YAMLs + dedicated pretrained-weight
# pipelines (exp1 probe sweep). Skipped here because they require fixed
# channel layouts / pretrained checkpoints not present in this fixture.
ALREADY_IMPLEMENTED = {
    "BENDR", "BIOT", "Labram",
    "InterpolatedBENDR", "InterpolatedBIOT",
    "InterpolatedLaBraM", "InterpolatedSignalJEPA",
}
# Architectures incompatible with the EEG fixture (22-ch, 1000 samples).
NON_EEG_CLASSIFIERS = {
    "AttnSleep", "USleep",                          # sleep-specific input
    "EMG2QwertyNet", "MetaNeuromotorHand",          # EMG / MEG modality
    "SignalJEPA", "SignalJEPA_Contextual",          # self-supervised, no head
}
CLASSIFIERS = sorted(
    name for name in models_dict
    if name not in ALREADY_IMPLEMENTED and name not in NON_EEG_CLASSIFIERS
)

N_CHANS, N_TIMES, N_OUTPUTS, SFREQ = 22, 1000, 4, 120.0
CH_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
            "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
            "Fz", "Cz", "Pz", "FC1", "FC2", "CP1"]


def _loader() -> SimpleNamespace:
    extractor = SimpleNamespace(frequency=SFREQ, _channels=dict.fromkeys(CH_NAMES))
    return SimpleNamespace(dataset=SimpleNamespace(extractors={"neuro": extractor}))


def test_build_raises_when_runtime_kwargs_overlap_config() -> None:
    """If a YAML pre-sets one of the six auto-injected params, the user is
    redefining a data-derived value -- ``BaseBrainDecodeModel.build()`` must
    raise so the mismatch surfaces instead of being silently overridden."""
    config = bd_base.EEGNet(kwargs={"sfreq": 250.0})
    with pytest.raises(ValueError, match="overlap"):
        build_braindecode_model(
            brain_model_config=config,
            downstream_model_wrapper=None,
            train_loader=_loader(),
            n_in_channels=N_CHANS,
            n_times=N_TIMES,
            n_outputs=N_OUTPUTS,
        )


@pytest.mark.parametrize("name", CLASSIFIERS)
def test_build_braindecode_classifier(name: str) -> None:
    config_cls = getattr(bd_base, name, None)
    if config_cls is None:
        pytest.skip(f"{name} not registered as a BaseBrainDecodeModel config")
    model = build_braindecode_model(
        brain_model_config=config_cls(),
        downstream_model_wrapper=None,
        train_loader=_loader(),
        n_in_channels=N_CHANS,
        n_times=N_TIMES,
        n_outputs=N_OUTPUTS,
    )
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, N_CHANS, N_TIMES))
    assert isinstance(out, torch.Tensor) and out.shape[0] == 2
