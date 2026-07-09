# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import typing as tp
from collections import OrderedDict
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

from neuraltrain.models.base import BaseModelConfig

from .model_factory import build_brain_model


class _Loader:
    def __init__(self, batch: SimpleNamespace, channel_names: list[str]) -> None:
        self._batch = batch
        self.dataset = SimpleNamespace(
            extractors={
                "neuro": SimpleNamespace(
                    _channels=OrderedDict((name, object()) for name in channel_names)
                )
            }
        )

    def __iter__(self) -> tp.Iterator[SimpleNamespace]:
        yield self._batch


def test_build_brain_model_passes_channel_names_when_requested(monkeypatch) -> None:
    """Generic model configs that declare ``ch_names_required`` should receive
    the dataset channel names during build.

    Some custom models are plain ``BaseModelConfig`` instances rather than
    braindecode configs, so they bypass ``build_braindecode_model`` and would
    otherwise miss the per-dataset channel-name metadata they need during
    construction.
    """

    build_calls: list[dict[str, object]] = []

    class _RecordingConfig:
        ch_names_required = True

        def build(self, **kwargs: object) -> nn.Module:
            build_calls.append(kwargs)
            return nn.Identity()

    batch = SimpleNamespace(
        data={
            "neuro": torch.randn(2, 3, 16),
            "target": torch.randn(2, 4),
        }
    )
    loader = _Loader(batch, ["Fp1", "Cz", "O2"])

    monkeypatch.setattr(
        "neuralbench.model_factory.summary",
        lambda *args, **kwargs: SimpleNamespace(total_params=0, trainable_params=0),
    )

    build_brain_model(
        brain_model_config=tp.cast(BaseModelConfig, _RecordingConfig()),
        downstream_model_wrapper=None,
        pretrained_weights_fname=None,
        train_loader=tp.cast(DataLoader[tp.Any], loader),
    )

    assert len(build_calls) == 1
    assert build_calls[0]["n_in_channels"] == 3
    assert build_calls[0]["n_outputs"] == 4
    assert build_calls[0]["ch_names"] == ["Fp1", "Cz", "O2"]
