# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import importlib.util
import sys
import typing as tp
from pathlib import Path

import pytest
import torch
from torch import nn

from neuraltrain.models.base import BrainModelBuildContext

from .external import ExternalModel, check_forward, save_instance


class AdaptiveFm(nn.Module):
    """Any channel count, any window length, channels keyed by position."""

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(3, width)

    def forward(self, x: torch.Tensor, channel_positions: torch.Tensor) -> torch.Tensor:
        # (batch, channels, 1) * (batch, channels, width), pooled over channels.
        return (x.mean(-1).unsqueeze(-1) * self.proj(channel_positions)).mean(1)


class NoPositions(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(-1)


class WithVarKwargs(nn.Module):
    def forward(self, x: torch.Tensor, **kwargs: tp.Any) -> torch.Tensor:
        return x.mean(-1)


class FixedWidth(nn.Module):
    """Only works at 64 channels -- stands in for a non-adaptive model."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(64, 4)

    def forward(self, x: torch.Tensor, channel_positions: torch.Tensor) -> torch.Tensor:
        return self.layer(x.transpose(1, 2)).mean(1)


class NotBatchFirst(nn.Module):
    def forward(self, x: torch.Tensor, channel_positions: torch.Tensor) -> torch.Tensor:
        return x.mean(-1).transpose(0, 1)


def context() -> BrainModelBuildContext:
    return BrainModelBuildContext(
        n_spatial_locations=3, n_temporal_samples=5, n_outputs=2
    )


def test_check_forward_requires_channel_positions() -> None:
    check_forward(AdaptiveFm())
    # **kwargs absorbs anything neuralbench passes by keyword.
    check_forward(WithVarKwargs())

    with pytest.raises(ValueError, match="does not accept 'channel_positions'"):
        check_forward(NoPositions())


def test_instance_roundtrip_and_digest_check(tmp_path: Path) -> None:
    model = AdaptiveFm(width=3)
    path, digest = save_instance(model, tmp_path)
    assert path.name == f"{digest}.pt"

    loaded = ExternalModel(pickle_path=str(path), digest=digest).build_from_context(
        context()
    )
    assert isinstance(loaded, AdaptiveFm)
    assert loaded is not model
    torch.testing.assert_close(loaded.proj.weight, model.proj.weight)

    stale = ExternalModel(pickle_path=str(path), digest="0" * 64)
    with pytest.raises(ValueError, match="expected 0{64}"):
        stale.build_from_context(context())

    missing = ExternalModel(pickle_path=str(tmp_path / "absent.pt"), digest=digest)
    with pytest.raises(FileNotFoundError, match="reachable from the"):
        missing.build_from_context(context())


def test_instance_reloads_fresh_per_build(tmp_path: Path) -> None:
    model = AdaptiveFm(width=3)
    path, digest = save_instance(model, tmp_path)
    config = ExternalModel(pickle_path=str(path), digest=digest)

    first = config.build_from_context(context())
    assert isinstance(first, AdaptiveFm)  # also narrows the type below
    with torch.no_grad():
        first.proj.weight.add_(1.0)  # stand in for fine-tuning

    second = config.build_from_context(context())
    assert isinstance(second, AdaptiveFm)
    torch.testing.assert_close(second.proj.weight, model.proj.weight)


def test_a_model_from_an_uninstalled_module_loads_without_that_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker gets neither the caller's ``sys.path`` nor its working directory."""
    (tmp_path / "user_fm.py").write_text(
        "from torch import nn\n\n"
        "class UserFm(nn.Module):\n"
        "    def forward(self, x, channel_positions):\n"
        "        return x.mean(-1)\n"
    )
    monkeypatch.syspath_prepend(tmp_path)
    path, digest = save_instance(importlib.import_module("user_fm").UserFm(), tmp_path)

    del sys.modules["user_fm"]
    monkeypatch.undo()
    assert importlib.util.find_spec("user_fm") is None

    loaded = ExternalModel(pickle_path=str(path), digest=digest).build_from_context(
        context()
    )
    assert type(loaded).__name__ == "UserFm"


def test_build_rejects_a_pickle_it_cannot_use(tmp_path: Path) -> None:
    not_a_module = tmp_path / "not_a_module.pt"
    torch.save("just a string", not_a_module)
    with pytest.raises(TypeError, match="expected a torch.nn.Module"):
        ExternalModel(pickle_path=str(not_a_module), digest="0" * 64).build_from_context(
            context()
        )

    # Re-checked after unpickling, since the worker never saw the caller's check.
    path, digest = save_instance(NoPositions(), tmp_path)
    with pytest.raises(ValueError, match="does not accept 'channel_positions'"):
        ExternalModel(pickle_path=str(path), digest=digest).build_from_context(context())
