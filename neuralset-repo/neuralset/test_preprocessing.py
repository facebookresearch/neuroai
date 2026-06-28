# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neuralset import preprocessing as pp


def test_resolve_transform_requires_dotted_path():
    with pytest.raises(ValueError, match="dotted import path"):
        pp.resolve_transform("not_dotted")


def test_apply_transforms_runs_in_order_with_params(monkeypatch):
    def add(value, *, k):
        return value + k

    def double(value):
        return value * 2

    monkeypatch.setattr(
        pp,
        "resolve_transform",
        lambda name: {"m.add": add, "m.double": double}[name],
    )
    out = pp.apply_transforms(
        1, [{"name": "m.add", "params": {"k": 3}}, "m.double"]
    )
    assert out == 8  # (1 + 3) * 2


def test_apply_transforms_rejects_none_result(monkeypatch):
    monkeypatch.setattr(pp, "resolve_transform", lambda name: lambda v: None)
    with pytest.raises(ValueError, match="returned None"):
        pp.apply_transforms(1, ["m.bad"])


def test_apply_transforms_accepts_single_mapping(monkeypatch):
    monkeypatch.setattr(pp, "resolve_transform", lambda name: lambda v: v + 1)
    assert pp.apply_transforms(1, {"name": "m.inc"}) == 2


def test_apply_transforms_accepts_transform_model(monkeypatch):
    monkeypatch.setattr(pp, "resolve_transform", lambda name: lambda v, *, k: v + k)
    out = pp.apply_transforms(1, [pp.Transform(name="m.add", params={"k": 4})])
    assert out == 5


def test_transform_requires_name():
    with pytest.raises(Exception):  # pydantic ValidationError
        pp.Transform(params={"k": 1})


def test_apply_array_transforms_preserves_tensor_dtype_and_device(monkeypatch):
    monkeypatch.setattr(pp, "resolve_transform", lambda name: lambda a: a + 1.0)
    tensor = torch.zeros(2, 3, dtype=torch.float64)
    out = pp.apply_array_transforms(tensor, ["m.inc"])
    assert torch.is_tensor(out)
    assert out.dtype == torch.float64
    assert torch.allclose(out, torch.ones(2, 3, dtype=torch.float64))


def test_apply_array_transforms_noop_without_entries():
    tensor = torch.arange(4)
    assert pp.apply_array_transforms(tensor, None) is tensor


def test_fit_train_set_fits_train_only_and_wraps_loaders(monkeypatch):
    class Batch:
        def __init__(self, neuro, segments):
            self.data = {"neuro": neuro}
            self.segments = segments

    class Loader:
        def __init__(self, batches):
            self.batches = batches
            self.dataset = None  # forces _sequential_loader fallback

        def __iter__(self):
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)

    class Standardizer:
        fitted_means: list = []

        def fit(self, X, y=None):
            self.mean_ = float(np.mean(X))
            Standardizer.fitted_means.append(self.mean_)
            return self

        def transform(self, X):
            return np.asarray(X) - self.mean_

    monkeypatch.setattr(pp, "resolve_transform", lambda name: Standardizer)

    train = Loader([Batch(np.array([[1.0], [3.0]]), ["a", "b"])])
    val_batch = Batch(np.array([[10.0]]), ["v"])
    val = Loader([val_batch])

    transformers = pp.fit_train_set_transformers(train, [{"name": "m.std"}])
    wrapped = pp.wrap_loader(val, transformers)
    out_batch = next(iter(wrapped))

    assert Standardizer.fitted_means == [2.0]
    assert out_batch is not val_batch
    assert np.allclose(out_batch.data["neuro"], np.array([[8.0]]))
    assert np.allclose(val_batch.data["neuro"], np.array([[10.0]]))  # original intact


def test_wrap_loader_is_noop_without_transformers():
    sentinel = SimpleNamespace()
    assert pp.wrap_loader(sentinel, []) is sentinel
