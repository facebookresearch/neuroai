# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative, dotted-path preprocessing transforms.

A transform is declared as a dotted import path with optional keyword params::

    {"name": "my_pkg.my_module.my_transform", "params": {"gain": 2.0}}

or, for a no-argument transform, just the dotted string. Transforms are
resolved and applied in order; each must return the (possibly new) value.

Two consumers use this module:

* Extractors (``neuralset.extractors``) apply ``"raw"`` transforms to the MNE
  ``Raw`` before their pipeline and ``"array"`` transforms to the extracted
  tensor (see ``apply_transforms`` / ``apply_array_transforms``).
* Experiments fit ``train_set`` transformers (scikit-learn ``fit``/``transform``
  objects) on the training split only and apply them to every loader's
  ``neuro`` batch at iteration time (see ``fit_train_set_transformers`` /
  ``wrap_loader``).
"""

from __future__ import annotations

import copy
import importlib
import typing as tp

import numpy as np
import pydantic
import torch
from torch.utils.data import DataLoader


class Transform(pydantic.BaseModel):
    """A dotted-path callable transform with optional keyword params.

    ``name`` is an import path (e.g. ``"my_pkg.my_module.my_transform"``);
    ``params`` are passed as keyword arguments when the transform is applied
    (stage transforms) or instantiated (train_set transformers).
    """

    model_config = pydantic.ConfigDict(extra="forbid")  # exca config-uid requirement

    name: str
    params: dict[str, tp.Any] = {}


def resolve_transform(name: str) -> tp.Callable[..., tp.Any]:
    """Import and return the callable/class named by a dotted path."""
    if not isinstance(name, str) or "." not in name:
        raise ValueError(
            f"preprocessing transform must be a dotted import path, got {name!r}"
        )
    module_name, attr = name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr)


def _name_and_params(
    entry: "Transform | str | dict[str, tp.Any]",
) -> tuple[str, dict[str, tp.Any]]:
    if isinstance(entry, Transform):
        return entry.name, entry.params
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict):
        name = entry.get("name")
        params = entry.get("params") or {}
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"preprocessing entry needs a non-empty name, got {entry!r}"
            )
        if not isinstance(params, dict):
            raise TypeError(
                f"preprocessing params must be a mapping, got {type(params).__name__}"
            )
        return name, params
    raise TypeError(
        f"preprocessing entries must be strings or mappings, got {type(entry).__name__}"
    )


def _coerce_list(entries: tp.Any) -> list[tp.Any]:
    if entries is None:
        return []
    if isinstance(entries, (str, dict, Transform)):
        return [entries]
    return list(entries)


def apply_transforms(value: tp.Any, entries: tp.Any) -> tp.Any:
    """Apply a list of declarative transforms to ``value`` in order."""
    for entry in _coerce_list(entries):
        name, params = _name_and_params(entry)
        result = resolve_transform(name)(value, **params)
        if result is None:
            raise ValueError(f"preprocessing transform returned None: {entry!r}")
        value = result
    return value


def apply_array_transforms(tensor: tp.Any, entries: tp.Any) -> tp.Any:
    """Apply transforms to an extracted tensor, preserving device and dtype.

    Array transforms operate on a NumPy view (the common case for array ops);
    the result is converted back to a tensor on the original device/dtype.
    Non-tensor inputs are transformed in place without conversion.
    """
    if not _coerce_list(entries):
        return tensor
    if not torch.is_tensor(tensor):
        return apply_transforms(tensor, entries)
    device, dtype = tensor.device, tensor.dtype
    array = apply_transforms(tensor.detach().cpu().numpy(), entries)
    return torch.as_tensor(array, dtype=dtype, device=device)


# --- train-set-fitted preprocessing -----------------------------------------


def instantiate_train_set_transformers(specs: tp.Any) -> list[tp.Any]:
    """Instantiate ``fit``/``transform`` objects from declarative specs."""
    transformers = []
    for entry in _coerce_list(specs):
        name, params = _name_and_params(entry)
        transformer = resolve_transform(name)(**params)
        if not (hasattr(transformer, "fit") and hasattr(transformer, "transform")):
            raise TypeError(
                "train_set preprocessing entries must instantiate fit/transform "
                f"objects, got {transformer!r}"
            )
        transformers.append(transformer)
    return transformers


def _batch_neuro_numpy(batch: tp.Any) -> np.ndarray:
    data = getattr(batch, "data", None)
    if not isinstance(data, dict) or "neuro" not in data:
        raise KeyError("train_set preprocessing requires batches with data['neuro']")
    neuro = data["neuro"]
    if torch.is_tensor(neuro):
        return neuro.detach().cpu().numpy()
    return np.asarray(neuro)


def _sequential_loader(loader: tp.Any) -> tp.Any:
    """A non-shuffled iterator over the same dataset.

    Used to fit transformers without consuming the training loader's RNG.
    """
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return loader
    collate_fn = getattr(loader, "collate_fn", None) or getattr(
        dataset, "collate_fn", None
    )
    return DataLoader(
        dataset,
        batch_size=getattr(loader, "batch_size", None) or 1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )


def fit_train_set_transformers(train_loader: tp.Any, specs: tp.Any) -> list[tp.Any]:
    """Fit declarative train_set transformers on the training split only."""
    transformers = instantiate_train_set_transformers(specs)
    if not transformers:
        return []
    arrays = [_batch_neuro_numpy(batch) for batch in _sequential_loader(train_loader)]
    if not arrays:
        raise ValueError("cannot fit train_set preprocessing on an empty train loader")
    train = np.concatenate(arrays, axis=0)
    for transformer in transformers:
        transformer.fit(train)
        train = transformer.transform(train)
    return transformers


def _apply_fitted(neuro: tp.Any, transformers: list[tp.Any]) -> tp.Any:
    if torch.is_tensor(neuro):
        device, dtype = neuro.device, neuro.dtype
        array = neuro.detach().cpu().numpy()
        for transformer in transformers:
            array = transformer.transform(array)
        return torch.as_tensor(array, dtype=dtype, device=device)
    array = np.asarray(neuro)
    for transformer in transformers:
        array = transformer.transform(array)
    return array


def _transform_batch(batch: tp.Any, transformers: list[tp.Any]) -> tp.Any:
    data = getattr(batch, "data", None)
    if not isinstance(data, dict) or "neuro" not in data:
        return batch
    new_data = dict(data)
    new_data["neuro"] = _apply_fitted(data["neuro"], transformers)
    segments = getattr(batch, "segments", None)
    try:
        return batch.__class__(data=new_data, segments=segments)
    except (TypeError, ValueError):
        clone = copy.copy(batch)
        clone.data = new_data
        return clone


class _TransformingLoader:
    """DataLoader proxy applying fitted train_set transforms to neuro batches."""

    def __init__(self, loader: tp.Any, transformers: list[tp.Any]) -> None:
        self._loader = loader
        self._transformers = transformers

    def __iter__(self) -> tp.Iterator[tp.Any]:
        for batch in self._loader:
            yield _transform_batch(batch, self._transformers)

    def __len__(self) -> int:
        return len(self._loader)

    def __getattr__(self, name: str) -> tp.Any:
        return getattr(self._loader, name)


def wrap_loader(loader: tp.Any, transformers: list[tp.Any]) -> tp.Any:
    """Wrap ``loader`` so fitted train_set transforms apply at iteration time."""
    if loader is None or not transformers:
        return loader
    return _TransformingLoader(loader, transformers)
