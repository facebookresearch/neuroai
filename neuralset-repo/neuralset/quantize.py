# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import typing as tp

import numpy as np
from exca import cachedict


@cachedict.DumpContext.register
class Float16StoredArray:
    """Array cached as float16, read back as float32 in its original units."""

    def __init__(self, data: tp.Any, gains: np.ndarray | None = None) -> None:
        self._data = data  # float32 until dumped, the stored float16 once loaded
        # zero-stride: gains follow data through slices and transposes
        self._gains = None if gains is None else np.broadcast_to(gains, data.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def dtype(self) -> np.dtype[tp.Any]:
        return np.dtype(np.float32)

    def __repr__(self) -> str:
        return f"Float16StoredArray(shape={self.shape}, dtype={self.dtype})"

    def __getattr__(self, name: str) -> tp.Any:
        raise AttributeError(
            f"Float16StoredArray has no {name!r}: gains can only be undone by a "
            f"read, so call np.asarray() first, then {name!r} on its float32 result."
        )

    def __getitem__(self, key: tp.Any) -> tp.Any:
        data = self._data[key]
        gains = None if self._gains is None else self._gains[key]
        if not np.ndim(data):
            return data if gains is None else data / gains
        return Float16StoredArray(data, gains)

    def transpose(self, *axes: tp.Any) -> Float16StoredArray:
        gains = None if self._gains is None else self._gains.transpose(*axes)
        return Float16StoredArray(self._data.transpose(*axes), gains)

    def __array__(
        self, dtype: np.dtype[tp.Any] | None = None, copy: bool | None = None
    ) -> np.ndarray:
        if self._gains is None:
            return np.array(self._data, dtype=dtype, copy=copy)
        if copy is False:
            raise ValueError("Float16StoredArray cannot undo its gains without a copy")
        out = np.divide(np.asarray(self._data), self._gains, dtype=np.float32)
        return out if dtype is None else out.astype(dtype)

    def __dump_info__(self, ctx: cachedict.DumpContext) -> dict[str, tp.Any]:
        data = np.asarray(self)
        # powers of two invert exactly; 2**10 clears float16 subnormals (6.1e-5)
        _, exponent = np.frexp(np.abs(data).max(axis=0, keepdims=True, initial=0.0))
        gains = np.ldexp(np.float32(1.0), np.clip(10 - exponent, -126, 126))
        scaled = (data * gains).astype(np.float16)
        info = cachedict.handlers.MemmapArray.__dump_info__(ctx, scaled)
        info["gains"] = ctx.dump(gains)
        return info

    @classmethod
    def __load_from_info__(
        cls, ctx: cachedict.DumpContext, *, gains: dict[str, tp.Any], **array: tp.Any
    ) -> Float16StoredArray:
        loader = cachedict.handlers.ContiguousMemmapArray
        return cls(loader.__load_from_info__(ctx, **array), np.asarray(ctx.load(gains)))
