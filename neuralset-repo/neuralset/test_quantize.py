# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import numpy as np
import pytest
from exca import cachedict

from . import base, quantize


def _roundtrip(folder: Path, data: tp.Any, frequency: float) -> base.TimedArray:
    ta = base.TimedArray(data=data, frequency=frequency, start=0.0)
    ctx = cachedict.DumpContext(folder)
    with ctx:
        info = ctx.dump_entry("key", ta)
    return ctx.load(info["content"])


@pytest.mark.parametrize("shape", [(4, 500), (3, 3, 3, 20)])
def test_float16_array_roundtrip(tmp_path: Path, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(0)
    data = (rng.standard_normal(shape) * 1e-5).astype(np.float32)  # Volts
    data[0] = 0.0  # dead channel

    loaded = _roundtrip(tmp_path, quantize.Float16StoredArray(data), 100.0)
    on_disk = sum(p.stat().st_size for p in tmp_path.rglob("*.data"))
    gains = 4 * data[..., 0].size
    assert on_disk == data.nbytes // 2 + gains, "cache is not halved"
    assert loaded.data.dtype == np.float32, "storage must be invisible downstream"

    window = loaded.overlap(0.0, 0.05).data
    assert isinstance(window, quantize.Float16StoredArray), "window is not lazy"
    ref = data[..., :5]
    assert not np.any(np.asarray(window)[ref != 0] == 0), "signal flushed to zero"
    with pytest.raises(AttributeError, match="np.asarray"):
        window.mean(0)  # a gain-less view would read scaled values

    whole = np.asarray(loaded.data)
    assert np.all(whole[0] == 0), "dead channel is not exactly zero"
    peaks = np.abs(data).max(-1, keepdims=True)
    err = np.abs(whole - data).max(-1, keepdims=True) / np.where(peaks, peaks, 1.0)
    assert err.max() < 5e-4, f"peak-relative error {err.max():.2e}"

    again = _roundtrip(tmp_path / "again", loaded.data, 100.0)
    np.testing.assert_array_equal(np.asarray(again.data), whole, "re-dump is lossy")


def test_float16_array_keeps_counts_exact(tmp_path: Path) -> None:
    counts = np.tile(np.arange(2049, dtype=np.float32), (3, 1))  # 2**11 significand
    loaded = _roundtrip(tmp_path, quantize.Float16StoredArray(counts), 50.0)
    np.testing.assert_array_equal(np.asarray(loaded.data), counts)


def test_float16_array_iadd(tmp_path: Path) -> None:
    data = 1e-5 * (1.0 + np.arange(20, dtype=np.float32).reshape(2, 10))
    data[1] *= 1e-3  # channel with a gain of its own
    loaded = _roundtrip(tmp_path, quantize.Float16StoredArray(data), 1.0)
    target = base.TimedArray(frequency=1.0, start=2.0, duration=3.0)
    target += loaded
    assert target.data.dtype == np.float32, "float16 storage cannot be added into"
    np.testing.assert_allclose(target.data, data[:, 2:5], rtol=1e-3)
