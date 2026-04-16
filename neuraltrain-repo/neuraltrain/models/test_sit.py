# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import pytest
import torch

from .sit import DENSITY_TO_PATCH_VERTEX_SIZE, Density, SiT, SiTConfig, TokenizerConfig
from .transformer import TransformerEncoder

_MESH = Density.FS5
_GRID = Density.FS2
_N_PATCHES_PER_HEMI, _N_VERTICES = DENSITY_TO_PATCH_VERTEX_SIZE[_MESH][_GRID]
_N_PATCHES = _N_PATCHES_PER_HEMI * 2
_N_MESH_VERTICES = 10242 * 2
_DIM = 96


@pytest.fixture()
def fake_triangle_csv(tmp_path):
    """Create a fake cortical triangular patches indices CSV for testing."""
    rng = np.random.default_rng(42)
    data = rng.integers(0, _N_MESH_VERTICES // 2, size=(_N_VERTICES, _N_PATCHES))
    csv_dir = tmp_path / "fsaverage"
    csv_dir.mkdir()
    path = csv_dir / f"fs{_MESH.value}_vertices_from_fs{_GRID.value}_triangles.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return csv_dir


def _base_config(csv_dir) -> SiTConfig:
    """Base SiT configuration for testing."""
    return SiTConfig(
        dim=_DIM,
        transformer_config=TransformerEncoder(depth=2, heads=3),
        tokenizer_config=TokenizerConfig(
            mesh_resolution=_MESH,
            grid_resolution=_GRID,
            triangle_indices_csv_path=str(csv_dir),
        ),
    )


@pytest.mark.parametrize(
    "in_time_agg,n_frames,expected_seq",
    [
        ("in_channels", 1, _N_PATCHES),
        ("in_channels", 3, _N_PATCHES),
        ("in_mean", 3, _N_PATCHES),
        ("in_concat", 3, _N_PATCHES * 3),
        ("in_patch_mean", 3, 3),
    ],
)
def test_sit_agg(
    fake_triangle_csv, in_time_agg: str, n_frames: int, expected_seq: int
) -> None:
    """Each temporal aggregation strategy produces the correct sequence length."""
    cfg = _base_config(fake_triangle_csv).model_copy(
        update={"in_time_agg": in_time_agg},
    )
    model = cfg.build(n_frames=n_frames)
    assert isinstance(model, SiT)
    assert model.pos_embedding.pe_len == expected_seq
    out = model(torch.randn(2, _N_MESH_VERTICES, n_frames))
    assert out.shape == (2, expected_seq, _DIM)


@pytest.mark.parametrize("pool", ["mean", "cls"])
def test_sit_pool(fake_triangle_csv, pool: str) -> None:
    """Pooling collapses the sequence to a single vector."""
    cfg = _base_config(fake_triangle_csv).model_copy(
        update={"pool": pool, "use_cls_token": pool == "cls"},
    )
    model = cfg.build(n_frames=3)
    out = model(torch.randn(2, _N_MESH_VERTICES, 3))
    assert out.shape == (2, _DIM)
