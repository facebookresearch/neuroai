# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from .mae import MaeEncoder
from .transformer import TransformerEncoder


@pytest.fixture
def config() -> MaeEncoder:
    return MaeEncoder(
        dim=64,
        patch_size=20,
        transformer_config=TransformerEncoder(heads=2, depth=1, rotary_pos_emb=False),
    )


# One model instance must handle any window length, so that a checkpoint stays
# usable downstream; 205 also exercises the incomplete trailing patch.
@pytest.mark.parametrize("n_times", [200, 205, 2000])
@pytest.mark.parametrize("n_outputs", [None, 3])
def test_build_and_forward(config, n_times, n_outputs) -> None:
    model = config.build(n_spatial_locations=8, n_outputs=n_outputs)
    out = model(torch.randn(2, 8, n_times))

    expected = (2, 3) if n_outputs is not None else (2, n_times // 20, 64)
    assert out.shape == expected, f"unexpected output shape {tuple(out.shape)}"
    assert (n_outputs is not None) == any(
        name.startswith("head.") for name in model.state_dict()
    ), "output head must exist if and only if n_outputs is set"


def test_patchify_rejects_too_short_input(config) -> None:
    model = config.build(n_spatial_locations=8, n_outputs=None)
    with pytest.raises(ValueError, match="less than patch_size"):
        model.patchify(torch.randn(2, 8, 19))
