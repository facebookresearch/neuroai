# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import typing as tp

import pytest
import torch

from .diffusion_prior import DiffusionPrior, DiffusionPriorModel


@pytest.mark.parametrize("depth", [6, 10])
@pytest.mark.parametrize("dim_head", [64])
@pytest.mark.parametrize("prior_learned_query_mode", ["pos_emb", "all_pos_emb"])
@pytest.mark.parametrize("timesteps", [10, 20])
@pytest.mark.parametrize("cond_drop_prob", [0.2])
def test_diffusion_prior(
    depth: int,
    dim_head: int,
    prior_learned_query_mode: str,
    timesteps: int,
    cond_drop_prob: float,
) -> None:
    batch_size, num_in_tokens, num_out_tokens, dim = 1, 257, 128, 768
    out_emb = torch.randn(batch_size, num_out_tokens, dim)
    in_emb = torch.randn(batch_size, num_in_tokens, dim)

    config_kwargs: tp.Any = {
        "depth": depth,
        "dim_head": dim_head,
        "prior_learned_query_mode": prior_learned_query_mode,
        "timesteps": timesteps,
        "cond_drop_prob": cond_drop_prob,
    }
    model = DiffusionPrior(**config_kwargs).build(
        dim=dim, num_in_tokens=num_in_tokens, num_out_tokens=num_out_tokens
    )
    assert isinstance(model, DiffusionPriorModel)

    # training mode
    model.train()
    out = model(text_embed=in_emb, image_embed=out_emb)

    assert out["pred"].shape == (batch_size, num_out_tokens, dim)
    assert out["target"].shape == (batch_size, num_out_tokens, dim)

    # testing mode
    model.eval()
    out = model(text_embed=in_emb)
    assert out.shape == (batch_size, num_out_tokens, dim)
