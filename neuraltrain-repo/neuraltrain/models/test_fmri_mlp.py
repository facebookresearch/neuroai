# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from .fmri_mlp import FmriLinear, FmriLinearModel, FmriMlp, FmriMlpModel


@pytest.mark.parametrize("n_blocks", [4, 0])
@pytest.mark.parametrize("subject_layers_config", [{}, None])
def test_fmri_mlp(n_blocks: int, subject_layers_config: dict | None) -> None:
    batch_size, n_voxels, n_times = 10, 100, 1
    fmri = torch.randn(batch_size, n_voxels, n_times)
    subject_ids = torch.randint(0, 2, (batch_size, 1))
    out_dim = 6

    config_kwargs = {
        "hidden": 64,
        "n_blocks": n_blocks,
        "norm_type": "ln",
        "act_first": False,
        "subject_layers_config": subject_layers_config,
        "n_repetition_times": n_times,
    }
    model = FmriMlp(**config_kwargs).build(n_voxels, out_dim)  # type: ignore
    assert isinstance(model, FmriMlpModel)

    out = model(fmri, subject_ids=subject_ids)
    assert out.shape == (batch_size, out_dim)


@pytest.mark.parametrize("n_repetition_times", [5, 1])
@pytest.mark.parametrize("use_tr_layer", [True, False])
@pytest.mark.parametrize("use_tr_embeds", [True, False])
@pytest.mark.parametrize("time_agg", ["out_linear", "in_linear", "out_mean", "in_mean"])
def test_fmri_mlp_time_agg(
    n_repetition_times: int, use_tr_layer: bool, use_tr_embeds: bool, time_agg: str
) -> None:
    batch_size, n_voxels = 10, 100
    fmri = torch.randn(batch_size, n_voxels, n_repetition_times)
    subject_ids = torch.randint(0, 2, (batch_size, 1))
    out_dim = 8

    config_kwargs = {
        "hidden": 4,
        "n_blocks": 1,
        "norm_type": "ln",
        "act_first": False,
        "subject_layers_config": {},
        "n_repetition_times": n_repetition_times,
        "time_agg": time_agg,
        "use_tr_layer": use_tr_layer,
        "use_tr_embeds": use_tr_embeds,
        "output_head_config": {
            "clip": {
                "hidden_sizes": [out_dim * 2],
                "norm_layer": "layer",
                "activation_layer": "gelu",
            },
            "mse": {"hidden_sizes": []},
        },
    }
    model = FmriMlp(**config_kwargs).build(n_voxels, out_dim)  # type: ignore
    assert isinstance(model, FmriMlpModel)
    out = model(fmri, subject_ids=subject_ids)
    assert out["clip"].shape == (batch_size, out_dim * 2)
    assert out["mse"].shape == (batch_size, out_dim)


@pytest.mark.parametrize("time_agg", ["in_linear", "out_mean", "out_linear", "in_mean"])
@pytest.mark.parametrize("use_output_head_config", [False, True])
def test_fmri_linear(time_agg, use_output_head_config) -> None:
    batch_size, n_voxels, n_times = 10, 100, 5
    fake_fmri = torch.randn(batch_size, n_voxels, n_times)
    out_dim = 6

    config_kwargs = {
        "time_agg": time_agg,
        "output_head_config": (
            {
                "clip": {
                    "hidden_sizes": [out_dim],
                    "norm_layer": "layer",
                    "activation_layer": "gelu",
                },
                "mse": {"hidden_sizes": []},
            }
            if use_output_head_config
            else None
        ),
    }
    model = FmriLinear(**config_kwargs).build(n_voxels, out_dim)
    assert isinstance(model, FmriLinearModel)
    out = model(fake_fmri)
    if use_output_head_config:
        out = out["clip"]
    assert out.shape == (batch_size, out_dim)
