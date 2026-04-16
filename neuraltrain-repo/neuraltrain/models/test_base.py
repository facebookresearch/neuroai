# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pydantic
import pytest
import torch

from . import base


@pytest.fixture
def fake_eeg():
    batch_size = 2
    n_channels = 4
    n_times = 1200
    meg = torch.randn(batch_size, n_channels, n_times)
    return meg


@pytest.mark.parametrize("use_default_config", [False, True])
def test_biot(fake_eeg, use_default_config):
    batch_size, n_in_channels, n_times = fake_eeg.shape
    n_outputs = 3
    sfreq = 200.0

    if use_default_config:
        config_kwargs = {
            "sfreq": sfreq,
            "n_times": n_times,
        }
    else:
        config_kwargs = {
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 4,
            "hop_length": 100,
            "return_feature": False,
            "chs_info": None,
            "n_times": n_times,
            "input_window_seconds": int(n_times / sfreq),
        }

    BIOT = getattr(base, "BIOT")
    model = BIOT(kwargs=config_kwargs).build(n_chans=n_in_channels, n_outputs=n_outputs)
    from braindecode.models import BIOT

    assert isinstance(model, BIOT)

    out = model(fake_eeg)
    assert out.shape == (batch_size, n_outputs)


@pytest.mark.parametrize("use_default_config", [True, False])
def test_shallowconvnet(fake_eeg, use_default_config):
    batch_size, n_in_channels, n_times = fake_eeg.shape
    n_outputs = 3

    if use_default_config:
        config_kwargs = dict()
    else:
        config_kwargs = dict(
            n_filters_time=16,
            pool_mode="max",
            drop_prob=0.25,
        )

    with pytest.raises(pydantic.ValidationError):
        base.BaseModelConfig(name="BadName", kwargs=config_kwargs)
    with pytest.raises(ValueError):
        bad_kwargs = config_kwargs.copy()
        bad_kwargs["bad_key"] = "bad_value"
        getattr(base, "BIOT")(kwargs=bad_kwargs).build(
            n_chans=n_in_channels, n_outputs=n_outputs
        )

    ShallowFBCSPNet = getattr(base, "ShallowFBCSPNet")
    model = ShallowFBCSPNet(kwargs=config_kwargs).build(
        n_chans=n_in_channels, n_outputs=n_outputs, n_times=n_times
    )
    from braindecode.models import ShallowFBCSPNet

    assert isinstance(model, ShallowFBCSPNet)

    out = model(fake_eeg)
    assert out.shape == (batch_size, n_outputs)


def test_from_pretrained_field_accepted():
    """Config should accept ``from_pretrained_name`` without raising."""
    EEGNet = getattr(base, "EEGNet")
    cfg = EEGNet(from_pretrained_name="some-org/some-repo")
    assert cfg.from_pretrained_name == "some-org/some-repo"
    assert cfg.kwargs == {}


def test_from_pretrained_build_calls_from_pretrained(mocker):
    """When ``from_pretrained_name`` is set, ``build()`` must call
    ``_MODEL_CLASS.from_pretrained`` and return its result."""
    EEGNet = getattr(base, "EEGNet")
    cfg = EEGNet(from_pretrained_name="org/repo")

    sentinel = mocker.MagicMock(name="pretrained_model")
    mock_fp = mocker.patch.object(
        cfg._MODEL_CLASS, "from_pretrained", return_value=sentinel
    )
    result = cfg.build(n_outputs=2, chs_info=None)

    mock_fp.assert_called_once()
    assert result is sentinel


def test_from_pretrained_build_forwards_config_kwargs(mocker):
    """``self.kwargs`` from the config should be forwarded to
    ``from_pretrained``."""
    EEGNet = getattr(base, "EEGNet")
    cfg = EEGNet(
        from_pretrained_name="org/repo",
        kwargs={"drop_prob": 0.5},
    )

    mock_fp = mocker.patch.object(
        cfg._MODEL_CLASS, "from_pretrained", return_value=mocker.MagicMock()
    )
    cfg.build(n_outputs=2)

    call_kwargs = mock_fp.call_args[1]
    assert call_kwargs["drop_prob"] == 0.5
    assert call_kwargs["n_outputs"] == 2
