# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import inspect
import typing as tp

import pydantic
import pytest
import torch

from . import base
from .base import BaseBrainModelConfig, BrainModelBuildContext


@pytest.fixture
def fake_eeg():
    batch_size = 2
    n_channels = 4
    n_times = 1200
    meg = torch.randn(batch_size, n_channels, n_times)
    return meg


@pytest.mark.parametrize("use_default_config", [False, True])
def test_biot(fake_eeg, use_default_config):
    batch_size, n_channels, n_times = fake_eeg.shape
    n_outputs = 3
    sfreq = 200.0

    if use_default_config:
        config_kwargs = {
            "sfreq": sfreq,
        }
    else:
        config_kwargs = {
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 4,
            "hop_length": 100,
            "return_feature": False,
            "chs_info": None,
            "input_window_seconds": int(n_times / sfreq),
        }

    BIOT = getattr(base, "BIOT")
    model = BIOT(kwargs=config_kwargs).build(
        n_spatial_locations=n_channels, n_temporal_samples=n_times, n_outputs=n_outputs
    )
    from braindecode.models import BIOT

    assert isinstance(model, BIOT)

    out = model(fake_eeg)
    assert out.shape == (batch_size, n_outputs)


@pytest.mark.parametrize("use_default_config", [True, False])
def test_shallowconvnet(fake_eeg, use_default_config):
    batch_size, n_channels, n_times = fake_eeg.shape
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
            n_spatial_locations=n_channels,
            n_temporal_samples=n_times,
            n_outputs=n_outputs,
        )

    ShallowFBCSPNet = getattr(base, "ShallowFBCSPNet")
    model = ShallowFBCSPNet(kwargs=config_kwargs).build(
        n_spatial_locations=n_channels, n_temporal_samples=n_times, n_outputs=n_outputs
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
    result = cfg.build(n_spatial_locations=4, n_temporal_samples=1200, n_outputs=2)

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
    cfg.build(n_spatial_locations=4, n_temporal_samples=1200, n_outputs=2)

    call_kwargs = mock_fp.call_args[1]
    assert call_kwargs["drop_prob"] == 0.5
    assert call_kwargs["n_outputs"] == 2


def test_deprecated_build_kwargs_aliases(fake_eeg):
    """Old shape kwargs still build (with a warning); mixing old+new errors."""
    _, n_channels, n_times = fake_eeg.shape
    cfg = getattr(base, "EEGNet")()

    with pytest.warns(DeprecationWarning, match="n_in_channels"):
        model = cfg.build(n_in_channels=n_channels, n_times=n_times, n_outputs=2)
    assert isinstance(model, torch.nn.Module)

    with pytest.raises(TypeError, match="deprecated alias"):
        cfg.build(
            n_spatial_locations=n_channels,
            n_in_channels=n_channels,
            n_temporal_samples=n_times,
            n_outputs=2,
        )


def _iter_all_subclasses(
    cls: type[BaseBrainModelConfig],
) -> tp.Iterator[type[BaseBrainModelConfig]]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _iter_all_subclasses(sub)


def _context_type_map() -> dict[str, tp.Any]:
    """Name -> declared type for every BrainModelBuildContext field and
    read-only property (the legal ``build`` parameters for models that use the
    base ``build_from_context`` injector)."""
    hints = dict(tp.get_type_hints(BrainModelBuildContext))
    for name, attr in vars(BrainModelBuildContext).items():
        if isinstance(attr, property) and attr.fget is not None:
            ret = tp.get_type_hints(attr.fget).get("return")
            if ret is not None:
                hints[name] = ret
    return hints


def test_build_params_are_context_backed() -> None:
    """Merge gate for the ``build`` contract.

    Every model that relies on the base ``build_from_context`` injector (i.e.
    does *not* override it) must declare ``build`` parameters whose names *and*
    types match a ``BrainModelBuildContext`` field/property.  This catches the
    reflective ctx->build seam that mypy cannot see, at merge time, without
    disturbing prototyping via ``build(...)``.  Models that override
    ``build_from_context`` (e.g. to narrow the context type) are exempt -- they
    own their own mapping.
    """
    # Import for side effects: registers every model subclass so the walk below
    # sees them all.
    import neuraltrain.models  # noqa: F401  # pylint: disable=unused-import

    allowed = _context_type_map()
    checked: list[str] = []
    for cls in set(_iter_all_subclasses(BaseBrainModelConfig)):
        # Overriding build_from_context is the explicit opt-out from the gate.
        if cls.build_from_context is not BaseBrainModelConfig.build_from_context:
            continue
        build = cls.__dict__.get("build")
        if build is None:
            continue
        # ``build`` may be wrapped to accept deprecated kwarg aliases; inspect
        # the undecorated function so annotations resolve in the model's module.
        build = inspect.unwrap(build)
        checked.append(cls.__name__)
        hints = tp.get_type_hints(build)
        for name, param in inspect.signature(build).parameters.items():
            if name == "self" or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            assert name in allowed, (
                f"{cls.__name__}.build declares '{name}', which is not a "
                f"BrainModelBuildContext field/property. Promote it to a typed "
                f"context field (and populate it in the factory) before merging, "
                f"or override build_from_context to map it yourself."
            )
            # The param must match the context type exactly, or widen it with
            # ``| None`` -- a param that also accepts None (for an ergonomic
            # ``build()`` default, e.g. ``frequency``) is compatible with
            # the injected value.
            param_type = hints.get(name)
            allowed_type = allowed[name]
            assert param_type in (allowed_type, tp.Optional[allowed_type]), (
                f"{cls.__name__}.build param '{name}' is typed "
                f"{param_type!r}, but BrainModelBuildContext.{name} is "
                f"{allowed_type!r}. Match it exactly (optionally widened with "
                f"| None), or override build_from_context to narrow the type."
            )
    # Guard against the check silently covering nothing (e.g. if Linear
    # regresses to a build_from_context override).
    assert "Linear" in checked, (
        f"expected Linear to be exercised by this gate; only checked {checked}"
    )
