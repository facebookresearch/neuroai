# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Reflection helpers for the model-config build interface.

These support the ``build`` / ``build_from_context`` contract on
:class:`neuraltrain.models.base.BaseBrainModelConfig` (see its docstring).  They
live here rather than in ``base`` to keep that module focused on the config
classes themselves.
"""

from __future__ import annotations

import inspect
import typing as tp

if tp.TYPE_CHECKING:
    from .base import BrainModelBuildContext


def iter_build_params(
    fn: tp.Callable[..., tp.Any],
) -> tp.Iterator[tuple[str, inspect.Parameter]]:
    """Yield ``(name, Parameter)`` for a build method, skipping ``self`` and
    ``*args`` / ``**kwargs``."""
    for name, param in inspect.signature(fn).parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        yield name, param


def inject_from_context(
    fn: tp.Callable[..., tp.Any], ctx: "BrainModelBuildContext"
) -> dict[str, tp.Any]:
    """Pull ``fn``'s declared parameters off ``ctx`` by name.

    Used by the default :meth:`BaseBrainModelConfig.build_from_context` to feed
    a model's ``build`` from the context.  Legal names are the context's
    fields *and* properties (``ctx.chs_info``, ``ctx.is_downstream``, ...).  A
    required parameter with no matching context attribute is an error; the
    ``test_base`` merge-gate test catches this statically before it can reach
    the factory.
    """
    out: dict[str, tp.Any] = {}
    for name, param in iter_build_params(fn):
        if hasattr(ctx, name):
            out[name] = getattr(ctx, name)
        elif param.default is inspect.Parameter.empty:
            raise AttributeError(
                f"{fn.__qualname__} requires '{name}', which is not a "
                f"BrainModelBuildContext field/property."
            )
    return out
