# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic configurations for models."""

import typing as tp

import pydantic
from exca import helpers
from torch import nn


class BaseModelConfig(helpers.DiscriminatedModel, discriminator_key="name"):
    """Base class for model configurations."""

    def build(self, *args, **kwargs) -> nn.Module:
        raise NotImplementedError


# Base class for braindecode model configs (using kwargs pattern)
class BaseBrainDecodeModel(BaseModelConfig):
    """Base class for braindecode model configurations.

    Attributes
    ----------
    kwargs : dict
        Free-form keyword arguments forwarded to the braindecode model
        constructor. Validated against the model's ``__init__`` signature at
        config creation time.
    from_pretrained_name : str or None
        Optional HuggingFace Hub repository ID (e.g.
        ``"braindecode/labram-pretrained"``).  When set, ``build()`` calls
        ``_MODEL_CLASS.from_pretrained()`` instead of the regular constructor.
    """

    _MODEL_CLASS: tp.ClassVar[tp.Any]
    kwargs: dict[str, tp.Any] = {}
    from_pretrained_name: str | None = None

    def model_post_init(self, __context__: tp.Any) -> None:
        super().model_post_init(__context__)
        helpers.validate_kwargs(self._MODEL_CLASS, self.kwargs)

    def build(self, **kwargs: tp.Any) -> nn.Module:
        if overlap := set(self.kwargs) & set(kwargs):
            raise ValueError(
                f"Build kwargs overlap with config kwargs for keys: {overlap}."
            )
        kwargs = self.kwargs | kwargs
        if self.from_pretrained_name is not None:
            return self._MODEL_CLASS.from_pretrained(self.from_pretrained_name, **kwargs)
        return self._MODEL_CLASS(**kwargs)  # type: ignore


def _register_braindecode_models() -> None:
    """Register per-model config classes for all braindecode models.

    Called at import time only when braindecode is installed.
    """
    import braindecode.models
    from braindecode.models import __all__ as bd_models

    for name in bd_models:
        cls: type[BaseBrainDecodeModel] = pydantic.create_model(  # type: ignore[assignment]
            name,
            __base__=BaseBrainDecodeModel,
        )
        cls._MODEL_CLASS = getattr(braindecode.models, name)  # type: ignore[attr-defined]
        globals()[name] = cls


try:
    _register_braindecode_models()
except ImportError:
    pass
