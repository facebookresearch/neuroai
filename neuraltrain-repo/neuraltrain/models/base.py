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

    Subclasses set ``_MODEL_CLASS_PATH`` (e.g.
    ``"braindecode.models.Labram"``) to resolve the underlying class lazily,
    avoiding an unconditional braindecode import at module load time.
    Subclasses that need custom resolution (e.g. optional-dependency
    handling) can instead override ``_ensure_model_class`` directly.

    The dynamic registration in :func:`_register_braindecode_models` sets
    ``_MODEL_CLASS`` directly at import time for the common braindecode
    models, which short-circuits the lazy path.

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

    _MODEL_CLASS: tp.ClassVar[tp.Any] = None
    _MODEL_CLASS_PATH: tp.ClassVar[str | None] = None
    kwargs: dict[str, tp.Any] = {}
    from_pretrained_name: str | None = None

    @classmethod
    def _ensure_model_class(cls) -> None:
        """Resolve ``_MODEL_CLASS`` on first use.

        Called from both ``model_post_init`` and ``build`` because
        submitit deserialization on SLURM workers does not invoke
        ``model_post_init``.
        """
        if cls._MODEL_CLASS is not None:
            return
        if cls._MODEL_CLASS_PATH is None:
            raise RuntimeError(
                f"{cls.__name__} has neither `_MODEL_CLASS` nor `_MODEL_CLASS_PATH` set."
            )
        import importlib

        module_name, attr = cls._MODEL_CLASS_PATH.rsplit(".", 1)
        cls._MODEL_CLASS = getattr(importlib.import_module(module_name), attr)

    def model_post_init(self, __context__: tp.Any) -> None:
        type(self)._ensure_model_class()
        super().model_post_init(__context__)
        helpers.validate_kwargs(self._MODEL_CLASS, self.kwargs)

    def build(self, **kwargs: tp.Any) -> nn.Module:
        type(self)._ensure_model_class()
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
