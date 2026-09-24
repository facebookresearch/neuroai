# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Pydantic configurations for models."""

import dataclasses
import functools
import importlib
import inspect
import typing as tp
import warnings

import pydantic
from exca import helpers
from torch import nn

from .utils import inject_from_context


@dataclasses.dataclass(frozen=True)
class BrainModelBuildContext:
    """Data-derived inputs a model config needs to build a model.

    A single frozen context is passed to every config's
    :meth:`BaseBrainModelConfig.build_from_context`, so each model reads only
    the fields it needs without the factory special-casing model families.  It
    is immutable: to vary one field, copy it with
    ``dataclasses.replace(ctx, n_outputs=2)``.  This powers the factory-facing
    build path only; see :class:`BaseBrainModelConfig` for the two ways to
    build a model.  Field names are modality-neutral so one context fits every
    device.

    Attributes
    ----------
    n_spatial_locations : int
        Size of the spatial axis (EEG/MEG/iEEG/fNIRS channels, fMRI
        voxels/vertices/parcels).  The size the model actually receives: with a
        downstream channel adapter the factory sets it to the adapter's output
        width (and clears ``ch_names``).
    n_temporal_samples : int
        Size of the temporal axis (M/EEG samples, fMRI TRs).
    n_outputs : int or None
        Output head width.  ``None`` when a downstream wrapper handles the
        output projection (encoder-only build).
    frequency : float or None
        Sampling rate in Hz (``1 / TR`` for fMRI).  ``None`` when unresolved;
        models that need it fall back to their config value.
    ch_names : list of str or None
        Ordered dataset channel names (sensor modalities only; ``None`` for
        fMRI).
    mesh : str or None
        Surface-mesh id for surface-sampled data (e.g. ``"FS6"``), consumed by
        surface models such as :class:`SiTConfig`.  ``None`` otherwise.

    Notes
    -----
    Model-family-specific build inputs (e.g. the fit-once baselines' lazy
    fit-data materializers and CTC blank index) live on subclasses of this
    context, not here.
    """

    n_spatial_locations: int
    n_temporal_samples: int
    n_outputs: int | None
    frequency: float | None = None
    ch_names: list[str] | None = None
    mesh: str | None = None

    def __post_init__(self) -> None:
        # Channel names, when present, must line up with the spatial axis -- a
        # mismatch silently corrupts name-based channel embeddings (LaBraM,
        # REVE, ...) and ``chs_info``-consuming models.
        if self.ch_names is not None and len(self.ch_names) != self.n_spatial_locations:
            raise ValueError(
                f"ch_names has {len(self.ch_names)} entries but "
                f"n_spatial_locations={self.n_spatial_locations}; they must match."
            )

    @property
    def chs_info(self) -> list[dict[str, tp.Any]] | None:
        """Channel metadata in braindecode/MNE ``chs_info`` form, or ``None``.

        Returns a list of per-channel dicts (``{"ch_name": ...}``) mirroring
        MNE's ``mne.Info["chs"]`` structure -- the format braindecode models
        accept via their ``chs_info`` constructor argument (see
        ``braindecode.models.base.EEGModuleMixin``).
        """
        if self.ch_names is None:
            return None
        return [{"ch_name": name} for name in self.ch_names]

    @property
    def is_downstream(self) -> bool:
        """Whether a downstream wrapper handles the output head."""
        return self.n_outputs is None


class BaseModelConfig(helpers.DiscriminatedModel, discriminator_key="name"):
    """Base class for model configurations.

    Two families share this base:

    * **Brain-model configs** -- the top-level models built from a build
      context.  They subclass :class:`BaseBrainModelConfig` and implement
      ``build(...)`` with context-named parameters; the factory-facing
      ``build_from_context(ctx)`` is provided for free.
    * **Sub-component configs** (e.g. transformers, mergers, MLP heads) which
      are composed manually inside larger models and whose ``build`` takes
      bespoke arguments.

    Both families therefore implement ``build``; it is kept permissive on this
    base so subclasses can override it with their own signatures.
    """

    def build(self, *args: tp.Any, **kwargs: tp.Any) -> nn.Module:
        raise NotImplementedError


#: Deprecated ``build`` keyword aliases -> their current names.  Lets pre-rename
#: scripts keep calling ``config.build(n_in_channels=..., n_times=..., sfreq=...)``
#: (with a :class:`DeprecationWarning`).  Remove once downstream callers migrate.
_DEPRECATED_BUILD_KWARGS: dict[str, str] = {
    "n_in_channels": "n_spatial_locations",
    "n_times": "n_temporal_samples",
    "sfreq": "frequency",
}

_BuildFn = tp.Callable[..., nn.Module]


def _support_deprecated_build_kwargs(build: _BuildFn) -> _BuildFn:
    """Wrap a ``build`` method so it also accepts the deprecated kwarg aliases.

    An alias is only remapped when the current name is an actual parameter of
    ``build`` (so e.g. passing a temporal size to a model that has none still
    raises, exactly as before the rename).  Signature/annotations are preserved
    via :func:`functools.wraps`, so the ``build_from_context`` injector and the
    ``test_base`` merge gate keep seeing the real parameters (use
    :func:`inspect.unwrap` to reach the undecorated function).
    """
    params = inspect.signature(build).parameters

    @functools.wraps(build)
    def wrapper(self: tp.Any, *args: tp.Any, **kwargs: tp.Any) -> nn.Module:
        for old, new in _DEPRECATED_BUILD_KWARGS.items():
            if old not in kwargs or new not in params:
                continue
            if new in kwargs:
                raise TypeError(
                    f"build() got both '{new}' and its deprecated alias '{old}'"
                )
            warnings.warn(
                f"build(): '{old}' is deprecated, use '{new}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs[new] = kwargs.pop(old)
        return build(self, *args, **kwargs)

    return wrapper


class BaseBrainModelConfig(BaseModelConfig):
    """Base class for top-level brain models, built in one of two ways.

    * :meth:`build` -- the only method a subclass implements, and both the
      user-facing entry point (prototyping, tests, manual construction) and the
      innermost constructor.  Its parameters are named/typed like
      :class:`BrainModelBuildContext` fields so the factory can inject them;
      any renaming to the module's own argument names happens inside ``build``.
      Called directly it has no coupling to the context, so a model under
      development may declare any argument.
    * :meth:`build_from_context` -- the factory-facing contract, provided for
      free: it injects ``build``'s parameters from the context by name.
      Subclasses override it only to narrow the context type (e.g. a fit-once
      baseline reading a context subclass with extra fields).

    The ``build`` name/type contract is enforced softly (the merge-gate test in
    ``test_base.py`` plus code review, not at import time), so prototyping with
    a novel argument never raises while merging an unbacked one fails.
    """

    def __init_subclass__(cls, **kwargs: tp.Any) -> None:
        # Transparently let every brain-model ``build`` accept the deprecated
        # kwarg aliases (see :data:`_DEPRECATED_BUILD_KWARGS`).  Applied once,
        # centrally, so no per-model ``**kwargs`` shims are needed.
        super().__init_subclass__(**kwargs)
        build = cls.__dict__.get("build")
        if build is not None and getattr(build, "__wrapped__", None) is None:
            cls.build = _support_deprecated_build_kwargs(build)  # type: ignore[method-assign]

    def build_from_context(self, ctx: "BrainModelBuildContext") -> nn.Module:
        """Build a model from a fully-populated context (factory contract).

        Default: inject :meth:`build`'s declared parameters from the context by
        name.  Subclasses override this only to narrow the context type
        (e.g. a fit-once baseline reading from a context subclass).
        """
        return self.build(**inject_from_context(self.build, ctx))


#: Data-derived build inputs a braindecode model can require forwarded from the
#: context (see :attr:`BaseBrainDecodeModel.required_fields`).
RequiredBuildField = tp.Literal["ch_names", "n_times", "sfreq"]


# Base class for braindecode model configs (using kwargs pattern)
class BaseBrainDecodeModel(BaseBrainModelConfig):
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
    #: Which data-derived build inputs this braindecode model requires forwarded
    #: from the context.  Members:
    #:
    #: * ``"ch_names"`` -- forward ``chs_info`` from the dataset's channel names
    #:   (e.g. LaBraM, REVE);
    #: * ``"n_times"`` -- forward ``n_times`` even on the pretrained path
    #:   (non-pretrained builds always receive it);
    #: * ``"sfreq"`` -- forward the sampling rate (``frequency`` -> braindecode
    #:   ``sfreq``, e.g. models with a fixed spectral front end).
    #:
    #: Empty by default: most models take ``sfreq`` via config ``kwargs`` and
    #: need neither channel names nor a pretrained-path ``n_times``.
    required_fields: tp.ClassVar[list[RequiredBuildField]] = []
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
        module_name, attr = cls._MODEL_CLASS_PATH.rsplit(".", 1)
        cls._MODEL_CLASS = getattr(importlib.import_module(module_name), attr)

    def model_post_init(self, __context__: tp.Any) -> None:
        type(self)._ensure_model_class()
        super().model_post_init(__context__)
        helpers.validate_kwargs(self._MODEL_CLASS, self.kwargs)

    def _construct(self, **kwargs: tp.Any) -> nn.Module:
        """Instantiate the underlying braindecode model from raw kwargs.

        Merges config ``kwargs`` with the build-time kwargs and dispatches to
        either ``from_pretrained`` or the plain constructor.  Subclasses that
        override :meth:`build` call this for the low-level instantiation.
        """
        type(self)._ensure_model_class()
        if overlap := set(self.kwargs) & set(kwargs):
            raise ValueError(
                f"Build kwargs overlap with config kwargs for keys: {overlap}."
            )
        kwargs = self.kwargs | kwargs
        if self.from_pretrained_name is not None:
            return self._MODEL_CLASS.from_pretrained(self.from_pretrained_name, **kwargs)
        return self._MODEL_CLASS(**kwargs)  # type: ignore

    def _bd_shape_kwargs(
        self,
        *,
        n_spatial_locations: int,
        n_temporal_samples: int,
        n_outputs: int | None,
        chs_info: list[dict[str, tp.Any]] | None,
        frequency: float | None,
    ) -> dict[str, tp.Any]:
        """Map context values onto braindecode constructor kwarg names.

        Single home for the neuraltrain -> braindecode renaming
        (``n_spatial_locations`` -> ``n_chans``, ``n_temporal_samples`` ->
        ``n_times``) plus the pretrained / downstream / ``chs_info`` / ``sfreq``
        forwarding rules (see body).  Custom subclasses call this from their
        ``build`` and adjust individual keys (e.g. a fixed or channel-reduced
        ``n_chans``).
        """
        bd_kwargs: dict[str, tp.Any] = {"n_chans": n_spatial_locations}
        if self.from_pretrained_name is not None:
            if "n_times" in self.required_fields:
                bd_kwargs["n_times"] = n_temporal_samples
        else:
            bd_kwargs["n_times"] = n_temporal_samples
        if n_outputs is not None:
            bd_kwargs["n_outputs"] = n_outputs
        if "ch_names" in self.required_fields:
            # Length vs n_spatial_locations is guaranteed by __post_init__.
            assert chs_info is not None, (
                f"{type(self).__name__} requires channel names (ctx.ch_names), "
                "but none were provided."
            )
            bd_kwargs["chs_info"] = chs_info
        if "sfreq" in self.required_fields:
            assert frequency is not None, (
                f"{type(self).__name__} requires the sampling rate "
                "(ctx.frequency), but none was provided."
            )
            bd_kwargs["sfreq"] = frequency
        return bd_kwargs

    def build(
        self,
        n_spatial_locations: int,
        n_temporal_samples: int,
        n_outputs: int | None = None,
        chs_info: list[dict[str, tp.Any]] | None = None,
        frequency: float | None = None,
    ) -> nn.Module:
        """Build the braindecode model from context-named shape parameters.

        Parameters are named/typed like
        :class:`BrainModelBuildContext` fields/properties, so the base
        ``build_from_context`` injects them (``chs_info`` / ``frequency`` from
        the matching context properties/fields).  Covers every auto-registered
        braindecode model (EEGNet, Deep4Net, ShallowFBCSPNet, BIOT, ...);
        custom configs (LaBraM, REVE, LUNA, BENDR) override this and reuse
        :meth:`_bd_shape_kwargs` for the name mapping.
        """
        return self._construct(
            **self._bd_shape_kwargs(
                n_spatial_locations=n_spatial_locations,
                n_temporal_samples=n_temporal_samples,
                n_outputs=n_outputs,
                chs_info=chs_info,
                frequency=frequency,
            )
        )


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
