# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import pydantic
import torch

from neuralset import base

T = tp.TypeVar("T", bound=torch.Tensor | np.ndarray)


class HuggingFaceConfig(base.BaseModel):
    """Common HuggingFace model construction options.

    This config is shared by text, audio, image, and video extractors so that
    HuggingFace models are instantiated consistently across modalities.

    Parameters
    ----------
    model_cls_name : str, default="AutoModel"
        Name of the model class to load from ``transformers``. Use this for a
        specific extractor instance when ``AutoModel`` is not the right class.
    processor_cls_name : str, default="AutoProcessor"
        Name of the processor class to load from ``transformers``. Use this for
        a specific extractor instance when ``AutoProcessor`` is not the right class.
    model_kwargs : dict | None, default=None
        Extra keyword arguments forwarded to model construction for
        non-standard HuggingFace models.
    processor_kwargs : dict | None, default=None
        Extra keyword arguments forwarded to processor construction. These are
        forwarded to processor construction for non-standard HuggingFace
        processors.

    Notes
    -----
    Non-standard HuggingFace model or processor classes can be specified with
    ``hf_config=HuggingFaceConfig(model_cls_name="...",
    processor_cls_name="...")``. Modality-specific configs may define
    ``HF_CLASS_DEFAULTS`` as a mapping from model-name patterns to class defaults.
    """

    model_cls_name: str = "AutoModel"
    processor_cls_name: str = "AutoProcessor"
    model_kwargs: dict[str, tp.Any] | None = None
    processor_kwargs: dict[str, tp.Any] | None = None
    HF_CLASS_DEFAULTS: tp.ClassVar[dict[str, dict[str, str]]] = {}

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return []

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        forbidden = {"device_map", "torch_dtype"} & set(self.model_kwargs or {})
        if forbidden:
            bad_keys = ", ".join(sorted(forbidden))
            msg = (
                f"Do not define {bad_keys} in hf_config.model_kwargs. "
                "Use HuggingFaceMixin.device and HuggingFaceMixin.dtype instead."
            )
            raise ValueError(msg)

    def _transformers_cls(self, cls_name: str, kind: str) -> tp.Any:
        import transformers

        try:
            return getattr(transformers, cls_name)
        except AttributeError as e:
            msg = f"transformers has no {kind} class {cls_name!r}"
            raise ValueError(msg) from e

    def _resolved_cls_name(self, field: str, model_name: str) -> str:
        configured_cls_name: str = getattr(self, field)
        if field in self.model_fields_set:
            return configured_cls_name
        lower_model_name = model_name.lower()
        for pattern, defaults_for_pattern in self.HF_CLASS_DEFAULTS.items():
            if pattern not in lower_model_name:
                continue
            default_cls_name = defaults_for_pattern.get(field)
            if default_cls_name is None:
                continue
            return default_cls_name
        return configured_cls_name

    def model_cls(self, model_name: str) -> tp.Any:
        cls_name = self._resolved_cls_name("model_cls_name", model_name)
        return self._transformers_cls(cls_name, "model")

    def processor_cls(self, model_name: str) -> tp.Any:
        cls_name = self._resolved_cls_name("processor_cls_name", model_name)
        return self._transformers_cls(
            cls_name,
            "processor",
        )


class HuggingFaceMixin(base.BaseModel):
    """Mixin for extractors that use a HuggingFace model.
    These extractors all return a tensor of shape (n_layers, n_tokens, *embedding_shape).
    This mixin determines how to aggregate the layers and tokens.

    Parameters
    ----------
    model_name: str
        Name of the model to use.
    hf_config: HuggingFaceConfig
        Shared HuggingFace loading options such as model and processor classes,
        plus extra model and processor kwargs.
    device: {"auto", "cpu", "cuda", "accelerate"}, default="auto"
        Device strategy for model placement. ``"auto"`` resolves to CUDA when
        available, otherwise CPU. ``"accelerate"`` forwards
        ``device_map="auto"`` to HuggingFace Accelerate.
    dtype: {"auto", "float16", "float32", "float64", "bfloat16"} | None, default=None
        Optional dtype forwarded to ``from_pretrained`` as ``torch_dtype``.
        If ``None``, Transformers chooses its default dtype.
    pretrained: bool
        If True, load pretrained model weights. If False, instantiate from the
        pretrained config without loading pretrained weights.
    layers: float | list[float] | "all"
        Specifies the layers to keep.
        - "all": keep all layers
        - a float between 0 and 1 (or list of): the relative depth of the layer(s) to use,
        where 0 stands for the first layer and 1 for the last layer.
    cache_n_layers: None, int
        if provided, n equidistributed layers will be cached
        If None, only cache the output of the layers specified by `layers`.
    layer_aggregation: str
        How to aggregate the layers (first dimension of the tensor of activations).
        Can be "mean", "sum", "group_mean" or None (in which case we keep the original dimension).
        "group_mean" will average the layers by groups defined by the layers parameter,
        e.g. if layers=[0, 0.5, 1] and there are 10 layers, the layers will be grouped as [0:5], [5:10].
    token_aggregation: str
        How to aggregate the tokens (second dimension of the tensor of activations).
        Can be "first", "last", "mean", "sum", "max", or None (in which case we keep the original dimension).

    Requires `transformers` and `huggingface_hub`. You can install them manually
    or via `pip install "neuralset[all]"`.
    """

    requirements: tp.ClassVar[tuple[str, ...]] = (
        "transformers>=4.29.2",
        "huggingface_hub>=0.27.0",
    )
    model_name: str
    hf_config: HuggingFaceConfig = HuggingFaceConfig()
    device: tp.Literal["auto", "cpu", "cuda", "accelerate"] = "auto"
    dtype: (
        tp.Literal[
            "auto",
            "float16",
            "float32",
            "float64",
            "bfloat16",
        ]
        | None
    ) = None
    pretrained: bool = True
    layers: float | list[float] | tp.Literal["all"] = 2 / 3
    cache_n_layers: int | None = None
    layer_aggregation: tp.Literal["mean", "sum", "group_mean"] | None = "mean"
    token_aggregation: tp.Literal["first", "last", "mean", "sum", "max"] | None = "mean"
    _model: torch.nn.Module | None = pydantic.PrivateAttr(default=None)
    _processor: tp.Any | None = pydantic.PrivateAttr(default=None)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        name = self.__class__.__name__
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.layers != "all":
            layers = self.layers if isinstance(self.layers, list) else [self.layers]
            if not all(isinstance(layer, float) and 0 <= layer <= 1 for layer in layers):
                raise ValueError(f"The layers must be floats between 0 and 1 for {name}")
        if self.cache_n_layers == 1:
            msg = f"Set {name}.cache_n_layers=None instead of 1"
            raise ValueError(msg)
        self._download_huggingface_snapshot()

    def _download_huggingface_snapshot(self) -> None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        revisions: list[tp.Any] = []
        for load_kwargs in (
            self.hf_config.model_kwargs,
            self.hf_config.processor_kwargs,
        ):
            revision = None if load_kwargs is None else load_kwargs.get("revision")
            if revision not in revisions:
                revisions.append(revision)
        for revision in revisions:
            kwargs = {} if revision is None else {"revision": revision}
            try:  # fast path once the snapshot is already cached
                snapshot_download(
                    repo_id=self.model_name,
                    local_files_only=True,
                    **kwargs,
                )
            except LocalEntryNotFoundError:  # first instantiation populates the cache
                snapshot_download(
                    repo_id=self.model_name,
                    **kwargs,
                )

    @classmethod
    def _exclude_from_cls_uid(cls) -> list[str]:
        return ["device"]

    def _exclude_from_cache_uid(self) -> list[str]:
        excluded: list[str] = ["device"]
        if self.cache_n_layers is not None:
            excluded.extend(["layers", "layer_aggregation"])
        return excluded

    @property
    def model(self) -> torch.nn.Module:
        if not hasattr(self, "_model") or self._model is None:
            self._model = self.load_model()
        return self._model

    @property
    def model_device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.model.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")

    @property
    def processor(self) -> tp.Any:
        if not hasattr(self, "_processor") or self._processor is None:
            self._processor = self.load_processor()
        return self._processor

    def load_model(self) -> torch.nn.Module:
        from transformers import AutoConfig

        hf_config = self.hf_config
        Model = hf_config.model_cls(self.model_name)
        if self.pretrained:
            kwargs = (hf_config.model_kwargs or {}).copy()
            kwargs["local_files_only"] = True
            if self.device == "accelerate":
                kwargs["device_map"] = "auto"
            if self.dtype is not None and "torch_dtype" not in kwargs:
                kwargs["torch_dtype"] = (
                    self.dtype if self.dtype == "auto" else getattr(torch, self.dtype)
                )
            try:
                model = Model.from_pretrained(
                    self.model_name,
                    **kwargs,
                )
            except Exception as e:
                msg = (
                    f"Failed to instantiate HuggingFace model {self.model_name!r} "
                    f"with {self.__class__.__name__}. Try adjusting hf_config, "
                    "for example model_cls_name or model_kwargs, or adjusting dtype. "
                    f"Original error: {type(e).__name__}: {e}"
                )
                raise RuntimeError(msg) from e
            if self.device != "accelerate":
                model.to(self.device)
        else:
            config_kwargs: dict[str, tp.Any] = {
                "output_hidden_states": True,
                "local_files_only": True,
            }
            if (
                hf_config.model_kwargs is not None
                and "revision" in hf_config.model_kwargs
            ):
                config_kwargs["revision"] = hf_config.model_kwargs["revision"]
            config = AutoConfig.from_pretrained(
                self.model_name,
                **config_kwargs,
            )
            constructor = getattr(Model, "from_config", Model._from_config)  # type: ignore[attr-defined]
            model = constructor(config, **(hf_config.model_kwargs or {}))
            if isinstance(self.dtype, str) and self.dtype != "auto":
                model.to(dtype=getattr(torch, self.dtype))
            if self.device == "accelerate":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
            else:
                model.to(self.device)
        model.eval()
        return model

    def load_processor(self) -> tp.Any:
        Processor = self.hf_config.processor_cls(self.model_name)
        kwargs = (self.hf_config.processor_kwargs or {}).copy()
        kwargs["local_files_only"] = True
        return Processor.from_pretrained(
            self.model_name,
            **kwargs,
        )

    def _aggregate_layers(self, latents: np.ndarray) -> np.ndarray:
        """
        Input:
            a tensor of activations of shape (n_model_layers, *embedding_shape)
        Output:
            a tensor of activations of following shape:
            - (len(self.layers), *embedding_shape) if layer_aggregation is None,
            - (len(self.layers)-1, *embedding_shape) if layer_aggregation is "group_mean",
            - (*embedding_shape) if layer_aggregation is "mean" or "sum".
        This function should be called before caching to reduce the size of the cached tensors,
        except if cache_n_layers > 1, in which case it should be called after caching.
        """
        n_model_layers = latents.shape[0]
        if self.layers == "all":
            layer_indices = list(range(n_model_layers))
        else:
            layers = self.layers if isinstance(self.layers, list) else [self.layers]
            layer_indices = np.unique(
                [int(i * (n_model_layers - 1)) for i in layers]
            ).tolist()  # type: ignore
        if len(layer_indices) == 1:
            if self.layer_aggregation is None:
                return latents[layer_indices[0]][None, :]
            else:
                return latents[layer_indices[0]]
        else:  # aggregate
            latents = np.asarray(latents)  # ContiguousMemmap must be loaded first
            if self.layer_aggregation == "mean":
                return latents[layer_indices].mean(0)  # type: ignore
            elif self.layer_aggregation == "sum":
                return latents[layer_indices].sum(0)  # type: ignore
            elif self.layer_aggregation == "group_mean":
                groups = []
                layer_indices[-1] += 1
                for l1, l2 in zip(layer_indices[:-1], layer_indices[1:]):
                    groups.append(latents[l1:l2].mean(0))
                return np.stack(groups)
            elif self.layer_aggregation is None:
                return latents[layer_indices]
            else:
                raise ValueError(f"Unknown layer aggregation: {self.layer_aggregation}")

    def _layer_subselection(self, latents: T) -> T:
        n_layers = latents.shape[0]
        if self.cache_n_layers is None or self.cache_n_layers >= n_layers:
            return latents
        selected = [
            int(round(x)) for x in np.linspace(0, n_layers - 1, self.cache_n_layers)
        ]
        return latents[selected, ...]  # type: ignore

    def _aggregate_tokens(self, latents: T) -> T:
        """Aggregate tokens and subselects the layers

        Input:
            a tensor of activations of shape (layer_idx, token_idx, *embedding_shape)
        Output:
            a tensor of activations with possibly downsampled number of layers, and token
            dimension removed through the aggregation method (if not None)
        This function should always be called before caching, to reduce the size
        of the cached tensors.
        """
        latents = self._layer_subselection(latents)
        match self.token_aggregation:
            case "mean":
                out = latents.mean(axis=1)  # type: ignore
            case "sum":
                out = latents.sum(axis=1)  # type: ignore
            case "max":
                out = latents.max(axis=1)  # type: ignore
                if isinstance(latents, torch.Tensor):
                    out = out.values  # type: ignore[union-attr]
            case "first":
                # np.take in numpy, torch.select in torch
                out = latents[:, 0, ...]
            case "last":
                out = latents[:, -1, ...]
            case None:
                out = latents
        if isinstance(out, torch.Tensor):
            out = out.float()  # recast bf16
        return out  # type: ignore
