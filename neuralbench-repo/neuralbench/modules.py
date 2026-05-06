# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""``nn.Module`` classes and pydantic config wrappers used by neuralbench.

This module groups the small custom ``nn.Module`` building blocks
(``IndexSelect``, ``Mean``, ``ConcatGroupedMean``) together with the
pydantic configs that build trainable adapters / probes on top of a
pretrained model (``ChannelProjection``, ``DownstreamWrapper``) and the
runtime wrapper they produce (``DownstreamWrapperModel``).
"""

import inspect
import logging
import typing as tp

import pydantic
import torch
from torch import nn

from neuraltrain.models.common import ChannelMerger, Mlp
from neuraltrain.models.preprocessor import OnTheFlyPreprocessor

LOGGER = logging.getLogger(__name__)


class IndexSelect(nn.Module):
    """Select specific indices along a dimension, squeezing if only one index is selected."""

    def __init__(self, dim: int, index: torch.Tensor) -> None:
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.index_select(x, dim=self.dim, index=self.index.to(x.device))
        if len(self.index) == 1:
            out = out.squeeze(self.dim)
        return out


class Mean(nn.Module):
    """Reduce a tensor by averaging over one or more dimensions."""

    def __init__(self, dim: int | tuple[int, ...]) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class ConcatGroupedMean(nn.Module):
    """Split a tensor into ``n_splits`` groups along ``dim``, average each group, then concatenate."""

    def __init__(self, dim: int, n_splits: int) -> None:
        super().__init__()
        self.dim = dim
        self.n_splits = n_splits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concat(
            [
                xi.mean(dim=self.dim)
                for xi in torch.tensor_split(x, self.n_splits, dim=self.dim)
            ],
            dim=self.dim,
        )


class ChannelProjection(pydantic.BaseModel):
    """Configuration for a Conv1d(kernel_size=1) channel projection adapter.

    Projects from an arbitrary number of input channels to a fixed target count
    via a learned pointwise (1x1) convolution.  Unlike ``ChannelMerger``, this
    does not use channel positions -- it is a simple linear mixing matrix applied
    identically at every time step.

    Parameters
    ----------
    n_target_channels : int
        Number of output channels (must match what the pretrained model expects).
    max_norm : float or None
        If set, applies a max-norm weight constraint on the Conv1d kernel
        (following braindecode's ``Conv1dWithConstraint``). Default is 1.0.
        Set to None to disable the constraint.
    init : {"random", "identity", "bipolar"}
        Initialisation scheme for the Conv1d kernel.

        * ``"random"`` (default): PyTorch's default Kaiming-uniform
          initialisation.
        * ``"identity"``: rows of the kernel that correspond to target channel
          names present (possibly under a rename) in the input channel list
          are set to a one-hot vector selecting that input; all other rows are
          zero.  On step 0 this makes the adapter a pass-through for the
          matching channels and a zero pad for missing ones, mirroring the
          behaviour of the zero-fill channel remapping that the adapter
          replaces.  Requires :attr:`target_channel_names` and, at build time,
          ``input_channel_names``.
        * ``"bipolar"``: rows correspond to bipolar derivations specified as
          ``"A-B"`` strings in ``target_channel_names`` (unipolar entries
          ``"A"`` are also accepted and treated as ``"A-None"``, i.e. +1 only
          on A).  For each covered pair, +1 is added to ``weight[target,
          idx(A), 0]`` and -1 to ``weight[target, idx(B), 0]`` **on top of**
          the default Kaiming-uniform init (additive, not overwriting).
          Rows whose pair is partially or fully missing retain the Kaiming
          baseline so they can still learn -- important for models whose
          first op is ``|STFT|`` (e.g. BIOT), where a fully zero row would
          yield a zero gradient through the ``abs`` singularity.  Requires
          :attr:`target_channel_names` and, at build time,
          ``input_channel_names``.
    target_channel_names : list[str] or None
        Required for ``init="identity"`` and ``init="bipolar"``.  Names of the
        target channels in the order the pretrained model expects them.  Must
        have length ``n_target_channels``.  Matching against input channel
        names is case-insensitive.  For ``init="bipolar"`` entries may be
        bipolar pairs ``"A-B"``; entries without ``-`` are treated as
        unipolar.
    rename_mapping : dict[str, str] or None
        Optional canonicalisation map applied to input channel names before
        identity/bipolar-init matching.  E.g. ``{"T3": "T7", "E9": "Fp2"}``.
        Ignored when ``init="random"``.
    """

    model_config = pydantic.ConfigDict(extra="forbid")
    n_target_channels: int
    max_norm: float | None = 1.0
    init: tp.Literal["random", "identity", "bipolar"] = "random"
    target_channel_names: list[str] | None = None
    rename_mapping: dict[str, str] | None = None

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.init in ("identity", "bipolar"):
            if self.target_channel_names is None:
                raise ValueError(
                    f"init={self.init!r} requires target_channel_names to be set."
                )
            if len(self.target_channel_names) != self.n_target_channels:
                raise ValueError(
                    f"target_channel_names has length {len(self.target_channel_names)} "
                    f"but n_target_channels is {self.n_target_channels}."
                )

    def build(
        self,
        n_in_channels: int,
        input_channel_names: list[str] | None = None,
    ) -> nn.Module:
        if self.max_norm is not None:
            from braindecode.modules import Conv1dWithConstraint

            conv: nn.Conv1d = Conv1dWithConstraint(
                n_in_channels,
                self.n_target_channels,
                kernel_size=1,
                max_norm=self.max_norm,
            )
        else:
            conv = nn.Conv1d(n_in_channels, self.n_target_channels, kernel_size=1)

        if self.init == "identity":
            self._apply_identity_init(conv, input_channel_names)
        elif self.init == "bipolar":
            self._apply_bipolar_init(conv, input_channel_names)

        return conv

    def _apply_identity_init(
        self,
        conv: nn.Conv1d,
        input_channel_names: list[str] | None,
    ) -> None:
        """Overwrite ``conv`` weights with a name-matched identity pattern.

        For each target channel name, finds the (possibly renamed) input
        channel whose name matches and sets ``weight[target, input, 0] = 1``.
        All other weights and the bias are zero.
        """
        if input_channel_names is None:
            raise ValueError(
                "init='identity' requires input_channel_names at build time; "
                "the calling DownstreamWrapper must forward them."
            )
        assert self.target_channel_names is not None  # enforced in post-init
        n_in = conv.weight.shape[1]
        if len(input_channel_names) != n_in:
            raise ValueError(
                f"input_channel_names has length {len(input_channel_names)} "
                f"but conv expects {n_in} input channels."
            )

        rename = self.rename_mapping or {}
        canon_inputs = [rename.get(name, name).upper() for name in input_channel_names]
        target_upper_to_idx: dict[str, int] = {}
        for i, tname in enumerate(self.target_channel_names):
            target_upper_to_idx.setdefault(tname.upper(), i)

        weight = torch.zeros_like(conv.weight)
        covered: set[int] = set()
        for j, canon in enumerate(canon_inputs):
            target_idx = target_upper_to_idx.get(canon)
            if target_idx is None:
                continue
            if target_idx in covered:
                LOGGER.warning(
                    "ChannelProjection identity init: target channel %r is "
                    "already covered by another input; input %r (%r) "
                    "contributes additively.",
                    self.target_channel_names[target_idx],
                    input_channel_names[j],
                    canon,
                )
            weight[target_idx, j, 0] = 1.0
            covered.add(target_idx)

        missing = [
            self.target_channel_names[i]
            for i in range(self.n_target_channels)
            if i not in covered
        ]
        LOGGER.info(
            "ChannelProjection identity init: %d/%d target channels covered "
            "(missing: %s); %d/%d input channels contribute.",
            len(covered),
            self.n_target_channels,
            missing if missing else "none",
            sum(1 for canon in canon_inputs if canon in target_upper_to_idx),
            n_in,
        )

        with torch.no_grad():
            _raw_conv_weight(conv).copy_(weight)
            if conv.bias is not None:
                conv.bias.zero_()

    def _apply_bipolar_init(
        self,
        conv: nn.Conv1d,
        input_channel_names: list[str] | None,
    ) -> None:
        """Add a name-matched bipolar pattern on top of the Kaiming baseline.

        For each target entry parsed as ``"A-B"`` (or unipolar ``"A"``), finds
        the (possibly renamed) input channels for A and B and **adds** +1 /
        -1 to ``weight[target, idx(A), 0]`` / ``weight[target, idx(B), 0]`` on
        top of the default Kaiming-uniform init produced by
        ``nn.Conv1d.__init__``.  Rows whose pair is partially or fully
        missing keep the Kaiming baseline.

        This additive choice is critical for models whose first op is a
        magnitude spectrogram (e.g. BIOT's ``|torch.stft(x)|``): an
        identically-zero row would produce a zero output, and the ``abs``
        singularity at zero would freeze the row's gradient, so the adapter
        could never recover.  The small Kaiming noise lets every row learn
        while the +1/-1 pattern preserves the pretrained channel-token
        semantics on covered pairs at step 0.
        """
        if input_channel_names is None:
            raise ValueError(
                "init='bipolar' requires input_channel_names at build time; "
                "the calling DownstreamWrapper must forward them."
            )
        assert self.target_channel_names is not None  # enforced in post-init
        n_in = conv.weight.shape[1]
        if len(input_channel_names) != n_in:
            raise ValueError(
                f"input_channel_names has length {len(input_channel_names)} "
                f"but conv expects {n_in} input channels."
            )

        rename = self.rename_mapping or {}
        canon_inputs = [rename.get(name, name).upper() for name in input_channel_names]
        canon_to_idx: dict[str, int] = {}
        for j, canon in enumerate(canon_inputs):
            canon_to_idx.setdefault(canon, j)

        pattern = torch.zeros_like(conv.weight)
        fully_covered: list[str] = []
        partial: list[str] = []
        missing: list[str] = []
        for i, tname in enumerate(self.target_channel_names):
            if "-" in tname:
                pos_name, neg_name = tname.split("-", 1)
            else:
                pos_name, neg_name = tname, None
            pos_idx = canon_to_idx.get(pos_name.upper())
            neg_idx = canon_to_idx.get(neg_name.upper()) if neg_name else None

            if neg_name is None:
                if pos_idx is not None:
                    pattern[i, pos_idx, 0] = 1.0
                    fully_covered.append(tname)
                else:
                    missing.append(tname)
            else:
                if pos_idx is not None and neg_idx is not None:
                    pattern[i, pos_idx, 0] = 1.0
                    pattern[i, neg_idx, 0] = -1.0
                    fully_covered.append(tname)
                elif pos_idx is not None or neg_idx is not None:
                    partial.append(tname)
                else:
                    missing.append(tname)

        LOGGER.info(
            "ChannelProjection bipolar init: %d/%d fully covered (+1/-1 added "
            "over Kaiming baseline), %d partial (kept at Kaiming baseline so "
            "they can still learn), %d missing (kept at Kaiming baseline). "
            "Covered: %s. Partial: %s. Missing: %s.",
            len(fully_covered),
            self.n_target_channels,
            len(partial),
            len(missing),
            fully_covered if fully_covered else "none",
            partial if partial else "none",
            missing if missing else "none",
        )

        with torch.no_grad():
            # Additive on top of the Kaiming-uniform init that
            # ``nn.Conv1d.__init__`` already applied.
            _raw_conv_weight(conv).add_(pattern)
            if conv.bias is not None:
                conv.bias.zero_()


def _raw_conv_weight(conv: nn.Conv1d) -> torch.Tensor:
    """Return the underlying trainable weight tensor for ``conv``.

    ``Conv1dWithConstraint`` (and any other ``torch.nn.utils.parametrize``
    user) registers parametrizations on ``weight`` -- reading ``conv.weight``
    yields the parametrised *view* and writing to it is rejected unless the
    parametrization defines a ``right_inverse``.  This helper returns the
    raw, trainable tensor so callers can mutate it in-place during custom
    initialisation.
    """
    parametrizations = getattr(conv, "parametrizations", None)
    if parametrizations is not None and "weight" in parametrizations:
        return parametrizations["weight"].original  # type: ignore[no-any-return]
    return conv.weight


class DownstreamWrapper(pydantic.BaseModel):
    """Configuration for wrapping a (pretrained) model for downstream fine-tuning or linear probing.

    This class provides a declarative way to configure how a pretrained model should be
    adapted for downstream tasks, including optional on-the-fly preprocessing, layer freezing,
    output aggregation, and adding a trainable probe on top of the model.

    Parameters
    ----------
    on_the_fly_preprocessor : OnTheFlyPreprocessor | None, optional
        On-the-fly preprocessing applied to the input before the model forward pass.
        Typically model-specific (e.g. QuantileAbsScaler for BIOT). Default is None.
    channel_adapter_config : ChannelMerger | ChannelProjection | None, optional
        Configuration for a channel adapter that projects from arbitrary input
        channels to a fixed number of target channels.  Supply a ``ChannelMerger``
        for position-based spatial attention, or a ``ChannelProjection`` for a
        simple Conv1d(kernel_size=1) linear mixing.  Default is None.
    model_output_key : str | int | None, optional
        Key or index to extract from model output dictionary. If None, assumes the model returns a
        tensor directly. Default is None.
    layers_to_freeze : list[str] | None, optional
        List of layer name patterns to freeze (set requires_grad=False). Cannot be used
        together with layers_to_unfreeze. Default is None.
    layers_to_unfreeze : list[str] | tp.Literal["last"] | None, optional
        List of layer name patterns to unfreeze (set requires_grad=True), while freezing all
        others. Cannot be used together with layers_to_freeze. If "last", unfreezes the last
        layer (nn.Module) of the model. Default is None.
    strict_matching : bool, optional
        If True, when freezing/unfreezing layers, only the first part of the layer name
        (before the first dot) must match exactly. If False, any part of the layer name
        can match the patterns. Default is True.
    aggregation : {"flatten", "mean", "first"} or int, optional
        Method to aggregate the model output.
        ``"flatten"`` flattens all dimensions except batch;
        ``"mean"`` averages over the temporal/sequence dimension (dim=1);
        ``"first"`` selects only the first timestep/token;
        an ``int`` splits into n groups, averages each group, then concatenates;
        ``None`` performs no aggregation.
    probe_config : Mlp | "linear" | None, optional
        Configuration for the probe layer added on top.
        ``None`` uses identity (no additional layer), e.g. if the model already
        has a linear layer of the right output size.
        ``"linear"`` adds a single linear layer.
        An ``Mlp`` instance adds a multi-layer perceptron with specified configuration.
    """

    model_config = pydantic.ConfigDict(extra="forbid")
    on_the_fly_preprocessor: OnTheFlyPreprocessor | None = None
    channel_adapter_config: ChannelMerger | ChannelProjection | None = None
    model_output_key: str | int | None = None
    layers_to_freeze: list[str] | None = None
    layers_to_unfreeze: list[str] | tp.Literal["last"] | None = None
    strict_matching: bool = True
    aggregation: tp.Literal["flatten", "mean", "first"] | int | None = "flatten"
    probe_config: Mlp | tp.Literal["linear"] | None = "linear"

    @property
    def n_adapter_target_channels(self) -> int | None:
        """Target channel count of the adapter, or ``None`` if no adapter is configured."""
        if self.channel_adapter_config is None:
            return None
        if isinstance(self.channel_adapter_config, ChannelMerger):
            return self.channel_adapter_config.n_virtual_channels
        return self.channel_adapter_config.n_target_channels

    def model_post_init(self, __context):
        super().model_post_init(__context)

        if self.layers_to_freeze is not None and self.layers_to_unfreeze is not None:
            raise ValueError(
                "Only one of layers_to_freeze and layers_to_unfreeze can be specified at once."
            )

    def build(
        self,
        model: nn.Module,
        dummy_batch: dict[str, torch.Tensor | None],
        n_outputs: int,
        input_channel_names: list[str] | None = None,
    ) -> "DownstreamWrapperModel":
        preprocessor = None
        if self.on_the_fly_preprocessor is not None:
            preprocessor = self.on_the_fly_preprocessor.build()

        channel_adapter: nn.Module | None = None
        adapter_needs_positions = False
        if self.channel_adapter_config is not None:
            if isinstance(self.channel_adapter_config, ChannelMerger):
                channel_adapter = self.channel_adapter_config.build()
                adapter_needs_positions = True
            else:
                input_key = next(iter(dummy_batch))
                x = dummy_batch[input_key]
                assert x is not None
                channel_adapter = self.channel_adapter_config.build(
                    x.shape[1], input_channel_names=input_channel_names
                )

        with torch.no_grad():
            model.eval()
            if channel_adapter is not None:
                input_key = next(iter(dummy_batch))
                x = dummy_batch[input_key]
                assert x is not None
                if adapter_needs_positions:
                    subject_ids = dummy_batch.get(
                        "subject_ids",
                        torch.zeros(x.shape[0], dtype=torch.long),
                    )
                    ch_pos = dummy_batch.get("channel_positions")
                    x_adapted = channel_adapter(x, subject_ids, ch_pos)
                else:
                    x_adapted = channel_adapter(x)
                model_batch = {input_key: x_adapted}
                orig_output = model(**model_batch)
            else:
                orig_output = model(**dummy_batch)
            if self.model_output_key is not None:
                orig_output = orig_output[self.model_output_key]
            model.train()

        wrapper_model = DownstreamWrapperModel(
            model,
            orig_output.shape[1:],
            preprocessor=preprocessor,
            channel_adapter=channel_adapter,
            adapter_needs_positions=adapter_needs_positions,
            model_output_key=self.model_output_key,
            wrapper_n_outputs=n_outputs,
            layers_to_freeze=self.layers_to_freeze,
            layers_to_unfreeze=self.layers_to_unfreeze,
            strict_matching=self.strict_matching,
            aggregation=self.aggregation,
            probe_config=self.probe_config,
        )

        # Sanity check (wrapper handles preprocessing internally)
        wrapper_output = wrapper_model(**dummy_batch)
        assert wrapper_output.shape[-1] == n_outputs

        return wrapper_model


class DownstreamWrapperModel(nn.Module):
    """Wrapper for downstream evaluation of pretrained models.

    Handles the full pipeline: optional preprocessing -> channel adapter ->
    model -> output key selection -> aggregation -> probe.
    """

    def __init__(
        self,
        model: nn.Module,
        brain_model_output_size: torch.Size,
        model_output_key: str | int | None,
        wrapper_n_outputs: int,
        preprocessor: nn.Module | None = None,
        channel_adapter: nn.Module | None = None,
        adapter_needs_positions: bool = False,
        layers_to_freeze: list[str] | None = None,
        layers_to_unfreeze: list[str] | tp.Literal["last"] | None = None,
        strict_matching: bool = True,
        aggregation: tp.Literal["flatten", "mean", "first"] | int | None = "flatten",
        probe_config: Mlp | tp.Literal["linear"] | None = None,
    ):
        super().__init__()

        self.preprocessor = preprocessor
        self.channel_adapter = channel_adapter
        self._adapter_needs_positions = adapter_needs_positions
        self.wrapped_model = model
        self.model_output_key = model_output_key

        inner_sig = inspect.signature(model.forward)
        self._inner_param_names = set(inner_sig.parameters.keys())
        self._inner_accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in inner_sig.parameters.values()
        )

        self._apply_freeze(layers_to_freeze, layers_to_unfreeze, strict_matching)
        n_inputs = self._build_aggregation(aggregation, brain_model_output_size)
        self._build_probe(probe_config, n_inputs, wrapper_n_outputs)

    def _apply_freeze(
        self,
        layers_to_freeze: list[str] | None,
        layers_to_unfreeze: list[str] | tp.Literal["last"] | None,
        strict_matching: bool,
    ) -> None:
        """Freeze or unfreeze model parameters based on layer name patterns."""
        if layers_to_freeze is not None:
            for name, param in self.wrapped_model.named_parameters():
                if strict_matching:
                    requires_grad = name.split(".")[0] not in layers_to_freeze
                else:
                    requires_grad = not any(
                        pattern in name for pattern in layers_to_freeze
                    )
                param.requires_grad = requires_grad

        elif layers_to_unfreeze == "last":
            param_names = list(self.wrapped_model.named_parameters())
            last_layer_name = param_names[-1][0].rsplit(".", 1)[0]
            unfrozen_layers = []
            for name, param in self.wrapped_model.named_parameters():
                if name.startswith(last_layer_name):
                    unfrozen_layers.append(name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            LOGGER.warning(f"Unfreezing {unfrozen_layers}")

        elif layers_to_unfreeze is not None:
            for name, param in self.wrapped_model.named_parameters():
                if strict_matching:
                    requires_grad = name.split(".")[0] in layers_to_unfreeze
                else:
                    requires_grad = any(pattern in name for pattern in layers_to_unfreeze)
                param.requires_grad = requires_grad

    def _build_aggregation(
        self,
        aggregation: tp.Literal["flatten", "mean", "first"] | int | None,
        brain_model_output_size: torch.Size,
    ) -> int:
        """Build the aggregation module and return the flattened input size for the probe."""
        self.aggregation: nn.Module
        if aggregation is None:
            self.aggregation = nn.Identity()
            return brain_model_output_size.numel()
        elif aggregation == "flatten":
            self.aggregation = nn.Flatten(start_dim=1)
            return brain_model_output_size.numel()
        elif aggregation == "first":
            assert len(brain_model_output_size) == 2
            self.aggregation = IndexSelect(dim=1, index=torch.LongTensor([0]))
            return brain_model_output_size[1]
        elif aggregation == "mean":
            dim: int | tuple[int, ...] = 1
            if len(brain_model_output_size) == 2:  # (n_patches, emb_dim)
                dim = 1
            elif len(brain_model_output_size) == 3:  # (n_chans, n_patches, emb_dim)
                dim = (1, 2)
            else:
                raise ValueError(
                    f"aggregation='mean' requires model output of 3D or 4D "
                    f"(got brain_model_output_size={brain_model_output_size})"
                )
            self.aggregation = Mean(dim=dim)
            return brain_model_output_size[-1]
        elif isinstance(aggregation, int):
            assert len(brain_model_output_size) == 2
            self.aggregation = ConcatGroupedMean(dim=1, n_splits=aggregation)
            return aggregation * brain_model_output_size[1]
        else:
            raise NotImplementedError()

    def _build_probe(
        self,
        probe_config: Mlp | tp.Literal["linear"] | None,
        n_inputs: int,
        n_outputs: int,
    ) -> None:
        """Build the probe (classification/regression head) on top of the aggregated representations."""
        self.probe: nn.Module
        if probe_config is None:
            self.probe = nn.Identity()
        elif probe_config == "linear":
            self.probe = nn.Linear(n_inputs, n_outputs)
        else:
            assert not isinstance(probe_config, str)
            self.probe = probe_config.build(
                input_size=n_inputs,
                output_size=n_outputs,
            )

    def forward(self, *args, return_embedding: bool = False, **kwargs) -> torch.Tensor:
        if self.preprocessor is not None:
            if args:
                x, *rest_args = args
                x, ch_pos = self.preprocessor(x, kwargs.get("channel_positions"))
                args = (x, *rest_args)
            else:
                input_key = next(iter(kwargs))
                x = kwargs[input_key]
                x, ch_pos = self.preprocessor(x, kwargs.get("channel_positions"))
                kwargs = {**kwargs, input_key: x}
            if ch_pos is not None and "channel_positions" in kwargs:
                kwargs["channel_positions"] = ch_pos

        if self.channel_adapter is not None:
            input_key = next(iter(kwargs))
            x = kwargs[input_key]
            if self._adapter_needs_positions:
                subject_ids = kwargs.get(
                    "subject_ids",
                    torch.zeros(x.shape[0], dtype=torch.long, device=x.device),
                )
                ch_pos = kwargs.get("channel_positions")
                x = self.channel_adapter(x, subject_ids, ch_pos)
            else:
                x = self.channel_adapter(x)
            kwargs = {**kwargs, input_key: x}

        if not self._inner_accepts_var_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in self._inner_param_names}

        out = self.wrapped_model(*args, **kwargs)
        if self.model_output_key is not None:
            out = out[self.model_output_key]
        out = self.aggregation(out)
        if return_embedding:
            return out
        out = self.probe(out)
        return out
