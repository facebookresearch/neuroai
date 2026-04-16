# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SimpleConv, taken and slightly modified from brainmagick."""

import logging
import typing as tp
from functools import partial

import torch
from torch import nn

from .base import BaseModelConfig
from .common import (
    BahdanauAttention,
    ChannelMerger,
    FourierEmb,
    LayerScale,
    Mlp,
    SubjectLayers,
)
from .transformer import TransformerEncoder

logger = logging.getLogger(__name__)


class ConvSequence(nn.Module):
    def __init__(
        self,
        channels: tp.Sequence[int],
        kernel: int = 4,
        dilation_growth: int = 1,
        dilation_period: int | None = None,
        stride: int = 2,
        dropout: float = 0.0,
        leakiness: float = 0.0,
        groups: int = 1,
        decode: bool = False,
        batch_norm: bool = False,
        dropout_input: float = 0.0,
        skip: bool = False,
        scale: float | None = None,
        rewrite: bool = False,
        activation_on_last: bool = True,
        post_skip: bool = False,
        glu: int = 0,
        glu_context: int = 0,
        glu_glu: bool = True,
        activation: tp.Any = None,
    ) -> None:
        super().__init__()
        dilation = 1
        channels = tuple(channels)
        self.skip = skip
        self.sequence = nn.ModuleList()
        self.glus = nn.ModuleList()
        if activation is None:
            activation = partial(nn.LeakyReLU, leakiness)
        Conv = nn.Conv1d if not decode else nn.ConvTranspose1d
        # build layers
        for k, (chin, chout) in enumerate(zip(channels[:-1], channels[1:])):
            layers: tp.List[nn.Module] = []
            is_last = k == len(channels) - 2

            # Set dropout for the input of the conv sequence if defined
            if k == 0 and dropout_input:
                if not (0.0 < dropout_input < 1.0):
                    raise ValueError(f"{dropout_input=} must be in (0,1)")
                layers.append(nn.Dropout(dropout_input))

            # conv layer
            if dilation_growth > 1:
                if kernel % 2 == 0:
                    raise ValueError(f"Odd kernel required with dilation, got {kernel}")
            if dilation_period and (k % dilation_period) == 0:
                dilation = 1
            pad = kernel // 2 * dilation
            layers.append(
                Conv(
                    chin,
                    chout,
                    kernel,
                    stride,
                    pad,
                    dilation=dilation,
                    groups=groups if k > 0 else 1,
                )
            )
            dilation *= dilation_growth
            # non-linearity
            if activation_on_last or not is_last:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(num_features=chout))
                layers.append(activation())
                if dropout:
                    layers.append(nn.Dropout(dropout))
                if rewrite:
                    layers += [nn.Conv1d(chout, chout, 1), nn.LeakyReLU(leakiness)]
                    # layers += [nn.Conv1d(chout, 2 * chout, 1), nn.GLU(dim=1)]
            if chin == chout and skip:
                if scale is not None:
                    layers.append(LayerScale(chout, scale))
                if post_skip:
                    layers.append(Conv(chout, chout, 1, groups=chout, bias=False))

            self.sequence.append(nn.Sequential(*layers))
            if glu and (k + 1) % glu == 0:
                ch = 2 * chout if glu_glu else chout
                act = nn.GLU(dim=1) if glu_glu else activation()
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(chout, ch, 1 + 2 * glu_context, padding=glu_context),
                        act,
                    )
                )
            else:
                self.glus.append(None)  # type: ignore

    def forward(self, x: tp.Any) -> tp.Any:
        for module_idx, module in enumerate(self.sequence):
            old_x = x
            x = module(x)
            if self.skip and x.shape == old_x.shape:
                x = x + old_x
            glu = self.glus[module_idx]
            if glu is not None:
                x = glu(x)
        return x


class SimpleConv(BaseModelConfig):
    """1-D convolutional encoder, adapted from brainmagick.

    Parameters
    ----------
    hidden : int
        Number of channels in the first convolutional layer.
    depth : int
        Number of convolutional layers.
    linear_out : bool
        Use a single transposed convolution as the output projection.
    complex_out : bool
        Use a two-layer transposed-convolution output projection with a
        non-linearity in between. Mutually exclusive with *linear_out*.
    kernel_size : int
        Kernel size for every convolutional layer (must be odd).
    growth : float
        Multiplicative channel growth factor per layer.
    dilation_growth : int
        Multiplicative dilation growth factor per layer.
    dilation_period : int or None
        If set, reset dilation to 1 every *dilation_period* layers.
    skip : bool
        Add residual skip connections when input and output shapes match.
    post_skip : bool
        Append a depth-wise convolution after each skip connection.
    scale : float or None
        If set, apply :class:`LayerScale` with this initial value after
        each skip connection.
    rewrite : bool
        Append a 1x1 convolution + LeakyReLU after each layer.
    groups : int
        Number of groups for grouped convolutions (first layer always uses 1).
    glu : int
        If non-zero, insert a GLU gate every *glu* layers.
    glu_context : int
        Context (padding) size for the GLU convolution.
    glu_glu : bool
        If True the gate uses ``nn.GLU``; otherwise the layer activation.
    gelu : bool
        Use GELU activation instead of (Leaky)ReLU.
    dropout : float
        Channel-dropout probability (currently raises ``NotImplementedError``).
    dropout_rescale : bool
        Rescale activations after channel dropout.
    conv_dropout : float
        Dropout probability inside each convolutional block.
    dropout_input : float
        Dropout probability applied to the input of the convolutional stack.
    batch_norm : bool
        Apply batch normalization after each convolution.
    relu_leakiness : float
        Negative slope for ``LeakyReLU`` (0 gives standard ReLU).
    transformer_config : TransformerEncoder or None
        If set, append a Transformer encoder after the convolutional stack.
    subject_layers_config : SubjectLayers or None
        If set, prepend a per-subject linear projection.
    subject_layers_dim : {"input", "hidden"}
        Dimension used for the subject-layer projection.
    merger_config : ChannelMerger or None
        If set, prepend a :class:`ChannelMerger` for multi-montage support.
    initial_linear : int
        If non-zero, prepend a 1x1 convolution projecting to this many channels.
    initial_depth : int
        Number of 1x1 convolution layers in the initial projection.
    initial_nonlin : bool
        Append a non-linearity after the initial 1x1 projection stack.
    backbone_out_channels : int or None
        If set, force the backbone output to this many channels.
    """

    # Channels
    hidden: int = 16
    # Overall structure
    depth: int = 4
    linear_out: bool = False
    complex_out: bool = False
    # Conv layer
    kernel_size: int = 5
    growth: float = 1.0
    dilation_growth: int = 2
    dilation_period: int | None = None
    skip: bool = False
    post_skip: bool = False
    scale: float | None = None
    rewrite: bool = False
    groups: int = 1
    glu: int = 0
    glu_context: int = 0
    glu_glu: bool = True
    gelu: bool = False
    # Dropouts, BN, activations
    dropout: float = 0.0
    dropout_rescale: bool = True
    conv_dropout: float = 0.0
    dropout_input: float = 0.0
    batch_norm: bool = False
    relu_leakiness: float = 0.0
    # Optional transformer
    transformer_config: TransformerEncoder | None = None
    # Subject-specific settings
    subject_layers_config: SubjectLayers | None = None
    subject_layers_dim: tp.Literal["input", "hidden"] = "hidden"
    # Channel attention for multi-montage support
    merger_config: ChannelMerger | None = ChannelMerger(
        n_virtual_channels=270,
        fourier_emb_config=FourierEmb(
            n_freqs=None,
            total_dim=2048,
            n_dims=2,
        ),
        dropout=0.2,
        usage_penalty=0.0,
        per_subject=False,
        embed_ref=False,
    )
    # Architectural details
    initial_linear: int = 0
    initial_depth: int = 1
    initial_nonlin: bool = False
    backbone_out_channels: int | None = None  # If provided, the output of the
    # backbone (i.e. layer before the output heads) will have this dimensionality

    def build(self, n_in_channels: int, n_outputs: int) -> "SimpleConvModel":
        return SimpleConvModel(n_in_channels, n_outputs, config=self)


class SimpleConvModel(nn.Module):
    """``nn.Module`` implementation of :class:`SimpleConv`."""

    def __init__(
        self,
        # Channels
        in_channels: int,
        out_channels: int,
        config: SimpleConv | None = None,
    ):
        super().__init__()
        config = config if config is not None else SimpleConv()

        self.out_channels = out_channels
        self.backbone_out_channels = config.backbone_out_channels or out_channels

        activation: nn.Module | tp.Callable
        if config.gelu:
            activation = nn.GELU
        elif config.relu_leakiness:
            activation = partial(nn.LeakyReLU, config.relu_leakiness)
        else:
            activation = nn.ReLU

        if config.kernel_size % 2 != 1:
            raise ValueError(f"{config.kernel_size=}, must be odd for padding")

        self.dropout = None

        self.initial_linear = None
        if config.dropout > 0.0:
            raise NotImplementedError("To be reimplemented here.")
            # self.dropout = ChannelDropout(dropout, dropout_rescale)

        self.merger = None
        if config.merger_config is not None:
            self.merger = config.merger_config.build()
            in_channels = config.merger_config.n_virtual_channels

        if config.initial_linear:
            init: list[nn.Module | tp.Callable] = [
                nn.Conv1d(in_channels, config.initial_linear, 1)
            ]
            for _ in range(config.initial_depth - 1):
                init += [
                    activation(),
                    nn.Conv1d(config.initial_linear, config.initial_linear, 1),
                ]
            if config.initial_nonlin:
                init += [activation()]
            self.initial_linear = nn.Sequential(*init)  # type: ignore[arg-type]
            in_channels = config.initial_linear

        self.subject_layers = None
        if config.subject_layers_config:
            dim = {"hidden": config.hidden, "input": in_channels}[
                config.subject_layers_dim
            ]
            self.subject_layers = config.subject_layers_config.build(in_channels, dim)
            in_channels = dim

        # compute the sequences of channel sizes
        sizes = [in_channels]
        sizes += [
            int(round(config.hidden * config.growth**k)) for k in range(config.depth)
        ]

        params: dict[str, tp.Any]
        params = dict(
            kernel=config.kernel_size,
            stride=1,
            leakiness=config.relu_leakiness,
            dropout=config.conv_dropout,
            dropout_input=config.dropout_input,
            batch_norm=config.batch_norm,
            dilation_growth=config.dilation_growth,
            groups=config.groups,
            dilation_period=config.dilation_period,
            skip=config.skip,
            post_skip=config.post_skip,
            scale=config.scale,
            rewrite=config.rewrite,
            glu=config.glu,
            glu_context=config.glu_context,
            glu_glu=config.glu_glu,
            activation=activation,
        )

        final_channels = sizes[-1]

        self.final: nn.Module | nn.Sequential | None = None
        pad = 0
        kernel = 1
        stride = 1

        if config.linear_out:
            if config.complex_out:
                raise ValueError("linear_out and complex_out are mutually exclusive")
            self.final = nn.ConvTranspose1d(
                final_channels, self.backbone_out_channels, kernel, stride, pad
            )
        elif config.complex_out:
            self.final = nn.Sequential(
                nn.Conv1d(final_channels, 2 * final_channels, 1),
                activation(),
                nn.ConvTranspose1d(
                    2 * final_channels, self.backbone_out_channels, kernel, stride, pad
                ),
            )
        else:
            params["activation_on_last"] = False
            sizes[-1] = self.backbone_out_channels

        self.encoder = ConvSequence(sizes, **params)

        self.transformer = None
        if config.transformer_config:
            self.transformer = config.transformer_config.build(
                dim=self.backbone_out_channels
            )

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        channel_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the convolutional encoder.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, C, T)``.
        subject_ids : Tensor or None
            Per-example subject indices, shape ``(B,)``.
        channel_positions : Tensor or None
            Normalised electrode coordinates, shape ``(B, C, D)``.
        """
        length = x.shape[-1]

        # if self.dropout is not None:
        #     x = self.dropout(x, batch)

        if self.merger is not None:
            x = self.merger(x, subject_ids, channel_positions)

        if self.initial_linear is not None:
            x = self.initial_linear(x)

        if self.subject_layers is not None:
            x = self.subject_layers(x, subject_ids)

        x = self.encoder(x)
        if self.final is not None:
            x = self.final(x)
        if x.shape[-1] < length:
            raise ValueError(f"Expected output time dim >= {length}, got {x.shape[-1]}")
        x = x[:, :, :length]

        if self.transformer:
            x = self.transformer(x.transpose(1, 2)).transpose(1, 2)

        return x


class SimpleConvTimeAgg(SimpleConv):
    """SimpleConv with temporal aggregation layer and optional output heads.

    Parameters
    ----------
    time_agg_out :
        - "gap" : Global average pooling
        - "linear" : Linear layer with one output
        - "att" : Bahdanau attention layer
    n_time_groups :
        Number of groups within which to apply temporal aggregation, e.g. 4 means the time
        dimension will be split into 4 groups and each group will be aggregated (and optionally
        projected) separately.
    """

    # Temporal aggregation
    time_agg_out: tp.Literal["gap", "linear", "att"] = "gap"
    n_time_groups: int | None = None

    # Output head(s)
    output_head_config: Mlp | dict[str, Mlp] | None = None

    def build(self, n_in_channels: int, n_outputs: int) -> "SimpleConvTimeAggModel":
        return SimpleConvTimeAggModel(n_in_channels, n_outputs, config=self)


class SimpleConvTimeAggModel(SimpleConvModel):
    """``nn.Module`` implementation of :class:`SimpleConvTimeAgg`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        config: SimpleConvTimeAgg | None = None,
    ):
        config = config if config is not None else SimpleConvTimeAgg()
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, config=config
        )

        # Output aggregation layer
        self.n_time_groups = config.n_time_groups
        self.time_agg_out: nn.Module | None
        if config.time_agg_out == "gap":
            self.time_agg_out = nn.AdaptiveAvgPool1d(1)
        elif config.time_agg_out == "linear":
            self.time_agg_out = nn.LazyLinear(1)
        elif config.time_agg_out == "att":
            self.time_agg_out = BahdanauAttention(input_size=None, hidden_size=256)
        else:
            self.time_agg_out = None

        # Separate output head(s)
        self.output_head: nn.Module | None
        if config.output_head_config is None:
            self.output_head = None
        else:
            if self.time_agg_out is None:
                raise NotImplementedError("Output heads require temporal aggregation.")
            if isinstance(config.output_head_config, Mlp):
                head_cfg, output_size = self._resolve_head_sentinel(
                    config.output_head_config
                )
                self.output_head = head_cfg.build(
                    input_size=self.backbone_out_channels,
                    output_size=output_size,
                )
            elif isinstance(config.output_head_config, dict):
                self.output_head = nn.ModuleDict()
                for name, head_config in config.output_head_config.items():
                    head_cfg, output_size = self._resolve_head_sentinel(head_config)
                    self.output_head[name] = head_cfg.build(
                        input_size=self.backbone_out_channels,
                        output_size=output_size,
                    )

    def _resolve_head_sentinel(self, cfg: Mlp) -> tuple[Mlp, int | None]:
        """Strip the ``-1`` output-size sentinel from an ``Mlp`` config.

        Returns a (possibly updated) config and the resolved ``output_size``.
        """
        if cfg.hidden_sizes and cfg.hidden_sizes[-1] == -1:
            cfg = cfg.model_copy(update={"hidden_sizes": cfg.hidden_sizes[:-1]})
            return cfg, self.out_channels
        return cfg, None

    def forward(  # type: ignore
        self, x, subject_ids=None, channel_positions=None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        B = x.shape[0]
        x = super().forward(
            x, subject_ids=subject_ids, channel_positions=channel_positions
        )

        # Break down time dimension into groups, with padding that mirrors what
        # ns.extractors.meta.TimeAggregatedExtractor does
        if self.n_time_groups is not None:
            x = x.permute(2, 0, 1)  # (B, F, T) -> (T, B, F)
            x = torch.tensor_split(
                x, self.n_time_groups, dim=0
            )  # -> N x (~(T // N), B, F)
            x = nn.utils.rnn.pad_sequence(x, batch_first=True)  # -> (N, ~(T // N), B, F)
            x = x.permute(0, 2, 3, 1)  # -> (N, B, F, ~(T // N))
            x = x.flatten(end_dim=1)  # -> (NxB, F, ~(T // N))

        if self.time_agg_out is not None:
            x = self.time_agg_out(x)
            if x.ndim == 3:
                x = x.squeeze(2)  # Remove singleton dimension

        # Apply output heads (e.g. for separate CLIP and MSE losses)
        if isinstance(self.output_head, nn.ModuleDict):
            x = {name: head(x) for name, head in self.output_head.items()}
        elif self.output_head is not None:
            x = self.output_head(x)

        # Build the time dimension back up
        if self.n_time_groups is not None:
            if isinstance(x, dict):
                raise NotImplementedError
            x = x.reshape(self.n_time_groups, B, -1)  # (NxB, F) -> (N, B, F)
            x = x.permute(1, 2, 0)  # -> (B, N, F)

        return x
