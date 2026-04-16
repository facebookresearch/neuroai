# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging

import torch
from torch import nn

from .base import BaseModelConfig
from .common import TemporalDownsampling
from .conformer import Conformer
from .simpleconv import SimpleConv
from .simplerconv import SimplerConv
from .transformer import TransformerEncoder

logger = logging.getLogger(__name__)


class ConvTransformer(BaseModelConfig):
    """Convolutional encoder followed by optional temporal aggregation and a transformer.

    Parameters
    ----------
    dim :
        Internal token dimension.
    encoder_config :
        Configuration for the convolutional encoder.
    temporal_downsampling_config :
        Configuration for the optional temporal downsampling module.
    conv_pos_emb_kernel_size :
        If provided, use convolutional positional embedding with this kernel size.
    neuro_device_types :
        List of expected neuro device types that can be used to embed the device type in the
        transformer.
    add_cls_token :
        If True, add a [CLS] token to the input of the transformer.
    pre_transformer_layer_norm :
        If True, apply layer normalization before the transformer.
    transformer_config :
        Configuration for the transformer encoder.
    output_avg_pool :
        If True, average the tokens outputted by the transformer.
    output_layer_dim :
        Set to 0 for no output layer, or None to use the same dimension as the transformer. Of
        note, both Bendr and Wav2vec2.0 use an output linear projection though it's not mentioned
        in their respective papers.
    """

    dim: int = 512
    encoder_config: SimplerConv | SimpleConv
    temporal_downsampling_config: TemporalDownsampling | None = None
    conv_pos_emb_kernel_size: int | None = None
    neuro_device_types: list[str] | None = None
    add_cls_token: bool = False
    pre_transformer_layer_norm: bool = False
    transformer_config: TransformerEncoder | Conformer | None = None
    output_avg_pool: bool = False
    output_layer_dim: int | None = 0

    def build(
        self, n_in_channels: int, n_outputs: int | None = None
    ) -> "ConvTransformerModel":
        """Build ConvTransformer model.

        Parameters
        ----------
        n_in_channels :
            Number of input channels.
        n_outputs :
            Number of output dimensions. If None, use the `output_layer_dim` parameter from the
            config.
        """
        return ConvTransformerModel(
            n_in_channels,
            n_outputs or self.output_layer_dim,
            config=self,
        )


class ConvTransformerModel(nn.Module):
    """``nn.Module`` implementation of :class:`ConvTransformer`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None,  # For output linear layer
        config: ConvTransformer,
    ):
        super().__init__()

        # Encoder
        self.dim = config.dim
        self.encoder = config.encoder_config.build(in_channels, self.dim)

        # Temporal downsampling
        self.temporal_downsampling = None
        if config.temporal_downsampling_config is not None:
            self.temporal_downsampling = config.temporal_downsampling_config.build(
                self.dim
            )

        self.pre_transformer_layer_norm = None
        if config.pre_transformer_layer_norm:
            self.pre_transformer_layer_norm = nn.LayerNorm(self.dim)

        # Transformer
        self.transformer = None
        if config.transformer_config is not None:
            self.transformer = config.transformer_config.build(dim=self.dim)

            # [CLS] token
            self.cls_token = None
            if config.add_cls_token:
                self.cls_token = nn.Parameter(
                    torch.empty(
                        (
                            1,
                            1,
                            self.dim,
                        )
                    ),
                    requires_grad=True,
                )
                nn.init.normal_(self.cls_token, mean=0.0, std=1.0)

            # Positional embedding
            # See https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/models/wav2vec2/position_encoder.py
            # See https://github.com/SPOClab-ca/BENDR/blob/main/dn3_ext.py#L522
            self.rel_pos_emb = None
            if config.conv_pos_emb_kernel_size is not None:
                kernel_size = config.conv_pos_emb_kernel_size
                conv = nn.Conv1d(
                    self.dim,
                    self.dim,
                    kernel_size,
                    padding="same",
                    groups=16,  # XXX Parametrize
                )
                nn.init.normal_(
                    conv.weight, mean=0.0, std=(4.0 / (kernel_size * self.dim)) ** 0.5
                )
                nn.init.constant_(conv.bias, 0.0)  # type: ignore
                conv = nn.utils.weight_norm(
                    conv, dim=2
                )  # XXX Will be deprecated in favour of parametrizations, but not yet compatible
                # conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
                self.rel_pos_emb = nn.Sequential(conv, nn.GELU())

            # Device embedding
            self.neuro_device_types = None
            if config.neuro_device_types is not None:
                self.neuro_device_types = sorted(config.neuro_device_types)
                self.neuro_device_emb = nn.Embedding(
                    len(self.neuro_device_types), self.dim
                )

        # Output layer
        self.output_avg_pool = config.output_avg_pool
        self.output_layer = None
        if out_channels != 0:
            self.output_layer = nn.Linear(
                self.dim,
                out_channels or self.dim,
            )

    def _encoder_and_downsampling_forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        channel_positions: torch.Tensor | None = None,
    ):
        z = self.encoder.forward(
            x, subject_ids=subject_ids, channel_positions=channel_positions
        )
        z = z.transpose(2, 1)  # (B, F, T) -> (B, T, F)
        if self.temporal_downsampling is not None:
            z = self.temporal_downsampling(z.unsqueeze(dim=1)).squeeze(dim=1)

        return z

    def _pre_transformer_forward(
        self,
        z: torch.Tensor,
        neuro_device_type: str | None = None,
    ) -> torch.Tensor:
        if self.rel_pos_emb is not None:
            pos_emb = self.rel_pos_emb(z.transpose(2, 1)).transpose(2, 1)
            z = z + pos_emb

        if neuro_device_type is not None and self.neuro_device_types is not None:
            device_ind = torch.tensor(
                self.neuro_device_types.index(neuro_device_type), device=z.device
            )
            device_emb = self.neuro_device_emb(device_ind).unsqueeze(0).unsqueeze(0)
            z = z + device_emb

        if self.pre_transformer_layer_norm is not None:
            z = self.pre_transformer_layer_norm(z)

        if self.cls_token is not None:
            z = torch.cat(
                [
                    self.cls_token.repeat((z.shape[0], 1, 1)),
                    z,
                ],
                dim=1,
            )

        return z

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
        channel_positions: torch.Tensor | None = None,
        neuro_device_type: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through encoder, optional transformer, and output layer.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(B, C, T)``.
        subject_ids : Tensor or None
            Per-example subject indices, shape ``(B,)``.
        channel_positions : Tensor or None
            Normalised electrode coordinates, shape ``(B, C, D)``.
        neuro_device_type : str or None
            Name of device represented in the batch (e.g. ``"Eeg"``,
            ``"Meg"``).  If ``None``, the device embedding is not applied.
        """
        z = self._encoder_and_downsampling_forward(
            x, subject_ids=subject_ids, channel_positions=channel_positions
        )

        if self.transformer is None:
            c_out = z
        else:
            c_in = self._pre_transformer_forward(
                z,
                neuro_device_type=neuro_device_type,
            )
            c_out = self.transformer(c_in)

        if self.output_avg_pool:
            if c_out.ndim != 3:
                raise ValueError(f"Expected 3D tensor for avg pool, got {c_out.ndim}D")
            c_out = c_out.mean(dim=1)

        if self.output_layer is not None:
            c_out = self.output_layer(c_out)

        return {
            "z": z,
            "c_out": c_out,
        }
