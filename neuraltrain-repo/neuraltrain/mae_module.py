# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning module for self-supervised masked autoencoder pretraining."""

import typing as tp

import lightning.pytorch as pl
import torch
from torch import nn

from .models.mae import MaeEncoderModel
from .models.transformer import TransformerEncoder
from .optimizers import BaseOptimizer


def random_masking(
    tokens: torch.Tensor, mask_ratio: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop a random subset of the tokens of each example.

    Parameters
    ----------
    tokens :
        Token sequence of shape ``(B, N, D)``.
    mask_ratio :
        Fraction of the ``N`` tokens to drop.

    Returns
    -------
    kept : torch.Tensor
        Surviving tokens, of shape ``(B, N_kept, D)``, in shuffled order.
    mask : torch.Tensor
        ``1`` on dropped positions and ``0`` on kept ones, of shape ``(B, N)``.
    restore : torch.Tensor
        Indices that put a ``[kept, dropped]`` sequence back into input order,
        of shape ``(B, N)``.
    """
    batch_size, n_tokens, dim = tokens.shape
    if n_tokens < 2:
        raise ValueError(
            f"masking needs at least 2 tokens to leave one of each kind, "
            f"got {n_tokens}: shorten patch_size or lengthen the input window."
        )
    n_kept = min(max(1, round(n_tokens * (1 - mask_ratio))), n_tokens - 1)

    shuffle = torch.rand(batch_size, n_tokens, device=tokens.device).argsort(dim=1)
    restore = shuffle.argsort(dim=1)
    kept = torch.gather(tokens, 1, shuffle[:, :n_kept, None].expand(-1, -1, dim))
    mask = torch.ones(batch_size, n_tokens, device=tokens.device)
    mask[:, :n_kept] = 0.0
    return kept, torch.gather(mask, 1, restore), restore


class MaeModule(pl.LightningModule):
    """Pretrain a :class:`~neuraltrain.models.mae.MaeEncoderModel` by masked reconstruction.

    The input is its own target, so batches need no ``"target"`` key: this
    trains on the unlabelled sliding windows of a ``neuralset`` segmenter
    configured with ``stride``.  Only ``model`` is meant to outlive
    pretraining; the decoder is scaffolding, which is why it lives here and not
    on the encoder config.

    Parameters
    ----------
    model :
        Encoder to pretrain, built with ``n_outputs=None``.
    loss :
        Reconstruction loss, called as ``loss(estimate, target, mask)`` -- see
        :class:`~neuraltrain.losses.losses.MaskedReconstructionLoss`.
    optim_config :
        Optimizer configuration.
    mask_ratio :
        Fraction of time patches hidden from the encoder.
    decoder_config :
        Transformer that reconstructs the masked patches from the encoded ones.
        Defaults to a shallower version of the encoder's own transformer.
    x_name :
        Batch key holding the neuro input.
    """

    def __init__(
        self,
        model: MaeEncoderModel,
        loss: nn.Module,
        optim_config: BaseOptimizer,
        mask_ratio: float = 0.5,
        decoder_config: TransformerEncoder | None = None,
        x_name: str = "input",
    ) -> None:
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must lie in (0, 1), got {mask_ratio}.")
        self.model = model
        self.loss = loss
        self.optim_config = optim_config
        self.mask_ratio = mask_ratio
        self.x_name = x_name

        if decoder_config is None:
            decoder_config = TransformerEncoder(heads=8, depth=2, rotary_pos_emb=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model.dim))
        self.decoder = decoder_config.build(dim=model.dim)
        self.decoder_pred = nn.Linear(model.dim, model.patch_dim)

    def _decode(self, encoded: torch.Tensor, restore: torch.Tensor) -> torch.Tensor:
        """Reconstruct every patch from the encoded ones, in input order."""
        n_dropped = restore.shape[1] - encoded.shape[1]
        mask_tokens = self.mask_token.expand(encoded.shape[0], n_dropped, -1)
        tokens = torch.cat([encoded, mask_tokens], dim=1)
        tokens = torch.gather(
            tokens, 1, restore[:, :, None].expand(-1, -1, tokens.shape[-1])
        )
        tokens = tokens + self.model.positional_embedding(tokens.shape[1]).to(tokens)
        return self.decoder_pred(self.decoder(tokens))

    def _run_step(self, batch: tp.Any, step_name: str) -> torch.Tensor:
        x = batch.data[self.x_name]
        kept, mask, restore = random_masking(self.model.embed(x), self.mask_ratio)
        estimate = self._decode(self.model.encoder(kept), restore)
        loss = self.loss(estimate, self.model.patchify(x), mask)

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=step_name == "train",
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )
        return loss

    def training_step(self, batch: tp.Any, batch_idx: int) -> torch.Tensor:
        return self._run_step(batch, step_name="train")

    def validation_step(self, batch: tp.Any, batch_idx: int) -> torch.Tensor:
        return self._run_step(batch, step_name="val")

    def configure_optimizers(self) -> tp.Any:
        return self.optim_config.build(self.parameters())
