# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning module for self-supervised masked-prediction pretraining."""

import typing as tp

import lightning.pytorch as pl
import torch
from torch import nn

from neuraltrain.models.mae import MaeEncoderModel
from neuraltrain.optimizers import LightningOptimizer


def random_mask(valid: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """Pick a ``mask_ratio`` fraction of each example's ``valid`` tokens to hide.

    Returns boolean ``(B, N)`` flags, ``True`` on hidden positions.  Invalid
    tokens are never hidden, and every example keeps at least one token of each
    kind whatever the ratio rounds to.
    """
    n_valid = valid.sum(dim=1, keepdim=True)
    if int(n_valid.min()) < 2:
        raise ValueError(
            f"masking needs at least 2 valid tokens to leave one of each kind, "
            f"got {int(n_valid.min())}: shorten patch_size, lengthen the input "
            f"window, or check that channel positions are not all invalid."
        )
    # invalid rank last -> the fraction is taken from valid tokens only
    scores = torch.rand_like(valid, dtype=torch.float).masked_fill(~valid, torch.inf)
    ranks = scores.argsort(dim=1).argsort(dim=1)
    n_hidden = (n_valid * mask_ratio).round().long().clamp(min=1)
    return ranks < torch.minimum(n_hidden, n_valid - 1)


class MaeModule(pl.LightningModule):
    """Pretrain a :class:`~neuraltrain.models.mae.MaeEncoderModel` by masked prediction.

    The input is its own target, so batches need no ``"target"`` key: this
    trains on the unlabelled sliding windows of a ``neuralset`` segmenter
    configured with ``stride``.  Hidden patches are replaced by a learned mask
    token, encoded along with the visible ones, and read back by a single
    linear layer.  Only ``model`` outlives pretraining; the mask token and that
    layer are scaffolding.

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
        Fraction of each example's channel-time patches hidden from the encoder.
    x_name, channel_positions_name :
        Batch keys holding the neuro input and its channel positions.
    """

    def __init__(
        self,
        model: MaeEncoderModel,
        loss: nn.Module,
        optim_config: LightningOptimizer,
        mask_ratio: float = 0.5,
        x_name: str = "input",
        channel_positions_name: str = "channel_positions",
    ) -> None:
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must lie in (0, 1), got {mask_ratio}.")
        self.model = model
        self.loss = loss
        self.optim_config = optim_config
        self.mask_ratio = mask_ratio
        self.x_name = x_name
        self.channel_positions_name = channel_positions_name

        self.mask_token = nn.Parameter(torch.zeros(1, 1, model.dim))
        self.reconstruct = nn.Linear(model.dim, model.patch_size)

    def _run_step(self, batch: tp.Any, step_name: str) -> torch.Tensor:
        x = batch.data[self.x_name]
        channel_positions = batch.data[self.channel_positions_name]

        patches = self.model.patchify(x)
        valid = self.model.valid_tokens(channel_positions, patches.shape[2])
        hidden = random_mask(valid, self.mask_ratio)

        tokens = self.model.patch_tokens(x)
        # substitute before positions: a hidden token keeps its place, loses content
        tokens = torch.where(hidden[..., None], self.mask_token.to(tokens), tokens)
        tokens = self.model.add_positions(tokens, channel_positions)
        # drops absent channels only; hidden tokens stay, predicting them is the task
        encoded = self.model.encoder(tokens, mask=valid)
        loss = self.loss(
            self.reconstruct(encoded), patches.flatten(1, 2), hidden.to(tokens)
        )

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=step_name == "train",
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=x.shape[0],
            sync_dist=True,  # epoch metrics gate checkpointing: ranks must agree
        )
        return loss

    def training_step(self, batch: tp.Any, batch_idx: int) -> torch.Tensor:
        return self._run_step(batch, step_name="train")

    def validation_step(self, batch: tp.Any, batch_idx: int) -> torch.Tensor:
        return self._run_step(batch, step_name="val")

    def configure_optimizers(self) -> tp.Any:
        # OneCycleLR and friends need the trainer's step count; others reject it
        try:
            return self.optim_config.build(
                self.parameters(), total_steps=self.trainer.estimated_stepping_batches
            )
        except TypeError:
            return self.optim_config.build(self.parameters())
