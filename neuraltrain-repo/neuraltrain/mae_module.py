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

from .models.mae import MaeEncoderModel
from .optimizers import LightningOptimizer


def random_mask(
    batch_size: int, n_tokens: int, mask_ratio: float, device: torch.device
) -> torch.Tensor:
    """Choose which tokens to hide from each example.

    Parameters
    ----------
    batch_size, n_tokens :
        Shape of the token sequence to mask.
    mask_ratio :
        Fraction of the ``n_tokens`` to hide.  At least one token is hidden and
        at least one is left visible, whatever the ratio rounds to.
    device :
        Device to build the mask on.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape ``(batch_size, n_tokens)``, ``True`` on hidden
        positions.
    """
    if n_tokens < 2:
        raise ValueError(
            f"masking needs at least 2 tokens to leave one of each kind, "
            f"got {n_tokens}: shorten patch_size or lengthen the input window."
        )
    n_hidden = min(max(1, round(n_tokens * mask_ratio)), n_tokens - 1)
    # Ranking random scores hides exactly `n_hidden` tokens per example, which
    # keeps the loss comparable across the batch.
    ranks = torch.rand(batch_size, n_tokens, device=device).argsort(dim=1)
    return ranks < n_hidden


class MaeModule(pl.LightningModule):
    """Pretrain a :class:`~neuraltrain.models.mae.MaeEncoderModel` by masked prediction.

    The input is its own target, so batches need no ``"target"`` key: this
    trains on the unlabelled sliding windows of a ``neuralset`` segmenter
    configured with ``stride``.

    Hidden patches are replaced by a learned mask token and run through the
    encoder along with the visible ones, so the encoder sees full-length
    sequences here exactly as it will downstream, and a single linear layer
    reads the reconstruction off its output.  The original MAE instead encodes
    only the visible patches and restores the rest with a transformer decoder,
    which is cheaper per step and a natural thing to try.

    Only ``model`` outlives pretraining; the mask token and the linear
    reconstruction layer are scaffolding, which is why they live here rather
    than on the encoder config.

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
        self.reconstruct = nn.Linear(model.dim, model.patch_dim)

    def _run_step(self, batch: tp.Any, step_name: str) -> torch.Tensor:
        x = batch.data[self.x_name]
        merged = self.model.merge(x, batch.data[self.channel_positions_name])
        tokens = self.model.patch_tokens(merged)
        mask = random_mask(
            tokens.shape[0], tokens.shape[1], self.mask_ratio, tokens.device
        )
        # Substituting before positions are added leaves a hidden token its
        # place in the sequence and takes only its content.
        tokens = torch.where(mask[..., None], self.mask_token.to(tokens), tokens)
        encoded = self.model.encoder(self.model.add_positions(tokens))
        # The target is detached so that the merger is trained to feed the
        # encoder, never to make its own output easier to predict.
        loss = self.loss(
            self.reconstruct(encoded),
            self.model.patchify(merged).detach(),
            mask.to(tokens),
        )

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=step_name == "train",
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=x.shape[0],
            # Epoch metrics drive early stopping and checkpoint selection, so
            # every rank must agree on them under DDP.
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: tp.Any, batch_idx: int) -> torch.Tensor:
        return self._run_step(batch, step_name="train")

    def validation_step(self, batch: tp.Any, batch_idx: int) -> torch.Tensor:
        return self._run_step(batch, step_name="val")

    def configure_optimizers(self) -> tp.Any:
        # Schedules like OneCycleLR need the total number of steps, which only
        # the trainer can know; the ones that do not reject the argument.
        try:
            return self.optim_config.build(
                self.parameters(), total_steps=self.trainer.estimated_stepping_batches
            )
        except TypeError:
            return self.optim_config.build(self.parameters())
