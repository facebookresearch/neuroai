# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Lightning module for self-supervised masked-prediction pretraining."""

import math
import typing as tp

import lightning.pytorch as pl
import torch
from torch import nn

from neuraltrain.models.common import INVALID_POS_VALUE
from neuraltrain.models.mae import MaeEncoderModel
from neuraltrain.optimizers import LightningOptimizer


def geometric_mask(
    channel_positions: torch.Tensor,
    n_patches: int,
    n_hidden_patches: int,
    radius: float,
    mask_ratio: float,
) -> torch.Tensor:
    """Hide caps of *radius* on the scalp over *n_hidden_patches* consecutive patches.

    Returns boolean ``(B, C * n_patches)`` flags, ``True`` on hidden positions,
    in the token order of :meth:`MaeEncoderModel.valid_tokens`.  Blocks are
    centred on a channel drawn at random and so overlap: their count comes from
    ``1 - (1 - f) ** n_blocks = mask_ratio``, ``f`` being the share of tokens one
    block hides.  Channels a recording lacks neither centre a block nor join one.

    *radius* is in the unit of *channel_positions*, i.e. metres for the head
    frame ``ns.extractors.ChannelPositions`` returns with ``normalize=False``.
    """
    device = channel_positions.device
    present = (channel_positions != INVALID_POS_VALUE).any(dim=-1)  # (B, C)
    if not present.any(dim=1).all():
        raise ValueError(
            "an example has no channel with a valid position: check that the "
            "montage of `channel_positions` names the channels of every study."
        )
    n_hidden_patches = min(n_hidden_patches, n_patches)

    caps = torch.cdist(channel_positions, channel_positions) <= radius
    caps &= present[:, None, :] & present[:, :, None]  # (B, C centres, C)
    # what one block hides on average, over the channels it may be centred on
    covered = caps.sum(dim=(1, 2)) / present.sum(dim=1) ** 2
    f = covered * n_hidden_patches / n_patches
    n_blocks = (math.log1p(-mask_ratio) / torch.log1p(-f)).ceil().long().clamp(min=1)

    n_drawn = int(n_blocks.max())
    centres = torch.multinomial(present.float(), n_drawn, replacement=True)
    channels = caps.gather(1, centres[..., None].expand(-1, -1, caps.shape[-1]))
    # examples needing fewer blocks than the batch's worst case drop the extra ones
    channels &= (torch.arange(n_drawn, device=device) < n_blocks[:, None])[..., None]

    starts = torch.randint(n_patches - n_hidden_patches + 1, centres.shape, device=device)
    windows = starts[..., None] + torch.arange(n_hidden_patches, device=device)
    times = torch.zeros(centres.shape + (n_patches,), dtype=torch.bool, device=device)
    times.scatter_(2, windows, True)

    return (channels[..., None] & times[:, :, None, :]).any(dim=1).flatten(1)


class MaeModule(pl.LightningModule):
    """Pretrain a :class:`~neuraltrain.models.mae.MaeEncoderModel` by masked prediction.

    The input is its own target, so batches need no ``"target"`` key: this
    trains on the unlabelled windows a ``neuralset`` segmenter cuts when
    configured with ``stride``.  Hidden patches are replaced by a learned mask
    token, encoded along with the visible ones, and read back by a single
    linear layer.  Only ``model`` outlives pretraining; the mask token and that
    layer are scaffolding.

    What is hidden is a cap of scalp over a window of time rather than tokens
    scattered over the montage, so a hidden patch has no visible neighbour to
    interpolate from and reconstructing it asks for more than local smoothness.

    Parameters
    ----------
    model :
        Encoder to pretrain, built with ``n_outputs=None``.
    loss :
        Reconstruction loss over the hidden patches, called as
        ``loss(estimate, target)`` on ``(n_hidden, patch_size)`` tensors.
    optim_config :
        Optimizer configuration.
    frequency :
        Sampling rate of the input, in Hz: what turns ``mask_duration`` into a
        number of time patches.
    mask_ratio :
        Fraction of each example's channel-time patches hidden from the encoder.
    mask_radius :
        Radius of the hidden scalp caps, in the unit of the channel positions.
    mask_duration :
        Duration of the hidden time windows, in seconds.
    x_name, channel_positions_name :
        Batch keys holding the neuro input and its channel positions.
    """

    def __init__(
        self,
        model: MaeEncoderModel,
        loss: nn.Module,
        optim_config: LightningOptimizer,
        frequency: float,
        mask_ratio: float = 0.5,
        mask_radius: float = 0.09,
        mask_duration: float = 2.0,
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
        self.mask_radius = mask_radius
        self.n_hidden_patches = max(
            1, round(mask_duration * frequency / model.patch_size)
        )
        self.x_name = x_name
        self.channel_positions_name = channel_positions_name

        self.mask_token = nn.Parameter(torch.zeros(1, 1, model.dim))
        self.reconstruct = nn.Linear(model.dim, model.patch_size)

    def _run_step(self, batch: tp.Any, step_name: str) -> torch.Tensor:
        x = batch.data[self.x_name]
        channel_positions = batch.data[self.channel_positions_name]

        patches = self.model.patchify(x)
        n_patches = patches.shape[2]
        valid = self.model.valid_tokens(channel_positions, n_patches)
        hidden = geometric_mask(
            channel_positions,
            n_patches,
            n_hidden_patches=self.n_hidden_patches,
            radius=self.mask_radius,
            mask_ratio=self.mask_ratio,
        )

        tokens = self.model.patch_tokens(patches)
        # substitute before positions: a hidden token keeps its place, loses content
        tokens = torch.where(hidden[..., None], self.mask_token.to(tokens), tokens)
        tokens = self.model.add_positions(tokens, channel_positions, n_patches)
        # drops absent channels only; hidden tokens stay, predicting them is the task
        encoded = self.model.encoder(tokens, mask=valid)
        # scoring the hidden patches alone is what stops the model from copying
        loss = self.loss(self.reconstruct(encoded)[hidden], patches.flatten(1, 2)[hidden])

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
