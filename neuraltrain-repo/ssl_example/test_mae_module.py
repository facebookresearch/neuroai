# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import typing as tp
from pathlib import Path

import lightning.pytorch as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neuraltrain.models.common import INVALID_POS_VALUE, FourierEmb
from neuraltrain.models.mae import MaeEncoder
from neuraltrain.models.transformer import TransformerEncoder
from neuraltrain.optimizers.base import LightningOptimizer

from .mae_module import MaeModule, geometric_mask

N_CHANNELS, N_TIMES, PATCH_SIZE = 4, 200, 20
FREQUENCY, N_PATCHES = 100.0, N_TIMES // PATCH_SIZE
# seeded: convergence below depends on the positions, `seed_everything` comes too late
POSITIONS = torch.rand(N_CHANNELS, 3, generator=torch.Generator().manual_seed(0))
# positions are drawn in [0, 1), where 0.5 plays the part 9 cm plays on a head
MASK_RADIUS = 0.5


@dataclasses.dataclass
class _Batch:
    """Minimal stand-in for a ``neuralset`` ``Batch``."""

    data: dict[str, torch.Tensor]


class _Windows(Dataset):
    """Unlabelled sinusoidal windows, i.e. what a strided segmenter would emit."""

    def __init__(self, n_windows: int) -> None:
        cycle = torch.linspace(0, 1, N_TIMES)
        phases = torch.rand(n_windows, N_CHANNELS, 1)
        self.windows = torch.sin(2 * torch.pi * (cycle + phases))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.windows[idx]


def _collate(windows: list[torch.Tensor]) -> _Batch:
    return _Batch(
        data={
            "input": torch.stack(windows),
            "channel_positions": POSITIONS.expand(len(windows), -1, -1),
        }
    )


def _build_module() -> MaeModule:
    config = MaeEncoder(
        dim=32,
        patch_size=PATCH_SIZE,
        channel_emb_config=FourierEmb(n_freqs=2, n_dims=3),
        transformer_config=TransformerEncoder(
            heads=2, depth=1, rotary_pos_emb=False, attn_dropout=0.0
        ),
    )
    return MaeModule(
        model=config.build(n_outputs=None),
        loss=nn.MSELoss(),
        optim_config=LightningOptimizer(optimizer={"name": "Adam", "lr": 3e-3}),  # type: ignore
        frequency=FREQUENCY,
        mask_ratio=0.5,
        mask_radius=MASK_RADIUS,
        mask_duration=0.4,  # two of the ten patches a window holds
    )


def test_geometric_mask_hides_a_cap_over_a_window() -> None:
    # five channels in a row, `radius` reaching the immediate neighbours only
    positions = torch.zeros(1, 5, 3)
    positions[0, :, 0] = torch.arange(5) * 0.25
    # one block already covers `mask_ratio`, so the mask is that block alone
    mask = geometric_mask(
        positions, n_patches=10, n_hidden_patches=5, radius=0.25, mask_ratio=0.25
    )

    hidden = mask.reshape(5, 10)
    channels = hidden.any(dim=1).nonzero().flatten()
    patches = hidden.any(dim=0).nonzero().flatten()
    assert int(hidden.sum()) == len(channels) * len(patches), "not one block"
    assert 2 <= len(channels) <= 3 and (channels.diff() == 1).all(), "cap is not a cap"
    assert len(patches) == 5 and (patches.diff() == 1).all(), "window is not contiguous"


# the block count assumes blocks overlap by chance alone, which nearby caps beat,
# so the ratio is approached rather than hit
@pytest.mark.parametrize("mask_ratio", [0.25, 0.5, 0.75])
def test_geometric_mask_reaches_its_ratio(mask_ratio) -> None:
    positions = POSITIONS.expand(64, -1, -1).clone()
    positions[:32, 3] = INVALID_POS_VALUE  # half the batch is missing a channel
    present = (positions != INVALID_POS_VALUE).any(dim=-1)
    present = present[:, :, None].expand(-1, -1, N_PATCHES).flatten(1)

    torch.manual_seed(0)
    masks = torch.stack(
        [
            geometric_mask(positions, N_PATCHES, 2, MASK_RADIUS, mask_ratio)
            for _ in range(20)
        ]
    )

    assert not (masks & ~present).any(), "an absent channel's token was hidden"
    ratio = float(masks.sum() / present.sum() / len(masks))
    assert abs(ratio - mask_ratio) < 0.1, f"hid {ratio:.2f} of the tokens"


def test_masking_needs_a_channel_position() -> None:
    positions = torch.full((2, N_CHANNELS, 3), INVALID_POS_VALUE)
    with pytest.raises(ValueError, match="no channel with a valid position"):
        geometric_mask(positions, N_PATCHES, 2, MASK_RADIUS, 0.5)


def test_rejects_degenerate_mask_ratio() -> None:
    with pytest.raises(ValueError, match=r"mask_ratio must lie in \(0, 1\)"):
        MaeModule(
            model=_build_module().model,
            loss=nn.MSELoss(),
            optim_config=LightningOptimizer(optimizer={"name": "Adam", "lr": 1e-3}),  # type: ignore
            frequency=FREQUENCY,
            mask_ratio=0.0,
        )


def test_step_ignores_absent_channels() -> None:
    pl.seed_everything(0)
    module = _build_module()
    positions = POSITIONS.expand(2, -1, -1).clone()
    positions[:, 3] = INVALID_POS_VALUE

    x = torch.randn(2, N_CHANNELS, N_TIMES)
    other = x.clone()
    other[:, 3] = torch.randn(2, N_TIMES)

    losses = []
    for data in (x, other):
        torch.manual_seed(0)  # same mask draw for both
        batch = _Batch(data={"input": data, "channel_positions": positions})
        losses.append(float(module._run_step(batch, "val")))

    assert losses[0] == pytest.approx(losses[1], abs=1e-6), (
        "loss changed when an absent channel's padding did"
    )


def test_pretraining_needs_no_target_and_checkpoints_the_encoder(
    tmp_path: Path,
) -> None:
    pl.seed_everything(0)
    module = _build_module()
    losses: list[float] = []

    class _Record(pl.Callback):
        def on_train_batch_end(
            self, trainer: tp.Any, pl_module: tp.Any, outputs: tp.Any, *args: tp.Any
        ) -> None:
            losses.append(float(outputs["loss"]))

    trainer = pl.Trainer(
        max_epochs=40,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        callbacks=[_Record()],
    )
    loader = DataLoader(
        _Windows(32),
        batch_size=8,
        shuffle=True,
        collate_fn=_collate,  # type: ignore[arg-type]
    )
    trainer.fit(module, train_dataloaders=loader)
    trainer.save_checkpoint(tmp_path / "last.ckpt")

    start, end = sum(losses[:2]) / 2, sum(losses[-2:]) / 2
    assert end < 0.35 * start, f"loss did not decrease: {start:.4f} -> {end:.4f}"

    # neuralbench strips the "model." prefix, then matches a freshly built encoder
    saved = torch.load(tmp_path / "last.ckpt", weights_only=True)["state_dict"]
    saved = {k[len("model.") :]: v for k, v in saved.items() if k.startswith("model.")}
    encoder = _build_module().model.state_dict()
    assert {k: v.shape for k, v in encoder.items()} == {
        k: saved[k].shape for k in encoder if k in saved
    }, f"encoder keys missing from checkpoint: {sorted(set(encoder) - set(saved))}"
