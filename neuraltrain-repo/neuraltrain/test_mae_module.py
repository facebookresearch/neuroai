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
from torch.utils.data import DataLoader, Dataset

from .losses.losses import MaskedReconstructionLoss
from .mae_module import MaeModule, random_masking
from .models.mae import MaeEncoder
from .models.transformer import TransformerEncoder
from .optimizers.base import LightningOptimizer

N_CHANNELS, N_TIMES, PATCH_SIZE = 4, 200, 20


@dataclasses.dataclass
class _Batch:
    """Minimal stand-in for a ``neuralset`` ``Batch`` (not a neuraltrain dependency)."""

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
    return _Batch(data={"input": torch.stack(windows)})


def _build_module() -> MaeModule:
    config = MaeEncoder(
        dim=32,
        patch_size=PATCH_SIZE,
        transformer_config=TransformerEncoder(
            heads=2, depth=1, rotary_pos_emb=False, attn_dropout=0.0
        ),
    )
    return MaeModule(
        model=config.build(n_spatial_locations=N_CHANNELS, n_outputs=None),
        loss=MaskedReconstructionLoss(),
        optim_config=LightningOptimizer(optimizer={"name": "Adam", "lr": 3e-3}),  # type: ignore
        mask_ratio=0.5,
        decoder_config=TransformerEncoder(
            heads=2, depth=1, rotary_pos_emb=False, attn_dropout=0.0
        ),
    )


@pytest.mark.parametrize("mask_ratio", [0.1, 0.5, 0.9])
def test_random_masking(mask_ratio) -> None:
    tokens = torch.randn(3, 10, 8)
    kept, mask, restore = random_masking(tokens, mask_ratio)

    assert kept.shape == (3, 10 - int(mask[0].sum()), 8)
    assert (mask.sum(dim=1) == mask[0].sum()).all(), "mask count must be per-batch equal"
    assert 0 < mask[0].sum() < 10, "must leave at least one token of each kind"

    # `restore` must reorder [kept, dropped] back to the original positions.
    dropped = torch.zeros(3, int(mask[0].sum()), 8)
    reordered = torch.gather(
        torch.cat([kept, dropped], dim=1), 1, restore[:, :, None].expand(-1, -1, 8)
    )
    assert torch.equal(reordered[mask == 0], tokens[mask == 0])


def test_masking_needs_at_least_two_tokens() -> None:
    with pytest.raises(ValueError, match="at least 2 tokens"):
        random_masking(torch.randn(2, 1, 8), 0.5)


def test_rejects_degenerate_mask_ratio() -> None:
    with pytest.raises(ValueError, match=r"mask_ratio must lie in \(0, 1\)"):
        MaeModule(
            model=_build_module().model,
            loss=MaskedReconstructionLoss(),
            optim_config=LightningOptimizer(optimizer={"name": "Adam", "lr": 1e-3}),  # type: ignore
            mask_ratio=0.0,
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
    assert end < 0.25 * start, f"loss did not decrease: {start:.4f} -> {end:.4f}"

    # `neuralbench.utils.load_checkpoint` strips the LightningModule's "model."
    # prefix and then matches against a freshly built encoder, so the encoder's
    # every parameter must be in the checkpoint under exactly that name.
    saved = torch.load(tmp_path / "last.ckpt", weights_only=True)["state_dict"]
    saved = {k[len("model.") :]: v for k, v in saved.items() if k.startswith("model.")}
    encoder = _build_module().model.state_dict()
    assert {k: v.shape for k, v in encoder.items()} == {
        k: saved[k].shape for k in encoder if k in saved
    }, f"encoder keys missing from checkpoint: {sorted(set(encoder) - set(saved))}"
