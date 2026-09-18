"""Multi-GPU inference helpers for the NeuralBench NeuroLM backend.

This follows the official NeuroLM launch model (one process per GPU), but
uses an evaluation sampler that does not pad or duplicate samples.  The
model is replicated on each GPU; this is data-parallel inference, not model
parallelism.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

from .runner import NeuroLMPrediction, predict
from .task_spec import NeuroLMTaskSpec


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation indices across ranks without padding duplicates."""

    def __init__(self, dataset: Dataset[Any], rank: int, world_size: int) -> None:
        self.dataset_size = len(dataset)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.dataset_size:
            return 0
        return (self.dataset_size - self.rank + self.world_size - 1) // self.world_size


def distributed_context() -> tuple[bool, int, int, torch.device]:
    """Initialize NCCL from torchrun environment and return rank metadata."""
    rank_value = os.environ.get("RANK")
    if rank_value is None:
        return False, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("NeuroLM multi-GPU execution requires CUDA.")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, torch.device(f"cuda:{local_rank}")


def shard_dataloader(loader: DataLoader[Any], rank: int, world_size: int) -> DataLoader[Any]:
    """Rebuild a NeuralBench loader with a non-padding distributed sampler."""
    kwargs: dict[str, Any] = {
        "dataset": loader.dataset,
        "batch_size": loader.batch_size,
        "sampler": DistributedEvalSampler(loader.dataset, rank, world_size),
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": False,
    }
    if loader.num_workers > 0:
        kwargs["persistent_workers"] = loader.persistent_workers
        if loader.prefetch_factor is not None:
            kwargs["prefetch_factor"] = loader.prefetch_factor
    return DataLoader(**kwargs)


def distributed_predict(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    task_spec: NeuroLMTaskSpec,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
) -> list[NeuroLMPrediction]:
    """Predict on one rank and gather the complete result list on rank zero."""
    local_loader = shard_dataloader(loader, rank, world_size)
    local_results = predict(model, local_loader, task_spec, device=device)
    gathered: list[list[NeuroLMPrediction] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    if rank != 0:
        return []
    return [row for shard in gathered for row in (shard or [])]


def close_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
