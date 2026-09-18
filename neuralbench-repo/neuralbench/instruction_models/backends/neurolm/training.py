"""Multi-task instruction tuning for NeuroLM on NeuralBench loaders."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import load_neurolm_checkpoint
from .distributed import distributed_predict
from .instruction import build_instruction_batch
from .multitask import (
    build_seven_task_loaders,
    iter_multitask_train_batches,
    select_task_configs,
)
from .presets import align_task_spec_to_dataset, builtin_task_spec
from .runner import predict


@dataclass
class NeuroLMTrainingConfig:
    source_root: Path
    checkpoint: Path
    output_dir: Path
    text_data_dir: Path
    device: torch.device
    rank: int = 0
    world_size: int = 1
    epochs: int = 5
    batch_size: int = 8
    text_batch_size: int = 16
    num_workers: int = 4
    learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    block_size: int = 1024
    seed: int = 1337
    log_interval: int = 10
    debug: bool = False
    force: bool = False
    no_text_loss: bool = False
    task_names: tuple[str, ...] = ()


def _rebuild_train_loader(
    loader: DataLoader[Any], *, rank: int, world_size: int
) -> DataLoader[Any]:
    if world_size == 1:
        return loader
    sampler = DistributedSampler(
        loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    kwargs: dict[str, Any] = {
        "dataset": loader.dataset,
        "batch_size": loader.batch_size,
        "sampler": sampler,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": True,
    }
    if loader.num_workers > 0:
        kwargs["persistent_workers"] = loader.persistent_workers
        if loader.prefetch_factor is not None:
            kwargs["prefetch_factor"] = loader.prefetch_factor
    return DataLoader(**kwargs)


class TextBatcher:
    """Sample GPT-2 token sequences from official NeuroLM text bins."""

    def __init__(
        self, root: Path, batch_size: int, block_size: int, device: torch.device
    ) -> None:
        self.root = root
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        for split in ("train", "val"):
            path = root / f"{split}.bin"
            if not path.is_file():
                raise FileNotFoundError(f"NeuroLM text token file not found: {path}")

    def get(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        import numpy as np

        data = np.memmap(self.root / f"{split}.bin", dtype=np.uint16, mode="r")
        if len(data) <= self.block_size:
            raise ValueError(f"{split}.bin is shorter than block_size={self.block_size}")
        indices = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy(
                    data[int(i) : int(i) + self.block_size].astype("int64")
                )
                for i in indices
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    data[int(i) + 1 : int(i) + 1 + self.block_size].astype("int64")
                )
                for i in indices
            ]
        )
        return x.to(self.device, non_blocking=True), y.to(
            self.device, non_blocking=True
        )


def _cosine_lr(
    step: int, total_steps: int, warmup_steps: int, maximum: float, minimum: float
) -> float:
    if warmup_steps and step < warmup_steps:
        return maximum * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def _metric_payload(rows: list[Any]) -> dict[str, float]:
    valid = [row for row in rows if row.prediction is not None]
    if not rows:
        return {"parse_failure_rate": 1.0}
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
    )

    truth = [row.target for row in rows]
    pred = [row.prediction if row.prediction is not None else -1 for row in rows]
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "f1_weighted": float(f1_score(truth, pred, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(truth, pred)),
        "parse_failure_rate": 1.0 - len(valid) / len(rows),
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    step: int,
    config: NeuroLMTrainingConfig,
    validation: dict[str, Any],
) -> None:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": {
            key: getattr(raw_model.GPT2.config, key)
            for key in ("n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size")
        },
        "epoch": epoch,
        "iter_num": step,
        "neuralbench_config": {
            **asdict(config),
            "source_root": str(config.source_root),
            "checkpoint": str(config.checkpoint),
            "output_dir": str(config.output_dir),
            "text_data_dir": str(config.text_data_dir),
            "device": str(config.device),
        },
        "validation": validation,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _evaluate(
    model: torch.nn.Module,
    loaders: dict[str, dict[str, Any]],
    *,
    task_configs: tuple[Any, ...],
    rank: int,
    world_size: int,
    device: torch.device,
    debug: bool,
) -> dict[str, Any]:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model.eval()
    output: dict[str, Any] = {}
    for task in task_configs:
        loader = loaders[task.name]["val"]
        spec = align_task_spec_to_dataset(
            builtin_task_spec(task.neuralbench_task, task.dataset),
            loader.dataset,
            task.neuralbench_task,
        )
        if debug:
            import itertools

            rows = predict(
                raw_model,
                itertools.islice(loader, 2),
                spec,
                channel_names=list(
                    loader.dataset.extractors["neuro"]._channels.keys()
                ),
                device=device,
            )
        elif world_size > 1:
            rows = distributed_predict(
                raw_model,
                loader,
                spec,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        else:
            rows = predict(raw_model, loader, spec, device=device)
        if rank == 0:
            output[task.name] = _metric_payload(rows)
        if world_size > 1:
            dist.barrier()
    raw_model.train()
    return output


def train_multitask(config: NeuroLMTrainingConfig) -> dict[str, Any]:
    """Run official-style EEG+text instruction tuning on seven tasks."""
    torch.manual_seed(config.seed + config.rank)
    random.seed(config.seed + config.rank)
    if config.device.type == "cuda":
        torch.cuda.set_device(config.device)

    task_configs = select_task_configs(config.task_names)
    loaders = build_seven_task_loaders(
        batch_size=config.batch_size,
        num_workers=0 if config.debug else config.num_workers,
        debug=config.debug,
        task_names=config.task_names,
    )
    for parts in loaders.values():
        parts["train"] = _rebuild_train_loader(
            parts["train"], rank=config.rank, world_size=config.world_size
        )

    specs = {
        task.name: align_task_spec_to_dataset(
            builtin_task_spec(task.neuralbench_task, task.dataset),
            loaders[task.name]["train"].dataset,
            task.neuralbench_task,
        )
        for task in task_configs
    }
    model = load_neurolm_checkpoint(
        config.checkpoint, config.source_root, device=config.device
    )
    raw_model = model
    optimizer = model.configure_optimizers(
        config.weight_decay,
        config.learning_rate,
        (config.beta1, config.beta2),
        config.device.type,
    )
    run_name = "neurolm-7task" if not config.task_names else "neurolm-" + "-".join(config.task_names)
    checkpoint_path = config.output_dir / "checkpoints" / run_name / "ckpt.pt"
    start_epoch = 0
    global_step = 0
    if checkpoint_path.is_file() and not config.force:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = int(saved["epoch"]) + 1
        global_step = int(saved.get("iter_num", 0))
        if config.rank == 0:
            print(f"Resuming NeuralBench NeuroLM from {checkpoint_path}")
    elif checkpoint_path.is_file() and config.force and config.rank == 0:
        backup = checkpoint_path.with_name(
            f"ckpt.before-force-{time.strftime('%Y%m%d-%H%M%S')}.pt"
        )
        checkpoint_path.replace(backup)
        print(f"Preserved previous checkpoint as {backup}")

    if config.world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[config.device.index], output_device=config.device.index
        )

    text_batcher = None if config.no_text_loss else TextBatcher(
        config.text_data_dir,
        config.text_batch_size,
        config.block_size,
        config.device,
    )
    batches_per_epoch = sum(len(parts["train"]) for parts in loaders.values())
    if config.debug:
        batches_per_epoch = min(batches_per_epoch, len(loaders) * 2)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / config.gradient_accumulation_steps
    )
    total_steps = max(1, optimizer_steps_per_epoch * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    amp_dtype = torch.bfloat16
    if config.device.type == "cuda" and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler(
        config.device.type, enabled=config.device.type == "cuda" and amp_dtype == torch.float16
    )
    autocast = (
        lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
        if config.device.type == "cuda"
        else contextlib.nullcontext()
    )
    optimizer.zero_grad(set_to_none=True)
    last_validation: dict[str, Any] = {}

    for epoch in range(start_epoch, config.epochs):
        for parts in loaders.values():
            sampler = getattr(parts["train"], "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
        model.train()
        micro_step = 0
        running = 0.0
        iterator = iter_multitask_train_batches(loaders, seed=config.seed + epoch)
        for task_name, batch in iterator:
            if config.debug and micro_step >= len(loaders) * 2:
                break
            instruction = build_instruction_batch(
                batch,
                loaders[task_name]["train"],
                specs[task_name],
                device=config.device,
                eeg_ignore_index=-1 - raw_model.GPT2.config.vocab_size,
            )
            should_step = (micro_step + 1) % config.gradient_accumulation_steps == 0
            # Synchronize every DDP micro-batch. This is slightly less
            # optimized than no_sync(), but remains correct when task loaders
            # have different lengths and the final accumulation is partial.
            with autocast():
                instruction_loss, _, _ = model(**instruction.as_model_kwargs())
                text_loss = torch.zeros((), device=config.device)
                if text_batcher is not None:
                    text_x, text_y = text_batcher.get("train")
                    text_loss, _, _ = model(None, None, text_x, text_y)
                loss = (
                    instruction_loss + text_loss
                ) / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running += float(loss.detach()) * config.gradient_accumulation_steps
            micro_step += 1
            if should_step:
                lr = _cosine_lr(
                    global_step, total_steps, warmup_steps,
                    config.learning_rate, config.min_learning_rate
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                if config.grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                interval = 1 if config.debug else max(1, config.log_interval)
                if config.rank == 0 and global_step % interval == 0:
                    print(
                        f"epoch={epoch + 1}/{config.epochs} step={global_step} "
                        f"task={task_name} loss={running / micro_step:.4f} lr={lr:.3e}",
                        flush=True,
                    )
        if micro_step % config.gradient_accumulation_steps:
            if config.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if config.world_size > 1:
            dist.barrier()
        last_validation = _evaluate(
            model,
            loaders,
            task_configs=task_configs,
            rank=config.rank,
            world_size=config.world_size,
            device=config.device,
            debug=config.debug,
        )
        if config.rank == 0:
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch=epoch,
                step=global_step,
                config=config,
                validation=last_validation,
            )
            metrics_path = config.output_dir / "neurolm-7task-validation.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(last_validation, indent=2))
            print(f"Saved checkpoint to {checkpoint_path}", flush=True)
        if config.world_size > 1:
            dist.barrier()

    return last_validation if config.rank == 0 else {}
