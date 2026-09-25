"""CLI and main-NeuralBench dispatch for NeuroLM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_neurolm_checkpoint
from .distributed import close_distributed, distributed_context, distributed_predict
from .presets import align_task_spec_to_dataset, builtin_task_spec
from .runner import predict
from .task_spec import NeuroLMTaskSpec


def _load_preprocessing_config() -> dict[str, object]:
    import yaml

    config_path = Path(__file__).resolve().parents[3] / "models" / "neurolm.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    data = config.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("neuro"), dict):
        raise ValueError(f"Invalid NeuroLM preprocessing config: {config_path}")
    return data


def _workspace_candidate() -> Path:
    return Path(__file__).resolve().parents[6] / "NeuroLM"


def resolve_paths(
    source_root: str | Path | None,
    checkpoint: str | Path | None,
    text_data_dir: str | Path | None,
) -> tuple[Path, Path, Path]:
    source = Path(
        source_root or os.environ.get("NEUROLM_SOURCE_ROOT") or _workspace_candidate()
    ).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(
            f"Official NeuroLM checkout not found at {source}. Pass "
            "--official-source-root or set NEUROLM_SOURCE_ROOT."
        )
    if checkpoint is None:
        checkpoint_value = os.environ.get("NEUROLM_CHECKPOINT")
        candidates = [
            Path(checkpoint_value) if checkpoint_value else None,
            source / "outputs" / "checkpoints" / "NeuroLM-B.pt",
            source / "checkpoints" / "NeuroLM-B.pt",
        ]
        checkpoint_path = next(
            (candidate for candidate in candidates if candidate and candidate.is_file()),
            candidates[1],
        )
    else:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Pretrained NeuroLM checkpoint not found: {checkpoint_path}. Pass "
            "--checkpoint or set NEUROLM_CHECKPOINT."
        )
    text_root = Path(
        text_data_dir
        or os.environ.get("NEUROLM_TEXT_DATA_DIR")
        or source / "outputs" / "text"
    ).expanduser().resolve()
    return source, checkpoint_path, text_root


def _metrics(results: list[Any]) -> dict[str, float]:
    if not results:
        return {}
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
    )

    y_pred = [
        row.prediction if row.prediction is not None else -1 for row in results
    ]
    y_true = [row.target for row in results]
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_score_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def _launch_multi_gpu(argv: list[str], gpus: str) -> int:
    selected = [item.strip() for item in gpus.split(",") if item.strip()]
    if len(selected) < 2:
        return -1
    child_argv: list[str] = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item == "--gpus":
            skip_next = True
            continue
        if item.startswith("--gpus="):
            continue
        child_argv.append(item)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(selected)}",
            "-m",
            "neuralbench.instruction_models.backends.neurolm.cli",
            "--gpus",
            "__distributed__",
            *child_argv,
        ],
        env=env,
    )


def _prepare_loaders(
    task: str,
    dataset: str | None,
    *,
    batch_size: int,
    workers: int,
    multi_task: bool,
    debug: bool,
) -> dict[str, Any]:
    if multi_task:
        from .multitask import build_seven_task_loaders

        return build_seven_task_loaders(
            batch_size=batch_size, num_workers=workers, debug=debug
        )
    from neuralbench import get_default_dataloaders
    from .multitask import task_neuro_overrides

    preprocessing = _load_preprocessing_config()
    overrides = {
        f"neuro.{key}": value for key, value in preprocessing["neuro"].items()
    }
    overrides.update(task_neuro_overrides(task, dataset))
    return get_default_dataloaders(
        "eeg",
        task,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=workers,
        **overrides,
    )


def _batch_contract_summary(
    *,
    name: str,
    task: str,
    dataset: str | None,
    parts: dict[str, Any],
) -> dict[str, Any]:
    """Summarize one train batch before an expensive instruction-tuning run.

    This deliberately reports the data *after* NeuralBench preprocessing and
    before NeuroLM tokenization.  It makes channel order, amplitude scale,
    label-to-answer order, and padding visible without loading a checkpoint.
    """
    from .instruction import build_instruction_batch
    from .runner import _channel_names_from_dataset

    spec = align_task_spec_to_dataset(
        builtin_task_spec(task, dataset), parts["train"].dataset, task
    )
    batch = next(iter(parts["train"]))
    data = getattr(batch, "data", batch)
    neuro = data["neuro"].float()
    instruction = build_instruction_batch(
        batch, parts["train"], spec, device="cpu"
    )
    targets = instruction.targets.cpu()
    return {
        "name": name,
        "splits": {split: len(loader.dataset) for split, loader in parts.items()},
        "channels": _channel_names_from_dataset(parts["train"].dataset),
        "raw_eeg_shape": list(neuro.shape),
        "raw_eeg_stats": {
            "min": float(neuro.min()),
            "max": float(neuro.max()),
            "mean": float(neuro.mean()),
            "std": float(neuro.std()),
        },
        "eeg_shape": list(instruction.x_eeg.shape),
        "valid_eeg_tokens": int(instruction.input_mask[0].sum()),
        "text_shape": list(instruction.x_text.shape),
        "batch_label_counts": torch.bincount(
            targets, minlength=len(spec.answers)
        ).tolist(),
        "answers_by_neuralbench_index": list(spec.answers),
        "background_label": spec.background_label,
        "prompt": spec.prompt,
    }


def _contract_summary(
    loaders: dict[str, Any], *, multi_task: bool, task: str | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Materialize batch contracts for either the seven-task suite or one task."""
    if not multi_task:
        if task is None:
            raise ValueError("Single-task contract summary requires task metadata.")
        return _batch_contract_summary(
            name=task, task=task, dataset=dataset, parts=loaders
        )

    from .multitask import BLPM_SEVEN_TASKS

    return {
        item.name: _batch_contract_summary(
            name=item.name,
            task=item.neuralbench_task,
            dataset=item.dataset,
            parts=loaders[item.name],
        )
        for item in BLPM_SEVEN_TASKS
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_tasks and not args.multi_task:
        raise ValueError("--train-tasks requires --multi-task.")
    if args.gpus and args.gpus != "__distributed__" and "RANK" not in os.environ:
        launch_argv = getattr(args, "_neurolm_argv", sys.argv[1:])
        exit_code = _launch_multi_gpu(launch_argv, args.gpus)
        if exit_code >= 0:
            if exit_code:
                raise RuntimeError(f"NeuroLM torchrun failed with exit code {exit_code}")
            return {}

    distributed, rank, world_size, detected_device = distributed_context()
    device = detected_device if distributed else __import__("torch").device(args.device)
    try:
        if args.prepare or args.download or args.audit:
            loaders = _prepare_loaders(
                args.task,
                args.dataset,
                batch_size=args.batch_size,
                workers=0 if args.debug else args.workers,
                multi_task=args.multi_task,
                debug=args.debug,
            )
            if rank == 0:
                if args.multi_task or args.audit:
                    summary = _contract_summary(
                        loaders,
                        multi_task=args.multi_task,
                        task=args.task,
                        dataset=args.dataset,
                    )
                else:
                    summary = {
                        split: len(loader.dataset) for split, loader in loaders.items()
                    }
                print(json.dumps({"prepared": summary, "audit": args.audit}, indent=2))
            return {"prepared": True, "audit": args.audit}

        source, checkpoint, text_root = resolve_paths(
            args.official_source_root, args.checkpoint, args.text_data_dir
        )
        if args.multi_task:
            from .training import NeuroLMTrainingConfig, train_multitask

            task_names = tuple(
                name.strip()
                for name in (args.train_tasks or "").split(",")
                if name.strip()
            )

            config = NeuroLMTrainingConfig(
                source_root=source,
                checkpoint=checkpoint,
                output_dir=Path(args.output_dir).expanduser().resolve(),
                text_data_dir=text_root,
                device=device,
                rank=rank,
                world_size=world_size,
                epochs=1 if args.debug else args.epochs,
                batch_size=args.batch_size,
                text_batch_size=args.text_batch_size,
                num_workers=args.workers,
                learning_rate=args.learning_rate,
                min_learning_rate=args.min_learning_rate,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_ratio=args.warmup_ratio,
                log_interval=args.log_interval,
                seed=args.seed,
                debug=args.debug,
                force=args.force,
                no_text_loss=args.no_text_loss,
                task_names=task_names,
            )
            return train_multitask(config)

        loaders = _prepare_loaders(
            args.task,
            args.dataset,
            batch_size=args.batch_size,
            workers=0 if args.debug else args.workers,
            multi_task=False,
            debug=args.debug,
        )
        if args.task_spec is not None:
            spec = NeuroLMTaskSpec.from_yaml(args.task_spec)
        else:
            spec = builtin_task_spec(args.task, args.dataset)
            spec = align_task_spec_to_dataset(
                spec, loaders[args.split].dataset, args.task
            )
        model = load_neurolm_checkpoint(checkpoint, source, device=device)
        if distributed:
            results = distributed_predict(
                model,
                loaders[args.split],
                spec,
                rank=rank,
                world_size=world_size,
                device=device,
            )
        else:
            results = predict(model, loaders[args.split], spec, device=device)
        if distributed and rank != 0:
            return {}
        valid = [row for row in results if row.prediction is not None]
        payload = {
            "task": spec.name,
            "split": args.split,
            "n_samples": len(results),
            "n_parsed": len(valid),
            "parse_failure_rate": 1.0
            - (len(valid) / len(results) if results else 0.0),
            "metrics": _metrics(results),
            "predictions": [row.prediction for row in results],
            "targets": [row.target for row in results],
            "generated_text": [row.text for row in results],
        }
        output = Path(args.output) if args.output else None
        if output is None:
            print(json.dumps(payload, indent=2))
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2))
        return payload
    finally:
        if distributed:
            close_distributed()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Instruction-tune or evaluate NeuroLM with NeuralBench data."
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--official-source-root")
    parser.add_argument("--text-data-dir")
    parser.add_argument("--task", default="pathology")
    parser.add_argument("--dataset")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--task-spec", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--multi-task", action="store_true")
    parser.add_argument(
        "--train-tasks",
        help=(
            "Comma-separated subset of the fixed NeuroLM suite for a control run, "
            "for example: hmc or hmc,tuab. Requires --multi-task."
        ),
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Print one pre-tokenization NeuroLM batch contract without loading a checkpoint.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--min-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-text-loss", action="store_true")
    return parser


def run_from_neuralbench(
    *,
    device: str,
    task: str | list[str],
    checkpoint: str | None,
    dataset: str | list[str] | None,
    debug: bool,
    force: bool,
    retry: bool,
    prepare: bool,
    download: bool,
    multi_task: bool,
    gpus: str | None,
    official_source_root: str | None,
    text_data_dir: str | None,
    output_dir: str,
    workers: int,
    batch_size: int,
    text_batch_size: int,
    epochs: int,
    gradient_accumulation_steps: int,
    log_interval: int,
    learning_rate: float,
    seed: int,
    no_text_loss: bool,
) -> list[dict[str, Any]]:
    """Translate the normal NeuralBench CLI into the NeuroLM runner."""
    del retry  # Existing checkpoint auto-resume is NeuroLM's retry behavior.
    if device != "eeg":
        raise ValueError("The NeuroLM backend currently supports EEG only.")
    tasks = [task] if isinstance(task, str) else list(task)
    if multi_task:
        if tasks != ["all"]:
            raise ValueError(
                "NeuroLM --multi-task denotes the fixed seven-task suite; use task 'all'."
            )
        selected_task = "all"
        selected_dataset = None
    else:
        if len(tasks) != 1 or tasks[0] in {"all", "all_multi_dataset"}:
            raise ValueError(
                "Single-task NeuroLM evaluation requires one explicit task, or "
                "use 'eeg all --model neurolm --multi-task'."
            )
        selected_task = tasks[0]
        selected_dataset = dataset if isinstance(dataset, str) else None
    namespace = argparse.Namespace(
        checkpoint=checkpoint,
        official_source_root=official_source_root,
        text_data_dir=text_data_dir,
        task=selected_task,
        dataset=selected_dataset,
        split="test",
        device="cuda",
        gpus=gpus,
        workers=workers,
        batch_size=batch_size,
        text_batch_size=text_batch_size,
        task_spec=None,
        output=None,
        output_dir=output_dir,
        multi_task=multi_task,
        train_tasks=None,
        prepare=prepare,
        download=download,
        audit=False,
        debug=debug,
        force=force,
        epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_interval=log_interval,
        learning_rate=learning_rate,
        min_learning_rate=5e-5,
        weight_decay=0.1,
        warmup_ratio=0.1,
        grad_clip=1.0,
        seed=seed,
        no_text_loss=no_text_loss,
    )
    launch_argv = [
        "--task", selected_task, "--device", "cuda",
        "--workers", str(workers), "--batch-size", str(batch_size),
        "--text-batch-size", str(text_batch_size),
        "--output-dir", output_dir, "--epochs", str(epochs),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--log-interval", str(log_interval),
        "--learning-rate", str(learning_rate), "--seed", str(seed),
    ]
    for flag, value in (
        ("--checkpoint", checkpoint),
        ("--dataset", selected_dataset),
        ("--official-source-root", official_source_root),
        ("--text-data-dir", text_data_dir),
    ):
        if value is not None:
            launch_argv.extend([flag, str(value)])
    for enabled, flag in (
        (multi_task, "--multi-task"), (prepare, "--prepare"),
        (download, "--download"), (debug, "--debug"), (force, "--force"),
        (no_text_loss, "--no-text-loss"),
    ):
        if enabled:
            launch_argv.append(flag)
    namespace._neurolm_argv = launch_argv
    payload = run(namespace)
    return [payload] if payload else []


def main() -> None:
    run(_parser().parse_args())


if __name__ == "__main__":
    main()
