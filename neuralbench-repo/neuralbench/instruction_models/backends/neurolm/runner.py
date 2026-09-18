"""Inference runner for the NeuroLM instruction/generation evaluation path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .adapter import NeuroLMInputAdapter, build_generation_inputs
from .task_spec import NeuroLMTaskSpec


@dataclass
class NeuroLMPrediction:
    """One generated answer and its parsed NeuralBench label."""

    text: str
    prediction: int | None
    target: int


def _target_labels(
    target: torch.Tensor, *, background_index: int | None = None
) -> torch.Tensor:
    """Convert common NeuralBench target layouts to integer class labels."""
    if target.ndim == 1:
        return target.long()
    if target.ndim == 2 and target.shape[1] == 1:
        return target[:, 0].long()
    if target.ndim == 2:
        labels = target.argmax(dim=1).long()
        if background_index is not None:
            missing = target.reshape(target.shape[0], -1).sum(dim=1) <= 0
            labels[missing] = background_index
        return labels
    raise ValueError(f"Unsupported target shape for NeuroLM: {tuple(target.shape)}")


def _channel_names_from_dataset(dataset: Any) -> list[str]:
    extractors = getattr(dataset, "extractors", None)
    neuro_extractor = None if extractors is None else extractors.get("neuro")
    names = getattr(neuro_extractor, "_channels", None)
    if names is None:
        raise ValueError(
            "NeuroLM requires EEG channel names. Pass channel_names explicitly "
            "or use a NeuralBench neuro extractor exposing _channels."
        )
    return list(names.keys())


def _move_batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    data = getattr(batch, "data", batch)
    if not isinstance(data, dict) or "neuro" not in data or "target" not in data:
        raise ValueError("Expected a NeuralBench Batch with neuro and target fields.")
    return data["neuro"].to(device), data["target"].to(device)


def predict(
    model: torch.nn.Module,
    dataloader: Iterable[Any],
    task_spec: NeuroLMTaskSpec,
    *,
    channel_names: list[str] | tuple[str, ...] | None = None,
    device: str | torch.device = "cuda",
    adapter: NeuroLMInputAdapter | None = None,
    tokenizer_name: str = "gpt2",
    top_k: int = 1,
) -> list[NeuroLMPrediction]:
    """Generate and parse NeuroLM answers for a NeuralBench dataloader.

    This function deliberately returns raw text as well as parsed labels so
    failed parses remain auditable instead of silently becoming a class.
    """
    import tiktoken

    device = torch.device(device)
    model = model.to(device).eval()
    adapter = adapter or NeuroLMInputAdapter()
    encoding = tiktoken.get_encoding(tokenizer_name)
    prompt = torch.tensor(
        [50257, *encoding.encode(task_spec.prompt)], dtype=torch.long, device=device
    )
    results: list[NeuroLMPrediction] = []

    for batch in dataloader:
        neuro, target = _move_batch(batch, device)
        names = list(channel_names) if channel_names is not None else _channel_names_from_dataset(
            dataloader.dataset
        )
        eeg, input_chans, input_time, eeg_mask, eeg_attention = adapter(neuro, names)
        inputs = build_generation_inputs(
            eeg,
            input_chans,
            input_time,
            eeg_mask,
            eeg_attention,
            prompt,
        )
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=task_spec.max_new_tokens,
                top_k=top_k,
            )
        target_labels = _target_labels(
            target, background_index=task_spec.background_label
        ).detach().cpu().tolist()
        generated = generated.detach().cpu()
        prompt_len = prompt.numel()
        for row, target_label in zip(generated, target_labels, strict=True):
            answer_text = encoding.decode(row[prompt_len:].tolist())
            results.append(
                NeuroLMPrediction(
                    text=answer_text,
                    prediction=task_spec.parse_answer(answer_text),
                    target=int(target_label),
                )
            )
    return results
