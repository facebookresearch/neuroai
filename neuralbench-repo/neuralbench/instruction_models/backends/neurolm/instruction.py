"""Convert NeuralBench batches to the official NeuroLM instruction contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .adapter import STANDARD_1020, NeuroLMInputAdapter
from .runner import _channel_names_from_dataset, _target_labels
from .task_spec import NeuroLMTaskSpec


@dataclass
class NeuroLMInstructionBatch:
    """Inputs consumed by ``NeuroLM.forward`` plus labels for auditing."""

    x_eeg: torch.Tensor
    y_eeg: torch.Tensor
    x_text: torch.Tensor
    y_text: torch.Tensor
    input_chans: torch.Tensor
    input_time: torch.Tensor
    input_mask: torch.Tensor
    eeg_mask: torch.Tensor
    eeg_text_mask: torch.Tensor
    targets: torch.Tensor

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            "x_eeg": self.x_eeg,
            "y_eeg": self.y_eeg,
            "x_text": self.x_text,
            "y_text": self.y_text,
            "input_chans": self.input_chans,
            "input_time": self.input_time,
            "input_mask": self.input_mask,
            "eeg_mask": self.eeg_mask,
            "eeg_text_mask": self.eeg_text_mask,
        }


def _encoding(tokenizer_name: str):
    import tiktoken

    return tiktoken.get_encoding(tokenizer_name)


def _pad_rows(rows: list[list[int]], pad_value: int) -> torch.Tensor:
    width = max(len(row) for row in rows)
    result = torch.full((len(rows), width), pad_value, dtype=torch.long)
    for index, row in enumerate(rows):
        result[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return result


def build_instruction_batch(
    batch: Any,
    dataloader: Any,
    task_spec: NeuroLMTaskSpec,
    *,
    device: torch.device | str,
    adapter: NeuroLMInputAdapter | None = None,
    tokenizer_name: str = "gpt2",
    max_eeg_tokens: int = 320,
    max_text_tokens: int = 80,
    eeg_ignore_index: int = -50258,
) -> NeuroLMInstructionBatch:
    """Create padded EEG/text tensors for one NeuralBench batch.

    The generated training target is ``prompt + answer + EOS``.  Tokens before
    the answer are set to ``-1`` in ``y_text``, matching the official loaders'
    masked language-model objective.
    """
    device = torch.device(device)
    data = getattr(batch, "data", batch)
    neuro = data["neuro"]
    targets = _target_labels(
        data["target"], background_index=task_spec.background_label
    ).long()
    channel_names = _channel_names_from_dataset(dataloader.dataset)
    adapter = adapter or NeuroLMInputAdapter()
    eeg, input_chans, input_time, eeg_mask, _ = adapter(neuro, channel_names)
    if eeg.shape[1] > max_eeg_tokens:
        raise ValueError(
            f"NeuroLM input has {eeg.shape[1]} EEG tokens, exceeding "
            f"max_eeg_tokens={max_eeg_tokens}. Add model-specific channel picks."
        )

    batch_size, eeg_tokens, _ = eeg.shape
    padded_eeg = torch.zeros(
        batch_size, max_eeg_tokens, eeg.shape[-1], dtype=eeg.dtype, device=eeg.device
    )
    padded_eeg[:, :eeg_tokens] = eeg
    padded_chans = torch.full(
        (batch_size, max_eeg_tokens),
        STANDARD_1020.index("PAD"),
        dtype=input_chans.dtype,
        device=input_chans.device,
    )
    padded_chans[:, :eeg_tokens] = input_chans
    padded_time = torch.zeros(
        batch_size, max_eeg_tokens, dtype=input_time.dtype, device=input_time.device
    )
    padded_time[:, :eeg_tokens] = input_time
    padded_mask = torch.zeros(
        batch_size, max_eeg_tokens, dtype=torch.bool, device=eeg.device
    )
    padded_mask[:, :eeg_tokens] = eeg_mask

    encoding = _encoding(tokenizer_name)
    prompt_ids = [50257, *encoding.encode(task_spec.prompt)]
    text_rows = []
    target_rows = []
    for target in targets.detach().cpu().tolist():
        continuation = task_spec.answer_continuation(int(target))
        full_ids = [
            *prompt_ids,
            *encoding.encode(
                f"{continuation} <|endoftext|>",
                allowed_special={"<|endoftext|>"},
            ),
        ]
        if len(full_ids) > max_text_tokens:
            raise ValueError(
                f"Instruction for {task_spec.name!r} has {len(full_ids)} tokens, "
                f"exceeding max_text_tokens={max_text_tokens}."
            )
        text_rows.append(full_ids)
        y = [-1] * len(full_ids)
        # Match official NeuroLM: predict the answer/EOS after the prompt.
        for index in range(len(prompt_ids) - 1, len(full_ids) - 1):
            y[index] = full_ids[index + 1]
        target_rows.append(y)
    x_text = _pad_rows(text_rows, 50256)
    y_text = _pad_rows(target_rows, -1)
    if x_text.shape[1] < max_text_tokens:
        x_text = torch.nn.functional.pad(
            x_text, (0, max_text_tokens - x_text.shape[1]), value=50256
        )
        y_text = torch.nn.functional.pad(
            y_text, (0, max_text_tokens - y_text.shape[1]), value=-1
        )

    total = max_eeg_tokens + max_text_tokens
    eeg_text_mask = torch.tril(
        torch.ones(total, total, dtype=torch.bool, device=device)
    ).unsqueeze(0)
    # Enable same-time-channel attention for the actual EEG tokens. NeuralBench
    # batches are task-homogeneous, so the channel count is shared here.
    n_channels = len(channel_names)
    n_blocks = eeg_tokens // n_channels
    for block in range(n_blocks):
        start = block * n_channels
        stop = min(start + n_channels, eeg_tokens)
        eeg_text_mask[:, start:stop, start:stop] = True
    eeg_text_mask[:, :, eeg_tokens:max_eeg_tokens] = False
    eeg_text_mask = eeg_text_mask.unsqueeze(1).expand(
        batch_size, 1, total, total
    ).clone()

    return NeuroLMInstructionBatch(
        x_eeg=padded_eeg.to(device),
        y_eeg=torch.full(
            (batch_size, max_eeg_tokens),
            fill_value=eeg_ignore_index,
            dtype=torch.long,
            device=device,
        ),
        x_text=x_text.to(device),
        y_text=y_text.to(device),
        input_chans=padded_chans.to(device),
        input_time=padded_time.to(device),
        input_mask=padded_mask.to(device),
        eeg_mask=padded_mask.to(device),
        eeg_text_mask=eeg_text_mask.to(device),
        targets=targets.to(device),
    )
