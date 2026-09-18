"""Convert NeuralBench EEG batches to the official NeuroLM tensor contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch


# This is the positional vocabulary used by the official NeuroLM dataset
# helpers.  It is kept as a small, dependency-free adapter constant so the
# benchmark does not need to import the official repository at discovery time.
STANDARD_1020 = (
    "FP1 FPZ FP2 AF9 AF7 AF5 AF3 AF1 AFZ AF2 AF4 AF6 AF8 AF10 "
    "F9 F7 F5 F3 F1 FZ F2 F4 F6 F8 F10 FT9 FT7 FC5 FC3 FC1 FCZ FC2 FC4 FC6 FT8 FT10 "
    "T9 T7 C5 C3 C1 CZ C2 C4 C6 T8 T10 TP9 TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6 TP8 TP10 "
    "P9 P7 P5 P3 P1 PZ P2 P4 P6 P8 P10 PO9 PO7 PO5 PO3 PO1 POZ PO2 PO4 PO6 PO8 PO10 "
    "O1 OZ O2 O9 CB1 CB2 IZ O10 T3 T5 T4 T6 M1 M2 A1 A2 CFC1 CFC2 CFC3 CFC4 CFC5 CFC6 CFC7 CFC8 "
    "CCP1 CCP2 CCP3 CCP4 CCP5 CCP6 CCP7 CCP8 T1 T2 FTT9H TTP7H TPP9H FTT10H TPP8H TPP10H "
    "FP1-F7 F7-T7 T7-P7 P7-O1 FP2-F8 F8-T8 T8-P8 P8-O2 FP1-F3 F3-C3 C3-P3 P3-O1 "
    "FP2-F4 F4-C4 C4-P4 P4-O2 PAD I1 I2"
).split()


def _canonical_channel(name: str) -> str:
    """Normalize common NeuralBench/EEG naming suffixes."""
    value = name.upper().strip().replace("EEG ", "")
    for suffix in ("-REF", "-LE", "-AVG"):
        value = value.removesuffix(suffix)
    if "-" in value and value.split("-", 1)[1] in {"M1", "M2", "A1", "A2"}:
        value = value.split("-", 1)[0]
    return value


def _channel_indices(channel_names: list[str] | tuple[str, ...]) -> list[int]:
    indices = []
    for name in channel_names:
        canonical = _canonical_channel(name)
        try:
            indices.append(STANDARD_1020.index(canonical))
        except ValueError as exc:
            raise ValueError(
                f"Channel {name!r} is not in NeuroLM's standard_1020 vocabulary."
            ) from exc
    return indices


@dataclass(frozen=True)
class NeuroLMInputAdapter:
    """Prepare a NeuralBench EEG window for NeuroLM.

    NeuroLM consumes non-overlapping 200-sample chunks.  The adapter therefore
    requires a 200 Hz input window whose sample count is divisible by 200.
    ``amplitude_scale`` matches the scale used by the official downstream
    loaders, which divide their stored signal by 100 before inference.
    """

    sampling_rate: int = 200
    token_samples: int = 200
    amplitude_scale: float = 100.0

    def __call__(
        self,
        neuro: torch.Tensor,
        channel_names: list[str] | tuple[str, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if neuro.ndim != 3:
            raise ValueError(f"Expected neuro shape (B, C, T), got {tuple(neuro.shape)}")
        if self.sampling_rate != 200:
            raise ValueError("NeuroLM checkpoints require 200 Hz input.")
        batch_size, n_channels, n_samples = neuro.shape
        if n_channels != len(channel_names):
            raise ValueError("channel_names must have one entry per EEG channel.")
        if n_samples % self.token_samples:
            raise ValueError(
                f"Window has {n_samples} samples; it must be divisible by "
                f"{self.token_samples}."
            )

        n_time_blocks = n_samples // self.token_samples
        eeg = neuro.float() / self.amplitude_scale
        eeg = eeg.reshape(batch_size, n_channels, n_time_blocks, self.token_samples)
        eeg = eeg.permute(0, 2, 1, 3).reshape(
            batch_size, n_time_blocks * n_channels, self.token_samples
        )

        one_block_channels = torch.tensor(
            _channel_indices(channel_names), dtype=torch.long, device=neuro.device
        )
        input_chans = one_block_channels.repeat(n_time_blocks).expand(batch_size, -1)
        input_time = torch.arange(n_time_blocks, device=neuro.device).repeat_interleave(
            n_channels
        )
        input_time = input_time.expand(batch_size, -1)
        eeg_mask = torch.ones(
            batch_size, eeg.shape[1], dtype=torch.bool, device=neuro.device
        )
        return eeg, input_chans, input_time, eeg_mask, _eeg_attention_mask(
            n_time_blocks, n_channels, device=neuro.device
        )


def _eeg_attention_mask(
    n_time_blocks: int, n_channels: int, *, device: torch.device
) -> torch.Tensor:
    n_tokens = n_time_blocks * n_channels
    mask = torch.tril(torch.ones(n_tokens, n_tokens, device=device, dtype=torch.bool))
    for block in range(n_time_blocks):
        start = block * n_channels
        stop = start + n_channels
        mask[start:stop, start:stop] = True
    return mask.unsqueeze(0)


def build_generation_inputs(
    eeg: torch.Tensor,
    input_chans: torch.Tensor,
    input_time: torch.Tensor,
    eeg_mask: torch.Tensor,
    eeg_attention_mask: torch.Tensor,
    prompt_tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the keyword tensors expected by ``NeuroLM.generate``."""
    if prompt_tokens.ndim == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0).expand(eeg.shape[0], -1)
    if prompt_tokens.shape[0] != eeg.shape[0]:
        raise ValueError("prompt_tokens batch dimension must match EEG batch size.")

    n_eeg = eeg.shape[1]
    n_text = prompt_tokens.shape[1]
    text_attention = torch.tril(
        torch.ones(n_eeg + n_text, n_eeg + n_text, dtype=torch.bool, device=eeg.device)
    )
    text_attention[:n_eeg, :n_eeg] = eeg_attention_mask[0]
    # Keep a singleton head dimension.  The official ``generate`` method
    # appends rows/columns to a 4-D mask after every generated token.
    text_attention = text_attention.unsqueeze(0).unsqueeze(0)
    return {
        "x_eeg": eeg,
        "x_text": prompt_tokens,
        "input_chans": input_chans,
        "input_time": input_time,
        "input_mask": eeg_mask,
        "eeg_mask": eeg_mask,
        "eeg_text_mask": text_attention,
    }
