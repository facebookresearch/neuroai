"""Lazy loader for official NeuroLM checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


def load_neurolm_checkpoint(
    checkpoint_path: str | Path,
    official_source_root: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Load a public NeuroLM checkpoint without vendoring its source code.

    ``official_source_root`` must point to a checkout of the official NeuroLM
    repository.  The path is added temporarily to ``sys.path`` because the
    upstream code uses imports such as ``from model.model import GPT``.
    """
    source_root = Path(official_source_root).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"NeuroLM source directory not found: {source_root}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"NeuroLM checkpoint not found: {checkpoint_path}")

    source_text = str(source_root)
    added = source_text not in sys.path
    if added:
        sys.path.insert(0, source_text)
    try:
        from model.model import GPTConfig
        from model.model_neurolm import NeuroLM

        checkpoint: dict[str, Any] = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model_args = dict(checkpoint["model_args"])
        model = NeuroLM(GPTConfig(**model_args), init_from="scratch")
        state_dict = dict(checkpoint["model"])
        prefix = "_orig_mod."
        state_dict = {
            key.removeprefix(prefix): value for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval()
    finally:
        if added:
            sys.path.remove(source_text)
