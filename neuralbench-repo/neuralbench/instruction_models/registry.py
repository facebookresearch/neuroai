"""Discovery contract for instruction-model backends."""

from __future__ import annotations

import importlib
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InstructionModelBackend(Protocol):
    """Interface implemented by LLM-style NeuralBench backends."""

    name: str

    def run_from_neuralbench(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Run preparation, training, or evaluation from the shared CLI."""


def load_backend(model_name: str) -> InstructionModelBackend | None:
    """Load backends/<model_name> without importing every optional model."""
    module_name = (
        f"neuralbench.instruction_models.backends.{model_name.lower()}"
    )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise
    backend = getattr(module, "BACKEND", None)
    if backend is None or not isinstance(backend, InstructionModelBackend):
        raise TypeError(
            f"{module_name} must expose BACKEND implementing "
            "InstructionModelBackend."
        )
    return backend
