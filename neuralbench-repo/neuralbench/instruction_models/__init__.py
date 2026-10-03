"""Pluggable instruction/generative model execution for NeuralBench."""

from .registry import InstructionModelBackend, load_backend

__all__ = ["InstructionModelBackend", "load_backend"]
