"""NeuroLM backend for instruction-based EEG evaluation.

The official NeuroLM source is intentionally not vendored here.  The loader
accepts a separately checked-out official source directory, while this package
contains only the NeuralBench-facing input and task adapters.
"""

from .adapter import NeuroLMInputAdapter, build_generation_inputs
from .checkpoint import load_neurolm_checkpoint
from .distributed import DistributedEvalSampler, distributed_predict
from .runner import NeuroLMPrediction, predict
from .presets import (
    align_task_spec_to_dataset,
    builtin_task_spec,
    cog_bci_workload,
    tuab_pathology,
)
from .task_spec import NeuroLMTaskSpec


class NeuroLMBackend:
    """Registration object consumed by the generic instruction-model CLI."""

    name = "neurolm"

    def run_from_neuralbench(self, **kwargs):
        from .cli import run_from_neuralbench

        return run_from_neuralbench(**kwargs)


BACKEND = NeuroLMBackend()

__all__ = [
    "NeuroLMInputAdapter",
    "NeuroLMTaskSpec",
    "NeuroLMPrediction",
    "build_generation_inputs",
    "load_neurolm_checkpoint",
    "DistributedEvalSampler",
    "distributed_predict",
    "predict",
    "tuab_pathology",
    "cog_bci_workload",
    "builtin_task_spec",
    "align_task_spec_to_dataset",
    "BACKEND",
]
