# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Model config for an instantiated model defined outside this repo."""

import hashlib
import importlib.metadata
import inspect
import io
import sys
import uuid
from pathlib import Path

import cloudpickle
import torch
from torch import nn

from neuraltrain.models.base import BaseBrainModelConfig, BrainModelBuildContext


def model_digest(model: nn.Module) -> str:
    """Hash *model*'s layout and weights, identically in any process.

    Not the serialized file: cloudpickle tags each class definition it stores
    with a fresh id, so the bytes differ every run and a uid built from them
    would never find the results of the previous call.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(str(model).encode() + buffer.getvalue()).hexdigest()


def _pickle_code_by_value(model: nn.Module) -> None:
    """Make the pickle carry the code of *model*'s classes, not just import paths.

    Pickle stores a class by reference, which a worker resolves by importing it.
    That works for an installed package, editable ones included, but a model
    being tried out lives in a script or a plain checkout, importable from the
    caller's ``sys.path`` alone.  Those modules are pickled by value instead, so
    the job needs nothing on its ``PYTHONPATH``; the cost is that the classes it
    rebuilds are not the ones the caller passed, which only ``isinstance`` in
    the caller's own process would notice.
    """
    installed = importlib.metadata.packages_distributions()
    for name in {type(module).__module__.split(".")[0] for module in model.modules()}:
        module = sys.modules.get(name)
        if module is not None and name not in installed:
            cloudpickle.register_pickle_by_value(module)


def save_instance(model: nn.Module, folder: Path) -> tuple[Path, str]:
    """Serialize *model* under its digest, as ``(path, digest)``."""
    folder.mkdir(parents=True, exist_ok=True)
    _pickle_code_by_value(model)
    # uuid4, not id(model): two processes can hold objects at one address, then
    # interleave writes into one file and each read back the other's bytes.
    tmp = folder / f".{uuid.uuid4().hex}.tmp"
    torch.save(model, tmp, pickle_module=cloudpickle)
    digest = model_digest(model)
    path = folder / f"{digest}.pt"
    tmp.replace(path)
    return path, digest


def check_forward(model: nn.Module) -> None:
    """Raise unless *model* reads channel identity from ``forward``."""
    params = inspect.signature(model.forward).parameters
    if "channel_positions" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ):
        return
    raise ValueError(
        f"forward() of {type(model).__name__} does not accept 'channel_positions', "
        "so it cannot tell one montage from another. NeuralBench passes it by "
        f"keyword as (batch, channels, 3), so the name must match exactly; got "
        f"{sorted(params)}."
    )


class ExternalModel(BaseBrainModelConfig):
    """An instantiated model built by out-of-tree code.

    Parameters
    ----------
    pickle_path : str
        Path to a serialized :class:`~torch.nn.Module`, as written by
        :func:`save_instance`.
    digest : str
        :func:`model_digest` of that model, so the weights take part in the
        cache uid rather than just the file name.
    """

    pickle_path: str
    digest: str

    def build_from_context(self, ctx: BrainModelBuildContext) -> nn.Module:
        """Load a fresh copy of the instance, so fine-tuned weights never leak.

        *ctx* is unused: the model adapts to the data rather than being built
        for it, which is the condition for accepting an instance at all.
        """
        path = Path(self.pickle_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Serialized model {path} not found. It is written to the "
                "neuralbench cache folder, which must be reachable from the "
                "worker running the experiment."
            )
        # weights_only=False: a whole pickled module, not a state dict.
        model = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"{path} contains {type(model).__name__}, expected a torch.nn.Module."
            )
        if (digest := model_digest(model)) != self.digest:
            raise ValueError(
                f"Model in {path} has digest {digest}, expected {self.digest}. "
                "The file was overwritten with a different model."
            )
        check_forward(model)
        return model
