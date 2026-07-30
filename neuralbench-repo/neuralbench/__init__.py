# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . import metrics as _metrics  # noqa: F401  # registers custom metric configs
from . import (
    transforms as _transforms,  # noqa: F401  # registers custom Event/Step subclasses
)
from .cli import run_benchmark, run_benchmark_cli
from .data import get_default_dataloaders
from .utils import SequenceLabelEncoder

# ``SequenceLabelEncoder`` is re-exported so importing ``neuralbench``
# registers it in the ``exca`` discriminator and YAML configs (e.g.
# ``emg/typing/config.yaml``) can resolve ``name: SequenceLabelEncoder``
# without an explicit import.
__all__ = [
    "SequenceLabelEncoder",
    "run_benchmark",
    "run_benchmark_cli",
    "get_default_dataloaders",
]
