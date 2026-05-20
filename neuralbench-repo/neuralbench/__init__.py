# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .cli import run_benchmark, run_benchmark_cli
from .utils import SequenceLabelEncoder

# ``SequenceLabelEncoder`` is re-exported so importing ``neuralbench``
# registers it in the ``exca`` discriminator and YAML configs (e.g.
# ``emg/typing/config.yaml``) can resolve ``name: SequenceLabelEncoder``
# without an explicit import.
__all__ = ["SequenceLabelEncoder", "run_benchmark", "run_benchmark_cli"]
