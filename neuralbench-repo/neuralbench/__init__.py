# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .cli import run_benchmark, run_benchmark_cli
# Register custom extractors in the ``exca`` discriminator at import
# time so YAML configs (e.g. ``emg/typing/config.yaml``) can resolve
# ``name: SequenceLabelEncoder`` without an explicit import.
from .utils import SequenceLabelEncoder  # noqa: F401
