# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Import modules to ensure configs are registered with DiscriminatedModel
from . import metrics as _metrics  # noqa: F401
from .base import BaseMetric as BaseMetric  # used for configs
