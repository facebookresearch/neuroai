# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Import module to ensure configs are registered with DiscriminatedModel
from . import base as _base  # noqa: F401
from .base import BaseOptimizer as BaseOptimizer  # used for configs
from .base import LightningOptimizer as LightningOptimizer  # noqa
