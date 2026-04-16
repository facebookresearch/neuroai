# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Import modules to ensure configs are registered with DiscriminatedModel
from . import base as _base  # noqa: F401
from . import losses as _losses  # noqa: F401
from .base import BaseLoss as BaseLoss  # used for configs
