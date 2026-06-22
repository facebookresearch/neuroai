# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Custom loss functions for neuralbench tasks.

Importing this module has the side-effect of registering the loss classes
defined here with the ``BaseLoss`` discriminated-union machinery from
``neuraltrain.losses.base``, so they can be referenced by class name in
task YAML configs.
"""
