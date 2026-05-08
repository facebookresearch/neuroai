# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""emg2qwerty CTC task package.

Submodule imports below trigger ``__init_subclass__`` registration of
the task's pydantic-discriminated subclasses (Study sources, extractors,
the CER metric config) with neuralset / neuraltrain.
"""

from . import callbacks, extractors, metrics, study  # noqa: F401
