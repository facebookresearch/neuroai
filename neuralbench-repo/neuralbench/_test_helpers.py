# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Module-level helper classes for tests.

Some tests need to spawn worker processes that re-import the test
fixtures. ``pytest --import-mode=importlib`` keeps test modules off
``sys.modules`` under their dotted name, which breaks pickling for
classes defined inside a ``test_*.py`` file. Placing helpers here keeps
them importable from a spawned child.
"""

import exca

from neuraltrain.utils import BaseExperiment


class ValueExperiment(BaseExperiment):
    value: int
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def run(self):
        return self.value


class FailingExperiment(BaseExperiment):
    """Raises ``RuntimeError`` from :meth:`run` for error-propagation tests."""

    value: int
    infra: exca.TaskInfra = exca.TaskInfra(version="1")

    @infra.apply
    def run(self):
        raise RuntimeError(f"intentional failure for value={self.value}")
