# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared pytest fixtures for neuralbench tests."""

import typing as tp
from collections.abc import Callable
from pathlib import Path

import pytest

from .data import Data

# Bypass multiprocessing in the test sandbox; ``Test2024Eeg`` is tiny enough
# that in-process execution is fast.
_NO_CLUSTER: tp.Any = {"cluster": None}


@pytest.fixture(scope="session")
def test2024eeg_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped temp directory for the ``Test2024Eeg`` synthetic study.

    The synthetic ``.fif`` files (~15 MB / timeline) are generated lazily on the
    first ``Study.run()`` call.  Sharing one directory across test modules
    (``test_data.py``, ``test_utils.py``) avoids regenerating them per-module.
    """
    root = tmp_path_factory.mktemp("test2024eeg")
    (root / "Test2024Eeg").mkdir(exist_ok=True)
    return root


@pytest.fixture(scope="session")
def build_data(
    test2024eeg_path: Path,
) -> Callable[..., Data]:
    """Factory fixture that builds a tiny ``Data`` over the ``Test2024Eeg`` study.

    Returns a callable so each test can vary ``seed`` (and optionally
    ``sampler``) without re-threading the study path or the
    rest of the config.  ``event_field="subject"`` keeps all 3 subjects in
    the train split so ``compute_class_weights_from_dataset`` sees no
    class-index gaps -- a quiet workaround for a separate latent bug.
    """

    def _factory(
        *,
        seed: int | None,
        sampler: tp.Any | None = None,
    ) -> Data:
        config: tp.Any = dict(
            study={
                "name": "Test2024Eeg",
                "path": test2024eeg_path,
                "infra_timelines": _NO_CLUSTER,
            },
            neuro={"name": "MneRaw", "event_types": "Eeg"},
            target={
                "name": "LabelEncoder",
                "event_field": "subject",
                "event_types": "Word",
            },
            channel_positions={"name": "ChannelPositions"},
            trigger_event_type="Word",
            start=0.0,
            duration=0.5,
            batch_size=4,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
            seed=seed,
            sampler=sampler,
        )
        return Data(**config)

    return _factory
