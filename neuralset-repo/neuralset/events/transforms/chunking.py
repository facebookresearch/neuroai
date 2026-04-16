# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import pandas as pd

from ..study import EventsTransform
from .utils import chunk_events


class ChunkEvents(EventsTransform):
    """
    This functions chunks long events (audio or video) into shorter events. It supports two modes:

    1. **Duration-based chunking** (if `event_type_to_use` is None):

       - Each event of type `event_type_to_chunk` is split into consecutive chunks with duration between `min_duration` and `max_duration`.
       - Ensures that no chunk exceeds `max_duration`.

    This avoids OOM errors when feeding long events into a deep learning model (e.g. Wav2Vec).

        Example::

            input:
                min_duration: 2
                max_duration: 3
                events:
                    sound:    [.......]
            out:
                events:
                    sound1:   [...]
                    sound2:      [....]

    2. **Split-based chunking** (if :code:`event_type_to_use` is not :code:`None`):

       - Events of type :code:`event_type_to_chunk` are split according to the train/val/test splits of :code:`event_type_to_use`.
       - Ensures each chunk respects :code:`min_duration` and :code:`max_duration`.
       - Prevents data leakage when processing long events with a deep learning model (e.g. Wav2Vec.
       - Requires that the splits for :code:`event_type_to_use` are already assigned.

        Example::

            input:
                max_duration: 2
                event_type_to_use: Word
                events:
                    sound:    [.......]
                    word :    [1112233]
            out:
                events:
                    sound1:   [..]
                    sound2:     [.]
                    sound3:      [..]
                    sound4:        [..]

    """

    event_type_to_chunk: tp.Literal["Audio", "Video"]
    event_type_to_use: str | None = None
    min_duration: float | None = None
    max_duration: float = np.inf

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        return chunk_events(
            events,
            self.event_type_to_chunk,
            self.event_type_to_use,
            self.min_duration,
            self.max_duration,
        )
