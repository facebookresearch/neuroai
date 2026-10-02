# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import typing as tp

import numpy as np
import torch

import neuralset as ns


class SleepOnsetTargetExtractor(ns.extractors.BaseStatic):
    """Compute the time-to-N2-onset target dynamically from the segment's ``stop``.

    Reads the absolute ``n2_onset`` timestamp from a ``SleepOnsetMarker``
    event (emitted by :class:`~neuralbench.transforms.AddSleepOnsetTargets` or
    :class:`~neuralbench.transforms.AddSleepOnsetCutTargets`) and returns
    ``clip(n2_onset - segment.stop, floor_s, cap_s)``.  Computing the target
    from the actual segment boundary -- rather than reading a value baked into
    the event at transform time -- guarantees the label always matches the EEG
    window being fed to the model, even if the segmenter's ``duration`` or
    ``stride`` changes.

    With the default ``floor_s=0`` the target is the non-negative time until
    onset (a window ending after onset reads 0). A negative ``floor_s`` keeps
    the sign, so a window whose ``stop`` falls past onset reports how long ago
    onset happened -- used by the single-window task where the cut is drawn on
    both sides of onset.

    Parameters
    ----------
    event_types : str or tuple of str
        Type of event(s) to read the ``n2_onset`` field from.  Defaults to
        ``"SleepOnsetMarker"``.
    cap_s : float
        Upper clip bound on the target (largest reported time-to-onset).
    floor_s : float
        Lower clip bound on the target. ``0`` clips post-onset windows to 0;
        a negative value keeps the signed time relative to onset.
    """

    event_types: str | tuple[str, ...] = "SleepOnsetMarker"
    cap_s: float = 600.0
    floor_s: float = 0.0

    def prepare(self, obj: tp.Any) -> None:
        pass

    def get_static(self, event: ns.events.Event) -> torch.Tensor:
        """Unused: ``_get_timed_arrays`` is overridden to depend on the segment bounds."""
        raise NotImplementedError

    def _get_timed_arrays(
        self, events: list[ns.events.Event], start: float, duration: float
    ) -> tp.Iterable[ns.base.TimedArray]:
        stop = start + duration
        for event in events:
            n2_onset = float(event._get_field_or_extra("n2_onset"))
            time_to_onset = np.clip(n2_onset - stop, self.floor_s, self.cap_s)
            embedding = torch.tensor([time_to_onset], dtype=torch.float32)
            yield ns.base.TimedArray(
                frequency=0,
                duration=event.duration,
                start=event.start,
                data=embedding.numpy(),
            )
