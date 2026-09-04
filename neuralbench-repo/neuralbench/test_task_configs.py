# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression guards for shipped task ``config.yaml`` semantics.

These are data-free: they read the committed YAML and exercise the
segmenter primitives, so they run without any downloaded study.
"""

from __future__ import annotations

import pandas as pd
import pytest

from neuralset.events import standardize_events
from neuralset.segments import list_segments

from .registry import _resolve_task_dir, load_yaml_config


@pytest.fixture
def reaction_time_config() -> dict:
    config = load_yaml_config(
        _resolve_task_dir("eeg", "reaction_time") / "config.yaml", safe=True
    )
    assert config is not None
    return config["data"]


def test_reaction_time_is_stimulus_locked(reaction_time_config: dict) -> None:
    """The EEG reaction-time window is locked to stimulus onset, not the response.

    Regression guard for #195. ``Keystroke.start`` is the button-press onset, so
    triggering on it puts the window entirely *after* the response. The EEG
    Foundation Challenge 2025 spec (arXiv:2506.19141) defines the window as
    0.5-2.5 s after *stimulus* onset.

    ``target.event_types`` has to move with the trigger: with
    ``aggregation: trigger`` the extractor reads the label off the trigger
    itself and rejects a trigger it does not accept.
    """
    assert reaction_time_config["trigger_event_type"] == "Stimulus"
    assert reaction_time_config["target"]["event_types"] == "Stimulus"
    assert reaction_time_config["start"] == 0.5
    assert reaction_time_config["duration"] == 2.0


def test_reaction_time_window_follows_stimulus_onset(
    reaction_time_config: dict,
) -> None:
    """Segments built with the shipped offsets sit after the stimulus, not the press.

    Mirrors one CCD trial: contrast change at t=11.0, button press at t=12.2
    (``reaction_time`` 1.2). Only the ``target`` Stimulus carries a
    ``reaction_time``, which is what ``filter_reaction_time`` selects on.
    """
    start = reaction_time_config["start"]
    duration = reaction_time_config["duration"]
    stimulus_onset, press_onset = 11.0, 12.2

    events = standardize_events(
        pd.DataFrame(
            [
                dict(
                    type="Stimulus",
                    start=stimulus_onset,
                    duration=2.4,
                    timeline="tl0",
                    reaction_time=press_onset - stimulus_onset,
                ),
                dict(
                    type="Keystroke",
                    start=press_onset,
                    duration=0.1,
                    timeline="tl0",
                    text="left_buttonPress",
                    reaction_time=press_onset - stimulus_onset,
                ),
            ]
        )
    )
    triggers = (events.type == reaction_time_config["trigger_event_type"]) & events[
        "reaction_time"
    ].notna()

    segments = list_segments(events, triggers=triggers, start=start, duration=duration)

    assert len(segments) == 1
    assert segments[0].start == pytest.approx(stimulus_onset + start)
    # The response falls inside the window -- the premise of the task.
    assert segments[0].start < press_onset < segments[0].stop
