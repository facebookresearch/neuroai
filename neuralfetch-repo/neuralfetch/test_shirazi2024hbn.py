# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the ``Shirazi2024Hbn`` contrast-change-detection events."""

from __future__ import annotations

import pytest

from neuralfetch.studies.shirazi2024hbn import Shirazi2024Hbn

_RELEASE, _SUBJECT, _TASK = "R5", "sub-NDARAA000AAA", "task-contrastChangeDetection"

# Three trials: a single response, two responses after the contrast change, and a
# premature response before it. Onsets are spaced so trial numbering (a cumsum over
# ``contrastTrial_start``) assigns every row to the intended trial.
_EVENTS_TSV = "\n".join(
    [
        "onset\tduration\tvalue\tfeedback",
        # Trial 1: one response, 1.2 s after the target.
        "10.0\t0.0\tcontrastTrial_start\tn/a",
        "11.0\t0.0\tleft_target\tn/a",
        "12.2\t0.0\tleft_buttonPress\tsmiley_face",
        # Trial 2: the response is at 1.0 s; the second press is 2.5 s out.
        "20.0\t0.0\tcontrastTrial_start\tn/a",
        "21.0\t0.0\tright_target\tn/a",
        "22.0\t0.0\tright_buttonPress\tsmiley_face",
        "23.5\t0.0\tright_buttonPress\tsmiley_face",
        # Trial 3: a press 0.5 s *before* the target, then the response at 1.4 s.
        "30.0\t0.0\tcontrastTrial_start\tn/a",
        "30.5\t0.0\tleft_buttonPress\tnon_target",
        "31.0\t0.0\tleft_target\tn/a",
        "32.4\t0.0\tleft_buttonPress\tsad_face",
        "40.0\t0.0\tcontrastTrial_start\tn/a",
    ]
)


@pytest.fixture
def targets(tmp_path):
    """The ``target`` Stimulus rows the reaction-time task triggers on."""
    study = Shirazi2024Hbn(path=str(tmp_path))
    eeg_dir = study.path / _RELEASE / "download" / _SUBJECT / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / f"{_SUBJECT}_{_TASK}_events.tsv").write_text(_EVENTS_TSV)

    events = study._load_contrast_change_detection_events(
        {"subject": _SUBJECT, "task": _TASK, "run": None, "release": _RELEASE}
    )
    targets = events[events["event_type"] == "target"]
    return targets.set_index("trial_num")


def test_reaction_time_uses_the_first_press_after_the_target(targets) -> None:
    """A trial with several responses is labelled with the first one.

    Trial 2 has presses 1.0 s and 2.5 s after the contrast change. The reaction
    time is the first, as in the EEG Foundation Challenge's own trial table;
    taking the last would overstate it by 1.5 s.
    """
    assert targets.loc[1.0, "reaction_time"] == pytest.approx(1.2)
    assert targets.loc[2.0, "reaction_time"] == pytest.approx(1.0)


def test_reaction_time_ignores_presses_before_the_target(targets) -> None:
    """A press preceding the contrast change is premature, not a response.

    Trial 3 opens with a press 0.5 s before the target (``non_target``
    feedback), which would give a negative reaction time.
    """
    assert targets.loc[3.0, "reaction_time"] == pytest.approx(1.4)
    assert not targets.loc[3.0, "is_correct"]
