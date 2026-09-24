# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import pytest

from neuralset.events import etypes, eventquery


def make_event(
    *, start: float = 0.0, duration: float = 0.0, timeline: str = "t", **extra: tp.Any
) -> etypes.Event:
    """A bare ``Event`` whose unknown keys are grouped into ``extra``."""
    row = {"type": "Event", "start": start, "timeline": timeline, "duration": duration}
    row.update(extra)
    return etypes.Event.from_dict(row)


def run(expr: str, event: etypes.Event) -> bool:
    return eventquery.compile_event_query(expr)(event)


def test_event_query_supported_behavior() -> None:
    event = make_event(space="T1w", preproc="fmriprep")

    supported_true = [
        "space == 'T1w'",
        "space != 'MNI152NLin2009cAsym'",
        "space in ['T1w', 'fsaverage']",
        "preproc not in ['deepprep', 'hcp']",
        "space == 'fsaverage' or preproc == 'fmriprep'",
        "not (space == 'fsaverage')",
    ]
    for expr in supported_true:
        assert run(expr, event) is True, expr

    # Regression: boolean operators must short-circuit before resolving names.
    assert run("space == 'T1w' or missing == 'x'", event) is True
    assert run("space == 'Nope' and missing == 'x'", event) is False

    # fields= validates known names at compile time while still allowing
    # the reduced fMRI variant selector grammar.
    pred = eventquery.compile_event_query(
        "space == 'T1w' and preproc in ['fmriprep']",
        fields={"space", "preproc"},
    )
    assert pred(event) is True


def test_event_query_rejects_unsafe_or_invalid_queries() -> None:
    event = make_event(space="T1w", preproc="fmriprep")

    # Evaluated missing fields still fail fast.
    with pytest.raises(ValueError, match="unknown field 'missing'; queryable fields:"):
        run("missing == 'x'", event)

    # fields= catches typos early, even inside branches that would short-circuit.
    with pytest.raises(ValueError, match="unknown field 'spaace'; queryable fields:"):
        eventquery.compile_event_query(
            "space == 'T1w' or spaace == 'x'", fields={"space"}
        )

    unsafe_or_invalid = [
        "stop > '0'",
        "space.attr == 1",
        "space == preproc",
        "space in 'T1w'",
        "space in [1]",
        "space == 'T1w' == preproc",
        "a & b",
        "__import__('os')",
        "dunder__attr__ == 'x'",
        "",
    ]
    for expr in unsafe_or_invalid:
        with pytest.raises(ValueError):
            eventquery.compile_event_query(expr, fields={"space", "preproc"})
