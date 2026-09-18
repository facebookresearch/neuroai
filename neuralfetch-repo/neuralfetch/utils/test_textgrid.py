# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for neuralfetch.utils.textgrid (read_tiers_lenient)."""

from pathlib import Path

import pytest

from neuralfetch.utils import textgrid

# A normal ("long") ooTextFile grid: two labelled words around an empty
# interval, so include_empty is observable.
LONG = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.5
            text = " hello "
        intervals [2]:
            xmin = 0.5
            xmax = 1.0
            text = ""
        intervals [3]:
            xmin = 1.0
            xmax = 2.0
            text = "world"
"""

# The same two words, but the second interval starts a hair before the first
# ends -- the ~1e-4 s overlap praatio's IntervalTier validator rejects.
OVERLAPPING = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 1.0001
            text = "hello"
        intervals [2]:
            xmin = 1.0
            xmax = 2.0
            text = "world"
"""

# Chronological format: every entry is prefixed by its 1-based tier index and
# listed in time order. praatio cannot parse this at all.
CHRONOLOGICAL = """\
"Praat chronological TextGrid text file"
0 2
2
"IntervalTier" "words" 0 2
"TextTier" "marks" 0 2
1 0 0.5 " hello "
2 0.25 "beep"
1 0.5 1.0 ""
1 1.0 2.0 "world"
"""


def _write(tmp_path: Path, name: str, data: str, encoding: str = "utf-8") -> Path:
    fp = tmp_path / name
    fp.write_text(data, encoding=encoding)
    return fp


def test_long_format_strips_labels_and_drops_empties(tmp_path: Path) -> None:
    """Labels come back stripped, and empty intervals are dropped by default."""
    pytest.importorskip("praatio")
    tiers = textgrid.read_tiers_lenient(_write(tmp_path, "long.TextGrid", LONG))
    assert [name for name, _ in tiers] == ["words"]
    assert tiers[0][1] == [(0.0, 0.5, "hello"), (1.0, 2.0, "world")]


def test_include_empty_keeps_positional_indexing(tmp_path: Path) -> None:
    """include_empty=True retains the blank interval so positions are stable."""
    pytest.importorskip("praatio")
    tiers = textgrid.read_tiers_lenient(
        _write(tmp_path, "long.TextGrid", LONG), include_empty=True
    )
    assert tiers[0][1] == [
        (0.0, 0.5, "hello"),
        (0.5, 1.0, ""),
        (1.0, 2.0, "world"),
    ]


def test_tolerates_boundary_overlaps(tmp_path: Path) -> None:
    """A ~1e-4 s overlap is read through rather than rejected.

    ``praatio.openTextgrid`` raises on this grid; going through the lower-level
    ``parseTextgridStr`` skips the per-tier overlap validation.
    """
    ptg = pytest.importorskip("praatio.textgrid")
    errors = pytest.importorskip("praatio.utilities.errors")
    fp = _write(tmp_path, "overlap.TextGrid", OVERLAPPING)

    with pytest.raises(errors.TextgridStateError, match="overlap in time"):
        ptg.openTextgrid(str(fp), includeEmptyIntervals=False)

    tiers = textgrid.read_tiers_lenient(fp)
    assert tiers[0][1] == [(0.0, 1.0001, "hello"), (1.0, 2.0, "world")]


def test_chronological_format(tmp_path: Path) -> None:
    """The chronological format parses without praatio, across both tier types."""
    tiers = textgrid.read_tiers_lenient(
        _write(tmp_path, "chrono.TextGrid", CHRONOLOGICAL)
    )
    assert dict(tiers) == {
        "words": [(0.0, 0.5, "hello"), (1.0, 2.0, "world")],
        "marks": [(0.25, 0.25, "beep")],  # point tier -> zero-length interval
    }


def test_chronological_include_empty(tmp_path: Path) -> None:
    """include_empty applies to the chronological path too."""
    tiers = dict(
        textgrid.read_tiers_lenient(
            _write(tmp_path, "chrono.TextGrid", CHRONOLOGICAL), include_empty=True
        )
    )
    assert tiers["words"] == [
        (0.0, 0.5, "hello"),
        (0.5, 1.0, ""),
        (1.0, 2.0, "world"),
    ]


def test_reads_utf16(tmp_path: Path) -> None:
    """Praat writes UTF-16 by default; the reader falls back to UTF-8."""
    fp = _write(tmp_path, "chrono16.TextGrid", CHRONOLOGICAL, encoding="utf-16")
    tiers = dict(textgrid.read_tiers_lenient(fp))
    assert tiers["words"] == [(0.0, 0.5, "hello"), (1.0, 2.0, "world")]


def test_quoted_label_escapes(tmp_path: Path) -> None:
    """A literal double-quote is written "" inside a quoted label."""
    data = CHRONOLOGICAL.replace('"world"', '"say ""hi"""')
    tiers = dict(textgrid.read_tiers_lenient(_write(tmp_path, "esc.TextGrid", data)))
    assert tiers["words"][-1] == (1.0, 2.0, 'say "hi"')


def test_rejects_non_chronological_body() -> None:
    """The chronological parser refuses a grid that is not in that format."""
    with pytest.raises(ValueError, match="not a Praat chronological TextGrid"):
        textgrid._parse_chronological_textgrid(LONG)
