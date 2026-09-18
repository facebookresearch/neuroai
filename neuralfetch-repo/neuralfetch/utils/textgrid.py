# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Lenient Praat TextGrid reading for study loaders.

This wraps ``praatio`` rather than replacing it: standard (normal/short) grids
are read through ``praatio.utilities.textgrid_io.parseTextgridStr``, and only
Praat's *chronological* format -- which praatio cannot parse in any form -- has
a parser of its own here.

``praatio.openTextgrid`` is strict in two ways that trip up the TextGrids some
studies ship: its ``IntervalTier`` constructor rejects the ~1e-4 s
floating-point boundary overlaps that Praat exports carry, and it cannot read
the chronological format.  :func:`read_tiers_lenient` skips that validation by
going through the lower-level ``parseTextgridStr``, handles chronological files
directly, and returns a plain
``[(tier_name, [(start, end, label), ...]), ...]`` structure with labels
stripped.

No other library removes the need for this module.  Measured over the 84
ds003020 grids (76 with overlaps, 2 chronological), ``praat-textgrids`` and
``tgt`` both tolerate the overlaps but read 82/84 -- neither parses the
chronological format -- and both are stale (last released 2022 and 2023);
``nltk_contrib.textgrid``, which these loaders used before praatio, is not on
PyPI at all and shipped a broken chronological regex we had to monkey-patch.
``parselmouth`` does read 84/84, and cross-checking it against this module is
what validates the parser below: both return an identical 730,184 intervals.
It is not used because it embeds Praat and exposes intervals one call at a
time, which makes bulk annotation reading ~50x slower.

``praatio`` is an opt-in extra, so it is imported inside
:func:`read_tiers_lenient` -- importing this module never requires it, and the
chronological path does not need it at all.  For that reason this is not
re-exported from :mod:`neuralfetch.utils` (same rationale as ``bids`` /
``runner``); use ``from neuralfetch.utils import textgrid`` and call
``textgrid.read_tiers_lenient(...)``.
"""

from __future__ import annotations

from pathlib import Path

_CHRONOLOGICAL_HEADER = '"Praat chronological TextGrid text file"'

# A chronological tier row is ``<tier-index> <start> <end>`` (IntervalTier) or
# ``<tier-index> <time>`` (point/TextTier); the label is the next token.
_CHRONO_INTERVAL_CLASS = "IntervalTier"


def _tokenize_praat(data: str) -> list[tuple[str, str]]:
    """Tokenize a Praat text file into ``("str" | "bare", value)`` tokens.

    Praat's text formats are a flat stream of whitespace-separated tokens: quoted
    strings (``"..."`` with a literal double-quote written as ``""``) and bare
    numeric/keyword tokens.  ``!`` starts a comment that runs to end-of-line, but
    only outside a quoted string.  This is enough to read the header + interval
    records of the *chronological* format, which praatio itself cannot parse.
    """
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(data)
    while i < n:
        c = data[i]
        if c in " \t\r\n":
            i += 1
        elif c == "!":  # comment to end of line
            while i < n and data[i] != "\n":
                i += 1
        elif c == '"':  # quoted string; "" is an escaped literal quote
            i += 1
            buf: list[str] = []
            while i < n:
                if data[i] == '"':
                    if i + 1 < n and data[i + 1] == '"':
                        buf.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                buf.append(data[i])
                i += 1
            tokens.append(("str", "".join(buf)))
        else:  # bare token (number / keyword) up to the next whitespace
            start = i
            while i < n and data[i] not in " \t\r\n":
                i += 1
            tokens.append(("bare", data[start:i]))
    return tokens


def _parse_chronological_textgrid(
    data: str,
) -> list[tuple[str, list[tuple[float, float, str]]]]:
    """Parse a Praat *chronological* TextGrid into ``[(tier_name, entries), ...]``.

    The chronological format lists every interval/point across all tiers in time
    order, each prefixed by its 1-based tier index, after a header that declares
    the time domain and the tiers.  praatio only supports the normal/short/JSON
    formats, so this fills that gap.
    """
    tokens = _tokenize_praat(data)
    values = [value for _, value in tokens]
    if not values or values[0] != _CHRONOLOGICAL_HEADER.strip('"'):
        raise ValueError("not a Praat chronological TextGrid")

    idx = 1
    idx += 2  # time domain (xmin, xmax) -- not needed downstream
    n_tiers = int(float(values[idx]))
    idx += 1

    tier_names: list[str] = []
    tier_classes: list[str] = []
    for _ in range(n_tiers):
        tier_classes.append(values[idx])
        tier_names.append(values[idx + 1])
        idx += 4  # class, name, tmin, tmax

    entries: list[list[tuple[float, float, str]]] = [[] for _ in range(n_tiers)]
    while idx < len(values):
        tier_no = int(float(values[idx]))
        idx += 1
        if tier_classes[tier_no - 1] == _CHRONO_INTERVAL_CLASS:
            start, end = float(values[idx]), float(values[idx + 1])
            label = values[idx + 2]
            idx += 3
        else:  # point tier: single timestamp, zero-length interval
            start = end = float(values[idx])
            label = values[idx + 1]
            idx += 2
        entries[tier_no - 1].append((start, end, label))

    return list(zip(tier_names, entries))


def read_tiers_lenient(
    path: Path,
    *,
    include_empty: bool = False,
) -> list[tuple[str, list[tuple[float, float, str]]]]:
    """Read a Praat TextGrid into ``[(tier_name, [(start, end, label), ...]), ...]``.

    Lenient counterpart to ``praatio.openTextgrid``: it (a) tolerates the ~1e-4 s
    floating-point boundary overlaps praatio's ``IntervalTier`` validator rejects
    (by reading via the lower-level ``parseTextgridStr``), (b) understands Praat's
    ``chronological`` format praatio cannot parse, and (c) strips surrounding
    whitespace from labels so word/phoneme tokens are clean.  ``include_empty``
    mirrors praatio's ``includeEmptyIntervals`` -- keep it True where a caller
    relies on positional indexing over every interval (e.g. word-index numbering).
    """
    try:
        data = path.read_text(encoding="utf-16")
    except UnicodeError:
        data = path.read_text(encoding="utf-8")

    if data.lstrip().startswith(_CHRONOLOGICAL_HEADER):
        tiers = _parse_chronological_textgrid(data)
    else:
        from praatio.utilities import textgrid_io

        parsed = textgrid_io.parseTextgridStr(data, includeEmptyIntervals=include_empty)
        tiers = [
            (tier["name"], [(float(s), float(e), lbl) for s, e, lbl in tier["entries"]])
            for tier in parsed["tiers"]
        ]

    result: list[tuple[str, list[tuple[float, float, str]]]] = []
    for name, entries in tiers:
        stripped = [(s, e, lbl.strip()) for s, e, lbl in entries]
        if not include_empty:
            stripped = [(s, e, lbl) for s, e, lbl in stripped if lbl != ""]
        result.append((name, stripped))
    return result
