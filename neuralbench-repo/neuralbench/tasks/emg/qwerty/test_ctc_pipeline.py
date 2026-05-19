# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests specific to the emg/qwerty task — vocabulary tables (this
package owns ``charset.py``) and the ``Emg2qwerty`` BIDS study source.

KeystrokeSequence / CharacterErrorRates / CtcSeqLoss / BandRotation
each have their own test files in the package that owns them
(``neuralset.extractors.test_text``,
``neuraltrain.metrics.test_metrics``, ``neuraltrain.losses.test_losses``,
and braindecode's augmentation tests).
"""

from __future__ import annotations

import pytest

from .charset import (
    COMPACT_KEY_TO_LABEL,
    COMPACT_NULL_CLASS,
    COMPACT_NUM_CLASSES,
    PAPER_KEY_TO_LABEL,
    PAPER_NULL_CLASS,
    PAPER_NUM_CLASSES,
)


def test_charset_class_count_invariants():
    paper_keys = {k for k, _ in PAPER_KEY_TO_LABEL}
    assert (PAPER_NULL_CLASS, PAPER_NUM_CLASSES, len(paper_keys)) == (98, 99, 98)

    compact_keys = {k for k, _ in COMPACT_KEY_TO_LABEL}
    assert (COMPACT_NULL_CLASS, COMPACT_NUM_CLASSES, len(compact_keys)) == (50, 51, 50)
    # compact drops uppercase, shifted symbols, and Key.shift.
    assert {"Key.shift", "A", "!"}.isdisjoint(compact_keys)


# --- Emg2qwerty study (synthetic BIDS tree) ------------------------------


_EVENTS_TSV = (
    "onset\tduration\tvalue\tprompt_text\tkey\n"
    "0.10\t1.5\tprompt\thello\t\n"
    "0.20\t0.05\tkeystroke_press\t\th\n"
    "0.30\t0.05\tkeystroke_press\t\te\n"
    "0.40\t0.05\tkeystroke_press\t\tKey.space\n"
)


def _make_bids_tree(root, subdir=""):
    """Build a synthetic single-(subject, session) BIDS tree under ``root``
    or ``root / subdir``. Returns ``(subject, session, bids_root)``."""
    sub, ses = "00000001", "0000000001"
    bids_root = root / subdir if subdir else root
    emg_dir = bids_root / f"sub-{sub}" / f"ses-{ses}" / "emg"
    emg_dir.mkdir(parents=True)
    stem = f"sub-{sub}_ses-{ses}_task-typing"
    (emg_dir / f"{stem}_emg.bdf").write_bytes(b"\x00" * 16)  # iter_timelines / _bids_paths only check existence.
    (emg_dir / f"{stem}_events.tsv").write_text(_EVENTS_TSV)
    return sub, ses, bids_root


@pytest.fixture
def bids_tree(tmp_path):
    sub, ses, _ = _make_bids_tree(tmp_path)
    return tmp_path, sub, ses


@pytest.mark.parametrize("subdir", ["", "download"])
def test_emg2qwerty_study_source(tmp_path, subdir):
    """``iter_timelines`` / ``_load_timeline_events`` / ``_bids_root`` all
    work whether the BIDS tree sits at the path root or under
    ``download/`` (the latter is the layout ``Study.download`` produces)."""
    from neuralfetch.studies.emg2qwerty import Emg2qwerty

    sub, ses, bids_root = _make_bids_tree(tmp_path, subdir)
    study = Emg2qwerty(path=str(tmp_path))

    assert study._bids_root() == bids_root
    assert list(study.iter_timelines()) == [{"subject": sub, "session": ses}]

    df = study._load_timeline_events({"subject": sub, "session": ses})
    types = df["type"].tolist()
    assert types.count("Emg2qwertyRaw") == 1 and types.count("Sentence") == 1
    assert df.loc[df["type"] == "Keystroke", "text"].tolist() == ["h", "e", "Key.space"]


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        # rstrip("\\n") would treat its arg as a char-set; need exact-suffix match.
        (r"fun\n", "fun"),
        (r"running\n", "running"),
        ("hello", "hello"),
        (r"\n\n", r"\n"),
    ],
)
def test_load_timeline_events_strips_only_literal_suffix(bids_tree, raw_text, expected):
    from neuralfetch.studies.emg2qwerty import Emg2qwerty

    root, sub, ses = bids_tree
    stem = f"sub-{sub}_ses-{ses}_task-typing"
    events_path = root / f"sub-{sub}" / f"ses-{ses}" / "emg" / f"{stem}_events.tsv"
    events_path.write_text(
        "onset\tduration\tvalue\tprompt_text\tkey\n"
        f"0.10\t1.5\tprompt\t{raw_text}\t\n"
    )
    df = Emg2qwerty(path=str(root))._load_timeline_events(
        {"subject": sub, "session": ses}
    )
    sentences = df.loc[df["type"] == "Sentence", "text"].tolist()
    assert sentences == [expected]


