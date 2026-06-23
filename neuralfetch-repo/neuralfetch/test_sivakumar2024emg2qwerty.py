# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the ``Sivakumar2024Emg2qwerty`` BIDS study source."""

from __future__ import annotations

import pytest

from neuralfetch.studies.sivakumar2024emg2qwerty import (
    PAPER_KEY_TO_LABEL,
    PAPER_NULL_CLASS,
    PAPER_NUM_CLASSES,
    Sivakumar2024Emg2qwerty,
)

_EVENTS_TSV = (
    "onset\tduration\tvalue\tprompt_text\tkey\n"
    "0.10\t1.5\tprompt\thello\t\n"
    "0.20\t0.05\tkeystroke_press\t\th\n"
    "0.30\t0.05\tkeystroke_press\t\te\n"
    "0.40\t0.05\tkeystroke_press\t\tKey.space\n"
)


def _make_bids_tree(root, subdir="download"):
    """Build a synthetic single-(subject, session) BIDS tree under
    ``root/subdir/<dataset_id>`` (eegdash's ``cache_dir/<dataset_id>``
    layout). Returns ``(subject, session, bids_root)``."""
    sub, ses = "00000001", "0000000001"
    base = root / subdir if subdir else root
    bids_root = base / Sivakumar2024Emg2qwerty.NEMAR_DATASET_ID
    emg_dir = bids_root / f"sub-{sub}" / f"ses-{ses}" / "emg"
    emg_dir.mkdir(parents=True)
    stem = f"sub-{sub}_ses-{ses}_task-typing"
    # iter_timelines / _bids_paths only check existence.
    (emg_dir / f"{stem}_emg.bdf").write_bytes(b"\x00" * 16)
    (emg_dir / f"{stem}_events.tsv").write_text(_EVENTS_TSV)
    return sub, ses, bids_root


@pytest.fixture
def bids_tree(tmp_path):
    # build under the study's resolved path so it matches ``bids_root``
    study = Sivakumar2024Emg2qwerty(path=str(tmp_path))
    sub, ses, _ = _make_bids_tree(study.path)
    return study.path, sub, ses


def test_emg2qwerty_study_source(tmp_path):
    """``iter_timelines`` / ``_load_timeline_events`` / ``bids_root`` work
    on a BIDS tree placed under ``download/`` (the layout
    ``Study.download`` produces)."""
    study = Sivakumar2024Emg2qwerty(path=str(tmp_path))
    # the study subfolder is resolved in model_post_init, so build the tree
    # under the study's resolved path (``<tmp_path>/Sivakumar2024Emg2qwerty``).
    sub, ses, bids_root = _make_bids_tree(study.path)

    assert study.bids_root == bids_root
    assert list(study.iter_timelines()) == [{"subject": sub, "session": ses}]

    df = study._load_timeline_events({"subject": sub, "session": ses})
    types = df["type"].tolist()
    assert types.count("BidsEmg") == 1 and types.count("Sentence") == 1
    keystrokes = df.loc[df["type"] == "Keystroke"]
    assert keystrokes["text"].tolist() == ["h", "e", "Key.space"]
    # Keystroke events carry a pre-computed integer ``label`` in
    # ``[0, PAPER_NUM_CLASSES)`` (the CTC blank lives at
    # ``PAPER_NULL_CLASS``); the SequenceLabelEncoder consumes this
    # column directly without any string→int lookup at encode time.
    assert keystrokes["label"].tolist() == [
        PAPER_KEY_TO_LABEL["h"],
        PAPER_KEY_TO_LABEL["e"],
        PAPER_KEY_TO_LABEL["Key.space"],
    ]
    # Nullable ``Int64`` survives the downstream ``pd.concat`` with
    # rows that lack the ``label`` field (raw / sentences) without
    # demoting to float64.
    assert keystrokes["label"].dtype == "Int64"
    assert keystrokes["label"].between(0, PAPER_NULL_CLASS - 1).all()


def test_paper_vocab_invariants():
    """The Sivakumar et al. (2024) vocabulary is 98 dense labels +
    one CTC blank, totalling 99 classes for the head."""
    assert (PAPER_NULL_CLASS, PAPER_NUM_CLASSES, len(PAPER_KEY_TO_LABEL)) == (98, 99, 98)
    # Labels are dense in [0, PAPER_NULL_CLASS); the blank sits at
    # PAPER_NULL_CLASS and the CTC head emits PAPER_NUM_CLASSES logits.
    assert set(PAPER_KEY_TO_LABEL.values()) == set(range(PAPER_NULL_CLASS))


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
    root, sub, ses = bids_tree
    stem = f"sub-{sub}_ses-{ses}_task-typing"
    events_path = (
        root
        / "download"
        / Sivakumar2024Emg2qwerty.NEMAR_DATASET_ID
        / f"sub-{sub}"
        / f"ses-{ses}"
        / "emg"
        / f"{stem}_events.tsv"
    )
    events_path.write_text(
        f"onset\tduration\tvalue\tprompt_text\tkey\n0.10\t1.5\tprompt\t{raw_text}\t\n"
    )
    df = Sivakumar2024Emg2qwerty(path=str(root))._load_timeline_events(
        {"subject": sub, "session": ses}
    )
    sentences = df.loc[df["type"] == "Sentence", "text"].tolist()
    assert sentences == [expected]
