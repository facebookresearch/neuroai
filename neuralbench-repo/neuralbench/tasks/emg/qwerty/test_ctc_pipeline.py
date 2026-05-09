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


@pytest.fixture
def bids_tree(tmp_path):
    sub, ses = "00000001", "0000000001"
    emg_dir = tmp_path / f"sub-{sub}" / f"ses-{ses}" / "emg"
    emg_dir.mkdir(parents=True)
    stem = f"sub-{sub}_ses-{ses}_task-typing"
    # Stub BDF — iter_timelines / _bids_paths only check existence.
    (emg_dir / f"{stem}_emg.bdf").write_bytes(b"\x00" * 16)
    (emg_dir / f"{stem}_events.tsv").write_text(
        "onset\tduration\tvalue\tprompt_text\tkey\n"
        "0.10\t1.5\tprompt\thello\t\n"
        "0.20\t0.05\tkeystroke_press\t\th\n"
        "0.30\t0.05\tkeystroke_press\t\te\n"
        "0.40\t0.05\tkeystroke_press\t\tKey.space\n"
    )
    return tmp_path, sub, ses


def test_emg2qwerty_bids_loader(bids_tree):
    from neuralfetch.studies.emg2qwerty import Emg2qwerty

    root, sub, ses = bids_tree
    study = Emg2qwerty(path=str(root))

    assert list(study.iter_timelines()) == [{"subject": sub, "session": ses}]

    df = study._load_timeline_events({"subject": sub, "session": ses})
    types = df["type"].tolist()
    assert types.count("Emg2qwertyRaw") == 1 and types.count("Sentence") == 1
    assert df.loc[df["type"] == "Keystroke", "text"].tolist() == ["h", "e", "Key.space"]


@pytest.mark.parametrize(
    ("subject", "session"),
    [("..", "0000000001"), ("00000001", "../../../etc"), ("$ub", "ses!")],
)
def test_emg2qwerty_bids_id_validation_rejects_unsafe(bids_tree, subject, session):
    from neuralfetch.studies.emg2qwerty import Emg2qwerty

    root, _, _ = bids_tree
    with pytest.raises(ValueError, match="unsafe BIDS id"):
        Emg2qwerty(path=str(root))._bids_paths(subject, session)


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


def test_emg2qwerty_bids_root_handles_download_subfolder(tmp_path):
    # ``Study.download`` lands BIDS under ``self.path/download/``.
    from neuralfetch.studies.emg2qwerty import Emg2qwerty

    sub, ses = "00000002", "0000000002"
    download_root = tmp_path / "download" / f"sub-{sub}" / f"ses-{ses}" / "emg"
    download_root.mkdir(parents=True)
    stem = f"sub-{sub}_ses-{ses}_task-typing"
    (download_root / f"{stem}_emg.bdf").write_bytes(b"\x00" * 16)

    study = Emg2qwerty(path=str(tmp_path))
    assert study._bids_root() == tmp_path / "download"
    assert list(study.iter_timelines()) == [{"subject": sub, "session": ses}]
