# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CTC machinery smoke tests (synthetic data only)."""

from __future__ import annotations

import pathlib

import pytest
import torch
from torch import nn

from neuralset.extractors.text import KeystrokeSequence
from neuraltrain.metrics.metrics import CharacterErrorRates

from .charset import (
    COMPACT_KEY_TO_LABEL,
    COMPACT_NULL_CLASS,
    COMPACT_NUM_CLASSES,
    PAPER_KEY_TO_LABEL,
    PAPER_NULL_CLASS,
    PAPER_NUM_CLASSES,
    vocab_kwargs,
)

_PAPER_LABEL = dict(PAPER_KEY_TO_LABEL)


def _y_true(seqs, max_len=8):
    """``(length-prefix, padded labels)`` packing for CtcSeqLoss + CER."""
    lengths = torch.tensor([len(s) for s in seqs])
    labels = torch.full((len(seqs), max_len), PAPER_NULL_CLASS, dtype=torch.long)
    for i, s in enumerate(seqs):
        labels[i, : len(s)] = torch.tensor([_PAPER_LABEL[c] for c in s])
    return torch.cat([lengths.unsqueeze(1), labels], dim=1)


@pytest.fixture
def make_keystrokes():
    from neuralset.events.etypes import Keystroke

    def _make(texts, starts=None):
        starts = starts or [0.1 * i for i in range(len(texts))]
        return [
            Keystroke(start=s, duration=0.05, text=t, timeline="t")
            for s, t in zip(starts, texts, strict=False)
        ]

    return _make


# --- KeystrokeSequence ---------------------------------------------------


def test_keystroke_sequence_pad_layout(make_keystrokes):
    ext = KeystrokeSequence(
        max_target_length=8, event_types="Keystroke", **vocab_kwargs(),
    )
    events = make_keystrokes(["h", "i", "Key.space"])
    ext.prepare(events)
    out = ext(events, start=0.0, duration=1.0)

    assert out.shape == (9,) and int(out[0]) == 3
    assert out[1:4].tolist() == [_PAPER_LABEL[k] for k in ("h", "i", "Key.space")]
    assert (out[4:] == PAPER_NULL_CLASS).all()


def test_keystroke_sequence_truncation_warns_once(make_keystrokes, caplog):
    ext = KeystrokeSequence(
        max_target_length=2, event_types="Keystroke", **vocab_kwargs(),
    )
    events = make_keystrokes(list("hello"))
    ext.prepare(events)
    with caplog.at_level("WARNING"):
        ext(events, start=0.0, duration=1.0)
        ext(events, start=0.0, duration=1.0)
    assert sum("truncating" in r.message for r in caplog.records) == 1


def test_keystroke_sequence_core_window_filters(make_keystrokes):
    # core [10.9, 14.9): only 'c'@11.0 + 'd'@14.5 qualify.
    ext = KeystrokeSequence(
        max_target_length=8, event_types="Keystroke",
        core_start_offset=0.9, core_duration=4.0,
        **vocab_kwargs(),
    )
    events = make_keystrokes(list("abcde"), starts=[10.0, 10.5, 11.0, 14.5, 14.95])
    ext.prepare(events)
    out = ext(events, start=10.0, duration=5.0)
    assert int(out[0]) == 2
    assert out[1:3].tolist() == [_PAPER_LABEL["c"], _PAPER_LABEL["d"]]


# --- Vocabulary tables + CER metric --------------------------------------


def test_charset_class_count_invariants():
    paper_keys = {k for k, _ in PAPER_KEY_TO_LABEL}
    assert (PAPER_NULL_CLASS, PAPER_NUM_CLASSES, len(paper_keys)) == (98, 99, 98)

    compact_keys = {k for k, _ in COMPACT_KEY_TO_LABEL}
    assert (COMPACT_NULL_CLASS, COMPACT_NUM_CLASSES, len(compact_keys)) == (50, 51, 50)
    # compact drops uppercase, shifted symbols, and Key.shift.
    assert {"Key.shift", "A", "!"}.isdisjoint(compact_keys)


def test_cer_perfect_predictions():
    seqs = ("hey", "world")
    y_true = _y_true(seqs)

    T_out = 30
    y_pred = torch.full((len(seqs), T_out, PAPER_NUM_CLASSES), -100.0)
    for i, s in enumerate(seqs):
        for t, c in enumerate(s):
            y_pred[i, t, _PAPER_LABEL[c]] = 0.0
        y_pred[i, len(s):, PAPER_NULL_CLASS] = 0.0

    metric = CharacterErrorRates(blank_idx=PAPER_NULL_CLASS)
    metric.update(torch.log_softmax(y_pred, dim=-1), y_true)
    assert float(metric.compute()) == 0.0


@pytest.mark.parametrize(
    ("kwargs", "mode", "n_chans", "should_change"),
    [
        # train mode + matching channels: input is mutated.
        (dict(band_offsets=(-1, 1), max_temporal_jitter=4), "train", 32, True),
        # eval mode: identity passthrough.
        ({}, "eval", 32, False),
        # train mode but channels don't divide num_bands*electrodes_per_band.
        ({}, "train", 7, False),
    ],
)
def test_band_rotation_module(kwargs, mode, n_chans, should_change):
    from neuraltrain.augmentations import BandRotation

    aug = BandRotation(num_bands=2, electrodes_per_band=16, **kwargs)
    getattr(aug, mode)()
    x = torch.randn(2, n_chans, 64)
    assert torch.equal(aug(x), x) is not should_change


def test_band_rotation_delegates_to_braindecode_functional(monkeypatch):
    from braindecode.augmentation import functional as bd_functional
    from neuraltrain.augmentations import BandRotation

    captured: dict = {}
    real = bd_functional.band_rotation

    def spy(X, y, **kwargs):
        captured["called"] = True
        captured["kwargs"] = kwargs
        return real(X, y, **kwargs)

    monkeypatch.setattr(bd_functional, "band_rotation", spy)

    aug = BandRotation(
        num_bands=2, electrodes_per_band=16, band_offsets=(-1, 1),
        max_temporal_jitter=4,
    ).train()
    aug(torch.randn(2, 32, 64))
    assert captured.get("called")
    assert captured["kwargs"]["num_bands"] == 2


def test_band_rotation_config_builds_module():
    from neuraltrain.augmentations import BandRotation, BandRotationConfig

    cfg = BandRotationConfig(num_bands=2, electrodes_per_band=16, max_temporal_jitter=4)
    built = cfg.build()
    assert isinstance(built, BandRotation)
    assert built.num_bands == 2 and built.max_temporal_jitter == 4


# --- CtcSeqLoss + CER share the (y_pred, y_true) signature ----------------


def test_brain_module_ctc_loss_and_metric_share_signature():
    from neuraltrain.losses.losses import CtcSeqLoss

    seqs = ("hey", "world")
    y_true = _y_true(seqs)
    y_pred = torch.log_softmax(torch.randn(len(seqs), 64, PAPER_NUM_CLASSES), dim=-1)

    loss = CtcSeqLoss(blank=PAPER_NULL_CLASS, zero_infinity=True)
    assert torch.isfinite(loss(y_pred, y_true))

    metric = CharacterErrorRates(blank_idx=PAPER_NULL_CLASS)
    metric.update(y_pred, y_true)
    assert float(metric.compute()) >= 0.0


# --- n_outputs override path ---------------------------------------------


@pytest.mark.parametrize(
    ("override", "target_last_dim", "expected"),
    [(5, 17, 5), (None, 17, 17), (99, 99, 99)],
)
def test_n_outputs_override_routes_to_braindecode_builder(
    monkeypatch, override, target_last_dim, expected
):
    """``n_outputs_override`` wins over the inferred ``target.shape[-1]``."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from neuralbench import model_factory
    from neuraltrain.models.base import BaseBrainDecodeModel

    captured: dict[str, int] = {}

    def fake_build(cfg, wrapper, loader, n_in_channels, n_times, n_outputs):
        captured["n_outputs"] = n_outputs
        return nn.Identity()

    monkeypatch.setattr(model_factory, "build_braindecode_model", fake_build)
    monkeypatch.setattr(model_factory, "build_dummy_batch", lambda *a, **k: (None, "x"))
    monkeypatch.setattr(model_factory, "init_lazy_layers", lambda *a, **k: None)
    monkeypatch.setattr(
        model_factory, "summary",
        lambda *a, **k: SimpleNamespace(total_params=0, trainable_params=0),
    )

    batch = SimpleNamespace(data={
        "neuro": torch.randn(2, 4, 64),
        "target": torch.zeros(2, target_last_dim),
        "subject_id": torch.zeros(2, dtype=torch.long),
    })
    # ``DataLoader``-shaped stub: needs ``__iter__`` + ``.dataset.metadata``.
    loader = MagicMock(spec=["__iter__", "dataset"])
    loader.__iter__.return_value = iter([batch])
    loader.dataset = SimpleNamespace(metadata={})

    model_factory.build_brain_model(
        brain_model_config=MagicMock(spec=BaseBrainDecodeModel),
        downstream_model_wrapper=None,
        pretrained_weights_fname=None,
        train_loader=loader,
        n_outputs_override=override,
    )
    assert captured["n_outputs"] == expected


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


def test_emg2qwerty_iter_timelines(bids_tree):
    from neuralfetch.studies.emg2qwerty import Emg2qwerty
    root, sub, ses = bids_tree
    assert list(Emg2qwerty(path=str(root)).iter_timelines()) == [
        {"subject": sub, "session": ses}
    ]


def test_emg2qwerty_load_timeline_events(bids_tree):
    from neuralfetch.studies.emg2qwerty import Emg2qwerty
    root, sub, ses = bids_tree
    df = Emg2qwerty(path=str(root))._load_timeline_events(
        {"subject": sub, "session": ses}
    )
    types = df["type"].tolist()
    assert types.count("Emg2qwertyRaw") == 1
    assert types.count("Sentence") == 1
    assert df.loc[df["type"] == "Keystroke", "text"].tolist() == [
        "h", "e", "Key.space"
    ]


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
def test_load_timeline_events_strips_only_literal_suffix(
    bids_tree, raw_text, expected
):
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


def test_emg2qwerty_download_wires_neuralfetch_eegdash():
    from unittest import mock

    import neuralfetch.download as dl

    from neuralfetch.studies.emg2qwerty import Emg2qwerty

    with mock.patch.object(dl, "Eegdash") as patched:
        Emg2qwerty(path="/tmp/_nbqwerty_test")._download()

    patched.assert_called_once_with(
        study="nm000104", dset_dir=pathlib.Path("/tmp/_nbqwerty_test")
    )
    patched.return_value.download.assert_called_once()


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
