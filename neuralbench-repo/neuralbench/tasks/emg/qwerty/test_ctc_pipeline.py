# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained smoke tests for the CTC machinery (synthetic data only)."""

from __future__ import annotations

import pathlib

import pytest
import torch
from torch import nn

from neuralset.extractors.text import KeystrokeSequence
from neuraltrain.metrics.metrics import CharacterErrorRates

from .charset import CharacterSet

CS = CharacterSet.paper()


def _vocab_kwargs(preset: str = "paper") -> dict:
    """Build the dicts ``KeystrokeSequence`` consumes from a preset."""
    cs = CharacterSet.from_preset(preset)
    return {
        "key_to_label": dict(cs._key_to_index),
        "unichar_to_key": dict(cs.UNICHAR_TO_KEY),
        "input_folds": dict(cs._input_folds),
    }


@pytest.fixture
def make_keystrokes():
    """Factory: build N Keystroke events at staggered start times."""
    from neuralset.events.etypes import Keystroke

    def _make(texts, starts=None):
        starts = starts or [0.1 * i for i in range(len(texts))]
        return [
            Keystroke(start=s, duration=0.05, text=t, timeline="t")
            for s, t in zip(starts, texts, strict=False)
        ]

    return _make


# ---------------------------------------------------------------------------
# KeystrokeSequence
# ---------------------------------------------------------------------------


def test_keystroke_sequence_pad_layout(make_keystrokes):
    ext = KeystrokeSequence(
        max_target_length=8, event_types="Keystroke", **_vocab_kwargs(),
    )
    events = make_keystrokes(["h", "i", "Key.space"])
    ext.prepare(events)
    out = ext(events, start=0.0, duration=1.0)

    assert out.shape == (9,) and int(out[0]) == 3
    assert out[1:4].tolist() == [CS.key_to_label(k) for k in ("h", "i", "Key.space")]
    assert (out[4:] == CS.null_class).all()


def test_keystroke_sequence_truncation_warns_once(make_keystrokes, caplog):
    ext = KeystrokeSequence(
        max_target_length=2, event_types="Keystroke", **_vocab_kwargs(),
    )
    events = make_keystrokes(list("hello"))
    ext.prepare(events)
    with caplog.at_level("WARNING"):
        ext(events, start=0.0, duration=1.0)
        ext(events, start=0.0, duration=1.0)
    assert sum("truncating" in r.message for r in caplog.records) == 1


def test_keystroke_sequence_core_window_filters(make_keystrokes):
    # Segment [10,15], core [10.9,14.9). Only 'c'@11.0 + 'd'@14.5 qualify.
    ext = KeystrokeSequence(
        max_target_length=8, event_types="Keystroke",
        core_start_offset=0.9, core_duration=4.0,
        **_vocab_kwargs(),
    )
    events = make_keystrokes(
        list("abcde"), starts=[10.0, 10.5, 11.0, 14.5, 14.95]
    )
    ext.prepare(events)
    out = ext(events, start=10.0, duration=5.0)
    assert int(out[0]) == 2
    assert out[1:3].tolist() == [CS.key_to_label("c"), CS.key_to_label("d")]


def test_keystroke_sequence_empty_segment(make_keystrokes):
    ext = KeystrokeSequence(
        max_target_length=8, event_types="Keystroke", allow_missing=True,
        **_vocab_kwargs(),
    )
    ext.prepare(make_keystrokes(["a"]))
    out = ext([], start=0.0, duration=1.0)
    assert out.shape == (9,) and int(out[0]) == 0


def test_keystroke_sequence_per_instance_vocab():
    """Distinct preset kwargs give distinct vocab sizes — no shared state."""
    paper = KeystrokeSequence(
        max_target_length=4, event_types="Keystroke", **_vocab_kwargs("paper"),
    )
    compact = KeystrokeSequence(
        max_target_length=4, event_types="Keystroke",
        **_vocab_kwargs("qwerty_compact"),
    )
    assert len(paper.key_to_label) == 98
    assert len(compact.key_to_label) == 50


# ---------------------------------------------------------------------------
# CharacterSet + CER metric
# ---------------------------------------------------------------------------


def test_charset_blank_and_size_invariants():
    assert CS.null_class == 98 == max(CS._key_to_index.values()) + 1
    assert CS.num_classes == 99


def test_qwerty_compact_has_51_classes():
    cs = CharacterSet.qwerty_compact()
    assert cs.num_classes == 51
    assert cs.null_class == 50
    keys = list(cs.KEY_TO_UNICODE.keys())
    assert len(keys) == 50
    assert "Key.shift" not in keys  # explicitly dropped
    assert "A" not in keys          # uppercase not in vocab
    assert "!" not in keys          # shifted symbol not in vocab


@pytest.mark.parametrize(
    ("input_keys", "expected"),
    [
        (["a"], ["a"]),
        (["⏎", "⌫", "⇧"], ["Key.enter", "Key.backspace", "Key.shift"]),
        ([" ", "\n", "\b"], ["Key.space", "Key.enter", "Key.backspace"]),
        (["Key.shift", "Key.shift_l"], ["Key.shift"]),  # _l variant dropped
        (["ZZZZ", "Key.unknown"], []),                  # out-of-vocab dropped
    ],
)
def test_charset_paper_clean_keys(input_keys, expected):
    assert CS.clean_keys(input_keys) == expected


@pytest.mark.parametrize(
    ("input_keys", "expected"),
    [
        (list("Hello"), list("hello")),                    # case fold
        (list("!@#$%^&*()"), list("1234567890")),          # shift-digit fold
        (list('~_+{}|:"<>?'), list("`-=[]\\;',./")),       # shift-punct fold
        (["Key.shift", "Key.shift_l", "Key.shift_r"], []), # shift dropped
        (["⇧", "a", "⇧", "b"], ["a", "b"]),                # unicode shift dropped
    ],
)
def test_charset_compact_clean_keys_folds_inputs(input_keys, expected):
    cs = CharacterSet.qwerty_compact()
    assert cs.clean_keys(input_keys) == expected


def test_cer_perfect_predictions():
    # "hey" / "world" have unique chars per word → labels can pack densely.
    seqs = ("hey", "world")
    target_lengths = torch.tensor([len(s) for s in seqs])
    targets = torch.full((len(seqs), 8), CS.null_class, dtype=torch.long)
    for i, s in enumerate(seqs):
        targets[i, : len(s)] = torch.tensor([CS.key_to_label(c) for c in s])
    y_true = torch.cat([target_lengths.unsqueeze(1), targets], dim=1)

    T_out = 30
    y_pred = torch.full((len(seqs), T_out, CS.num_classes), -100.0)
    for i, s in enumerate(seqs):
        for t, c in enumerate(s):
            y_pred[i, t, CS.key_to_label(c)] = 0.0
        y_pred[i, len(s):, CS.null_class] = 0.0

    metric = CharacterErrorRates(blank_idx=98)
    metric.update(torch.log_softmax(y_pred, dim=-1), y_true)
    assert float(metric.compute()) == 0.0


# ---------------------------------------------------------------------------
# Lightning callbacks (qwerty Python-only API; not YAML-wired)
# ---------------------------------------------------------------------------


class _StubModule:
    training = True
    def __init__(self, model=None):
        self.model = model


class _StubTrainer:
    current_epoch = 5
    world_size = 1


def test_specaugment_callback_attaches_and_detaches():
    pytest.importorskip("torchaudio")
    try:
        from braindecode.models import EMG2QwertyNet
    except ImportError:
        pytest.skip("EMG2QwertyNet missing from installed braindecode")
    from neuralbench.callbacks import SpecAugmentCallback

    model = EMG2QwertyNet(
        n_outputs=99, n_chans=32, n_times=8000, sfreq=2000.0, log_softmax=True
    )
    cb = SpecAugmentCallback(prob=1.0, start_epoch=0)
    cb.on_train_start(_StubTrainer(), _StubModule(model))
    cb.on_train_epoch_start(_StubTrainer(), _StubModule(model))
    assert cb._handle is not None
    model.train()
    assert model(torch.randn(2, 32, 8000)).shape == (2, 373, 99)
    cb.on_train_end(_StubTrainer(), _StubModule(model))
    assert cb._handle is None


def test_band_rotation_module_changes_input_in_train_mode():
    from neuraltrain.augmentations import BandRotation

    x = torch.randn(2, 32, 64)
    aug = BandRotation(
        num_bands=2, electrodes_per_band=16, band_offsets=(-1, 1),
        max_temporal_jitter=4,
    )
    aug.train()
    assert not torch.equal(aug(x), x)


def test_band_rotation_module_is_noop_in_eval_mode():
    from neuraltrain.augmentations import BandRotation

    x = torch.randn(2, 32, 64)
    aug = BandRotation(num_bands=2, electrodes_per_band=16).eval()
    assert torch.equal(aug(x), x)


def test_band_rotation_module_skips_when_channel_layout_mismatches():
    from neuraltrain.augmentations import BandRotation

    x = torch.randn(2, 7, 64)  # 7 channels, not divisible by num_bands*electrodes_per_band
    aug = BandRotation(num_bands=2, electrodes_per_band=16).train()
    assert torch.equal(aug(x), x)


def test_band_rotation_delegates_to_braindecode_functional(monkeypatch):
    """The per-batch math lives in braindecode."""
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


# ---------------------------------------------------------------------------
# BrainModule CTC path: CtcSeqLoss + (y_pred, y_true) metric — no special fork
# ---------------------------------------------------------------------------


def test_brain_module_ctc_loss_and_metric_share_signature():
    pytest.importorskip("Levenshtein")
    from neuraltrain.losses.losses import CtcSeqLoss

    seqs = ("hey", "world")
    target_lengths = torch.tensor([len(s) for s in seqs])
    targets = torch.full((len(seqs), 8), CS.null_class, dtype=torch.long)
    for i, s in enumerate(seqs):
        targets[i, : len(s)] = torch.tensor([CS.key_to_label(c) for c in s])
    y_true = torch.cat([target_lengths.unsqueeze(1), targets], dim=1)

    # (B, T_out, C) log-probs — braindecode convention; same shape both
    # the loss adapter and CER metric consume.
    B, T_out, V = len(seqs), 64, CS.num_classes
    y_pred = torch.log_softmax(torch.randn(B, T_out, V), dim=-1)

    loss = CtcSeqLoss(blank=CS.null_class, zero_infinity=True)
    assert torch.isfinite(loss(y_pred, y_true))

    metric = CharacterErrorRates(blank_idx=CS.null_class)
    metric.update(y_pred, y_true)
    assert float(metric.compute()) >= 0.0


# ---------------------------------------------------------------------------
# Plumbing affected by the CTC integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "target_last_dim", "expected"),
    [(5, 17, 5), (None, 17, 17), (99, 99, 99)],
)
def test_n_outputs_override_routes_to_braindecode_builder(
    monkeypatch, override, target_last_dim, expected
):
    """``n_outputs_override`` (when not None) wins over the inferred
    ``target.shape[-1]`` (CTC targets carry a length prefix)."""
    from neuraltrain.models.base import BaseBrainDecodeModel

    from neuralbench import model_factory

    captured: dict[str, int] = {}

    def fake_build(cfg, wrapper, loader, n_in_channels, n_times, n_outputs):
        captured["n_outputs"] = n_outputs
        return nn.Identity()

    class _FakeSummary:
        total_params = trainable_params = 0

    monkeypatch.setattr(model_factory, "build_braindecode_model", fake_build)
    monkeypatch.setattr(model_factory, "build_dummy_batch", lambda *a, **k: (None, "x"))
    monkeypatch.setattr(model_factory, "init_lazy_layers", lambda *a, **k: None)
    monkeypatch.setattr(model_factory, "summary", lambda *a, **k: _FakeSummary())

    class _Batch: ...
    class _Loader:
        def __iter__(self):
            b = _Batch()
            b.data = {
                "neuro": torch.randn(2, 4, 64),
                "target": torch.zeros(2, target_last_dim),
                "subject_id": torch.zeros(2, dtype=torch.long),
            }
            yield b

        @property
        def dataset(self):
            class _DS:
                metadata: dict = {}
            return _DS()

    from unittest.mock import MagicMock
    cfg = MagicMock(spec=BaseBrainDecodeModel)

    model_factory.build_brain_model(
        brain_model_config=cfg,
        downstream_model_wrapper=None,
        pretrained_weights_fname=None,
        train_loader=_Loader(),  # type: ignore[arg-type]
        n_outputs_override=override,
    )
    assert captured["n_outputs"] == expected


# ---------------------------------------------------------------------------
# Study source: synthetic BIDS tree
# ---------------------------------------------------------------------------


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
        # Bug 4: rstrip("\\n") used to treat the arg as a char-set, eating
        # any trailing 'n' or '\\'.  Fix is exact-suffix match.
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
    """``_download`` delegates to ``neuralfetch.download.Eegdash`` (no IO)."""
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
    """``Study.download`` lands BIDS under ``self.path/download/``."""
    from neuralfetch.studies.emg2qwerty import Emg2qwerty

    sub, ses = "00000002", "0000000002"
    download_root = tmp_path / "download" / f"sub-{sub}" / f"ses-{ses}" / "emg"
    download_root.mkdir(parents=True)
    stem = f"sub-{sub}_ses-{ses}_task-typing"
    (download_root / f"{stem}_emg.bdf").write_bytes(b"\x00" * 16)

    study = Emg2qwerty(path=str(tmp_path))
    assert study._bids_root() == tmp_path / "download"
    assert list(study.iter_timelines()) == [{"subject": sub, "session": ses}]
