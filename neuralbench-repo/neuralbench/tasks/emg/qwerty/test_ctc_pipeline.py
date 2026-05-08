# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained smoke tests for the CTC machinery (synthetic data only)."""

from __future__ import annotations

import importlib
import pathlib

import pytest
import torch
import yaml
from torch import nn

from .charset import CharacterSet
from .extractors import KeystrokeSequence
from .metrics import CharacterErrorRates

CS = CharacterSet.paper()
HERE = pathlib.Path(__file__).parent


@pytest.fixture
def make_keystrokes():
    """Factory: build N Keystroke events, optionally at custom start times."""
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
    ext = KeystrokeSequence(max_target_length=8, event_types="Keystroke")
    events = make_keystrokes(["h", "i", "Key.space"])
    ext.prepare(events)
    out = ext(events, start=0.0, duration=1.0)

    assert out.shape == (9,) and int(out[0]) == 3
    assert out[1:4].tolist() == [CS.key_to_label(k) for k in ("h", "i", "Key.space")]
    assert (out[4:] == CS.null_class).all()


def test_keystroke_sequence_truncation_warns_once(make_keystrokes, caplog):
    ext = KeystrokeSequence(max_target_length=2, event_types="Keystroke")
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
        max_target_length=8, event_types="Keystroke", allow_missing=True
    )
    ext.prepare(make_keystrokes(["a"]))
    out = ext([], start=0.0, duration=1.0)
    assert out.shape == (9,) and int(out[0]) == 0


# ---------------------------------------------------------------------------
# CharacterSet + CER metric
# ---------------------------------------------------------------------------


def test_charset_blank_and_size_invariants():
    assert CS.null_class == 98 == max(CS._key_to_index.values()) + 1
    assert CS.num_classes == 99


# ---------------------------------------------------------------------------
# qwerty_compact preset (case-fold + US-QWERTY shift-fold).
# ---------------------------------------------------------------------------


def test_qwerty_compact_has_51_classes():
    """26 lowercase + 10 digits + 11 unshifted punctuation + 3 modifiers
    (no Shift) + 1 blank = 51 output classes."""
    from .charset import CharacterSet

    cs = CharacterSet.qwerty_compact()
    assert cs.num_classes == 51
    assert cs.null_class == 50
    keys = list(cs.KEY_TO_UNICODE.keys())
    assert len(keys) == 50
    assert "Key.shift" not in keys  # explicitly dropped
    assert "A" not in keys          # uppercase not in vocab
    assert "!" not in keys          # shifted symbol not in vocab


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        # Case fold
        (list("Hello"), list("hello")),
        # Shift-digit fold (! → 1, @ → 2, etc.)
        (list("!@#$%^&*()"), list("1234567890")),
        # Shift-punct fold (~ → `, _ → -, etc.)
        (list('~_+{}|:"<>?'), list("`-=[]\\;',./")),
        # Shift modifiers dropped entirely
        (["Key.shift", "Key.shift_l", "Key.shift_r"], []),
        # Mixed real-world input (the kind a typed sentence produces).
        # ``" "`` normalizes to ``"Key.space"`` via UNICHAR_TO_KEY.
        (
            list("Hello, World!"),
            ["h", "e", "l", "l", "o", ",", "Key.space",
             "w", "o", "r", "l", "d", "1"],
        ),
        # Unicode shift sentinel "⇧" also dropped
        (["⇧", "a", "⇧", "b"], ["a", "b"]),
        # Backspace / Enter / Space retained as in paper preset
        (
            ["Key.backspace", "Key.enter", "Key.space"],
            ["Key.backspace", "Key.enter", "Key.space"],
        ),
    ],
)
def test_qwerty_compact_clean_keys_folds_inputs(inputs, expected):
    """``clean_keys`` collapses uppercase + shifted variants to their
    unshifted base before vocabulary lookup."""
    from .charset import CharacterSet

    cs = CharacterSet.qwerty_compact()
    assert cs.clean_keys(inputs) == expected


def test_qwerty_compact_round_trips_via_extractor(make_keystrokes):
    """End-to-end: ``KeystrokeSequence(vocab_preset='qwerty_compact')``
    produces compact-vocabulary labels via its own per-instance charset
    (no process-global mutation)."""
    from .extractors import KeystrokeSequence

    ext = KeystrokeSequence(
        max_target_length=8, event_types="Keystroke",
        vocab_preset="qwerty_compact",
    )
    events = make_keystrokes(["H", "i", "!"])  # mixed case + shift symbol
    ext.prepare(events)
    out = ext(events, start=0.0, duration=1.0)

    cs = ext._charset
    assert cs.num_classes == 51
    # All three inputs got folded + accepted: H→h, i→i, !→1.
    assert int(out[0]) == 3
    assert out[1:4].tolist() == [
        cs.key_to_label("h"),
        cs.key_to_label("i"),
        cs.key_to_label("1"),
    ]


def test_two_extractors_with_different_presets_dont_clobber_each_other():
    """Bug-3 regression: two ``KeystrokeSequence`` instances with different
    ``vocab_preset`` values must keep their charsets independent.  Used
    to share a process-global that the second-built instance clobbered."""
    from .extractors import KeystrokeSequence

    paper_ext = KeystrokeSequence(
        max_target_length=4, event_types="Keystroke", vocab_preset="paper"
    )
    compact_ext = KeystrokeSequence(
        max_target_length=4, event_types="Keystroke", vocab_preset="qwerty_compact"
    )
    assert paper_ext._charset.num_classes == 99
    assert compact_ext._charset.num_classes == 51
    # Order doesn't matter: building paper after compact still gives 99.
    paper2 = KeystrokeSequence(
        max_target_length=4, event_types="Keystroke", vocab_preset="paper"
    )
    assert paper2._charset.num_classes == 99
    assert compact_ext._charset.num_classes == 51  # unchanged


def test_ctc_metric_factory_builder_binds_to_extractor_charset():
    """The ``qwerty`` task registers a builder that captures the
    extractor's charset; metrics built from two different extractors
    end up with two different charsets."""
    import neuralbench.tasks.emg.qwerty  # noqa: F401  — registers the builder
    from neuralbench.pl_module import get_ctc_metric_factory_builder

    from .extractors import KeystrokeSequence

    builder = get_ctc_metric_factory_builder("qwerty")
    assert builder is not None

    paper_ext = KeystrokeSequence(
        max_target_length=4, event_types="Keystroke", vocab_preset="paper"
    )
    compact_ext = KeystrokeSequence(
        max_target_length=4, event_types="Keystroke", vocab_preset="qwerty_compact"
    )
    paper_metric = builder(paper_ext)()
    compact_metric = builder(compact_ext)()
    assert paper_metric._charset.num_classes == 99
    assert compact_metric._charset.num_classes == 51


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
def test_charset_clean_keys(input_keys, expected):
    cleaned = CS.clean_keys(input_keys)
    assert cleaned == expected
    for k in cleaned:
        assert isinstance(CS.key_to_label(k), int)


def test_cer_perfect_predictions():
    # "hey" / "world" have unique chars per word → no blank separators
    # needed for CTC's collapse-repeats step; we can pack labels densely.
    seqs = ("hey", "world")
    target_lengths = torch.tensor([len(s) for s in seqs])
    targets = torch.full((len(seqs), 8), CS.null_class, dtype=torch.long)
    for i, s in enumerate(seqs):
        targets[i, : len(s)] = torch.tensor([CS.key_to_label(c) for c in s])

    T_out = 30
    emissions = torch.full((T_out, len(seqs), CS.num_classes), -100.0)
    for i, s in enumerate(seqs):
        for t, c in enumerate(s):
            emissions[t, i, CS.key_to_label(c)] = 0.0
        emissions[len(s):, i, CS.null_class] = 0.0

    metric = CharacterErrorRates()
    metric.update(torch.log_softmax(emissions, dim=-1), targets, target_lengths)
    assert float(metric.compute()) == 0.0
    for attr in ("insertions", "deletions", "substitutions"):
        assert int(getattr(metric, attr)) == 0


# ---------------------------------------------------------------------------
# Lightning callbacks (qwerty-task)
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
    from .callbacks import SpecAugmentCallback

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


def test_band_rotation_callback_modifies_neuro_in_place():
    from .callbacks import BandRotationCallback

    x = torch.randn(2, 32, 64)

    class _Batch: ...
    batch = _Batch()
    batch.data = {"neuro": x.clone()}

    BandRotationCallback(
        num_bands=2, electrodes_per_band=16, band_offsets=(-1, 1),
        max_temporal_jitter=4,
    ).on_train_batch_start(None, _StubModule(), batch, 0)
    assert not torch.equal(batch.data["neuro"], x)


def test_band_rotation_callback_respects_start_epoch():
    """``start_epoch`` defers augmentation.  Used by the fine-tune recipe
    to keep the first few epochs clean while the optimizer adapts the
    pretrained head."""
    from .callbacks import BandRotationCallback

    x = torch.randn(2, 32, 64)

    class _Batch: ...
    class _Trainer:
        current_epoch = 0

    cb = BandRotationCallback(
        num_bands=2, electrodes_per_band=16, band_offsets=(-1, 1),
        max_temporal_jitter=4, start_epoch=3,
    )
    trainer = _Trainer()

    # Epochs 0-2: gated → input untouched.
    for trainer.current_epoch in (0, 1, 2):
        batch = _Batch()
        batch.data = {"neuro": x.clone()}
        cb.on_train_batch_start(trainer, _StubModule(), batch, 0)
        assert torch.equal(batch.data["neuro"], x)

    # Epoch 3: gate opens → input mutated.
    trainer.current_epoch = 3
    batch = _Batch()
    batch.data = {"neuro": x.clone()}
    cb.on_train_batch_start(trainer, _StubModule(), batch, 0)
    assert not torch.equal(batch.data["neuro"], x)


def test_band_rotation_callback_delegates_to_braindecode_functional(monkeypatch):
    """The callback's per-batch math lives in braindecode; verify the
    callback actually calls ``braindecode.augmentation.functional.band_rotation``
    so a future braindecode rename / move would break this test."""
    from braindecode.augmentation import functional as bd_functional

    from .callbacks import BandRotationCallback

    captured = {}

    real = bd_functional.band_rotation

    def spy(X, y, **kwargs):
        captured["called"] = True
        captured["kwargs"] = kwargs
        return real(X, y, **kwargs)

    monkeypatch.setattr(bd_functional, "band_rotation", spy)

    x = torch.randn(2, 32, 64)

    class _Batch: ...
    batch = _Batch()
    batch.data = {"neuro": x.clone()}

    BandRotationCallback(
        num_bands=2, electrodes_per_band=16, band_offsets=(-1, 1),
        max_temporal_jitter=4,
    ).on_train_batch_start(None, _StubModule(), batch, 0)

    assert captured.get("called"), "callback did not invoke braindecode.band_rotation"
    assert captured["kwargs"]["num_bands"] == 2
    assert captured["kwargs"]["electrodes_per_band"] == 16
    assert captured["kwargs"]["max_temporal_jitter"] == 4


# ---------------------------------------------------------------------------
# BrainModule CTC dispatch
# ---------------------------------------------------------------------------


def _make_brain_module(ctc_metric_factory):
    from neuraltrain.optimizers import LightningOptimizer
    from neuralbench.pl_module import BrainModule

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, CS.num_classes)
        def forward(self, x):
            return torch.log_softmax(self.linear(x.transpose(1, 2)), dim=-1)

    return BrainModule(
        model=Toy(),
        loss=nn.CTCLoss(blank=CS.null_class, zero_infinity=True),
        metrics={},
        lightning_optimizer_config=LightningOptimizer(
            optimizer={"name": "Adam", "lr": 1e-3, "kwargs": {}},
            scheduler=None, interval="step",
        ),
        ctc_metric_factory=ctc_metric_factory,
    )


def test_brain_module_ctc_step_runs_e2e():
    pytest.importorskip("Levenshtein")
    bm = _make_brain_module(CharacterErrorRates)

    seqs = ("hey", "world")  # B=2; covers padded + length-prefix layout
    target_lengths = torch.tensor([len(s) for s in seqs])
    targets = torch.full((len(seqs), 8), CS.null_class, dtype=torch.long)
    for i, s in enumerate(seqs):
        targets[i, : len(s)] = torch.tensor([CS.key_to_label(c) for c in s])
    y_true = torch.cat([target_lengths.unsqueeze(1), targets], dim=1)

    class _Batch: ...
    batch = _Batch()
    batch.data = {
        "neuro": torch.randn(len(seqs), 4, 64), "target": y_true,
        "subject_id": torch.zeros(len(seqs), dtype=torch.long),
    }
    log_probs = bm.model_forward(batch).transpose(0, 1).contiguous()
    input_lengths = torch.full((len(seqs),), log_probs.shape[0], dtype=torch.long)
    assert torch.isfinite(bm.loss(log_probs, targets, input_lengths, target_lengths))

    metric = bm._get_ctc_metric("val")
    metric.update(log_probs.detach(), targets, target_lengths)
    assert float(metric.compute()) >= 0.0
    for attr in ("insertions", "deletions", "substitutions"):
        assert int(getattr(metric, attr)) >= 0


def test_brain_module_ctc_without_factory_raises():
    bm = _make_brain_module(ctc_metric_factory=None)
    with pytest.raises(RuntimeError, match="ctc_metric_factory"):
        bm._get_ctc_metric("val")


# ---------------------------------------------------------------------------
# Callback config plumbing (BaseCallbackConfig + train_callbacks)
# ---------------------------------------------------------------------------


def test_emg2qwerty_model_yaml_resolves_to_callbacks():
    """The paper recipe lives in models/emg2qwerty.yaml so it follows the
    model wherever it's used; the dataset overlays only carry data-subset
    deltas (query + split_by)."""
    pytest.importorskip("torchaudio")
    from neuralbench.callbacks import BaseCallbackConfig

    model_yaml = HERE.parents[2] / "models" / "emg2qwerty.yaml"
    cfg = yaml.safe_load(model_yaml.read_text())
    cbs = [BaseCallbackConfig.model_validate(c) for c in cfg["train_callbacks"]]
    assert sorted(type(c).__name__ for c in cbs) == ["BandRotation", "SpecAugment"]
    for c in cbs:
        c.build()


@pytest.mark.parametrize(
    ("recipe", "expected_split_by"),
    [("paper_personalized", "session"), ("paper_generic", "subject")],
)
def test_paper_recipe_yaml_carries_split_only(recipe, expected_split_by):
    """From-scratch paper-recipe overlays only carry query + split_by;
    the rest of the recipe lives in shared task / model configs."""
    cfg = yaml.safe_load((HERE / "datasets" / f"{recipe}.yaml").read_text())
    assert cfg["data"]["study"]["split"]["split_by"] == expected_split_by
    assert "query" in cfg["data"]["study"]["source"]
    # No duplicated optimizer / callbacks / batch_size in the overlay.
    assert "lightning_optimizer_config" not in cfg
    assert "train_callbacks" not in cfg


def test_paper_personalized_finetune_overlay_overrides_optimizer_and_callbacks():
    """The fine-tune recipe overrides the from-scratch defaults: lower
    max_lr, shorter warmup, and deferred augmentation start_epochs."""
    cfg = yaml.safe_load(
        (HERE / "datasets" / "paper_personalized_finetune.yaml").read_text()
    )
    opt = cfg["lightning_optimizer_config"]
    assert opt["optimizer"]["lr"] == 1e-5
    assert opt["scheduler"]["kwargs"]["max_lr"] == 1e-5
    assert opt["scheduler"]["kwargs"]["pct_start"] == 0.1

    cbs = {c["name"]: c for c in cfg["train_callbacks"]}
    assert cbs["SpecAugment"]["start_epoch"] == 5
    assert cbs["BandRotation"]["start_epoch"] == 3


def test_basecallbackconfig_field_overrides():
    pytest.importorskip("torchaudio")
    import pydantic
    from neuralbench.callbacks import BaseCallbackConfig
    from .callbacks import BandRotationCallback, SpecAugmentCallback

    class _Holder(pydantic.BaseModel):
        train_callbacks: list[BaseCallbackConfig] = []

    sa, br = _Holder(train_callbacks=[
        {"name": "SpecAugment", "start_epoch": 2, "n_time_masks": 1},
        {"name": "BandRotation", "max_temporal_jitter": 60},
    ]).train_callbacks
    assert (sa.start_epoch, sa.n_time_masks) == (2, 1)
    assert br.max_temporal_jitter == 60
    assert isinstance(sa.build(), SpecAugmentCallback)
    assert isinstance(br.build(), BandRotationCallback)


# ---------------------------------------------------------------------------
# CTC factory registry + neuralbench plumbing
# ---------------------------------------------------------------------------


def test_ctc_metric_registry_lookup():
    """The qwerty package registers a *factory builder* (not the metric
    class itself) so the metric can capture per-extractor charset state."""
    import neuralbench.tasks.emg.qwerty  # noqa: F401 — triggers registration
    from neuralbench.pl_module import get_ctc_metric_factory_builder

    builder = get_ctc_metric_factory_builder("qwerty")
    assert builder is not None
    # Builder bound to a paper-preset extractor produces a paper-preset metric.
    from .extractors import KeystrokeSequence
    ext = KeystrokeSequence(
        max_target_length=4, event_types="Keystroke", vocab_preset="paper"
    )
    metric = builder(ext)()
    assert isinstance(metric, CharacterErrorRates)
    assert metric._charset.num_classes == 99
    assert get_ctc_metric_factory_builder("nope_task") is None


def test_register_ctc_metric_warns_on_overwrite(caplog):
    """Re-registering a different builder under the same task name should
    log a WARNING so silent overwrites in long-running grids stay visible."""
    from neuralbench.pl_module import CtcMetricRegistry

    # Use a fresh registry instance (no shared state with the global one).
    reg = CtcMetricRegistry()
    name = "_test_overwrite_task"
    builder1 = lambda extractor: CharacterErrorRates  # noqa: E731
    builder2 = lambda extractor: CharacterErrorRates  # noqa: E731

    reg.register(name, builder1)
    with caplog.at_level("WARNING"):
        reg.register(name, builder1)
    assert not [r for r in caplog.records if "overwritten" in r.message]
    caplog.clear()
    with caplog.at_level("WARNING"):
        reg.register(name, builder2)
    assert any("overwritten" in r.message for r in caplog.records)


def test_emg2qwerty_raw_rescale_runs_once_per_read(monkeypatch):
    """``Emg2qwertyRaw._read`` multiplies the EMG by 1e6 once per call.
    Two consecutive calls must each see fresh raw values, not a chained
    1e12 / 1e18 / ... rescale."""
    pytest.importorskip("mne")
    import mne
    import numpy as np

    from neuralset.events import etypes

    from .study import Emg2qwertyRaw

    base_value = 1e-5  # ~10 µV in volts (matches BDF native units)
    n_channels, n_times = 2, 16
    sfreq = 2000.0

    def _make_raw():
        info = mne.create_info(
            ch_names=[f"ch{i}" for i in range(n_channels)],
            sfreq=sfreq, ch_types="eeg",  # Emg2qwertyRaw coerces to "emg"
        )
        return mne.io.RawArray(
            np.full((n_channels, n_times), base_value), info, verbose=False
        )

    # Patch the parent ``etypes.Emg._read`` so we don't need a real BDF on
    # disk — every super() call returns a fresh Raw.
    monkeypatch.setattr(etypes.Emg, "_read", lambda self: _make_raw())

    inst = Emg2qwertyRaw(
        filepath="/dev/null", start=0.0, subject="x", timeline="t"
    )
    raw1 = inst._read()
    raw2 = inst._read()
    expected = base_value * Emg2qwertyRaw.BDF_TO_MICROVOLT_SCALE  # 10.0
    np.testing.assert_allclose(raw1.get_data(), expected)
    np.testing.assert_allclose(raw2.get_data(), expected)


def test_maybe_import_task_module_caches_and_skips_missing(tmp_path):
    from neuralbench import experiment_config

    experiment_config._maybe_import_task_module.cache_clear()
    qwerty_dir = pathlib.Path(
        importlib.import_module("neuralbench.tasks.emg.qwerty").__file__
    ).parent
    # Real path imports OK; second call is a cached no-op.
    for _ in range(2):
        experiment_config._maybe_import_task_module("emg", "qwerty", qwerty_dir)

    yaml_only = tmp_path / "fake_task"
    yaml_only.mkdir()
    experiment_config._maybe_import_task_module("emg", "fake_task", yaml_only)


@pytest.mark.parametrize(
    ("override", "target_last_dim", "expected"),
    [(5, 17, 5), (None, 17, 17), (99, 99, 99)],
)
def test_n_outputs_override_routes_to_braindecode_builder(
    monkeypatch, override, target_last_dim, expected
):
    """``n_outputs_override`` (when not None) reaches ``build_braindecode_model``
    instead of the inferred ``target.shape[-1]``.  Verified by monkeypatching
    the inner builder + summary call so we don't need a real braindecode
    config."""
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

    # Mock with ``spec=`` so ``isinstance(cfg, BaseBrainDecodeModel)`` is
    # True without actually constructing a config (which would pull in
    # optional braindecode deps unrelated to this test).
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
    # Intentional 16-byte stub — iter_timelines / _bids_paths only stat
    # for existence; no test in this module reads BDF content.  If a
    # future test needs it, write a real header via ``mne.export``.
    (emg_dir / f"{stem}_emg.bdf").write_bytes(b"\x00" * 16)
    (emg_dir / f"{stem}_events.tsv").write_text(
        "onset\tduration\tvalue\tprompt_text\tkey\n"
        "0.10\t1.5\tprompt\thello\t\n"
        "0.20\t0.05\tkeystroke_press\t\th\n"
        "0.30\t0.05\tkeystroke_press\t\te\n"
        "0.40\t0.05\tkeystroke_press\t\tKey.space\n"
    )
    return tmp_path, sub, ses


def test_ctrllabs_iter_timelines(bids_tree):
    from .study import Emg2qwerty
    root, sub, ses = bids_tree
    assert list(Emg2qwerty(path=str(root)).iter_timelines()) == [
        {"subject": sub, "session": ses}
    ]


def test_ctrllabs_load_timeline_events(bids_tree):
    from .study import Emg2qwerty
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
def test_ctrllabs_bids_id_validation_rejects_unsafe(bids_tree, subject, session):
    from .study import Emg2qwerty
    root, _, _ = bids_tree
    with pytest.raises(ValueError, match="unsafe BIDS id"):
        Emg2qwerty(path=str(root))._bids_paths(subject, session)


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        # The bug: prior code did rstrip("\\n") which strips ANY trailing
        # 'n' or '\' chars (treats arg as a char-set, not a suffix), so
        # "fun\\n" silently became "fu" and "running\\n" became "running"
        # by chance (correct).  After the fix, only the literal
        # 2-character "\n" suffix is removed.
        (r"fun\n", "fun"),         # literal backslash-n suffix → strip
        (r"running\n", "running"),  # trailing 'n' MUST NOT be eaten
        ("hello", "hello"),         # no suffix → unchanged
        (r"\n\n", r"\n"),           # only one occurrence stripped (matches doc)
    ],
)
def test_load_timeline_events_prompt_text_strips_only_literal_suffix(
    bids_tree, raw_text, expected,
):
    """Bug-4 regression: prompt_text suffix stripping is exact-match,
    not character-set."""
    from .study import Emg2qwerty

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
    """``_download`` delegates to ``neuralfetch.download.Eegdash`` with the
    NEMAR dataset id; no real network IO."""
    from unittest import mock

    import neuralfetch.download as dl

    from .study import Emg2qwerty

    with mock.patch.object(dl, "Eegdash") as patched:
        Emg2qwerty(path="/tmp/_nbqwerty_test")._download()

    patched.assert_called_once_with(
        study="nm000104", dset_dir=pathlib.Path("/tmp/_nbqwerty_test")
    )
    patched.return_value.download.assert_called_once()


def test_emg2qwerty_bids_root_handles_download_subfolder(tmp_path):
    """``Study.download`` lands BIDS under ``self.path/download/``;
    ``_bids_root`` must find it there when the direct path is empty."""
    from .study import Emg2qwerty

    sub, ses = "00000002", "0000000002"
    download_root = tmp_path / "download" / f"sub-{sub}" / f"ses-{ses}" / "emg"
    download_root.mkdir(parents=True)
    stem = f"sub-{sub}_ses-{ses}_task-typing"
    (download_root / f"{stem}_emg.bdf").write_bytes(b"\x00" * 16)

    study = Emg2qwerty(path=str(tmp_path))
    assert study._bids_root() == tmp_path / "download"
    assert list(study.iter_timelines()) == [{"subject": sub, "session": ses}]


# ---------------------------------------------------------------------------
# Pretrained-weight loading: upstream emg2qwerty checkpoint → braindecode
# EMG2QwertyNet via ``load_checkpoint`` + ``EMG2QwertyNet.mapping``.
# ---------------------------------------------------------------------------


def test_load_checkpoint_applies_model_mapping_before_key_match(tmp_path, caplog):
    """Generic regression for the ``load_checkpoint`` rename hook.

    Models can declare a ``mapping`` ClassVar (e.g. EMG2QwertyNet maps the
    upstream emg2qwerty TDS classifier head ``model.4.{weight,bias}`` →
    ``final_layer.{weight,bias}``).  The loader must consult that mapping
    on every backend (``.ckpt`` / ``.pt`` / ``.safetensors``), not just
    safetensors, otherwise upstream-shaped state_dicts silently fail to
    load the renamed parameters.
    """
    import logging

    from neuralbench.utils import load_checkpoint

    # Tiny model that mimics EMG2QwertyNet's mapping shape: an inner
    # ``model.0.linear`` plus a separate ``final_layer``.  Source keys
    # (``model.4.{weight,bias}``) get renamed to the model-side names
    # via ``mapping``; the other keys must round-trip unchanged.
    class _Toy(nn.Module):
        mapping = {
            "model.4.weight": "final_layer.weight",
            "model.4.bias": "final_layer.bias",
        }

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(4, 4))  # → model.0.weight/bias
            self.final_layer = nn.Linear(4, 2)

    target = _Toy()
    truth_sd = target.state_dict()  # has model.0.* + final_layer.*

    # Upstream shape: rename final_layer.* → model.4.* and wrap in PL's
    # ``state_dict`` envelope.
    upstream_sd: dict[str, torch.Tensor] = {}
    for k, v in truth_sd.items():
        if k == "final_layer.weight":
            upstream_sd["model.4.weight"] = v.clone()
        elif k == "final_layer.bias":
            upstream_sd["model.4.bias"] = v.clone()
        else:
            upstream_sd[k] = v.clone()
    ckpt_path = tmp_path / "upstream.ckpt"
    torch.save({"state_dict": upstream_sd}, ckpt_path)

    fresh = _Toy()
    pre_final_w = fresh.final_layer.weight.detach().clone()

    with caplog.at_level(logging.INFO, logger="test_logger"):
        loaded = load_checkpoint(fresh, ckpt_path, logging.getLogger("test_logger"))

    # Renamed head landed in the right place …
    assert torch.allclose(loaded.final_layer.weight, truth_sd["final_layer.weight"])
    assert torch.allclose(loaded.final_layer.bias, truth_sd["final_layer.bias"])
    assert not torch.equal(loaded.final_layer.weight, pre_final_w)
    # … and the unrenamed key was populated too.
    assert torch.allclose(
        loaded.state_dict()["model.0.weight"], truth_sd["model.0.weight"]
    )
    # Loader logged zero missing keys — full coverage.  The loader uses
    # ``sorted(set(...))`` which renders empty as ``[]``, not ``set()``.
    missing_logs = [
        r for r in caplog.records if "Missing keys" in r.getMessage()
    ]
    assert missing_logs and missing_logs[-1].getMessage().endswith(": []"), (
        f"expected zero missing keys, got: "
        f"{missing_logs[-1].getMessage() if missing_logs else None}"
    )


def test_load_upstream_emg2qwerty_checkpoint_round_trips(tmp_path):
    """End-to-end version of the test above against a real
    ``EMG2QwertyNet``.  Skips when braindecode's wheel-installed copy
    doesn't expose ``EMG2QwertyNet`` (the in-repo editable install does)."""
    try:
        from braindecode.models import EMG2QwertyNet
    except ImportError:
        pytest.skip("EMG2QwertyNet missing from installed braindecode")

    import logging

    from neuralbench.utils import load_checkpoint

    src = EMG2QwertyNet(
        n_outputs=99, n_chans=32, n_times=8000, sfreq=2000.0, log_softmax=True
    )
    sd = src.state_dict()
    upstream_sd = {
        ("model.4.weight" if k == "final_layer.weight"
         else "model.4.bias" if k == "final_layer.bias"
         else k): v.clone()
        for k, v in sd.items()
    }
    ckpt_path = tmp_path / "upstream.ckpt"
    torch.save({"state_dict": upstream_sd}, ckpt_path)

    dst = EMG2QwertyNet(
        n_outputs=99, n_chans=32, n_times=8000, sfreq=2000.0, log_softmax=True
    )
    load_checkpoint(dst, ckpt_path, logging.getLogger("test_logger"))
    for k, v in sd.items():
        assert torch.allclose(dst.state_dict()[k], v), f"mismatch on {k}"


def test_load_checkpoint_safetensors_with_lightning_keys_loads_backbone(tmp_path):
    """Bug-1 regression: a safetensors checkpoint whose keys still carry
    the Lightning ``model.`` prefix used to silently drop the backbone
    because the safetensors path applied ``mapping`` early, creating a
    partial overlap that disabled the auto-prefix step.  After the fix,
    mapping runs once after strip and per-key auto-prefix recovers the
    backbone keys."""
    pytest.importorskip("safetensors")
    import logging

    from safetensors.torch import save_file

    from neuralbench.utils import load_checkpoint

    class _Toy(nn.Module):
        # Mimic the EMG2QwertyNet shape: an inner ``model.0.linear`` plus a
        # separate ``final_layer``; declare a Lightning-side mapping.
        mapping = {
            "model.4.weight": "final_layer.weight",
            "model.4.bias": "final_layer.bias",
        }

        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(nn.Linear(4, 4))
            self.final_layer = nn.Linear(4, 2)

    src = _Toy()
    truth = src.state_dict()  # has model.0.weight + final_layer.weight, etc.

    # Lightning-prefixed safetensors checkpoint: model.0.weight + model.4.weight.
    upstream = {
        "model.0.weight": truth["model.0.weight"].clone(),
        "model.0.bias": truth["model.0.bias"].clone(),
        "model.4.weight": truth["final_layer.weight"].clone(),
        "model.4.bias": truth["final_layer.bias"].clone(),
    }
    sf_path = tmp_path / "upstream.safetensors"
    save_file(upstream, str(sf_path))

    dst = _Toy()
    load_checkpoint(dst, sf_path, logging.getLogger("test_logger"))

    for k, v in truth.items():
        assert torch.allclose(dst.state_dict()[k], v), (
            f"backbone key {k} not loaded — auto-prefix regression"
        )


def test_load_checkpoint_rejects_duplicate_mapping_targets(tmp_path):
    """Bug-2 regression: mapping with two source keys aliasing the same
    target key produces silent clobber depending on dict-iteration order.
    Loader must fail loudly."""
    import logging

    from neuralbench.utils import load_checkpoint

    class _Bad(nn.Module):
        mapping = {
            "src1.weight": "final_layer.weight",
            "src2.weight": "final_layer.weight",  # duplicate target
        }

        def __init__(self):
            super().__init__()
            self.final_layer = nn.Linear(2, 2)

    ckpt_path = tmp_path / "x.ckpt"
    torch.save({"state_dict": {"src1.weight": torch.zeros(2, 2)}}, ckpt_path)

    with pytest.raises(ValueError, match="duplicate target keys"):
        load_checkpoint(_Bad(), ckpt_path, logging.getLogger("t"))
