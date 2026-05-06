# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the classical sklearn/pyriemann baselines."""

import logging

import numpy as np
import pydantic
import pytest
import torch
from pytest_mock import MockerFixture
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from neuraltrain.models.base import BaseModelConfig

from .sklearn_baseline import (
    CoSpectraLogLR,
    CovTsLR,
    CovTsRidge,
    SklearnBaseline,
    SklearnBaselineModel,
    SklearnClassifier,
    XdawnTsLR,
    _UpperTriLog1p,
)

# Keep synthetic datasets small so pyriemann stays well below its PD-stability
# limits and the tests run fast.
_N = 80
_C = 8
_T = 128
_SEED = 0

_ALL_LEAVES: list[type[SklearnBaseline]] = [
    XdawnTsLR,
    CovTsLR,
    CoSpectraLogLR,
    CovTsRidge,
]


# ---------------------------------------------------------------------------
# Synthetic signals
# ---------------------------------------------------------------------------
# Each leaf gets a signal that exercises its feature extractor: a per-class
# channel-mean offset for the covariance-based classifiers, a per-class
# narrow-band sinusoid for the spectral classifier, and a class-independent
# gaussian for the regressor.  ``_synth_for`` routes between them so the
# parametrized tests below don't need per-leaf ``if/elif`` chains.


def _make_class_signal(
    n: int = _N, n_classes: int = 2, seed: int = _SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Class-conditional channel-mean offset for covariance-based classifiers."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, _C, _T)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n).astype(np.int64)
    for k in range(n_classes):
        X[y == k, 0, :] += 0.5 * k
    return X, y


def _make_reg_signal(
    n: int = _N, d: int = 1, seed: int = _SEED
) -> tuple[np.ndarray, np.ndarray]:
    """White-noise regression signal (used for shape/dtype checks only)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, _C, _T)).astype(np.float32)
    shape: tuple[int, ...] = (n,) if d == 1 else (n, d)
    y = rng.standard_normal(shape).astype(np.float32)
    return X, y


def _make_ssvep_signal(
    n: int = _N,
    n_classes: int = 3,
    seed: int = _SEED,
    sfreq: float = 120.0,
    duration: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class narrow-band sinusoid (``8 + 3*k`` Hz) for spectral classifiers.

    Uses a 4 s / 120 Hz window so ``CoSpectra`` has enough samples for at
    least one full Welch window at ``window=128``.
    """
    rng = np.random.default_rng(seed)
    n_times = int(sfreq * duration)
    t = np.arange(n_times) / sfreq
    X = rng.standard_normal((n, _C, n_times)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n).astype(np.int64)
    for k in range(n_classes):
        freq_hz = 8.0 + 3.0 * k
        X[y == k, 0, :] += 2.0 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    return X, y


def _make_reg_signal_with_variance_modulation(
    n: int = _N, seed: int = _SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Regression signal where channel-0 variance monotonically tracks ``y``.

    This shows up in the covariance matrix as a y-correlated ``cov[0, 0]``
    entry, which ``CovTsRidge`` should be able to recover.  Guards against
    the failure mode where a fixed-``alpha`` Ridge on un-standardized
    tangent-space features collapses to a constant prediction
    (``std(y_pred) -> 0``, Pearson r -> 0).
    """
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((n,)).astype(np.float32)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
    amp = (0.5 + 2.0 * y_norm).astype(np.float32)
    X = rng.standard_normal((n, _C, _T)).astype(np.float32)
    X[:, 0, :] *= amp[:, None]
    return X, y


def _synth_for(
    cls: type[SklearnBaseline], *, out_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a synthetic ``(X, y)`` matched to *cls*'s feature extractor."""
    if cls is CoSpectraLogLR:
        return _make_ssvep_signal(n_classes=out_dim)
    if cls._kind == "regressor":
        return _make_reg_signal(d=out_dim)
    return _make_class_signal(n_classes=out_dim)


def _unwrap_head(pipe: Pipeline) -> object:
    """Return the final estimator, unwrapping a :class:`OneVsRestClassifier`."""
    head = pipe.steps[-1][1]
    return head.estimator if isinstance(head, OneVsRestClassifier) else head


# ---------------------------------------------------------------------------
# Per-leaf fit / forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,kind,out_dim",
    [
        (XdawnTsLR, "classifier", 2),
        (CovTsLR, "classifier", 3),
        (CoSpectraLogLR, "classifier", 3),
        (CovTsRidge, "regressor", 1),
        (CovTsRidge, "regressor", 3),
    ],
)
def test_leaf_fit_forward(cls: type[SklearnBaseline], kind: str, out_dim: int) -> None:
    """Every concrete leaf fits on synthetic data and forwards batched tensors."""
    X, y = _synth_for(cls, out_dim=out_dim)
    model = cls().build(X, y)

    assert isinstance(model, SklearnBaselineModel)
    assert model.output_dim == out_dim

    out = model(torch.from_numpy(X[:5]))
    assert out.shape == (5, out_dim)
    assert out.dtype == torch.float32
    if kind == "classifier":
        assert torch.allclose(out.sum(dim=-1), torch.ones(5), atol=1e-5)


@pytest.mark.parametrize(
    "cls,expected_head",
    [
        (XdawnTsLR, LogisticRegressionCV),
        (CovTsLR, LogisticRegressionCV),
        (CoSpectraLogLR, LogisticRegressionCV),
        (CovTsRidge, RidgeCV),
    ],
)
def test_pipeline_includes_standard_scaler_and_cv_head(
    cls: type[SklearnBaseline], expected_head: type
) -> None:
    """Every leaf places a ``StandardScaler`` immediately before a CV head.

    The standardization step is load-bearing for regression -- without it a
    fixed-``alpha`` Ridge on tangent-space features collapses toward the
    training mean on real EEG, producing Pearson r ~ 0.
    """
    X, y = _synth_for(cls, out_dim=1 if cls._kind == "regressor" else 2)
    pipe = cls().build(X, y)._pipe

    assert isinstance(pipe.steps[-2][1], StandardScaler), (
        f"{cls.__name__}: StandardScaler must be the step immediately before the head; "
        f"got steps {[type(s) for _, s in pipe.steps]}."
    )
    head = _unwrap_head(pipe)
    assert isinstance(head, expected_head), (
        f"{cls.__name__} head is {type(head).__name__}, expected {expected_head.__name__}"
    )


def test_covtsridge_recovers_variance_signal() -> None:
    """Guard against the Pearson-r ~ 0 failure mode on regression tasks.

    Fits ``CovTsRidge`` on a signal where channel-0 variance monotonically
    tracks ``y``, then asserts predictions are (a) not constant and (b)
    positively correlated with the ground truth on the fit set.  A linear
    Ridge on tangent-space features collapsed to a constant (pre-fix) would
    fail both checks.
    """
    X, y = _make_reg_signal_with_variance_modulation(n=120, seed=_SEED)
    model = CovTsRidge().build(X, y)
    preds = model(torch.from_numpy(X)).squeeze(-1).numpy()

    assert float(preds.std()) > 1e-3, (
        f"CovTsRidge predictions collapsed to a constant (std={preds.std():.2e})."
    )
    r = float(np.corrcoef(preds, y)[0, 1])
    assert r > 0.3, f"CovTsRidge Pearson r={r:.3f} is too low; expected > 0.3."


def test_cospectra_log_lr_recovers_ssvep_peak() -> None:
    """``CoSpectraLogLR`` must pick up a class-specific narrow-band oscillation.

    Injects a distinct sinusoid per class and checks that the fitted pipeline
    classifies held-out synthetic trials well above chance.  Guards against
    regressions in the ``CoSpectra`` ``fs``/``fmin``/``fmax`` plumbing and in
    the ``_UpperTriLog1p`` flattening.
    """
    X_fit, y_fit = _make_ssvep_signal(n=120, n_classes=3, seed=_SEED)
    X_test, y_test = _make_ssvep_signal(n=60, n_classes=3, seed=_SEED + 1)

    model = CoSpectraLogLR().build(X_fit, y_fit)
    probs = model(torch.from_numpy(X_test)).numpy()
    preds = probs.argmax(axis=-1)
    acc = float((preds == y_test).mean())
    assert acc > 0.7, (
        f"CoSpectraLogLR accuracy on synthetic SSVEP is {acc:.2f}; expected > 0.7 "
        f"(chance = 1/3)."
    )


def test_upper_tri_log1p_small_example() -> None:
    """``_UpperTriLog1p`` log1p's the diagonal, keeps off-diagonal raw, and
    flattens ``(N, C, C, F)`` to ``(N, F * C*(C+1)/2)``."""
    # One trial, 2x2 symmetric co-spectral matrix, 2 frequency bins.
    # Diagonals (power): bin 0 -> [3, 7];  bin 1 -> [0, e - 1].
    # Off-diagonal (cross-spectrum): bin 0 -> 2;  bin 1 -> -0.5.
    X = np.array(
        [[[[3.0, 0.0], [2.0, -0.5]], [[2.0, -0.5], [7.0, np.e - 1.0]]]],
        dtype=np.float64,
    )  # shape (1, 2, 2, 2)

    feats = _UpperTriLog1p().transform(X)

    # Upper-tri order from ``np.triu_indices(2)`` is (0,0), (0,1), (1,1).
    # Flattening ``(N, K, F) -> (N, K*F)`` in C-order interleaves frequency
    # within each upper-tri entry: [diag0(f0), diag0(f1), off(f0), off(f1),
    # diag1(f0), diag1(f1)].
    expected = np.array([[np.log1p(3.0), np.log1p(0.0), 2.0, -0.5, np.log1p(7.0), 1.0]])
    np.testing.assert_allclose(feats, expected, rtol=1e-7, atol=1e-12)


# ---------------------------------------------------------------------------
# Target normalisation paths (one-hot collapse, multilabel detection)
# ---------------------------------------------------------------------------


def test_normalise_targets_variants() -> None:
    """``SklearnClassifier._normalise_targets`` handles 1D, one-hot, and multilabel."""
    cfg = CovTsLR()
    n_classes = 3

    y_1d = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    y_norm, info = cfg._normalise_targets(y_1d)
    assert y_norm.ndim == 1 and y_norm.dtype == np.int64
    assert info == {"n_classes": n_classes, "is_multilabel": False}

    y_onehot = np.eye(n_classes, dtype=np.int64)[y_1d]
    y_norm, info = cfg._normalise_targets(y_onehot)
    assert y_norm.ndim == 1
    assert info == {"n_classes": n_classes, "is_multilabel": False}

    y_multilabel = np.zeros((4, n_classes), dtype=np.int64)
    y_multilabel[0] = [1, 1, 0]  # >1 active label forces multilabel
    _, info = cfg._normalise_targets(y_multilabel)
    assert info == {"n_classes": n_classes, "is_multilabel": True}


@pytest.mark.parametrize(
    "cls,expect",
    [(CovTsLR, "ovr"), (XdawnTsLR, "raise")],
)
def test_multilabel_handling(cls: type[SklearnClassifier], expect: str) -> None:
    """``CovTsLR`` wraps the head in ``OneVsRestClassifier``; ``XdawnTsLR`` rejects."""
    rng = np.random.default_rng(_SEED)
    n_classes = 3
    X, _ = _make_class_signal(n_classes=n_classes)
    y = (rng.random((_N, n_classes)) > 0.5).astype(np.int64)
    y[0] = [1, 1, 0]
    y[1] = [0, 0, 0]

    if expect == "raise":
        with pytest.raises(ValueError, match="multilabel targets are not supported"):
            cls().build(X, y)
        return

    model = cls().build(X, y)
    assert isinstance(model, SklearnBaselineModel)
    pipe = model._pipe
    assert isinstance(pipe, Pipeline)
    assert isinstance(pipe.steps[-1][1], OneVsRestClassifier)
    assert model(torch.from_numpy(X[:4])).shape == (4, n_classes)


# ---------------------------------------------------------------------------
# Shared fit-time plumbing
# ---------------------------------------------------------------------------


def test_drop_flat_trials(caplog: pytest.LogCaptureFixture) -> None:
    X, y = _make_class_signal(n_classes=2)
    X[:5] = 0.0

    caplog.set_level(logging.INFO, logger="neuralbench.sklearn_baseline")
    X_kept, y_kept = CovTsLR()._drop_flat_trials(X, y)

    assert X_kept.shape[0] == _N - 5
    assert y_kept.shape[0] == _N - 5
    assert "dropping 5/" in caplog.text


@pytest.mark.parametrize(
    "mode,cap,expected",
    [("stratified", 60, 60), ("uniform", 40, 40), ("disabled", 500, 200)],
)
def test_maybe_subsample(mode: str, cap: int, expected: int) -> None:
    """Stratified for multiclass, uniform for regression, no-op when below cap."""
    if mode == "uniform":
        X, y = _make_reg_signal(n=200)
        cfg: SklearnBaseline = CovTsRidge(max_fit_samples=cap, random_state=7)
    else:
        X, y = _make_class_signal(n=200, n_classes=3)
        cfg = CovTsLR(max_fit_samples=cap, random_state=123)

    X_sub, y_sub = cfg._maybe_subsample(X, y)
    assert X_sub.shape[0] == expected
    assert y_sub.shape[0] == expected

    if mode == "stratified":
        orig = np.bincount(y, minlength=3)
        new = np.bincount(y_sub, minlength=3)
        # Allow a few trials of slack (``_maybe_subsample`` rounds up to hit cap).
        assert np.all(np.abs(new - (cap * orig // 200)) <= 3)


def test_pd_fallback_retries_with_logeuclid(
    caplog: pytest.LogCaptureFixture, mocker: MockerFixture
) -> None:
    """The first fit raises a PD error; the fallback metric should succeed."""
    X, y = _make_class_signal(n_classes=2)
    real_fit = Pipeline.fit
    n_calls = {"n": 0}

    def flaky_fit(self: Pipeline, X_: np.ndarray, y_: np.ndarray) -> Pipeline:
        n_calls["n"] += 1
        if n_calls["n"] == 1:
            raise ValueError("Matrices must be positive definite")
        return real_fit(self, X_, y_)

    mocker.patch.object(Pipeline, "fit", flaky_fit)

    caplog.set_level(logging.WARNING, logger="neuralbench.sklearn_baseline")
    model = CovTsLR().build(X, y)
    assert isinstance(model, SklearnBaselineModel)
    assert n_calls["n"] == 2
    assert "logeuclid" in caplog.text


def test_pd_fallback_reraises_non_pd_errors(mocker: MockerFixture) -> None:
    """Errors that don't mention 'positive definite' bypass the retry loop."""
    X, y = _make_class_signal(n_classes=2)
    mocker.patch.object(Pipeline, "fit", side_effect=ValueError("unrelated failure"))
    with pytest.raises(ValueError, match="unrelated failure"):
        CovTsLR().build(X, y)


# ---------------------------------------------------------------------------
# Leaf-specific behaviour
# ---------------------------------------------------------------------------


def test_xdawn_target_class_auto_uses_minority(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rng = np.random.default_rng(_SEED)
    # Imbalanced binary task: class 1 is the minority (~17%).
    X = rng.standard_normal((120, _C, _T)).astype(np.float32)
    y = np.zeros(120, dtype=np.int64)
    y[:20] = 1
    X[y == 1, 0, :] += 0.8
    rng.shuffle(y)

    caplog.set_level(logging.INFO, logger="neuralbench.sklearn_baseline")
    XdawnTsLR(xdawn_target_class=-1).build(X, y)
    assert "auto-detected Xdawn target class = 1" in caplog.text


# ---------------------------------------------------------------------------
# Pydantic / discriminator wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", _ALL_LEAVES)
def test_discriminator_roundtrip(cls: type[SklearnBaseline]) -> None:
    """Leaves are reachable via the shared ``BaseModelConfig`` discriminator."""
    cfg = BaseModelConfig(name=cls.__name__)  # type: ignore[call-arg]
    assert isinstance(cfg, cls)


def test_leaf_rejects_unknown_fields() -> None:
    """``extra='forbid'`` rejects YAML keys that belong to a sibling leaf."""
    with pytest.raises(pydantic.ValidationError):
        CovTsLR(xdawn_nfilter=4)  # type: ignore[call-arg]
