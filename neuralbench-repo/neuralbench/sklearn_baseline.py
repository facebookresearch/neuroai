# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Classical sklearn-based baselines for neuralbench tasks.

These baselines are fit once at model-build time from the training (optionally
train+val) set and wrapped as a ``torch.nn.Module`` so they flow through the
standard Lightning eval loop (metrics, logging, caching).  Intended to be used
with ``eval_only: true`` to skip the training loop entirely.

Supported pipelines
-------------------
``XdawnTsLR``
    ``XdawnCovariances -> TangentSpace -> StandardScaler -> LogisticRegressionCV``.
    Classical ERP / P300 classification convention (moabb).
``CovTsLR``
    ``Covariances -> TangentSpace -> StandardScaler -> LogisticRegressionCV``.
    Classical oscillatory / motor-imagery classification convention (moabb
    ``TSLR.yml``).  Keeps the full tangent-space feature vector, which
    strictly dominates CSP+LR (log-variance components) on the same
    covariance features.
``CoSpectraLogLR``
    ``CoSpectra -> upper-tri log1p flatten -> StandardScaler ->
    LogisticRegressionCV``.  Spectral classification baseline (SSVEP / steady
    state).  Unlike the other pyriemann leaves it linearises the co-spectral
    matrices with a plain ``log1p`` on the diagonal (per-channel PSD) rather
    than a Riemannian tangent space, trading geometric symmetry for a much
    simpler pipeline with no frequency-specific plumbing.
``CovTsRidge``
    ``Covariances -> TangentSpace -> StandardScaler -> RidgeCV``.
    Continuous-target regression baseline (e.g. age prediction); handles both
    univariate and multi-output targets via sklearn's native multi-output
    Ridge.

The ``StandardScaler`` step puts all tangent-space features on a common scale
so the linear head's regularization penalty is not dominated by the most
heavily-scaled (typically diagonal) entries.  The ``CV`` heads pick the
regularizer via LOO (Ridge, closed form) or k-fold (LogReg) so behaviour is
stable across tasks/modalities without per-task hyper-tuning.

This module mirrors the :class:`neuraltrain.models.dummy_predictor.DummyPredictor`
pattern: a pydantic :class:`BaseModelConfig` subclass whose ``build`` consumes
the training data once and returns an eval-only ``nn.Module``.
"""

import logging
import typing as tp

import numpy as np
import torch
from pyriemann.estimation import CoSpectra, Covariances, Shrinkage, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from neuraltrain.models.base import BaseModelConfig

LOGGER = logging.getLogger(__name__)

# Soft threshold for logging a memory warning; no hard cap.
_MEMORY_WARN_BYTES = 5 * 1024**3  # 5 GiB

# Maximum number of training trials used to fit the classical pipelines.  The
# iterative Riemannian mean (``mean_riemann``) is numerically fragile at very
# large N (observed failure at N=223k on the multilabel artifact task with
# OAS-shrunk covariances), and the linear classifier head is already saturated
# well below this cap.  Large datasets are uniformly subsampled (stratified by
# label where possible) prior to ``pipe.fit``.
_DEFAULT_MAX_FIT_SAMPLES = 20_000


class SklearnBaseline(BaseModelConfig):
    """Abstract base for classical sklearn/pyriemann baselines.

    Subclasses implement :meth:`_normalise_targets` and :meth:`_build_pipeline`;
    this class owns the shared fit-time plumbing (flat-trial dropping,
    subsampling, PD-fallback metric retry), the ``build`` template method,
    and the shared covariance-pipeline hyperparameters.

    Parameters
    ----------
    max_fit_samples
        Cap on the number of training trials passed to ``pipe.fit``.  Larger
        datasets are uniformly (or stratified) subsampled; set ``<= 0`` to
        disable the cap.
    random_state
        Seed used for stratified / uniform subsampling.
    min_trial_variance
        Trials whose variance (over channels and time) is ``<=`` this threshold
        are dropped before fitting, because OAS-shrunk covariances of a flat
        trial are still all-zero (not PD) and would break ``logm`` downstream.
        Set to ``<= 0`` to disable flat-trial dropping.
    cov_estimator
        Covariance shrinkage estimator passed to
        :class:`pyriemann.estimation.Covariances` / :class:`XdawnCovariances`
        (``"oas"``, ``"lwf"``, ``"scm"``, ...).
    ts_metric
        Riemannian metric for :class:`pyriemann.tangentspace.TangentSpace`
        (``"riemann"``, ``"logeuclid"``, ``"euclid"``, ...).  ``"riemann"``
        matches the moabb ``TSLR.yml`` pipeline.
    cov_shrinkage
        Additive covariance regularization for numerical stability (fraction
        of the mean diagonal element, applied via
        :class:`pyriemann.estimation.Shrinkage`).
    """

    max_fit_samples: int = _DEFAULT_MAX_FIT_SAMPLES
    random_state: int = 0
    min_trial_variance: float = 1e-8
    cov_estimator: str = "oas"
    ts_metric: str = "riemann"
    cov_shrinkage: float = 0.1

    # Each kind-specific base sets this to ``"classifier"`` / ``"regressor"``.
    _kind: tp.ClassVar[tp.Literal["classifier", "regressor"]]

    def build(  # type: ignore[override]
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "SklearnBaselineModel":
        """Fit the underlying sklearn pipeline and wrap as ``nn.Module``."""
        # Keep ``X`` in its incoming dtype (typically f32) through the
        # flat-trial drop and subsampling steps so we don't transiently
        # double its footprint.  The f32 -> f64 widen is deferred until
        # after ``_maybe_subsample`` caps N at ``max_fit_samples``, making
        # the unavoidable copy bounded.  pyriemann's ``Covariances``
        # promotes f32 inputs internally, so the early pipeline stages are
        # unaffected.
        X = np.asarray(X_train)
        y = (
            y_train.detach().cpu().numpy()
            if isinstance(y_train, torch.Tensor)
            else np.asarray(y_train)
        )
        if X.nbytes > _MEMORY_WARN_BYTES:
            LOGGER.warning(
                "SklearnBaseline: X_train is %.1f GiB (> 5 GiB). "
                "Classical pipeline may need substantial RAM; proceeding anyway.",
                X.nbytes / 1024**3,
            )

        X, y = self._drop_flat_trials(X, y)
        X, y = self._maybe_subsample(X, y)
        X = X.astype(np.float64, copy=False)

        y_norm, target_info = self._normalise_targets(y)
        LOGGER.info(
            "SklearnBaseline: fitting %s on X=%s y=%s (%s)...",
            type(self).__name__,
            X.shape,
            y_norm.shape,
            self._describe_target(target_info),
        )
        pipe = self._fit_with_fallback(
            X,
            y_norm,
            lambda ts_metric: self._build_pipeline(
                y=y_norm, target_info=target_info, ts_metric=ts_metric
            ),
        )
        output_dim = self._compute_output_dim(y_norm, target_info)
        LOGGER.info(
            "SklearnBaseline: fit complete (kind=%s, output_dim=%d).",
            self._kind,
            output_dim,
        )
        return SklearnBaselineModel(pipe, kind=self._kind, output_dim=output_dim)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _normalise_targets(
        self, y_train: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray, dict[str, tp.Any]]:
        """Convert raw targets to sklearn-friendly form.

        Returns ``(y, info)`` where ``info`` is forwarded to
        :meth:`_build_pipeline` and :meth:`_compute_output_dim`.
        """
        raise NotImplementedError

    def _build_pipeline(
        self,
        *,
        y: np.ndarray,
        target_info: dict[str, tp.Any],
        ts_metric: str,
    ) -> Pipeline:
        """Build (but do not fit) the sklearn pipeline for this baseline.

        ``ts_metric`` is supplied by :meth:`_fit_with_fallback`; leaves that
        have no :class:`TangentSpace` step should simply ignore it.
        """
        raise NotImplementedError

    def _compute_output_dim(self, y: np.ndarray, target_info: dict[str, tp.Any]) -> int:
        """Output dimensionality of the wrapped :class:`SklearnBaselineModel`."""
        raise NotImplementedError

    def _describe_target(self, target_info: dict[str, tp.Any]) -> str:
        """One-line human-readable description of the target for logging."""
        return self._kind

    def _cov_ts_steps(self, ts_metric: str) -> list[tp.Any]:
        """``Covariances -> Shrinkage -> TangentSpace -> StandardScaler``
        prefix shared by the non-Xdawn Riemannian leaves (:class:`CovTsLR`,
        :class:`CovTsRidge`).

        The trailing :class:`~sklearn.preprocessing.StandardScaler` puts all
        tangent-space features on a common scale so the linear head's
        regularization penalty is not dominated by the most heavily-scaled
        (typically diagonal) entries -- without it, a fixed regularizer
        collapses predictions toward the training mean on tasks where the
        raw-feature variance is small.
        """
        return [
            Covariances(estimator=self.cov_estimator),
            Shrinkage(shrinkage=self.cov_shrinkage),
            TangentSpace(metric=ts_metric),
            StandardScaler(),
        ]

    # ------------------------------------------------------------------
    # Shared fit-time helpers
    # ------------------------------------------------------------------
    def _drop_flat_trials(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drop trials whose per-channel variance is near zero.

        OAS-shrunk covariances of a flat (all-zero) trial are still all-zero
        and therefore not positive definite, which causes ``logm`` inside
        ``pyriemann.TangentSpace`` to fail.  Removing such trials up front
        avoids the need for opaque downstream regularization tweaks.
        """
        thresh = float(self.min_trial_variance)
        if thresh <= 0:
            return X, y
        per_trial_var = X.reshape(X.shape[0], -1).var(axis=-1)
        keep = per_trial_var > thresh
        dropped = int((~keep).sum())
        if dropped == 0:
            return X, y
        LOGGER.info(
            "SklearnBaseline: dropping %d/%d trials with variance <= %.1e "
            "before classical fit.",
            dropped,
            X.shape[0],
            thresh,
        )
        return X[keep], y[keep]

    def _maybe_subsample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Uniformly subsample ``(X, y)`` to ``self.max_fit_samples`` trials.

        Keeps the full dataset when N <= cap.  Uses stratified sampling for 1D
        integer (multiclass) targets to preserve class balance; falls back to
        uniform sampling for 2D (one-hot or multilabel) and regression targets.
        """
        n = X.shape[0]
        cap = int(self.max_fit_samples)
        if cap <= 0 or n <= cap:
            return X, y

        rng = np.random.default_rng(self.random_state)

        if y.ndim == 1 and np.issubdtype(y.dtype, np.integer):
            classes, inverse = np.unique(y, return_inverse=True)
            n_per_class = np.maximum(1, (cap * np.bincount(inverse) // n))
            total = int(n_per_class.sum())
            if total < cap:
                extra = cap - total
                bumped = rng.choice(len(classes), size=extra, replace=True)
                for idx in bumped:
                    n_per_class[idx] += 1
            indices = np.concatenate(
                [
                    rng.choice(
                        np.flatnonzero(inverse == cls_idx), size=n_k, replace=False
                    )
                    for cls_idx, n_k in enumerate(n_per_class)
                ]
            )
        else:
            indices = rng.choice(n, size=cap, replace=False)

        indices = np.sort(indices)
        LOGGER.info(
            "SklearnBaseline: subsampling %d -> %d trials for classical fit "
            "(random_state=%d).",
            n,
            indices.size,
            self.random_state,
        )
        return X[indices], y[indices]

    def _fit_with_fallback(
        self,
        X: np.ndarray,
        y: np.ndarray,
        build_fn: tp.Callable[[str], Pipeline],
    ) -> Pipeline:
        """Fit the pipeline with progressively more robust TangentSpace metrics.

        The default (usually ``"riemann"``) uses an iterative Frechet mean on
        the cone of PD matrices and occasionally fails with ``ValueError:
        Matrices must be positive definite`` on large / noisy EEG datasets
        (observed on the multilabel artifact task).  We retry first with
        ``"logeuclid"`` (closed-form log-Euclidean mean) and, if that also
        fails -- which can happen because ``pyriemann``'s ``expm`` rejects
        symmetric matrices with negative eigenvalues -- we fall back to
        ``"euclid"`` (simple arithmetic mean of covariances), which is
        numerically bulletproof for PD inputs.  Pipelines without a
        TangentSpace step are unaffected because they never raise this error.
        """
        pd_error = "positive definite"
        candidates = list(dict.fromkeys([self.ts_metric, "logeuclid", "euclid"]))
        last_err: ValueError | None = None
        for idx, metric in enumerate(candidates):
            try:
                pipe = build_fn(metric)
                pipe.fit(X, y)
                if idx > 0:
                    LOGGER.warning(
                        "SklearnBaseline: %s succeeded with ts_metric=%r "
                        "after %d fallback(s).",
                        type(self).__name__,
                        metric,
                        idx,
                    )
                return pipe
            except ValueError as err:
                if pd_error not in str(err):
                    raise
                LOGGER.warning(
                    "SklearnBaseline: %s fit failed with ts_metric=%r (%s); %s",
                    type(self).__name__,
                    metric,
                    err,
                    "retrying with a more robust metric..."
                    if idx < len(candidates) - 1
                    else "no more fallbacks available.",
                )
                last_err = err
        assert last_err is not None
        raise last_err


# ----------------------------------------------------------------------
# Classifier hierarchy
# ----------------------------------------------------------------------
_DEFAULT_LR_CS: tp.Final[tuple[float, ...]] = tuple(
    10.0**k for k in range(-3, 4)
)  # (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000)
_DEFAULT_LR_CV: tp.Final[int] = 5


class SklearnClassifier(SklearnBaseline):
    """Abstract classifier base.

    Owns the target-normalisation logic (single-label vs multilabel, one-hot
    collapse) and the :class:`~sklearn.linear_model.LogisticRegressionCV` head.

    Parameters
    ----------
    lr_Cs
        Grid of inverse regularisation strengths for
        :class:`~sklearn.linear_model.LogisticRegressionCV`.  A length-one
        grid degenerates to a fixed-``C`` fit.
    lr_cv
        Number of stratified CV folds used to select ``C`` from ``lr_Cs``.
    """

    lr_Cs: tuple[float, ...] = _DEFAULT_LR_CS
    lr_cv: int = _DEFAULT_LR_CV

    _kind: tp.ClassVar[tp.Literal["classifier", "regressor"]] = "classifier"
    # Whether this leaf supports ``(N, L)`` multilabel targets.  Set False on
    # leaves whose spatial-filter stage requires binary / single-label
    # multiclass targets (e.g. Xdawn).
    _supports_multilabel: tp.ClassVar[bool] = True

    def _normalise_targets(
        self, y_train: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray, dict[str, tp.Any]]:
        y, n_classes, is_multilabel = _normalise_classification_targets(y_train)
        if is_multilabel and not self._supports_multilabel:
            raise ValueError(
                f"SklearnBaseline: multilabel targets are not supported for "
                f"{type(self).__name__}.  Use CovTsLR for multilabel tasks instead."
            )
        return y, {"n_classes": n_classes, "is_multilabel": is_multilabel}

    def _compute_output_dim(self, y: np.ndarray, target_info: dict[str, tp.Any]) -> int:
        return int(target_info["n_classes"])

    def _describe_target(self, target_info: dict[str, tp.Any]) -> str:
        n_classes = target_info["n_classes"]
        suffix = ", multilabel" if target_info["is_multilabel"] else ""
        return f"n_classes={n_classes}{suffix}"

    def _build_lr_head(
        self, is_multilabel: bool
    ) -> LogisticRegressionCV | OneVsRestClassifier:
        """``LogisticRegressionCV`` (wrapped in OvR for multilabel targets)."""
        lr = LogisticRegressionCV(
            Cs=list(self.lr_Cs),
            cv=self.lr_cv,
            max_iter=1000,
            class_weight="balanced",
            scoring="accuracy",
        )
        if is_multilabel:
            return OneVsRestClassifier(lr)
        return lr


class XdawnTsLR(SklearnClassifier):
    """XdawnCovariances + Shrinkage + TangentSpace + LogisticRegression.

    Classical ERP / P300 classification convention (moabb).

    Parameters
    ----------
    xdawn_nfilter
        Number of Xdawn spatial filters per class.
    xdawn_target_class
        Class index for which to compute Xdawn filters.  ``-1`` means
        auto-detect (use the minority class, which is the standard choice for
        ERP-style binary classification where one class evokes the ERP).
    """

    xdawn_nfilter: int = 4
    xdawn_target_class: int = -1
    # Xdawn benefits from Ledoit-Wolf shrinkage because it fits class-conditional
    # means; override the ``oas`` base default.
    cov_estimator: str = "lwf"
    _supports_multilabel: tp.ClassVar[bool] = False

    def _build_pipeline(
        self,
        *,
        y: np.ndarray,
        target_info: dict[str, tp.Any],
        ts_metric: str,
    ) -> Pipeline:
        n_classes = int(target_info["n_classes"])
        if self.xdawn_target_class == -1:
            counts = np.bincount(y, minlength=n_classes)
            target_class = int(np.argmin(counts))
            LOGGER.info(
                "SklearnBaseline: auto-detected Xdawn target class = %d "
                "(class counts: %s)",
                target_class,
                counts.tolist(),
            )
        else:
            target_class = self.xdawn_target_class
        return make_pipeline(
            XdawnCovariances(
                nfilter=self.xdawn_nfilter,
                classes=[target_class],
                estimator=self.cov_estimator,
            ),
            Shrinkage(shrinkage=self.cov_shrinkage),
            TangentSpace(metric=ts_metric),
            StandardScaler(),
            self._build_lr_head(is_multilabel=False),
        )


class CovTsLR(SklearnClassifier):
    """Covariances + Shrinkage + TangentSpace + LogisticRegression.

    Classical oscillatory / motor-imagery classification convention
    (moabb ``TSLR.yml``).  Supports multilabel targets via
    :class:`OneVsRestClassifier`.
    """

    def _build_pipeline(
        self,
        *,
        y: np.ndarray,
        target_info: dict[str, tp.Any],
        ts_metric: str,
    ) -> Pipeline:
        return make_pipeline(
            *self._cov_ts_steps(ts_metric),
            self._build_lr_head(is_multilabel=target_info["is_multilabel"]),
        )


class _UpperTriLog1p(TransformerMixin, BaseEstimator):
    """Flatten ``(N, C, C, F)`` spectral matrices to ``(N, F * C*(C+1)/2)``.

    The diagonal entries of each frequency bin are per-channel PSD estimates
    (always non-negative), with a high dynamic range between SSVEP peaks and
    broadband noise: ``log1p`` compresses that range so the downstream linear
    classifier's regularizer does not get dominated by a single peak.  The
    off-diagonal entries are signed real cross-spectra (CoSpectra is the
    real part of the cross-spectrum) and are kept raw to preserve sign
    information; negative values would make ``log1p`` undefined anyway.
    """

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "_UpperTriLog1p":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 4 or X.shape[1] != X.shape[2]:
            raise ValueError(
                f"_UpperTriLog1p expects (N, C, C, F) input, got shape {X.shape}"
            )
        n, c, _, f = X.shape
        iu, ju = np.triu_indices(c)
        feats = X[:, iu, ju, :]  # (N, K, F) with K = C*(C+1)/2
        is_diag = iu == ju
        # Diagonal entries are power (>= 0 for a valid co-spectrum) but numerical
        # noise can produce small negative values -- clamp before ``log1p``.
        feats = feats.astype(np.float64, copy=True)
        feats[:, is_diag, :] = np.log1p(np.clip(feats[:, is_diag, :], 0.0, None))
        return feats.reshape(n, -1)


class CoSpectraLogLR(SklearnClassifier):
    """CoSpectra + upper-tri log1p + StandardScaler + LogisticRegressionCV.

    Spectral classification baseline for steady-state paradigms (SSVEP,
    cVEP, FBCCA-style tasks) where class identity is encoded in a narrow
    band of sustained oscillations rather than an event-locked ERP.
    Computes per-trial co-spectral matrices via Welch-style overlapping
    windows (:class:`pyriemann.estimation.CoSpectra`), flattens the upper
    triangle of each frequency bin with ``log1p`` on the diagonal (power)
    and raw values on the off-diagonal (cross-spectra), then feeds the
    concatenated features to a CV-selected logistic regression.

    Unlike the other pyriemann-based leaves this baseline does **not**
    tangent-space the spectral matrices: the asymmetry buys a much simpler
    pipeline with no per-frequency geometry, at the cost of a slightly
    noisier linearisation.  In internal benchmarks (MAMEM3, Nakanishi2015)
    it matched the tangent-space variant on mean accuracy while avoiding
    the filter-bank / stimulus-frequency plumbing that MOABB's
    ``SSVEP_TS+LR`` relies on.

    Parameters
    ----------
    sfreq
        Sampling frequency of the inputs in Hz.  **Must** match the rate at
        which the :class:`~neuralset.extractors.neuro.EegExtractor` resamples
        the raw data (default ``120.0`` in the neuralbench defaults).  Only
        used to map FFT bins to physical frequencies when ``fmin``/``fmax``
        are set.
    window
        Welch-style FFT window length in samples; rounded up to the next
        power of two by pyriemann.  The resulting frequency resolution is
        ``sfreq / next_pow2(window)`` Hz.
    overlap
        Fractional overlap between consecutive Welch windows (0--1).
    fmin, fmax
        Optional band limits in Hz; frequencies outside ``[fmin, fmax]`` are
        dropped before flattening, reducing feature dimensionality.  Set
        both to ``None`` to keep every FFT bin up to Nyquist.
    """

    sfreq: float = 120.0
    window: int = 128
    overlap: float = 0.75
    fmin: float | None = 2.0
    fmax: float | None = 40.0

    def _build_pipeline(
        self,
        *,
        y: np.ndarray,
        target_info: dict[str, tp.Any],
        ts_metric: str,
    ) -> Pipeline:
        # ``ts_metric`` is irrelevant here (no TangentSpace step); the
        # PD-fallback loop will never trigger because CoSpectra + log1p
        # never raises "positive definite" errors.
        del ts_metric
        return make_pipeline(
            CoSpectra(
                window=self.window,
                overlap=self.overlap,
                fmin=self.fmin,
                fmax=self.fmax,
                fs=self.sfreq,
            ),
            _UpperTriLog1p(),
            StandardScaler(),
            self._build_lr_head(is_multilabel=target_info["is_multilabel"]),
        )


# ----------------------------------------------------------------------
# Regressor hierarchy
# ----------------------------------------------------------------------
_DEFAULT_RIDGE_ALPHAS: tp.Final[tuple[float, ...]] = tuple(
    10.0**k for k in range(-3, 4)
)  # (1e-3, 1e-2, 1e-1, 1, 10, 100, 1000)


class SklearnRegressor(SklearnBaseline):
    """Abstract regressor base.

    Parameters
    ----------
    ridge_alphas
        Grid of regularisation strengths for
        :class:`~sklearn.linear_model.RidgeCV`.  ``RidgeCV`` selects the best
        alpha via closed-form leave-one-out CV (effectively free on top of a
        single ``Ridge.fit``).  A length-one grid degenerates to a fixed-alpha
        fit.
    """

    ridge_alphas: tuple[float, ...] = _DEFAULT_RIDGE_ALPHAS
    _kind: tp.ClassVar[tp.Literal["classifier", "regressor"]] = "regressor"

    def _normalise_targets(
        self, y_train: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray, dict[str, tp.Any]]:
        y = _normalise_regression_targets(y_train)
        return y, {}

    def _compute_output_dim(self, y: np.ndarray, target_info: dict[str, tp.Any]) -> int:
        return 1 if y.ndim == 1 else int(y.shape[1])

    def _describe_target(self, target_info: dict[str, tp.Any]) -> str:
        return "regression"


class CovTsRidge(SklearnRegressor):
    """Covariances + Shrinkage + TangentSpace + StandardScaler + RidgeCV.

    Continuous-target regression baseline (e.g. age prediction); handles both
    univariate and multi-output targets via sklearn's native multi-output
    Ridge.  Covariance-pipeline hyperparameters (``cov_estimator``,
    ``ts_metric``, ``cov_shrinkage``) are inherited from
    :class:`SklearnBaseline`; the ``alpha`` grid is inherited from
    :class:`SklearnRegressor` and selected via leave-one-out CV inside
    :class:`~sklearn.linear_model.RidgeCV`.
    """

    def _build_pipeline(
        self,
        *,
        y: np.ndarray,
        target_info: dict[str, tp.Any],
        ts_metric: str,
    ) -> Pipeline:
        return make_pipeline(
            *self._cov_ts_steps(ts_metric),
            RidgeCV(alphas=list(self.ridge_alphas)),
        )


class SklearnBaselineModel(nn.Module):
    """Wrap a fitted sklearn pipeline as an eval-only ``nn.Module``.

    The fitted pipeline is stored as a plain attribute (not a submodule) so
    torch does not try to move it to GPU or register parameters.  A dummy
    zero-requires-grad parameter keeps Lightning's optimiser-construction
    happy when paired with ``lr: 0.0`` + ``eval_only: true``.

    For classifiers, :meth:`forward` returns ``predict_proba`` (shape
    ``(B, n_classes)``).  For regressors, it returns ``predict`` reshaped
    to ``(B, output_dim)``.
    """

    def __init__(
        self,
        pipe: Pipeline,
        kind: tp.Literal["classifier", "regressor"],
        output_dim: int,
    ):
        super().__init__()
        self._pipe = pipe
        self._kind = kind
        self.output_dim = output_dim
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,
        **_: tp.Any,
    ) -> torch.Tensor:
        """Run the fitted sklearn pipeline on ``(B, C, T)`` input.

        Returns
        -------
        torch.Tensor
            Shape ``(B, output_dim)``, dtype float32, device matching ``x``.
        """
        x_np = x.detach().cpu().numpy().astype(np.float64, copy=False)

        # Walk the transform chain explicitly so we can sanitize non-finite
        # intermediate features.  The covariance -> tangent-space leg can
        # emit NaN at predict time on DC-railed / saturated trials: the
        # per-trial covariance is effectively rank-deficient and the fixed
        # ``Shrinkage`` diagonal is not large enough to make it PD, so
        # ``TangentSpace.logm`` returns NaN.  Such trials would otherwise
        # crash the final estimator's ``predict`` / ``predict_proba``.
        # Imputing with zeros (the tangent-space / standardized mean) makes
        # the final linear head fall back to the training-mean prediction
        # for those trials, which is the principled chance-level answer.
        x_feat = x_np
        for _, step in self._pipe.steps[:-1]:
            x_feat = step.transform(x_feat)
        if not np.isfinite(x_feat).all():
            n_bad = int((~np.isfinite(x_feat)).any(axis=-1).sum())
            LOGGER.warning(
                "SklearnBaseline: %d/%d test trials produced non-finite "
                "features after the transform chain (likely saturated / "
                "rank-deficient covariances); imputing with zeros so the "
                "final estimator falls back to the training-mean prediction.",
                n_bad,
                x_feat.shape[0],
            )
            x_feat = np.nan_to_num(x_feat, nan=0.0, posinf=0.0, neginf=0.0)
        final = self._pipe.steps[-1][1]
        if self._kind == "classifier":
            out = final.predict_proba(x_feat)
        else:
            out = final.predict(x_feat)
            if out.ndim == 1:
                out = out[:, None]
        return torch.from_numpy(out).to(device=x.device, dtype=torch.float32)


def _normalise_classification_targets(
    y_train: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, int, bool]:
    """Convert raw targets to class indices (single-label) or an ``(N, L)``
    binary label matrix (multilabel).

    Multilabel is detected as a 2D target whose per-row sum is not always 1
    (either zero labels or multiple labels per sample).  One-hot (exactly one
    ``1`` per row) is treated as single-label multiclass and collapsed to 1D
    integer indices, mirroring
    :func:`neuralbench.utils.compute_class_weights_from_dataset`.

    Returns
    -------
    tuple
        ``(y, n_classes, is_multilabel)`` where ``y`` is ``(N,)`` int64 for
        single-label or ``(N, L)`` int64 for multilabel, ``n_classes`` is the
        output dimension, and ``is_multilabel`` is ``True`` when the target
        matrix has more than one (or zero) active labels per row.
    """
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.cpu().numpy()
    if y_train.ndim == 2:
        n_classes = int(y_train.shape[1])
        row_sums = y_train.sum(axis=1)
        if np.any(row_sums != 1):
            # Multilabel: force strictly {0, 1} so OneVsRestClassifier's
            # LabelBinarizer recognises it as ``multilabel-indicator`` rather
            # than ``multiclass-multioutput`` (any cell with value > 1 would
            # otherwise trigger ``ValueError: Multioutput target data is not
            # supported with label binarization``).
            y_bin = (np.asarray(y_train) > 0).astype(np.int64)
            return y_bin, n_classes, True
        return y_train.argmax(axis=1).astype(np.int64), n_classes, False
    if y_train.ndim == 1:
        y_int = y_train.astype(np.int64)
        return y_int, int(y_int.max()) + 1, False
    raise ValueError(
        f"SklearnBaseline classifier expects 1D or 2D targets, got shape {y_train.shape}"
    )


def _normalise_regression_targets(
    y_train: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Cast regression targets to float64.

    Accepts ``(N,)`` scalar targets and ``(N, D)`` multi-output targets;
    sklearn's :class:`~sklearn.linear_model.Ridge` handles both natively
    (``(N, 1)`` is treated identically to ``(N,)``).
    """
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.cpu().numpy()
    y = np.asarray(y_train, dtype=np.float64)
    if y.ndim > 2:
        raise ValueError(
            f"SklearnBaseline regressor expects 1D or 2D targets, got shape {y.shape}"
        )
    return y
