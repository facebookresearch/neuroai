# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch
from torch import nn

from .base import BaseModelConfig


class DummyPredictor(BaseModelConfig):
    """
    Dummy predictor that makes predictions using simple rules based on the
    target distribution, analogous to ``sklearn.dummy.DummyClassifier``.

    Parameters
    ----------
    mode: tp.Literal[
        "most_frequent",
        "most_frequent_multilabel",
        "stratified_multilabel",
        "mean",
        "auto",
    ]
        Strategy used to derive predictions from the training targets.

        - ``"most_frequent"``: predict the most frequent class (single-label
          classification, constant output).
        - ``"most_frequent_multilabel"``: predict the most frequent binary
          value per class independently (multilabel classification, constant
          output per class).
        - ``"stratified_multilabel"``: sample each class label independently
          from a Bernoulli distribution with probability equal to the class
          prevalence in the training set (multilabel classification,
          stochastic output).  Produces macro-F1 scores that reflect the
          class prior rather than collapsing to 0 on rare-class tasks.
        - ``"mean"``: predict the mean of the targets (regression).
        - ``"auto"``: automatically pick a mode based on the dtype and shape
          of the targets.  Multilabel integer targets resolve to
          ``"stratified_multilabel"``.
    random_state: int | None
        Seed used to initialize the ``torch.Generator`` that drives the
        Bernoulli sampling in ``stratified_multilabel`` mode.  ``None``
        (default) falls back to the global RNG state (controlled e.g. by
        Lightning's ``seed_everything``).  Ignored by the other modes.
    """

    mode: tp.Literal[
        "most_frequent",
        "most_frequent_multilabel",
        "stratified_multilabel",
        "mean",
        "auto",
    ] = "auto"
    random_state: int | None = None

    def build(self, y_train: torch.Tensor) -> "DummyPredictorModel":  # type: ignore[override]
        if self.mode == "auto":
            if y_train.dtype in (torch.int, torch.int64, torch.long):
                mode = "most_frequent"
                if y_train.ndim == 2:
                    n_classes_per_example = (y_train > 0).sum(dim=1)
                    if (n_classes_per_example == 0).any() or (
                        n_classes_per_example > 1
                    ).any():
                        mode = "stratified_multilabel"
            elif torch.is_floating_point(y_train):
                mode = "mean"
            else:
                raise ValueError(f"Unsupported dtype: {y_train.dtype}")
        else:
            mode = self.mode

        if mode == "most_frequent":
            if y_train.ndim == 1:
                most_frequent_ind, _ = torch.mode(y_train)
                n_classes = int(y_train.max().item()) + 1
            elif y_train.ndim == 2:
                most_frequent_ind = y_train.sum(dim=0).argmax()
                n_classes = y_train.shape[1]
            else:
                raise NotImplementedError()
            out = torch.nn.functional.one_hot(most_frequent_ind, num_classes=n_classes)
            return DummyPredictorModel(out=out.float())
        if mode == "most_frequent_multilabel":
            out = (y_train > 0).int().mode(dim=0)[0]
            return DummyPredictorModel(out=out.float())
        if mode == "stratified_multilabel":
            if y_train.ndim != 2:
                raise ValueError(
                    f"stratified_multilabel requires 2D targets, got ndim={y_train.ndim}."
                )
            probs = (y_train > 0).float().mean(dim=0)
            return DummyPredictorModel(probs=probs, random_state=self.random_state)
        if mode == "mean":
            out = y_train.mean(dim=0)
            return DummyPredictorModel(out=out.float())
        raise ValueError(f"Unsupported mode: {mode}")


class DummyPredictorModel(nn.Module):
    """Evaluation-only module that implements the dummy prediction strategies.

    Constructed by :meth:`DummyPredictor.build`; not intended to be built
    directly by users.  Depending on the mode chosen at build time, either
    returns a constant tensor tiled across the batch dimension (``out``) or
    samples fresh Bernoulli draws per call using a cached ``torch.Generator``
    (``probs`` + ``random_state``).
    """

    out: torch.Tensor
    probs: torch.Tensor

    def __init__(
        self,
        out: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__()
        if (out is None) == (probs is None):
            raise ValueError(
                "Exactly one of `out` or `probs` must be provided to DummyPredictorModel."
            )
        self._stratified = probs is not None
        if out is not None:
            self.register_buffer("out", out)
        if probs is not None:
            self.register_buffer("probs", probs)
        self._random_state = random_state
        # Generator objects are device-specific and cannot be registered as
        # buffers; cache them lazily per device on first forward call.
        self._generators: dict[torch.device, torch.Generator] = {}

    def _get_generator(self, device: torch.device) -> torch.Generator | None:
        if self._random_state is None:
            return None
        gen = self._generators.get(device)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self._random_state)
            self._generators[device] = gen
        return gen

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if not self._stratified:
            return self.out.repeat(X.shape[0], 1)
        probs = self.probs.expand(X.shape[0], -1)
        generator = self._get_generator(probs.device)
        return torch.bernoulli(probs, generator=generator)
