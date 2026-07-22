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

    def build(  # type: ignore[override]
        self,
        y_train: torch.Tensor,
        blank_idx: int | None = None,
        n_classes: int | None = None,
    ) -> "DummyPredictorModel | DummyCtcSequenceModel":
        # CTC sequence tasks: ``y_train`` is ``(n_samples, max_length)`` of
        # blank-padded label ids rather than a class / multilabel row.  When
        # the caller signals a CTC objective (``blank_idx`` set), ignore the
        # classification ``mode`` and emit a constant most-frequent-character
        # sequence (see ``DummyCtcSequenceModel``).
        if blank_idx is not None:
            return self._build_ctc_sequence(y_train, blank_idx, n_classes)

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

    @staticmethod
    def _build_ctc_sequence(
        y_train: torch.Tensor, blank_idx: int, n_classes: int | None
    ) -> "DummyCtcSequenceModel":
        """Most-frequent-character CTC baseline (analogue of ``most_frequent``).

        Given blank-padded label sequences ``y_train`` of shape
        ``(n_samples, max_length)``, predict the most frequent non-blank
        character repeated ``L_median`` times, where ``L_median`` is the
        median number of real (non-blank) tokens per sequence.  The median
        minimises the L1 length penalty under edit-distance error rates.
        """
        y = y_train.long()
        if y.ndim != 2:
            raise ValueError(
                f"CTC dummy expects 2D (n_samples, max_length) targets; "
                f"got ndim={y.ndim}."
            )
        if n_classes is None:
            n_classes = int(y.max().item()) + 1
        non_blank_mask = y != blank_idx
        median_length = int(non_blank_mask.sum(dim=1).float().median().item())
        non_blank = y[non_blank_mask]
        if non_blank.numel() == 0 or median_length == 0:
            # No real keystrokes -> predict the empty string.
            modal_char = blank_idx
            median_length = 0
        else:
            modal_char = int(torch.mode(non_blank).values.item())
        return DummyCtcSequenceModel(
            modal_char=modal_char,
            length=median_length,
            blank_idx=blank_idx,
            n_classes=n_classes,
        )


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


class DummyCtcSequenceModel(nn.Module):
    """Constant CTC emitter for the most-frequent-character baseline.

    Emits input-independent per-frame scores of shape
    ``(batch, n_frames, n_classes)`` whose greedy CTC decode (collapse
    consecutive repeats + drop the blank, as
    :class:`neuraltrain.metrics.CharacterErrorRates` does) is exactly
    ``modal_char`` repeated ``length`` times.  The modal character is
    interleaved with the blank class (``[c, blank, c, blank, ..., c]``) so
    the repeats survive consecutive-collapse and the decoded length stays
    ``length`` even when ``modal_char`` would otherwise merge.

    The non-target classes are filled with a large negative score so the
    emissions double as valid (near one-hot) log-probabilities for
    :class:`torch.nn.CTCLoss`; ``argmax`` is unaffected by the magnitude.
    """

    emissions: torch.Tensor

    def __init__(
        self, modal_char: int, length: int, blank_idx: int, n_classes: int
    ) -> None:
        super().__init__()
        n_frames = max(2 * length - 1, 1)
        emissions = torch.full((n_frames, n_classes), -30.0)
        for t in range(n_frames):
            cls = modal_char if t % 2 == 0 else blank_idx
            emissions[t, cls] = 0.0
        self.register_buffer("emissions", emissions)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.emissions.to(X.device).unsqueeze(0).expand(X.shape[0], -1, -1)
