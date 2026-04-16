# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Loss functions.

Ensure each new loss function has its own Pydantic configuration object under `losses/config.py` so
that a loss function can be defined in an experiment configuration.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ClipLoss(nn.Module):
    """CLIP constrastive loss.

    Contrastive Language-Image Pretraining (CLIP) loss from [1]_. Default values reflect the
    configuration of the CLIP loss used in [2]_.

    Parameters
    ----------
    norm_kind : {"x", "y", "xy"} or None
        How to normalize the estimates and/or candidates before computing their dot products.
            ``'x'``: normalize estimates only.
            ``'y'``: normalize candidates only (approach originally used in brainmagick).
            ``'xy'``: normalize both estimates and candidates.
            ``None``: do not normalize.
    temperature : bool
        If True, use a learnable temperature parameter.
    symmetric : bool
        If True, compute loss in both retrieval directions, i.e. retrieve candidates given
        estimates and retrieve estimates given candidates (requires estimates and candidates to be
        of the same shape). If False, only do the former.
    reduction : str
        Reduction applied to the per-example cross-entropy loss (forwarded to
        ``F.cross_entropy``).

    References
    ----------
    .. [1] Radford, Alec, et al. "Learning transferable visual models from natural language
        supervision." International conference on machine learning. PMLR, 2021.
    .. [2] Défossez, Alexandre, et al. "Decoding speech perception from non-invasive brain
        recordings." Nature Machine Intelligence (2023): 1-11.
    """

    def __init__(
        self,
        norm_kind: str | None = "y",
        temperature: bool = True,
        symmetric: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.norm_kind = norm_kind
        self.temperature = (
            nn.Parameter(torch.tensor(1 / 0.07).log())
            if temperature
            else nn.Parameter(torch.tensor(0.0), requires_grad=False)
        )
        self.symmetric = symmetric
        self.reduction = reduction

    @staticmethod
    def _compute_similarity(
        x: torch.Tensor, y: torch.Tensor, norm: str | None = None, eps=1e-15
    ) -> torch.Tensor:
        if norm is None:
            eq, inv_norms = "b", torch.ones(x.shape[0])
        elif norm == "x":
            eq, inv_norms = "b", 1 / (eps + x.norm(dim=(1), p=2))
        elif norm == "y":
            eq, inv_norms = "o", 1 / (eps + y.norm(dim=(1), p=2))
        elif norm == "xy":
            eq = "bo"
            inv_norms = 1 / (
                eps + torch.outer(x.norm(dim=(1), p=2), y.norm(dim=(1), p=2))
            )
        else:
            raise ValueError(f"norm must be None, x, y or xy, got {norm}.")

        # Normalize inside einsum to avoid creating a copy of candidates which can be pretty big
        return torch.einsum(f"bc,oc,{eq}->bo", x, y, inv_norms)

    def get_scores(self, estimate: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """Given estimates of shape [B, F] and candidates of shape [B', F], return a [B, B'] matrix
        of similarity scores.
        """
        scores = self._compute_similarity(estimate, candidate, norm=self.norm_kind)
        scores = self.temperature.exp() * scores
        return scores

    def get_probabilities(
        self, estimate: torch.Tensor, candidate: torch.Tensor
    ) -> torch.Tensor:
        """Given estimates of shape [B, F] and candidates of shape [B', F], return a [B, B'] matrix
        of matching probability.
        """
        scores = self.get_scores(estimate, candidate)
        return F.softmax(scores, dim=1)

    def forward(self, estimate: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """Warning: estimate and candidate are not necessarily symmetrical.

        If estimate of shape [B, C] and candidate of shape [B', C] with B'>=B, the first B samples
        of candidate are targets, while the remaining B'-B samples of candidate are only used as
        negatives.
        """
        n_est = estimate.size(0)
        n_cand = candidate.size(0)
        if n_est > n_cand:
            raise ValueError(f"need candidates >= estimates, got {n_cand} vs {n_est}")
        scores = self.get_scores(estimate, candidate)
        target = torch.arange(len(scores), device=estimate.device)

        loss_e = F.cross_entropy(scores, target, reduction=self.reduction)
        if self.symmetric:
            if scores.shape[0] != scores.shape[1]:
                raise ValueError(f"need square scores, got {scores.shape}")
            loss_c = F.cross_entropy(
                scores.transpose(1, 0), target, reduction=self.reduction
            )
            loss = (loss_e + loss_c) / 2
        else:
            loss = loss_e
        return loss


class SigLipLoss(ClipLoss):
    """SigLIP contrastive loss.

    Sigmoid loss for Language-Image Pretraining (SigLIP) from [1]_.

    Parameters
    ----------
    norm_kind : {"x", "y", "xy"} or None
        How to normalize the estimates and/or candidates before computing their dot products.
            ``'x'``: normalize estimates only.
            ``'y'``: normalize candidates only (approach originally used in brainmagick).
            ``'xy'``: normalize both estimates and candidates.
            ``None``: do not normalize.
    temperature : bool
        If True, use a learnable temperature parameter initialized to ``ln(10)``.
    bias : bool
        If True, use a learnable bias parameter initialized to ``-10`` (since
        most pairs are negative).
    identical_candidates_threshold : float or None
        If given, estimates are matched not only to their candidate, but all candidates
        that have a large cosine similarity to their candidate (larger or equal this threshold).
        Assumes such other candidates with high cosine similarity are duplicates.
        Intended to use only if candidate generator is frozen.
    reduction : str
        Reduction applied to the binary cross-entropy loss (forwarded to
        ``F.binary_cross_entropy_with_logits``).
    reweigh_positives : bool
        If True and *identical_candidates_threshold* is set, down-weight
        duplicate positive pairs so only one copy contributes to the loss.

    References
    ----------
    .. [1] Zhai, Xiaohua, et al. "Sigmoid loss for language image pre-training." arXiv preprint
        arXiv:2303.15343 (2023).

    Note
    ----
    Official jax implementation: https://github.com/google-research/big_vision/blob/474dd2ebde37268db4ea44decef14c7c1f6a0258/big_vision/trainers/proj/image_text/siglip.py
    """

    def __init__(
        self,
        norm_kind: str | None = "y",
        temperature: bool = True,
        bias: bool = True,
        identical_candidates_threshold: float | None = 0.999,
        reduction: str = "sum",
        reweigh_positives: bool = False,
    ):
        super().__init__(
            norm_kind=norm_kind, temperature=False, symmetric=True, reduction=reduction
        )
        self.temperature = (
            nn.Parameter(torch.tensor(10).log())
            if temperature
            else nn.Parameter(torch.tensor(0.0), requires_grad=False)
        )
        self.bias = (
            nn.Parameter(torch.tensor(-10.0))
            if bias
            else nn.Parameter(torch.tensor(0.0), requires_grad=False)
        )
        self.identical_candidates_threshold = identical_candidates_threshold
        self.reweigh_positives = reweigh_positives

    def get_scores(self, estimate: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        """Given estimates of shape [B, F] and candidates of shape [B', F], return a [B, B'] matrix
        of similarity scores.
        """
        return super().get_scores(estimate, candidate) + self.bias

    def forward(self, estimate: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        n_est = estimate.size(0)
        n_cand = candidate.size(0)
        if n_est > n_cand:
            raise ValueError(f"need candidates >= estimates, got {n_cand} vs {n_est}")
        scores = self.get_scores(estimate, candidate)
        if self.identical_candidates_threshold is not None:
            candidate_sim = self._compute_similarity(
                candidate, candidate, "xy", eps=1e-15
            )
            targets = 1.0 * (candidate_sim >= self.identical_candidates_threshold)
            targets = targets[: len(estimate)]
            if self.reweigh_positives:
                weights = 1.0 * (candidate_sim >= self.identical_candidates_threshold)
                weights = 1 - weights  # remove all duplicates
                weights += torch.eye(
                    *weights.shape, device=weights.device
                )  # keep only one
            else:
                weights = None
        else:
            weights = None
            targets = torch.eye(*scores.shape, device=scores.device)
        loss = F.binary_cross_entropy_with_logits(
            scores, targets, weights, reduction=self.reduction
        )
        return loss


class MultiLoss(nn.Module):
    """Weighted combination of multiple loss terms.

    Can be parametrized through MultiLossConfig.

    Parameters
    ----------
    losses :
        Different loss terms.
    weights :
        Weights associated with each loss term. If None, use weight of 1 for each term.
    """

    def __init__(
        self,
        losses: dict[str, nn.Module],
        weights: dict[str, float] | None = None,
    ):
        super().__init__()

        self.losses = nn.ModuleDict(losses)
        if weights is None:
            weights = {name: 1.0 for name in losses}
        self.weights = weights
        assert len(losses) == len(weights)

    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        y: torch.Tensor | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Evaluate the different loss terms.

        Parameters
        ----------
        input, target :
            If provided as a dictionary, the keys must match the loss names provided when
            the class was instantiated.
        """
        loss_values = {"total": torch.tensor(0.0)}
        for name in self.losses:
            _x = x[name] if isinstance(x, dict) else x
            _y = y[name] if isinstance(y, dict) else y
            loss_values[name] = self.losses[name](_x, _y)
            loss_values["total"] += self.weights[name] * loss_values[name]

        return loss_values
