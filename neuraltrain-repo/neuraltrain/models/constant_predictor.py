# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch
from torch import nn

from .base import BaseModelConfig


class ConstantPredictor(BaseModelConfig):
    """
    Constant predictor that predicts the most frequent class or the mean of the targets.

    Parameters
    ----------
    mode: tp.Literal["most_frequent", "most_frequent_multilabel", "mean", "auto"]
        Mode to use for obtaining the constant from the targets. Options:
        - "most_frequent": Predicts the most frequent class (for single-label classification).
        - "most_frequent_multilabel": Predicts the most frequent value for each class independently
          (for multilabel classification where multiple classes can be active simultaneously).
        - "mean": Predicts the mean of the targets (for regression).
        - "auto": Automatically determines the mode based on the dtype and shape of the targets.
    """

    mode: tp.Literal["most_frequent", "most_frequent_multilabel", "mean", "auto"] = "auto"

    def build(self, y_train: torch.Tensor) -> "ConstantPredictorModel":  # type: ignore[override]
        if self.mode == "auto":
            if y_train.dtype in (torch.int, torch.int64, torch.long):
                mode = "most_frequent"
                if y_train.ndim == 2:
                    n_classes_per_example = (y_train > 0).sum(dim=1)
                    if (n_classes_per_example == 0).any() or (
                        n_classes_per_example > 1
                    ).any():
                        mode = "most_frequent_multilabel"
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
        elif mode == "most_frequent_multilabel":
            out = (y_train > 0).int().mode(dim=0)[0]
        elif mode == "mean":
            out = y_train.mean(dim=0)
        else:
            raise ValueError(f"Unsupported dtype: {y_train.dtype}")

        return ConstantPredictorModel(out=out.float())


class ConstantPredictorModel(nn.Module):
    out: torch.Tensor

    def __init__(self, out: torch.Tensor):
        super().__init__()
        self.register_buffer("out", out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.out.repeat(X.shape[0], 1)
