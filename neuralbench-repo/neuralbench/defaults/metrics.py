# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp


def get_classification_metric_configs(
    n_classes_or_labels: int,
    task: tp.Literal["multiclass", "multilabel"] = "multiclass",
) -> list[dict[str, tp.Any]]:
    cardinality_type = "num_classes" if task == "multiclass" else "num_labels"
    metrics = [
        {
            "log_name": "acc",
            "name": "Accuracy",
            "kwargs": {"task": task, cardinality_type: n_classes_or_labels},
        },
        {
            "log_name": "f1_score_micro",
            "name": "F1Score",
            "kwargs": {
                "task": task,
                "average": "micro",
                cardinality_type: n_classes_or_labels,
            },
        },
        {
            "log_name": "f1_score_macro",
            "name": "F1Score",
            "kwargs": {
                "task": task,
                "average": "macro",
                cardinality_type: n_classes_or_labels,
            },
        },
    ]
    if task == "multiclass":
        metrics.extend(
            [
                {
                    "log_name": "bal_acc",
                    "name": "Accuracy",
                    "kwargs": {
                        "task": "multiclass",
                        "num_classes": n_classes_or_labels,
                        "average": "macro",
                    },
                },
                {
                    "log_name": "auroc",
                    "name": "AUROC",
                    "kwargs": {"task": "multiclass", "num_classes": n_classes_or_labels},
                },
                {
                    "log_name": "auprc",
                    "name": "AveragePrecision",
                    "kwargs": {
                        "task": "multiclass",
                        "num_classes": n_classes_or_labels,
                        "average": "macro",
                    },
                },
                {
                    "log_name": "confusion_matrix",
                    "name": "ConfusionMatrix",
                    "kwargs": {"task": "multiclass", "num_classes": n_classes_or_labels},
                },
            ]
        )
    elif task == "multilabel":
        metrics.extend(
            [
                {
                    "log_name": "exact_match",
                    "name": "ExactMatch",
                    "kwargs": {"task": "multilabel", "num_labels": n_classes_or_labels},
                },
                {
                    "log_name": "hamming",
                    "name": "HammingDistance",
                    "kwargs": {"task": "multilabel", "num_labels": n_classes_or_labels},
                },
                {
                    "log_name": "confusion_matrix",
                    "name": "ConfusionMatrix",
                    "kwargs": {"task": "multilabel", "num_labels": n_classes_or_labels},
                },
            ]
        )

    return metrics


def get_regression_metric_configs(output_size: int) -> list[dict[str, tp.Any]]:
    return [
        {
            "log_name": "rmse",
            "name": "MeanSquaredError",
            "kwargs": {"num_outputs": output_size, "squared": False},
        },
        {
            "log_name": "mae",
            "name": "MeanAbsoluteError",
            "kwargs": {"num_outputs": output_size},
        },
        {
            "log_name": "pearsonr",
            "name": "PearsonCorrCoef",
            "kwargs": {"num_outputs": output_size},
        },
        {
            "log_name": "r2_score",
            "name": "R2Score",
            "kwargs": {},
        },
        {
            "log_name": "normalized_rmse",
            "name": "NormalizedRMSE",
            "num_outputs": output_size,
        },
    ]


retrieval_metrics = [
    {"log_name": "batch_median_rank", "name": "Rank", "reduction": "median"},
    {"log_name": "batch_mean_rank", "name": "Rank", "reduction": "mean"},
    {"log_name": "batch_std_rank", "name": "Rank", "reduction": "std"},
    {"log_name": "batch_top1_acc", "name": "TopkAcc", "topk": 1},
    {"log_name": "batch_top5_acc", "name": "TopkAcc", "topk": 5},
    {"log_name": "batch_top10_acc", "name": "TopkAcc", "topk": 10},
    {
        "log_name": "batch_pearson_corr",
        "name": "OnlinePearsonCorr",
        "reduction": "mean",
        "dim": 0,
    },
]


test_full_retrieval_metrics = [
    {
        "log_name": "median_retrieval_rank",
        "name": "Rank",
        "reduction": "median",
    },
    {
        "log_name": "median_retrieval_rank_subject-ind",
        "name": "Rank",
        "reduction": "median",
    },
    {
        "log_name": "mean_retrieval_rank",
        "name": "Rank",
        "reduction": "mean",
    },
    {
        "log_name": "std_retrieval_rank",
        "name": "Rank",
        "reduction": "std",
    },
    {
        "log_name": "median_retrieval_rank_instance-agg",
        "name": "Rank",
        "reduction": "median",
    },
    {
        "log_name": "median_retrieval_rank_subject-agg",
        "name": "Rank",
        "reduction": "median",
    },
    {"log_name": "top1_acc", "name": "TopkAcc", "topk": 1},
    {
        "log_name": "top1_acc_instance-agg",
        "name": "TopkAcc",
        "topk": 1,
    },
    {
        "log_name": "top1_acc_subject-agg",
        "name": "TopkAcc",
        "topk": 1,
    },
    {"log_name": "top5_acc", "name": "TopkAcc", "topk": 5},
    {
        "log_name": "top5_acc_instance-agg",
        "name": "TopkAcc",
        "topk": 5,
    },
    {
        "log_name": "top5_acc_subject-agg",
        "name": "TopkAcc",
        "topk": 5,
    },
    {"log_name": "top10_acc", "name": "TopkAcc", "topk": 10},
    {
        "log_name": "top10_acc_instance-agg",
        "name": "TopkAcc",
        "topk": 10,
    },
    {
        "log_name": "top10_acc_subject-agg",
        "name": "TopkAcc",
        "topk": 10,
    },
]
