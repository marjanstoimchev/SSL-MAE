import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, coverage_error,
    f1_score, hamming_loss, label_ranking_loss,
    precision_score, recall_score,
)


def findmax(outputs):
    Max = -float("inf")
    index = 0
    for i in range(outputs.shape[0]):
        if outputs[i] > Max:
            Max = outputs[i]
            index = i
    return Max, index


def OneError(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    num = 0
    one_error = 0
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            Max, index = findmax(outputs[i])
            num = num + 1
            if test_target[i][index] != 1:
                one_error = one_error + 1
    return one_error / num


def calculate_mcc_metrics(Y):
    """Evaluation metrics for multi-class classification."""
    y_true, y_targs_hot = Y['y_true'], Y['one_hot']
    y_pred, y_scores = Y['y_pred'], Y['y_scores']

    metric_names = [
        "micro f1", "micro recall", "micro precision",
        "macro f1", "macro recall", "macro precision",
        "accuracy", "hamming loss", "auprc",
    ]

    metrics = [
        f1_score(y_true, y_pred, average='micro'),
        recall_score(y_true, y_pred, average='micro'),
        precision_score(y_true, y_pred, average='micro'),
        f1_score(y_true, y_pred, average='macro'),
        recall_score(y_true, y_pred, average="macro"),
        precision_score(y_true, y_pred, average="macro", zero_division=0),
        accuracy_score(y_true, y_pred),
        hamming_loss(y_true, y_pred),
        average_precision_score(y_targs_hot, y_scores, average="macro"),
    ]

    return pd.DataFrame(
        {name: metric for name, metric in zip(metric_names, metrics)},
        index=["Metric Value"],
    ).T


def calculate_mlc_metrics(Y):
    """Evaluation metrics for multi-label classification."""
    y_true, y_pred, y_scores = Y['y_true'], Y['y_pred'], Y['y_scores']

    metric_names = [
        "ranking loss", "one error", "coverage",
        "average auprc", "weighted auprc",
        "micro f1", "micro recall", "micro precision",
        "macro f1", "macro recall", "macro precision",
        "subset accuracy", "hamming loss",
        "ml_f_one", "ml_recall", "ml_precision",
    ]

    metrics = [
        label_ranking_loss(y_true, y_scores),
        OneError(y_scores, y_true),
        coverage_error(y_true, y_scores) - 1,
        average_precision_score(y_true, y_scores, average="macro"),
        average_precision_score(y_true, y_scores, average="weighted"),
        f1_score(y_true, y_pred, average='micro'),
        recall_score(y_true, y_pred, average="micro"),
        precision_score(y_true, y_pred, average="micro"),
        f1_score(y_true, y_pred, average='macro'),
        recall_score(y_true, y_pred, average="macro"),
        precision_score(y_true, y_pred, average="macro", zero_division=0),
        accuracy_score(y_true, y_pred),
        hamming_loss(y_true, y_pred),
        f1_score(y_true, y_pred, average='samples'),
        recall_score(y_true, y_pred, average="samples"),
        precision_score(y_true, y_pred, average="samples", zero_division=0),
    ]

    return pd.DataFrame(
        {name: metric for name, metric in zip(metric_names, metrics)},
        index=["Metric Value"],
    ).T
