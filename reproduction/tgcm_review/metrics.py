"""Per-sequence metrics used by the released inference notebooks."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def align_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_labels = np.unique(y_true)
    true_labels = true_labels[true_labels > 0]
    pred_labels = np.unique(y_pred)
    aligned = y_pred.copy()
    if not len(true_labels) or not len(pred_labels):
        return aligned
    overlap = np.zeros((len(true_labels), len(pred_labels)), dtype=np.int64)
    for row, true_value in enumerate(true_labels):
        for column, pred_value in enumerate(pred_labels):
            overlap[row, column] = np.sum((y_true == true_value) & (y_pred == pred_value))
    rows, columns = linear_sum_assignment(-overlap)
    mapping = {int(pred_labels[c]): int(true_labels[r]) for r, c in zip(rows, columns)}
    next_label = int(true_labels.max()) + 1
    for pred_value in pred_labels:
        source = int(pred_value)
        destination = mapping.get(source, next_label)
        if source not in mapping:
            next_label += 1
        aligned[y_pred == source] = destination
    return aligned


def _fmi_nmi(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    _, true_inverse, true_counts = np.unique(y_true, return_inverse=True, return_counts=True)
    _, pred_inverse, pred_counts = np.unique(y_pred, return_inverse=True, return_counts=True)
    contingency = np.zeros((len(true_counts), len(pred_counts)), dtype=np.int64)
    np.add.at(contingency, (true_inverse, pred_inverse), 1)
    comb2 = lambda values: values.astype(np.float64) * (values - 1.0) / 2.0
    denominator = math.sqrt(float(comb2(true_counts).sum() * comb2(pred_counts).sum()))
    fmi = float(comb2(contingency).sum()) / denominator if denominator else 0.0
    n = len(y_true)
    joint = contingency.astype(np.float64) / n
    true_probability = true_counts.astype(np.float64) / n
    pred_probability = pred_counts.astype(np.float64) / n
    expected = np.outer(true_probability, pred_probability)
    nonzero = joint > 0
    mutual_information = float(np.sum(joint[nonzero] * np.log(joint[nonzero] / expected[nonzero])))
    h_true = float(-np.sum(true_probability[true_probability > 0] * np.log(true_probability[true_probability > 0])))
    h_pred = float(-np.sum(pred_probability[pred_probability > 0] * np.log(pred_probability[pred_probability > 0])))
    if h_true == h_pred == 0.0:
        nmi = 1.0
    elif h_true == 0.0 or h_pred == 0.0:
        nmi = 0.0
    else:
        nmi = 2.0 * mutual_information / (h_true + h_pred)
    return float(fmi), float(nmi)


def sequence_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    hungarian: bool = True,
    include_predicted_labels: bool = False,
) -> dict[str, float] | None:
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    length = min(len(truth), len(prediction))
    if not length:
        return None
    truth, prediction = truth[:length], prediction[:length]
    valid = truth > 0
    if not valid.any():
        return None
    truth, prediction = truth[valid], prediction[valid]
    raw_prediction = prediction.copy()
    if hungarian:
        prediction = align_labels(truth, prediction)
        labels = np.union1d(truth, prediction) if include_predicted_labels else np.unique(truth)
    else:
        labels = np.union1d(truth, prediction)
    precision_values, recall_values, f1_values = [], [], []
    for label in labels:
        true_mask, pred_mask = truth == label, prediction == label
        tp = int(np.sum(true_mask & pred_mask))
        fp = int(np.sum(~true_mask & pred_mask))
        fn = int(np.sum(true_mask & ~pred_mask))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    fmi, nmi = _fmi_nmi(truth, raw_prediction)
    true_k = len(np.unique(truth))
    pred_k = len(np.unique(raw_prediction[raw_prediction > 0]))
    return {
        "Acc": float(np.mean(truth == prediction)),
        "FMI": fmi,
        "NMI": nmi,
        "Precision": float(np.mean(precision_values)),
        "Recall": float(np.mean(recall_values)),
        "Macro-F1": float(np.mean(f1_values)),
        "K-MAE": float(abs(true_k - pred_k)),
    }


def mean_metrics(items: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(items)
    if not rows:
        raise ValueError("No valid metric rows")
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
