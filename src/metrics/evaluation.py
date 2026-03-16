"""Core evaluation metrics for RTB prediction models.

Pure functions depending only on numpy/sklearn. No project-specific imports.
"""

from typing import NamedTuple, Optional

import numpy as np


def _numpy_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Pure numpy ROC AUC using the Mann-Whitney U statistic."""
    y_true = np.asarray(y_true, dtype=np.intp)
    y_score = np.asarray(y_score, dtype=np.float64)
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("Only one class present in y_true.")
    order = np.argsort(y_score)
    ranks = np.empty(len(y_true), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_true) + 1, dtype=np.float64)
    # Tie correction
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j > i + 1:
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg_rank
        i = j
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class EvalMetrics(NamedTuple):
    """Evaluation metrics."""
    auc: float
    log_loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    ece: float  # Expected calibration error
    ieb: float = 0.0  # Inherent estimation bias


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        n_bins: Number of bins

    Returns:
        ECE value (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        in_bin = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        prop_in_bin = in_bin.sum() / total

        if prop_in_bin > 0:
            avg_confidence = y_pred[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return ece


def compute_ieb(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute Inherent Estimation Bias (IEB).

    IEB = |mean(pred) - mean(actual)| / max(mean(actual), 1e-8)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities

    Returns:
        IEB value (lower is better, 0 = perfectly calibrated mean)
    """
    pred_mean = float(np.mean(y_pred))
    actual_mean = float(np.mean(y_true))
    return abs(pred_mean - actual_mean) / max(actual_mean, 1e-8)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> EvalMetrics:
    """Compute evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold

    Returns:
        EvalMetrics
    """
    # Filter out NaN
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Clip predictions for log_loss
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # AUC (pure numpy, ranking-based)
    try:
        auc = _numpy_roc_auc(y_true, y_pred)
    except ValueError:
        auc = 0.5  # Fallback if all same class

    # Log loss (pure numpy)
    logloss = float(-np.mean(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    ))

    # Binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Classification metrics (pure numpy)
    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))
    accuracy = float(np.mean(y_pred_binary == y_true))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Expected Calibration Error (ECE)
    ece = compute_ece(y_true, y_pred)

    # Inherent Estimation Bias (IEB)
    ieb = compute_ieb(y_true, y_pred)

    return EvalMetrics(
        auc=auc,
        log_loss=logloss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        ece=ece,
        ieb=ieb,
    )
