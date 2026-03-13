"""Core evaluation metrics for RTB prediction models.

Pure functions depending only on numpy/sklearn. No project-specific imports.
"""

from typing import NamedTuple, Optional

import numpy as np


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
    from sklearn.metrics import (
        roc_auc_score,
        log_loss,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    # Filter out NaN
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Clip predictions for log_loss
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Compute metrics
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.5  # Fallback if all same class

    logloss = log_loss(y_true, y_pred_clipped)

    # Binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

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
