"""Reusable 3-panel prediction diagnostics: Calibration + ROC + Score Distribution."""

from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve

COMPARISON_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12"]


class DiagnosticsPlotConfig(NamedTuple):
    figsize: Tuple[int, int] = (15, 4)
    dpi: int = 150
    n_cal_bins: int = 10
    cal_strategy: str = "quantile"
    score_percentile: float = 99.5


def plot_prediction_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None,
    config: DiagnosticsPlotConfig = DiagnosticsPlotConfig(),
) -> plt.Figure:
    """3-panel diagnostics: Calibration (quantile bins) + ROC + Score Distribution.

    Args:
        y_true: Binary ground truth labels (0/1).
        y_pred: Predicted probabilities.
        model_name: Display name for the figure title.
        save_path: If provided, save figure to this path.
        config: Plot configuration.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=config.figsize)

    # 1. Calibration curve
    ax = axes[0]
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred, strategy=config.cal_strategy, n_bins=config.n_cal_bins
    )
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    ax.plot(prob_pred, prob_true, "o-", label=f"Model (ECE={ece:.6f})")
    # Auto-zoom: reference line & axes to data range (extreme class imbalance)
    xy_max = max(prob_pred.max(), prob_true.max()) * 1.15
    ax.plot([0, xy_max], [0, xy_max], "k--", label="Perfect calibration")
    ax.set_xlim(0, xy_max)
    ax.set_ylim(0, xy_max)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration ({config.cal_strategy} bins)")
    ax.legend()

    # 2. ROC curve
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    # 3. Score distribution (zoomed to P99.5, log y-axis)
    ax = axes[2]
    x_upper = np.percentile(y_pred, config.score_percentile)
    mask_neg = y_true == 0
    mask_pos = y_true == 1
    bins = np.linspace(0, x_upper, 51)
    ax.hist(y_pred[mask_neg], bins=bins, alpha=0.5, label="Negative", density=True)
    ax.hist(y_pred[mask_pos], bins=bins, alpha=0.5, label="Positive", density=True)
    ax.set_xlim(0, x_upper)
    ax.set_yscale("log")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density (log)")
    ax.set_title(f"Score Distribution (x <= P{config.score_percentile}={x_upper:.4f})")
    ax.legend()

    fig.suptitle(f"{model_name} — Prediction Diagnostics", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=config.dpi, bbox_inches="tight")

    return fig


def plot_comparison_diagnostics(
    models: List[Tuple[str, np.ndarray, np.ndarray]],
    save_path: Optional[Path] = None,
    config: DiagnosticsPlotConfig = DiagnosticsPlotConfig(),
) -> plt.Figure:
    """3-panel comparison diagnostics: overlay multiple models on Calibration + ROC + Score Distribution.

    Args:
        models: List of (model_name, y_true, y_pred) tuples.
        save_path: If provided, save figure to this path.
        config: Plot configuration.

    Returns:
        matplotlib Figure.
    """
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors = COMPARISON_COLORS[: len(models)]
        scale = 1e3  # display in x1e-3

        # 1. Calibration curve overlay (x1e-3 scale)
        ax = axes[0]
        xy_max = 0.0
        for (name, y_true, y_pred), color in zip(models, colors):
            prob_true, prob_pred = calibration_curve(
                y_true, y_pred, strategy=config.cal_strategy, n_bins=config.n_cal_bins
            )
            ece = float(np.mean(np.abs(prob_true - prob_pred)))
            ax.plot(
                prob_pred * scale, prob_true * scale, "o-", color=color, lw=2, markersize=6,
                label=f"{name}\n(ECE={ece:.4f})",
            )
            xy_max = max(xy_max, prob_pred.max(), prob_true.max())
        xy_max *= scale * 1.15
        ax.plot([0, xy_max], [0, xy_max], "k--", alpha=0.4, lw=1)
        ax.set_xlim(0, xy_max)
        ax.set_ylim(0, xy_max)
        ax.set_xlabel("Mean Predicted (x1e-3)", fontsize=11)
        ax.set_ylabel("Fraction of Positives (x1e-3)", fontsize=11)
        ax.set_title("Calibration", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8.5, framealpha=0.9, loc="upper left")

        # 2. ROC curve overlay
        ax = axes[1]
        for (name, y_true, y_pred), color in zip(models, colors):
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} ({auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curve", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8.5, framealpha=0.9, loc="lower right", title="AUC", title_fontsize=9)

        # 3. IEB comparison bar chart
        ax = axes[2]
        names, iebs = [], []
        for (name, y_true, y_pred) in models:
            ieb = abs(y_pred.mean() - y_true.mean()) / y_true.mean()
            names.append(name)
            iebs.append(ieb)
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, iebs, color=colors, height=0.5, edgecolor="white", lw=1.5)
        for bar, ieb in zip(bars, iebs):
            ax.text(
                bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{ieb:.3f}", va="center", fontsize=11, fontweight="bold",
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel("IEB (|mean bias| / actual CTR)", fontsize=11)
        ax.set_title("Calibration Bias (IEB)", fontsize=12, fontweight="bold")
        ax.set_xlim(0, max(iebs) * 1.35)
        ax.invert_yaxis()

        fig.suptitle(
            "Model Comparison — Prediction Diagnostics",
            fontsize=14, fontweight="bold", y=1.01,
        )
        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=config.dpi, bbox_inches="tight", facecolor="white")

    return fig
