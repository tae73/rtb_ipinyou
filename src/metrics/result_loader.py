"""JSON result loading and schema normalization.

Loads baseline (LGB/LR) and neural (ESMM-WC/ESCM2-WC) result JSONs
into a unified schema for cross-model comparison.
"""

import json
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional


class UnifiedMetrics(NamedTuple):
    """Unified metrics across all model types."""
    model_name: str
    model_type: str           # "lgb_ctr", "lr_win", "esmmwc", "escm2wc_dr" etc.
    eval_context: str         # "won_only" | "all_bids"
    auc: float
    log_loss: Optional[float]  # Neural models may not have this
    ece: float
    ieb: Optional[float]      # Baseline requires recomputation


def load_baseline_result(path: Path) -> dict:
    """Load baseline JSON (LGB/LR) into standard dict.

    Baseline JSON schema:
        model_name, task, n_features, training_time,
        train_metrics/val_metrics/test_metrics: {auc, log_loss, accuracy, ...}
    """
    with open(path) as f:
        return json.load(f)


def load_neural_result(path: Path) -> dict:
    """Load neural model JSON (ESMM-WC/ESCM2-WC) into standard dict.

    Neural JSON schema:
        model_name, model_type, debiasing, config, training_time,
        test_win_auc, test_ctr_biased_auc, test_ctr_biased_ece,
        test_wctr_auc, test_wctr_ece, test_ctr_ieb, test_wctr_ieb
    """
    with open(path) as f:
        return json.load(f)


def _normalize_baseline(result: dict, eval_context: str) -> UnifiedMetrics:
    """Normalize baseline result dict to UnifiedMetrics."""
    test = result.get("test_metrics", {})
    model_name = result.get("model_name", "unknown")
    task = result.get("task", "")

    # Model type from model_name (e.g. "lgb_ctr", "lr_win")
    model_type = model_name

    return UnifiedMetrics(
        model_name=model_name,
        model_type=model_type,
        eval_context=eval_context,
        auc=test.get("auc", 0.0),
        log_loss=test.get("log_loss"),
        ece=test.get("ece", 0.0),
        ieb=test.get("ieb"),  # None if not computed
    )


def _normalize_neural_won_only(result: dict) -> UnifiedMetrics:
    """Normalize neural result for won-only (CTR biased) context."""
    model_name = result.get("model_name", "unknown")
    debiasing = result.get("debiasing", "none")
    # Use model_type field if present, else derive from name + debiasing
    model_type = result.get("model_type", model_name)
    if debiasing != "none" and not model_type.endswith(f"_{debiasing}"):
        model_type = f"{model_type}_{debiasing}"

    return UnifiedMetrics(
        model_name=f"{model_name} CTR (biased)",
        model_type=model_type,
        eval_context="won_only",
        auc=result.get("test_ctr_biased_auc", 0.0),
        log_loss=None,
        ece=result.get("test_ctr_biased_ece", 0.0),
        ieb=result.get("test_ctr_ieb"),
    )


def _normalize_neural_all_bids(result: dict) -> UnifiedMetrics:
    """Normalize neural result for all-bids (WCTR) context."""
    model_name = result.get("model_name", "unknown")
    debiasing = result.get("debiasing", "none")
    model_type = result.get("model_type", model_name)
    if debiasing != "none" and not model_type.endswith(f"_{debiasing}"):
        model_type = f"{model_type}_{debiasing}"

    return UnifiedMetrics(
        model_name=f"{model_name} WCTR",
        model_type=model_type,
        eval_context="all_bids",
        auc=result.get("test_wctr_auc", 0.0),
        log_loss=None,
        ece=result.get("test_wctr_ece", 0.0),
        ieb=result.get("test_wctr_ieb"),
    )


def normalize_to_unified(
    result: dict,
    eval_context: str,
) -> UnifiedMetrics:
    """Convert any result dict to UnifiedMetrics.

    Args:
        result: Loaded JSON dict (baseline or neural)
        eval_context: "won_only" or "all_bids"

    Returns:
        UnifiedMetrics with normalized fields
    """
    # Detect neural vs baseline by presence of neural-specific fields
    is_neural = "test_win_auc" in result or "test_wctr_auc" in result

    if is_neural:
        if eval_context == "won_only":
            return _normalize_neural_won_only(result)
        else:
            return _normalize_neural_all_bids(result)
    else:
        return _normalize_baseline(result, eval_context)


def load_and_normalize(
    path: Path,
    eval_context: str,
) -> UnifiedMetrics:
    """Load JSON and normalize in one step.

    Args:
        path: Path to JSON result file
        eval_context: "won_only" or "all_bids"
    """
    with open(path) as f:
        result = json.load(f)
    return normalize_to_unified(result, eval_context)
