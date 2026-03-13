"""Comparison table generation for unified model evaluation."""

from typing import List

import pandas as pd

from .result_loader import UnifiedMetrics


def build_comparison_table(metrics_list: List[UnifiedMetrics]) -> pd.DataFrame:
    """Build comparison DataFrame from UnifiedMetrics list.

    Args:
        metrics_list: List of UnifiedMetrics from different models

    Returns:
        DataFrame with columns: Model, Type, Context, AUC, Log Loss, ECE, IEB
    """
    rows = [
        {
            "Model": m.model_name,
            "Type": m.model_type,
            "Context": m.eval_context,
            "AUC": m.auc,
            "Log Loss": m.log_loss,
            "ECE": m.ece,
            "IEB": m.ieb,
        }
        for m in metrics_list
    ]
    return pd.DataFrame(rows)


def highlight_best(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Highlight best value per numeric column.

    - AUC: higher is better (max)
    - Log Loss, ECE, IEB: lower is better (min)

    Args:
        df: Comparison DataFrame from build_comparison_table()

    Returns:
        Styled DataFrame
    """
    def _highlight_col(s: pd.Series, higher_better: bool = True) -> list:
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.isna().all():
            return [""] * len(s)
        best_idx = numeric.idxmax() if higher_better else numeric.idxmin()
        return [
            "font-weight: bold; background-color: #d4edda" if i == best_idx else ""
            for i in s.index
        ]

    styler = df.style
    if "AUC" in df.columns:
        styler = styler.apply(_highlight_col, higher_better=True, subset=["AUC"])
    for col in ["Log Loss", "ECE", "IEB"]:
        if col in df.columns:
            styler = styler.apply(_highlight_col, higher_better=False, subset=[col])

    return styler.format(
        {
            "AUC": "{:.4f}",
            "Log Loss": lambda x: f"{x:.4f}" if x is not None else "—",
            "ECE": "{:.2e}",
            "IEB": lambda x: f"{x:.4f}" if x is not None else "—",
        },
        na_rep="—",
    )
