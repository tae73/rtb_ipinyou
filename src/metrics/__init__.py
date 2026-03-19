"""Metrics modules — evaluation, result loading, comparison, and diagnostics."""

from .evaluation import EvalMetrics, compute_ece, compute_ieb, compute_metrics
from .result_loader import (
    UnifiedMetrics,
    load_and_normalize,
    load_baseline_result,
    load_neural_result,
    normalize_to_unified,
)
from .comparison import build_comparison_table, highlight_best
from .diagnostics_plot import DiagnosticsPlotConfig, plot_prediction_diagnostics

__all__ = [
    "EvalMetrics",
    "compute_ece",
    "compute_ieb",
    "compute_metrics",
    "UnifiedMetrics",
    "load_and_normalize",
    "load_baseline_result",
    "load_neural_result",
    "normalize_to_unified",
    "build_comparison_table",
    "highlight_best",
    "DiagnosticsPlotConfig",
    "plot_prediction_diagnostics",
]
