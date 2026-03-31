"""
Impression Value Computation for RTB Bid Optimization.

Connects debiased pCTR (SP1 output) to economic impression value V(x).
Primary: V(x) = debiased_pCTR(x) × CPC_target (CPM units).

⚠ Must use debiased pCTR (ESCM²-WC(DR)):
  - Biased pCTR → V(x) overestimation → overbidding
  - See NB05 Section 11: IEB 0.014 → oracle surplus, IEB 0.362 → 25.9× overbid
"""

from typing import Dict, NamedTuple, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration & Result Types
# ---------------------------------------------------------------------------

class ValueConfig(NamedTuple):
    """Value computation configuration.

    All monetary values in CPM units (cost per mille impressions).
    CPC campaign (primary): V(x) = pCTR × cpc_target
      - cpc_target = 200,000 CPM/click (NB05 Section 11 기준)
      - V_TRUE = 0.0008 × 200,000 = 160 CPM ≈ market price median 68
    """
    goal_type: str = "CPC"            # CPC, CPA, CPM
    cpc_target: float = 200_000.0     # CPM per click
    cpa_target: float = 0.0           # CPM per action (CPA only)
    cpm_target: float = 0.0           # fixed CPM value (CPM only)


class ValueResult(NamedTuple):
    """Per-impression value computation result."""
    values: np.ndarray                # V(x) array (CPM)
    mean_value: float
    median_value: float
    std_value: float
    pct_above_market_median: float    # fraction V(x) > market median (68 CPM)


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def compute_impression_values(
    p_ctr: np.ndarray,
    config: Optional[ValueConfig] = None,
    market_median: float = 68.0,
) -> ValueResult:
    """Compute per-impression values V(x) = pCTR × CPC_target.

    Args:
        p_ctr: Predicted click-through rates (debiased, from ESCM²-WC(DR)).
        config: Value configuration. Defaults to CPC with 200K CPM/click.
        market_median: Market price median for pct_above_market_median stat.

    Returns:
        ValueResult with V(x) array and summary statistics.
    """
    if config is None:
        config = ValueConfig()

    p_ctr = np.asarray(p_ctr, dtype=np.float64)

    if config.goal_type == "CPC":
        values = p_ctr * config.cpc_target
    elif config.goal_type == "CPM":
        values = np.full_like(p_ctr, config.cpm_target / 1000.0)
    else:
        raise ValueError(f"Unknown goal_type: {config.goal_type}. Use compute_impression_values_cpa for CPA.")

    return ValueResult(
        values=values,
        mean_value=float(np.mean(values)),
        median_value=float(np.median(values)),
        std_value=float(np.std(values)),
        pct_above_market_median=float(np.mean(values > market_median)),
    )


def compute_impression_values_cpa(
    p_ctr: np.ndarray,
    p_cvr: np.ndarray,
    config: Optional[ValueConfig] = None,
    market_median: float = 68.0,
) -> ValueResult:
    """Compute per-impression values for CPA campaigns.

    V(x) = pCTR × pCVR × CPA_target.
    Optional path for retargeting advertisers (CVR near-trivial for branding).

    Args:
        p_ctr: Predicted CTR (debiased).
        p_cvr: Predicted CVR (click → conversion).
        config: Value configuration with cpa_target set.
        market_median: Market price median.

    Returns:
        ValueResult with V(x) array and summary statistics.
    """
    if config is None:
        config = ValueConfig(goal_type="CPA", cpa_target=50_000.0)

    p_ctr = np.asarray(p_ctr, dtype=np.float64)
    p_cvr = np.asarray(p_cvr, dtype=np.float64)
    values = p_ctr * p_cvr * config.cpa_target

    return ValueResult(
        values=values,
        mean_value=float(np.mean(values)),
        median_value=float(np.median(values)),
        std_value=float(np.std(values)),
        pct_above_market_median=float(np.mean(values > market_median)),
    )


def compare_value_distributions(
    values_dict: Dict[str, np.ndarray],
    market_stats: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compare value distributions across multiple pCTR sources.

    Quantifies how IEB (calibration error) distorts impression values.

    Args:
        values_dict: {model_name: V(x) array}. E.g., {"ESCM2-DR": v1, "LGB_biased": v2}.
        market_stats: Market price statistics dict (mean, median, p75, p90).

    Returns:
        DataFrame with per-model value statistics and overbid ratios.
    """
    if market_stats is None:
        market_stats = {"mean": 78.05, "median": 68.0, "p75": 93.0, "p90": 166.0}

    records = []
    for name, values in values_dict.items():
        v = np.asarray(values, dtype=np.float64)
        records.append({
            "model": name,
            "mean_value": float(np.mean(v)),
            "median_value": float(np.median(v)),
            "std_value": float(np.std(v)),
            "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)),
            "p90": float(np.percentile(v, 90)),
            "p95": float(np.percentile(v, 95)),
            "pct_above_market_median": float(np.mean(v > market_stats["median"])),
            "pct_above_market_p90": float(np.mean(v > market_stats["p90"])),
            "overbid_ratio_mean": float(np.mean(v)) / market_stats["mean"],
        })

    return pd.DataFrame(records).set_index("model")
