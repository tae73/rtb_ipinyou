"""Win rate analysis modules.

Non-parametric win rate curves, survival-based market price CDF estimation,
and serving lookup tables for SP3 bid optimization.
"""

from src.win_rate.nonparametric import (
    MarketPriceStats,
    WinRateCurveResult,
    WinRateLookup,
    build_win_rate_lookup,
    compute_market_price_stats,
    empirical_market_cdf,
    empirical_win_rate_curve,
    lookup_win_prob,
    wilson_ci,
)
from src.win_rate.survival import (
    ParametricFitResult,
    SurvivalFitResult,
    compare_parametric_fits,
    estimate_market_cdf_km,
    estimate_market_cdf_parametric,
    segment_market_cdf,
    validate_cdf_vs_empirical,
)

__all__ = [
    # NamedTuples
    "WinRateCurveResult",
    "MarketPriceStats",
    "WinRateLookup",
    "SurvivalFitResult",
    "ParametricFitResult",
    # Non-parametric
    "wilson_ci",
    "empirical_win_rate_curve",
    "compute_market_price_stats",
    "empirical_market_cdf",
    "build_win_rate_lookup",
    "lookup_win_prob",
    # Survival
    "estimate_market_cdf_km",
    "estimate_market_cdf_parametric",
    "compare_parametric_fits",
    "segment_market_cdf",
    "validate_cdf_vs_empirical",
]
