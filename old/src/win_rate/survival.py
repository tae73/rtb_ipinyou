"""
Survival Analysis–based Market Price CDF Estimation.

Uses right-censored bid data to estimate the true market price
distribution via Kaplan-Meier and parametric (Weibull, LogNormal,
Exponential) survival models.

Key insight:
- Win=1 → market_price (=payprice) observed → "event"
- Win=0 → market_price > bidprice → right-censored at bidprice
- S(p) = P(market_price > p), F(p) = 1 - S(p) = P(Win | bid=p)

The resulting CDF F(p) is the direct input for SP3 bid shading:
  shade(x) = argmax_b (v - b) * F(b)
"""

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from lifelines import (
        ExponentialFitter,
        KaplanMeierFitter,
        LogNormalFitter,
        WeibullFitter,
    )
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result NamedTuples
# ---------------------------------------------------------------------------

class SurvivalFitResult(NamedTuple):
    """Result of Kaplan-Meier market price CDF estimation."""
    price_grid: np.ndarray
    survival: np.ndarray       # S(p) = P(market_price > p)
    cdf: np.ndarray            # F(p) = 1 - S(p) = P(Win|bid=p)
    ci_lower: np.ndarray       # S(p) lower CI
    ci_upper: np.ndarray       # S(p) upper CI
    median_price: float        # Median market price (S(p)=0.5)
    fitter: object             # KaplanMeierFitter instance (or None)


class ParametricFitResult(NamedTuple):
    """Result of parametric survival model fit."""
    distribution: str
    params: Dict[str, float]
    aic: float
    bic: float
    cdf_fn: Callable[[np.ndarray], np.ndarray]
    pdf_fn: Callable[[np.ndarray], np.ndarray]
    fitter: object


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_survival_data(
    bidprice: np.ndarray,
    payprice: np.ndarray,
    win: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert RTB data to survival analysis format.

    Returns:
        durations: Observed time (payprice if won, bidprice if lost).
        event_observed: 1 if won (price observed), 0 if censored.
    """
    bidprice = np.asarray(bidprice, dtype=np.float64)
    payprice = np.asarray(payprice, dtype=np.float64)
    win = np.asarray(win, dtype=np.float64)

    durations = np.where(win == 1, payprice, bidprice)
    event_observed = win.astype(bool)

    # Ensure positive durations (lifelines requirement)
    durations = np.maximum(durations, 0.01)

    return durations, event_observed


def _numpy_km(
    durations: np.ndarray,
    event_observed: np.ndarray,
    price_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback Kaplan-Meier product-limit estimator (numpy only).

    No confidence intervals — use lifelines for full output.
    """
    # Sort by duration
    order = np.argsort(durations)
    t_sorted = durations[order]
    e_sorted = event_observed[order]

    n = len(t_sorted)
    unique_times = np.unique(t_sorted[e_sorted])

    survival = np.ones(len(unique_times) + 1)
    times = np.concatenate([[0.0], unique_times])

    for i, t in enumerate(unique_times):
        at_risk = np.sum(t_sorted >= t)
        events = np.sum((t_sorted == t) & e_sorted)
        if at_risk > 0:
            survival[i + 1] = survival[i] * (1.0 - events / at_risk)
        else:
            survival[i + 1] = survival[i]

    # Interpolate to price_grid (step function)
    surv_on_grid = np.ones(len(price_grid))
    for j, p in enumerate(price_grid):
        idx = np.searchsorted(times, p, side="right") - 1
        idx = max(0, min(idx, len(survival) - 1))
        surv_on_grid[j] = survival[idx]

    cdf_on_grid = 1.0 - surv_on_grid
    return surv_on_grid, cdf_on_grid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_market_cdf_km(
    bidprice: np.ndarray,
    payprice: np.ndarray,
    win: np.ndarray,
    price_grid: Optional[np.ndarray] = None,
) -> SurvivalFitResult:
    """Estimate market price CDF using Kaplan-Meier estimator.

    Treats won auctions as observed events (at payprice) and lost
    auctions as right-censored (at bidprice).

    Args:
        bidprice: Bid prices for all auctions.
        payprice: Pay prices (meaningful only for winners).
        win: Win indicator (0/1).
        price_grid: Evaluation grid. If None, uses linspace(0, max, 500).

    Returns:
        SurvivalFitResult with S(p), F(p), CIs, and median.
    """
    durations, event_observed = _prepare_survival_data(bidprice, payprice, win)

    if price_grid is None:
        price_grid = np.linspace(0, np.percentile(durations, 99.5), 500)
    price_grid = np.asarray(price_grid, dtype=np.float64)

    if LIFELINES_AVAILABLE:
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=event_observed)

        # Evaluate on price grid
        timeline = pd.Series(price_grid)
        surv_df = kmf.survival_function_at_times(timeline)
        survival = surv_df.values.flatten()

        ci_df = kmf.confidence_interval_survival_function_
        ci_times = ci_df.index.values
        ci_lo_vals = ci_df.iloc[:, 0].values
        ci_hi_vals = ci_df.iloc[:, 1].values

        ci_lower = np.interp(price_grid, ci_times, ci_lo_vals)
        ci_upper = np.interp(price_grid, ci_times, ci_hi_vals)

        median_price = float(kmf.median_survival_time_)
        fitter = kmf
    else:
        survival, cdf = _numpy_km(durations, event_observed, price_grid)
        ci_lower = np.full_like(survival, np.nan)
        ci_upper = np.full_like(survival, np.nan)
        median_idx = np.searchsorted(1.0 - survival, 0.5)
        median_price = float(
            price_grid[min(median_idx, len(price_grid) - 1)]
        )
        fitter = None

    cdf = 1.0 - survival

    return SurvivalFitResult(
        price_grid=price_grid,
        survival=survival,
        cdf=cdf,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        median_price=median_price,
        fitter=fitter,
    )


def estimate_market_cdf_parametric(
    bidprice: np.ndarray,
    payprice: np.ndarray,
    win: np.ndarray,
    distribution: str = "lognormal",
) -> ParametricFitResult:
    """Fit parametric survival distribution to market price data.

    Supported distributions: 'weibull', 'lognormal', 'exponential'.

    Args:
        bidprice: Bid prices for all auctions.
        payprice: Pay prices (meaningful only for winners).
        win: Win indicator (0/1).
        distribution: Distribution family to fit.

    Returns:
        ParametricFitResult with params, AIC/BIC, and callable CDF/PDF.

    Raises:
        ImportError: If lifelines is not installed.
        ValueError: If distribution name is unknown.
    """
    if not LIFELINES_AVAILABLE:
        raise ImportError(
            "lifelines is required for parametric fitting. "
            "Install with: pip install lifelines>=0.29.0"
        )

    durations, event_observed = _prepare_survival_data(bidprice, payprice, win)

    fitter_map = {
        "weibull": WeibullFitter,
        "lognormal": LogNormalFitter,
        "exponential": ExponentialFitter,
    }

    if distribution not in fitter_map:
        raise ValueError(
            f"Unknown distribution: {distribution}. "
            f"Choose from: {list(fitter_map.keys())}"
        )

    fitter = fitter_map[distribution]()
    fitter.fit(durations, event_observed=event_observed)

    # Extract parameters
    params = {col: float(fitter.summary.loc[col, "coef"]) for col in fitter.summary.index}

    # AIC / BIC
    n = len(durations)
    k = len(params)
    aic = float(-2 * fitter.log_likelihood_ + 2 * k) if hasattr(fitter, "log_likelihood_") else float("nan")
    bic = float(-2 * fitter.log_likelihood_ + k * np.log(n)) if hasattr(fitter, "log_likelihood_") else float("nan")

    def cdf_fn(prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=np.float64)
        timeline = pd.Series(np.maximum(prices, 0.01))
        sf = fitter.survival_function_at_times(timeline).values.flatten()
        return 1.0 - sf

    def pdf_fn(prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=np.float64)
        cdf_vals = cdf_fn(prices)
        # Numerical differentiation
        dp = np.maximum(prices * 0.001, 0.1)
        cdf_plus = cdf_fn(prices + dp)
        return np.maximum((cdf_plus - cdf_vals) / dp, 0.0)

    return ParametricFitResult(
        distribution=distribution,
        params=params,
        aic=aic,
        bic=bic,
        cdf_fn=cdf_fn,
        pdf_fn=pdf_fn,
        fitter=fitter,
    )


def compare_parametric_fits(
    bidprice: np.ndarray,
    payprice: np.ndarray,
    win: np.ndarray,
) -> pd.DataFrame:
    """Compare Weibull, LogNormal, and Exponential fits via AIC/BIC.

    Args:
        bidprice: Bid prices for all auctions.
        payprice: Pay prices (meaningful only for winners).
        win: Win indicator (0/1).

    Returns:
        DataFrame with distribution, AIC, BIC, and parameters.
    """
    distributions = ["weibull", "lognormal", "exponential"]
    rows = []

    for dist in distributions:
        try:
            result = estimate_market_cdf_parametric(bidprice, payprice, win, dist)
            rows.append({
                "distribution": dist,
                "aic": result.aic,
                "bic": result.bic,
                "params": str(result.params),
            })
        except Exception as e:
            rows.append({
                "distribution": dist,
                "aic": float("nan"),
                "bic": float("nan"),
                "params": f"ERROR: {e}",
            })

    return (
        pd.DataFrame(rows)
        .sort_values("aic")
        .reset_index(drop=True)
    )


def segment_market_cdf(
    df: pd.DataFrame,
    segment_col: str,
    price_grid: Optional[np.ndarray] = None,
    min_samples: int = 1000,
) -> Dict[str, SurvivalFitResult]:
    """Estimate KM market price CDF per segment.

    Args:
        df: DataFrame with 'bidprice', 'payprice', 'win', and segment_col.
        segment_col: Column to segment by (e.g., 'adexchange').
        price_grid: Shared evaluation grid across segments.
        min_samples: Minimum observations per segment.

    Returns:
        Dict mapping segment value → SurvivalFitResult.
    """
    results: Dict[str, SurvivalFitResult] = {}

    for seg_val, grp in df.groupby(segment_col):
        if len(grp) < min_samples:
            continue
        results[str(seg_val)] = estimate_market_cdf_km(
            bidprice=grp["bidprice"].values,
            payprice=grp["payprice"].values,
            win=grp["win"].values,
            price_grid=price_grid,
        )

    return results


def validate_cdf_vs_empirical(
    survival_result: SurvivalFitResult,
    payprice_observed: np.ndarray,
) -> Dict[str, float]:
    """Validate KM CDF against empirical CDF of observed pay prices.

    Computes KS statistic and mean absolute difference.  Note: the
    empirical CDF (winners-only) is *biased* — differences are expected
    because KM corrects for censoring.

    Args:
        survival_result: KM fit result.
        payprice_observed: Observed pay prices (winners only).

    Returns:
        Dict with 'ks_statistic', 'ks_pvalue', 'mean_abs_diff'.
    """
    from scipy.stats import ks_2samp

    payprice_observed = np.asarray(payprice_observed, dtype=np.float64)

    # Empirical CDF of observed prices
    sorted_obs = np.sort(payprice_observed)
    ecdf_obs = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs)

    # KM CDF at same points
    km_cdf_at_obs = np.interp(
        sorted_obs,
        survival_result.price_grid,
        survival_result.cdf,
    )

    mean_abs_diff = float(np.mean(np.abs(km_cdf_at_obs - ecdf_obs)))

    # KS test (2-sample approximation)
    ks_stat, ks_pvalue = ks_2samp(
        km_cdf_at_obs,
        ecdf_obs,
    )

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "mean_abs_diff": mean_abs_diff,
    }
