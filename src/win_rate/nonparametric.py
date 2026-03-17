"""
Non-parametric Win Rate Analysis & Market Price Statistics.

Provides empirical win rate curves, market price statistics,
Wilson confidence intervals, and serving lookup tables for
RTB bid optimization (SP3 input).
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result NamedTuples
# ---------------------------------------------------------------------------

class WinRateCurveResult(NamedTuple):
    """Result of empirical win rate curve estimation."""
    bid_midpoints: np.ndarray
    win_rates: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    n_per_bin: np.ndarray


class MarketPriceStats(NamedTuple):
    """Summary statistics for observed market prices (winners only)."""
    mean: float
    median: float
    std: float
    p25: float
    p75: float
    p90: float
    p95: float
    n_observed: int
    floor_binding_rate: float


class WinRateLookup(NamedTuple):
    """Serving lookup table: price grid → win rates per segment."""
    price_grid: np.ndarray
    win_rates: Dict[str, np.ndarray]
    default_win_rates: np.ndarray


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def wilson_ci(
    successes: np.ndarray,
    total: np.ndarray,
    z: float = 1.96,
) -> Tuple[np.ndarray, np.ndarray]:
    """Wilson score confidence interval for binomial proportions.

    More accurate than normal approximation for small samples or
    extreme proportions — avoids overshooting [0, 1].

    Args:
        successes: Number of successes per bin.
        total: Total observations per bin.
        z: Z-score for confidence level (default 1.96 = 95%).

    Returns:
        Tuple of (ci_lower, ci_upper) arrays.
    """
    successes = np.asarray(successes, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)

    # Avoid division by zero
    safe_total = np.maximum(total, 1)
    p_hat = successes / safe_total
    z2 = z ** 2

    denom = 1 + z2 / safe_total
    center = (p_hat + z2 / (2 * safe_total)) / denom
    margin = (z / denom) * np.sqrt(
        p_hat * (1 - p_hat) / safe_total + z2 / (4 * safe_total ** 2)
    )

    ci_lower = np.clip(center - margin, 0.0, 1.0)
    ci_upper = np.clip(center + margin, 0.0, 1.0)

    # Zero-total bins → NaN
    mask_zero = total == 0
    ci_lower[mask_zero] = np.nan
    ci_upper[mask_zero] = np.nan

    return ci_lower, ci_upper


def empirical_win_rate_curve(
    bidprice: np.ndarray,
    win: np.ndarray,
    n_bins: int = 50,
    min_samples: int = 100,
) -> WinRateCurveResult:
    """Compute non-parametric win rate curve with Wilson CIs.

    Groups bids into equal-frequency bins (quantiles) by bid price
    and computes win rate + confidence interval per bin.  Bins with
    fewer than *min_samples* observations are dropped.

    Note: iPinYou uses flat bidding (6 discrete bid values), so this
    function uses cross-advertiser pooling and may produce fewer bins
    than requested.

    Args:
        bidprice: Bid prices array.
        win: Win indicator array (0/1).
        n_bins: Target number of bins (may be fewer due to ties).
        min_samples: Minimum observations per bin.

    Returns:
        WinRateCurveResult with midpoints, rates, CIs, and counts.
    """
    bidprice = np.asarray(bidprice, dtype=np.float64)
    win = np.asarray(win, dtype=np.float64)

    # Use unique bid values when fewer than n_bins
    unique_bids = np.unique(bidprice)
    if len(unique_bids) <= n_bins:
        # Group by exact bid value
        midpoints, rates, lowers, uppers, counts = [], [], [], [], []
        for b in unique_bids:
            mask = bidprice == b
            n = mask.sum()
            if n < min_samples:
                continue
            w = win[mask].sum()
            rate = w / n
            ci_lo, ci_hi = wilson_ci(np.array([w]), np.array([n]))
            midpoints.append(b)
            rates.append(rate)
            lowers.append(ci_lo[0])
            uppers.append(ci_hi[0])
            counts.append(n)
    else:
        # Quantile binning
        try:
            bin_edges = np.unique(
                np.percentile(bidprice, np.linspace(0, 100, n_bins + 1))
            )
        except Exception:
            bin_edges = np.linspace(bidprice.min(), bidprice.max(), n_bins + 1)

        bin_indices = np.digitize(bidprice, bin_edges[1:-1])

        midpoints, rates, lowers, uppers, counts = [], [], [], [], []
        for i in range(len(bin_edges) - 1):
            mask = bin_indices == i
            n = mask.sum()
            if n < min_samples:
                continue
            w = win[mask].sum()
            rate = w / n
            mid = (bin_edges[i] + bin_edges[i + 1]) / 2
            ci_lo, ci_hi = wilson_ci(np.array([w]), np.array([n]))
            midpoints.append(mid)
            rates.append(rate)
            lowers.append(ci_lo[0])
            uppers.append(ci_hi[0])
            counts.append(n)

    return WinRateCurveResult(
        bid_midpoints=np.array(midpoints),
        win_rates=np.array(rates),
        ci_lower=np.array(lowers),
        ci_upper=np.array(uppers),
        n_per_bin=np.array(counts, dtype=np.int64),
    )


def compute_market_price_stats(
    payprice: np.ndarray,
    slotprice: np.ndarray,
) -> MarketPriceStats:
    """Compute summary statistics of observed market prices (winners only).

    Args:
        payprice: Observed pay prices (winners only, second-price or floor).
        slotprice: Floor prices (slot reserve prices).

    Returns:
        MarketPriceStats with mean, median, percentiles, floor binding rate.
    """
    payprice = np.asarray(payprice, dtype=np.float64)
    slotprice = np.asarray(slotprice, dtype=np.float64)

    floor_binding_rate = float(np.mean(np.abs(payprice - slotprice) < 1.0))

    return MarketPriceStats(
        mean=float(np.mean(payprice)),
        median=float(np.median(payprice)),
        std=float(np.std(payprice)),
        p25=float(np.percentile(payprice, 25)),
        p75=float(np.percentile(payprice, 75)),
        p90=float(np.percentile(payprice, 90)),
        p95=float(np.percentile(payprice, 95)),
        n_observed=len(payprice),
        floor_binding_rate=floor_binding_rate,
    )


def empirical_market_cdf(
    payprice: np.ndarray,
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical CDF of observed market prices (winners only).

    This is a *biased* baseline: only includes won auctions.
    The true market price distribution includes unobserved prices
    above the bid — use survival analysis for unbiased estimation.

    Args:
        payprice: Observed pay prices (winners only).
        n_points: Number of evaluation points on the price grid.

    Returns:
        Tuple of (price_grid, cdf_values).
    """
    payprice = np.asarray(payprice, dtype=np.float64)
    sorted_prices = np.sort(payprice)
    price_grid = np.linspace(sorted_prices.min(), sorted_prices.max(), n_points)
    cdf_values = np.searchsorted(sorted_prices, price_grid, side="right") / len(
        sorted_prices
    )
    return price_grid, cdf_values


def build_win_rate_lookup(
    df: pd.DataFrame,
    price_grid: np.ndarray,
    segment_cols: Optional[List[str]] = None,
) -> WinRateLookup:
    """Build a serving-ready win rate lookup table.

    For each segment (defined by *segment_cols*), computes empirical
    win rate at each price in *price_grid* using linear interpolation
    from observed bid-level win rates.

    Args:
        df: DataFrame with 'bidprice' and 'win' columns.
        price_grid: Array of price points for lookup.
        segment_cols: Columns defining segments (e.g., ['adexchange']).
            If None, builds a single default curve.

    Returns:
        WinRateLookup with interpolated win rates per segment.
    """
    price_grid = np.asarray(price_grid, dtype=np.float64)

    # Default (overall) curve
    curve = empirical_win_rate_curve(
        df["bidprice"].values, df["win"].values, n_bins=50
    )
    default_wr = np.interp(price_grid, curve.bid_midpoints, curve.win_rates)

    win_rates: Dict[str, np.ndarray] = {"__default__": default_wr}

    if segment_cols:
        for keys, grp in df.groupby(segment_cols):
            key_str = str(keys)
            if len(grp) < 200:
                continue
            seg_curve = empirical_win_rate_curve(
                grp["bidprice"].values, grp["win"].values, n_bins=50
            )
            if len(seg_curve.bid_midpoints) < 2:
                win_rates[key_str] = default_wr.copy()
            else:
                win_rates[key_str] = np.interp(
                    price_grid, seg_curve.bid_midpoints, seg_curve.win_rates
                )

    return WinRateLookup(
        price_grid=price_grid,
        win_rates=win_rates,
        default_win_rates=default_wr,
    )


def lookup_win_prob(
    lookup: WinRateLookup,
    price: float,
    segment_key: Optional[str] = None,
) -> float:
    """Look up interpolated win probability at a given price.

    Args:
        lookup: Pre-built WinRateLookup table.
        price: Bid price to query.
        segment_key: Segment identifier (falls back to default if missing).

    Returns:
        Estimated win probability.
    """
    key = segment_key if segment_key and segment_key in lookup.win_rates else "__default__"
    wr_array = lookup.win_rates[key]
    return float(np.interp(price, lookup.price_grid, wr_array))
