"""
First-Price Bid Shading for RTB Bid Optimization.

Implements distribution-based optimal bid shading using market price CDF
(Kaplan-Meier estimates from SP2).

Core formula: b* = argmax_b (V - b) × F(b)
  where F(b) = P(market_price ≤ b) = KM CDF

Survey ref: Ou et al. (2024) Sec 6.2.1 — distribution estimators for
first-price auctions. The expected surplus (v-b)×P_win(b) has a unique
global maximum for log-normal, gamma, and truncated-normal distributions.

Strategies:
  1. optimal: numerical argmax via grid search (primary)
  2. linear: b = α×V (Ou et al. Eq.5, widely used in practice)
  3. percentile: bid at market price percentile
  4. dual_regime: floor-aware + competitive (EDA-driven, 32.24% floor binding)
"""

from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration & Result Types
# ---------------------------------------------------------------------------

class ShadingConfig(NamedTuple):
    """Bid shading configuration."""
    strategy: str = "optimal"           # optimal, linear, percentile, dual_regime
    min_bid: float = 1.0
    max_bid: float = 300.0
    linear_alpha: float = 0.8           # for linear strategy
    percentile_target: float = 0.75     # for percentile strategy
    floor_shading_factor: float = 1.05  # for floor-bound regime (bid slightly above floor)
    exchange_conditional: bool = True


class MarketCDF(NamedTuple):
    """Loaded market price CDF for shading computations."""
    price_grid: np.ndarray    # monotone price points
    cdf: np.ndarray           # F(p) = P(market_price ≤ p)
    median_price: float       # S(p)=0.5 point (inf if not reached)
    source: str               # "km_overall", "km_exchange_1", etc.


class ShadingResult(NamedTuple):
    """Per-impression shading result."""
    bids: np.ndarray              # final bid prices
    shading_factors: np.ndarray   # shade = bid / value (0-1)
    expected_surplus: np.ndarray  # (V - bid) × F(bid)
    expected_win_prob: np.ndarray # F(bid)
    regime: np.ndarray            # 0=competitive, 1=floor_bound


# ---------------------------------------------------------------------------
# CDF Loading
# ---------------------------------------------------------------------------

def load_market_cdf(path: str) -> MarketCDF:
    """Load KM CDF from .npz file.

    Args:
        path: Path to km_cdf_*.npz (price_grid, cdf, survival, median_price).

    Returns:
        MarketCDF NamedTuple.
    """
    data = np.load(path, allow_pickle=True)
    median_p = float(data["median_price"])
    source = path.rsplit("/", 1)[-1].replace(".npz", "")
    return MarketCDF(
        price_grid=data["price_grid"].astype(np.float64),
        cdf=data["cdf"].astype(np.float64),
        median_price=median_p,
        source=source,
    )


def load_exchange_cdfs(cdf_dir: str) -> Dict[str, MarketCDF]:
    """Load exchange-conditional CDFs.

    Args:
        cdf_dir: Directory containing km_cdf_exchange_*.npz files.

    Returns:
        Dict mapping exchange ID string to MarketCDF.
    """
    import os
    cdfs = {}
    for fname in sorted(os.listdir(cdf_dir)):
        if fname.startswith("km_cdf_exchange_") and fname.endswith(".npz"):
            ex_id = fname.replace("km_cdf_exchange_", "").replace(".npz", "")
            cdfs[ex_id] = load_market_cdf(os.path.join(cdf_dir, fname))
    return cdfs


# ---------------------------------------------------------------------------
# CDF Interpolation Helper
# ---------------------------------------------------------------------------

def _interpolate_cdf(bid_values: np.ndarray, market_cdf: MarketCDF) -> np.ndarray:
    """Interpolate CDF at arbitrary bid values.

    Args:
        bid_values: Bid prices to evaluate F(b) at.
        market_cdf: MarketCDF with price_grid and cdf.

    Returns:
        F(b) values, clipped to [0, 1].
    """
    return np.interp(bid_values, market_cdf.price_grid, market_cdf.cdf, left=0.0, right=market_cdf.cdf[-1])


# ---------------------------------------------------------------------------
# Shading Strategies
# ---------------------------------------------------------------------------

def optimal_bid_vectorized(
    values: np.ndarray,
    market_cdf: MarketCDF,
    n_candidates: int = 1000,
    min_bid: float = 1.0,
    max_bid: float = 300.0,
    batch_size: int = 100_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal first-price bids via distribution-based grid search.

    For each impression: b* = argmax_b (V - b) × F(b), subject to min_bid ≤ b ≤ min(V, max_bid).

    Uses batched numpy broadcasting for efficient computation on large arrays.
    Survey ref: Ou et al. (2024) Sec 6.2.1 distribution estimators.

    Args:
        values: V(x) array, shape (N,).
        market_cdf: Market price CDF (KM estimate).
        n_candidates: Number of bid candidates per impression.
        min_bid: Minimum allowed bid.
        max_bid: Maximum allowed bid.
        batch_size: Process this many impressions at a time.

    Returns:
        (optimal_bids, optimal_surplus) arrays, each shape (N,).
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    optimal_bids = np.zeros(n, dtype=np.float64)
    optimal_surplus = np.zeros(n, dtype=np.float64)

    # Pre-compute CDF on a shared fine grid for interpolation
    bid_grid = np.linspace(min_bid, max_bid, n_candidates)
    cdf_grid = _interpolate_cdf(bid_grid, market_cdf)  # shape (n_candidates,)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        v_batch = values[start:end]  # (B,)

        # Mask: bid must be ≤ value (can't bid more than value)
        # surplus_matrix[i, j] = (V[i] - bid_grid[j]) × F(bid_grid[j])
        margins = v_batch[:, None] - bid_grid[None, :]  # (B, n_candidates)
        surplus_matrix = margins * cdf_grid[None, :]     # (B, n_candidates)

        # Zero out bids above value (negative margin)
        surplus_matrix = np.where(margins > 0, surplus_matrix, -np.inf)

        # Find optimal bid index per impression
        best_idx = np.argmax(surplus_matrix, axis=1)  # (B,)
        optimal_bids[start:end] = bid_grid[best_idx]
        optimal_surplus[start:end] = np.maximum(
            surplus_matrix[np.arange(len(v_batch)), best_idx], 0.0
        )

    return optimal_bids, optimal_surplus


def linear_bid(
    values: np.ndarray,
    alpha: float = 0.8,
    min_bid: float = 1.0,
    max_bid: float = 300.0,
) -> np.ndarray:
    """Linear bid shading: b = α × V.

    Survey ref: Ou et al. Eq.5 — widely used in practice.
    b_i = α × v_i, where α < 1 for bid shading.

    Args:
        values: V(x) array.
        alpha: Shading factor (0 < α ≤ 1).
        min_bid: Minimum bid.
        max_bid: Maximum bid.

    Returns:
        Bid array clipped to [min_bid, max_bid].
    """
    bids = np.asarray(values, dtype=np.float64) * alpha
    return np.clip(bids, min_bid, max_bid)


def percentile_bid(
    values: np.ndarray,
    market_cdf: MarketCDF,
    target_percentile: float = 0.75,
    min_bid: float = 1.0,
    max_bid: float = 300.0,
) -> np.ndarray:
    """Bid at a target market price percentile.

    Finds price p such that F(p) = target_percentile, then bids min(p, V).

    Args:
        values: V(x) array.
        market_cdf: Market price CDF.
        target_percentile: Target win probability (e.g., 0.75).
        min_bid: Minimum bid.
        max_bid: Maximum bid.

    Returns:
        Bid array.
    """
    # Find price at target percentile via inverse CDF
    idx = np.searchsorted(market_cdf.cdf, target_percentile)
    idx = min(idx, len(market_cdf.price_grid) - 1)
    target_price = market_cdf.price_grid[idx]

    values = np.asarray(values, dtype=np.float64)
    bids = np.minimum(values, target_price)
    return np.clip(bids, min_bid, max_bid)


def dual_regime_shading(
    values: np.ndarray,
    market_cdf: MarketCDF,
    slotprice: np.ndarray,
    config: Optional[ShadingConfig] = None,
) -> ShadingResult:
    """Dual-regime bid shading: floor-bound vs competitive.

    EDA finding: 32.24% of won bids have payprice ≈ floor price (floor-binding).
    - Floor-bound regime: market_price ≈ floor → bid just above floor
    - Competitive regime: standard optimal shading

    The regime is detected using slotprice > 0 as a proxy for floor presence.
    Exchange 3 has active floors, Exchange 1 has minimal floors.

    Args:
        values: V(x) array.
        market_cdf: Market price CDF.
        slotprice: Floor prices per impression (0 if no floor).
        config: Shading configuration.

    Returns:
        ShadingResult with per-impression bids and regime labels.
    """
    if config is None:
        config = ShadingConfig(strategy="dual_regime")

    values = np.asarray(values, dtype=np.float64)
    slotprice = np.asarray(slotprice, dtype=np.float64)
    n = len(values)

    # Detect floor-bound regime: slotprice > 0 and significant relative to value
    # Only classify as floor-bound if floor is meaningful (> 10% of value)
    is_floor_bound = (slotprice > 0) & (slotprice > values * 0.1)
    regime = is_floor_bound.astype(np.int32)

    bids = np.zeros(n, dtype=np.float64)

    # Floor-bound regime: bid slightly above floor to minimize overpayment
    # Since market_price ≈ floor in this regime, bid = floor + small margin
    # floor_shading_factor > 1.0 means bid = floor * factor (e.g., 1.05 = 5% above floor)
    floor_mask = is_floor_bound
    if floor_mask.any():
        # Bid at floor * factor (just above floor to win)
        floor_factor = max(config.floor_shading_factor, 1.01)  # ensure above floor
        floor_bids = slotprice[floor_mask] * floor_factor
        floor_bids = np.maximum(floor_bids, config.min_bid)
        # Don't bid more than value
        floor_bids = np.minimum(floor_bids, values[floor_mask])
        bids[floor_mask] = floor_bids

    # Competitive regime: optimal shading
    comp_mask = ~is_floor_bound
    if comp_mask.any():
        opt_bids, _ = optimal_bid_vectorized(
            values[comp_mask], market_cdf,
            min_bid=config.min_bid, max_bid=config.max_bid,
        )
        bids[comp_mask] = opt_bids

    bids = np.clip(bids, config.min_bid, config.max_bid)
    shading_factors = np.where(values > 0, bids / values, 0.0)
    win_probs = _interpolate_cdf(bids, market_cdf)
    surplus = (values - bids) * win_probs

    return ShadingResult(
        bids=bids,
        shading_factors=shading_factors,
        expected_surplus=surplus,
        expected_win_prob=win_probs,
        regime=regime,
    )


def exchange_conditional_shading(
    values: np.ndarray,
    exchange_ids: np.ndarray,
    exchange_cdfs: Dict[str, MarketCDF],
    overall_cdf: MarketCDF,
    config: Optional[ShadingConfig] = None,
) -> ShadingResult:
    """Exchange-conditional optimal bid shading.

    Exchange 1 (median 153, F(300)=0.69) vs Exchange 2 (F(300)=0.29) vs
    Exchange 3 (F(300)=0.12) have dramatically different competition levels.

    Args:
        values: V(x) array.
        exchange_ids: Exchange ID per impression (1, 2, 3).
        exchange_cdfs: Dict of exchange-specific MarketCDF.
        overall_cdf: Fallback CDF for unknown exchanges.
        config: Shading configuration.

    Returns:
        ShadingResult.
    """
    if config is None:
        config = ShadingConfig()

    values = np.asarray(values, dtype=np.float64)
    # Handle pyarrow-backed or NA-containing exchange_ids
    exchange_ids = pd.array(exchange_ids).fillna(-1) if hasattr(pd.array(exchange_ids), 'fillna') else np.asarray(exchange_ids)
    exchange_ids = np.asarray(exchange_ids, dtype=np.int64)
    n = len(values)

    bids = np.zeros(n, dtype=np.float64)
    surplus = np.zeros(n, dtype=np.float64)

    unique_exchanges = np.unique(exchange_ids)
    for ex_id in unique_exchanges:
        mask = exchange_ids == ex_id
        ex_str = str(int(ex_id))
        cdf = exchange_cdfs.get(ex_str, overall_cdf)

        ex_bids, ex_surplus = optimal_bid_vectorized(
            values[mask], cdf,
            min_bid=config.min_bid, max_bid=config.max_bid,
        )
        bids[mask] = ex_bids
        surplus[mask] = ex_surplus

    bids = np.clip(bids, config.min_bid, config.max_bid)
    shading_factors = np.where(values > 0, bids / values, 0.0)
    # Win prob uses overall CDF as common reference
    win_probs = _interpolate_cdf(bids, overall_cdf)
    regime = np.zeros(n, dtype=np.int32)  # all competitive

    return ShadingResult(
        bids=bids,
        shading_factors=shading_factors,
        expected_surplus=surplus,
        expected_win_prob=win_probs,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# Convenience: apply any strategy
# ---------------------------------------------------------------------------

def compute_shaded_bids(
    values: np.ndarray,
    market_cdf: MarketCDF,
    config: Optional[ShadingConfig] = None,
    slotprice: Optional[np.ndarray] = None,
    exchange_ids: Optional[np.ndarray] = None,
    exchange_cdfs: Optional[Dict[str, MarketCDF]] = None,
) -> ShadingResult:
    """Apply bid shading strategy specified in config.

    Dispatches to the appropriate shading function based on config.strategy.

    Args:
        values: V(x) array.
        market_cdf: Overall market price CDF.
        config: Shading configuration.
        slotprice: Floor prices (required for dual_regime).
        exchange_ids: Exchange IDs (required if exchange_conditional=True).
        exchange_cdfs: Exchange CDFs (required if exchange_conditional=True).

    Returns:
        ShadingResult.
    """
    if config is None:
        config = ShadingConfig()

    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    if config.strategy == "linear":
        bids = linear_bid(values, config.linear_alpha, config.min_bid, config.max_bid)
        shading_factors = np.where(values > 0, bids / values, 0.0)
        win_probs = _interpolate_cdf(bids, market_cdf)
        surplus = (values - bids) * win_probs
        return ShadingResult(bids, shading_factors, surplus, win_probs, np.zeros(n, dtype=np.int32))

    elif config.strategy == "percentile":
        bids = percentile_bid(values, market_cdf, config.percentile_target, config.min_bid, config.max_bid)
        shading_factors = np.where(values > 0, bids / values, 0.0)
        win_probs = _interpolate_cdf(bids, market_cdf)
        surplus = (values - bids) * win_probs
        return ShadingResult(bids, shading_factors, surplus, win_probs, np.zeros(n, dtype=np.int32))

    elif config.strategy == "optimal":
        if config.exchange_conditional and exchange_ids is not None and exchange_cdfs is not None:
            return exchange_conditional_shading(values, exchange_ids, exchange_cdfs, market_cdf, config)
        bids, surplus = optimal_bid_vectorized(values, market_cdf, min_bid=config.min_bid, max_bid=config.max_bid)
        shading_factors = np.where(values > 0, bids / values, 0.0)
        win_probs = _interpolate_cdf(bids, market_cdf)
        expected_surplus = (values - bids) * win_probs
        return ShadingResult(bids, shading_factors, expected_surplus, win_probs, np.zeros(n, dtype=np.int32))

    elif config.strategy == "dual_regime":
        if slotprice is None:
            raise ValueError("slotprice required for dual_regime strategy")
        return dual_regime_shading(values, market_cdf, slotprice, config)

    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")
