"""
Offline Auction Simulation Engine for RTB Bid Optimization.

Evaluates bidding strategies using iPinYou test data.
- Primary: won-only simulation (payprice observed, ~4.23M impressions)
- Secondary: full simulation with KM CDF sampling (supplementary)

Auction types:
- first_price: winner pays own bid (current industry standard)
- second_price: winner pays market price (iPinYou original)

Survey ref: Ou et al. (2024) — iPinYou dataset as benchmark,
  used by [9, 65, 94, 95, 97] for bid optimization research.
"""

from typing import Callable, Dict, List, NamedTuple, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result Types
# ---------------------------------------------------------------------------

class AuctionResult(NamedTuple):
    """Per-auction simulation result."""
    bids: np.ndarray           # bid prices submitted
    wins: np.ndarray           # 1 if bid >= market_price
    payments: np.ndarray       # first_price: bid if win; second_price: market_price if win
    surplus: np.ndarray        # (V - payment) if win, else 0
    clicks: np.ndarray         # actual click labels


class SimulationMetrics(NamedTuple):
    """Aggregate simulation metrics for strategy comparison."""
    strategy_name: str
    n_bids: int
    n_wins: int
    win_rate: float
    total_clicks: int
    click_rate: float              # clicks / wins
    total_spend: float             # sum of payments (CPM)
    total_surplus: float           # sum of (V - payment) for wins
    avg_cpm: float                 # avg payment per win
    avg_cpc: float                 # spend / clicks
    avg_surplus_per_win: float
    overpayment_ratio: float       # mean (bid - market_price) / market_price for wins
    roi: float                     # total value of clicks / total spend


class SimulationConfig(NamedTuple):
    """Simulation configuration."""
    auction_type: str = "first_price"  # first_price, second_price
    budget_per_day: Optional[float] = None  # None = unlimited
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Core Auction Simulation
# ---------------------------------------------------------------------------

def run_auction_simulation(
    bids: np.ndarray,
    market_prices: np.ndarray,
    values: np.ndarray,
    clicks: np.ndarray,
    auction_type: str = "first_price",
) -> AuctionResult:
    """Run offline auction simulation.

    For iPinYou (second-price data), market_price = payprice for won impressions.
    In first-price auction: win if bid ≥ market_price, pay = bid.
    In second-price auction: win if bid ≥ market_price, pay = market_price.

    Args:
        bids: Bid prices per impression.
        market_prices: Observed market prices (payprice for won impressions).
        values: V(x) per impression.
        clicks: Actual click labels (0/1).
        auction_type: "first_price" or "second_price".

    Returns:
        AuctionResult with per-impression outcomes.
    """
    bids = np.asarray(bids, dtype=np.float64)
    market_prices = np.asarray(market_prices, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    clicks = np.asarray(clicks, dtype=np.int32)

    wins = (bids >= market_prices).astype(np.int32)

    if auction_type == "first_price":
        payments = bids * wins  # pay own bid
    elif auction_type == "second_price":
        payments = market_prices * wins  # pay market price
    else:
        raise ValueError(f"Unknown auction_type: {auction_type}")

    surplus = (values - payments) * wins

    return AuctionResult(
        bids=bids,
        wins=wins,
        payments=payments,
        surplus=surplus,
        clicks=clicks * wins,  # only count clicks on won impressions
    )


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_simulation_metrics(
    result: AuctionResult,
    values: np.ndarray,
    market_prices: np.ndarray,
    strategy_name: str,
    cpc_target: float = 200_000.0,
) -> SimulationMetrics:
    """Compute aggregate metrics from auction result.

    Args:
        result: AuctionResult from run_auction_simulation.
        values: V(x) array (for ROI computation).
        market_prices: Market prices (for overpayment ratio).
        strategy_name: Name for display.
        cpc_target: CPC target for ROI computation.

    Returns:
        SimulationMetrics.
    """
    n_bids = len(result.bids)
    n_wins = int(np.sum(result.wins))
    total_clicks = int(np.sum(result.clicks))
    total_spend = float(np.sum(result.payments))
    total_surplus = float(np.sum(result.surplus))

    win_rate = n_wins / n_bids if n_bids > 0 else 0.0
    click_rate = total_clicks / n_wins if n_wins > 0 else 0.0
    avg_cpm = total_spend / n_wins if n_wins > 0 else 0.0
    avg_cpc = total_spend / total_clicks if total_clicks > 0 else float("inf")
    avg_surplus = total_surplus / n_wins if n_wins > 0 else 0.0

    # Overpayment ratio: (bid - market_price) / market_price for wins
    win_mask = result.wins.astype(bool)
    if win_mask.any():
        mp_wins = market_prices[win_mask]
        bid_wins = result.bids[win_mask]
        valid = mp_wins > 0
        if valid.any():
            overpayment = float(np.mean((bid_wins[valid] - mp_wins[valid]) / mp_wins[valid]))
        else:
            overpayment = 0.0
    else:
        overpayment = 0.0

    # ROI: total click value / total spend
    click_value = total_clicks * cpc_target
    roi = click_value / total_spend if total_spend > 0 else float("inf")

    return SimulationMetrics(
        strategy_name=strategy_name,
        n_bids=n_bids,
        n_wins=n_wins,
        win_rate=win_rate,
        total_clicks=total_clicks,
        click_rate=click_rate,
        total_spend=total_spend,
        total_surplus=total_surplus,
        avg_cpm=avg_cpm,
        avg_cpc=avg_cpc,
        avg_surplus_per_win=avg_surplus,
        overpayment_ratio=overpayment,
        roi=roi,
    )


# ---------------------------------------------------------------------------
# Strategy Comparison
# ---------------------------------------------------------------------------

def compare_strategies(
    strategies: Dict[str, np.ndarray],
    market_prices: np.ndarray,
    values: np.ndarray,
    clicks: np.ndarray,
    auction_type: str = "first_price",
    cpc_target: float = 200_000.0,
) -> pd.DataFrame:
    """Run all strategies and produce comparison DataFrame.

    Args:
        strategies: {strategy_name: bids_array}.
        market_prices: Observed market prices.
        values: V(x) per impression.
        clicks: Actual click labels.
        auction_type: "first_price" or "second_price".
        cpc_target: CPC target for ROI computation.

    Returns:
        DataFrame with one row per strategy, all SimulationMetrics fields.
    """
    records = []
    for name, bids in strategies.items():
        result = run_auction_simulation(bids, market_prices, values, clicks, auction_type)
        metrics = compute_simulation_metrics(result, values, market_prices, name, cpc_target)
        records.append(metrics._asdict())

    return pd.DataFrame(records).set_index("strategy_name")


def compare_auction_types(
    bids: np.ndarray,
    market_prices: np.ndarray,
    values: np.ndarray,
    clicks: np.ndarray,
    strategy_name: str = "strategy",
    cpc_target: float = 200_000.0,
) -> pd.DataFrame:
    """Compare first-price vs second-price for the same bids.

    Demonstrates why bid shading matters for first-price auctions.

    Args:
        bids: Bid prices.
        market_prices: Observed market prices.
        values: V(x) per impression.
        clicks: Actual click labels.
        strategy_name: Base name for display.
        cpc_target: CPC target for ROI computation.

    Returns:
        DataFrame with 2 rows (first_price, second_price).
    """
    records = []
    for atype in ["first_price", "second_price"]:
        result = run_auction_simulation(bids, market_prices, values, clicks, atype)
        metrics = compute_simulation_metrics(
            result, values, market_prices,
            f"{strategy_name}_{atype}", cpc_target,
        )
        records.append(metrics._asdict())

    return pd.DataFrame(records).set_index("strategy_name")


# ---------------------------------------------------------------------------
# Prepare Test Data (Won-only Simulation)
# ---------------------------------------------------------------------------

def prepare_won_only_data(
    test_df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Prepare won-only simulation data.

    Filters to impressions where y_win=1 AND payprice > 0 (market price observed).
    This is the primary defensible simulation approach.

    Args:
        test_df: Test DataFrame with bidprice, payprice, slotprice, adexchange, hour, click.
        predictions: Dict with p_win, p_ctr, y_win, y_click arrays.

    Returns:
        Dict with filtered arrays: p_ctr, market_prices, clicks, slotprice, adexchange, hour, bidprice.
    """
    y_win = predictions["y_win"].astype(bool)
    # Convert pyarrow-backed columns to numpy for compatibility
    payprice = np.asarray(test_df["payprice"].values, dtype=np.float64) if "payprice" in test_df.columns else np.zeros(len(test_df))
    valid = y_win & (payprice > 0)

    def _col_np(col: str, dtype=np.float64) -> np.ndarray:
        """Extract column as numpy, handling pyarrow NA."""
        arr = test_df[col].values if col in test_df.columns else None
        if arr is None:
            return np.zeros(int(valid.sum()), dtype=dtype)
        arr = np.asarray(arr)
        # Replace NA/nan with 0
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.nan_to_num(arr, nan=0.0)
        return arr[valid].astype(dtype)

    n_valid = int(valid.sum())
    return {
        "p_ctr": predictions["p_ctr"][valid],
        "market_prices": payprice[valid],
        "clicks": predictions["y_click"][valid].astype(np.int32),
        "slotprice": _col_np("slotprice", np.float64),
        "adexchange": _col_np("adexchange", np.int64),
        "hour": _col_np("hour", np.int32),
        "bidprice": _col_np("bidprice", np.float64),
        "n_total": n_valid,
        "n_filtered": len(y_win) - n_valid,
    }


# ---------------------------------------------------------------------------
# Standard Strategy Definitions
# ---------------------------------------------------------------------------

def build_standard_strategies(
    values: np.ndarray,
    market_cdf_path: str,
    exchange_cdf_dir: str,
    bidprice: np.ndarray,
    slotprice: np.ndarray,
    exchange_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build all 8 standard bidding strategies for comparison.

    Args:
        values: V(x) array (won-only).
        market_cdf_path: Path to km_cdf_overall.npz.
        exchange_cdf_dir: Directory with exchange CDFs.
        bidprice: Original iPinYou flat bids.
        slotprice: Floor prices.
        exchange_ids: Exchange IDs.

    Returns:
        Dict of {strategy_name: bids_array}.
    """
    from .shading import (
        ShadingConfig,
        dual_regime_shading,
        exchange_conditional_shading,
        linear_bid,
        load_exchange_cdfs,
        load_market_cdf,
        optimal_bid_vectorized,
    )

    overall_cdf = load_market_cdf(market_cdf_path)
    exchange_cdfs = load_exchange_cdfs(exchange_cdf_dir)

    strategies = {}

    # 1. iPinYou flat-bid baseline (original)
    strategies["ipinyou_flat"] = bidprice.copy()

    # 2. Truthful: bid = V(x)
    strategies["truthful"] = np.clip(values, 1.0, 300.0)

    # 3. Linear α=0.8
    strategies["linear_08"] = linear_bid(values, alpha=0.8)

    # 4. Linear α=0.6
    strategies["linear_06"] = linear_bid(values, alpha=0.6)

    # 5. Optimal (overall KM CDF)
    opt_bids, _ = optimal_bid_vectorized(values, overall_cdf)
    strategies["optimal_km"] = opt_bids

    # 6. Optimal exchange-conditional
    config_ex = ShadingConfig(exchange_conditional=True)
    ex_result = exchange_conditional_shading(values, exchange_ids, exchange_cdfs, overall_cdf, config_ex)
    strategies["optimal_exchange"] = ex_result.bids

    # 7. Dual-regime (floor-aware)
    dr_result = dual_regime_shading(values, overall_cdf, slotprice)
    strategies["dual_regime"] = dr_result.bids

    # 8. Optimal + paced (placeholder — pacing applied in notebook)
    strategies["optimal_paced"] = opt_bids.copy()

    return strategies
