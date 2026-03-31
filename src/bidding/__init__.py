"""Bidding strategy modules for RTB bid optimization (SP3).

Core formula: bid(x) = V(x) × shade(x) × pace(t)

Modules:
  - value: Impression value computation (V(x) = pCTR × CPC_target)
  - shading: First-price bid shading (distribution-based optimal, linear, dual-regime)
  - pacing: Budget pacing (PID controller, throttling)
  - simulator: Offline auction simulation engine
"""

from .value import ValueConfig, ValueResult, compute_impression_values
from .shading import (
    MarketCDF,
    ShadingConfig,
    ShadingResult,
    compute_shaded_bids,
    load_exchange_cdfs,
    load_market_cdf,
    optimal_bid_vectorized,
)
from .pacing import PacingConfig, PacingResult, PacingState, simulate_pacing
from .simulator import (
    AuctionResult,
    SimulationConfig,
    SimulationMetrics,
    build_standard_strategies,
    compare_strategies,
    run_auction_simulation,
)

__all__ = [
    # Value
    "ValueConfig",
    "ValueResult",
    "compute_impression_values",
    # Shading
    "MarketCDF",
    "ShadingConfig",
    "ShadingResult",
    "compute_shaded_bids",
    "load_exchange_cdfs",
    "load_market_cdf",
    "optimal_bid_vectorized",
    # Pacing
    "PacingConfig",
    "PacingResult",
    "PacingState",
    "simulate_pacing",
    # Simulator
    "AuctionResult",
    "SimulationConfig",
    "SimulationMetrics",
    "build_standard_strategies",
    "compare_strategies",
    "run_auction_simulation",
]
