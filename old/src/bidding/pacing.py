"""
Budget Pacing for RTB Bid Optimization.

Implements PID controller and throttling approaches for budget management.
Functional style: immutable PacingState NamedTuple, pure update functions.

Survey ref: Ou et al. (2024) Sec 5.1.2 — PID controller widely used,
  control signal ϕ = λP×e + λI×Σe + λD×Δe/Δt (Eq.10).
EDA insight: Hourly WR U-shape (dawn 43% → afternoon 8.59%) →
  WR-weighted budget allocation improves efficiency.
"""

from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration & State Types
# ---------------------------------------------------------------------------

class PacingConfig(NamedTuple):
    """Budget pacing configuration."""
    pacing_type: str = "pid"          # pid, throttle, uniform, wr_weighted
    daily_budget: float = 10_000.0    # CPM units
    total_hours: int = 24
    # PID parameters (Ou et al. Eq.10)
    kp: float = 0.5
    ki: float = 0.1
    kd: float = 0.1
    multiplier_min: float = 0.3
    multiplier_max: float = 2.0
    # Hourly weights (None = uniform)
    hourly_weights: Optional[Dict[int, float]] = None


class PacingState(NamedTuple):
    """Immutable pacing state for functional updates."""
    spent: float = 0.0
    integral_error: float = 0.0
    prev_error: float = 0.0
    current_hour: int = 0


class PacingSnapshot(NamedTuple):
    """Single-step pacing record for diagnostics."""
    hour: int
    spent: float
    ideal_spent: float
    error: float
    multiplier: float
    hourly_spend: float


class PacingResult(NamedTuple):
    """Pacing simulation result."""
    multipliers: np.ndarray       # per-hour multipliers applied
    hourly_spend: np.ndarray      # actual spend per hour
    hourly_budget: np.ndarray     # ideal budget per hour
    budget_utilization: float     # total_spend / daily_budget
    total_spend: float
    snapshots: Tuple[PacingSnapshot, ...]


# ---------------------------------------------------------------------------
# PID Controller (Pure Functions)
# ---------------------------------------------------------------------------

def compute_pid_multiplier(
    state: PacingState,
    config: PacingConfig,
) -> Tuple[float, PacingState]:
    """Compute PID pacing multiplier (pure function).

    PID formula (Ou et al. Eq.10):
      ϕ = Kp × e(t) + Ki × Σe + Kd × Δe/Δt
      multiplier = clip(1 + ϕ / normalization, [min, max])

    Args:
        state: Current immutable pacing state.
        config: Pacing configuration.

    Returns:
        (multiplier, new_state) tuple.
    """
    weights = _get_hourly_weights(config)
    ideal_spent = _compute_ideal_spent(state.current_hour, config.daily_budget, weights)
    error = ideal_spent - state.spent  # positive = underspend, negative = overspend

    new_integral = state.integral_error + error
    derivative = error - state.prev_error

    pid_output = config.kp * error + config.ki * new_integral + config.kd * derivative
    normalization = max(config.daily_budget * 0.1, 1.0)
    multiplier = 1.0 + pid_output / normalization
    multiplier = max(config.multiplier_min, min(config.multiplier_max, multiplier))

    new_state = PacingState(
        spent=state.spent,
        integral_error=new_integral,
        prev_error=error,
        current_hour=state.current_hour,
    )
    return multiplier, new_state


def _get_hourly_weights(config: PacingConfig) -> Dict[int, float]:
    """Get hourly budget weights, defaulting to uniform."""
    if config.hourly_weights is not None:
        return config.hourly_weights
    return {h: 1.0 for h in range(config.total_hours)}


def _compute_ideal_spent(
    current_hour: int,
    daily_budget: float,
    weights: Dict[int, float],
) -> float:
    """Compute ideal cumulative spend up to current_hour."""
    total_weight = sum(weights.values())
    if total_weight == 0:
        return daily_budget * (current_hour + 1) / 24
    cum_weight = sum(weights.get(h, 0) for h in range(current_hour + 1))
    return daily_budget * cum_weight / total_weight


# ---------------------------------------------------------------------------
# Throttling
# ---------------------------------------------------------------------------

def throttle_decision(
    state: PacingState,
    config: PacingConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[bool, PacingState]:
    """Decide whether to participate in auction (throttling pacing).

    If overspending, probabilistically skip bids.
    Survey ref: Ou et al. Sec 5.1.2 — throttling modifies participation probability.

    Args:
        state: Current pacing state.
        config: Pacing configuration.
        rng: Random generator (for reproducibility).

    Returns:
        (should_bid: bool, state: PacingState).
    """
    if rng is None:
        rng = np.random.default_rng()

    remaining_hours = max(config.total_hours - state.current_hour, 1)
    remaining_budget = config.daily_budget - state.spent
    ideal_hourly = remaining_budget / remaining_hours

    # Estimate current spending rate
    current_rate = state.spent / max(state.current_hour, 1) if state.current_hour > 0 else ideal_hourly

    if current_rate > ideal_hourly * 1.2:
        # Overspending → throttle
        throttle_prob = 1 - (ideal_hourly / current_rate)
        should_bid = rng.random() > throttle_prob
    else:
        should_bid = True

    return should_bid, state


# ---------------------------------------------------------------------------
# Hourly Budget Weights (Data-Driven)
# ---------------------------------------------------------------------------

def compute_hourly_budget_weights(
    hourly_win_rates: Dict[int, float],
    hourly_volumes: Optional[Dict[int, float]] = None,
) -> Dict[int, float]:
    """Compute data-driven hourly budget weights.

    EDA U-shape insight: high WR = low competition = more efficient spend.
    Weight hours with higher win rates more heavily.

    Args:
        hourly_win_rates: {hour: win_rate}. From EDA: dawn ~43%, afternoon ~8.6%.
        hourly_volumes: {hour: bid_volume}. Optional volume weighting.

    Returns:
        Normalized hourly weights dict.
    """
    weights = {}
    for h in range(24):
        wr = hourly_win_rates.get(h, 0.2)
        vol = hourly_volumes.get(h, 1.0) if hourly_volumes else 1.0
        # Higher WR → more efficient → higher weight
        weights[h] = wr * vol

    total = sum(weights.values())
    if total > 0:
        weights = {h: w / total * 24 for h, w in weights.items()}  # normalize to mean=1
    return weights


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_pacing(
    bids: np.ndarray,
    payments: np.ndarray,
    hours: np.ndarray,
    config: PacingConfig,
) -> PacingResult:
    """Run full-day budget pacing simulation.

    Processes bids chronologically, applying PID/throttle multipliers per hour.

    Args:
        bids: Base bid prices (before pacing adjustment).
        payments: Actual payment if won (0 if lost). For budget tracking.
        hours: Hour of day for each bid (0-23).
        config: Pacing configuration.

    Returns:
        PacingResult with per-hour metrics and snapshots.
    """
    bids = np.asarray(bids, dtype=np.float64)
    payments = np.asarray(payments, dtype=np.float64)
    hours = np.asarray(hours, dtype=np.int32)

    multipliers = np.ones(config.total_hours, dtype=np.float64)
    hourly_spend = np.zeros(config.total_hours, dtype=np.float64)
    weights = _get_hourly_weights(config)
    total_weight = sum(weights.values())

    hourly_budget = np.array([
        config.daily_budget * weights.get(h, 1.0) / total_weight
        for h in range(config.total_hours)
    ])

    state = PacingState()
    snapshots = []

    for hour in range(config.total_hours):
        # Compute multiplier for this hour
        state = state._replace(current_hour=hour)
        multiplier, state = compute_pid_multiplier(state, config)
        multipliers[hour] = multiplier

        # Apply multiplier to bids in this hour
        hour_mask = hours == hour
        hour_payments = payments[hour_mask]

        # Scale payments by multiplier (simplified: multiplier affects bid→win→payment)
        scaled_spend = float(np.sum(hour_payments)) * multiplier
        hourly_spend[hour] = scaled_spend

        # Update spent
        state = state._replace(spent=state.spent + scaled_spend)

        ideal = _compute_ideal_spent(hour, config.daily_budget, weights)
        snapshots.append(PacingSnapshot(
            hour=hour,
            spent=state.spent,
            ideal_spent=ideal,
            error=ideal - state.spent,
            multiplier=multiplier,
            hourly_spend=scaled_spend,
        ))

    total_spend = float(np.sum(hourly_spend))
    utilization = total_spend / config.daily_budget if config.daily_budget > 0 else 0.0

    return PacingResult(
        multipliers=multipliers,
        hourly_spend=hourly_spend,
        hourly_budget=hourly_budget,
        budget_utilization=min(utilization, 1.0),
        total_spend=total_spend,
        snapshots=tuple(snapshots),
    )
