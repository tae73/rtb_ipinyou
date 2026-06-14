"""
SCM, DAG & Model-Based Counterfactual for RTB (SP4 Part B).

Provides:
  - RTB DAG specification with surplus/payment nodes (2-channel structure)
  - DoWhy backdoor identification & estimation
  - Refutation tests (random common cause, placebo, data subset)
  - Model-based counterfactual using existing auction simulator

DAG key insight:
  Bid affects Surplus through two channels:
    Volume: bid → win → surplus (positive)
    Cost: bid → payment → surplus (negative)
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from src.bidding.simulator import (
    SimulationMetrics,
    compute_simulation_metrics,
    run_auction_simulation,
)

try:
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RTB_DAG_DOT = """
digraph {
    /* Exogenous (unobserved) */
    U_market [label="U_market", observed="no"];
    U_user [label="U_user", observed="no"];

    /* Observed */
    context [label="Context"];
    user [label="User"];
    campaign [label="Campaign"];
    floor [label="Floor Price"];
    market_price [label="Market Price"];
    bid [label="Bid Price"];
    win [label="Win"];
    payment [label="Payment"];
    click [label="Click"];
    value [label="V(x)"];
    surplus [label="Surplus"];

    /* Exogenous -> Observed */
    U_market -> market_price;
    U_user -> user;

    /* Context -> Market */
    context -> market_price;

    /* Bidding decision */
    campaign -> bid;
    context -> bid;
    user -> bid;

    /* Auction outcome */
    bid -> win;
    market_price -> win;
    floor -> win;
    context -> floor;

    /* Two channels to surplus */
    bid -> payment;
    win -> payment;

    /* Click & Value */
    win -> click;
    user -> click;
    context -> click;

    click -> value;
    user -> value;
    context -> value;
    campaign -> value;

    /* Surplus composition */
    value -> surplus;
    payment -> surplus;
    win -> surplus;
}
"""

_DEFAULT_METHODS: List[str] = [
    "backdoor.linear_regression",
    "backdoor.propensity_score_weighting",
]


# ---------------------------------------------------------------------------
# NamedTuples
# ---------------------------------------------------------------------------

class DAGSpec(NamedTuple):
    """DAG specification for DoWhy."""
    graph_dot: str
    treatment: str
    outcome: str


class SCMResult(NamedTuple):
    """Result of SCM identification and estimation."""
    estimates: Dict[str, float]
    estimate_cis: Dict[str, Optional[Tuple[float, float]]]
    identified_estimand: Optional[Any] = None


class RefutationResult(NamedTuple):
    """Result of refutation tests."""
    tests: Dict[str, Dict[str, Any]]   # test_name -> {estimate, p_value, ...}
    is_robust: bool


class CounterfactualResult(NamedTuple):
    """Result of model-based counterfactual simulation."""
    scenario_name: str
    original: SimulationMetrics
    counterfactual: SimulationMetrics
    delta_surplus: float
    delta_win_rate: float


# ---------------------------------------------------------------------------
# DAG Construction
# ---------------------------------------------------------------------------

def build_rtb_dag(
    treatment: str = "bid",
    outcome: str = "surplus",
) -> DAGSpec:
    """Build RTB DAG with surplus/payment nodes.

    The DAG encodes two channels from bid to surplus:
      Volume: bid -> win -> surplus (winning more auctions)
      Cost: bid -> payment -> surplus (paying more per win)

    Args:
        treatment: Treatment variable name in the DAG.
        outcome: Outcome variable name in the DAG.

    Returns:
        DAGSpec with DOT graph string.
    """
    return DAGSpec(
        graph_dot=_RTB_DAG_DOT,
        treatment=treatment,
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# SCM Estimation
# ---------------------------------------------------------------------------

def _prepare_scm_data(
    df: pd.DataFrame,
    subsample_n: int,
    preds: Optional[Dict[str, np.ndarray]] = None,
    value_config: Optional[Any] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Prepare DataFrame for DoWhy (subsample + rename).

    Args:
        df: Feature DataFrame with bidprice, payprice, win, etc.
        subsample_n: Max rows to keep.
        preds: Model predictions dict (must contain 'p_ctr' for surplus).
        value_config: ValueConfig for V(x) computation. Defaults to CPC 200K.
        seed: Random seed for subsampling.
    """
    idx = None
    if len(df) > subsample_n:
        idx = df.sample(n=subsample_n, random_state=seed).index
        df = df.loc[idx]

    scm_df = pd.DataFrame({
        "bid": df["bidprice"].values.astype(float),
        "win": df["win"].values.astype(float),
        "click": df["click"].values.astype(float),
        "floor": df["slotprice"].values.astype(float),
        "context": df["adexchange"].values.astype(float),
        "user": df["region"].values.astype(float),
        "campaign": df["advertiser"].values.astype(float),
    })

    # Payment = payprice × win (0 for losers)
    payprice = df["payprice"].values.astype(float)
    scm_df["payment"] = payprice * scm_df["win"].values
    scm_df["market_price"] = payprice  # observed only for winners

    # Value & Surplus (requires preds with p_ctr)
    if preds is not None and "p_ctr" in preds:
        from src.bidding.value import ValueConfig as VC, compute_impression_values

        p_ctr = np.asarray(preds["p_ctr"])
        if idx is not None:
            p_ctr = p_ctr[idx.values]
        cfg = value_config if value_config is not None else VC()
        scm_df["value"] = compute_impression_values(p_ctr, cfg).values
        scm_df["surplus"] = (
            (scm_df["value"].values - payprice) * scm_df["win"].values
        )

    return scm_df


def estimate_causal_effect(
    df: pd.DataFrame,
    dag: Optional[DAGSpec] = None,
    methods: Optional[List[str]] = None,
    subsample_n: int = 500_000,
    preds: Optional[Dict[str, np.ndarray]] = None,
    value_config: Optional[Any] = None,
) -> SCMResult:
    """Estimate causal effect of bid on outcome using DoWhy.

    Args:
        df: Feature DataFrame with bidprice, win, click, payprice, etc.
        dag: DAG specification. Defaults to build_rtb_dag().
        methods: Estimation methods. Defaults to linear_regression + PS weighting.
        subsample_n: Subsample for computational feasibility.
        preds: Model predictions dict (must contain 'p_ctr' when outcome is surplus).
        value_config: ValueConfig for V(x) computation.

    Returns:
        SCMResult with estimates from each method.
    """
    if not DOWHY_AVAILABLE:
        raise ImportError("dowhy>=0.11.0 required. Install: pip install dowhy")

    if dag is None:
        dag = build_rtb_dag()
    if methods is None:
        methods = list(_DEFAULT_METHODS)

    scm_df = _prepare_scm_data(df, subsample_n, preds=preds, value_config=value_config)

    model = CausalModel(
        data=scm_df,
        treatment=dag.treatment,
        outcome=dag.outcome,
        graph=dag.graph_dot,
    )

    identified = model.identify_effect(proceed_when_unidentifiable=True)

    estimates: Dict[str, float] = {}
    estimate_cis: Dict[str, Optional[Tuple[float, float]]] = {}

    for method in methods:
        try:
            est = model.estimate_effect(identified, method_name=method)
            estimates[method] = float(est.value)
            try:
                ci = est.get_confidence_intervals()
                estimate_cis[method] = (float(ci[0]), float(ci[1]))
            except Exception:
                estimate_cis[method] = None
        except Exception as e:
            estimates[method] = float("nan")
            estimate_cis[method] = None

    return SCMResult(
        estimates=estimates,
        estimate_cis=estimate_cis,
        identified_estimand=identified,
    )


# ---------------------------------------------------------------------------
# Refutation Tests
# ---------------------------------------------------------------------------

def run_refutation_tests(
    df: pd.DataFrame,
    dag: Optional[DAGSpec] = None,
    n_simulations: int = 50,
    subsample_n: int = 500_000,
    preds: Optional[Dict[str, np.ndarray]] = None,
    value_config: Optional[Any] = None,
) -> RefutationResult:
    """Run 3 refutation tests for robustness validation.

    Tests:
      1. random_common_cause: Adding random confounder shouldn't change estimate.
      2. placebo_treatment: Permuting treatment should give ~0 effect.
      3. data_subset: Estimate should be stable across 80% subsets.

    Args:
        df: Feature DataFrame.
        dag: DAG specification. Defaults to build_rtb_dag().
        n_simulations: Number of simulations per refutation.
        subsample_n: Subsample size.
        preds: Model predictions dict (must contain 'p_ctr' when outcome is surplus).
        value_config: ValueConfig for V(x) computation.

    Returns:
        RefutationResult with test details and overall robustness flag.
    """
    if not DOWHY_AVAILABLE:
        raise ImportError("dowhy>=0.11.0 required. Install: pip install dowhy")

    if dag is None:
        dag = build_rtb_dag()

    scm_df = _prepare_scm_data(df, subsample_n, preds=preds, value_config=value_config)

    model = CausalModel(
        data=scm_df,
        treatment=dag.treatment,
        outcome=dag.outcome,
        graph=dag.graph_dot,
    )

    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified, method_name="backdoor.linear_regression",
    )

    tests: Dict[str, Dict[str, Any]] = {}
    is_robust = True

    # 1. Random common cause
    try:
        refute = model.refute_estimate(
            identified, estimate,
            method_name="random_common_cause",
            num_simulations=n_simulations,
        )
        tests["random_common_cause"] = {
            "new_effect": float(refute.new_effect),
            "original_effect": float(estimate.value),
            "change_pct": abs(float(refute.new_effect) - float(estimate.value))
                          / max(abs(float(estimate.value)), 1e-10) * 100,
        }
        if tests["random_common_cause"]["change_pct"] > 10:
            is_robust = False
    except Exception as e:
        tests["random_common_cause"] = {"error": str(e)}
        is_robust = False

    # 2. Placebo treatment
    try:
        refute = model.refute_estimate(
            identified, estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
        )
        tests["placebo_treatment"] = {
            "new_effect": float(refute.new_effect),
        }
        if abs(float(refute.new_effect)) > abs(float(estimate.value)) * 0.1:
            is_robust = False
    except Exception as e:
        tests["placebo_treatment"] = {"error": str(e)}
        is_robust = False

    # 3. Data subset
    try:
        refute = model.refute_estimate(
            identified, estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8,
            num_simulations=min(n_simulations, 10),
        )
        tests["data_subset"] = {
            "new_effect": float(refute.new_effect),
        }
    except Exception as e:
        tests["data_subset"] = {"error": str(e)}

    return RefutationResult(tests=tests, is_robust=is_robust)


# ---------------------------------------------------------------------------
# Model-Based Counterfactual
# ---------------------------------------------------------------------------

def simulate_counterfactual(
    market_prices: np.ndarray,
    original_bids: np.ndarray,
    values: np.ndarray,
    clicks: np.ndarray,
    bid_multiplier: float = 1.1,
    auction_type: str = "first_price",
) -> CounterfactualResult:
    """Run model-based counterfactual using auction simulator.

    "What if all bids were multiplied by bid_multiplier?"

    Uses existing run_auction_simulation() from src.bidding.simulator,
    avoiding the need for structural equations.

    Args:
        market_prices: Observed market prices (payprice for won impressions).
        original_bids: Original bid prices.
        values: V(x) per impression.
        clicks: Actual click labels.
        bid_multiplier: Factor to multiply all bids by.
        auction_type: "first_price" or "second_price".

    Returns:
        CounterfactualResult comparing original vs counterfactual outcomes.
    """
    market_prices = np.asarray(market_prices, dtype=np.float64)
    original_bids = np.asarray(original_bids, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    clicks = np.asarray(clicks, dtype=np.int32)

    # Original
    orig_result = run_auction_simulation(
        original_bids, market_prices, values, clicks, auction_type,
    )
    orig_metrics = compute_simulation_metrics(
        orig_result, values, market_prices, "original",
    )

    # Counterfactual
    cf_bids = original_bids * bid_multiplier
    cf_result = run_auction_simulation(
        cf_bids, market_prices, values, clicks, auction_type,
    )
    scenario_name = f"bid×{bid_multiplier:.2f}"
    cf_metrics = compute_simulation_metrics(
        cf_result, values, market_prices, scenario_name,
    )

    return CounterfactualResult(
        scenario_name=scenario_name,
        original=orig_metrics,
        counterfactual=cf_metrics,
        delta_surplus=cf_metrics.total_surplus - orig_metrics.total_surplus,
        delta_win_rate=cf_metrics.win_rate - orig_metrics.win_rate,
    )


def simulate_counterfactual_scenarios(
    market_prices: np.ndarray,
    original_bids: np.ndarray,
    values: np.ndarray,
    clicks: np.ndarray,
    multipliers: Optional[List[float]] = None,
    auction_type: str = "first_price",
) -> pd.DataFrame:
    """Run multiple counterfactual scenarios and return comparison table.

    Args:
        market_prices: Observed market prices.
        original_bids: Original bid prices.
        values: V(x) per impression.
        clicks: Actual click labels.
        multipliers: Bid multipliers. Defaults to [0.8, 0.9, 1.0, 1.1, 1.2].
        auction_type: Auction type.

    Returns:
        DataFrame with one row per scenario: multiplier, win_rate, surplus,
        overpayment_ratio, delta_surplus, delta_win_rate.
    """
    if multipliers is None:
        multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]

    records = []
    for mult in multipliers:
        cf = simulate_counterfactual(
            market_prices, original_bids, values, clicks, mult, auction_type,
        )
        records.append({
            "multiplier": mult,
            "scenario": cf.scenario_name,
            "win_rate": cf.counterfactual.win_rate,
            "total_surplus": cf.counterfactual.total_surplus,
            "avg_surplus_per_win": cf.counterfactual.avg_surplus_per_win,
            "overpayment_ratio": cf.counterfactual.overpayment_ratio,
            "delta_surplus": cf.delta_surplus,
            "delta_win_rate": cf.delta_win_rate,
        })

    return pd.DataFrame(records)
