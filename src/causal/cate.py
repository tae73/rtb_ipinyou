"""
Multi-Outcome CATE Estimation for RTB Bid Optimization (SP4 Part A).

Estimates Conditional Average Treatment Effects across 4 outcomes:
  - Win: bid elasticity of winning (τ_win)
  - Payment: cost elasticity (τ_pay, winners-only)
  - Click: value elasticity (τ_click)
  - Surplus: net economic effect (τ_surplus, validation)

Surplus decomposition: τ_surplus ≈ V(x)·τ_win - τ_pay
Mediation: NIE (volume channel) vs NDE (cost channel)

Treatment: T = log(bid_price + 1)
V(x) is NOT a CATE outcome (bid-invariant model prediction).
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from src.bidding.value import ValueConfig, compute_impression_values

try:
    from econml.dml import CausalForestDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADVERTISER_TAXONOMY: Dict[str, List[int]] = {
    "retargeting": [2821, 3358, 2259],
    "branding": [1458, 3386, 3427, 2261, 2997],
    "mixed": [3476],
}

_ADV_TO_TAXONOMY: Dict[int, str] = {
    adv: tax
    for tax, advs in ADVERTISER_TAXONOMY.items()
    for adv in advs
}

_TAXONOMY_ENCODE: Dict[str, int] = {
    "branding": 0,
    "retargeting": 1,
    "mixed": 2,
}

HETEROGENEITY_FEATURES: List[str] = [
    "advertiser_taxonomy", "hour", "adexchange", "region",
]

CONFOUNDER_FEATURES: List[str] = [
    "slotprice", "slot_area", "domain_freq",
]

_DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": -1,
}


# ---------------------------------------------------------------------------
# NamedTuples
# ---------------------------------------------------------------------------

class CATEConfig(NamedTuple):
    """CausalForestDML configuration."""
    n_estimators: int = 1000
    honest: bool = True
    inference: bool = True
    random_state: int = 42
    max_samples: float = 0.5
    min_samples_leaf: int = 50
    alpha: float = 0.05
    subsample_n: Optional[int] = 2_000_000
    discrete_treatment: bool = True         # iPinYou: 2-6 discrete bid levels
    model_y_params: Optional[Dict[str, Any]] = None
    model_t_params: Optional[Dict[str, Any]] = None


class OutcomeSpec(NamedTuple):
    """Specification for a single CATE outcome."""
    name: str                       # "win", "payment", "click", "surplus"
    is_binary: bool                 # True for win, click
    winners_only: bool = False      # True for payment


DEFAULT_OUTCOMES: List[OutcomeSpec] = [
    OutcomeSpec("win", is_binary=True),
    OutcomeSpec("payment", is_binary=False, winners_only=True),
    OutcomeSpec("click", is_binary=True),
    OutcomeSpec("surplus", is_binary=False),
]


class CATEResult(NamedTuple):
    """Result from a single CausalForestDML fit."""
    outcome_name: str
    cate: np.ndarray                # (n_samples,)
    cate_ci_lower: np.ndarray       # (n_samples,)
    cate_ci_upper: np.ndarray       # (n_samples,)
    ate: float
    ate_ci: Tuple[float, float]
    feature_importances: np.ndarray
    feature_names: List[str]
    n_samples: int


class MultiOutcomeCATEResult(NamedTuple):
    """Aggregated multi-outcome CATE results."""
    results: Dict[str, CATEResult]
    decomposition: Optional[Dict[str, float]]
    segment_summary: Optional[pd.DataFrame]


class MediationResult(NamedTuple):
    """Mediation decomposition: NIE (volume) vs NDE (cost)."""
    total_effect: float
    nie: float                      # Volume channel: E[V(x) × τ_win(x)]
    nde: float                      # Cost channel: Total - NIE
    mediation_proportion: float     # NIE / Total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assign_advertiser_taxonomy(advertiser: np.ndarray) -> np.ndarray:
    """Map advertiser IDs to taxonomy labels.

    Args:
        advertiser: Integer array of advertiser IDs.

    Returns:
        String array of taxonomy labels ("branding", "retargeting", "mixed").
    """
    return np.array(
        [_ADV_TO_TAXONOMY.get(int(a), "unknown") for a in advertiser],
        dtype=object,
    )


def _encode_taxonomy(taxonomy: np.ndarray) -> np.ndarray:
    """Encode taxonomy labels to integers for CausalForestDML."""
    return np.array(
        [_TAXONOMY_ENCODE.get(str(t), -1) for t in taxonomy],
        dtype=np.float64,
    )


def _safe_col(df: pd.DataFrame, col: str, dtype: type = np.float64) -> np.ndarray:
    """Extract column as numpy, handling pyarrow NA."""
    arr = np.asarray(df[col].values)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0)
    return arr.astype(dtype)


def _subsample_stratified(
    n_total: int,
    win: np.ndarray,
    n_target: int,
    seed: int = 42,
) -> np.ndarray:
    """Stratified subsample indices preserving win rate.

    Args:
        n_total: Total sample size.
        win: Binary win indicator.
        n_target: Target subsample size.
        seed: Random seed.

    Returns:
        Sorted integer index array of length <= n_target.
    """
    if n_total <= n_target:
        return np.arange(n_total)

    rng = np.random.RandomState(seed)
    ratio = n_target / n_total

    won_idx = np.where(win == 1)[0]
    lost_idx = np.where(win == 0)[0]

    n_won = int(len(won_idx) * ratio)
    n_lost = n_target - n_won

    sampled_won = rng.choice(won_idx, size=min(n_won, len(won_idx)), replace=False)
    sampled_lost = rng.choice(lost_idx, size=min(n_lost, len(lost_idx)), replace=False)

    return np.sort(np.concatenate([sampled_won, sampled_lost]))


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def _filter_bid_variation(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    min_bid_levels: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Filter to advertisers with bid variation (exclude flat-bid advertisers).

    iPinYou flat-bid structure: some advertisers always bid the same price.
    For CATE estimation, only advertisers with >= min_bid_levels are useful.

    Args:
        df: Feature DataFrame.
        predictions: Dict with 'p_ctr' array (aligned to df rows).
        min_bid_levels: Minimum unique bid prices per advertiser.

    Returns:
        Filtered (df, predictions) tuple.
    """
    adv_col = _safe_col(df, "advertiser", np.int64)
    bid_col = _safe_col(df, "bidprice", np.float64)

    # Find advertisers with sufficient bid variation
    adv_bid_df = pd.DataFrame({"advertiser": adv_col, "bidprice": bid_col})
    levels = adv_bid_df.groupby("advertiser")["bidprice"].nunique()
    valid_advs = set(levels[levels >= min_bid_levels].index)

    mask = np.isin(adv_col, list(valid_advs))
    idx = np.where(mask)[0]

    filtered_preds = {k: np.asarray(v)[idx] for k, v in predictions.items()}
    return df.iloc[idx].reset_index(drop=True), filtered_preds


def prepare_cate_data(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    value_config: Optional[ValueConfig] = None,
    subsample_n: Optional[int] = None,
    discrete_treatment: bool = True,
    filter_bid_variation: bool = True,
) -> Dict[str, np.ndarray]:
    """Prepare T, X, W, Y arrays for multi-outcome CATE estimation.

    Args:
        df: Feature DataFrame (test or train split, or merged).
        predictions: Dict with 'p_ctr' key (debiased pCTR array).
        value_config: V(x) computation config. Defaults to CPC 200K.
        subsample_n: If set, stratified subsample to this size.
        discrete_treatment: If True, T = bidprice (categorical).
            If False, T = log(bidprice + 1) (continuous).
        filter_bid_variation: If True, exclude advertisers with only 1 bid level.

    Returns:
        Dict with keys: T, X, W, feature_names,
            Y_win, Y_payment, Y_click, Y_surplus,
            values, win_mask,
            T_won, X_won, W_won, idx (subsample indices).
    """
    # Filter to advertisers with bid variation
    if filter_bid_variation:
        df, predictions = _filter_bid_variation(df, predictions)

    n = len(df)
    win = _safe_col(df, "win", np.int32)

    # Subsample
    if subsample_n is not None and subsample_n < n:
        idx = _subsample_stratified(n, win, subsample_n)
        df = df.iloc[idx].reset_index(drop=True)
        p_ctr = np.asarray(predictions["p_ctr"])[idx]
        win = win[idx]
    else:
        idx = np.arange(n)
        p_ctr = np.asarray(predictions["p_ctr"])

    n_sub = len(df)

    # Treatment
    if discrete_treatment:
        T = _safe_col(df, "bidprice")  # categorical (e.g., 227, 238, 241)
    else:
        T = np.log(_safe_col(df, "bidprice") + 1.0)

    # Heterogeneity features X
    taxonomy = assign_advertiser_taxonomy(_safe_col(df, "advertiser", np.int64))
    taxonomy_enc = _encode_taxonomy(taxonomy)
    hour = _safe_col(df, "hour")
    adexchange = _safe_col(df, "adexchange")
    region = _safe_col(df, "region")
    X = np.column_stack([taxonomy_enc, hour, adexchange, region])
    feature_names = list(HETEROGENEITY_FEATURES)

    # Confounders W
    slotprice = _safe_col(df, "slotprice")
    slot_area = _safe_col(df, "slot_area")
    domain_freq = _safe_col(df, "domain_freq")
    W = np.column_stack([slotprice, slot_area, domain_freq])

    # V(x)
    vr = compute_impression_values(p_ctr, value_config)
    values = vr.values

    # Outcomes
    payprice = _safe_col(df, "payprice")
    click = _safe_col(df, "click", np.int32)

    Y_win = win.astype(np.float64)
    Y_click = click.astype(np.float64)
    Y_surplus = (values - payprice) * win.astype(np.float64)

    # Winners-only for payment
    win_mask = win == 1
    Y_payment = payprice[win_mask]
    T_won = T[win_mask]
    X_won = X[win_mask]
    W_won = W[win_mask]

    return {
        "T": T, "X": X, "W": W,
        "feature_names": feature_names,
        "Y_win": Y_win,
        "Y_payment": Y_payment,
        "Y_click": Y_click,
        "Y_surplus": Y_surplus,
        "values": values,
        "win_mask": win_mask,
        "T_won": T_won, "X_won": X_won, "W_won": W_won,
        "taxonomy": taxonomy,
        "idx": idx,
        "n_samples": n_sub,
    }


# ---------------------------------------------------------------------------
# Core CATE Estimation
# ---------------------------------------------------------------------------

def _make_forest(config: CATEConfig) -> "CausalForestDML":
    """Build CausalForestDML with LGBMRegressor nuisance models."""
    if not ECONML_AVAILABLE:
        raise ImportError("econml>=0.15.0 required. Install: pip install econml")
    if not LGBM_AVAILABLE:
        raise ImportError("lightgbm required. Install: pip install lightgbm")

    y_params = config.model_y_params or dict(_DEFAULT_MODEL_PARAMS)
    t_params = config.model_t_params or dict(_DEFAULT_MODEL_PARAMS)

    # discrete_treatment requires a classifier for model_t
    model_t = LGBMClassifier(**t_params) if config.discrete_treatment else LGBMRegressor(**t_params)

    return CausalForestDML(
        model_y=LGBMRegressor(**y_params),
        model_t=model_t,
        discrete_treatment=config.discrete_treatment,
        n_estimators=config.n_estimators,
        honest=config.honest,
        inference=config.inference,
        max_samples=config.max_samples,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
    )


def estimate_cate(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    W: np.ndarray,
    outcome_name: str,
    config: Optional[CATEConfig] = None,
    feature_names: Optional[List[str]] = None,
) -> CATEResult:
    """Estimate CATE for a single outcome using CausalForestDML.

    Args:
        Y: Outcome array (n,).
        T: Treatment array (n,) — log(bid + 1).
        X: Heterogeneity features (n, p).
        W: Confounders (n, q).
        outcome_name: Label for this outcome.
        config: Forest configuration.
        feature_names: Names for X columns.

    Returns:
        CATEResult with per-sample CATE, CI, ATE, and feature importances.
    """
    if config is None:
        config = CATEConfig()
    if feature_names is None:
        feature_names = [f"X{i}" for i in range(X.shape[1])]

    forest = _make_forest(config)
    forest.fit(Y=Y, T=T, X=X, W=W)

    # For discrete treatment, must specify T0/T1 as actual treatment values
    effect_kwargs: Dict[str, Any] = {}
    if config.discrete_treatment:
        t_vals = np.unique(T)
        effect_kwargs = {"T0": t_vals.min(), "T1": t_vals.max()}

    cate = forest.effect(X, **effect_kwargs).flatten()
    ci_lower, ci_upper = forest.effect_interval(X, alpha=config.alpha, **effect_kwargs)
    ci_lower = ci_lower.flatten()
    ci_upper = ci_upper.flatten()

    ate = float(forest.ate(X, **effect_kwargs))
    ate_ci_raw = forest.ate_interval(X, alpha=config.alpha, **effect_kwargs)
    ate_ci = (float(ate_ci_raw[0]), float(ate_ci_raw[1]))

    importances = forest.feature_importances_

    return CATEResult(
        outcome_name=outcome_name,
        cate=cate,
        cate_ci_lower=ci_lower,
        cate_ci_upper=ci_upper,
        ate=ate,
        ate_ci=ate_ci,
        feature_importances=importances,
        feature_names=list(feature_names),
        n_samples=len(Y),
    )


def estimate_multi_outcome_cate(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    config: Optional[CATEConfig] = None,
    value_config: Optional[ValueConfig] = None,
    outcomes: Optional[List[OutcomeSpec]] = None,
) -> MultiOutcomeCATEResult:
    """Estimate CATE for multiple outcomes with decomposition validation.

    Fits CausalForestDML for each outcome:
      - win, click, surplus: full sample
      - payment: winners-only (avoids point mass at 0)

    Validates surplus decomposition: τ_surplus ≈ V(x)·τ_win - τ_pay.

    Args:
        df: Feature DataFrame.
        predictions: Dict with 'p_ctr' (debiased pCTR).
        config: CausalForestDML configuration.
        value_config: V(x) computation config.
        outcomes: Outcome specifications. Defaults to DEFAULT_OUTCOMES.

    Returns:
        MultiOutcomeCATEResult with per-outcome results, decomposition, summary.
    """
    if config is None:
        config = CATEConfig()
    if outcomes is None:
        outcomes = DEFAULT_OUTCOMES

    data = prepare_cate_data(df, predictions, value_config, config.subsample_n)

    outcome_arrays = {
        "win": data["Y_win"],
        "payment": data["Y_payment"],
        "click": data["Y_click"],
        "surplus": data["Y_surplus"],
    }

    results: Dict[str, CATEResult] = {}
    for spec in outcomes:
        Y = outcome_arrays[spec.name]
        if spec.winners_only:
            T, X, W = data["T_won"], data["X_won"], data["W_won"]
        else:
            T, X, W = data["T"], data["X"], data["W"]

        results[spec.name] = estimate_cate(
            Y=Y, T=T, X=X, W=W,
            outcome_name=spec.name,
            config=config,
            feature_names=data["feature_names"],
        )

    # Decomposition validation
    decomp = None
    if "win" in results and "surplus" in results:
        decomp = validate_decomposition(
            results, data["values"], data["win_mask"],
        )

    # Segment summary
    summary = compute_segment_cate_summary(
        results, data["taxonomy"], segment_name="advertiser_taxonomy",
    )

    return MultiOutcomeCATEResult(
        results=results,
        decomposition=decomp,
        segment_summary=summary,
    )


# ---------------------------------------------------------------------------
# Decomposition & Mediation
# ---------------------------------------------------------------------------

def validate_decomposition(
    results: Dict[str, CATEResult],
    values: np.ndarray,
    win_mask: np.ndarray,
) -> Dict[str, float]:
    """Compare direct surplus CATE vs decomposed V(x)·τ_win - τ_pay.

    Comparison on winners-only (non-winners are trivially ~0 for both).

    Args:
        results: Must contain 'win', 'surplus', and optionally 'payment'.
        values: V(x) array (full sample).
        win_mask: Boolean mask for winners.

    Returns:
        Dict with 'corr', 'mae', 'bias' comparing direct vs decomposed.
    """
    tau_win = results["win"].cate
    tau_surplus_direct = results["surplus"].cate

    # Build decomposed surplus CATE
    tau_pay_full = np.zeros_like(tau_win)
    if "payment" in results:
        tau_pay_full[win_mask] = results["payment"].cate

    tau_surplus_decomposed = values * tau_win - tau_pay_full

    # Compare on winners only (avoid trivial zero correlation)
    direct_won = tau_surplus_direct[win_mask]
    decomposed_won = tau_surplus_decomposed[win_mask]

    if len(direct_won) < 2:
        return {"corr": float("nan"), "mae": float("nan"), "bias": float("nan")}

    corr = float(np.corrcoef(direct_won, decomposed_won)[0, 1])
    mae = float(np.mean(np.abs(direct_won - decomposed_won)))
    bias = float(np.mean(decomposed_won - direct_won))

    return {"corr": corr, "mae": mae, "bias": bias}


def estimate_mediation(
    results: Dict[str, CATEResult],
    values: np.ndarray,
) -> MediationResult:
    """Decompose surplus CATE into volume (NIE) and cost (NDE) channels.

    Algebraic decomposition (not Pearl's formal NDE/NIE):
      NIE = E[V(x) × τ_win(x)]   — winning more auctions
      NDE = Total - NIE            — paying more per auction
      Total = E[τ_surplus(x)]      — from direct estimation

    Args:
        results: Must contain 'win' and 'surplus' CATEResults.
        values: V(x) array (same length as win CATE).

    Returns:
        MediationResult with total, NIE, NDE, and mediation proportion.
    """
    tau_win = results["win"].cate
    tau_surplus = results["surplus"].cate

    nie = float(np.mean(values * tau_win))
    total = float(np.mean(tau_surplus))
    nde = total - nie

    proportion = nie / total if abs(total) > 1e-10 else float("nan")

    return MediationResult(
        total_effect=total,
        nie=nie,
        nde=nde,
        mediation_proportion=proportion,
    )


# ---------------------------------------------------------------------------
# Segment Analysis
# ---------------------------------------------------------------------------

def compute_segment_cate_summary(
    results: Dict[str, CATEResult],
    segment_labels: np.ndarray,
    segment_name: str = "segment",
) -> pd.DataFrame:
    """Compute per-segment CATE summary across all outcomes.

    Args:
        results: Outcome name → CATEResult.
        segment_labels: Array of segment labels (same length as full-sample CATE).
        segment_name: Name for the segment column.

    Returns:
        DataFrame with columns: segment, outcome, mean_cate, std_cate, count.
    """
    records = []
    unique_segments = np.unique(segment_labels)

    for outcome_name, res in results.items():
        cate = res.cate
        # Payment is winners-only (shorter), skip segment analysis
        if len(cate) != len(segment_labels):
            continue

        for seg in unique_segments:
            mask = segment_labels == seg
            seg_cate = cate[mask]
            records.append({
                segment_name: str(seg),
                "outcome": outcome_name,
                "mean_cate": float(np.mean(seg_cate)),
                "std_cate": float(np.std(seg_cate)),
                "count": int(mask.sum()),
            })

    return pd.DataFrame(records) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Alternative CATE Methods (for limited treatment variation)
# ---------------------------------------------------------------------------

class StratifiedCATEResult(NamedTuple):
    """Result from stratified mean difference CATE."""
    outcome_name: str
    ate: float
    ate_se: float
    strata_df: pd.DataFrame         # per-stratum CATE
    cate_by_sample: np.ndarray      # per-sample CATE (mapped from strata)
    n_strata: int


class TLearnerCATEResult(NamedTuple):
    """Result from T-Learner CATE."""
    outcome_name: str
    ate: float
    cate: np.ndarray                # per-sample CATE
    mu_1: np.ndarray                # E[Y|T=high, X]
    mu_0: np.ndarray                # E[Y|T=low, X]
    feature_importances_1: np.ndarray
    feature_importances_0: np.ndarray
    feature_names: List[str]


def estimate_stratified_cate(
    Y: np.ndarray,
    T: np.ndarray,
    strata: np.ndarray,
    outcome_name: str,
) -> StratifiedCATEResult:
    """Estimate CATE via stratified mean differences.

    Within each stratum, computes E[Y|T=T_high] - E[Y|T=T_low].
    No residualization — works even when DML fails due to limited
    treatment variation (iPinYou flat-bid).

    Args:
        Y: Outcome array (n,).
        T: Treatment array (n,) — discrete values.
        strata: Stratum labels (n,) — e.g., (exchange, hour_bin) encoded.
        outcome_name: Label for this outcome.

    Returns:
        StratifiedCATEResult with per-stratum and per-sample CATE.
    """
    t_vals = np.unique(T)
    if len(t_vals) < 2:
        raise ValueError(f"Need >= 2 treatment levels, got {len(t_vals)}")
    t_low, t_high = t_vals.min(), t_vals.max()

    unique_strata = np.unique(strata)
    records = []

    for s in unique_strata:
        s_mask = strata == s
        y_high = Y[(s_mask) & (T == t_high)]
        y_low = Y[(s_mask) & (T == t_low)]
        if len(y_high) < 5 or len(y_low) < 5:
            continue
        diff = float(np.mean(y_high) - np.mean(y_low))
        se = float(np.sqrt(np.var(y_high) / len(y_high) + np.var(y_low) / len(y_low)))
        records.append({
            "stratum": s, "cate": diff, "se": se,
            "n_high": len(y_high), "n_low": len(y_low),
            "mean_high": float(np.mean(y_high)),
            "mean_low": float(np.mean(y_low)),
        })

    strata_df = pd.DataFrame(records)
    if len(strata_df) == 0:
        return StratifiedCATEResult(outcome_name, 0.0, 0.0, strata_df,
                                    np.zeros(len(Y)), 0)

    # Weighted ATE
    weights = strata_df["n_high"] + strata_df["n_low"]
    ate = float(np.average(strata_df["cate"], weights=weights))
    ate_se = float(np.sqrt(np.average(strata_df["se"] ** 2, weights=weights)))

    # Map per-stratum CATE to per-sample
    stratum_to_cate = dict(zip(strata_df["stratum"], strata_df["cate"]))
    cate_by_sample = np.array([stratum_to_cate.get(s, ate) for s in strata])

    return StratifiedCATEResult(
        outcome_name=outcome_name,
        ate=ate, ate_se=ate_se,
        strata_df=strata_df,
        cate_by_sample=cate_by_sample,
        n_strata=len(strata_df),
    )


def estimate_tlearner_cate(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    outcome_name: str,
    model_params: Optional[Dict[str, Any]] = None,
) -> TLearnerCATEResult:
    """Estimate CATE via T-Learner (separate outcome models per treatment).

    Fits μ_1(x) = E[Y|T=high, X=x] and μ_0(x) = E[Y|T=low, X=x] separately.
    CATE(x) = μ_1(x) - μ_0(x).

    No residualization — avoids the DML identification issue with iPinYou flat-bid.

    Args:
        Y: Outcome array (n,).
        T: Treatment array (n,) — discrete values.
        X: Feature matrix (n, p).
        outcome_name: Label for this outcome.
        model_params: LGBMRegressor kwargs.

    Returns:
        TLearnerCATEResult with per-sample CATE and feature importances.
    """
    if not LGBM_AVAILABLE:
        raise ImportError("lightgbm required")

    params = model_params or dict(_DEFAULT_MODEL_PARAMS)
    t_vals = np.unique(T)
    t_low, t_high = t_vals.min(), t_vals.max()

    # Fit separate models
    mask_high = T == t_high
    mask_low = T == t_low

    model_1 = LGBMRegressor(**params)
    model_1.fit(X[mask_high], Y[mask_high])

    model_0 = LGBMRegressor(**params)
    model_0.fit(X[mask_low], Y[mask_low])

    # Predict on full X
    mu_1 = model_1.predict(X)
    mu_0 = model_0.predict(X)
    cate = mu_1 - mu_0

    ate = float(np.mean(cate))

    return TLearnerCATEResult(
        outcome_name=outcome_name,
        ate=ate,
        cate=cate,
        mu_1=mu_1,
        mu_0=mu_0,
        feature_importances_1=model_1.feature_importances_,
        feature_importances_0=model_0.feature_importances_,
        feature_names=[f"X{i}" for i in range(X.shape[1])],
    )


def build_strata(
    X: np.ndarray,
    feature_names: List[str],
    hour_bins: int = 4,
) -> np.ndarray:
    """Build stratum labels from X for stratified CATE.

    Creates strata from (advertiser_taxonomy, exchange, hour_bin).

    Args:
        X: Feature matrix with columns matching feature_names.
        feature_names: Column names of X.
        hour_bins: Number of hour bins (default 4: 0-6, 6-12, 12-18, 18-24).

    Returns:
        String array of stratum labels.
    """
    tax_idx = feature_names.index("advertiser_taxonomy") if "advertiser_taxonomy" in feature_names else 0
    ex_idx = feature_names.index("adexchange") if "adexchange" in feature_names else 2
    hour_idx = feature_names.index("hour") if "hour" in feature_names else 1

    tax = X[:, tax_idx].astype(int)
    ex = X[:, ex_idx].astype(int)
    hour_bin = np.clip(X[:, hour_idx].astype(int) // (24 // hour_bins), 0, hour_bins - 1)

    return np.array([f"{t}_{e}_{h}" for t, e, h in zip(tax, ex, hour_bin)])
