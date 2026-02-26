"""Selection Bias Diagnostics for RTB.

Reusable diagnostic functions for quantifying selection bias:
- Covariate shift analysis (KS test + effect size)
- IPW-weighted CTR estimation by propensity bucket
- Subgroup-level bias decomposition
- Propensity model specification sensitivity analysis
"""

from typing import NamedTuple, List, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    LOGISTIC_AVAILABLE = True
except ImportError:
    LOGISTIC_AVAILABLE = False


# =============================================================================
# Result Types
# =============================================================================

class CovariateShiftResult(NamedTuple):
    """Result of covariate shift analysis."""
    ks_df: pd.DataFrame  # columns: feature, ks_stat, p_value, cohens_d, significant
    n_significant: int
    avg_ks_stat: float


class BucketCTRResult(NamedTuple):
    """Result of bucket-level CTR estimation."""
    bucket_df: pd.DataFrame  # bucket-level stats: naive_ctr, ipw_ctr, n_bids, n_wins, n_clicks, ess
    naive_ctr: float
    ipw_ctr: float
    horvitz_thompson_ctr: float  # Σ(click/p) / N


class SubgroupBiasResult(NamedTuple):
    """Result of subgroup bias analysis."""
    subgroup_df: pd.DataFrame  # subgroup, naive_ctr, ipw_ctr, bias_pct, n_samples
    max_abs_bias: float
    heterogeneity: bool  # True if subgroups have different bias directions


class SensitivityResult(NamedTuple):
    """Result of propensity model sensitivity analysis."""
    results_df: pd.DataFrame  # model_spec, auc, naive_ctr, ipw_ctr, bias_pct, overlap


# =============================================================================
# Covariate Shift Analysis
# =============================================================================

def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size between two samples."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def run_covariate_shift(
    df: pd.DataFrame,
    group_col: str,
    features: List[str],
    alpha: float = 0.05,
    sample_n: int = 500_000,
) -> CovariateShiftResult:
    """Run KS test + Cohen's d for covariate shift between groups.

    Args:
        df: DataFrame with features and group column
        group_col: Binary group column (e.g. 'win', 'click')
        features: List of numerical feature names
        alpha: Significance level
        sample_n: Max samples per group for KS test (memory efficiency)

    Returns:
        CovariateShiftResult with per-feature statistics
    """
    df_pos = df[df[group_col] == 1]
    df_neg = df[df[group_col] == 0]

    # Sample for memory efficiency
    if len(df_pos) > sample_n:
        df_pos = df_pos.sample(sample_n, random_state=42)
    if len(df_neg) > sample_n:
        df_neg = df_neg.sample(sample_n, random_state=42)

    results = []
    for feature in features:
        if feature not in df.columns:
            continue
        pos_vals = df_pos[feature].dropna().values
        neg_vals = df_neg[feature].dropna().values

        if len(pos_vals) == 0 or len(neg_vals) == 0:
            continue

        ks_stat, p_value = stats.ks_2samp(pos_vals, neg_vals)
        d = _cohens_d(pos_vals, neg_vals)

        results.append({
            "feature": feature,
            "ks_stat": ks_stat,
            "p_value": p_value,
            "cohens_d": d,
            "abs_cohens_d": abs(d),
            "significant": p_value < alpha,
        })

    ks_df = pd.DataFrame(results).sort_values("ks_stat", ascending=False).reset_index(drop=True)
    n_significant = int(ks_df["significant"].sum()) if len(ks_df) > 0 else 0
    avg_ks = float(ks_df["ks_stat"].mean()) if len(ks_df) > 0 else 0.0

    return CovariateShiftResult(
        ks_df=ks_df,
        n_significant=n_significant,
        avg_ks_stat=avg_ks,
    )


# =============================================================================
# Bucket-level IPW-weighted CTR
# =============================================================================

def compute_bucket_ctr(
    propensity: np.ndarray,
    win: np.ndarray,
    click: np.ndarray,
    n_buckets: int = 10,
    clip_range: Tuple[float, float] = (0.01, 0.99),
) -> BucketCTRResult:
    """Compute naive and IPW-weighted CTR by propensity bucket.

    Unlike naive per-bucket CTR (which is tautologically flat),
    this computes IPW-weighted CTR within each bucket to reveal
    whether bias varies with propensity.

    Args:
        propensity: Win propensity P(Win|X)
        win: Win indicator (binary)
        click: Click indicator (binary)
        n_buckets: Number of propensity quantile buckets
        clip_range: Propensity clipping range

    Returns:
        BucketCTRResult with bucket-level and aggregate statistics
    """
    prop_clipped = np.clip(propensity, clip_range[0], clip_range[1])

    # Create bucket assignments using all bids
    bucket_labels = pd.qcut(prop_clipped, q=n_buckets, labels=False, duplicates="drop")

    tmp = pd.DataFrame({
        "propensity": prop_clipped,
        "win": win.astype(float),
        "click": click.astype(float),
        "bucket": bucket_labels,
    })

    bucket_rows = []
    for bucket_id, grp in tmp.groupby("bucket"):
        n_bids = len(grp)
        n_wins = int(grp["win"].sum())
        n_clicks = int(grp["click"].sum())
        avg_ps = float(grp["propensity"].mean())

        # Naive CTR (won impressions only)
        naive_ctr = n_clicks / max(n_wins, 1)

        # Self-normalized IPW CTR: Σ(click * win/p) / Σ(win/p)
        w = grp["win"] / grp["propensity"]
        numerator = (grp["click"] * w).sum()
        denominator = w.sum()
        ipw_ctr = float(numerator / max(denominator, 1e-8))

        # Horvitz-Thompson: Σ(click * win/p) / N_bucket
        ht_ctr = float(numerator / max(n_bids, 1))

        # Effective Sample Size
        w_vals = w.values
        ess = float((w_vals.sum() ** 2) / max((w_vals ** 2).sum(), 1e-8))

        bucket_rows.append({
            "bucket": bucket_id,
            "avg_propensity": avg_ps,
            "n_bids": n_bids,
            "n_wins": n_wins,
            "n_clicks": n_clicks,
            "naive_ctr": naive_ctr,
            "ipw_ctr": ipw_ctr,
            "ht_ctr": ht_ctr,
            "ess": ess,
        })

    bucket_df = pd.DataFrame(bucket_rows)

    # Aggregate CTR estimates
    w_all = tmp["win"] / tmp["propensity"]
    naive_ctr_total = float(tmp.loc[tmp["win"] == 1, "click"].mean())
    ipw_ctr_total = float(
        (tmp["click"] * w_all).sum() / max(w_all.sum(), 1e-8)
    )
    ht_ctr_total = float(
        (tmp["click"] * w_all).sum() / max(len(tmp), 1)
    )

    return BucketCTRResult(
        bucket_df=bucket_df,
        naive_ctr=naive_ctr_total,
        ipw_ctr=ipw_ctr_total,
        horvitz_thompson_ctr=ht_ctr_total,
    )


# =============================================================================
# Subgroup Bias Analysis
# =============================================================================

def compute_subgroup_bias(
    df: pd.DataFrame,
    propensity: np.ndarray,
    group_col: str,
    win_col: str = "win",
    click_col: str = "click",
    clip_range: Tuple[float, float] = (0.01, 0.99),
    min_clicks: int = 5,
) -> SubgroupBiasResult:
    """Compute naive vs IPW CTR bias per subgroup.

    Detects Simpson's Paradox: overall bias may be small
    but subgroup-level biases can be large and heterogeneous.

    Args:
        df: DataFrame with group_col, win_col, click_col
        propensity: Win propensity P(Win|X)
        group_col: Categorical column defining subgroups
        win_col: Win indicator column name
        click_col: Click indicator column name
        clip_range: Propensity clipping range
        min_clicks: Minimum clicks in a subgroup to include

    Returns:
        SubgroupBiasResult with per-subgroup bias decomposition
    """
    prop_clipped = np.clip(propensity, clip_range[0], clip_range[1])

    tmp = df[[group_col, win_col, click_col]].copy()
    tmp["propensity"] = prop_clipped

    subgroup_rows = []
    for name, grp in tmp.groupby(group_col):
        n_total = len(grp)
        n_wins = int(grp[win_col].sum())
        n_clicks = int(grp[click_col].sum())

        if n_clicks < min_clicks:
            continue

        naive_ctr = n_clicks / max(n_wins, 1)

        w = grp[win_col] / grp["propensity"]
        ipw_ctr = float(
            (grp[click_col] * w).sum() / max(w.sum(), 1e-8)
        )

        bias_pct = float(
            (naive_ctr - ipw_ctr) / max(abs(ipw_ctr), 1e-10) * 100
        ) if ipw_ctr != 0 else 0.0

        subgroup_rows.append({
            "subgroup": str(name),
            "n_bids": n_total,
            "n_wins": n_wins,
            "n_clicks": n_clicks,
            "win_rate": n_wins / max(n_total, 1),
            "naive_ctr": naive_ctr,
            "ipw_ctr": ipw_ctr,
            "bias_pct": bias_pct,
        })

    subgroup_df = pd.DataFrame(subgroup_rows)

    if len(subgroup_df) == 0:
        return SubgroupBiasResult(
            subgroup_df=subgroup_df,
            max_abs_bias=0.0,
            heterogeneity=False,
        )

    max_abs_bias = float(subgroup_df["bias_pct"].abs().max())

    # Check heterogeneity: different bias directions across subgroups
    signs = subgroup_df["bias_pct"].apply(np.sign)
    heterogeneity = bool(signs.nunique() > 1 and (signs != 0).any())

    return SubgroupBiasResult(
        subgroup_df=subgroup_df,
        max_abs_bias=max_abs_bias,
        heterogeneity=heterogeneity,
    )


# =============================================================================
# Propensity Model Sensitivity Analysis
# =============================================================================

def _fit_propensity_crossfit(
    X: np.ndarray,
    win: np.ndarray,
    model_type: str = "lgb",
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Fit propensity model with cross-fitting and return out-of-fold predictions.

    Args:
        X: Feature matrix
        win: Binary win indicator
        model_type: 'lgb' or 'logistic'
        n_folds: Number of cross-fitting folds
        seed: Random seed

    Returns:
        Out-of-fold propensity scores
    """
    propensity = np.zeros(len(win))
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, val_idx in kfold.split(X, win):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = win[train_idx]

        if model_type == "lgb" and LGB_AVAILABLE:
            model = lgb.LGBMClassifier(
                objective="binary",
                num_leaves=31,
                learning_rate=0.05,
                n_estimators=200,
                min_child_samples=20,
                verbose=-1,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
        else:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            model = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
            model.fit(X_train, y_train)

        propensity[val_idx] = model.predict_proba(X_val)[:, 1]

    return propensity


def propensity_sensitivity(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    win_col: str = "win",
    click_col: str = "click",
    n_folds: int = 5,
    clip_range: Tuple[float, float] = (0.01, 0.99),
    model_type: str = "lgb",
    sample_n: Optional[int] = 2_000_000,
) -> SensitivityResult:
    """Test sensitivity of bias estimates to propensity model specification.

    Fits multiple propensity models with different feature sets and compares
    the resulting bias estimates.

    Args:
        df: DataFrame with features, win, click columns
        feature_sets: Dict mapping spec name to list of feature names
        win_col: Win indicator column
        click_col: Click indicator column
        n_folds: Cross-fitting folds
        clip_range: Propensity clipping range
        model_type: 'lgb' or 'logistic'
        sample_n: Subsample size for speed (None = use all)

    Returns:
        SensitivityResult with per-specification metrics
    """
    # Subsample for speed
    if sample_n is not None and len(df) > sample_n:
        df_sub = df.sample(sample_n, random_state=42).reset_index(drop=True)
    else:
        df_sub = df.reset_index(drop=True)

    win = df_sub[win_col].values.astype(float)
    click = df_sub[click_col].values.astype(float)

    results = []
    for spec_name, features in feature_sets.items():
        available = [f for f in features if f in df_sub.columns]
        if len(available) == 0:
            continue

        X = df_sub[available].fillna(0).values

        propensity = _fit_propensity_crossfit(
            X, win, model_type=model_type, n_folds=n_folds,
        )

        prop_clipped = np.clip(propensity, clip_range[0], clip_range[1])

        # Metrics
        auc = roc_auc_score(win, propensity)

        # Naive CTR
        win_mask = win == 1
        naive_ctr = float(click[win_mask].mean()) if win_mask.sum() > 0 else 0.0

        # IPW CTR (self-normalized)
        w = win / prop_clipped
        ipw_ctr = float((click * w).sum() / max(w.sum(), 1e-8))

        bias_pct = float(
            (naive_ctr - ipw_ctr) / max(abs(ipw_ctr), 1e-10) * 100
        ) if ipw_ctr != 0 else 0.0

        # Overlap
        overlap = float(np.mean((propensity > 0.1) & (propensity < 0.9)))

        # ESS
        ess = float((w.sum() ** 2) / max((w ** 2).sum(), 1e-8))
        ess_ratio = ess / max(len(w), 1)

        results.append({
            "model_spec": spec_name,
            "n_features": len(available),
            "auc": auc,
            "naive_ctr": naive_ctr,
            "ipw_ctr": ipw_ctr,
            "bias_pct": bias_pct,
            "overlap_01_09": overlap,
            "ess_ratio": ess_ratio,
        })

    results_df = pd.DataFrame(results)
    return SensitivityResult(results_df=results_df)
