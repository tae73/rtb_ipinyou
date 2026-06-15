"""Clustered-inference utilities for the advertiser-level surplus-gap analysis.

Tools to adjudicate "underpowered homogeneous effect" vs "genuine heterogeneity" when comparing two
bidding policies across a SMALL number of advertiser clusters (here 5). Pure numpy/scipy.

- `cochrans_q`     — between-cluster heterogeneity (Q, p, I², DerSimonian–Laird τ²). The decisive test:
                     does the per-advertiser gap vary beyond within-advertiser sampling noise?
- `cluster_t_mde`  — minimum detectable effect / power for the cluster-mean gap (cluster-t, df=k−1).
                     Calibrates "underpowered" honestly; report as order-of-magnitude at small k.
- `sign_test`      — fraction of clusters with a positive gap (exact binomial). Near-zero power at k=5;
                     a non-rejection licenses nothing — descriptive footnote only.
- `loocv_means`    — leave-one-cluster-out means (exposes single-cluster leverage).
"""

from typing import NamedTuple

import numpy as np
from scipy import stats


class HeterogeneityResult(NamedTuple):
    Q: float          # Cochran's Q
    df: int           # k - 1
    p_value: float    # chi2.sf(Q, df) — small ⇒ reject homogeneity
    I2: float         # proportion of total variation due to heterogeneity (0..1)
    tau2: float       # DerSimonian–Laird between-cluster variance estimate
    fixed_effect: float   # inverse-variance-weighted mean gap


class MDEResult(NamedTuple):
    k: int
    mean_gap: float
    sd_between: float       # std of per-cluster gaps (ddof=1)
    se_mean: float          # sd_between / sqrt(k)
    mde_80: float           # smallest |mean| detectable at 80% power, two-sided α=0.05 (cluster-t)
    observed_above_mde: bool
    n_clusters_for_power: float  # clusters needed to detect the OBSERVED mean (normal approx)


def cochrans_q(gaps: np.ndarray, ses: np.ndarray) -> HeterogeneityResult:
    """Cochran's Q heterogeneity test + I² + DerSimonian–Laird τ² over cluster estimates.

    Args:
        gaps: per-cluster point estimates (e.g. per-advertiser surplus gap), shape (k,).
        ses:  per-cluster standard errors (e.g. within-advertiser bootstrap SE), shape (k,).
    """
    gaps = np.asarray(gaps, np.float64)
    ses = np.asarray(ses, np.float64)
    k = gaps.shape[0]
    w = 1.0 / np.clip(ses ** 2, 1e-30, None)
    sw = w.sum()
    fe = float((w * gaps).sum() / sw)              # fixed-effect (inverse-variance) mean
    Q = float((w * (gaps - fe) ** 2).sum())
    df = k - 1
    p = float(stats.chi2.sf(Q, df)) if df > 0 else float("nan")
    I2 = float(max(0.0, (Q - df) / Q)) if Q > 0 else 0.0
    C = sw - (w ** 2).sum() / sw
    tau2 = float(max(0.0, (Q - df) / C)) if C > 0 else 0.0
    return HeterogeneityResult(Q=Q, df=df, p_value=p, I2=I2, tau2=tau2, fixed_effect=fe)


def cluster_t_mde(gaps: np.ndarray, alpha: float = 0.05, power: float = 0.80) -> MDEResult:
    """Minimum detectable effect for the cluster-MEAN gap (cluster-t, df=k−1).

    At small k the t-quantiles make this fragile — report as an ORDER-OF-MAGNITUDE reality check
    ("the design can only detect ~X effects; the observed effect is ~Y"), never a precise number.
    """
    gaps = np.asarray(gaps, np.float64)
    k = gaps.shape[0]
    mean_g = float(gaps.mean())
    sd_b = float(gaps.std(ddof=1)) if k > 1 else float("nan")
    se_mean = sd_b / np.sqrt(k) if k > 0 else float("nan")
    df = k - 1
    if df > 0 and np.isfinite(se_mean):
        mde = float((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * se_mean)
    else:
        mde = float("nan")
    # clusters needed to detect the OBSERVED mean (large-sample normal approx, fixed sd_between)
    if mean_g != 0 and np.isfinite(sd_b):
        z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
        n_needed = float((z * sd_b / abs(mean_g)) ** 2)
    else:
        n_needed = float("inf")
    return MDEResult(k=k, mean_gap=mean_g, sd_between=sd_b, se_mean=se_mean, mde_80=mde,
                     observed_above_mde=bool(abs(mean_g) > mde) if np.isfinite(mde) else False,
                     n_clusters_for_power=n_needed)


def sign_test(gaps: np.ndarray, alternative: str = "greater") -> dict:
    """Exact binomial sign test: are the per-cluster gaps consistently positive?

    Near-zero power at small k (k=5: even 5/5 gives p=0.031; 4/5 → 0.187). Footnote only.
    """
    gaps = np.asarray(gaps, np.float64)
    k = int(gaps.shape[0])
    k_pos = int((gaps > 0).sum())
    p = float(stats.binomtest(k_pos, k, 0.5, alternative=alternative).pvalue) if k > 0 else float("nan")
    return {"k_pos": k_pos, "n": k, "p_greater": p}


def loocv_means(gaps: np.ndarray) -> np.ndarray:
    """Leave-one-cluster-out means (exposes single-cluster leverage)."""
    gaps = np.asarray(gaps, np.float64)
    n = gaps.shape[0]
    if n <= 1:
        return np.array([], np.float64)
    total = gaps.sum()
    return (total - gaps) / (n - 1)
