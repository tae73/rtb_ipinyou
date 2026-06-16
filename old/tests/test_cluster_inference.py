"""Tests for src.metrics.cluster_inference — clustered heterogeneity / power tools."""

import numpy as np
import pytest

from src.metrics.cluster_inference import (
    cluster_t_mde,
    cochrans_q,
    loocv_means,
    sign_test,
)


def test_cochrans_q_homogeneous_low_I2():
    """Identical cluster estimates (only sampling noise) -> Q small, I² ≈ 0, can't reject homogeneity."""
    gaps = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    ses = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    r = cochrans_q(gaps, ses)
    assert r.Q == pytest.approx(0.0, abs=1e-9)
    assert r.I2 == pytest.approx(0.0)
    assert r.tau2 == pytest.approx(0.0)
    assert r.p_value > 0.5
    assert r.fixed_effect == pytest.approx(10.0)


def test_cochrans_q_heterogeneous_high_I2():
    """Widely-varying cluster estimates with tight SEs -> Q large, I² high, reject homogeneity."""
    gaps = np.array([11.0, 1.0, -0.6, -2.6, -5.1])   # the observed-shape pattern (in millions)
    ses = np.array([1.0, 1.0, 1.0, 1.0, 1.0])         # tight relative to spread
    r = cochrans_q(gaps, ses)
    assert r.Q > r.df                                  # excess over df
    assert r.I2 > 0.5                                  # substantial heterogeneity
    assert r.tau2 > 0.0
    assert r.p_value < 0.05                            # reject homogeneity


def test_cluster_t_mde_order_of_magnitude():
    """MDE ≈ (t.975,4 + t.80,4)·SD/√5; observed mean below MDE -> not detectable."""
    gaps = np.array([11.0, 0.9, -0.6, -2.6, -5.1])
    r = cluster_t_mde(gaps, alpha=0.05, power=0.80)
    assert r.k == 5
    sd = gaps.std(ddof=1)
    from scipy import stats
    expected_mde = (stats.t.ppf(0.975, 4) + stats.t.ppf(0.80, 4)) * sd / np.sqrt(5)
    assert r.mde_80 == pytest.approx(expected_mde, rel=1e-9)
    # mean ≈ 0.72, MDE is several -> observed below MDE
    assert r.mean_gap < r.mde_80
    assert r.observed_above_mde is False


def test_sign_test_consistency():
    """5/5 positive -> p=0.031; 2/5 -> not significant."""
    assert sign_test(np.array([1.0, 2, 3, 4, 5]))["p_greater"] == pytest.approx(0.5 ** 5, abs=1e-6)
    s = sign_test(np.array([11.0, 0.9, -0.6, -2.6, -5.1]))
    assert s["k_pos"] == 2 and s["n"] == 5
    assert s["p_greater"] > 0.5                        # nowhere near significant


def test_loocv_means_drop_one_leverage():
    """Leave-one-out means; dropping the single large positive flips the mean negative."""
    gaps = np.array([11.0, 0.9, -0.6, -2.6, -5.1])    # full mean ≈ +0.72
    lo = loocv_means(gaps)
    assert lo.shape == (5,)
    # drop index 0 (the +11 leverage point) -> mean of the other 4 is negative
    assert lo[0] == pytest.approx(np.mean(gaps[1:]))
    assert lo[0] < 0
    # identity: each loo mean = (total - g_i)/(n-1)
    assert lo[2] == pytest.approx((gaps.sum() - gaps[2]) / 4)


def test_cochrans_q_fixed_effect_inverse_variance_weighted():
    """Fixed-effect mean down-weights high-SE clusters."""
    gaps = np.array([10.0, 0.0])
    ses = np.array([1.0, 10.0])     # second estimate is imprecise -> ~ignored
    r = cochrans_q(gaps, ses)
    assert r.fixed_effect == pytest.approx(10.0, abs=0.2)
