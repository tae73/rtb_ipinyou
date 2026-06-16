"""Closed-form unit tests for src.metrics.evaluation.

These pin the mathematical contracts of the three public metric functions
(`compute_ece`, `compute_ieb`, `compute_metrics`) using small, hand-verifiable
inputs whose expected values are derived from the metric definitions rather
than from any recorded artifact. Everything is pure-numpy, fixed-seed, and
runs in well under a second.

Definitions under test (from evaluation.py docstrings):
  - ECE  = Σ_bin (frac_in_bin * |mean(y_true in bin) - mean(y_pred in bin)|),
           equal-width bins over [0, 1) via np.linspace(0, 1, n_bins+1),
           half-open [lo, hi); an exact 1.0 prediction falls in no bin.
  - IEB  = |mean(y_pred) - mean(y_true)| / max(mean(y_true), 1e-8).
  - AUC  = Mann-Whitney rank statistic (1.0 for a perfect separator).
"""

import numpy as np
import pytest

from src.metrics.evaluation import (
    EvalMetrics,
    _numpy_roc_auc,
    compute_ece,
    compute_ieb,
    compute_metrics,
)


# =============================================================================
# compute_ece
# =============================================================================


def test_ece_perfectly_calibrated_within_bin_is_zero():
    """Constant pred == constant accuracy within the only occupied bin -> ECE 0.

    All preds = 0.45 (single bin [0.4, 0.5)); exactly 45% of labels are 1.
    Within that bin mean(y_true) == mean(y_pred) == 0.45 so |gap| == 0.
    """
    n = 100
    y_pred = np.full(n, 0.45)
    y_true = np.zeros(n)
    y_true[:45] = 1.0  # mean 0.45 == confidence
    assert compute_ece(y_true, y_pred, n_bins=10) == pytest.approx(0.0, abs=1e-12)


def test_ece_closed_form_two_bins():
    """Hand-computed ECE across two occupied bins.

    Bin [0.0,0.1): preds 0.05, n=4, true rate 0.25 -> |0.25-0.05| = 0.20
    Bin [0.9,1.0): preds 0.95, n=6, true rate 0.50 -> |0.50-0.95| = 0.45
    frac: 4/10 and 6/10.
    ECE = 0.4*0.20 + 0.6*0.45 = 0.08 + 0.27 = 0.35
    """
    y_pred = np.array([0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95])
    y_true = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    assert compute_ece(y_true, y_pred, n_bins=10) == pytest.approx(0.35, abs=1e-12)


def test_ece_calibrated_synthetic_set_is_small():
    """A genuinely well-calibrated synthetic set has small (near-zero) ECE.

    Draw p_i ~ U(0,1), then y_i ~ Bernoulli(p_i). By construction the
    empirical accuracy in each bin tracks the bin's mean confidence, so ECE
    should be small for a large sample.
    """
    rng = np.random.default_rng(0)
    n = 50_000
    y_pred = rng.random(n)
    y_true = (rng.random(n) < y_pred).astype(np.float64)
    ece = compute_ece(y_true, y_pred, n_bins=10)
    assert 0.0 <= ece < 0.02


def test_ece_miscalibrated_set_is_larger_than_calibrated():
    """A systematically overconfident set has a clearly larger ECE."""
    rng = np.random.default_rng(1)
    n = 50_000
    p = rng.random(n)
    y_true = (rng.random(n) < p).astype(np.float64)
    calibrated_ece = compute_ece(y_true, p, n_bins=10)
    # Shift predictions toward 1.0 -> overconfident, miscalibrated.
    p_bad = np.clip(p + 0.3, 0.0, 0.999)
    bad_ece = compute_ece(y_true, p_bad, n_bins=10)
    assert bad_ece > calibrated_ece
    assert bad_ece > 0.1


def test_ece_exact_one_falls_in_no_bin():
    """A prediction of exactly 1.0 is excluded (half-open last bin [0.9,1.0)).

    With a single sample pred=1.0, no bin is occupied so ECE == 0.0 even though
    the label is 0 (the contract intentionally drops exact-1.0 predictions).
    """
    y_pred = np.array([1.0])
    y_true = np.array([0.0])
    assert compute_ece(y_true, y_pred, n_bins=10) == pytest.approx(0.0, abs=1e-12)


# =============================================================================
# compute_ieb
# =============================================================================


def test_ieb_zero_when_means_match():
    """IEB == 0 exactly when mean(pred) == mean(actual)."""
    rng = np.random.default_rng(2)
    y_true = (rng.random(1000) < 0.3).astype(np.float64)
    actual_mean = y_true.mean()
    # Predictions with identical mean but arbitrary spread.
    y_pred = np.full_like(y_true, actual_mean)
    assert compute_ieb(y_true, y_pred) == pytest.approx(0.0, abs=1e-12)


def test_ieb_closed_form_relative_gap():
    """IEB equals the relative gap |mean_pred - mean_true| / mean_true.

    mean_true = 0.20, mean_pred = 0.30 -> |0.30-0.20|/0.20 = 0.5.
    """
    y_true = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # mean 0.2
    y_pred = np.full(5, 0.30)  # mean 0.3
    assert compute_ieb(y_true, y_pred) == pytest.approx(0.5, abs=1e-12)


def test_ieb_floor_guards_zero_actual_mean():
    """When mean(actual) == 0 the denominator is floored at 1e-8 (no div-by-0)."""
    y_true = np.zeros(10)
    y_pred = np.full(10, 1e-9)
    # |1e-9 - 0| / max(0, 1e-8) = 1e-9 / 1e-8 = 0.1
    assert compute_ieb(y_true, y_pred) == pytest.approx(0.1, rel=1e-9)


# =============================================================================
# compute_metrics / AUC
# =============================================================================


def test_perfect_predictor_auc_is_one():
    """A predictor that perfectly separates the classes scores AUC == 1.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    y_pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = compute_metrics(y_true, y_pred)
    assert isinstance(m, EvalMetrics)
    assert m.auc == pytest.approx(1.0, abs=1e-12)


def test_inverted_predictor_auc_is_zero():
    """A perfectly anti-correlated predictor scores AUC == 0.0."""
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    y_pred = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert _numpy_roc_auc(y_true, y_pred) == pytest.approx(0.0, abs=1e-12)


def test_random_tie_predictor_auc_is_half():
    """All-equal scores give the Mann-Whitney tie value 0.5."""
    y_true = np.array([0, 1, 0, 1, 0, 1], dtype=np.float64)
    y_pred = np.full(6, 0.5)
    assert _numpy_roc_auc(y_true, y_pred) == pytest.approx(0.5, abs=1e-12)


def test_auc_single_class_falls_back_to_half():
    """compute_metrics catches the one-class ValueError and returns AUC 0.5."""
    y_true = np.ones(8, dtype=np.float64)
    y_pred = np.linspace(0.1, 0.9, 8)
    assert compute_metrics(y_true, y_pred).auc == pytest.approx(0.5, abs=1e-12)


def test_numpy_roc_auc_raises_on_single_class():
    """The bare AUC helper raises ValueError when only one class is present."""
    with pytest.raises(ValueError):
        _numpy_roc_auc(np.zeros(5), np.linspace(0, 1, 5))


def test_compute_metrics_perfect_threshold_classification():
    """At threshold 0.5 a perfect separator gives accuracy/precision/recall/f1 = 1."""
    y_true = np.array([0, 0, 1, 1], dtype=np.float64)
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, y_pred, threshold=0.5)
    assert m.accuracy == pytest.approx(1.0, abs=1e-12)
    assert m.precision == pytest.approx(1.0, abs=1e-12)
    assert m.recall == pytest.approx(1.0, abs=1e-12)
    assert m.f1 == pytest.approx(1.0, abs=1e-12)
    # Perfectly confident & correct -> tiny log loss, near-zero ECE.
    assert m.log_loss < 0.25
    assert m.ece < 0.5


def test_compute_metrics_drops_nans():
    """NaNs in either array are dropped before any metric is computed."""
    y_true = np.array([0.0, 1.0, np.nan, 1.0, 0.0])
    y_pred = np.array([0.2, 0.9, 0.5, np.nan, 0.1])
    # Surviving rows: (0,0.2), (1,0.9), (0,0.1) -> perfectly ordered -> AUC 1.0
    m = compute_metrics(y_true, y_pred)
    assert np.isfinite(m.auc)
    assert m.auc == pytest.approx(1.0, abs=1e-12)
    assert np.isfinite(m.log_loss)


def test_compute_metrics_ece_ieb_consistency():
    """compute_metrics' embedded ece/ieb equal the standalone functions.

    Guards against drift between the convenience wrapper and the primitives.
    """
    rng = np.random.default_rng(3)
    n = 2000
    y_pred = rng.random(n)
    y_true = (rng.random(n) < y_pred).astype(np.float64)
    m = compute_metrics(y_true, y_pred)
    # compute_metrics drops NaNs (none here) then calls the same primitives.
    assert m.ece == pytest.approx(compute_ece(y_true, y_pred, n_bins=10), rel=1e-12)
    assert m.ieb == pytest.approx(compute_ieb(y_true, y_pred), rel=1e-12)
