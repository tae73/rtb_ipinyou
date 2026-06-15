"""Closed-form unit tests for src.metrics.calibration.

Pins the mathematical contracts of the equal-frequency calibration tooling
added for the rare-event RTB regime:

  - ``quantile_reliability`` -> equal-frequency reliability table + count-
    weighted quantile-ECE.
  - ``slice_calibration``    -> per-slice (mean_pred, mean_true, bias, count,
    ece) following the compute_subgroup_bias grouping pattern.
  - ``safe_quantile_bins``   -> robust quantile labelling (ties / degenerate).
  - ``reliability_summary``  -> scalar contrast vs legacy equal-width ECE.

Definitions under test (from calibration.py docstrings):
  - quantile-ECE = Σ_b count_b |mean_pred_b - mean_true_b| / Σ_b count_b,
    over EQUAL-FREQUENCY (quantile) bins.
  - per-bin bias = mean_pred - mean_true (signed).

Everything is pure-numpy, fixed-seed, hand-verifiable, sub-second.
"""

import numpy as np
import pytest

from src.metrics.calibration import (
    ReliabilityBin,
    ReliabilityTable,
    SliceCalibrationResult,
    SliceCalibrationRow,
    cross_fit_isotonic,
    fit_isotonic,
    segment_cross_fit_isotonic,
    quantile_reliability,
    reliability_summary,
    safe_quantile_bins,
    slice_calibration,
)
from src.metrics.evaluation import _numpy_roc_auc, compute_ieb


# =============================================================================
# quantile_reliability — perfectly calibrated -> quantile-ECE == 0
# =============================================================================


def test_quantile_ece_perfectly_calibrated_is_zero():
    """Within every quantile bin mean_pred == empirical rate -> q-ECE 0.

    Construction: 10 score levels {0.05, 0.15, ..., 0.95}, each repeated 100x.
    For a level with score p we set exactly round(100*p) of its 100 rows to 1,
    so the empirical rate in that level equals p. With n_bins=10 each distinct
    score becomes its own equal-frequency bin, so per-bin gap == 0 everywhere.
    """
    levels = np.arange(0.05, 1.0, 0.10)  # 0.05 .. 0.95, 10 levels
    reps = 100
    y_prob = np.repeat(levels, reps)
    y_true = np.zeros_like(y_prob)
    for i, p in enumerate(levels):
        k = int(round(reps * p))
        seg = slice(i * reps, (i + 1) * reps)
        block = y_true[seg]
        block[:k] = 1.0
        y_true[seg] = block

    table = quantile_reliability(y_true, y_prob, n_bins=10)

    assert isinstance(table, ReliabilityTable)
    assert table.n_bins == 10
    assert table.n_samples == reps * len(levels)
    assert table.quantile_ece == pytest.approx(0.0, abs=1e-12)
    for b in table.bins:
        assert b.bias == pytest.approx(0.0, abs=1e-12)


# =============================================================================
# quantile_reliability — known miscalibrated -> hand-computed q-ECE
# =============================================================================


def test_quantile_ece_known_miscalibrated_closed_form():
    """Two equal-frequency bins with hand-computed gaps.

    n_bins=2 over a score vector with a clean median split:
      low  half:  pred 0.20 (x50), true rate 0.10 (5/50) -> gap 0.10
      high half:  pred 0.80 (x50), true rate 0.60 (30/50)-> gap 0.20
    Equal counts (50 each) => q-ECE = mean(0.10, 0.20) = 0.15.
    Signed bias is +0.10 (low) and +0.20 (high): both over-predicting.
    """
    n_half = 50
    y_prob = np.concatenate([np.full(n_half, 0.20), np.full(n_half, 0.80)])
    y_true = np.zeros(2 * n_half)
    y_true[:5] = 1.0  # 5/50 = 0.10 in low bin
    y_true[n_half : n_half + 30] = 1.0  # 30/50 = 0.60 in high bin

    table = quantile_reliability(y_true, y_prob, n_bins=2)

    assert table.n_bins == 2
    assert [b.count for b in table.bins] == [n_half, n_half]

    low, high = table.bins
    assert low.mean_pred == pytest.approx(0.20)
    assert low.mean_true == pytest.approx(0.10)
    assert low.bias == pytest.approx(0.10)
    assert high.mean_pred == pytest.approx(0.80)
    assert high.mean_true == pytest.approx(0.60)
    assert high.bias == pytest.approx(0.20)

    # count-weighted: (50*0.10 + 50*0.20) / 100 = 0.15
    assert table.quantile_ece == pytest.approx(0.15)


def test_quantile_ece_count_weighting_with_unequal_bins():
    """q-ECE is COUNT-weighted, not a plain bin mean (tie-collapse case).

    Three constant score levels 0.10 x50, 0.50 x30, 0.90 x20. With n_bins=4 the
    quantile edges land on the tied values, so duplicates-dropping yields three
    UNEQUAL-count bins (50 / 30 / 20) — one per distinct score:
      bin0: pred 0.10, true 5/50 = 0.10 -> |bias| 0.00, weight 50
      bin1: pred 0.50, true 12/30 = 0.40 -> |bias| 0.10, weight 30
      bin2: pred 0.90, true 14/20 = 0.70 -> |bias| 0.20, weight 20
    q-ECE = (50*0.00 + 30*0.10 + 20*0.20)/100 = (0 + 3 + 4)/100 = 0.07.
    A plain (unweighted) bin mean would give 0.10 — so this pins the weighting.
    """
    y_prob = np.concatenate([np.full(50, 0.10), np.full(30, 0.50), np.full(20, 0.90)])
    y_true = np.zeros(100)
    y_true[:5] = 1.0  # bin0: 5/50 = 0.10
    y_true[50:62] = 1.0  # bin1: 12/30 = 0.40
    y_true[80:94] = 1.0  # bin2: 14/20 = 0.70

    table = quantile_reliability(y_true, y_prob, n_bins=4)
    counts = sorted(b.count for b in table.bins)
    assert counts == [20, 30, 50]
    assert table.quantile_ece == pytest.approx(0.07)
    # not the unweighted bin mean
    assert table.quantile_ece != pytest.approx(0.10)


# =============================================================================
# safe_quantile_bins — equal-frequency counts & degenerate fallback
# =============================================================================


def test_equal_frequency_bins_hold_equal_counts():
    """With distinct scores each quantile bin holds ~n/n_bins samples."""
    rng = np.random.default_rng(0)
    y_prob = rng.random(1000)  # 1000 distinct-ish uniform draws
    labels = safe_quantile_bins(y_prob, n_bins=10)

    assert labels.shape == (1000,)
    _, counts = np.unique(labels, return_counts=True)
    assert len(counts) == 10
    # equal-frequency: every bin within +/-1 of 100
    assert counts.min() >= 99
    assert counts.max() <= 101
    assert int(counts.sum()) == 1000


def test_safe_quantile_bins_degenerate_constant_scores():
    """A (near-)constant score vector collapses to a single bin, no raise."""
    y_prob = np.full(500, 0.0003)  # rare-event constant score
    labels = safe_quantile_bins(y_prob, n_bins=10)
    assert labels.shape == (500,)
    assert np.unique(labels).tolist() == [0]

    # quantile_reliability must also survive and report a single bin.
    y_true = np.zeros(500)
    y_true[:1] = 1.0
    table = quantile_reliability(y_true, y_prob, n_bins=10)
    assert table.n_bins == 1
    assert table.bins[0].count == 500
    # gap = |0.0003 - 1/500| = |0.0003 - 0.002|
    assert table.quantile_ece == pytest.approx(abs(0.0003 - 1.0 / 500.0))


def test_safe_quantile_bins_empty_input():
    """Empty input returns an empty label array (no raise)."""
    labels = safe_quantile_bins(np.array([]), n_bins=10)
    assert labels.shape == (0,)
    table = quantile_reliability(np.array([]), np.array([]), n_bins=10)
    assert table.n_bins == 0
    assert table.n_samples == 0
    assert table.quantile_ece == 0.0


# =============================================================================
# slice_calibration — recovers injected per-slice bias
# =============================================================================


def test_slice_calibration_recovers_injected_bias():
    """Per-slice bias = mean_pred - mean_true recovers the injected offsets.

    Two slices "A" and "B":
      A: pred 0.30 (x200), true rate 0.50 (100/200) -> bias -0.20
      B: pred 0.70 (x100), true rate 0.40 (40/100)  -> bias +0.30
    max_abs_bias = 0.30; weighted_abs_bias = (200*0.20 + 100*0.30)/300 = 0.2333..
    """
    y_prob = np.concatenate([np.full(200, 0.30), np.full(100, 0.70)])
    y_true = np.concatenate(
        [
            np.array([1.0] * 100 + [0.0] * 100),  # A: 100/200 = 0.50
            np.array([1.0] * 40 + [0.0] * 60),  # B: 40/100  = 0.40
        ]
    )
    slice_values = np.array(["A"] * 200 + ["B"] * 100)

    result = slice_calibration(y_true, y_prob, slice_values, slice_name="grp")

    assert isinstance(result, SliceCalibrationResult)
    assert result.slice_name == "grp"
    assert {r.slice_value for r in result.rows} == {"A", "B"}

    by_val = {r.slice_value: r for r in result.rows}
    a, b = by_val["A"], by_val["B"]

    assert a.count == 200
    assert a.mean_pred == pytest.approx(0.30)
    assert a.mean_true == pytest.approx(0.50)
    assert a.bias == pytest.approx(-0.20)
    assert a.slice_name == "grp"

    assert b.count == 100
    assert b.mean_pred == pytest.approx(0.70)
    assert b.mean_true == pytest.approx(0.40)
    assert b.bias == pytest.approx(0.30)

    assert result.max_abs_bias == pytest.approx(0.30)
    assert result.weighted_abs_bias == pytest.approx(
        (200 * 0.20 + 100 * 0.30) / 300.0
    )


def test_slice_calibration_min_count_filters_levels():
    """Slice levels below min_count are dropped from the report."""
    y_prob = np.concatenate([np.full(100, 0.4), np.full(3, 0.9)])
    y_true = np.concatenate([np.zeros(100), np.ones(3)])
    slice_values = np.array(["big"] * 100 + ["tiny"] * 3)

    result = slice_calibration(
        y_true, y_prob, slice_values, slice_name="g", min_count=10
    )
    assert {r.slice_value for r in result.rows} == {"big"}


# =============================================================================
# reliability_summary — contrasts equal-frequency vs equal-width
# =============================================================================


def test_reliability_summary_fields_and_contrast():
    """Summary reports both ECE flavours and a non-negative max bin bias.

    Rare-event-like vector: scores mostly tiny with a small over-predicting
    cluster. The equal-frequency quantile-ECE must equal the table's q-ECE,
    and max_bin_abs_bias must equal the largest |bias| across bins.
    """
    y_prob = np.concatenate([np.full(80, 0.001), np.full(20, 0.50)])
    y_true = np.concatenate([np.zeros(80), np.ones(10).tolist() + [0.0] * 10])

    summary = reliability_summary(y_true, y_prob, n_bins=2)
    table = quantile_reliability(y_true, y_prob, n_bins=2)

    assert summary.quantile_ece == pytest.approx(table.quantile_ece)
    assert summary.n_samples == 100
    assert summary.equal_width_ece >= 0.0
    assert summary.max_bin_abs_bias == pytest.approx(
        max(abs(b.bias) for b in table.bins)
    )


# =============================================================================
# NamedTuple field-name contract (downstream probes depend on these)
# =============================================================================


# =============================================================================
# cross_fit_isotonic / fit_isotonic — leak-free recalibration contract
# =============================================================================


def _miscalibrated_winners(n: int = 20000, scale: float = 0.4, seed: int = 7):
    """Monotone *under-prediction*: raw = scale * p_true (60% under at scale=0.4).

    Latent rate p_true = 0.02 + 0.3*s with s~U(0,1); labels y~Bernoulli(p_true).
    raw is a strictly-monotone transform of p_true, so isotonic can recover the
    empirical rate and pull mean(raw) (=scale*mean) back toward mean(y).
    """
    rng = np.random.default_rng(seed)
    s = rng.random(n)
    p_true = 0.02 + 0.30 * s
    y = (rng.random(n) < p_true).astype(np.float64)
    raw = scale * p_true
    return raw, y


def test_cross_fit_isotonic_shape_and_determinism():
    """Returns input shape; identical given the same seed; empty -> empty."""
    raw, y = _miscalibrated_winners()
    r1 = cross_fit_isotonic(raw, y, n_folds=5, seed=0)
    r2 = cross_fit_isotonic(raw, y, n_folds=5, seed=0)
    assert r1.shape == raw.shape
    assert np.array_equal(r1, r2)  # deterministic

    empty = cross_fit_isotonic(np.array([]), np.array([]), n_folds=5)
    assert empty.shape == (0,)


def test_cross_fit_isotonic_reduces_under_prediction_ieb():
    """Cross-fitted isotonic removes the monotone under-prediction (IEB ~0.6 -> ~0)."""
    raw, y = _miscalibrated_winners(scale=0.4)
    ieb_before = compute_ieb(y, raw)
    recal = cross_fit_isotonic(raw, y, n_folds=5, seed=0)
    ieb_after = compute_ieb(y, recal)

    assert ieb_before > 0.5  # ~0.6 by construction (40% of true on average)
    assert ieb_after < 0.05  # recalibrated mean matches empirical rate
    assert ieb_after < ieb_before


def test_cross_fit_isotonic_is_rank_preserving():
    """Isotonic is monotone -> AUC essentially unchanged (only calibration moves)."""
    raw, y = _miscalibrated_winners()
    auc_before = _numpy_roc_auc(y, raw)
    recal = cross_fit_isotonic(raw, y, n_folds=5, seed=0)
    auc_after = _numpy_roc_auc(y, recal)
    # Step-function ties at fold boundaries can perturb AUC slightly, not materially.
    assert abs(auc_after - auc_before) < 0.02


def test_cross_fit_isotonic_no_leakage_each_score_held_out():
    """A constant-signal set: out-of-fold recal cannot exceed the global rate band.

    With raw carrying NO signal (constant), isotonic on any training fold returns
    ~the training-fold base rate, so every held-out score maps near the global
    mean — confirming the map is applied out-of-fold, not fit on the same rows.
    """
    rng = np.random.default_rng(1)
    n = 5000
    raw = np.full(n, 0.4)  # constant -> no ranking signal
    y = (rng.random(n) < 0.1).astype(np.float64)  # ~10% base rate
    recal = cross_fit_isotonic(raw, y, n_folds=5, seed=0)
    # All held-out scores collapse to ~the base rate (no over-fit to own labels).
    assert recal.mean() == pytest.approx(y.mean(), abs=0.01)
    assert recal.std() < 0.02


def test_fit_isotonic_monotone_map():
    """fit_isotonic returns a monotone non-decreasing recalibration map."""
    from sklearn.isotonic import IsotonicRegression

    raw, y = _miscalibrated_winners()
    iso = fit_isotonic(raw, y)
    assert isinstance(iso, IsotonicRegression)
    grid = np.linspace(raw.min(), raw.max(), 50)
    mapped = iso.transform(grid)
    assert np.all(np.diff(mapped) >= -1e-9)  # non-decreasing


# =============================================================================
# segment_cross_fit_isotonic — per-segment calibration the global map can't do
# =============================================================================


def _segmented_offset_bias(seed=11):
    """Two segments with the SAME pred distribution but OPPOSITE calibration.

    pred is drawn from the same range for both segments, but the true rate is a
    different multiple of pred per segment: seg A true = 1.5*pred (A
    under-predicts), seg B true = 0.5*pred (B over-predicts). Because the pred
    ranges overlap, a single GLOBAL isotonic map (which sees only pooled
    rate(pred) = pred) cannot separate them — it leaves a large per-segment
    residual. A per-segment map recovers 1.5*pred / 0.5*pred and zeroes each.
    """
    rng = np.random.default_rng(seed)
    n = 24000
    seg = np.where(np.arange(n) < n // 2, "A", "B")
    pred = rng.uniform(0.02, 0.25, n)               # same distribution both segments
    rate = np.where(seg == "A", 1.5 * pred, 0.5 * pred)
    y = (rng.random(n) < rate).astype(np.float64)
    return pred, y, seg


def _seg_bias(pred, y, seg, value):
    m = seg == value
    return abs(pred[m].mean() - y[m].mean()) / max(y[m].mean(), 1e-8)


def test_segment_isotonic_zeroes_each_segment_bias_global_cannot():
    """Per-segment maps zero each segment's mean bias; the global map cannot."""
    pred, y, seg = _segmented_offset_bias()

    glob = cross_fit_isotonic(pred, y, n_folds=5, seed=0)
    segr = segment_cross_fit_isotonic(pred, y, seg, n_folds=5, seed=0)

    # Global map leaves a per-segment residual on at least one segment...
    glob_resid = max(_seg_bias(glob, y, seg, "A"), _seg_bias(glob, y, seg, "B"))
    seg_resid = max(_seg_bias(segr, y, seg, "A"), _seg_bias(segr, y, seg, "B"))
    assert glob_resid > 0.10          # global can't fix opposite-direction offsets
    assert seg_resid < 0.03           # per-segment drives each segment's bias ~0
    assert seg_resid < glob_resid


def test_segment_isotonic_falls_back_on_tiny_segments():
    """A segment below min_positives uses the global map exactly on its rows."""
    pred, y, seg = _segmented_offset_bias()
    # carve out a tiny third segment with almost no positives
    seg = seg.copy()
    tiny = np.arange(20)
    seg[tiny] = "T"
    y = y.copy(); y[tiny] = 0.0  # 0 positives in T

    glob = cross_fit_isotonic(pred, y, n_folds=5, seed=0)
    segr = segment_cross_fit_isotonic(pred, y, seg, n_folds=5, seed=0, min_positives=50)
    # tiny segment T (0 positives < 50) must equal the global map on its rows
    assert np.allclose(segr[seg == "T"], glob[seg == "T"])


def test_segment_isotonic_deterministic_and_shape():
    pred, y, seg = _segmented_offset_bias()
    r1 = segment_cross_fit_isotonic(pred, y, seg, n_folds=5, seed=3)
    r2 = segment_cross_fit_isotonic(pred, y, seg, n_folds=5, seed=3)
    assert r1.shape == pred.shape
    assert np.array_equal(r1, r2)


def test_segment_isotonic_within_segment_rank_preserved():
    """Within a segment the map is monotone -> within-segment AUC preserved."""
    pred, y, seg = _segmented_offset_bias()
    segr = segment_cross_fit_isotonic(pred, y, seg, n_folds=5, seed=0)
    for v in ("A", "B"):
        m = seg == v
        # only meaningful if both classes present
        if 0 < y[m].sum() < m.sum():
            auc_before = _numpy_roc_auc(y[m], pred[m])
            auc_after = _numpy_roc_auc(y[m], segr[m])
            assert abs(auc_after - auc_before) < 0.02


def test_namedtuple_field_contracts():
    """Lock the public NamedTuple field names downstream probes will call."""
    assert ReliabilityBin._fields == (
        "bin_index",
        "mean_pred",
        "mean_true",
        "count",
        "bias",
    )
    assert ReliabilityTable._fields == (
        "bins",
        "quantile_ece",
        "n_bins",
        "n_samples",
    )
    assert SliceCalibrationRow._fields == (
        "slice_name",
        "slice_value",
        "mean_pred",
        "mean_true",
        "bias",
        "count",
        "ece",
    )
    assert SliceCalibrationResult._fields == (
        "slice_name",
        "rows",
        "max_abs_bias",
        "weighted_abs_bias",
    )
