"""Calibration measurement upgrades for rare-event RTB prediction.

The legacy metrics in :mod:`src.metrics.evaluation` are near-degenerate at the
~0.02% click base rate:

  - ``compute_ece`` uses 10 equal-WIDTH bins over ``[0, 1)``; at this base rate
    essentially every prediction lands in the first bin, so the reliability
    diagram collapses to a single point and ECE under-reports tail
    miscalibration.
  - ``compute_ieb`` is a single global mean-bias ratio with no resolution over
    the predicted-probability range.

This module adds EQUAL-FREQUENCY (quantile) calibration tooling that keeps
each reliability bin populated even when the score distribution is extremely
skewed, plus a per-slice calibration decomposition following the
``compute_subgroup_bias`` pattern in :mod:`src.debiasing.diagnostics`.

Pure NumPy (+ pandas only for the grouped quantile binning, matching the
project's existing ``pd.qcut(..., duplicates='drop')`` convention). The legacy
``compute_ece`` is imported and reused, never reimplemented, so equal-width ECE
remains available unchanged.
"""

from typing import List, NamedTuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.metrics.evaluation import compute_ece

__all__ = [
    "ReliabilityBin",
    "ReliabilityTable",
    "SliceCalibrationRow",
    "SliceCalibrationResult",
    "safe_quantile_bins",
    "quantile_reliability",
    "slice_calibration",
    "reliability_summary",
    "fit_isotonic",
    "cross_fit_isotonic",
    "segment_cross_fit_isotonic",
]


# =============================================================================
# Result Types
# =============================================================================


class ReliabilityBin(NamedTuple):
    """A single equal-frequency reliability bin."""

    bin_index: int
    mean_pred: float  # mean predicted probability in the bin
    mean_true: float  # empirical event rate in the bin
    count: int  # number of samples in the bin
    bias: float  # mean_pred - mean_true (signed; >0 = over-prediction)


class ReliabilityTable(NamedTuple):
    """Equal-frequency reliability table + count-weighted quantile-ECE."""

    bins: List[ReliabilityBin]
    quantile_ece: float  # Σ count_b |mean_pred_b - mean_true_b| / Σ count_b
    n_bins: int  # number of NON-EMPTY bins actually realised
    n_samples: int  # total scored samples (NaNs dropped)


class SliceCalibrationRow(NamedTuple):
    """Calibration summary for one level of a categorical slice."""

    slice_name: str  # e.g. "adexchange"
    slice_value: str  # the categorical level (stringified)
    mean_pred: float  # mean predicted probability in the slice
    mean_true: float  # empirical event rate in the slice
    bias: float  # mean_pred - mean_true (signed)
    count: int  # number of samples in the slice
    ece: float  # within-slice equal-width ECE (legacy compute_ece)


class SliceCalibrationResult(NamedTuple):
    """Per-slice calibration decomposition for a categorical slice."""

    slice_name: str
    rows: List[SliceCalibrationRow]
    max_abs_bias: float  # max |bias| over slices (count-eligible)
    weighted_abs_bias: float  # count-weighted mean |bias| over slices


# =============================================================================
# Quantile binning (safe fallback)
# =============================================================================


def safe_quantile_bins(y_prob: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Assign equal-frequency (quantile) bin labels, robust to ties.

    Mirrors the project's ``pd.qcut(..., duplicates='drop')`` convention but
    degrades gracefully: when the score distribution is so concentrated that
    fewer than two distinct quantile edges survive (e.g. an almost-constant
    score vector), every sample is placed in a single bin (label 0) rather
    than raising.

    Args:
        y_prob: Predicted probabilities, shape ``(n,)``.
        n_bins: Requested number of equal-frequency bins.

    Returns:
        Integer bin labels in ``[0, k)`` where ``k <= n_bins`` is the number of
        realised (non-degenerate) bins. Same length as ``y_prob``. Bin labels
        are monotonically increasing in the score (label 0 = lowest scores).
    """
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = y_prob.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.intp)

    try:
        labels = pd.qcut(y_prob, q=n_bins, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        # Not enough distinct values to form even one quantile edge.
        return np.zeros(n, dtype=np.intp)

    labels = np.asarray(labels)
    # pd.qcut returns NaN for values outside the (deduplicated) edge range only
    # in pathological cases; map any residual NaN to the lowest bin so every
    # sample is accounted for in the count-weighted aggregation.
    if np.isnan(labels).any():
        labels = np.where(np.isnan(labels), 0.0, labels)
    return labels.astype(np.intp)


# =============================================================================
# Equal-frequency reliability table
# =============================================================================


def quantile_reliability(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> ReliabilityTable:
    """Reliability table over EQUAL-FREQUENCY (quantile) bins.

    Unlike equal-WIDTH ECE (``compute_ece``), every bin holds ~the same number
    of samples, so reliability is resolved even when scores are extremely
    concentrated near 0 (rare-event regime). The scalar ``quantile_ece`` is the
    count-weighted mean absolute per-bin gap.

    Args:
        y_true: Ground-truth binary labels, shape ``(n,)``.
        y_prob: Predicted probabilities, shape ``(n,)``.
        n_bins: Requested number of equal-frequency bins.

    Returns:
        :class:`ReliabilityTable` with one :class:`ReliabilityBin` per realised
        (non-empty) bin and the count-weighted quantile-ECE scalar.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # Drop NaNs (mirrors compute_metrics masking) before binning.
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    n_samples = int(y_true.shape[0])

    if n_samples == 0:
        return ReliabilityTable(bins=[], quantile_ece=0.0, n_bins=0, n_samples=0)

    labels = safe_quantile_bins(y_prob, n_bins=n_bins)

    bins: List[ReliabilityBin] = []
    weighted_abs = 0.0
    # Iterate realised bins in ascending score order for a readable diagram.
    for out_idx, lab in enumerate(np.unique(labels)):
        in_bin = labels == lab
        count = int(in_bin.sum())
        if count == 0:
            continue
        mean_pred = float(y_prob[in_bin].mean())
        mean_true = float(y_true[in_bin].mean())
        bias = mean_pred - mean_true
        bins.append(
            ReliabilityBin(
                bin_index=out_idx,
                mean_pred=mean_pred,
                mean_true=mean_true,
                count=count,
                bias=bias,
            )
        )
        weighted_abs += count * abs(bias)

    quantile_ece = float(weighted_abs / n_samples)

    return ReliabilityTable(
        bins=bins,
        quantile_ece=quantile_ece,
        n_bins=len(bins),
        n_samples=n_samples,
    )


# =============================================================================
# Per-slice calibration
# =============================================================================


def slice_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    slice_values: np.ndarray,
    slice_name: str,
    n_ece_bins: int = 10,
    min_count: int = 1,
) -> SliceCalibrationResult:
    """Per-slice calibration decomposition for a categorical slice.

    Follows the ``compute_subgroup_bias`` pattern
    (``src/debiasing/diagnostics.py``): group by a categorical column and report
    per-level statistics. Here the statistics are calibration-oriented
    (predicted vs. empirical rate, signed bias, within-slice equal-width ECE)
    rather than naive-vs-IPW CTR.

    Args:
        y_true: Ground-truth binary labels, shape ``(n,)``.
        y_prob: Predicted probabilities, shape ``(n,)``.
        slice_values: Categorical slice key per row (e.g. ``adexchange``),
            shape ``(n,)``. Stringified for grouping/reporting.
        slice_name: Human-readable name of the slice column.
        n_ece_bins: Bins for the within-slice legacy equal-width ``compute_ece``.
        min_count: Minimum samples in a slice level to be reported.

    Returns:
        :class:`SliceCalibrationResult` with one :class:`SliceCalibrationRow`
        per (count-eligible) slice level, plus max / count-weighted absolute
        bias aggregates over the reported levels.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    slice_values = np.asarray(slice_values)

    frame = pd.DataFrame(
        {
            "_slice": slice_values.astype(str),
            "_y_true": y_true,
            "_y_prob": y_prob,
        }
    )

    rows: List[SliceCalibrationRow] = []
    for value, grp in frame.groupby("_slice", sort=True):
        count = int(len(grp))
        if count < min_count:
            continue
        gt = grp["_y_true"].to_numpy()
        gp = grp["_y_prob"].to_numpy()
        mean_pred = float(gp.mean())
        mean_true = float(gt.mean())
        bias = mean_pred - mean_true
        ece = float(compute_ece(gt, gp, n_bins=n_ece_bins))
        rows.append(
            SliceCalibrationRow(
                slice_name=slice_name,
                slice_value=str(value),
                mean_pred=mean_pred,
                mean_true=mean_true,
                bias=bias,
                count=count,
                ece=ece,
            )
        )

    if not rows:
        return SliceCalibrationResult(
            slice_name=slice_name,
            rows=[],
            max_abs_bias=0.0,
            weighted_abs_bias=0.0,
        )

    abs_biases = np.array([abs(r.bias) for r in rows], dtype=np.float64)
    counts = np.array([r.count for r in rows], dtype=np.float64)
    max_abs_bias = float(abs_biases.max())
    weighted_abs_bias = float((abs_biases * counts).sum() / counts.sum())

    return SliceCalibrationResult(
        slice_name=slice_name,
        rows=rows,
        max_abs_bias=max_abs_bias,
        weighted_abs_bias=weighted_abs_bias,
    )


# =============================================================================
# Convenience summary
# =============================================================================


class ReliabilitySummary(NamedTuple):
    """Compact scalar summary pairing equal-frequency and equal-width ECE."""

    quantile_ece: float  # equal-frequency ECE (this module)
    equal_width_ece: float  # legacy equal-width ECE (compute_ece)
    n_bins: int  # realised equal-frequency bins
    n_samples: int
    max_bin_abs_bias: float  # largest |mean_pred - mean_true| across q-bins


def reliability_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> ReliabilitySummary:
    """One-call scalar summary contrasting equal-frequency vs equal-width ECE.

    Useful as a single-line probe: a large gap between ``quantile_ece`` and
    ``equal_width_ece`` flags that the legacy equal-width metric is hiding tail
    miscalibration in the rare-event regime.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted probabilities.
        n_bins: Bins for both the quantile table and the legacy ECE.

    Returns:
        :class:`ReliabilitySummary`.
    """
    table = quantile_reliability(y_true, y_prob, n_bins=n_bins)
    eqw_ece = float(compute_ece(np.asarray(y_true), np.asarray(y_prob), n_bins=n_bins))
    max_bin_abs_bias = (
        float(max(abs(b.bias) for b in table.bins)) if table.bins else 0.0
    )
    return ReliabilitySummary(
        quantile_ece=table.quantile_ece,
        equal_width_ece=eqw_ece,
        n_bins=table.n_bins,
        n_samples=table.n_samples,
        max_bin_abs_bias=max_bin_abs_bias,
    )


# =============================================================================
# Post-hoc isotonic recalibration
# =============================================================================


def fit_isotonic(scores: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Fit a monotone isotonic recalibration map ``score -> P(event)``.

    Thin wrapper over the ``IsotonicRegression(out_of_bounds='clip')`` pattern
    already used for win-propensity calibration
    (``src/debiasing/win_propensity.py``). ``out_of_bounds='clip'`` makes the
    fitted map safe to apply to held-out scores outside the training range
    (clamped to the boundary calibrated value).

    The map is monotonically non-decreasing in ``scores`` and therefore
    **rank-preserving**: applying it cannot change AUC, only the predicted
    probability *level* (calibration).

    Args:
        scores: Uncalibrated predicted probabilities, shape ``(n,)``.
        y_true: Ground-truth binary labels, shape ``(n,)``.

    Returns:
        A fitted :class:`sklearn.isotonic.IsotonicRegression`. Apply with
        ``.transform(new_scores)``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, y_true)
    return iso


def cross_fit_isotonic(
    scores: np.ndarray,
    y_true: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Leak-free recalibrated scores via K-fold cross-fitted isotonic regression.

    For each fold, an isotonic map is fit on the *other* folds and applied to
    the held-out fold. Every returned score is therefore recalibrated by a map
    that never saw it, so calibration metrics computed on the result are honest
    out-of-fold estimates — no optimistic bias, no separate held-out set needed.

    Because each per-fold map is monotone (see :func:`fit_isotonic`), ranking is
    preserved *within* each fold; aggregate AUC is unchanged up to fold-boundary
    effects.

    Args:
        scores: Uncalibrated predicted probabilities, shape ``(n,)``.
        y_true: Ground-truth binary labels, shape ``(n,)``.
        n_folds: Number of cross-fitting folds (>= 2). Clamped to ``n`` samples.
        seed: Seed for the deterministic fold permutation (reproducible).

    Returns:
        Recalibrated scores in the original row order, shape ``(n,)``, dtype
        float64. Empty input returns an empty array.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    n = scores.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    k = int(max(2, min(n_folds, n)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_test_idx = np.array_split(perm, k)

    recal = np.empty(n, dtype=np.float64)
    for test_idx in fold_test_idx:
        if test_idx.size == 0:
            continue
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        iso = fit_isotonic(scores[train_mask], y_true[train_mask])
        recal[test_idx] = iso.transform(scores[test_idx])
    return recal


def segment_cross_fit_isotonic(
    scores: np.ndarray,
    y_true: np.ndarray,
    segment_ids: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
    min_positives: int = 50,
) -> np.ndarray:
    """Per-segment cross-fit isotonic recalibration (e.g. per advertiser).

    Fits a SEPARATE leak-free cross-fit isotonic map within each segment, so each
    segment's scores are calibrated to that segment's OWN event rate. This
    targets segment-specific level offsets that a single GLOBAL map cannot
    correct (a global monotone map zeroes only the aggregate mean bias).

    Segments with fewer than ``min_positives`` positive labels fall back to the
    global cross-fit map (avoids overfitting tiny segments).

    **Ranking caveat:** each per-segment map is monotone *within* its segment, so
    within-segment ranking (AUC) is preserved; but the maps differ across
    segments, so the map is NOT globally monotone — global cross-segment ranking
    may change. That is expected: per-segment calibration aligns levels so that
    cross-segment comparisons reflect calibrated probabilities.

    Args:
        scores: Uncalibrated predicted probabilities, shape ``(n,)``.
        y_true: Ground-truth binary labels, shape ``(n,)``.
        segment_ids: Segment key per row (e.g. advertiser id), shape ``(n,)``.
        n_folds: Cross-fitting folds used within each segment.
        seed: RNG seed (per-segment fits use the same seed deterministically).
        min_positives: Minimum positives for a per-segment map; below this the
            segment uses the global cross-fit map.

    Returns:
        Recalibrated scores in original row order, shape ``(n,)``, dtype float64.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    segment_ids = np.asarray(segment_ids)
    n = scores.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Global fallback map (used for under-populated segments).
    global_recal = cross_fit_isotonic(scores, y_true, n_folds=n_folds, seed=seed)

    recal = np.empty(n, dtype=np.float64)
    for seg in np.unique(segment_ids):
        m = segment_ids == seg
        if int(y_true[m].sum()) >= min_positives:
            recal[m] = cross_fit_isotonic(scores[m], y_true[m], n_folds=n_folds, seed=seed)
        else:
            recal[m] = global_recal[m]
    return recal
