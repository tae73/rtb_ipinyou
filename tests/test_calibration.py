"""Calibration regression test — pins the headline research claims.

This test re-derives the recorded ECE / IEB calibration numbers directly from
an already-present `*_test_predictions.npz` artifact and asserts they match the
values written into the matching `*_result.json`. It locks down the exact
masking / target / prediction wiring behind the project's headline calibration
claims so a refactor of the metric code (or the prediction-export code) cannot
silently change them.

Recompute wiring (verified against the recorded JSON, ~1e-7 relative agreement;
the residual is float32 npz storage vs float64 recompute):

  CTR-biased (won_only):  mask = y_win == 1
                          target = y_click[mask], pred = p_ctr[mask]
      compute_ece -> result["test_ctr_biased_ece"]
      compute_ieb -> result["test_ctr_ieb"]

  WCTR (all_bids):        no mask
                          target = y_click, pred = p_click_bid
      compute_ece -> result["test_wctr_ece"]
      compute_ieb -> result["test_wctr_ieb"]

`compute_ece` / `compute_ieb` default `n_bins=10`, which is how the JSON was
produced — so the recomputation is a faithful regression check, not merely an
internal-consistency one.

Performance: each npz is ~19.4M rows x 5 arrays (~114 MB on disk). We
`mmap_mode='r'` the file and upcast only the per-context slices we need to
float64. We exercise a SINGLE pair (esmmwc) to stay comfortably under the time
budget; the other two pairs are listed for documentation/availability and a
lightweight existence check.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.metrics.evaluation import compute_ece, compute_ieb

# Project root = parent of the tests/ directory holding this file.
_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "models"

# (npz stem, json stem) pairs that have a real prediction artifact.
_PAIRS = {
    "esmmwc": ("esmmwc_test_predictions.npz", "esmmwc_result.json"),
    "escm2wc_dr": ("escm2wc_dr_test_predictions.npz", "escm2wc_dr_result.json"),
    "escm2wc_dr_extps": (
        "escm2wc_dr_extps_test_predictions.npz",
        "escm2wc_dr_extps_result.json",
    ),
}

# Tolerances: the JSON stored float64 metrics computed from float32 arrays;
# our recompute casts the same float32 arrays to float64. Agreement is ~1e-7
# relative, so a slightly looser rtol with a small atol is robust.
_RTOL = 1e-4
_ATOL = 1e-7


def _load_json(json_name: str) -> dict:
    with open(_RESULTS_DIR / json_name) as f:
        return json.load(f)


def _recompute_from_npz(npz_name: str) -> dict:
    """Recompute the four calibration metrics from a prediction npz.

    Uses mmap + per-slice float64 upcast to keep peak memory modest.
    """
    npz_path = _RESULTS_DIR / npz_name
    with np.load(npz_path, mmap_mode="r") as d:
        y_win = np.asarray(d["y_win"])          # int8 0/1
        y_click = np.asarray(d["y_click"])      # int8 0/1
        p_ctr = np.asarray(d["p_ctr"])          # float32 P(click|win)
        p_click_bid = np.asarray(d["p_click_bid"])  # float32 P(click|bid)

        # --- CTR-biased (won_only): restrict to winners. ---
        won = y_win == 1
        ctr_true = y_click[won].astype(np.float64)
        ctr_pred = p_ctr[won].astype(np.float64)
        ctr_biased_ece = compute_ece(ctr_true, ctr_pred, n_bins=10)
        ctr_ieb = compute_ieb(ctr_true, ctr_pred)

        # --- WCTR (all_bids): no mask. ---
        wctr_true = y_click.astype(np.float64)
        wctr_pred = p_click_bid.astype(np.float64)
        wctr_ece = compute_ece(wctr_true, wctr_pred, n_bins=10)
        wctr_ieb = compute_ieb(wctr_true, wctr_pred)

    return {
        "test_ctr_biased_ece": ctr_biased_ece,
        "test_ctr_ieb": ctr_ieb,
        "test_wctr_ece": wctr_ece,
        "test_wctr_ieb": wctr_ieb,
    }


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"required artifact missing: {path}")


# =============================================================================
# Primary regression: recompute esmmwc calibration and pin to recorded JSON.
# =============================================================================


def test_esmmwc_calibration_matches_recorded_json():
    """Recomputed ECE/IEB from the esmmwc npz match esmmwc_result.json.

    This is the regression that pins the headline calibration claims: it
    re-derives test_ctr_biased_ece, test_ctr_ieb, test_wctr_ece, test_wctr_ieb
    from raw predictions and asserts equality with the recorded values.
    """
    npz_name, json_name = _PAIRS["esmmwc"]
    _require(_RESULTS_DIR / npz_name)
    _require(_RESULTS_DIR / json_name)

    recorded = _load_json(json_name)
    recomputed = _recompute_from_npz(npz_name)

    for key, recomputed_value in recomputed.items():
        assert key in recorded, f"{json_name} missing key {key}"
        assert recomputed_value == pytest.approx(
            recorded[key], rel=_RTOL, abs=_ATOL
        ), (
            f"{json_name}[{key}] = {recorded[key]!r} but recompute = "
            f"{recomputed_value!r} (float32->float64 rounding should be ~1e-7)"
        )


# =============================================================================
# Internal-consistency cross-checks (cheap, single npz already in memory path).
# =============================================================================


def test_ctr_biased_uses_won_only_mask():
    """The won-only mask is load-bearing: dropping it changes ECE materially.

    Documents WHY the regression test masks on y_win == 1. The CTR head is
    trained/evaluated on winners; computing its ECE over *all* bids (wrong
    denominator) yields a different number, proving the mask matters.
    """
    npz_name, json_name = _PAIRS["esmmwc"]
    _require(_RESULTS_DIR / npz_name)
    _require(_RESULTS_DIR / json_name)

    recorded = _load_json(json_name)
    with np.load(_RESULTS_DIR / npz_name, mmap_mode="r") as d:
        y_win = np.asarray(d["y_win"])
        y_click = np.asarray(d["y_click"])
        p_ctr = np.asarray(d["p_ctr"])

        won = y_win == 1
        masked_ece = compute_ece(
            y_click[won].astype(np.float64), p_ctr[won].astype(np.float64)
        )
        unmasked_ece = compute_ece(
            y_click.astype(np.float64), p_ctr.astype(np.float64)
        )

    # Masked recompute pins the JSON; the unmasked variant must differ.
    assert masked_ece == pytest.approx(
        recorded["test_ctr_biased_ece"], rel=_RTOL, abs=_ATOL
    )
    assert not np.isclose(masked_ece, unmasked_ece, rtol=_RTOL, atol=_ATOL)


def test_ieb_internal_consistency_against_recorded_means():
    """IEB is itself a pure function of two means -> re-derive it standalone.

    Independently of compute_ieb, IEB := |mean(pred) - mean(true)| /
    max(mean(true), 1e-8). We recompute the constituent means from the npz and
    confirm the manual formula reproduces the recorded test_wctr_ieb. This is
    an internal-consistency guard that does not depend on the compute_ieb impl.
    """
    npz_name, json_name = _PAIRS["esmmwc"]
    _require(_RESULTS_DIR / npz_name)
    _require(_RESULTS_DIR / json_name)

    recorded = _load_json(json_name)
    with np.load(_RESULTS_DIR / npz_name, mmap_mode="r") as d:
        mean_true = float(np.asarray(d["y_click"], dtype=np.float64).mean())
        mean_pred = float(np.asarray(d["p_click_bid"], dtype=np.float64).mean())

    manual_ieb = abs(mean_pred - mean_true) / max(mean_true, 1e-8)
    assert manual_ieb == pytest.approx(
        recorded["test_wctr_ieb"], rel=_RTOL, abs=_ATOL
    )


# =============================================================================
# Availability of the remaining documented pairs (cheap header-only check).
# =============================================================================


@pytest.mark.parametrize("name", ["escm2wc_dr", "escm2wc_dr_extps"])
def test_other_prediction_pairs_present_and_keyed(name):
    """The other two npz+json pairs exist and carry the expected JSON keys.

    Kept lightweight (no full-array recompute) so the suite stays fast; the
    esmmwc test above already exercises the full recompute path.
    """
    npz_name, json_name = _PAIRS[name]
    _require(_RESULTS_DIR / npz_name)
    _require(_RESULTS_DIR / json_name)

    recorded = _load_json(json_name)
    for key in (
        "test_ctr_biased_ece",
        "test_ctr_ieb",
        "test_wctr_ece",
        "test_wctr_ieb",
    ):
        assert key in recorded, f"{json_name} missing {key}"
        assert isinstance(recorded[key], (int, float))
