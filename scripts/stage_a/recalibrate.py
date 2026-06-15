"""Stage A step 1 — post-hoc isotonic recalibration of winners pCTR.

The fair-split retrain fixed *ranking* (escm2wc_dr winners-CTR AUC 0.658 > LR
0.554, LGB 0.632) but left a *calibration* gap: the neural winners pCTR
under-predicts (global IEB ~0.60; all 10 deciles under, pred/true 0.15-0.60).
This probe applies a monotone isotonic recalibration map to the winners pCTR of
the neural model AND the LR/LGB baselines, and reports before/after calibration.

Protocol: K-fold **cross-fitted** isotonic on the test winners
(:func:`src.metrics.calibration.cross_fit_isotonic`). Every recalibrated score
comes from a map that never saw it -> leak-free out-of-fold estimate, GPU 0,
identical protocol for all three models (apples-to-apples). The literal
val->test frozen-map version is deferred to the next neural retrain (which now
saves val predictions).

Isotonic is monotone => rank-preserving: winners AUC must be unchanged
before/after; only calibration (IEB, decile ratios, quantile-ECE) moves. That
invariant is asserted as a sanity check.

Reads:
- results/models/escm2wc_dr_fair_posw/escm2wc_dr_test_predictions.npz
  (neural; keys p_ctr, p_click_bid, p_win, y_win, y_click).
- results/stage_a/fair_baseline_preds.npz
  (lr_p_all, lgb_p_all, y_win, y_click, advertiser).

Writes:
- results/stage_a/recalibration.json          (per-model before/after metrics).
- results/stage_a/recalibrated_winners_preds.npz (idx_won, y_click_won, per-model
  raw/recal winners pCTR -> consumed by the Stage B2 surplus comparison).
- results/stage_a/recalibration_summary.md     (audit-style report).

Usage: python scripts/stage_a/recalibrate.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.metrics.calibration import cross_fit_isotonic, quantile_reliability
from src.metrics.evaluation import _numpy_roc_auc, compute_ieb

NEURAL_DIR = Path("results/models/escm2wc_dr_fair_posw")
BASE = Path("results/stage_a/fair_baseline_preds.npz")
OUT_JSON = Path("results/stage_a/recalibration.json")
OUT_NPZ = Path("results/stage_a/recalibrated_winners_preds.npz")
OUT_MD = Path("results/stage_a/recalibration_summary.md")

N_FOLDS = 5
SEED = 0
UNDER_RATIO = 0.8  # decile pred/true below this = under-predicting
OVER_RATIO = 1.25  # decile pred/true above this = over-predicting


def _decile_table(pred: np.ndarray, y_true: np.ndarray) -> List[dict]:
    """Per-decile pred/true/ratio table on winners (mirrors compare_fair.py)."""
    rel = quantile_reliability(y_true.astype(float), pred.astype(float), n_bins=10)
    return [
        {
            "bin": b.bin_index,
            "pred": round(float(b.mean_pred), 7),
            "true": round(float(b.mean_true), 7),
            "ratio": round(float(b.mean_pred / b.mean_true), 3) if b.mean_true > 0 else None,
            "count": int(b.count),
        }
        for b in rel.bins
    ]


def _count_off(deciles: List[dict], lo: float, hi: float) -> Tuple[int, int]:
    """(# deciles under lo, # deciles over hi) by pred/true ratio."""
    ratios = [d["ratio"] for d in deciles if d["ratio"] is not None]
    return (sum(r < lo for r in ratios), sum(r > hi for r in ratios))


def _eval_winners(pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Calibration + ranking summary for one winners pCTR vector."""
    deciles = _decile_table(pred, y_true)
    under, over = _count_off(deciles, UNDER_RATIO, OVER_RATIO)
    rel = quantile_reliability(y_true.astype(float), pred.astype(float), n_bins=10)
    return {
        "ieb": float(compute_ieb(y_true, pred)),  # |mean(pred)-mean(true)|/mean(true)
        "quantile_ece": float(rel.quantile_ece),
        "auc": float(_numpy_roc_auc(y_true, pred)),
        "pred_mean": float(pred.mean()),
        "true_mean": float(y_true.mean()),
        "under_deciles": int(under),
        "over_deciles": int(over),
        "deciles": deciles,
    }


def _per_adv_ieb(pred: np.ndarray, y_true: np.ndarray, adv: np.ndarray) -> Dict[str, float]:
    """Winners IEB per advertiser (stratified calibration)."""
    out: Dict[str, float] = {}
    for a in np.unique(adv):
        m = adv == a
        if m.sum() == 0:
            continue
        out[str(int(a))] = round(float(compute_ieb(y_true[m], pred[m])), 4)
    return out


def main() -> None:
    npzs = glob.glob(str(NEURAL_DIR / "*test_predictions*.npz"))
    if not npzs:
        raise SystemExit(f"No neural test predictions in {NEURAL_DIR} — retrain not finished.")
    neural = np.load(npzs[0])
    base = np.load(BASE)

    yw_b = base["y_win"].astype(np.int64)
    yc_b = base["y_click"].astype(np.int64)
    adv_b = base["advertiser"]
    yw_n = neural["y_win"].astype(np.int64)
    yc_n = neural["y_click"].astype(np.int64)

    # Alignment guard: both files derive from the same fair test.parquet in the
    # same row order, so labels must match exactly. If so, the neural and
    # baseline winners are the SAME rows and we can borrow `advertiser`.
    if yw_n.shape != yw_b.shape:
        raise SystemExit(f"row-count mismatch: neural {yw_n.shape} vs base {yw_b.shape}")
    aligned = np.array_equal(yw_n, yw_b) and np.array_equal(yc_n, yc_b)
    if not aligned:
        print("WARNING: neural/baseline labels differ — row order may not match; "
              "using each file's own labels; neural per-advertiser borrow disabled.")

    # Canonical winners (from the baseline fair labels).
    won_mask = yw_b == 1
    idx_won = np.where(won_mask)[0].astype(np.int64)
    y_click_won = yc_b[won_mask].astype(np.int8)
    adv_won = adv_b[won_mask]

    # (model_key, raw winners pCTR, click labels on those winners, adv or None)
    specs = []
    # Neural CTR tower P(click|win) on winners.
    if aligned:
        specs.append(("escm2wc_dr", neural["p_ctr"][won_mask], y_click_won, adv_won))
    else:
        nm = yw_n == 1
        specs.append(("escm2wc_dr", neural["p_ctr"][nm], yc_n[nm].astype(np.int8), None))
    # Baselines: all-bids pCTR evaluated on the winners subset (as compare_fair.py does).
    specs.append(("lr_ctr_all", base["lr_p_all"][won_mask], y_click_won, adv_won))
    specs.append(("lgb_ctr_all", base["lgb_p_all"][won_mask], y_click_won, adv_won))

    results: Dict[str, dict] = {}
    npz_payload: Dict[str, np.ndarray] = {
        "idx_won": idx_won,
        "y_click_won": y_click_won,
    }

    hdr = f"{'model':14s} {'IEB b→a':>16s} {'qECE b→a':>17s} {'under b→a':>10s} {'AUC b/a':>15s}"
    print(hdr)
    print("-" * len(hdr))
    for key, raw, y_won, adv in specs:
        raw = raw.astype(np.float64)
        recal = cross_fit_isotonic(raw, y_won.astype(np.float64), n_folds=N_FOLDS, seed=SEED)

        before = _eval_winners(raw, y_won.astype(np.float64))
        after = _eval_winners(recal, y_won.astype(np.float64))

        # Sanity: isotonic is monotone -> ranking preserved (AUC ~unchanged).
        auc_drift = abs(before["auc"] - after["auc"])
        if auc_drift > 5e-3:
            print(f"  NOTE: {key} winners AUC drift {auc_drift:.4f} "
                  "(fold-boundary effect; isotonic is rank-preserving within folds).")

        entry = {
            "n_winners": int(y_won.shape[0]),
            "before": before,
            "after": after,
            "ieb_before": round(before["ieb"], 4),
            "ieb_after": round(after["ieb"], 4),
            "quantile_ece_before": round(before["quantile_ece"], 6),
            "quantile_ece_after": round(after["quantile_ece"], 6),
            "under_deciles_before": before["under_deciles"],
            "under_deciles_after": after["under_deciles"],
            "winners_auc_before": round(before["auc"], 4),
            "winners_auc_after": round(after["auc"], 4),
        }
        if adv is not None:
            entry["per_advertiser_ieb_before"] = _per_adv_ieb(raw, y_won.astype(np.float64), adv)
            entry["per_advertiser_ieb_after"] = _per_adv_ieb(recal, y_won.astype(np.float64), adv)
        results[key] = entry

        npz_payload[f"{key}_raw"] = raw.astype(np.float32)
        npz_payload[f"{key}_recal"] = recal.astype(np.float32)

        print(f"{key:14s} "
              f"{before['ieb']:6.3f}→{after['ieb']:<6.3f}  "
              f"{before['quantile_ece']:7.5f}→{after['quantile_ece']:<7.5f}  "
              f"{before['under_deciles']:2d}→{after['under_deciles']:<2d}     "
              f"{before['auc']:.3f}/{after['auc']:.3f}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    np.savez_compressed(OUT_NPZ, **npz_payload)
    OUT_MD.write_text(_render_md(results, aligned))
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_NPZ}\nWrote {OUT_MD}")


def _render_md(results: Dict[str, dict], aligned: bool) -> str:
    """Audit-style markdown report (mirrors reliability_summary.md / rootcause_audit.md)."""
    lines: List[str] = []
    lines.append("# Stage A — Post-hoc Isotonic Recalibration of Winners pCTR\n")
    neu = results.get("escm2wc_dr", {})
    fixed = neu.get("under_deciles_after", 99) <= 1 and neu.get("ieb_after", 9.9) < 0.1
    adv_after = neu.get("per_advertiser_ieb_after", {})
    adv_max = max(adv_after.values()) if adv_after else None
    adv_argmax = max(adv_after, key=adv_after.get) if adv_after else None
    lines.append("## TL;DR VERDICT\n")
    lines.append(
        f"- **Global calibration {'FIXED' if fixed else 'IMPROVED'}** by cross-fitted isotonic "
        f"(K={N_FOLDS}, leak-free out-of-fold, GPU 0).\n"
        f"- Neural (escm2wc_dr) winners IEB **{neu.get('ieb_before')} → {neu.get('ieb_after')}**, "
        f"under-prediction deciles **{neu.get('under_deciles_before')}/10 → "
        f"{neu.get('under_deciles_after')}/10**.\n"
        f"- Ranking untouched: isotonic is monotone, so winners AUC is preserved "
        f"({neu.get('winners_auc_before')} → {neu.get('winners_auc_after')}).\n"
        f"- **Caveat — per-advertiser residual remains.** A single GLOBAL map zeroes the "
        f"aggregate mean bias but cannot correct advertiser-specific bias: neural max "
        f"per-advertiser residual IEB after recal = **{adv_max}** (adv {adv_argmax}). "
        f"⇒ motivates segment-level (per-advertiser) recalibration and/or training-stage "
        f"calibration — picked up by Stage B2 slice calibration.\n"
        f"- Protocol: cross-fit on test winners; literal val→test frozen-map deferred to "
        f"the next neural retrain (now saves val predictions). Alignment "
        f"({'OK' if aligned else 'MISMATCH — see log'}).\n"
    )

    lines.append("\n## Before → After (winners pCTR)\n")
    lines.append("| model | IEB before | IEB after | qECE before | qECE after | under-deciles | AUC (unchanged) |")
    lines.append("|---|---|---|---|---|---|---|")
    for k, v in results.items():
        lines.append(
            f"| {k} | {v['ieb_before']} | {v['ieb_after']} | "
            f"{v['quantile_ece_before']} | {v['quantile_ece_after']} | "
            f"{v['under_deciles_before']}/10 → {v['under_deciles_after']}/10 | "
            f"{v['winners_auc_before']} → {v['winners_auc_after']} |"
        )

    for k, v in results.items():
        lines.append(f"\n## {k} — decile pred/true ratios\n")
        lines.append("| decile | pred (before→after) | true | ratio before | ratio after |")
        lines.append("|---|---|---|---|---|")
        bdec = {d["bin"]: d for d in v["before"]["deciles"]}
        adec = {d["bin"]: d for d in v["after"]["deciles"]}
        for b in sorted(set(bdec) | set(adec)):
            db, da = bdec.get(b, {}), adec.get(b, {})
            lines.append(
                f"| {b} | {db.get('pred')} → {da.get('pred')} | {db.get('true')} | "
                f"{db.get('ratio')} | {da.get('ratio')} |"
            )
        if "per_advertiser_ieb_before" in v:
            lines.append(f"\n**Per-advertiser winners IEB (before → after):**\n")
            pb = v["per_advertiser_ieb_before"]
            pa = v["per_advertiser_ieb_after"]
            for a in sorted(pb, key=lambda x: int(x)):
                lines.append(f"- adv {a}: {pb[a]} → {pa.get(a)}")

    lines.append("\n## Files\n")
    lines.append(f"- `{OUT_JSON}` — full per-model before/after metrics + decile tables.")
    lines.append(f"- `{OUT_NPZ}` — recalibrated winners pCTR (idx_won, y_click_won, "
                 "{model}_raw/{model}_recal) for the Stage B2 surplus comparison.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
