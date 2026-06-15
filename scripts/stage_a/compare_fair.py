"""Probe A+C verdict: compare the fair-split retrained ESCM2-WC(DR) (calibration-fixed)
against the fair-split LR/LGB baselines on winners-CTR ranking + calibration.

Answers: (1) does the neural model's winners-CTR AUC converge to the baselines on the
SHARED-vocab fair split (-> the prior negative was an adversarial-split artifact)?
(2) does the calibration fix (ctr_weight=1.0 + ctr_pos_weight) remove the monotone
10/10-decile under-prediction of winners p_ctr?

Reads:
- results/models/escm2wc_dr_fair_posw/escm2wc_dr_test_predictions.npz (neural; keys
  p_win,p_ctr,p_click_bid,y_win,y_click) — written by the retrain.
- results/stage_a/fair_baseline_preds.npz (lr_p_all, lgb_p_all, y_win, y_click, advertiser).

Usage: python scripts/stage_a/compare_fair.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from src.metrics.evaluation import _numpy_roc_auc, compute_ieb
from src.metrics.calibration import quantile_reliability

NEURAL_DIR = Path("results/models/escm2wc_dr_fair_posw")
BASE = Path("results/stage_a/fair_baseline_preds.npz")
OUT = Path("results/stage_a/fair_comparison.json")


def _winners_auc(pctr: np.ndarray, y_click: np.ndarray, y_win: np.ndarray) -> float:
    m = y_win == 1
    return float(_numpy_roc_auc(y_click[m], pctr[m]))


def _decile_ratios(pctr: np.ndarray, y_click: np.ndarray, y_win: np.ndarray):
    m = y_win == 1
    rel = quantile_reliability(y_click[m].astype(float), pctr[m].astype(float), n_bins=10)
    return [
        {"bin": b.bin_index, "pred": round(float(b.mean_pred), 7),
         "true": round(float(b.mean_true), 7),
         "ratio": round(float(b.mean_pred / b.mean_true), 3) if b.mean_true > 0 else None}
        for b in rel.bins
    ], float(rel.quantile_ece)


def main() -> None:
    npzs = glob.glob(str(NEURAL_DIR / "*test_predictions*.npz"))
    if not npzs:
        raise SystemExit(f"No neural test predictions in {NEURAL_DIR} yet — retrain not finished.")
    neural = np.load(npzs[0])
    base = np.load(BASE)

    yw, yc = base["y_win"], base["y_click"]
    # Alignment guard: neural eval order must match the baseline order on the fair test.
    assert neural["y_win"].shape == yw.shape, "row count mismatch"
    if not (neural["y_win"].sum() == yw.sum() and neural["y_click"].sum() == yc.sum()):
        print(f"WARNING: label-sum mismatch neural(win={neural['y_win'].sum()},clk={neural['y_click'].sum()}) "
              f"vs base(win={yw.sum()},clk={yc.sum()}) — order may differ; using each file's own labels.")

    models = {
        "escm2wc_dr_fair (neural, fixed)": (neural["p_ctr"], neural["p_click_bid"], neural["y_win"], neural["y_click"]),
        "lr_ctr_all (fair)": (base["lr_p_all"], base["lr_p_all"], yw, yc),
        "lgb_ctr_all (fair)": (base["lgb_p_all"], base["lgb_p_all"], yw, yc),
    }

    rows = {}
    print(f"{'model':34s} {'winAUC':>7s} {'allAUC':>7s} {'win_qECE':>9s} {'winIEB':>7s}")
    for name, (pctr, pall, w, c) in models.items():
        wauc = _winners_auc(pctr, c, w)
        aauc = float(_numpy_roc_auc(c, pall))
        m = w == 1
        wieb = float(compute_ieb(pctr[m], c[m]))
        ratios, qece = _decile_ratios(pctr, c, w)
        rows[name] = {"winners_auc": wauc, "all_bids_auc": aauc, "winners_quantile_ece": qece,
                      "winners_ieb": wieb, "winners_deciles": ratios}
        print(f"{name:34s} {wauc:7.4f} {aauc:7.4f} {qece:9.5f} {wieb:7.4f}")

    # Verdict signals.
    n = rows["escm2wc_dr_fair (neural, fixed)"]
    lr_w = rows["lr_ctr_all (fair)"]["winners_auc"]
    lgb_w = rows["lgb_ctr_all (fair)"]["winners_auc"]
    under = sum(1 for d in n["winners_deciles"] if d["ratio"] is not None and d["ratio"] < 0.8)
    verdict = {
        "neural_winners_auc": n["winners_auc"],
        "lr_winners_auc": lr_w, "lgb_winners_auc": lgb_w,
        "neural_vs_lgb_gap": round(n["winners_auc"] - lgb_w, 4),
        "ranking_converged (|neural-lgb|<0.03)": abs(n["winners_auc"] - lgb_w) < 0.03,
        "neural_winners_ieb": n["winners_ieb"],
        "calibration_under_deciles (ratio<0.8, was 10/10)": under,
        "calibration_fixed (<=3 deciles under)": under <= 3,
    }
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))
    print("\nneural winners deciles (pred/true ratio; was 0.06..0.80 monotone-under before fix):")
    for d in n["winners_deciles"]:
        print(f"  bin {d['bin']}: pred {d['pred']:.6f}  true {d['true']:.6f}  ratio {d['ratio']}")

    OUT.write_text(json.dumps({"models": rows, "verdict": verdict}, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
