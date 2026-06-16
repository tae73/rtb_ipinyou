"""Phase 1a (finalize) — assemble the fair-split ablation ladder.

Closes the headline ablation on ONE consistent fair split, computed with ONE code path:
  Biased LR  →  Biased LGB  →  ESMM-WC  →  ESCM²-WC (IPW)  →  ESCM²-WC (DR)
The neural rungs share held-constant CTR supervision (ctr_weight=1, pos_weight=50, joint=0.1,
embed=16); only the debiasing mechanism varies (none → ESMM joint → IPW → DR). For each rung:
winners-only AUC (the bidding object), winners IEB raw→after cross-fit isotonic, recal AUC.

Inputs (fair-split test predictions):
  LR/LGB:  results/stage_a/fair_baseline_preds.npz (lr_p_all, lgb_p_all, y_win, y_click)
  ESMM-WC: results/models/esmmwc_fair/esmmwc_test_predictions.npz (p_ctr, y_win, y_click)
  IPW:     results/models/escm2wc_ipw_fair/escm2wc_ipw_test_predictions.npz
  DR:      results/models/escm2wc_dr_fair_posw/escm2wc_dr_test_predictions.npz
Output: results/stage_a/ablation_ladder.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.metrics.calibration import cross_fit_isotonic

PROJECT = Path(__file__).resolve().parents[2]
MODELS = PROJECT / "results/models"
OUT = PROJECT / "results/stage_a/ablation_ladder.json"


def _ieb(pred, true):
    tm = float(true.mean())
    return float((tm - float(pred.mean())) / tm) if tm > 0 else float("nan")


def _rung(label, mechanism, p_all, y_win, y_click):
    w = y_win == 1
    pw, yw = np.asarray(p_all)[w], np.asarray(y_click)[w]
    auc_w = float(roc_auc_score(yw, pw))
    ieb_raw = _ieb(pw, yw)
    p_recal = cross_fit_isotonic(pw, yw, n_folds=5, seed=0)
    return {
        "label": label, "mechanism": mechanism,
        "winners_auc": round(auc_w, 4),
        "winners_ieb_raw": round(ieb_raw, 4),
        "winners_ieb_recal": round(_ieb(p_recal, yw), 4),
        "winners_auc_recal": round(float(roc_auc_score(yw, p_recal)), 4),
        "n_winners": int(w.sum()), "n_clicks": int(yw.sum()),
    }


def main() -> None:
    base = np.load(PROJECT / "results/stage_a/fair_baseline_preds.npz")
    y_win, y_click = base["y_win"], base["y_click"]

    rungs = []
    rungs.append(_rung("LR (ctr_all)", "biased baseline (linear)", base["lr_p_all"], y_win, y_click))
    rungs.append(_rung("LGB (ctr_all)", "biased baseline (GBM)", base["lgb_p_all"], y_win, y_click))

    neural = [
        ("ESMM-WC", "ESMM joint constraint (implicit)", "esmmwc_fair/esmmwc_test_predictions.npz"),
        ("ESCM²-WC (IPW)", "inverse-propensity weighting", "escm2wc_ipw_fair/escm2wc_ipw_test_predictions.npz"),
        ("ESCM²-WC (DR)", "doubly robust (primary)", "escm2wc_dr_fair_posw/escm2wc_dr_test_predictions.npz"),
    ]
    for label, mech, rel in neural:
        path = MODELS / rel
        if not path.exists():
            print(f"  MISSING {rel} — skipping {label}")
            continue
        d = np.load(path)
        rungs.append(_rung(label, mech, d["p_ctr"], d["y_win"], d["y_click"]))

    for r in rungs:
        print(f"  {r['label']:18} winners-AUC {r['winners_auc']}  IEB {r['winners_ieb_raw']}→{r['winners_ieb_recal']}")

    out = {
        "_meta": {
            "split": "features_fair (per-advertiser temporal, shared vocab)",
            "object": "winners-only AUC = P(click|win), the object bidding ranks on",
            "neural_shared_config": {"embed_dim": 16, "hidden_dims": [128, 64], "win_hidden_dims": [64, 32],
                                     "ctr_weight": 1.0, "ctr_pos_weight": 50, "joint_weight": 0.1,
                                     "batch_size": 65536, "lr": 0.0005, "epochs": 50, "patience": 10},
            "recal": "cross-fit isotonic (K=5, leak-free), winners pCTR",
            "note": "Held-constant CTR supervision across neural rungs; only the debiasing mechanism "
                    "varies. ESMM-WC + IPW retrained on the fair split (2026-06) to close the ladder.",
        },
        "ladder": rungs,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
