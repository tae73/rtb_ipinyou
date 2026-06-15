"""Stage A — Step 2: A2-immediate EASY-NEGATIVES decomposition.

For each model we measure how much of its all-bids CTR-AUC is the TRIVIAL
won/not-won separation (the impressions that never won are near-certain
non-clicks) versus REAL skill at ranking clicks among winners.

  all_bids_auc     = AUC( all-bids pCTR , y_click )  over ALL rows
  winners_only_auc = AUC( winners pCTR  , y_click )  over y_win==1 subset
  easy_neg_gap     = all_bids_auc - winners_only_auc

  Neural   : all-bids = p_click_bid (=p_win*p_ctr); winners = p_ctr (P(click|win)).
  LR/LGB   : single all-bids predictor used for BOTH columns
             (all-rows AUC, and the same predictor restricted to the won subset).

A large positive gap means the headline all-bids AUC is mostly the easy
won/not-won contrast, not winners-CTR skill.

AUC via src.metrics.evaluation._numpy_roc_auc (the project's canonical pure-numpy
ROC-AUC, identical to what produced the saved result JSONs).

Also emits a SKEW-CHECK: the recomputed LR/LGB all-bids AUC must match the saved
lr_ctr_all_result.json / lgb_ctr_all_result.json test AUC.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.metrics.evaluation import _numpy_roc_auc

PROJECT = Path("/home/mail-agent/project/rtb_ipinyou")
NPZ = PROJECT / "results" / "stage_a" / "test_predictions_all.npz"
MODELS = PROJECT / "results" / "models"
OUT = PROJECT / "results" / "stage_a" / "easy_negatives.json"

NEURAL = ["esmmwc", "escm2wc_dr", "escm2wc_dr_extps"]
BASELINES = ["lr_ctr_all", "lgb_ctr_all"]


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return _numpy_roc_auc(
        np.asarray(y_true, dtype=np.intp),
        np.asarray(y_score, dtype=np.float64),
    )


def main() -> None:
    d = np.load(NPZ)
    y_click = d["y_click"].astype(np.intp)
    y_win = d["y_win"].astype(np.intp)
    won = y_win == 1
    n = int(y_click.shape[0])
    n_win = int(won.sum())
    y_click_won = y_click[won]
    print(f"n={n:,}  n_win={n_win:,}  clicks(all)={int(y_click.sum()):,}  "
          f"clicks(won)={int(y_click_won.sum()):,}")

    results: dict[str, dict] = {}

    # --- neural: all-bids = p_click_bid ; winners = p_ctr on won subset ---
    for m in NEURAL:
        all_pred = d[f"{m}_p_click_bid"].astype(np.float64)
        win_pred = d[f"{m}_p_ctr"].astype(np.float64)[won]
        a_all = _auc(y_click, all_pred)
        a_win = _auc(y_click_won, win_pred)
        results[m] = {
            "all_bids_auc": a_all,
            "winners_only_auc": a_win,
            "easy_neg_gap": a_all - a_win,
            "n": n,
            "n_win": n_win,
            "all_bids_pred": "p_click_bid (p_win*p_ctr)",
            "winners_pred": "p_ctr (P(click|win))",
        }
        print(f"{m:18s} all={a_all:.6f}  won={a_win:.6f}  gap={a_all - a_win:+.6f}")

    # --- baselines: one all-bids predictor used for both views ---
    for m in BASELINES:
        pred = d[m].astype(np.float64)
        a_all = _auc(y_click, pred)
        a_win = _auc(y_click_won, pred[won])
        results[m] = {
            "all_bids_auc": a_all,
            "winners_only_auc": a_win,
            "easy_neg_gap": a_all - a_win,
            "n": n,
            "n_win": n_win,
            "all_bids_pred": f"{m} all-bids pCTR",
            "winners_pred": f"{m} all-bids pCTR restricted to won subset",
        }
        print(f"{m:18s} all={a_all:.6f}  won={a_win:.6f}  gap={a_all - a_win:+.6f}")

    # --- SKEW-CHECK against saved result JSONs ---
    skew = {}
    for m, jp in [("lr_ctr_all", "lr_ctr_all_result.json"),
                  ("lgb_ctr_all", "lgb_ctr_all_result.json")]:
        saved = json.loads((MODELS / jp).read_text())["test_metrics"]["auc"]
        recomputed = results[m]["all_bids_auc"]
        skew[m] = {
            "saved_test_auc": saved,
            "recomputed_all_bids_auc": recomputed,
            "abs_diff": abs(saved - recomputed),
        }
        print(f"SKEW-CHECK {m}: saved={saved:.7f} recomputed={recomputed:.7f} "
              f"|diff|={abs(saved - recomputed):.2e}")
    results["_skew_check"] = skew

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
