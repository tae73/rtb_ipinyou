"""Recalibration trap — WHY naive recalibration hurts full-inventory bidding.

Mechanism: recalibrating a selection-biased pCTR model RAISES its level (the biased model under-predicts
on winners-only), so truthful bids `b = p̂·CPC` go UP → the policy WINS MORE impressions. But the extra
wins are disproportionately MARGINAL — impressions where the *true* value pctr·CPC is BELOW the clearing
price (true surplus < 0). Each such win subtracts surplus. Doubly-robust debiasing corrects the shape
without the blanket level inflation, so it avoids the over-bidding.

We decompose, for biased vs biased+recal vs debiased: n_wins, won-surplus, and the share of wins that
are UNPROFITABLE (true value < price). Run at a strong-selection cell where the trap bites.

Output: witnesses/recal_trap.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

import phase_diagram as P   # reuse the testbed + model helpers (no sweep runs on import)

OUT = Path(__file__).resolve().parent / "recal_trap.json"


def decompose(phat, pctr, mprice):
    bid = phat * P.CPC
    win = bid >= mprice
    true_val = pctr * P.CPC
    prof = win & (true_val >= mprice)
    unprof = win & (true_val < mprice)
    return {
        "mean_bid": round(float(bid.mean()), 1),
        "n_wins": int(win.sum()),
        "won_surplus": round(float(np.sum((true_val - mprice) * win)), 1),
        "n_unprofitable_wins": int(unprof.sum()),
        "unprofitable_win_share": round(float(unprof.sum() / max(win.sum(), 1)), 3),
        "surplus_from_unprofitable": round(float(np.sum((true_val - mprice) * unprof)), 1),
    }


def main():
    rows = {}
    for tag, (gamma, theta, cap) in {"linear_strong": (1.2, 0, "linear"),
                                     "gbm_strong": (1.2, 0, "gbm")}.items():
        X, pctr, click, mprice = P.make_pop(gamma, theta, seed=7)
        b0 = np.quantile(mprice, 0.5)
        win = (b0 >= mprice).astype(int)
        pw = LGBMClassifier(n_estimators=150, num_leaves=31, min_child_samples=80,
                            verbose=-1).fit(X, win).predict_proba(X)[:, 1]
        w = win == 1
        p_base = P.fit_pred(X, win, click, cap, False, pw)
        p_rec = P.xfit_isotonic(p_base[w], click[w], p_base)
        p_deb = P.fit_pred(X, win, click, "gbm", True, pw)
        rows[tag] = {
            "gamma": gamma, "baseline_capacity": cap,
            "biased": decompose(p_base, pctr, mprice),
            "biased_recal": decompose(p_rec, pctr, mprice),
            "debiased": decompose(p_deb, pctr, mprice),
            "oracle_surplus": round(float(P.surplus(pctr, pctr, mprice)), 1),
        }
        r = rows[tag]
        print(f"[{tag}] mean bid: base={r['biased']['mean_bid']} recal={r['biased_recal']['mean_bid']} "
              f"deb={r['debiased']['mean_bid']}")
        print(f"   unprofitable-win share: base={r['biased']['unprofitable_win_share']} "
              f"recal={r['biased_recal']['unprofitable_win_share']} deb={r['debiased']['unprofitable_win_share']}")
        print(f"   won surplus: base={r['biased']['won_surplus']:.0f} recal={r['biased_recal']['won_surplus']:.0f} "
              f"deb={r['debiased']['won_surplus']:.0f} (oracle {r['oracle_surplus']:.0f})")

    out = {"_meta": {"claim": "Recalibrating a selection-biased pCTR model raises bids -> wins more "
                              "marginal impressions where true value < clearing price (negative surplus); "
                              "DR debiasing avoids the blanket level inflation.",
                     "metric": "expected full-inventory 2nd-price surplus; unprofitable = won & true value < price"},
           "cases": rows}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
