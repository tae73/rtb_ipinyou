"""Recalibration trap — WHY naive recalibration hurts full-inventory bidding.

Mechanism: recalibrating a selection-biased pCTR model RAISES its level (the biased model under-predicts
on winners-only), so truthful bids `b = p̂·CPC` go UP → the policy WINS MORE impressions. But the extra
wins are disproportionately MARGINAL — impressions where the *true* value pctr·CPC is BELOW the clearing
price (true surplus < 0). Each such win subtracts surplus. Principled IPW debiasing corrects the SHAPE
(via win-propensity reweighting) without the blanket level inflation, so it avoids the over-bidding.

Honest framing: biased / biased+recal / IPW debiased are all AT THE SAME MODEL CAPACITY as the baseline
(so this isolates what the recalibration *does to bidding*, not a model-class swap). We decompose n_wins,
won-surplus, and the share of wins that are UNPROFITABLE (true value < price), at a strong-selection cell
where the trap bites, and report a multi-seed robustness block (the trap is not a single-cell artifact).

Output: witnesses/recal_trap.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

import phase_diagram as P   # reuse the testbed + model helpers (no sweep runs on import)

OUT = Path(__file__).resolve().parent / "recal_trap.json"


def _fit_all(gamma, theta, cap, seed):
    X, pctr, click, mprice = P.make_pop(gamma, theta, seed)
    b0 = np.quantile(mprice, 0.5)
    win = (b0 >= mprice).astype(int)
    pw = LGBMClassifier(n_estimators=150, num_leaves=31, min_child_samples=80,
                        verbose=-1).fit(X, win).predict_proba(X)[:, 1]
    w = win == 1
    p_base = P.fit_pred(X, win, click, cap, False, pw)
    p_rec = P.xfit_isotonic(p_base[w], click[w], p_base)
    p_deb = P.fit_pred(X, win, click, cap, True, pw)      # IPW at the SAME capacity as the baseline
    return pctr, mprice, p_base, p_rec, p_deb


def decompose(phat, pctr, mprice):
    bid = phat * P.CPC
    win = bid >= mprice
    true_val = pctr * P.CPC
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
        pctr, mprice, p_base, p_rec, p_deb = _fit_all(gamma, theta, cap, seed=7)
        rows[tag] = {
            "gamma": gamma, "baseline_capacity": cap, "debiaser": f"IPW ({cap} capacity)",
            "biased": decompose(p_base, pctr, mprice),
            "biased_recal": decompose(p_rec, pctr, mprice),
            "debiased": decompose(p_deb, pctr, mprice),
            "oracle_surplus": round(float(P.surplus(pctr, pctr, mprice)), 1),
        }
        r = rows[tag]
        print(f"[{tag}] mean bid: base={r['biased']['mean_bid']} recal={r['biased_recal']['mean_bid']} "
              f"ipw={r['debiased']['mean_bid']}")
        print(f"   unprofitable-win share: base={r['biased']['unprofitable_win_share']} "
              f"recal={r['biased_recal']['unprofitable_win_share']} ipw={r['debiased']['unprofitable_win_share']}")
        print(f"   won surplus: base={r['biased']['won_surplus']:.0f} recal={r['biased_recal']['won_surplus']:.0f} "
              f"ipw={r['debiased']['won_surplus']:.0f} (oracle {r['oracle_surplus']:.0f})")

    # robustness: the trap (recal surplus < biased) and IPW recovery across seeds, linear_strong regime
    seeds = [1, 2, 3, 7, 42]
    rec_drop, deb_gain, bids = [], [], {"biased": [], "recal": [], "ipw": []}
    for s in seeds:
        pctr, mprice, p_base, p_rec, p_deb = _fit_all(1.2, 0, "linear", s)
        b = decompose(p_base, pctr, mprice); rc = decompose(p_rec, pctr, mprice); db = decompose(p_deb, pctr, mprice)
        rec_drop.append(rc["won_surplus"] < b["won_surplus"])      # recal hurts
        deb_gain.append(db["won_surplus"] > b["won_surplus"])      # IPW helps
        bids["biased"].append(b["mean_bid"]); bids["recal"].append(rc["mean_bid"]); bids["ipw"].append(db["mean_bid"])
    robustness = {
        "regime": "linear_strong (γ=1.2, θ=0)", "seeds": seeds,
        "recal_lowers_surplus_frac": f"{sum(rec_drop)}/{len(seeds)}",
        "ipw_raises_surplus_frac": f"{sum(deb_gain)}/{len(seeds)}",
        "mean_bid_biased": round(float(np.mean(bids['biased'])), 1),
        "mean_bid_recal": round(float(np.mean(bids['recal'])), 1),
        "mean_bid_ipw": round(float(np.mean(bids['ipw'])), 1),
        "recal_bid_inflation_pp": round(float((np.mean(bids['recal']) / np.mean(bids['biased']) - 1) * 100), 1),
    }
    print(f"\nROBUSTNESS (5 seeds): recal lowers surplus {robustness['recal_lowers_surplus_frac']}, "
          f"IPW raises surplus {robustness['ipw_raises_surplus_frac']}; "
          f"recal inflates bid +{robustness['recal_bid_inflation_pp']}%")

    out = {"_meta": {"claim": "Recalibrating a selection-biased pCTR model raises bids -> wins more "
                              "marginal impressions where true value < clearing price (negative surplus); "
                              "IPW debiasing (same capacity) corrects the shape without the blanket level inflation.",
                     "metric": "expected full-inventory 2nd-price surplus; unprofitable = won & true value < price",
                     "debiaser": "IPW (winners-only, win-propensity weighted), fit at the baseline's own model capacity"},
           "cases": rows, "robustness": robustness}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
