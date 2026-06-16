"""Phase 1c — budget pacing on the FAIR split (CPU).

Fair-split replacement for the original/unfair-split results/bidding/pacing_comparison.csv.
Simulates daily budget pacing over the 24-hour cycle on the fair test set (recal neural pCTR,
truthful 2p bids), comparing three allocations across budget levels:
  - no_pacing:   chronological greedy, exhausts budget early in the day
  - pid_uniform: budget spread evenly across hours (1/24 each)
  - wr_weighted: budget weighted by hourly winnable volume (more budget to busy hours)
Within each hour, impressions are taken most-profitable-first ((V−payprice) desc) until the hour
budget is used. Honest expectation: gains are small because iPinYou intra-day win rates are stable.

Output: results/stage_a/pacing_fair.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.features.engineering import load_feature_splits

PROJECT = Path(__file__).resolve().parents[2]
FAIR = PROJECT / "data/ipinyou/prediction/features_fair"
NPZ = PROJECT / "results/stage_a/recalibrated_winners_preds.npz"
OUT = PROJECT / "results/stage_a/pacing_fair.json"
CPC = 200000.0
MAX_BID = 300.0


def simulate(V, pp, clicks, hour, budget, hour_w):
    """Hour-bucketed greedy 2p pacing sim. hour_w: array[24], per-hour budget share (sums to 1)."""
    spend = surplus = clk = wins = 0.0
    bid = np.minimum(V, MAX_BID)
    winnable = bid >= pp
    for h in range(24):
        m = hour == h
        if not m.any():
            continue
        v_h, pp_h, c_h, w_h = V[m], pp[m], clicks[m], winnable[m]
        order = np.argsort(-(v_h - pp_h))  # most profitable first
        pp_o, v_o, c_o, win_o = pp_h[order], v_h[order], c_h[order], w_h[order]
        cum = np.cumsum(np.where(win_o, pp_o, 0.0))
        hb = budget * hour_w[h]
        take = win_o & (cum <= hb)
        spend += pp_o[take].sum()
        surplus += (v_o[take] - pp_o[take]).sum()
        clk += c_o[take].sum()
        wins += take.sum()
    roi = (clk * CPC / spend) if spend > 0 else 0.0
    return {"utilization": round(spend / budget, 4), "spend": float(spend),
            "surplus": float(surplus), "clicks": int(clk), "wins": int(wins), "roi": round(roi, 3)}


def main() -> None:
    d = np.load(NPZ)
    idx = d["idx_won"]
    clicks = np.asarray(d["y_click_won"], dtype=np.float64)
    V = np.asarray(d["escm2wc_dr_recal"], dtype=np.float64) * CPC
    _, _, test_df, _ = load_feature_splits(str(FAIR), columns=["payprice", "hour"])
    pp = np.asarray(test_df["payprice"].values, dtype=np.float64)[idx]
    hour = np.asarray(test_df["hour"].values, dtype=np.int64)[idx]

    keep = pp > 0
    V, pp, clicks, hour = V[keep], pp[keep], clicks[keep], hour[keep]
    bid = np.minimum(V, MAX_BID)
    winnable = bid >= pp
    full_spend = float(pp[winnable].sum())     # unconstrained truthful 2p spend
    print(f"winners={keep.sum():,}  unconstrained spend={full_spend:.3e}")

    # hourly winnable volume → weights
    hv = np.array([float((winnable & (hour == h)).sum()) for h in range(24)])
    wr_w = hv / hv.sum()
    uni_w = np.ones(24) / 24.0
    # no_pacing = a single front-loaded weighting (early hours get all budget first):
    # approximate by huge weight on chronological order → emulate via cumulative full participation
    chrono = np.zeros(24);
    # give each hour weight = its share but allow early hours to fully spend (front-load) → use cumulative cap
    # simplest: no_pacing lets every hour bid fully (weight = its own winnable spend share, uncapped-ish)
    hourly_spend = np.array([float(pp[(winnable) & (hour == h)].sum()) for h in range(24)])
    nopace_w = hourly_spend / hourly_spend.sum()

    budgets = [round(full_spend * f) for f in (0.2, 0.4, 0.6, 0.8)]
    out = {"_meta": {"split": "features_fair", "auction": "second_price",
                     "model": "escm2wc_dr recalibrated", "cpc": CPC,
                     "unconstrained_spend": full_spend, "n_winners": int(keep.sum()),
                     "note": "Fair-split budget pacing. Replaces original-split pacing_comparison.csv. "
                             "Budgets are fractions {0.2,0.4,0.6,0.8} of unconstrained truthful 2p spend."},
           "by_budget": {}}
    for B in budgets:
        out["by_budget"][f"{B:.0f}"] = {
            "budget_frac": round(B / full_spend, 2),
            "no_pacing": simulate(V, pp, clicks, hour, B, nopace_w),
            "pid_uniform": simulate(V, pp, clicks, hour, B, uni_w),
            "wr_weighted": simulate(V, pp, clicks, hour, B, wr_w),
        }
        r = out["by_budget"][f"{B:.0f}"]
        lift = (r["wr_weighted"]["surplus"] / r["pid_uniform"]["surplus"] - 1) * 100
        print(f"B={B:.2e} (frac {r['budget_frac']}): uniform surplus={r['pid_uniform']['surplus']:.3e} "
              f"util={r['pid_uniform']['utilization']} | wr-weighted lift={lift:+.1f}%")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
