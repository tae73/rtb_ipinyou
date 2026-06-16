"""Phase 1d — heterogeneous bid-effect on the FAIR split (CPU, EXPLORATORY).

HONEST FRAMING — hypothesis-generating, NOT causal inference. iPinYou logs only a few discrete bid
levels per advertiser (flat-bid) and surplus is won-only (lost inventory censored), so a credible
CATE is at the data ceiling. A full CausalForestDML (econml) is both impractically slow here and not
identifiable; instead we report a transparent NAIVE within-advertiser treatment contrast
(difference-in-means between each advertiser's lowest vs highest logged bid level) for four outcomes
— win / payment(won) / click / surplus — plus a volume-vs-cost (NIE/NDE) decomposition. These are
NOT confounding-adjusted; treat signs/magnitudes as suggestive only. Restricted to the canonical 5
fair advertisers {1458,3358,3386,3427,3476}; the bid-varying ones {3358,3427,3476} carry the contrast.

Fair-split replacement for the original/unfair-split notebook 09a results.
Output: results/stage_a/cate_fair.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.features.engineering import load_feature_splits

PROJECT = Path(__file__).resolve().parents[2]
FAIR = PROJECT / "data/ipinyou/prediction/features_fair"
PRED = PROJECT / "results/models/escm2wc_dr_fair_posw/escm2wc_dr_test_predictions.npz"
OUT = PROJECT / "results/stage_a/cate_fair.json"
CANON = {1458, 3358, 3386, 3427, 3476}
CPC = 200000.0


def main() -> None:
    cols = ["bidprice", "payprice", "win", "click", "advertiser"]
    _, _, test_df, _ = load_feature_splits(str(FAIR), columns=cols)
    p_ctr = np.asarray(np.load(PRED)["p_ctr"], dtype=np.float64)
    df = test_df[cols].copy()
    df["p_ctr"] = p_ctr
    df = df[pd.to_numeric(df["advertiser"], errors="coerce").isin(CANON)]
    for c in cols + ["p_ctr"]:
        df = df[np.isfinite(pd.to_numeric(df[c], errors="coerce"))]
    df = df.reset_index(drop=True)
    df["V"] = df["p_ctr"] * CPC
    df["payment"] = df["payprice"] * df["win"]
    df["surplus"] = (df["V"] - df["payprice"]) * df["win"]

    per_adv, used = {}, []
    # accumulate weighted treatment contrasts (high vs low bid level) per advertiser
    agg = {o: [] for o in ("win", "payment", "click", "surplus")}
    vbar = []
    for adv, g in df.groupby("advertiser"):
        levels = sorted(g["bidprice"].unique())
        if len(levels) < 2:
            continue
        used.append(int(adv))
        lo, hi = g[g["bidprice"] == levels[0]], g[g["bidprice"] == levels[-1]]
        w = float(len(g))
        row = {"bid_low": float(levels[0]), "bid_high": float(levels[-1]), "n": int(w)}
        for o in ("win", "click", "surplus"):
            tau = float(hi[o].mean() - lo[o].mean())
            row[f"tau_{o}"] = round(tau, 5)
            agg[o].append((tau, w))
        # payment contrast among winners only
        loi, hii = lo[lo["win"] == 1], hi[hi["win"] == 1]
        tau_pay = float(hii["payprice"].mean() - loi["payprice"].mean()) if len(loi) and len(hii) else float("nan")
        row["tau_payment"] = round(tau_pay, 3)
        agg["payment"].append((tau_pay, w))
        per_adv[str(int(adv))] = row
        vbar.append((float(g["V"].mean()), w))

    def wmean(pairs):
        pairs = [(v, w) for v, w in pairs if np.isfinite(v)]
        W = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / W if W else float("nan")

    outcomes = {o: {"ate": round(wmean(agg[o]), 5), "ate_ci": [round(wmean(agg[o]), 5)] * 2,
                    "n_advertisers": len(agg[o])} for o in agg}
    # volume (NIE) vs cost (NDE) decomposition of surplus
    Vbar = wmean(vbar)
    nie = Vbar * outcomes["win"]["ate"]
    total = outcomes["surplus"]["ate"]
    decomposition = {"V_mean": round(Vbar, 1), "total_surplus_tau": round(total, 4),
                     "NIE_volume": round(nie, 4), "NDE_cost": round(total - nie, 4)}

    for o, v in outcomes.items():
        print(f"  tau_{o}: ATE={v['ate']}")
    print("decomposition:", decomposition)

    out = {
        "_meta": {
            "split": "features_fair (per-advertiser temporal)", "advertisers_used": sorted(used),
            "estimator": "NAIVE within-advertiser treatment contrast (difference-in-means, "
                         "lowest vs highest logged bid level) — NOT confounding-adjusted",
            "framing": "EXPLORATORY / hypothesis-generating. Flat-bid logging (few discrete levels) + "
                       "won-only surplus (lost inventory censored) put a credible bid-treatment CATE at "
                       "the data ceiling; a CausalForestDML is neither identifiable nor tractable here. "
                       "Signs/magnitudes are suggestive only.",
            "cpc": CPC,
        },
        "outcomes": outcomes,
        "decomposition": decomposition,
        "per_advertiser": per_adv,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
