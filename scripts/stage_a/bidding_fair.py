"""Phase 1b — bid-shading strategy comparison on the FAIR split (canonical, CPU).

Complements stage_b2_surplus.json (which holds the neural-vs-baseline decision-value gap for 3
strategies) by characterizing the SHADING STRATEGIES themselves on the fair split: a full strategy
comparison {truthful, exchange_optimal, dual_regime, linear, percentile} and a linear-shading
alpha-sensitivity sweep — the fair-split replacements for the original/unfair-split
results/bidding/{strategy_comparison_*,alpha_sensitivity}.csv (advertisers 2259/2261/2821/2997).

All on the FAIR per-advertiser test set, recalibrated neural pCTR, second-price auction on actual
payprice. Reuses src/bidding machinery. No model retraining.

Inputs:  results/stage_a/recalibrated_winners_preds.npz (idx_won, y_click_won, escm2wc_dr_recal),
         data/ipinyou/prediction/features_fair/test.parquet (payprice, adexchange, slotprice, advertiser),
         results/market_price_cdf/*.npz.
Output:  results/stage_a/bidding_fair.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.bidding.shading import (
    ShadingConfig,
    compute_shaded_bids,
    dual_regime_shading,
    exchange_conditional_shading,
    load_exchange_cdfs,
    load_market_cdf,
)
from src.bidding.simulator import compute_simulation_metrics, run_auction_simulation
from src.features.engineering import load_feature_splits

PROJECT = Path(__file__).resolve().parents[2]
FAIR = PROJECT / "data/ipinyou/prediction/features_fair"
NPZ = PROJECT / "results/stage_a/recalibrated_winners_preds.npz"
CDF_DIR = PROJECT / "results/market_price_cdf"
OUT = PROJECT / "results/stage_a/bidding_fair.json"

CPC = 200000.0
MAX_BID = 300.0
STRATEGIES = ["truthful", "exchange_optimal", "dual_regime", "linear", "percentile"]
ALPHAS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _shade(strategy, V, ax, slot, overall, exch, alpha=0.8, pct=0.75):
    if strategy == "truthful":
        return np.minimum(V, MAX_BID)
    if strategy == "exchange_optimal":
        cfg = ShadingConfig(strategy="optimal", exchange_conditional=True, max_bid=MAX_BID)
        return exchange_conditional_shading(V, ax, exch, overall, cfg).bids
    if strategy == "dual_regime":
        return dual_regime_shading(V, overall, slot, ShadingConfig(strategy="dual_regime", max_bid=MAX_BID)).bids
    if strategy == "linear":
        return compute_shaded_bids(V, overall, ShadingConfig(strategy="linear", linear_alpha=alpha, max_bid=MAX_BID)).bids
    if strategy == "percentile":
        return compute_shaded_bids(V, overall, ShadingConfig(strategy="percentile", percentile_target=pct, max_bid=MAX_BID)).bids
    raise ValueError(strategy)


def _metrics(strategy, V, pp, clicks, ax, slot, overall, exch, alpha=0.8):
    bids = _shade(strategy, V, ax, slot, overall, exch, alpha=alpha)
    res = run_auction_simulation(bids, pp, V, clicks, auction_type="second_price")
    m = compute_simulation_metrics(res, V, pp, strategy, cpc_target=CPC)
    return {
        "strategy": strategy, "n_wins": m.n_wins, "win_rate": round(m.win_rate, 4),
        "total_surplus": float(m.total_surplus), "avg_cpm": round(m.avg_cpm, 2),
        "avg_cpc": round(m.avg_cpc, 1), "roi": round(m.roi, 3),
        "overpayment_ratio": round(m.overpayment_ratio, 3), "mean_bid": round(float(np.mean(bids)), 2),
    }


def main() -> None:
    d = np.load(NPZ)
    idx = d["idx_won"]
    clicks = np.asarray(d["y_click_won"], dtype=np.int32)
    p_neural = np.asarray(d["escm2wc_dr_recal"], dtype=np.float64)

    _, _, test_df, _ = load_feature_splits(
        str(FAIR), columns=["payprice", "bidprice", "adexchange", "slotprice", "advertiser"])
    pp = np.asarray(test_df["payprice"].values, dtype=np.float64)[idx]
    bidp = np.asarray(test_df["bidprice"].values, dtype=np.float64)[idx]
    ax = np.asarray(test_df["adexchange"].values)[idx]
    slot = np.asarray(test_df["slotprice"].values, dtype=np.float64)[idx]
    adv = np.asarray(test_df["advertiser"].values)[idx]

    keep = pp > 0
    pp, bidp, ax, slot, adv = pp[keep], bidp[keep], ax[keep], slot[keep], adv[keep]
    clicks, p_neural = clicks[keep], p_neural[keep]
    V = p_neural * CPC

    overall = load_market_cdf(str(CDF_DIR / "km_cdf_overall.npz"))
    exch = load_exchange_cdfs(str(CDF_DIR))

    print(f"winners (payprice>0)={keep.sum():,}  clicks={int(clicks.sum()):,}  mean V={V.mean():.1f}")

    # 1) strategy comparison (neural recal)
    strat = {s: _metrics(s, V, pp, clicks, ax, slot, overall, exch) for s in STRATEGIES}
    best = max(strat, key=lambda s: strat[s]["total_surplus"])
    print("strategy surplus:", {s: f"{strat[s]['total_surplus']:.3e}" for s in STRATEGIES}, "| best:", best)

    # 2) alpha-sensitivity (linear shading)
    alpha_sweep = [{"alpha": a, **{k: v for k, v in _metrics("linear", V, pp, clicks, ax, slot, overall, exch, alpha=a).items()
                                   if k != "strategy"}} for a in ALPHAS]

    # 3) per-advertiser surplus for the best strategy
    bids_best = _shade(best, V, ax, slot, overall, exch)
    res_best = run_auction_simulation(bids_best, pp, V, clicks, auction_type="second_price")
    per_adv = {}
    for a in sorted(set(adv.tolist())):
        m = adv == a
        per_adv[str(int(a))] = {
            "n_winners": int(m.sum()),
            "surplus": float(res_best.surplus[m].sum()),
            "win_rate": round(float(res_best.wins[m].mean()), 4),
            "clicks": int(clicks[m].sum()),
        }

    out = {
        "_meta": {
            "split": "features_fair (per-advertiser temporal)", "auction": "second_price",
            "model": "escm2wc_dr recalibrated (cross-fit isotonic)", "cpc": CPC, "max_bid": MAX_BID,
            "n_winners": int(keep.sum()), "n_clicks": int(clicks.sum()),
            "advertisers": sorted(int(a) for a in set(adv.tolist())),
            "note": "Fair-split bid-shading characterization. Replaces the original/unfair-split "
                    "results/bidding/{strategy_comparison_*,alpha_sensitivity}.csv (advertisers 2259/2261/2821/2997). "
                    "Decision-value gap vs baselines is in stage_b2_surplus.json.",
        },
        "strategy_comparison": strat,
        "best_strategy": best,
        "alpha_sensitivity": alpha_sweep,
        "per_advertiser_best": per_adv,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
