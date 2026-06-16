"""Stage B2 — decision-level value of debiasing: realized surplus + slice calibration.

Stage A leveled GLOBAL calibration: cross-fitted isotonic drove winners-pCTR IEB to ~0 for
all three models (neural escm2wc_dr, LR, LGB). Cross-fit pins each recal model's MEAN pCTR to
the empirical base rate -> all three recal cells have *identical* mean V. So among-recal surplus
differences isolate **ranking + residual slice-calibration**, with the mean-bid-level confound
removed by construction. This probe asks: is there decision-level value from debiasing BEYOND
what cheap global recalibration gives a strong ranker?

Method (mirrors scripts/stage_a/surplus_corr.py): for each {model}×{raw,recal} cell and each
shading strategy, V = pCTR*CPC -> bids -> run_auction_simulation on the WON-ONLY set using ACTUAL
payprice -> realized (model-independent) surplus = Σ_{re-won}(click·CPC − payprice). The thesis
number is the among-RECAL gap neural−{lgb,lr}, with paired-Poisson + advertiser-cluster bootstrap
CIs (rare-click variance => a point estimate alone is untrustworthy).

Reads:
- results/stage_a/recalibrated_winners_preds.npz (idx_won, y_click_won, {model}_raw/_recal).
- data/ipinyou/prediction/features_fair/test.parquet (payprice, adexchange, slotprice, advertiser).
- results/market_price_cdf/{km_cdf_overall,km_cdf_exchange_*}.npz.

Writes:
- results/stage_a/stage_b2_surplus.json, results/stage_a/stage_b2_surplus_summary.md.

SCOPE LIMITATION: won-only surplus scores only impressions the ORIGINAL policy won (a cheap,
non-random slice). Debiasing's value on LOST inventory is unobservable offline (censored payprice).
This is a conservative lower bound on decision value, not a full policy evaluation.

Usage: python scripts/stage_a/stage_b2_surplus.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.bidding.shading import (
    ShadingConfig,
    dual_regime_shading,
    exchange_conditional_shading,
    load_exchange_cdfs,
    load_market_cdf,
)
from src.bidding.simulator import (
    cluster_bootstrap_surplus_gap,
    compute_simulation_metrics,
    paired_bootstrap_surplus_gap,
    run_auction_simulation,
)
from src.bidding.value import ValueConfig, compute_impression_values
from src.features.engineering import load_feature_splits
from src.metrics.calibration import quantile_reliability, slice_calibration
from src.metrics.evaluation import _numpy_roc_auc, compute_ieb

PROJECT = Path("/home/mail-agent/project/rtb_ipinyou")
NPZ = PROJECT / "results/stage_a/recalibrated_winners_preds.npz"
FAIR = PROJECT / "data/ipinyou/prediction/features_fair"
CDF_DIR = PROJECT / "results/market_price_cdf"
OUT_JSON = PROJECT / "results/stage_a/stage_b2_surplus.json"
OUT_MD = PROJECT / "results/stage_a/stage_b2_surplus_summary.md"

CPC = 200_000.0
MAX_BID = 300.0
MODELS = [("escm2wc_dr", "neural"), ("lr_ctr_all", "lr"), ("lgb_ctr_all", "lgb")]
VARIANTS = ["raw", "recal"]
STRATEGIES = ["exchange_optimal", "dual_regime", "truthful"]  # dual_regime = headline (NB07 best)
N_BOOT_PAIRED = 5000
N_BOOT_CLUSTER = 4000


# ---------------------------------------------------------------------------
# Shading + per-cell simulation
# ---------------------------------------------------------------------------

def _shade(strategy: str, V: np.ndarray, ax, slot, overall, exch, max_bid: float) -> np.ndarray:
    if strategy == "exchange_optimal":
        cfg = ShadingConfig(strategy="optimal", exchange_conditional=True, max_bid=max_bid)
        return exchange_conditional_shading(V, ax, exch, overall, cfg).bids
    if strategy == "dual_regime":
        cfg = ShadingConfig(max_bid=max_bid)
        return dual_regime_shading(V, overall, slot, cfg).bids
    if strategy == "truthful":
        return np.clip(V, 1.0, max_bid)
    raise ValueError(f"unknown strategy {strategy}")


def _simulate(p, pp, clicks, ax, slot, overall, exch, strategy, max_bid=MAX_BID, cpc=CPC):
    """Run one cell under SECOND-PRICE (canonical, the correct iPinYou mechanism).

    Winners pay the market clearing price (= payprice ≤ bid), NOT their bid. Returns
    (cell_metrics, second-price per-row surplus s_vec). Also reports the first-price surplus
    (`realized_surplus_1p`) from the SAME shaded bids — the historically-buggy headline — for the
    mechanism comparison. Re-won set (bid ≥ payprice) is identical across mechanisms.
    """
    V = compute_impression_values(p, ValueConfig(goal_type="CPC", cpc_target=cpc)).values
    bids = _shade(strategy, V, ax, slot, overall, exch, max_bid)
    res = run_auction_simulation(bids, pp, V, clicks, auction_type="second_price")
    sim = compute_simulation_metrics(res, V, pp, strategy, cpc_target=cpc)
    s_vec = res.clicks.astype(np.float64) * cpc - res.payments  # second-price: pay = market price (pp)
    realized = float(s_vec.sum())
    # first-price counterpart (same bids/wins, pay your bid) — diagnostic only
    realized_1p = float((res.clicks.astype(np.float64) * cpc - bids * res.wins).sum())
    cell = {
        "realized_surplus": realized,            # SECOND-PRICE (canonical)
        "realized_surplus_1p": realized_1p,      # first-price (old buggy headline; diagnostic)
        "sim_total_surplus": sim.total_surplus,  # model-own V (phantom-prone) diagnostic
        "n_wins": sim.n_wins,
        "win_rate": round(sim.win_rate, 4),
        "n_clicks_won": int(res.clicks.sum()),
        "total_spend": sim.total_spend,
        "roi": round(sim.roi, 4),
        "overpayment_ratio": round(sim.overpayment_ratio, 4),
        "mean_V": float(V.mean()),
        "mean_bid": float(bids.mean()),
        "frac_bid_clipped": float((bids >= max_bid - 1e-6).mean()),
    }
    return cell, s_vec


# ---------------------------------------------------------------------------
# Calibration (strategy-independent, per model x variant)
# ---------------------------------------------------------------------------

def _slice_dict(sc) -> dict:
    return {
        "max_abs_bias": round(sc.max_abs_bias, 6),
        "weighted_abs_bias": round(sc.weighted_abs_bias, 6),
        "rows": [
            {"value": r.slice_value, "mean_pred": round(r.mean_pred, 7),
             "mean_true": round(r.mean_true, 7), "bias": round(r.bias, 7), "count": r.count}
            for r in sc.rows
        ],
    }


def _calibration(p, y, ax, adv) -> dict:
    rel = quantile_reliability(y.astype(float), p.astype(float), n_bins=10)
    deciles = [{"bin": b.bin_index, "pred": round(b.mean_pred, 7), "true": round(b.mean_true, 7),
                "ratio": round(b.mean_pred / b.mean_true, 3) if b.mean_true > 0 else None}
               for b in rel.bins]
    return {
        "ieb": round(float(compute_ieb(y, p)), 4),
        "winners_auc": round(float(_numpy_roc_auc(y, p)), 4),
        "quantile_ece": round(float(rel.quantile_ece), 6),
        "deciles": deciles,
        "slices": {
            "adexchange": _slice_dict(slice_calibration(y.astype(float), p.astype(float), ax, "adexchange")),
            "advertiser": _slice_dict(slice_calibration(y.astype(float), p.astype(float), adv, "advertiser")),
        },
    }


def _gap_ci(s_a, s_b, adv) -> dict:
    paired = paired_bootstrap_surplus_gap(s_a, s_b, n_boot=N_BOOT_PAIRED, seed=0)
    clus = cluster_bootstrap_surplus_gap(s_a, s_b, adv, n_boot=N_BOOT_CLUSTER, seed=0)
    return {
        "point": round(paired.point, 1),
        "paired_ci95": [round(paired.ci95_lo, 1), round(paired.ci95_hi, 1)],
        "paired_p_gt_0": round(paired.p_gt_0, 4),
        "cluster_ci95": [round(clus.ci95_lo, 1), round(clus.ci95_hi, 1)],
        "cluster_p_gt_0": round(clus.p_gt_0, 4),
    }


def main() -> None:
    print("Loading recalibrated winners + fair-test covariates ...")
    d = np.load(NPZ)
    idx = d["idx_won"].astype(np.int64)
    y_full = d["y_click_won"].astype(np.int32)

    _, _, test_df, _ = load_feature_splits(
        FAIR, columns=["payprice", "bidprice", "adexchange", "slotprice", "advertiser", "win", "click"]
    )
    pp_all = np.asarray(test_df["payprice"].values, dtype=np.float64)[idx]
    bid_all = np.asarray(test_df["bidprice"].values, dtype=np.float64)[idx]
    ax_all = np.nan_to_num(np.asarray(test_df["adexchange"].values, dtype=np.float64)[idx], nan=-1).astype(np.int64)
    slot_all = np.nan_to_num(np.asarray(test_df["slotprice"].values, dtype=np.float64)[idx], nan=0.0)
    adv_all = np.asarray(test_df["advertiser"].values)[idx]
    win_all = np.asarray(test_df["win"].values, dtype=np.int64)[idx]
    clk_all = np.asarray(test_df["click"].values, dtype=np.int64)[idx]

    # Alignment sanity: idx_won must point at winners, and labels must match the npz.
    assert win_all.all(), "idx_won contains non-winner rows"
    assert np.array_equal(clk_all.astype(np.int32), y_full), "click label mismatch idx vs npz"

    # Same row mask for ALL cells: payprice observed (>0).
    keep = pp_all > 0
    n_drop = int((~keep).sum())
    pp = pp_all[keep]; bid = bid_all[keep]; ax = ax_all[keep]; slot = slot_all[keep]; adv = adv_all[keep]
    clicks = y_full[keep]
    preds = {(m, v): d[f"{m}_{v}"][keep].astype(np.float64) for m, _ in MODELS for v in VARIANTS}
    print(f"  winners kept={keep.sum():,} (dropped {n_drop} payprice==0)  clicks={int(clicks.sum()):,}")

    # P2 sanity: a logged-bid policy under SECOND-price reproduces realized 2p won-only surplus
    # on valid wins (logged_bid >= payprice). payprice>bidprice rows violate 2p and are excluded.
    realized_2p_clean = float(((clicks * CPC - pp) * (bid >= pp)).sum())
    n_anom = int((bid < pp).sum())
    print(f"  P2 sanity: realized 2nd-price won-only surplus (valid wins) = {realized_2p_clean:.3e} "
          f"({n_anom:,} payprice>bidprice rows excluded)")

    overall = load_market_cdf(str(CDF_DIR / "km_cdf_overall.npz"))
    exch = load_exchange_cdfs(str(CDF_DIR))

    # ---- Calibration (strategy-independent) + assertions licensing the decomposition ----
    calibration: Dict[str, Dict[str, dict]] = {}
    mean_V_recal = {}
    for m, key in MODELS:
        calibration[key] = {}
        for v in VARIANTS:
            calibration[key][v] = _calibration(preds[(m, v)], clicks, ax, adv)
        drift = abs(calibration[key]["raw"]["winners_auc"] - calibration[key]["recal"]["winners_auc"])
        assert drift < 5e-3, f"{key} AUC drift {drift} >= 5e-3 — isotonic should preserve ranking"
        mean_V_recal[key] = float(preds[(m, "recal")].mean() * CPC)
    mv = list(mean_V_recal.values())
    assert np.allclose(mv, mv[0], rtol=1e-2), f"recal mean-V not equalized: {mean_V_recal}"
    print(f"  assertions OK: AUC drift <5e-3; mean V_recal equalized ~{mv[0]:.1f} CPM")

    # ---- Main grid: {strategy} x {model} x {raw,recal} ----
    grid: Dict[str, Dict[str, Dict[str, dict]]] = {}
    s_recal: Dict[Tuple[str, str], np.ndarray] = {}  # (strategy, model_key) -> s_vec, recal only
    for strat in STRATEGIES:
        grid[strat] = {}
        print(f"strategy={strat} ...")
        for m, key in MODELS:
            grid[strat][key] = {}
            for v in VARIANTS:
                cell, s_vec = _simulate(preds[(m, v)], pp, clicks, ax, slot, overall, exch, strat)
                # mechanism invariant: paying the market price (2p) can't cost more than paying your bid (1p)
                assert cell["realized_surplus"] >= cell["realized_surplus_1p"] - 1.0, \
                    f"2p surplus < 1p for {strat}/{key}/{v}"
                grid[strat][key][v] = cell
                if v == "recal":
                    s_recal[(strat, key)] = s_vec
            print(f"    {key:7s} raw surplus={grid[strat][key]['raw']['realized_surplus']:.3e}  "
                  f"recal surplus={grid[strat][key]['recal']['realized_surplus']:.3e}")

    # ---- Decomposition ----
    decomposition = {"calibration_contribution": {}, "ranking_contribution_among_recal": {}}
    for strat in STRATEGIES:
        decomposition["calibration_contribution"][strat] = {
            key: {
                "raw_surplus": grid[strat][key]["raw"]["realized_surplus"],
                "recal_surplus": grid[strat][key]["recal"]["realized_surplus"],
                "delta_cal": grid[strat][key]["recal"]["realized_surplus"]
                - grid[strat][key]["raw"]["realized_surplus"],
            }
            for _, key in MODELS
        }
        decomposition["ranking_contribution_among_recal"][strat] = {
            "neural_minus_lgb": _gap_ci(s_recal[(strat, "neural")], s_recal[(strat, "lgb")], adv),
            "neural_minus_lr": _gap_ci(s_recal[(strat, "neural")], s_recal[(strat, "lr")], adv),
        }

    # ---- Per-advertiser surplus gap (headline dual_regime) vs residual IEB ----
    s_neu = s_recal[("dual_regime", "neural")]
    s_lgb = s_recal[("dual_regime", "lgb")]
    p_neu_recal = preds[("escm2wc_dr", "recal")]
    p_lgb_recal = preds[("lgb_ctr_all", "recal")]
    per_adv = {}
    for a in np.unique(adv):
        ma = adv == a
        per_adv[str(int(a))] = {
            "neural_surplus": float(s_neu[ma].sum()),
            "lgb_surplus": float(s_lgb[ma].sum()),
            "neural_minus_lgb": float((s_neu[ma] - s_lgb[ma]).sum()),
            "neural_adv_resid_ieb": round(float(compute_ieb(clicks[ma], p_neu_recal[ma])), 4),
            "lgb_adv_resid_ieb": round(float(compute_ieb(clicks[ma], p_lgb_recal[ma])), 4),
        }

    # ---- Sensitivity: among-recal dual_regime gap under CPC + max_bid sweeps ----
    sensitivity = {"cpc_target": {}, "max_bid": {}}
    for cpc in (1e5, 2e5, 4e5):
        cn, s_sn = _simulate(preds[("escm2wc_dr", "recal")], pp, clicks, ax, slot, overall, exch, "dual_regime", cpc=cpc)
        cl, s_sl = _simulate(preds[("lgb_ctr_all", "recal")], pp, clicks, ax, slot, overall, exch, "dual_regime", cpc=cpc)
        gap = paired_bootstrap_surplus_gap(s_sn, s_sl, n_boot=2000, seed=0)
        sensitivity["cpc_target"][f"{cpc:.0e}"] = {
            "neural_minus_lgb_point": round(gap.point, 1), "p_gt_0": round(gap.p_gt_0, 4),
            "neural_surplus": cn["realized_surplus"], "lgb_surplus": cl["realized_surplus"]}
    for mb in (300.0, 600.0):
        _, s_sn = _simulate(preds[("escm2wc_dr", "recal")], pp, clicks, ax, slot, overall, exch, "dual_regime", max_bid=mb)
        _, s_sl = _simulate(preds[("lgb_ctr_all", "recal")], pp, clicks, ax, slot, overall, exch, "dual_regime", max_bid=mb)
        gap = paired_bootstrap_surplus_gap(s_sn, s_sl, n_boot=2000, seed=0)
        sensitivity["max_bid"][f"{mb:.0f}"] = {
            "neural_minus_lgb_point": round(gap.point, 1), "p_gt_0": round(gap.p_gt_0, 4)}

    # ---- Mechanism comparison (first- vs second-price; the bug correction) ----
    def _neu_recal_surplus(strat, mech):  # mech: "realized_surplus"(2p) | "realized_surplus_1p"
        return grid[strat]["neural"]["recal"][mech]
    best_2p = max(STRATEGIES, key=lambda s: _neu_recal_surplus(s, "realized_surplus"))
    best_1p = max(STRATEGIES, key=lambda s: _neu_recal_surplus(s, "realized_surplus_1p"))
    mechanism_comparison = {
        "note": "stage_b2 originally ran FIRST-price on second-price data (bug). Canonical headline is "
                "now SECOND-price (winners pay the market clearing price = payprice ≤ bid). Under 2p, "
                "surplus is higher and the strategy ranking reverses (truthful = 2p-optimal).",
        "best_strategy_2p": best_2p, "best_strategy_1p": best_1p,
        "strategy_rank_reversed": bool(best_2p != best_1p),
        "per_strategy_recal": {
            s: {k: {"surplus_2p": round(grid[s][k]["recal"]["realized_surplus"], 1),
                    "surplus_1p": round(grid[s][k]["recal"]["realized_surplus_1p"], 1)}
                for k in ("neural", "lr", "lgb")}
            for s in STRATEGIES},
        "neural_minus_lgb_1p_point": {
            s: round(grid[s]["neural"]["recal"]["realized_surplus_1p"]
                     - grid[s]["lgb"]["recal"]["realized_surplus_1p"], 1) for s in STRATEGIES},
        "p2_realized_2p_won_only_surplus": round(realized_2p_clean, 1),
    }

    # ---- Verdict (headline = the SECOND-price-optimal strategy) ----
    headline_strat = best_2p
    headline = decomposition["ranking_contribution_among_recal"][headline_strat]["neural_minus_lgb"]
    sign_consistent = all(
        decomposition["ranking_contribution_among_recal"][s]["neural_minus_lgb"]["point"] > 0
        for s in STRATEGIES
    )
    cpc_sign_stable = all(v["neural_minus_lgb_point"] > 0 for v in sensitivity["cpc_target"].values())
    supported = (headline["cluster_ci95"][0] > 0) and sign_consistent and cpc_sign_stable
    verdict = {
        "auction": "second_price (corrected from first-price bug)",
        "thesis_supported": bool(supported),
        "headline_strategy": headline_strat,
        "headline_neural_minus_lgb": headline,
        "dual_regime_neural_minus_lgb": decomposition["ranking_contribution_among_recal"]["dual_regime"]["neural_minus_lgb"],
        "sign_consistent_across_strategies": bool(sign_consistent),
        "cpc_sweep_sign_stable": bool(cpc_sign_stable),
        "rule": f"SUPPORTED iff neural-lgb advertiser-cluster 95% CI excludes 0 under the 2p-optimal "
                f"strategy ({headline_strat}) AND sign-consistent across 3 strategies AND CPC-sweep sign-stable.",
        "scope_limitation": "won-only surplus cannot test debiasing's value on LOST inventory "
                            "(censored payprice); conservative lower bound. Full-inventory view: policy_value.py.",
    }

    out = {
        "_meta": {
            "auction": "second_price (canonical); first-price retained in mechanism_comparison",
            "n_winners_kept": int(keep.sum()), "n_dropped_payprice0": n_drop,
            "n_clicks_won": int(clicks.sum()), "cpc_target": CPC, "max_bid": MAX_BID,
            "strategies": STRATEGIES, "mean_V_recal_equalized": mean_V_recal,
            "n_boot": {"paired": N_BOOT_PAIRED, "cluster": N_BOOT_CLUSTER},
            "row_mask": "payprice>0 (same mask all cells)",
        },
        "verdict": verdict,
        "mechanism_comparison": mechanism_comparison,
        "grid": grid,
        "decomposition": decomposition,
        "per_advertiser_dual_regime": per_adv,
        "sensitivity": sensitivity,
        "calibration": calibration,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    OUT_MD.write_text(_render_md(out))
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")
    print(f"\n[SECOND-PRICE] best 2p strategy={best_2p} (1p best was {best_1p}; reversed={best_2p!=best_1p})")
    print(f"VERDICT: thesis_supported={supported}  {headline_strat} neural−lgb={headline['point']:.3e} "
          f"cluster_CI={headline['cluster_ci95']} p={headline['cluster_p_gt_0']}")


def _render_md(out: dict) -> str:
    L: List[str] = []
    v = out["verdict"]; meta = out["_meta"]
    L.append("# Stage B2 — Decision-level value of debiasing (SECOND-PRICE, realized surplus + slice calibration)\n")
    mc = out["mechanism_comparison"]
    L.append("## TL;DR VERDICT\n")
    h = v["headline_neural_minus_lgb"]
    L.append(
        f"- **Auction = SECOND-PRICE (corrected from a first-price bug).** Winners pay the market "
        f"clearing price (≤ bid), not their bid. Strategy ranking reversed: **2p-optimal = "
        f"`{mc['best_strategy_2p']}`** (1p-best was `{mc['best_strategy_1p']}`).\n"
        f"- **Thesis {'SUPPORTED' if v['thesis_supported'] else 'NOT SUPPORTED'}.** "
        f"Among-recal (mean V equalized ⇒ pure ranking + residual slice-calibration), headline "
        f"**{v['headline_strategy']}** neural−lgb realized-surplus gap = **{h['point']:.3e}** "
        f"(paired CI {h['paired_ci95']}, p={h['paired_p_gt_0']}; "
        f"advertiser-cluster CI {h['cluster_ci95']}, p={h['cluster_p_gt_0']}). "
        f"(dual_regime gap {v['dual_regime_neural_minus_lgb']['point']:.3e}, "
        f"cluster CI {v['dual_regime_neural_minus_lgb']['cluster_ci95']}.)\n"
        f"- Sign-consistent across 3 strategies: **{v['sign_consistent_across_strategies']}**; "
        f"CPC-sweep sign-stable: **{v['cpc_sweep_sign_stable']}**.\n"
        f"- Rule: {v['rule']}\n"
        f"- **Scope:** {v['scope_limitation']}\n"
        f"- Setup: winners kept {meta['n_winners_kept']:,} (clicks {meta['n_clicks_won']:,}), "
        f"CPC={meta['cpc_target']:.0f}, mean V_recal equalized = "
        f"{ {k: round(x,1) for k,x in meta['mean_V_recal_equalized'].items()} }.\n"
    )

    L.append("\n## Mechanism comparison: second-price (correct) vs first-price (old bug), recal surplus\n")
    L.append("| strategy | model | surplus 2p (correct) | surplus 1p (buggy) | neural−lgb 1p |")
    L.append("|---|---|---|---|---|")
    for s in out["_meta"]["strategies"]:
        for k in ("neural", "lr", "lgb"):
            ps = mc["per_strategy_recal"][s][k]
            extra = f"{mc['neural_minus_lgb_1p_point'][s]:.3e}" if k == "neural" else ""
            L.append(f"| {s} | {k} | {ps['surplus_2p']:.3e} | {ps['surplus_1p']:.3e} | {extra} |")
    L.append(f"\n*Under second-price you pay the market price, so surplus is higher and `{mc['best_strategy_2p']}` "
             f"is optimal (vs `{mc['best_strategy_1p']}` under first-price). The old first-price headline "
             f"(neural−lgb dual_regime ≈ +1.79e7) is superseded by the second-price numbers below.*\n")

    L.append("\n## Among-recal ranking gap (neural − baseline), by strategy [SECOND-PRICE]\n")
    L.append("| strategy | neural−lgb point | paired CI95 | p>0 | cluster CI95 | p>0 |  neural−lr point |")
    L.append("|---|---|---|---|---|---|---|")
    for s in out["_meta"]["strategies"]:
        g = out["decomposition"]["ranking_contribution_among_recal"][s]
        nl, nr = g["neural_minus_lgb"], g["neural_minus_lr"]
        L.append(f"| {s} | {nl['point']:.3e} | {nl['paired_ci95']} | {nl['paired_p_gt_0']} | "
                 f"{nl['cluster_ci95']} | {nl['cluster_p_gt_0']} | {nr['point']:.3e} |")

    L.append("\n## Realized surplus grid (raw → recal), by strategy\n")
    L.append("| strategy | model | raw surplus | recal surplus | Δ_cal | recal win_rate | recal clicks |")
    L.append("|---|---|---|---|---|---|---|")
    for s in out["_meta"]["strategies"]:
        for key in ("neural", "lr", "lgb"):
            c = out["grid"][s][key]
            dc = out["decomposition"]["calibration_contribution"][s][key]
            L.append(f"| {s} | {key} | {c['raw']['realized_surplus']:.3e} | "
                     f"{c['recal']['realized_surplus']:.3e} | {dc['delta_cal']:.3e} | "
                     f"{c['recal']['win_rate']} | {c['recal']['n_clicks_won']} |")

    L.append("\n## Per-advertiser (dual_regime, recal): surplus gap vs residual calibration\n")
    L.append("| advertiser | neural−lgb surplus | neural resid IEB | lgb resid IEB |")
    L.append("|---|---|---|---|")
    for a, r in sorted(out["per_advertiser_dual_regime"].items(), key=lambda kv: -kv[1]["neural_minus_lgb"]):
        L.append(f"| {a} | {r['neural_minus_lgb']:.3e} | {r['neural_adv_resid_ieb']} | {r['lgb_adv_resid_ieb']} |")
    L.append("\n*Reading: if neural's surplus edge concentrates in HIGH-residual-IEB advertisers, "
             "that is decision-level evidence for the slice-calibration mechanism global recal can't "
             "fix; if it sits in low-residual advertisers, the edge is pure ranking.*\n")

    L.append("\n## Sensitivity (dual_regime, among-recal neural−lgb)\n")
    L.append("| knob | value | neural−lgb point | p>0 |")
    L.append("|---|---|---|---|")
    for cpc, r in out["sensitivity"]["cpc_target"].items():
        L.append(f"| CPC | {cpc} | {r['neural_minus_lgb_point']:.3e} | {r['p_gt_0']} |")
    for mb, r in out["sensitivity"]["max_bid"].items():
        L.append(f"| max_bid | {mb} | {r['neural_minus_lgb_point']:.3e} | {r['p_gt_0']} |")

    L.append(f"\n## Files\n- `{OUT_JSON}` — full grid, decomposition, bootstrap CIs, calibration, sensitivity.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
