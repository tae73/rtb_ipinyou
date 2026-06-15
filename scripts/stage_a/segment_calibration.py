"""Stage 5 — segment-aware (per-advertiser) calibration: the lever no global map fixes.

Global cross-fit isotonic (Stages 1-3) zeroes the AGGREGATE winners-pCTR bias but leaves a
per-advertiser residual (neural max 0.226 @adv 3476). A single monotone map cannot correct
advertiser-specific level offsets. This probe fits a SEPARATE leak-free isotonic map PER ADVERTISER
(`segment_cross_fit_isotonic`) and asks:

  (a) does it drive per-advertiser residual IEB -> ~0 (where global cannot)?
  (b) does it IMPROVE realized bidding surplus vs the global map, or is the residual decision-neutral?

For each model {escm2wc_dr (neural), lr_ctr_all, lgb_ctr_all} on the fair-test winners, compares 3
regimes: raw / global (cross-fit isotonic) / segment (per-advertiser cross-fit isotonic). Reports
per-advertiser residual IEB, within-advertiser AUC (monotone per-seg map -> preserved) and global
winners AUC (may shift), and the segment-vs-global realized-surplus gap with paired + advertiser-
cluster bootstrap CIs.

Ranking caveat: per-advertiser maps are monotone WITHIN an advertiser (within-adv AUC preserved) but
not globally, so global cross-advertiser AUC can change — expected, not a bug.

Reads: results/stage_a/recalibrated_winners_preds.npz (idx_won, y_click_won, {model}_raw),
fair_baseline_preds.npz (advertiser), features_fair/test.parquet, market_price_cdf/*.
Writes: results/stage_a/segment_calibration.{json,_summary.md} + segment_recalibrated_winners_preds.npz.

Usage: python scripts/stage_a/segment_calibration.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

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
    paired_bootstrap_surplus_gap,
    run_auction_simulation,
)
from src.bidding.value import ValueConfig, compute_impression_values
from src.features.engineering import load_feature_splits
from src.metrics.calibration import (
    cross_fit_isotonic,
    quantile_reliability,
    segment_cross_fit_isotonic,
)
from src.metrics.evaluation import _numpy_roc_auc, compute_ieb

PROJECT = Path("/home/mail-agent/project/rtb_ipinyou")
NPZ = PROJECT / "results/stage_a/recalibrated_winners_preds.npz"
BASE = PROJECT / "results/stage_a/fair_baseline_preds.npz"
FAIR = PROJECT / "data/ipinyou/prediction/features_fair"
CDF_DIR = PROJECT / "results/market_price_cdf"
OUT_JSON = PROJECT / "results/stage_a/segment_calibration.json"
OUT_MD = PROJECT / "results/stage_a/segment_calibration_summary.md"
OUT_NPZ = PROJECT / "results/stage_a/segment_recalibrated_winners_preds.npz"

CPC = 200_000.0
MAX_BID = 300.0
ADVS = ["1458", "3358", "3386", "3427", "3476"]
GLOBAL_CEILING = 0.226   # global-isotonic max per-advertiser residual (neural) to beat
MODELS = [("escm2wc_dr", "neural"), ("lr_ctr_all", "lr"), ("lgb_ctr_all", "lgb")]
STRATEGIES = ["exchange_optimal", "dual_regime", "truthful"]


def _shade(strategy, V, ax, slot, overall, exch, max_bid=MAX_BID):
    if strategy == "exchange_optimal":
        return exchange_conditional_shading(
            V, ax, exch, overall, ShadingConfig(strategy="optimal", exchange_conditional=True, max_bid=max_bid)
        ).bids
    if strategy == "dual_regime":
        return dual_regime_shading(V, overall, slot, ShadingConfig(max_bid=max_bid)).bids
    if strategy == "truthful":
        return np.clip(V, 1.0, max_bid)
    raise ValueError(strategy)


def _surplus_vec(pred, pp, clicks, ax, slot, overall, exch, strategy, cpc=CPC):
    V = compute_impression_values(pred, ValueConfig(goal_type="CPC", cpc_target=cpc)).values
    bids = _shade(strategy, V, ax, slot, overall, exch)
    res = run_auction_simulation(bids, pp, V, clicks, auction_type="first_price")
    s_vec = res.clicks.astype(np.float64) * cpc - res.payments
    return s_vec, float(s_vec.sum())


def _cal_metrics(pred, y, adv) -> dict:
    rel = quantile_reliability(y.astype(float), pred.astype(float), n_bins=10)
    per_adv_ieb, per_adv_auc, weights = {}, {}, {}
    for a in np.unique(adv):
        m = adv == a
        per_adv_ieb[str(int(a))] = round(float(compute_ieb(y[m], pred[m])), 4)
        if 0 < y[m].sum() < m.sum():
            per_adv_auc[str(int(a))] = round(float(_numpy_roc_auc(y[m], pred[m])), 4)
            weights[str(int(a))] = int(m.sum())
    within = (sum(per_adv_auc[a] * weights[a] for a in per_adv_auc) / sum(weights.values())
              if weights else None)
    return {
        "global_ieb": round(float(compute_ieb(y, pred)), 4),
        "global_auc": round(float(_numpy_roc_auc(y, pred)), 4),
        "within_adv_auc": round(within, 4) if within is not None else None,
        "quantile_ece": round(float(rel.quantile_ece), 6),
        "per_advertiser_ieb": per_adv_ieb,
        "max_per_adv_ieb": round(max(per_adv_ieb.values()), 4),
        "per_advertiser_auc": per_adv_auc,
    }


def _gap(s_a, s_b, adv) -> dict:
    p = paired_bootstrap_surplus_gap(s_a, s_b, n_boot=3000, seed=0)
    c = cluster_bootstrap_surplus_gap(s_a, s_b, adv, n_boot=3000, seed=0)
    return {"point": round(p.point, 1),
            "paired_ci95": [round(p.ci95_lo, 1), round(p.ci95_hi, 1)], "paired_p_gt_0": round(p.p_gt_0, 4),
            "cluster_ci95": [round(c.ci95_lo, 1), round(c.ci95_hi, 1)], "cluster_p_gt_0": round(c.p_gt_0, 4)}


def main() -> None:
    print("Loading winners predictions + covariates ...")
    d = np.load(NPZ)
    idx = d["idx_won"].astype(np.int64)
    y = d["y_click_won"].astype(np.float64)
    adv_full = np.load(BASE)["advertiser"]
    adv = adv_full[idx]

    _, _, test_df, _ = load_feature_splits(FAIR, columns=["payprice", "adexchange", "slotprice"])
    pp = np.asarray(test_df["payprice"].values, np.float64)[idx]
    ax = np.nan_to_num(np.asarray(test_df["adexchange"].values, np.float64)[idx], nan=-1).astype(np.int64)
    slot = np.nan_to_num(np.asarray(test_df["slotprice"].values, np.float64)[idx], nan=0.0)
    overall = load_market_cdf(str(CDF_DIR / "km_cdf_overall.npz"))
    exch = load_exchange_cdfs(str(CDF_DIR))
    print(f"  winners={len(idx):,}  clicks={int(y.sum()):,}  advertisers={sorted(set(adv.tolist()))}")

    # Surplus subset: payprice>0 (same mask all cells).
    keep = pp > 0
    pk, axk, sk, advk, ck = pp[keep], ax[keep], slot[keep], adv[keep], y[keep]

    results: Dict[str, dict] = {}
    npz_out: Dict[str, np.ndarray] = {"idx_won": idx, "y_click_won": y.astype(np.int8)}
    for m, key in MODELS:
        raw = d[f"{m}_raw"].astype(np.float64)
        glob = cross_fit_isotonic(raw, y, n_folds=5, seed=0)
        seg = segment_cross_fit_isotonic(raw, y, adv, n_folds=5, seed=0, min_positives=50)
        regimes = {"raw": raw, "global": glob, "segment": seg}
        cal = {rk: _cal_metrics(rp, y, adv) for rk, rp in regimes.items()}

        # Surplus: segment vs global (both globally ~calibrated; segment additionally per-adv).
        surplus = {}
        for strat in STRATEGIES:
            s_glob, tot_g = _surplus_vec(glob[keep], pk, ck, axk, sk, overall, exch, strat)
            s_seg, tot_s = _surplus_vec(seg[keep], pk, ck, axk, sk, overall, exch, strat)
            entry = _gap(s_seg, s_glob, advk)  # segment - global
            entry["global_surplus"] = round(tot_g, 1)
            entry["segment_surplus"] = round(tot_s, 1)
            if strat == "dual_regime":
                # per-advertiser surplus delta (where does seg help/hurt?)
                entry["per_adv_delta"] = {
                    str(int(a)): round(float((s_seg[advk == a] - s_glob[advk == a]).sum()), 1)
                    for a in np.unique(advk)}
            surplus[strat] = entry

        results[key] = {"calibration": cal, "surplus_segment_vs_global": surplus}
        npz_out[f"{m}_global"] = glob.astype(np.float32)
        npz_out[f"{m}_segment"] = seg.astype(np.float32)
        print(f"  {key:7s} per-adv max IEB: raw {cal['raw']['max_per_adv_ieb']} | "
              f"global {cal['global']['max_per_adv_ieb']} | segment {cal['segment']['max_per_adv_ieb']}  "
              f"(within-adv AUC raw {cal['raw']['within_adv_auc']} -> seg {cal['segment']['within_adv_auc']})")

    # ---- Verdict (judged on the neural model, dual_regime headline) ----
    neu = results["neural"]
    seg_max = neu["calibration"]["segment"]["max_per_adv_ieb"]
    glob_max = neu["calibration"]["global"]["max_per_adv_ieb"]
    g = neu["surplus_segment_vs_global"]["dual_regime"]
    residual_fixed = seg_max < 0.10 and seg_max < glob_max
    ci_lo, ci_hi = g["cluster_ci95"]
    if residual_fixed and ci_lo > 0:
        v = "WIN"; stmt = (f"Per-advertiser calibration drives neural max residual {glob_max}->{seg_max} "
                           f"AND improves bidding surplus (segment-global dual_regime +{g['point']:.3e}, "
                           f"cluster CI {g['cluster_ci95']} excludes 0).")
    elif residual_fixed and ci_lo <= 0 <= ci_hi:
        v = "CALIBRATION_WIN_DECISION_NEUTRAL"; stmt = (
            f"Per-advertiser calibration drives neural max residual {glob_max}->{seg_max} (fixes the "
            f"one gap no global map could), BUT realized-surplus gap vs global is decision-neutral "
            f"(cluster CI {g['cluster_ci95']} contains 0) — the residual was a calibration nicety, not "
            f"a bidding lever on won-only inventory.")
    elif residual_fixed and ci_hi < 0:
        v = "CALIBRATION_WIN_SURPLUS_WORSE"; stmt = (
            f"Per-advertiser calibration fixes the residual ({glob_max}->{seg_max}) but realized surplus "
            f"is WORSE than global (cluster CI {g['cluster_ci95']} < 0) — over-fitting per-advertiser maps "
            f"hurts won-only bidding.")
    else:
        v = "NO_RESIDUAL_FIX"; stmt = f"Segment map did not drive residual below 0.10 ({seg_max})."
    verdict = {"verdict": v, "statement": stmt, "neural_global_max_resid": glob_max,
               "neural_segment_max_resid": seg_max, "global_ceiling": GLOBAL_CEILING}
    print(f"\nVERDICT [{v}]: {stmt}")

    OUT_JSON.write_text(json.dumps({"verdict": verdict, "models": results,
                                    "_meta": {"cpc_target": CPC, "advertisers": ADVS,
                                              "n_winners": int(len(idx)), "n_clicks": int(y.sum())}}, indent=2))
    np.savez_compressed(OUT_NPZ, **npz_out)
    OUT_MD.write_text(_render_md(verdict, results))
    print(f"Wrote {OUT_JSON}\nWrote {OUT_NPZ}\nWrote {OUT_MD}")


def _render_md(verdict: dict, results: dict) -> str:
    L: List[str] = ["# Stage 5 — Segment-aware (per-advertiser) calibration\n", "## TL;DR VERDICT\n"]
    L.append(f"- **[{verdict['verdict']}]** {verdict['statement']}\n")
    L.append(f"- Global-isotonic per-advertiser ceiling to beat = **{verdict['global_ceiling']}** "
             f"(neural). Segment map: {verdict['neural_global_max_resid']} → "
             f"**{verdict['neural_segment_max_resid']}**.\n")

    L.append("\n## Per-advertiser residual IEB (raw / global / segment), by model\n")
    L.append("| model | regime | 1458 | 3358 | 3386 | 3427 | 3476 | max | within-adv AUC | global AUC |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, r in results.items():
        for rk in ("raw", "global", "segment"):
            c = r["calibration"][rk]; pa = c["per_advertiser_ieb"]
            L.append(f"| {key} | {rk} | " + " | ".join(str(pa.get(a, "—")) for a in ADVS) +
                     f" | {c['max_per_adv_ieb']} | {c['within_adv_auc']} | {c['global_auc']} |")

    L.append("\n## Surplus: segment − global (does fixing the residual help bidding?)\n")
    L.append("| model | strategy | global surplus | segment surplus | seg−glob | cluster CI95 | p>0 |")
    L.append("|---|---|---|---|---|---|---|")
    for key, r in results.items():
        for strat, s in r["surplus_segment_vs_global"].items():
            L.append(f"| {key} | {strat} | {s['global_surplus']:.3e} | {s['segment_surplus']:.3e} | "
                     f"{s['point']:.3e} | {s['cluster_ci95']} | {s['cluster_p_gt_0']} |")

    nd = results["neural"]["surplus_segment_vs_global"]["dual_regime"].get("per_adv_delta", {})
    if nd:
        L.append("\n## Neural per-advertiser surplus delta (segment − global, dual_regime)\n")
        L.append("| advertiser | seg−glob surplus | (residual was: global → segment) |")
        L.append("|---|---|---|")
        gp = results["neural"]["calibration"]["global"]["per_advertiser_ieb"]
        sp = results["neural"]["calibration"]["segment"]["per_advertiser_ieb"]
        for a in sorted(nd, key=lambda x: -nd[x]):
            L.append(f"| {a} | {nd[a]:.3e} | {gp.get(a)} → {sp.get(a)} |")

    L.append("\n*Within-advertiser AUC is the ranking check (monotone per-advertiser maps preserve it); "
             "global AUC may shift because per-advertiser maps are not globally monotone — expected.*")
    L.append("\n## Files\n- `results/stage_a/segment_calibration.json`, "
             "`segment_recalibrated_winners_preds.npz`.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
