"""Stage 6 — full-inventory policy-value projection (escape won-only), de-risk ladder P0→P3.

Estimates the SECOND-PRICE value of truthful bidding (bid = V = pCTR·CPC) over ALL 19.4M fair-test
bids for neural vs LR/LGB, decomposed into V_exact (observable) + V_model (extrapolated increment
where the policy bids above the logged flat bid into the censored region). See
`src/bidding/policy_value.py` for the exact/modeled decomposition rationale.

Probe ladder (each narrows the surviving claim):
  P0  action-support audit: fraction of value that is MODELED (extrapolated) — how much rides on F(b|x)?
  P1  market-model calibration: does segment F(b|x) match realized win-rate at the 8 logged bid levels?
  P2  recover the observable: estimator on the LOGGED policy == realized second-price won-only surplus;
      also quantify the first→second-price correction vs the (buggy first-price) Stage-B2 baseline.
  P3  full V(π): neural vs LR/LGB, V_exact/V_model split + advertiser-cluster CI on the gap.

Honest framing: NOT off-policy evaluation (deterministic flat logging). Structural, mostly-observable
projection; the modeled component is bounded and reported separately.

Usage: python scripts/stage_a/policy_value.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.bidding.policy_value import MarketModel, project_policy_value
from src.bidding.simulator import cluster_bootstrap_surplus_gap, paired_bootstrap_surplus_gap
from src.metrics.calibration import fit_isotonic
from src.win_rate.nonparametric import wilson_ci
from src.win_rate.survival import estimate_market_cdf_km

PROJECT = Path("/home/mail-agent/project/rtb_ipinyou")
NEURAL = PROJECT / "results/models/escm2wc_dr_fair_posw"
BASE = PROJECT / "results/stage_a/fair_baseline_preds.npz"
FAIR = PROJECT / "data/ipinyou/prediction/features_fair"
OUT_JSON = PROJECT / "results/stage_a/policy_value.json"
OUT_MD = PROJECT / "results/stage_a/policy_value_summary.md"

CPC = 200_000.0
MAX_BID = 300.0
PRICE_GRID = np.linspace(0.0, 300.0, 301)


def _build_market_model(bidprice, payprice, win, seg_keys) -> MarketModel:
    """Per-segment KM market-price CDF F(b|seg) (exchange × floor-regime), + overall default."""
    cdf_by_seg: Dict[str, np.ndarray] = {}
    for k in np.unique(seg_keys):
        m = seg_keys == k
        if int(m.sum()) < 5000:
            continue
        r = estimate_market_cdf_km(bidprice[m], payprice[m], win[m], price_grid=PRICE_GRID)
        cdf_by_seg[str(k)] = r.cdf
    overall = estimate_market_cdf_km(bidprice, payprice, win, price_grid=PRICE_GRID)
    cdf_by_seg["__default__"] = overall.cdf
    return MarketModel(PRICE_GRID, cdf_by_seg)


def main() -> None:
    print("Loading predictions + covariates ...")
    neural = np.load(glob.glob(str(NEURAL / "*test_predictions*.npz"))[0])
    base = np.load(BASE)
    yw = neural["y_win"].astype(bool)
    yc = neural["y_click"].astype(np.float64)

    import pandas as pd
    df = pd.read_parquet(FAIR / "test.parquet",
                         columns=["bidprice", "payprice", "win", "adexchange", "slotprice", "advertiser"])
    bidprice = df["bidprice"].to_numpy(np.float64)
    payprice = df["payprice"].to_numpy(np.float64)
    win = df["win"].to_numpy(bool)
    adx = np.nan_to_num(df["adexchange"].to_numpy(np.float64), nan=-1).astype(np.int64)
    slot = np.nan_to_num(df["slotprice"].to_numpy(np.float64), nan=0.0)
    adv = df["advertiser"].to_numpy()
    assert np.array_equal(win, yw), "win mismatch between npz and parquet"

    # market segment = exchange × floor-regime
    floor_reg = (slot > 0).astype(int)
    seg_keys = np.char.add(np.char.add(adx.astype(str), "_"), floor_reg.astype(str))

    # calibrated pCTR for ALL rows: fit global isotonic on winners, apply everywhere
    iso_neu = fit_isotonic(neural["p_ctr"][yw].astype(np.float64), yc[yw])
    pctr = {"neural": iso_neu.transform(neural["p_ctr"].astype(np.float64))}
    for key, col in (("lr", "lr_p_all"), ("lgb", "lgb_p_all")):
        iso = fit_isotonic(base[col][yw].astype(np.float64), yc[yw])
        pctr[key] = iso.transform(base[col].astype(np.float64))
    print(f"  rows={len(win):,} winrate={win.mean():.3f} clicks={int(yc.sum()):,} "
          f"market segments={len(np.unique(seg_keys))}")

    print("Building market model F(b|x) (KM per exchange×floor) ...")
    mm = _build_market_model(bidprice, payprice, win, seg_keys)

    out: Dict = {"_meta": {"cpc": CPC, "n": int(len(win)), "winrate": float(win.mean()),
                           "n_clicks": int(yc.sum())}}

    # ---- P2 (do first: validates the exact path): logged policy recovers realized 2nd-price surplus ----
    logged_res = project_policy_value(pctr["neural"], bidprice, bidprice, win, payprice, yc,
                                      seg_keys, mm, cpc=CPC)
    clean = win & (payprice <= bidprice)                       # valid second-price wins
    realized_2p_clean = float(((yc * CPC - payprice) * clean).sum())
    realized_2p_all = float(((yc * CPC - payprice) * win).sum())
    n_anom = int((win & (payprice > bidprice)).sum())
    out["P2_recover_observable"] = {
        "logged_policy_value": round(logged_res.total, 1),
        "realized_2p_surplus_clean": round(realized_2p_clean, 1),
        "realized_2p_surplus_all": round(realized_2p_all, 1),
        "match_clean": bool(abs(logged_res.total - realized_2p_clean) < max(1.0, 1e-7 * abs(realized_2p_clean))),
        "n_payprice_gt_bidprice_anomalies": n_anom,
        "frac_anomalies": round(n_anom / int(win.sum()), 4),
        "anomaly_surplus_excluded": round(realized_2p_all - realized_2p_clean, 1),
        "note": "logged policy bids==logged bids → all EXACT; equals realized 2nd-price surplus on the "
                "valid (payprice<=bidprice) won rows. The payprice>bidprice rows VIOLATE second-price "
                "(iPinYou data quirk) and are correctly excluded — a data-quality finding.",
    }
    print(f"  P2: logged-policy value {logged_res.total:.3e} vs realized-clean {realized_2p_clean:.3e} "
          f"(match={out['P2_recover_observable']['match_clean']}); "
          f"{n_anom:,} payprice>bidprice anomalies excluded ({realized_2p_all-realized_2p_clean:.2e})")

    # ---- Truthful policy (second-price optimum) per model: bid = clip(V, 1, 300) ----
    results: Dict[str, dict] = {}
    s_vecs: Dict[str, np.ndarray] = {}
    for key in ("neural", "lr", "lgb"):
        V = pctr[key] * CPC
        bid = np.clip(V, 1.0, MAX_BID)
        r = project_policy_value(pctr[key], bid, bidprice, win, payprice, yc, seg_keys, mm, cpc=CPC)
        results[key] = {
            "total": round(r.total, 1), "v_exact": round(r.v_exact, 1), "v_model": round(r.v_model, 1),
            "frac_value_modeled": round(r.frac_value_modeled, 4),
            "frac_rows_extrapolated": round(r.frac_rows_extrapolated, 5),
            "n_winnable_exact": r.n_winnable_exact,
            "frac_bid_at_ceiling": float((bid >= MAX_BID - 1e-6).mean()),
            "mean_bid": round(float(bid.mean()), 2),
        }
        s_vecs[key] = r.s_vec
        print(f"  {key:6s} truthful V(π)={r.total:.3e}  exact={r.v_exact:.3e} model={r.v_model:.3e} "
              f"(modeled {100*r.frac_value_modeled:.1f}% of |value|, {100*r.frac_rows_extrapolated:.3f}% rows)")

    # ---- P0: action-support / how much rides on the market model ----
    out["P0_support_audit"] = {
        key: {"frac_value_modeled": results[key]["frac_value_modeled"],
              "frac_rows_extrapolated": results[key]["frac_rows_extrapolated"],
              "frac_bid_at_ceiling": round(results[key]["frac_bid_at_ceiling"], 4)}
        for key in results}

    # ---- P1: market-model calibration at the 8 logged bid levels, per segment ----
    levels = np.unique(bidprice)
    cal_rows = []
    for seg in np.unique(seg_keys):
        ms = seg_keys == seg
        kk = mm._key(str(seg))
        for lvl in levels:
            mm2 = ms & (bidprice == lvl)
            n = int(mm2.sum())
            if n < 2000:
                continue
            realized = float(win[mm2].mean())
            lo, hi = wilson_ci(np.array([win[mm2].sum()]), np.array([n]))
            F_pred = float(np.interp(lvl, mm.grid, mm._cdf[kk]))
            cal_rows.append({"seg": str(seg), "bid": float(lvl), "n": n,
                             "realized_wr": round(realized, 4), "F_pred": round(F_pred, 4),
                             "in_ci": bool(lo[0] <= F_pred <= hi[0])})
    in_ci_frac = float(np.mean([r["in_ci"] for r in cal_rows])) if cal_rows else None
    out["P1_market_calibration"] = {"cells": cal_rows, "frac_in_wilson_ci": round(in_ci_frac, 3) if in_ci_frac is not None else None,
                                    "note": "F(b|x) vs realized win-rate at logged bid levels (≤300 only)."}
    print(f"  P1: market-model F(b|x) in Wilson CI on {in_ci_frac:.0%} of {len(cal_rows)} (segment×bid) cells")

    # ---- P3: full-inventory value gap neural − baseline, with cluster CI ----
    gaps = {}
    for comp in ("lr", "lgb"):
        pg = paired_bootstrap_surplus_gap(s_vecs["neural"], s_vecs[comp], n_boot=3000, seed=0)
        cg = cluster_bootstrap_surplus_gap(s_vecs["neural"], s_vecs[comp], adv, n_boot=3000, seed=0)
        gaps[f"neural_minus_{comp}"] = {
            "point": round(pg.point, 1),
            "paired_ci95": [round(pg.ci95_lo, 1), round(pg.ci95_hi, 1)], "paired_p_gt_0": round(pg.p_gt_0, 4),
            "cluster_ci95": [round(cg.ci95_lo, 1), round(cg.ci95_hi, 1)], "cluster_p_gt_0": round(cg.p_gt_0, 4)}
    out["P3_full_inventory_gap"] = gaps
    out["models_truthful"] = results

    # ---- Verdict ----
    nl = gaps["neural_minus_lgb"]
    max_modeled = max(results[k]["frac_value_modeled"] for k in results)
    p2_ok = out["P2_recover_observable"]["match_clean"]
    verdict = {
        "p2_recovers_observable": p2_ok,
        "max_frac_value_modeled": round(max_modeled, 4),
        "headline_neural_minus_lgb": nl,
        "claim": (
            f"Full-inventory second-price truthful-bidding value: neural−lgb = {nl['point']:.3e} "
            f"(cluster CI {nl['cluster_ci95']}, p={nl['cluster_p_gt_0']}). "
            f"{'MOSTLY OBSERVABLE' if max_modeled < 0.1 else 'MODEL-DEPENDENT'}: ≤{100*max_modeled:.1f}% of "
            f"value is extrapolated (the rest is EXACT second-price surplus). "
            + ("P2 sanity holds (estimator reproduces realized surplus on the logged policy)." if p2_ok
               else "WARNING: P2 sanity FAILED — estimator does not reproduce realized surplus.")),
    }
    out["verdict"] = verdict
    print(f"\nVERDICT: {verdict['claim']}")

    OUT_JSON.write_text(json.dumps(out, indent=2))
    OUT_MD.write_text(_render_md(out))
    print(f"Wrote {OUT_JSON}\nWrote {OUT_MD}")


def _render_md(out: dict) -> str:
    L: List[str] = ["# Stage 6 — Full-inventory policy-value projection (escape won-only)\n", "## TL;DR\n"]
    v = out["verdict"]
    L.append(f"- {v['claim']}\n")
    L.append(f"- **Honest framing:** structural second-price policy-value projection (NOT OPE — deterministic "
             f"flat logging). V(π) = **V_exact** (observed) + **V_model** (extrapolated increment where the "
             f"policy bids above the logged flat bid). Max modeled share = **{100*v['max_frac_value_modeled']:.1f}%**.\n")
    p2 = out["P2_recover_observable"]
    L.append(f"- **P2 (recover observable):** logged-policy value {p2['logged_policy_value']:.3e} "
             f"== realized 2nd-price surplus on valid wins {p2['realized_2p_surplus_clean']:.3e} → "
             f"match={p2['match_clean']}. Data quirk: **{p2['n_payprice_gt_bidprice_anomalies']:,} "
             f"({100*p2['frac_anomalies']:.1f}%) won rows have payprice>bidprice** (violates second-price) "
             f"and are correctly excluded. (Separately: Stage-B2 ran *first-price* on second-price data — corrected here.)\n")
    p1 = out["P1_market_calibration"]
    L.append(f"- **P1 (market model):** F(b|x) within Wilson CI on **{p1['frac_in_wilson_ci']:.0%}** of logged "
             f"(segment×bid) cells (≤300 only; says nothing about b>300).\n")

    L.append("\n## Truthful (second-price) full-inventory value, by model\n")
    L.append("| model | V(π) total | V_exact | V_model | modeled % | rows extrapolated % | mean bid |")
    L.append("|---|---|---|---|---|---|---|")
    for k, r in out["models_truthful"].items():
        L.append(f"| {k} | {r['total']:.3e} | {r['v_exact']:.3e} | {r['v_model']:.3e} | "
                 f"{100*r['frac_value_modeled']:.1f}% | {100*r['frac_rows_extrapolated']:.3f}% | {r['mean_bid']} |")

    L.append("\n## Full-inventory value gap (neural − baseline)\n")
    L.append("| gap | point | paired CI95 | p>0 | cluster CI95 | p>0 |")
    L.append("|---|---|---|---|---|---|")
    for g, d in out["P3_full_inventory_gap"].items():
        L.append(f"| {g} | {d['point']:.3e} | {d['paired_ci95']} | {d['paired_p_gt_0']} | "
                 f"{d['cluster_ci95']} | {d['cluster_p_gt_0']} |")
    L.append("\n*Most of V(π) is EXACT second-price surplus (observed market prices on re-won inventory); "
             "the modeled increment is the censored region a value-driven policy wins by bidding above the "
             "logged flat bid. Won-only barely binds for truthful bids ≈ V < logged 227–300.*\n")
    L.append("\n## Files\n- `results/stage_a/policy_value.json`\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
