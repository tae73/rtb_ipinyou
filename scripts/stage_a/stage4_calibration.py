"""Stage 4 — training-stage calibration eval (native / frozen-map val->test / cross-fit) + surplus.

For each retrained ESCM2-WC model dir, evaluates winners pCTR calibration under THREE regimes and
asks whether training-time calibration beats the cheap post-hoc isotonic of Stages 1-3:

  1. native        — raw p_ctr, no recalibration (the Stage 4 headline: did training calibrate it?)
  2. frozen_val2test — isotonic FIT on saved VAL winners, APPLIED to test winners (4b; honest
                       temporal-shift test — the map never sees test). Needs {model}_val_predictions.npz.
  3. crossfit      — cross-fit isotonic on test winners (leak-free optimistic reference; matches
                     recalibration.json).

Verdict (per docs/evaluation_protocol.md): a training-calibration WIN requires the native (or frozen)
max hard-advertiser residual IEB (adv 3358/3476) to beat the global-isotonic ceiling 0.226 WITHOUT
winners-AUC dropping below 0.653 and WITHOUT over-prediction (IEB not negative). Otherwise NEGATIVE
(acceptable): "post-hoc cross-fit isotonic remains the recommended path".

Also runs a surplus check (reuse Stage B2 primitives): native vs cross-fit-isotonic of the SAME model
under dual_regime/exchange_optimal/truthful → realized-surplus gap with paired+cluster bootstrap CIs.

Reads: results/models/<dir>/{escm2wc_dr_test_predictions.npz, escm2wc_dr_val_predictions.npz},
data/.../features_fair/test.parquet, results/market_price_cdf/*.
Writes: results/stage_a/stage4_calibration.{json,_summary.md}.

Usage: python scripts/stage_a/stage4_calibration.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

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
from src.metrics.calibration import cross_fit_isotonic, fit_isotonic, quantile_reliability
from src.metrics.evaluation import _numpy_roc_auc, compute_ieb

PROJECT = Path("/home/mail-agent/project/rtb_ipinyou")
FAIR = PROJECT / "data/ipinyou/prediction/features_fair"
CDF_DIR = PROJECT / "results/market_price_cdf"
OUT_JSON = PROJECT / "results/stage_a/stage4_calibration.json"
OUT_MD = PROJECT / "results/stage_a/stage4_calibration_summary.md"

CPC = 200_000.0
MAX_BID = 300.0
HARD_ADV = ("3358", "3476")          # advertisers with the worst global-isotonic residual
ISO_CEILING = 0.226                    # global-isotonic max per-adv residual to beat (neural)
AUC_FLOOR = 0.653                      # ranking guard (~0.658 - 5e-3)

# label -> model dir (only those with predictions on disk are evaluated)
MODEL_DIRS = {
    "fair_baseline": "results/models/escm2wc_dr_fair_posw",     # dr-mse, joint 0.1, pos_weight 50 (inert) — IEB 0.597
    "A2_relax_jw003": "results/models/stage4_jw003",            # dr-mse, joint 0.03 (relax squeeze)
    "B2_pw20": "results/models/stage4_drbce_pw20",              # dr-bce, pos_weight 20 (over-corrected)
    "C_pw2": "results/models/stage4_drbce_pw2",                 # dr-bce, pos_weight 2 (sweet-spot probe)
}
STRATEGIES = ["exchange_optimal", "dual_regime", "truthful"]


def _load_preds(model_dir: Path):
    test_npz = glob.glob(str(model_dir / "*_test_predictions.npz"))
    if not test_npz:
        return None, None
    test = np.load(test_npz[0])
    val_npz = glob.glob(str(model_dir / "*_val_predictions.npz"))
    val = np.load(val_npz[0]) if val_npz else None
    return test, val


def _cal_metrics(pred: np.ndarray, y: np.ndarray, adv: np.ndarray) -> dict:
    rel = quantile_reliability(y.astype(float), pred.astype(float), n_bins=10)
    per_adv = {}
    for a in np.unique(adv):
        m = adv == a
        per_adv[str(int(a))] = round(float(compute_ieb(y[m], pred[m])), 4)
    hard = [per_adv[a] for a in HARD_ADV if a in per_adv]
    return {
        "ieb": round(float(compute_ieb(y, pred)), 4),
        "winners_auc": round(float(_numpy_roc_auc(y, pred)), 4),
        "quantile_ece": round(float(rel.quantile_ece), 6),
        "pred_mean": float(pred.mean()),
        "true_mean": float(y.mean()),
        "per_advertiser_ieb": per_adv,
        "max_hard_adv_ieb": round(max(hard), 4) if hard else None,
    }


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
    return s_vec, float(s_vec.sum()), float(V.mean())


def main() -> None:
    # Covariates for the fair test (winners subset per model via that model's y_win).
    print("Loading fair-test covariates ...")
    _, _, test_df, _ = load_feature_splits(
        FAIR, columns=["payprice", "adexchange", "slotprice", "advertiser"]
    )
    pp_all = np.asarray(test_df["payprice"].values, np.float64)
    ax_all = np.nan_to_num(np.asarray(test_df["adexchange"].values, np.float64), nan=-1).astype(np.int64)
    slot_all = np.nan_to_num(np.asarray(test_df["slotprice"].values, np.float64), nan=0.0)
    adv_all = np.asarray(test_df["advertiser"].values)
    overall = load_market_cdf(str(CDF_DIR / "km_cdf_overall.npz"))
    exch = load_exchange_cdfs(str(CDF_DIR))

    results: Dict[str, dict] = {}
    for label, rel_dir in MODEL_DIRS.items():
        mdir = PROJECT / rel_dir
        test, val = _load_preds(mdir)
        if test is None:
            print(f"  skip {label}: no test predictions in {rel_dir}")
            continue
        print(f"== {label} ({rel_dir}) ==")
        yw = test["y_win"].astype(np.int64)
        yc = test["y_click"].astype(np.int64)
        won = yw == 1
        p_native = test["p_ctr"][won].astype(np.float64)
        y_won = yc[won].astype(np.float64)
        adv_won = adv_all[won]

        regimes = {"native": p_native}
        # frozen-map val->test (honest): fit isotonic on saved VAL winners, apply to test winners
        if val is not None and "p_ctr" in val:
            vwon = val["y_win"].astype(np.int64) == 1
            iso = fit_isotonic(val["p_ctr"][vwon].astype(np.float64), val["y_click"][vwon].astype(np.float64))
            regimes["frozen_val2test"] = iso.transform(p_native)
        else:
            print(f"   (no val predictions — frozen_val2test unavailable for {label})")
        # cross-fit on test (leak-free optimistic reference)
        regimes["crossfit"] = cross_fit_isotonic(p_native, y_won, n_folds=5, seed=0)

        cal = {rk: _cal_metrics(rp, y_won, adv_won) for rk, rp in regimes.items()}
        # ranking guard: isotonic regimes are monotone → AUC must match native within tol
        for rk in ("frozen_val2test", "crossfit"):
            if rk in cal:
                drift = abs(cal[rk]["winners_auc"] - cal["native"]["winners_auc"])
                if drift > 5e-3:
                    print(f"   NOTE {label}/{rk}: AUC drift {drift:.4f} (>5e-3)")

        results[label] = {"calibration": cal, "n_winners": int(won.sum())}
        nat = cal["native"]
        print(f"   native: IEB {nat['ieb']} AUC {nat['winners_auc']} maxHardAdv {nat['max_hard_adv_ieb']} "
              f"| crossfit IEB {cal['crossfit']['ieb']}"
              + (f" | frozen IEB {cal['frozen_val2test']['ieb']} maxHardAdv {cal['frozen_val2test']['max_hard_adv_ieb']}"
                 if 'frozen_val2test' in cal else ""))

        # ---- Surplus: native vs cross-fit isotonic of the SAME model ----
        keep = pp_all > 0
        won_keep = won & keep
        pk = pp_all[won_keep]; axk = ax_all[won_keep]; sk = slot_all[won_keep]
        advk = adv_all[won_keep]; ck = yc[won_keep].astype(np.float64)
        # recompute regime preds on the keep subset (align to won_keep within winners)
        keep_in_won = keep[won]
        p_nat_k = p_native[keep_in_won]
        p_iso_k = regimes["crossfit"][keep_in_won]
        surplus = {}
        for strat in STRATEGIES:
            s_nat, tot_nat, mv_nat = _surplus_vec(p_nat_k, pk, ck, axk, sk, overall, exch, strat)
            s_iso, tot_iso, mv_iso = _surplus_vec(p_iso_k, pk, ck, axk, sk, overall, exch, strat)
            paired = paired_bootstrap_surplus_gap(s_nat, s_iso, n_boot=3000, seed=0)
            clus = cluster_bootstrap_surplus_gap(s_nat, s_iso, advk, n_boot=3000, seed=0)
            surplus[strat] = {
                "native_surplus": round(tot_nat, 1), "iso_surplus": round(tot_iso, 1),
                "native_minus_iso": round(paired.point, 1),
                "paired_ci95": [round(paired.ci95_lo, 1), round(paired.ci95_hi, 1)], "paired_p_gt_0": round(paired.p_gt_0, 4),
                "cluster_ci95": [round(clus.ci95_lo, 1), round(clus.ci95_hi, 1)], "cluster_p_gt_0": round(clus.p_gt_0, 4),
                "mean_V_native": round(mv_nat, 2), "mean_V_iso": round(mv_iso, 2),
            }
        results[label]["surplus_native_vs_iso"] = surplus

    # ---- Verdict (judge the best non-baseline training run) ----
    candidates = [k for k in results if k != "fair_baseline"]
    verdict = {"iso_ceiling": ISO_CEILING, "auc_floor": AUC_FLOOR, "runs_evaluated": list(results.keys())}
    best = None
    for label in candidates:
        cal = results[label]["calibration"]
        for regime in ("native", "frozen_val2test"):
            if regime not in cal:
                continue
            c = cal[regime]
            beats = (c["max_hard_adv_ieb"] is not None and c["max_hard_adv_ieb"] < ISO_CEILING
                     and c["winners_auc"] >= AUC_FLOOR and c["ieb"] >= -0.05)
            if beats and (best is None or c["max_hard_adv_ieb"] < best[2]):
                best = (label, regime, c["max_hard_adv_ieb"], c["ieb"], c["winners_auc"])
    if best:
        verdict["calibration_supported"] = True
        verdict["winner"] = {"run": best[0], "regime": best[1], "max_hard_adv_ieb": best[2],
                             "ieb": best[3], "winners_auc": best[4]}
        verdict["statement"] = (
            f"Training-stage calibration WINS: {best[0]}/{best[1]} brings max hard-advertiser residual "
            f"to {best[2]} (< {ISO_CEILING} global-isotonic ceiling) with winners AUC {best[4]} (≥{AUC_FLOOR}).")
    else:
        verdict["calibration_supported"] = False
        verdict["statement"] = (
            "NEGATIVE (acceptable): no training run beats the global-isotonic per-advertiser ceiling "
            f"({ISO_CEILING}) without losing ranking/over-predicting. Post-hoc cross-fit isotonic remains "
            "the recommended calibration path (rank-preserving, leak-free, 1-line, GPU 0).")
    print("\nVERDICT:", verdict["statement"])

    OUT_JSON.write_text(json.dumps({"verdict": verdict, "models": results,
                                    "_meta": {"cpc_target": CPC, "hard_advertisers": HARD_ADV}}, indent=2))
    OUT_MD.write_text(_render_md(verdict, results))
    print(f"Wrote {OUT_JSON}\nWrote {OUT_MD}")


def _render_md(verdict: dict, results: dict) -> str:
    L: List[str] = ["# Stage 4 — Training-stage calibration (native / frozen val→test / cross-fit)\n"]
    L.append("## TL;DR VERDICT\n")
    L.append(f"- **{'CALIBRATION SUPPORTED' if verdict['calibration_supported'] else 'NEGATIVE (acceptable)'}.** "
             f"{verdict['statement']}\n")
    L.append(f"- Ceiling to beat = global-isotonic max hard-advertiser ({'/'.join(HARD_ADV)}) residual "
             f"**{verdict['iso_ceiling']}**; ranking floor AUC **{verdict['auc_floor']}**.\n")

    L.append("\n## Calibration by regime (winners pCTR)\n")
    L.append("| run | regime | IEB | winners AUC | qECE | max hard-adv resid |")
    L.append("|---|---|---|---|---|---|")
    for label, r in results.items():
        for rk, c in r["calibration"].items():
            L.append(f"| {label} | {rk} | {c['ieb']} | {c['winners_auc']} | {c['quantile_ece']} | {c['max_hard_adv_ieb']} |")

    L.append("\n## Per-advertiser residual IEB (native / frozen / cross-fit)\n")
    L.append("| run | regime | 1458 | 3358 | 3386 | 3427 | 3476 |")
    L.append("|---|---|---|---|---|---|---|")
    for label, r in results.items():
        for rk, c in r["calibration"].items():
            pa = c["per_advertiser_ieb"]
            L.append(f"| {label} | {rk} | " + " | ".join(str(pa.get(a, "—")) for a in ("1458","3358","3386","3427","3476")) + " |")

    L.append("\n## Surplus: native vs own cross-fit isotonic (does native still need post-hoc?)\n")
    L.append("| run | strategy | native surplus | iso surplus | native−iso | cluster CI95 | p>0 | meanV nat/iso |")
    L.append("|---|---|---|---|---|---|---|---|")
    for label, r in results.items():
        for strat, s in r.get("surplus_native_vs_iso", {}).items():
            L.append(f"| {label} | {strat} | {s['native_surplus']:.3e} | {s['iso_surplus']:.3e} | "
                     f"{s['native_minus_iso']:.3e} | {s['cluster_ci95']} | {s['cluster_p_gt_0']} | "
                     f"{s['mean_V_native']}/{s['mean_V_iso']} |")
    L.append("\n*Reading: if native−iso ≈ 0 (CI contains 0) the native model bids as well as its post-hoc "
             "isotonic → training calibration is non-inferior; if native ≪ iso, native still under-bids "
             "(under-prediction not fixed at train time) and post-hoc isotonic stays necessary.*\n")
    L.append("\n## Files\n- `results/stage_a/stage4_calibration.json` — full metrics + per-adv + surplus CIs.\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
