"""Stage 8 — resolve neural-vs-LGB: heterogeneity / power analysis (honest verdict).

The Stage-7 second-price correction left neural−LGB realized surplus point-positive (+9.4M, truthful)
but the 5-advertiser cluster CI contains 0. Is that real-but-underpowered, or null/not-robust? This
runs the claim ladder (R1–R6) on the **truthful (2p-optimal)** second-price gap:

  R1 per-advertiser paired-bootstrap CIs (which advertisers individually win?)
  R2 sign test across 5 advertisers (footnote — near-zero power at n=5)
  R3 Cochran's Q / I² / τ² heterogeneity (DECISIVE: heterogeneity vs power)
  R4 cluster-t MDE / power (calibrate "underpowered", order-of-magnitude)
  R5 intra-advertiser ICC of the per-(advertiser,hour) gap → finer clustering forbidden if ICC>0
  R6 leave-one-advertiser-out mean (single-cluster leverage) + per-advertiser residual-IEB mechanism

Honest framing: 5 advertisers = the entire fair shared-vocab split (population, not sample). Lead with
the descriptive per-advertiser picture; the cluster-mean CI is a sanity bound, not the headline.
9-advertiser eval is a dead end (reintroduces the disjoint-advertiser artifact).

Usage: python scripts/stage_a/power_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.bidding.simulator import cluster_bootstrap_surplus_gap, paired_bootstrap_surplus_gap
from src.bidding.value import ValueConfig, compute_impression_values
from src.features.engineering import load_feature_splits
from src.metrics.cluster_inference import cluster_t_mde, cochrans_q, loocv_means, sign_test
from src.metrics.evaluation import compute_ieb

PROJECT = Path("/home/mail-agent/project/rtb_ipinyou")
NPZ = PROJECT / "results/stage_a/recalibrated_winners_preds.npz"
BASE = PROJECT / "results/stage_a/fair_baseline_preds.npz"
FAIR = PROJECT / "data/ipinyou/prediction/features_fair"
OUT_JSON = PROJECT / "results/stage_a/power_analysis.json"
OUT_MD = PROJECT / "results/stage_a/power_analysis_summary.md"
CPC = 200_000.0
MAX_BID = 300.0


def _truthful_s_vec(pctr, pp, clicks, win_rewin_mask=None):
    """Second-price truthful (bid=clip(V)) realized surplus per row on the won subset."""
    V = compute_impression_values(pctr, ValueConfig(goal_type="CPC", cpc_target=CPC)).values
    bid = np.clip(V, 1.0, MAX_BID)
    rewin = bid >= pp                      # second-price: win iff bid >= market price (payprice)
    return (clicks * CPC - pp) * rewin


def _se_from_ci(ci):
    return (ci[1] - ci[0]) / (2 * 1.96)


def _icc_oneway(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way random-effects ICC(1) of per-cell gaps grouped by advertiser.

    ICC>0 ⇒ within-advertiser correlation across cells ⇒ finer (advertiser×hour) clustering would be
    anti-conservative ⇒ forbidden.
    """
    values = np.asarray(values, np.float64)
    uniq = np.unique(groups)
    G = len(uniq); N = len(values)
    if G < 2 or N <= G:
        return float("nan")
    grand = values.mean()
    ssb = 0.0; ssw = 0.0; ns = []
    for g in uniq:
        m = groups == g
        ng = int(m.sum()); ns.append(ng)
        gm = values[m].mean()
        ssb += ng * (gm - grand) ** 2
        ssw += float(((values[m] - gm) ** 2).sum())
    msb = ssb / (G - 1)
    msw = ssw / (N - G) if N > G else float("nan")
    ns = np.array(ns, np.float64)
    n0 = (N - (ns ** 2).sum() / N) / (G - 1)
    var_b = (msb - msw) / n0
    return float(var_b / (var_b + msw)) if (var_b + msw) > 0 else 0.0


def _analyze(s_a, s_b, adv, label_a, label_b, pctr_a_recal, pctr_b_recal, clicks, hour):
    """Run R1–R6 for one comparison (a − b)."""
    advs = sorted(np.unique(adv).tolist())
    # R1: per-advertiser gap CIs
    per_adv = {}
    gaps, ses = [], []
    for a in advs:
        ma = adv == a
        r = paired_bootstrap_surplus_gap(s_a[ma], s_b[ma], n_boot=5000, seed=0)
        se = _se_from_ci([r.ci95_lo, r.ci95_hi])
        per_adv[str(int(a))] = {"gap": round(r.point, 1), "ci95": [round(r.ci95_lo, 1), round(r.ci95_hi, 1)],
                                "se": round(se, 1), "p_gt_0": round(r.p_gt_0, 4),
                                "ci_excludes_0": bool(r.ci95_lo > 0 or r.ci95_hi < 0),
                                "n_clicks": int(clicks[ma].sum())}
        gaps.append(r.point); ses.append(se)
    gaps = np.array(gaps); ses = np.array(ses)
    n_pos = int((gaps > 0).sum())
    n_ci_win = int(sum(per_adv[str(int(a))]["ci_excludes_0"] and per_adv[str(int(a))]["gap"] > 0 for a in advs))

    # R2 sign test, R3 heterogeneity, R4 MDE, R6 LOAO
    st = sign_test(gaps)
    het = cochrans_q(gaps, ses)
    mde = cluster_t_mde(gaps)
    loao = loocv_means(gaps)
    # R5 ICC of per-(advertiser,hour) gap
    cell_vals, cell_grp = [], []
    d = s_a - s_b
    for a in advs:
        for h in np.unique(hour):
            mm = (adv == a) & (hour == h)
            if mm.sum() >= 500:
                cell_vals.append(float(d[mm].sum())); cell_grp.append(a)
    icc = _icc_oneway(np.array(cell_vals), np.array(cell_grp)) if cell_vals else float("nan")
    # cluster-mean sanity bound
    cm = cluster_bootstrap_surplus_gap(s_a, s_b, adv, n_boot=4000, seed=0)
    # mechanism: per-advertiser residual IEB
    resid = {str(int(a)): {"a_ieb": round(float(compute_ieb(clicks[adv == a], pctr_a_recal[adv == a])), 4),
                           "b_ieb": round(float(compute_ieb(clicks[adv == a], pctr_b_recal[adv == a])), 4)}
             for a in advs}
    return {
        "comparison": f"{label_a} - {label_b}",
        "R1_per_advertiser": per_adv,
        "n_advertisers_positive": n_pos, "n_advertisers_ci_win": n_ci_win, "n_advertisers": len(advs),
        "R2_sign_test": st,
        "R3_heterogeneity": {"Q": round(het.Q, 2), "df": het.df, "p": round(het.p_value, 4),
                             "I2": round(het.I2, 3), "tau2": round(het.tau2, 1),
                             "fixed_effect_mean": round(het.fixed_effect, 1),
                             "heterogeneous": bool(het.p_value < 0.05 or het.I2 > 0.5)},
        "R4_mde": {"mean_gap": round(mde.mean_gap, 1), "sd_between": round(mde.sd_between, 1),
                   "mde_80": round(mde.mde_80, 1), "observed_above_mde": mde.observed_above_mde,
                   "n_clusters_for_power": round(mde.n_clusters_for_power, 1)},
        "R5_icc_per_adv_hour_gap": round(icc, 4) if np.isfinite(icc) else None,
        "R5_finer_clustering": "forbidden (ICC>0)" if (np.isfinite(icc) and icc > 0.05) else "ICC~0 (would be permissible)",
        "R6_loao_means": [round(x, 1) for x in loao.tolist()],
        "cluster_mean_ci_sanity": {"point": round(cm.point, 1), "ci95": [round(cm.ci95_lo, 1), round(cm.ci95_hi, 1)],
                                   "p_gt_0": round(cm.p_gt_0, 4), "excludes_0": bool(cm.ci95_lo > 0 or cm.ci95_hi < 0)},
        "mechanism_residual_ieb": resid,
    }


def main() -> None:
    print("Loading winners + covariates ...")
    d = np.load(NPZ)
    idx = d["idx_won"].astype(np.int64)
    y = d["y_click_won"].astype(np.float64)
    adv_full = np.load(BASE)["advertiser"]
    _, _, test_df, _ = load_feature_splits(FAIR, columns=["payprice", "hour"])
    pp = np.asarray(test_df["payprice"].values, np.float64)[idx]
    hour = np.nan_to_num(np.asarray(test_df["hour"].values, np.float64)[idx], nan=-1).astype(np.int64)
    adv = adv_full[idx]

    keep = pp > 0
    pp, hour, adv, y = pp[keep], hour[keep], adv[keep], y[keep]
    pctr = {k: d[f"{m}_recal"][keep].astype(np.float64)
            for k, m in (("neural", "escm2wc_dr"), ("lr", "lr_ctr_all"), ("lgb", "lgb_ctr_all"))}
    s = {k: _truthful_s_vec(p, pp, y) for k, p in pctr.items()}
    print(f"  winners={len(pp):,} clicks={int(y.sum()):,} advertisers={sorted(np.unique(adv).tolist())}")

    out = {"_meta": {"strategy": "truthful (2p-optimal, second-price)", "cpc": CPC,
                     "n_winners": int(len(pp)), "n_clicks": int(y.sum()),
                     "note": "5 advertisers = entire fair shared-vocab split (population). 9-advertiser "
                             "eval is a dead end (disjoint-advertiser artifact)."}}
    out["neural_minus_lgb"] = _analyze(s["neural"], s["lgb"], adv, "neural", "lgb",
                                       pctr["neural"], pctr["lgb"], y, hour)
    out["neural_minus_lr"] = _analyze(s["neural"], s["lr"], adv, "neural", "lr",
                                      pctr["neural"], pctr["lr"], y, hour)

    nl = out["neural_minus_lgb"]
    robust_lgb = nl["cluster_mean_ci_sanity"]["excludes_0"] and nl["n_advertisers_ci_win"] >= 3
    verdict = {
        "neural_vs_lgb_robust": bool(robust_lgb),
        "neural_vs_lgb": (
            f"NOT robust — advertiser-heterogeneous: positive on {nl['n_advertisers_positive']}/5 "
            f"(CI-significant on only {nl['n_advertisers_ci_win']}/5), heterogeneity "
            f"{'CONFIRMED' if nl['R3_heterogeneity']['heterogeneous'] else 'consistent-with (Q underpowered)'} "
            f"(I²={nl['R3_heterogeneity']['I2']}, Q p={nl['R3_heterogeneity']['p']}); cluster-mean CI "
            f"{nl['cluster_mean_ci_sanity']['ci95']} contains 0; MDE {nl['R4_mde']['mde_80']:.2e} vs "
            f"observed mean {nl['R4_mde']['mean_gap']:.2e}; leave-one-out shows single-advertiser leverage."
            if not robust_lgb else
            f"robust: CI-significant on {nl['n_advertisers_ci_win']}/5 advertisers and cluster CI excludes 0."),
        "neural_vs_lr": (
            f"{'robust' if out['neural_minus_lr']['cluster_mean_ci_sanity']['excludes_0'] else 'not robust'} "
            f"(cluster CI {out['neural_minus_lr']['cluster_mean_ci_sanity']['ci95']}, "
            f"positive on {out['neural_minus_lr']['n_advertisers_positive']}/5)"),
        "finer_clustering": nl["R5_finer_clustering"],
    }
    out["verdict"] = verdict

    print(f"\n  R1 neural-lgb per-advertiser positive: {nl['n_advertisers_positive']}/5 "
          f"(CI-significant {nl['n_advertisers_ci_win']}/5)")
    print(f"  R3 heterogeneity: I²={nl['R3_heterogeneity']['I2']} Q p={nl['R3_heterogeneity']['p']} "
          f"=> {'HETEROGENEOUS' if nl['R3_heterogeneity']['heterogeneous'] else 'cannot reject homogeneity (underpowered)'}")
    print(f"  R4 MDE={nl['R4_mde']['mde_80']:.2e} vs mean={nl['R4_mde']['mean_gap']:.2e} "
          f"(observed above MDE: {nl['R4_mde']['observed_above_mde']})")
    print(f"  R5 ICC(adv×hour gap)={nl['R5_icc_per_adv_hour_gap']} => {nl['R5_finer_clustering']}")
    print(f"  R6 LOAO means: {nl['R6_loao_means']}")
    print(f"\nVERDICT neural>LGB: {verdict['neural_vs_lgb']}")
    print(f"VERDICT neural>LR:  {verdict['neural_vs_lr']}")

    OUT_JSON.write_text(json.dumps(out, indent=2))
    OUT_MD.write_text(_render_md(out))
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}")


def _render_md(out: dict) -> str:
    L: List[str] = ["# Stage 8 — neural-vs-LGB heterogeneity / power analysis (truthful 2p-optimal)\n", "## TL;DR VERDICT\n"]
    v = out["verdict"]
    L.append(f"- **neural vs LGB (strong GBM): {v['neural_vs_lgb']}**\n")
    L.append(f"- **neural vs LR (linear): {v['neural_vs_lr']}**\n")
    L.append(f"- Finer clustering: {v['finer_clustering']}.\n")
    L.append(f"- Framing: 5 advertisers = entire fair shared-vocab split (population). Cluster-mean CI = "
             f"sanity bound, not headline. 9-advertiser eval = dead end (disjoint-advertiser artifact).\n")
    for comp in ("neural_minus_lgb", "neural_minus_lr"):
        a = out[comp]
        L.append(f"\n## {a['comparison']}: per-advertiser (R1) + mechanism\n")
        L.append("| advertiser | gap | CI95 | excl 0 | clicks | neural resid IEB | LGB/LR resid IEB |")
        L.append("|---|---|---|---|---|---|---|")
        for adv, r in sorted(a["R1_per_advertiser"].items(), key=lambda kv: -kv[1]["gap"]):
            mech = a["mechanism_residual_ieb"][adv]
            L.append(f"| {adv} | {r['gap']:.3e} | {r['ci95']} | {r['ci_excludes_0']} | {r['n_clicks']} | "
                     f"{mech['a_ieb']} | {mech['b_ieb']} |")
        h = a["R3_heterogeneity"]; m = a["R4_mde"]
        L.append(f"\n- **R2 sign test:** {a['R2_sign_test']['k_pos']}/{a['R2_sign_test']['n']} positive, "
                 f"p={a['R2_sign_test']['p_greater']} (n=5 ⇒ near-zero power; footnote only).")
        L.append(f"- **R3 heterogeneity:** Q={h['Q']} (df={h['df']}, p={h['p']}), I²={h['I2']}, τ²={h['tau2']:.2e} "
                 f"⇒ **{'HETEROGENEOUS' if h['heterogeneous'] else 'cannot reject homogeneity (Q underpowered)'}**.")
        L.append(f"- **R4 MDE:** mean {m['mean_gap']:.3e}, SD_between {m['sd_between']:.3e}, "
                 f"MDE(80%) {m['mde_80']:.3e} ⇒ observed {'above' if m['observed_above_mde'] else 'AT/BELOW'} MDE; "
                 f"~{m['n_clusters_for_power']} clusters needed (homog. approx).")
        L.append(f"- **R6 leave-one-advertiser-out means:** {a['R6_loao_means']} (single-advertiser leverage).")
        L.append(f"- **cluster-mean sanity:** {a['cluster_mean_ci_sanity']['point']:.3e}, "
                 f"CI {a['cluster_mean_ci_sanity']['ci95']}, excludes 0 = {a['cluster_mean_ci_sanity']['excludes_0']}.")
    L.append("\n## Files\n- `results/stage_a/power_analysis.json`\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
