"""Portfolio figures — generated from committed Stage A JSON ledgers.

Reads only ``results/stage_a/*.json`` (no model re-training, no data mutation) and writes
portfolio-grade PNG figures to ``results/figures/portfolio/``. Every value plotted is traceable to
``docs/NUMBERS_LEDGER.md``.

Usage:
    python scripts/portfolio/make_figures.py

Figures:
    fig_surplus_forest.png        per-advertiser surplus gaps + CI (the honest hero)
    fig_calibration_journey.png   reliability before/after + per-advertiser residual raw->global->segment
    fig_ablation_auc.png          winners-only vs all-bids AUC across LR/LGB/neural
    fig_surplus_grid.png          neural-baseline surplus gap by strategy (2nd-price)
    fig_policy_value_decomp.png   full-inventory V(pi) = exact + modeled (>=99% exact)
    fig_artifact_vs_fair.png      the retraction: old adversarial split vs fair split
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
STAGE_A = ROOT / "results" / "stage_a"
OUT = ROOT / "results" / "figures" / "portfolio"


class Palette(NamedTuple):
    neural: str = "#4F46E5"   # indigo
    lgb: str = "#0D9488"      # teal
    lr: str = "#6B7280"       # gray
    pos: str = "#16A34A"      # green  — CI excludes 0, favorable
    neg: str = "#DC2626"      # red    — CI excludes 0, unfavorable
    ns: str = "#9CA3AF"       # gray   — CI contains 0 (not significant)
    ink: str = "#111827"
    grid: str = "#E5E7EB"


C = Palette()
M = 1e6  # surplus unit -> "millions"


def _load(name: str) -> dict:
    with open(STAGE_A / name) as fh:
        return json.load(fh)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": C.ink,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": C.grid,
        "grid.linewidth": 0.7,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def _ci_color(gap: float, excludes_0: bool) -> str:
    if not excludes_0:
        return C.ns
    return C.pos if gap > 0 else C.neg


# --------------------------------------------------------------------------------------------------
# Fig 1 — surplus forest (HERO): per-advertiser neural-baseline gaps with 95% CI
# --------------------------------------------------------------------------------------------------
def fig_surplus_forest() -> None:
    pa = _load("power_analysis.json")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=False)

    panels = [
        ("neural_minus_lgb", "neural − LGB   (NOT robust)", axes[0]),
        ("neural_minus_lr", "neural − LR   (robust)", axes[1]),
    ]
    for key, title, ax in panels:
        block = pa[key]
        per_adv = block["R1_per_advertiser"]
        items = sorted(per_adv.items(), key=lambda kv: kv[1]["gap"])  # ascending -> 3427 on top
        ys = range(len(items))
        for y, (adv, d) in zip(ys, items):
            gap, (lo, hi) = d["gap"] / M, (d["ci95"][0] / M, d["ci95"][1] / M)
            col = _ci_color(d["gap"], d["ci_excludes_0"])
            ax.plot([lo, hi], [y, y], color=col, lw=2.4, solid_capstyle="round", zorder=2)
            ax.scatter([gap], [y], color=col, s=70, zorder=3, edgecolor="white", linewidth=1.0)
            tag = " ✓" if d["ci_excludes_0"] else ""
            ax.text(hi, y + 0.18, f"  {gap:+.1f}M{tag}", va="bottom", ha="left",
                    fontsize=8.5, color=col)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([f"adv {adv}\n(clk {d['n_clicks']})" for adv, d in items], fontsize=9)

        # cluster mean diamond
        cm = block["cluster_mean_ci_sanity"]
        cmean, clo, chi = cm["point"] / M, cm["ci95"][0] / M, cm["ci95"][1] / M
        ydia = len(items) + 0.6
        ccol = C.pos if cm["excludes_0"] and cmean > 0 else C.ns
        ax.plot([clo, chi], [ydia, ydia], color=ccol, lw=2.6, zorder=2)
        ax.scatter([cmean], [ydia], marker="D", s=95, color=ccol, zorder=3,
                   edgecolor="white", linewidth=1.0)
        ax.text(chi, ydia + 0.18, f"  cluster mean {cmean:+.1f}M "
                f"[{clo:.0f}, {chi:.0f}]", va="bottom", ha="left", fontsize=8.5,
                color=ccol, fontweight="bold")

        het = block["R3_heterogeneity"]
        sub = (f"Cochran Q={het['Q']:.1f} (p={het['p']:.4f}),  I²={het['I2']:.2f}")
        ax.axvline(0, color=C.ink, lw=1.0, zorder=1)
        ax.set_title(title, fontsize=12, fontweight="bold", color=C.ink, pad=22)
        ax.text(0.5, 1.005, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color="#374151")
        ax.set_xlabel("realized 2nd-price surplus gap (millions)")
        ax.set_ylim(-0.7, len(items) + 1.4)
        ax.margins(x=0.18)

    # LOAO annotation on the LGB panel
    loao = pa["neural_minus_lgb"]["R6_loao_means"]
    axes[0].text(0.02, 0.02,
                 f"Leave-one-advertiser-out: drop 3427 → mean {min(loao)/M:+.1f}M (flips negative)",
                 transform=axes[0].transAxes, fontsize=8.2, color=C.neg, style="italic")

    fig.suptitle("Does debiasing improve bidding? Per-advertiser surplus, truthful 2nd-price "
                 "(5 advertisers)", fontsize=13.5, fontweight="bold", y=1.02)
    _save(fig, "fig_surplus_forest.png")


# --------------------------------------------------------------------------------------------------
# Fig 2 — calibration journey: reliability before/after + per-advertiser residual raw->global->segment
# --------------------------------------------------------------------------------------------------
def fig_calibration_journey() -> None:
    recal = _load("recalibration.json")["escm2wc_dr"]
    seg = _load("segment_calibration.json")["models"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Panel A — neural winners reliability, before vs after global isotonic
    axA = axes[0]
    before = recal["before"]["deciles"]
    after = recal["after"]["deciles"]
    bp = [d["pred"] for d in before]
    bt = [d["true"] for d in before]
    ap = [d["pred"] for d in after]
    at = [d["true"] for d in after]
    lim_lo, lim_hi = 2e-5, 3e-3
    axA.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=C.ink, lw=1.0, ls="--",
             label="perfect calibration")
    axA.plot(bp, bt, "o-", color=C.neg, lw=1.8, ms=6, label=f"raw  (IEB {recal['ieb_before']:.3f})")
    axA.plot(ap, at, "o-", color=C.pos, lw=1.8, ms=6,
             label=f"+ isotonic  (IEB {recal['ieb_after']:.3f})")
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlim(lim_lo, lim_hi)
    axA.set_ylim(lim_lo, lim_hi)
    axA.set_xlabel("predicted pCTR (decile mean)")
    axA.set_ylabel("observed CTR (decile mean)")
    axA.set_title("Neural winners pCTR — global recalibration\n"
                  "(all 10 deciles under-predicted → on diagonal)", fontsize=10.5,
                  fontweight="bold", color=C.ink)
    axA.legend(fontsize=8.5, loc="upper left", framealpha=0.95)

    # Panel B — per-advertiser MAX residual IEB across raw -> global -> segment, 3 models
    axB = axes[1]
    stages = ["raw", "global", "segment"]
    series = {
        "neural": (C.neural, "o"),
        "lgb": (C.lgb, "s"),
        "lr": (C.lr, "^"),
    }
    for model, (col, mk) in series.items():
        cal = seg[model]["calibration"]
        ys = [cal[s]["max_per_adv_ieb"] for s in stages]
        axB.plot(stages, ys, marker=mk, color=col, lw=2.0, ms=8, label=model)
        if model == "neural":  # annotate only the protagonist to avoid label collision
            for x, yv in zip(stages, ys):
                axB.annotate(f"{yv:g}", (x, yv), textcoords="offset points",
                             xytext=(0, 11), ha="center", fontsize=9, color=col,
                             fontweight="bold")
    axB.set_yscale("log")
    axB.set_ylabel("per-advertiser MAX residual IEB (log)")
    axB.set_title("Per-advertiser calibration residual\nraw → global isotonic → per-advertiser "
                  "(segment)", fontsize=10.5, fontweight="bold", color=C.ink)
    axB.legend(fontsize=9, title="model")
    axB.margins(x=0.12)

    fig.suptitle("Calibration journey — global isotonic levels the mean, per-advertiser maps close "
                 "the residual", fontsize=12.5, fontweight="bold", y=1.0)
    _save(fig, "fig_calibration_journey.png")


# --------------------------------------------------------------------------------------------------
# Fig 3 — ablation AUC: winners-only vs all-bids, LR/LGB/neural (fair split)
# --------------------------------------------------------------------------------------------------
def fig_ablation_auc() -> None:
    fb = _load("fair_baselines.json")
    fc = _load("fair_comparison.json")["models"]
    neural = fc["escm2wc_dr_fair (neural, fixed)"]

    models = ["LR", "LGB", "neural\n(ESCM²-WC DR)"]
    cols = [C.lr, C.lgb, C.neural]
    winners = [fb["LR_ctr_all"]["winners_auc"], fb["LGB_ctr_all"]["winners_auc"],
               neural["winners_auc"]]
    allbids = [fb["LR_ctr_all"]["all_bids_auc"], fb["LGB_ctr_all"]["all_bids_auc"],
               neural["all_bids_auc"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharey=True)
    for ax, vals, title, note in [
        (axes[0], winners, "Winners-only AUC  —  P(click | win)\n(the object bidding ranks on)",
         "neural leads"),
        (axes[1], allbids, "All-bids AUC\n(easy negatives inflate; not the bidding object)",
         "LGB leads"),
    ]:
        bars = ax.bar(models, vals, color=cols, width=0.62, edgecolor="white", zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=10.5, fontweight="bold")
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=C.ink)
        ax.set_ylim(0.5, 0.78)
    axes[0].set_ylabel("ROC AUC   (0.5 = chance)")

    fig.suptitle("Fair-split ranking — debiasing wins the winners-only object, trails on all-bids",
                 fontsize=12.5, fontweight="bold", y=1.0)
    _save(fig, "fig_ablation_auc.png")


# --------------------------------------------------------------------------------------------------
# Fig 4 — surplus grid: neural - baseline gap by strategy (2nd-price, among-recal)
# --------------------------------------------------------------------------------------------------
def fig_surplus_grid() -> None:
    rc = _load("stage_b2_surplus.json")["decomposition"]["ranking_contribution_among_recal"]
    strategies = ["exchange_optimal", "dual_regime", "truthful"]
    labels = ["exchange_optimal", "dual_regime", "truthful\n(2p-optimal ★)"]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    x = range(len(strategies))
    width = 0.36
    # color encodes significance; hatch encodes which baseline (LGB solid / LR hatched)
    for cmp_key, off, hatch, short in [
        ("neural_minus_lgb", -width / 2, "", "LGB"),
        ("neural_minus_lr", width / 2, "//", "LR"),
    ]:
        for xi, s in zip(x, strategies):
            d = rc[s][cmp_key]
            gap = d["point"] / M
            lo, hi = d["cluster_ci95"][0] / M, d["cluster_ci95"][1] / M
            excl = not (lo <= 0 <= hi)
            col = _ci_color(d["point"], excl)
            bx = xi + off
            ax.bar(bx, gap, width=width, color=col, edgecolor=C.ink, linewidth=0.7,
                   hatch=hatch, zorder=3, alpha=0.92)
            ax.errorbar(bx, gap, yerr=[[gap - lo], [hi - gap]], fmt="none", ecolor=C.ink,
                        elinewidth=1.1, capsize=4, zorder=4)
            ax.text(bx, hi + 1.6, "✓" if excl else "✗", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color=col)
            ax.text(bx, -23.0, short, ha="center", va="top", fontsize=8.5, color="#374151")

    ax.axhline(0, color=C.ink, lw=1.1, zorder=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(-24, 52)
    ax.set_ylabel("realized 2nd-price surplus gap (millions)\nwith advertiser-cluster 95% CI")
    ax.set_title("Decision value by strategy — neural beats LR robustly; beats LGB only on point",
                 fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch
    leg_sig = ax.legend(handles=[
        Patch(facecolor=C.pos, label="✓ cluster CI excludes 0"),
        Patch(facecolor=C.ns, label="✗ cluster CI contains 0"),
    ], fontsize=9, loc="upper left", title="significance", framealpha=0.95)
    ax.add_artist(leg_sig)
    ax.legend(handles=[
        Patch(facecolor="white", edgecolor=C.ink, label="left bar: neural − LGB"),
        Patch(facecolor="white", edgecolor=C.ink, hatch="//", label="right bar: neural − LR"),
    ], fontsize=9, loc="upper center", title="baseline", framealpha=0.95)
    _save(fig, "fig_surplus_grid.png")


# --------------------------------------------------------------------------------------------------
# Fig 5 — full-inventory policy value: V(pi) = exact + modeled
# --------------------------------------------------------------------------------------------------
def fig_policy_value_decomp() -> None:
    pv = _load("policy_value.json")
    mt = pv["models_truthful"]
    gap = pv["P3_full_inventory_gap"]
    order = [("neural", C.neural), ("lgb", C.lgb), ("lr", C.lr)]
    labels = {"neural": "neural", "lgb": "LGB", "lr": "LR"}

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    B = 1e8  # plot in hundred-millions
    xs = range(len(order))
    for x, (m, col) in zip(xs, order):
        vex = mt[m]["v_exact"] / B
        vmod = mt[m]["v_model"] / B
        ax.bar(x, vex, color=col, width=0.58, edgecolor="white", zorder=3)
        ax.bar(x, vmod, bottom=vex, color=col, width=0.58, edgecolor="white", alpha=0.35,
               hatch="////", zorder=3)
        total = mt[m]["total"] / B
        modeled_pct = mt[m]["frac_value_modeled"] * 100
        ax.text(x, total + 0.22, f"{total:.2f}×1e8", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
        ax.text(x, total + 0.05, f"({modeled_pct:.1f}% modeled)", ha="center", va="bottom",
                fontsize=8.5, color="#374151")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([labels[m] for m, _ in order])
    ax.set_ylabel("full-inventory policy value V(π)   (×1e8, 2nd-price truthful)")
    ax.set_ylim(0, 5.6)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C.ink, label="V_exact  (observed 2nd-price surplus)"),
        Patch(facecolor=C.ink, alpha=0.35, hatch="////", label="V_model  (extrapolated, ≤0.74%)"),
    ], fontsize=9, loc="lower center")
    nl = gap["neural_minus_lr"]
    ng = gap["neural_minus_lgb"]
    sub = (f"neural−LR +{nl['point']/M:.1f}M (CI [{nl['cluster_ci95'][0]/M:.1f}, "
           f"{nl['cluster_ci95'][1]/M:.1f}] ✓)   ·   "
           f"neural−LGB +{ng['point']/M:.1f}M (CI [{ng['cluster_ci95'][0]/M:.1f}, "
           f"{ng['cluster_ci95'][1]/M:.1f}] ✗)")
    ax.set_title("Full-inventory value is ≥99% EXACT — won-only is barely binding",
                 fontsize=12.5, fontweight="bold", pad=26)
    ax.text(0.5, 1.012, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
            color="#374151")
    _save(fig, "fig_policy_value_decomp.png")


# --------------------------------------------------------------------------------------------------
# Fig 6 — the retraction: old adversarial split vs fair split (winners-only AUC)
# --------------------------------------------------------------------------------------------------
def fig_artifact_vs_fair() -> None:
    v = _load("fair_baselines.json")["verdict"]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    groups = ["LR", "LGB"]
    old = [v["old_adversarial_split_LR_winners_auc"], v["old_adversarial_split_LGB_winners_auc"]]
    fair = [v["fair_split_LR_winners_auc"], v["fair_split_LGB_winners_auc"]]
    x = range(len(groups))
    width = 0.36
    b1 = ax.bar([xi - width / 2 for xi in x], old, width, color=C.ns, edgecolor="white",
                label="old adversarial split (disjoint advertisers)", zorder=3)
    b2 = ax.bar([xi + width / 2 for xi in x], fair, width,
                color=[C.lr, C.lgb], edgecolor="white",
                label="fair per-advertiser temporal split", zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=10.5,
                    fontweight="bold")
    # arrows showing the reversal
    ax.annotate("", xy=(0 + width / 2, fair[0]), xytext=(0 - width / 2, old[0]),
                arrowprops=dict(arrowstyle="->", color=C.neg, lw=1.6))
    ax.annotate("", xy=(1 + width / 2, fair[1]), xytext=(1 - width / 2, old[1]),
                arrowprops=dict(arrowstyle="->", color=C.pos, lw=1.6))
    ax.text(0, old[0] + 0.02, "LR collapses\n(was the artifact)", ha="center", fontsize=8.5,
            color=C.neg)
    ax.text(1, fair[1] + 0.03, "LGB rises", ha="center", fontsize=8.5, color=C.pos)
    ax.axhline(0.5, color=C.ink, lw=1.0, ls="--", zorder=2)
    ax.text(1.45, 0.505, "chance (exclude adv 2997 → ≈0.499)", fontsize=8, ha="right", color=C.ink)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("winners-only AUC")
    ax.set_ylim(0.45, 0.78)
    ax.set_title("The retraction — the original 'debiasing loses' headline was a split artifact\n"
                 "LR 0.714→0.554 (rode on one unseen advertiser), LGB 0.479→0.632",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    _save(fig, "fig_artifact_vs_fair.png")


_LADDER_COLORS = ["#6B7280", "#0D9488", "#A5B4FC", "#818CF8", "#4F46E5"]


def fig_ablation_ladder() -> None:
    d = _load("ablation_ladder.json")["ladder"]
    labels = [r["label"].replace(" (", "\n(") for r in d]
    auc = [r["winners_auc"] for r in d]
    ieb_raw = [max(abs(r["winners_ieb_raw"]), 1e-3) for r in d]
    ieb_rec = [max(abs(r["winners_ieb_recal"]), 1e-3) for r in d]
    cols = _LADDER_COLORS[: len(d)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    ax.bar(range(len(d)), auc, color=cols, edgecolor="white", width=0.66, zorder=3)
    ax.axhline(0.5, ls="--", color=C.ink, lw=1)
    for i, v in enumerate(auc):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(labels, fontsize=8.4)
    ax.set_ylim(0.5, max(auc) + 0.03)
    ax.set_ylabel("winners-only AUC  (0.5 = chance)")
    ax.set_title("Ranking along the ladder\n(winners-only AUC = the bidding object)",
                 fontsize=11, fontweight="bold")
    ax = axes[1]
    x = range(len(d))
    w = 0.38
    ax.bar([i - w / 2 for i in x], ieb_raw, w, color=C.neg, label="raw", zorder=3)
    ax.bar([i + w / 2 for i in x], ieb_rec, w, color=C.pos, label="+ cross-fit isotonic", zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8.4)
    ax.set_ylabel("|winners IEB|  (log)")
    ax.legend(fontsize=9)
    ax.set_title("Calibration fixed post-hoc for every rung\n(raw IEB varies wildly → recal ≈ 0)",
                 fontsize=11, fontweight="bold")
    fig.suptitle("Ablation ladder (fair split): LR → LGB → ESMM-WC → ESCM²-WC(IPW) → ESCM²-WC(DR)",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, "fig_ablation_ladder.png")


def fig_bidding_strategies() -> None:
    d = _load("bidding_fair.json")
    strat, alpha, best = d["strategy_comparison"], d["alpha_sensitivity"], d["best_strategy"]
    B = 1e8
    names = list(strat)
    sur = [strat[s]["total_surplus"] / B for s in names]
    cols = [C.pos if s == best else C.lgb for s in names]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    ax.bar(names, sur, color=cols, edgecolor="white", zorder=3)
    for i, v in enumerate(sur):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("realized 2nd-price surplus (×1e8)")
    ax.tick_params(axis="x", labelrotation=18)
    ax.set_title(f"Strategy comparison — best: {best}", fontsize=11, fontweight="bold")
    ax = axes[1]
    a = [x["alpha"] for x in alpha]
    sa = [x["total_surplus"] / B for x in alpha]
    roi = [x["roi"] for x in alpha]
    ax.plot(a, sa, "o-", color=C.neural, lw=2)
    ax.set_xlabel("linear shading α  (bid = α·V)")
    ax.set_ylabel("surplus (×1e8)", color=C.neural)
    ax2 = ax.twinx()
    ax2.plot(a, roi, "s--", color=C.lgb, lw=2)
    ax2.set_ylabel("ROI", color=C.lgb)
    ax2.grid(False)
    ax.set_title("Shading α: surplus ↑, ROI ↓ (win-rate / cost tradeoff)", fontsize=11, fontweight="bold")
    fig.suptitle("Bid-shading on the fair split (recalibrated neural pCTR, 2nd-price)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    _save(fig, "fig_bidding_strategies.png")


def fig_pacing() -> None:
    d = _load("pacing_fair.json")["by_budget"]
    B = 1e8
    keys = sorted(d, key=lambda k: float(k))
    fr = [d[k]["budget_frac"] for k in keys]
    nop = [d[k]["no_pacing"]["surplus"] / B for k in keys]
    uni = [d[k]["pid_uniform"]["surplus"] / B for k in keys]
    wr = [d[k]["wr_weighted"]["surplus"] / B for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(fr, nop, "o-", color=C.ns, lw=2, label="no pacing")
    ax.plot(fr, uni, "s-", color=C.lr, lw=2, label="PID uniform")
    ax.plot(fr, wr, "^-", color=C.pos, lw=2, label="WR-weighted PID")
    ax.set_xlabel("daily budget (fraction of unconstrained spend)")
    ax.set_ylabel("surplus (×1e8)")
    ax.legend(fontsize=9)
    ax.set_title("Surplus vs budget", fontsize=11, fontweight="bold")
    ax = axes[1]
    lift = [(d[k]["wr_weighted"]["surplus"] / d[k]["pid_uniform"]["surplus"] - 1) * 100 for k in keys]
    ax.bar([f"{f:.1f}" for f in fr], lift, color=C.pos, edgecolor="white", zorder=3)
    for i, v in enumerate(lift):
        ax.text(i, v + 0.15, f"+{v:.1f}%", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("budget fraction")
    ax.set_ylabel("WR-weighted surplus lift over uniform (%)")
    ax.set_title("Smart hourly allocation lifts surplus", fontsize=11, fontweight="bold")
    fig.suptitle("Budget pacing on the fair split (truthful 2nd-price)", fontsize=12.5, fontweight="bold", y=1.0)
    _save(fig, "fig_pacing.png")


def fig_market_cdf() -> None:
    import numpy as np
    s = json.load(open(ROOT / "results/market_price_cdf/summary.json"))
    km = np.load(ROOT / "results/market_price_cdf/km_cdf_overall.npz")
    ks = km.files
    grid = km["price_grid"] if "price_grid" in ks else km[ks[0]]
    if "cdf" in ks:
        cdf = km["cdf"]
    elif "survival" in ks:
        cdf = 1 - km["survival"]
    else:
        cdf = km[ks[1]]
    st = s["market_price_stats"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axes[0]
    ax.plot(grid, cdf, color=C.neural, lw=2.2)
    ax.axvline(st["median"], ls="--", color=C.lgb, lw=1.2)
    ax.text(st["median"] + 5, 0.12, f"median {st['median']:.0f}", color=C.lgb, fontsize=9)
    ax.set_xlim(0, 320)
    ax.set_xlabel("market price (CPM)")
    ax.set_ylabel("win prob  F(b) = P(market ≤ b)")
    ax.set_title(f"Kaplan-Meier market-price CDF\nmean {st['mean']:.0f} · median {st['median']:.0f} · "
                 f"floor-binding {st['floor_binding_rate']*100:.0f}%", fontsize=10.5, fontweight="bold")
    ax = axes[1]
    exf = s["exchange_f300"]
    exs = sorted(exf)
    vals = [exf[e] for e in exs]
    ax.bar([f"exch {e}" for e in exs], vals, color=[C.lgb, C.neural, C.lr][: len(exs)],
           edgecolor="white", zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("F(300) = win prob at max bid")
    ax.set_ylim(0, 0.8)
    ax.set_title(f"Exchange competition differs sharply\n(temporal drift KS={s['temporal_ks']:.3f}, "
                 f"best fit {s['best_parametric']})", fontsize=10.5, fontweight="bold")
    fig.suptitle("Right-censored market-price modeling (first-price win curve)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    _save(fig, "fig_market_cdf.png")


def fig_cate() -> None:
    d = _load("cate_fair.json")
    o = d["outcomes"]
    dec = d["decomposition"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    # Panel A: surplus decomposition (volume NIE vs cost NDE -> total)
    ax = axes[0]
    labs = ["volume\n(NIE)", "cost\n(NDE)", "total\nτ_surplus"]
    vals = [dec["NIE_volume"], dec["NDE_cost"], dec["total_surplus_tau"]]
    cols = [C.neg if v < 0 else C.pos for v in vals[:2]] + [C.neural]
    ax.bar(labs, vals, color=cols, edgecolor="white", zorder=3)
    ax.axhline(0, color=C.ink, lw=1)
    for i, v in enumerate(vals):
        ax.text(i, v + (1.5 if v >= 0 else -1.5), f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Δ surplus per higher bid level (CPM)")
    ax.set_title("Surplus contrast decomposes into\nnegative volume + positive cost channel",
                 fontsize=10.5, fontweight="bold")
    # Panel B: per-advertiser τ_surplus heterogeneity
    ax = axes[1]
    pa = d.get("per_advertiser", {})
    advs = sorted(pa, key=lambda a: pa[a]["tau_surplus"])
    ts = [pa[a]["tau_surplus"] for a in advs]
    ax.barh([f"adv {a}" for a in advs], ts, color=[C.neg if v < 0 else C.pos for v in ts],
            edgecolor="white", zorder=3)
    ax.axvline(0, color=C.ink, lw=1)
    for i, v in enumerate(ts):
        ax.text(v, i, f" {v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=9.5)
    ax.set_xlabel("τ_surplus per advertiser (CPM)")
    ax.set_title("Heterogeneous across advertisers\n(bid-varying advertisers only)",
                 fontsize=10.5, fontweight="bold")
    other = "  ".join(f"τ_{n}={o[n]['ate']:.3g}" for n in ("win", "payment", "click") if n in o)
    fig.suptitle("Bid-effect contrast — fair split  ·  EXPLORATORY naive difference-in-means "
                 "(confounded; won-only / flat-bid ceiling)\n" + other,
                 fontsize=10.5, fontweight="bold", y=1.04)
    _save(fig, "fig_cate.png")


def fig_scm() -> None:
    import ast
    d = _load("scm_fair.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    rows = []
    for o in ["bid_to_surplus", "bid_to_win"]:
        est = d[o]["estimate"]["estimates"]
        ci = d[o]["estimate"]["estimate_cis"]
        if isinstance(est, str):
            est = ast.literal_eval(est.replace("nan", "None"))
        if isinstance(ci, str):
            ci = ast.literal_eval(ci.replace("nan", "None"))
        v = est.get("backdoor.linear_regression")
        c = ci.get("backdoor.linear_regression")
        rows.append((o.replace("bid_to_", "bid → "), v, c))
    for i, (lab, v, c) in enumerate(rows):
        col = C.neg if v < 0 else C.pos
        if c:
            ax.plot([c[0], c[1]], [i, i], color=col, lw=3)
        ax.scatter([v], [i], color=col, s=90, zorder=3)
        ax.text(v, i + 0.16, f"{v:.4f}", ha="center", fontsize=10, color=col, fontweight="bold")
    ax.axvline(0, color=C.ink, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.set_ylim(-0.5, len(rows) - 0.3)
    ax.set_xlabel("DoWhy backdoor estimate (linear regression)")
    ax.set_title("Causal effect of bid", fontsize=11, fontweight="bold")
    ax = axes[1]
    ax.axis("off")
    lines = ["Refutation tests (robustness diagnostics)", ""]
    for o in ["bid_to_surplus", "bid_to_win"]:
        r = d[o]["refutation"]
        rob = r.get("is_robust")
        lines.append(f"{o.replace('bid_to_', 'bid→')}:  {'✓ robust' if rob else '✗'}")
        for t, vals in r["tests"].items():
            lines.append(f"   • {t}: {vals.get('new_effect'):.2e}")
        lines.append("")
    ax.text(0.02, 0.96, "\n".join(lines), va="top", ha="left", fontsize=10, family="monospace")
    fig.suptitle("SCM / DAG — bid → {surplus, win} (fair split)  ·  EXPLORATORY (not identified)",
                 fontsize=11.5, fontweight="bold", y=1.0)
    _save(fig, "fig_scm.png")


def main() -> None:
    _style()
    print(f"Generating portfolio figures -> {OUT.relative_to(ROOT)}/")
    figs = [
        fig_surplus_forest, fig_calibration_journey, fig_ablation_auc, fig_surplus_grid,
        fig_policy_value_decomp, fig_artifact_vs_fair,
        fig_ablation_ladder, fig_bidding_strategies, fig_pacing, fig_market_cdf, fig_cate, fig_scm,
    ]
    for f in figs:
        try:
            f()
        except FileNotFoundError as e:
            print(f"  SKIP {f.__name__}: missing {e.filename}")
        except Exception as e:  # noqa: BLE001 — keep generating the rest
            print(f"  ERROR {f.__name__}: {type(e).__name__}: {e}")
    print("done.")


if __name__ == "__main__":
    main()
