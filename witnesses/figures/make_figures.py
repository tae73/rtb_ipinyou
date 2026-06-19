"""Research figures (JSON -> PNG). Every value traces to witnesses/{phase_diagram,recal_trap}.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WIT = HERE.parent
ANCHOR_PURPLE, LINEAR_C, GBM_C, NEG, POS, INK, MUTE = "#4F46E5", "#6B7280", "#0D9488", "#DC2626", "#16A34A", "#111827", "#374151"
CAP_GREY = "#9CA3AF"


def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK,
                         "axes.linewidth": 0.8, "axes.grid": True, "grid.color": "#E5E7EB",
                         "grid.linewidth": 0.7, "axes.axisbelow": True, "figure.dpi": 150,
                         "savefig.dpi": 150, "savefig.bbox": "tight"})


def _save(fig, name):
    fig.savefig(HERE / name, facecolor="white")
    plt.close(fig)
    print("  wrote", (HERE / name).relative_to(WIT.parent))


def fig_phase_diagram():
    d = json.load(open(WIT / "phase_diagram.json"))
    cells, summ = d["cells"], d["summary"]
    gammas = sorted({c["gamma"] for c in cells})

    def edge_by_gamma(cap):
        return [np.mean([c["debias_edge_ipw_pp"] for c in cells if c["capacity"] == cap and c["gamma"] == g])
                for g in gammas]
    sd_lin = float(np.mean([c["debias_edge_ipw_sd"] for c in cells if c["capacity"] == "linear"]))
    sd_gbm = float(np.mean([c["debias_edge_ipw_sd"] for c in cells if c["capacity"] == "gbm"]))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3))
    # Panel A: within-capacity debiasing edge vs selection strength
    ax = axes[0]
    ax.plot(gammas, edge_by_gamma("linear"), "o-", color=POS, lw=2.4, ms=8, label="IPW-debias a WEAK (linear ≈ LR) model")
    ax.plot(gammas, edge_by_gamma("gbm"), "s-", color=NEG, lw=2.4, ms=8, label="IPW-debias a STRONG (gbm ≈ LGB) model")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("win-selection-bias strength  γ")
    ax.set_ylabel("WITHIN-CAPACITY debiasing edge (pp)\n>0 = debiasing improves the bid")
    ax.legend(fontsize=9.5, loc="upper left")
    ax.set_title("Debiasing helps a weak model (grows with γ),\nnot a strong one — capacity held fixed",
                 fontsize=11, fontweight="bold")
    # Panel B: honest headline + the capacity confound shown SEPARATELY
    ax = axes[1]
    vals = [summ["debias_edge_ipw_within_linear_pp"], summ["debias_edge_ipw_within_gbm_pp"], summ["capacity_gap_pp"]]
    cols = [POS, NEG, CAP_GREY]
    labels = ["debiasing\n(within linear)", "debiasing\n(within GBM)", "model CAPACITY\n(GBM>LR — NOT debiasing)"]
    bars = ax.bar(labels, vals, color=cols, edgecolor="white", width=0.66, zorder=3,
                  yerr=[sd_lin / np.sqrt(summ["n_seeds"]), sd_gbm / np.sqrt(summ["n_seeds"]), 0],
                  error_kw={"ecolor": INK, "capsize": 4, "lw": 1})
    bars[2].set_hatch("//")
    ax.axhline(0, color=INK, lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (1.0 if v >= 0 else -1.0), f"{v:+.1f}pp",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=11, fontweight="bold")
    ax.set_ylabel("mean edge (pp of decision-value regret)")
    ax.set_title("Honest decomposition: most of the apparent edge\nis model capacity, not debiasing",
                 fontsize=10.5, fontweight="bold")
    ax.text(0.5, -0.30, "real-world anchor — iPinYou fair split: robust vs LR, NOT robust vs LGB (I²=0.82)",
            transform=ax.transAxes, ha="center", fontsize=9, color=ANCHOR_PURPLE, style="italic")
    fig.suptitle("WHEN does win-selection-bias debiasing improve the bid? — within-capacity phase diagram",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, "fig_phase_diagram.png")


def fig_recal_trap():
    d = json.load(open(WIT / "recal_trap.json"))["cases"]["linear_strong"]
    models = ["biased", "biased_recal", "debiased"]
    labels = ["biased\n(winners-only)", "+ recalibration", "debiased\n(IPW)"]
    cols = [MUTE, NEG, POS]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
    for ax, key, ttl, fmt in [
        (axes[0], "mean_bid", "Mean bid  (recal inflates the bid)", lambda v: f"{v:.0f}"),
        (axes[1], "unprofitable_win_share", "Share of wins that are UNPROFITABLE\n(true value < clearing price)", lambda v: f"{v:.0%}"),
        (axes[2], "won_surplus", "Realized won surplus  (recal ↓, debiasing ↑)", lambda v: f"{v/1e6:.1f}M"),
    ]:
        vals = [d[m][key] for m in models]
        bars = ax.bar(labels, vals, color=cols, edgecolor="white", width=0.62, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, "  " + fmt(v), ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold")
        ax.set_title(ttl, fontsize=10, fontweight="bold")
        ax.margins(y=0.16)
        if key == "won_surplus":
            ax.axhline(d["oracle_surplus"], ls="--", color=ANCHOR_PURPLE, lw=1.2)
            ax.text(2.4, d["oracle_surplus"], f" oracle {d['oracle_surplus']/1e6:.1f}M", color=ANCHOR_PURPLE,
                    fontsize=8.5, va="bottom", ha="right")
    fig.suptitle("The recalibration trap — naive recalibration over-bids marginal inventory "
                 "(strong selection, linear baseline; debiaser = IPW at same capacity)",
                 fontsize=11.5, fontweight="bold", y=1.03)
    _save(fig, "fig_recal_trap.png")


def fig_neural_anchor():
    p = WIT / "neural_anchor.json"
    if not p.exists():
        print("  (skip fig_neural_anchor — neural_anchor.json not found)")
        return
    j = json.load(open(p))
    s = j["summary"]
    frozen = j["_meta"]["frozen_prefix_result"]["truthful_edge_neural_by_gamma"]
    gammas = sorted(float(g) for g in s["truthful_edge_neural_by_gamma"])
    gk = lambda d, g: d[f"{g:g}"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    # Panel A: the -47pp was a WIRING BUG — censoring click (click*win) fixes the over-bidding
    ax = axes[0]
    unc = [gk(frozen, g) for g in gammas]
    cen = [gk(s["truthful_edge_neural_by_gamma"], g) for g in gammas]
    ax.plot(gammas, unc, "s--", color=NEG, lw=2.2, ms=8, label="uncensored click (the bug)")
    ax.plot(gammas, cen, "o-", color=POS, lw=2.4, ms=8, label="censored click·win (the fix)")
    ax.axhline(0, color=INK, lw=1)
    for g, u, c in zip(gammas, unc, cen):
        ax.text(g, u - 6, f"{u:+.0f}", ha="center", va="top", fontsize=8.5, color=NEG, fontweight="bold")
        ax.text(g, c + 4, f"{c:+.1f}", ha="center", va="bottom", fontsize=8.5, color=POS, fontweight="bold")
    ax.set_xlabel("win-selection-bias strength  γ")
    ax.set_ylabel("neural TRUTHFUL-bid edge (pp)")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("The −47pp over-bidding was a WIRING BUG:\ncensoring click (click·win) fixes it",
                 fontsize=10, fontweight="bold")
    # Panel B: the intuitive fix (post-hoc calibration) does NOT help — IPW-cal even HURTS at strong selection
    ax = axes[1]
    raw = [gk(s["truthful_edge_neural_by_gamma"], g) for g in gammas]
    ipw = [gk(s["truthful_edge_neural_ipwcal_by_gamma"], g) for g in gammas]
    nai = [gk(s["truthful_edge_neural_naivecal_by_gamma"], g) for g in gammas]
    ax.plot(gammas, raw, "o-", color=POS, lw=2.4, ms=8, label="debiased (censored, no cal)")
    ax.plot(gammas, ipw, "D-", color=GBM_C, lw=2.2, ms=7, label="+ IPW-weighted calibration")
    ax.plot(gammas, nai, "v-", color=LINEAR_C, lw=2.0, ms=7, label="+ naive calibration")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("win-selection-bias strength  γ")
    ax.set_ylabel("neural TRUTHFUL-bid edge (pp)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Calibration further helps (+7.5→+11pp avg) —\nbut IPW-cal's gain vanishes at the strongest selection (ESS↓)",
                 fontsize=10, fontweight="bold")
    fig.suptitle("Neural anchor (corrected) — the over-bidding was a censoring bug; fixed, the ESCM²-WC helps truthfully (+7.5pp)",
                 fontsize=11.5, fontweight="bold", y=1.03)
    _save(fig, "fig_neural_anchor.png")


def main():
    _style()
    print("research figures ->", HERE.relative_to(WIT.parent))
    fig_phase_diagram()
    fig_recal_trap()
    fig_neural_anchor()
    print("done.")


if __name__ == "__main__":
    main()
