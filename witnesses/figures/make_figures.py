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
        return [np.mean([c["edge_vs_baseline_pp"] for c in cells if c["baseline"] == cap and c["gamma"] == g])
                for g in gammas]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    ax.plot(gammas, edge_by_gamma("linear"), "o-", color=LINEAR_C, lw=2.4, ms=8, label="vs LINEAR baseline (≈ LR)")
    ax.plot(gammas, edge_by_gamma("gbm"), "s-", color=GBM_C, lw=2.4, ms=8, label="vs GBM baseline (≈ LGB)")
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("win-selection-bias strength  γ")
    ax.set_ylabel("debiasing edge (pp of decision-value regret)\n>0 = debiasing wins the bid")
    ax.legend(fontsize=9.5, loc="upper left")
    ax.set_title("Debiasing's bid value depends on COMPETITOR strength", fontsize=11.5, fontweight="bold")
    # panel B: the asymmetry headline + iPinYou anchor
    ax = axes[1]
    bars = ax.bar(["vs LINEAR\n(≈ LR)", "vs GBM\n(≈ LGB)"],
                  [summ["mean_edge_vs_linear_pp"], summ["mean_edge_vs_gbm_pp"]],
                  color=[POS, NEG], edgecolor="white", width=0.6, zorder=3)
    ax.axhline(0, color=INK, lw=1)
    for b, v in zip(bars, [summ["mean_edge_vs_linear_pp"], summ["mean_edge_vs_gbm_pp"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.8 if v >= 0 else -0.8), f"{v:+.1f}pp",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=12, fontweight="bold")
    ax.set_ylabel("mean debiasing edge (pp)")
    ax.set_title("Reproduces the real iPinYou result:\nrobust vs LR · NOT robust vs LGB (I²=0.82)",
                 fontsize=11, fontweight="bold")
    ax.text(0.5, 0.04, "◆ real-world anchor: iPinYou fair split", transform=ax.transAxes,
            ha="center", fontsize=9, color=ANCHOR_PURPLE, style="italic")
    fig.suptitle("WHEN does win-selection-bias debiasing improve the bid? — a controllable phase diagram",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, "fig_phase_diagram.png")


def fig_recal_trap():
    d = json.load(open(WIT / "recal_trap.json"))["cases"]["linear_strong"]
    models = ["biased", "biased_recal", "debiased"]
    labels = ["biased\n(winners-only)", "biased\n+ recalibration", "debiased\n(DR)"]
    cols = [MUTE, NEG, POS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
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
                 "(strong selection, linear baseline)", fontsize=12, fontweight="bold", y=1.03)
    _save(fig, "fig_recal_trap.png")


def main():
    _style()
    print("research figures ->", HERE.relative_to(WIT.parent))
    fig_phase_diagram()
    fig_recal_trap()
    print("done.")


if __name__ == "__main__":
    main()
