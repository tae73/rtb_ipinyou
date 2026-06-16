"""Regenerate NB03/NB04 figures from result JSONs (retrained values)."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_DIR = Path("results/models")
FIG_DIR = Path("results/figures")


def load_result(name: str) -> dict:
    path = MODEL_DIR / name
    return json.load(open(path))


# === Retrained result JSONs ===
J = load_result("esmmwc_result.json")
AL = load_result("escm2wc_dr_result.json")
AW = load_result("escm2wc_dr_extps_result.json")

# Baseline JSONs have nested test_metrics — flatten
def _flatten_baseline(name: str) -> dict:
    raw = load_result(name)
    tm = raw.get("test_metrics", {})
    return {"test_auc": tm.get("auc", 0), "test_ece": tm.get("ece", 0),
            "test_ieb": tm.get("ieb", 0), **raw}

LGB_CTR = _flatten_baseline("lgb_ctr_result.json")
LGB_WIN = _flatten_baseline("lgb_win_result.json")
LGB_ALL = _flatten_baseline("lgb_ctr_all_result.json")
LR_CTR = _flatten_baseline("lr_ctr_result.json")
LR_ALL = _flatten_baseline("lr_ctr_all_result.json")

# Ablation runs (original training, not retrained)
AK = load_result("escm2wc_dr_result_AK.json")  # cfr=0.0
AL_orig = load_result("escm2wc_dr_result_AL.json")  # cfr=0.2 original
AQ = load_result("escm2wc_dr_result_AQ.json")  # cfr=0.3
AR = load_result("escm2wc_dr_result_AR.json")  # cfr=0.5
AJ = load_result("escm2wc_dr_result_AJ.json")  # Huber
AM = load_result("escm2wc_dr_result_AM.json")  # per-tower dropout
AP = load_result("escm2wc_dr_result_AP.json")  # checkpoint avg
AV = load_result("escm2wc_dr_result_AV.json")  # ExtPS variant
AW_orig = load_result("escm2wc_dr_result_AW.json")  # ExtPS original


def fig_auc_ece_comparison():
    """Figure 4: 4-panel AUC/ECE comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Won-only CTR
    won_models = {
        "lgb_ctr": (LGB_CTR["test_auc"], LGB_CTR.get("test_ece", 0)),
        "lr_ctr": (LR_CTR["test_auc"], LR_CTR.get("test_ece", 0)),
        "esmmwc\nCTR (biased)": (J["test_ctr_biased_auc"], 0),
        "escm2wc_dr\nCTR (biased)": (AL["test_ctr_biased_auc"], 0),
    }

    # All-bids CTR
    all_models = {
        "lgb_ctr_all": (LGB_ALL["test_auc"], LGB_ALL.get("test_ece", 0)),
        "lr_ctr_all": (LR_ALL["test_auc"], LR_ALL.get("test_ece", 0)),
        "esmmwc\nWCTR": (J["test_wctr_auc"], J["test_wctr_ece"]),
        "escm2wc_dr\nWCTR": (AL["test_wctr_auc"], AL["test_wctr_ece"]),
    }

    colors = ["#2ca02c", "#9467bd", "#1f77b4", "#ff7f0e"]

    for row, (title_prefix, models) in enumerate([
        ("Won-Only CTR", won_models), ("All-Bids CTR", all_models)
    ]):
        names = list(models.keys())
        aucs = [v[0] for v in models.values()]
        eces = [v[1] for v in models.values()]

        # AUC panel
        ax = axes[row][0]
        bars = ax.bar(range(len(names)), aucs, color=colors, alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        for i, v in enumerate(aucs):
            ax.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("AUC")
        ax.set_title(f"AUC — {title_prefix}")

        # ECE panel
        ax = axes[row][1]
        ece_nonzero = [max(e, 1e-7) for e in eces]
        bars = ax.bar(range(len(names)), ece_nonzero, color=colors, alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        for i, v in enumerate(eces):
            if v > 0:
                ax.text(i, v * 1.5, f"{v:.1e}", ha="center", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_yscale("log")
        ax.set_ylabel("ECE (log scale)")
        suffix = "*" if row == 1 else ""
        ax.set_title(f"ECE — {title_prefix}{suffix}")

    plt.tight_layout()
    path = FIG_DIR / "03_auc_ece_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_ieb_comparison():
    """IEB comparison bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Won-only IEB
    won_ieb = {
        "lgb_ctr": LGB_CTR.get("test_ieb", 0),
        "lr_ctr": LR_CTR.get("test_ieb", 0),
        "esmmwc\nCTR (biased)": J["test_ctr_ieb"],
        "escm2wc_dr\nCTR (biased)": AL["test_ctr_ieb"],
    }

    # All-bids IEB
    all_ieb = {
        "lgb_ctr": LGB_CTR.get("test_ieb", 0),
        "lr_ctr_all": LR_ALL.get("test_ieb", 0.122),
        "esmmwc\nWCTR": J["test_wctr_ieb"],
        "escm2wc_dr\nWCTR": AL["test_wctr_ieb"],
    }

    colors = ["#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]

    for ax_idx, (title, data) in enumerate([
        ("IEB Comparison — Won-Only CTR", won_ieb),
        ("IEB Comparison — All-Bids CTR", all_ieb),
    ]):
        ax = axes[ax_idx]
        names = list(data.keys())
        vals = list(data.values())
        bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.8,
                      edgecolor="black", linewidth=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("IEB (lower is better)")
        ax.set_title(title)
        ax.axhline(y=0.1, color="green", linestyle="--", alpha=0.5,
                   label="IEB<0.1 threshold")
        ax.legend(fontsize=8)

    best_model = "ESCM2-WC(DR)"
    best_ieb = AL["test_wctr_ieb"]
    plt.suptitle(f"Inherent Estimation Bias — {best_model} IEB={best_ieb:.3f}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "03_ieb_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_ablation_ladder():
    """Figure 5: Debiasing ablation ladder."""
    models = [
        ("LGB CTR_all\n(baseline)", LGB_ALL["test_auc"], None, "#888888"),
        ("ESMM-WC\n(Run J)", J["test_wctr_auc"], J["test_wctr_ieb"], "#1f77b4"),
        ("ESCM²-WC DR\n(Run AL)", AL["test_wctr_auc"], AL["test_wctr_ieb"], "#2ca02c"),
        ("ESCM²-WC DR+ExtPS\n(Run AW)", AW["test_wctr_auc"], AW["test_wctr_ieb"], "#ff7f0e"),
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    names = [m[0] for m in models]
    aucs = [m[1] for m in models]
    colors = [m[3] for m in models]

    y_pos = np.arange(len(models))
    bars = ax.barh(y_pos, aucs, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    for i, m in enumerate(models):
        ieb_text = f"IEB={m[2]:.3f}" if m[2] is not None else "IEB=N/A"
        ax.text(m[1] + 0.002, i, f"AUC={m[1]:.4f} ({ieb_text})",
                va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("WCTR AUC")
    ax.set_title("Debiasing Ablation Ladder — WCTR AUC Progression")
    ax.set_xlim(0.5, 0.75)
    ax.invert_yaxis()
    plt.tight_layout()

    path = FIG_DIR / "04_debiasing_ablation_ladder.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_auc_vs_calibration():
    """Figure 6: AUC vs Calibration trade-off scatter."""
    runs = [
        ("J (ESMM-WC)", J["test_wctr_auc"], J["test_wctr_ieb"], "D", "C3", 80),
        ("AL (DR cfr=0.2)", AL["test_wctr_auc"], AL["test_wctr_ieb"], "*", "C2", 150),
        ("AW (DR+ExtPS)", AW["test_wctr_auc"], AW["test_wctr_ieb"], "*", "C0", 150),
        ("AK (cfr=0)", AK["test_wctr_auc"], AK["test_wctr_ieb"], "^", "C4", 60),
        ("AJ (Huber)", AJ["test_wctr_auc"], AJ["test_wctr_ieb"], "^", "C5", 60),
        ("AQ (cfr=0.3)", AQ["test_wctr_auc"], AQ["test_wctr_ieb"], "s", "C6", 60),
        ("AR (cfr=0.5)", AR["test_wctr_auc"], AR["test_wctr_ieb"], "s", "C7", 60),
        ("AM (per-tower DO)", AM["test_wctr_auc"], AM["test_wctr_ieb"], "v", "C8", 60),
        ("AP (ckpt avg)", AP["test_wctr_auc"], AP["test_wctr_ieb"], "v", "C9", 60),
        ("AV (J+ExtPS)", AV["test_wctr_auc"], AV["test_wctr_ieb"], "o", "C1", 60),
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    for name, auc, ieb, marker, color, size in runs:
        ax.scatter(auc, ieb, marker=marker, c=color, s=size, edgecolors="black",
                   linewidth=0.5, zorder=5)
        ax.annotate(name, (auc, ieb), fontsize=7, xytext=(5, 5),
                    textcoords="offset points")

    # Pareto frontier (AL and AW)
    pareto_x = [AL["test_wctr_auc"], AW["test_wctr_auc"]]
    pareto_y = [AL["test_wctr_ieb"], AW["test_wctr_ieb"]]
    ax.plot(pareto_x, pareto_y, "g--", alpha=0.5, label="Pareto frontier")

    ax.set_xlabel("WCTR AUC (higher is better)")
    ax.set_ylabel("WCTR IEB (lower is better, log scaled)")
    ax.set_yscale("log")
    ax.set_title("AUC vs Calibration Trade-off — All ESCM2-WC Variants")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = FIG_DIR / "04_auc_vs_calibration_tradeoff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_cfr_lambda_ablation():
    """Figure 9: CFR lambda ablation (AL retrained, others original)."""
    cfr_data = [
        (0.0, AK["test_wctr_auc"], AK["test_wctr_ieb"]),
        (0.2, AL["test_wctr_auc"], AL["test_wctr_ieb"]),  # retrained
        (0.3, AQ["test_wctr_auc"], AQ["test_wctr_ieb"]),
        (0.5, AR["test_wctr_auc"], AR["test_wctr_ieb"]),
    ]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    cfrs = [d[0] for d in cfr_data]
    aucs = [d[1] for d in cfr_data]
    iebs = [d[2] for d in cfr_data]

    ax1.plot(cfrs, aucs, "o--", color="C0", label="WCTR AUC", markersize=8)
    ax1.set_xlabel("CFR Lambda")
    ax1.set_ylabel("WCTR AUC (higher is better)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    for i, d in enumerate(cfr_data):
        label = "AL*" if d[0] == 0.2 else ""
        ax1.annotate(f"AUC={d[1]:.4f}", (d[0], d[1]), fontsize=7,
                     xytext=(5, 10), textcoords="offset points")

    ax2 = ax1.twinx()
    ax2.plot(cfrs, iebs, "s--", color="C3", label="WCTR IEB", markersize=8)
    ax2.set_ylabel("WCTR IEB (lower is better, log scale)", color="C3")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="C3")

    # Highlight sweet spot
    ax1.axvspan(0.15, 0.25, alpha=0.1, color="green", label="Sweet spot")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax1.set_title("CFR Lambda Ablation: AUC vs Calibration Trade-off\n"
                  "(*AL = retrained, others = original training)")
    plt.tight_layout()

    path = FIG_DIR / "04_cfr_lambda_ablation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_external_ps_impact():
    """ExtPS impact comparison."""
    models = [
        ("AL\n(internal PS)", AL["test_wctr_auc"], AL["test_wctr_ieb"], "#2ca02c"),
        ("AW\n(external PS)", AW["test_wctr_auc"], AW["test_wctr_ieb"], "#1f77b4"),
        ("AV\n(J cfg+ExtPS)", AV["test_wctr_auc"], AV["test_wctr_ieb"], "#9467bd"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = [m[0] for m in models]
    colors = [m[3] for m in models]

    # AUC
    ax = axes[0]
    aucs = [m[1] for m in models]
    ax.bar(range(len(models)), aucs, color=colors, alpha=0.8,
           edgecolor="black", linewidth=0.5)
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("WCTR AUC")
    ax.set_title("External PS Impact — AUC")
    ax.set_ylim(0.65, 0.70)

    # IEB
    ax = axes[1]
    iebs = [m[2] for m in models]
    ax.bar(range(len(models)), iebs, color=colors, alpha=0.8,
           edgecolor="black", linewidth=0.5)
    for i, v in enumerate(iebs):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("WCTR IEB (lower is better)")
    ax.set_title("External PS Impact — IEB")

    plt.suptitle("External Win PS Impact on ESCM²-WC(DR)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = FIG_DIR / "04_external_ps_impact.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_training_dynamics():
    """Figure 8: Training dynamics from result JSON training_history."""
    # Check if training_history exists
    j_hist = J.get("training_history")
    al_hist = AL.get("training_history")

    if not j_hist or not al_hist:
        print("[SKIP] 04_training_dynamics.png — no training_history in result JSONs")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    def plot_loss(ax, key, title, j_data, al_data):
        j_vals = [e.get(key) for e in j_data if e.get(key) is not None]
        al_vals = [e.get(key) for e in al_data if e.get(key) is not None]
        if j_vals:
            ax.plot(range(1, len(j_vals) + 1), j_vals, "o-", label="J (ESMM-WC)", color="C1")
        if al_vals:
            ax.plot(range(1, len(al_vals) + 1), al_vals, "s-", label="AL (DR)", color="C2")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend(fontsize=7)

    plot_loss(axes[0][0], "val_win", "Win Loss (val)", j_hist, al_hist)
    plot_loss(axes[0][1], "val_ctr", "CTR Loss (val)", j_hist, al_hist)
    plot_loss(axes[0][2], "val_joint", "Joint Loss (val)", j_hist, al_hist)

    # AL-only: CFR, Imputation
    al_cfr = [e.get("val_cfr") for e in al_hist if e.get("val_cfr") is not None]
    if al_cfr:
        axes[1][0].plot(range(1, len(al_cfr) + 1), al_cfr, "s-", color="C2")
        axes[1][0].set_title("CFR Loss — AL only (val)")
        axes[1][0].set_xlabel("Epoch")
        axes[1][0].set_yscale("log")

    al_imp = [e.get("val_impute") for e in al_hist if e.get("val_impute") is not None]
    if al_imp:
        axes[1][1].plot(range(1, len(al_imp) + 1), al_imp, "s-", color="C2")
        axes[1][1].set_title("Imputation Loss — AL only (val)")
        axes[1][1].set_xlabel("Epoch")

    # Val AUC trajectories
    ax = axes[1][2]
    for data, name, color, marker in [(j_hist, "J WCTR", "C1", "o"),
                                       (al_hist, "AL WCTR", "C2", "s")]:
        vals = [e.get("val_wctr_auc") or e.get("val_ctr_auc") for e in data]
        vals = [v for v in vals if v is not None]
        if vals:
            ax.plot(range(1, len(vals) + 1), vals, f"{marker}-", label=name, color=color)
    ax.set_title("Validation AUC Trajectories")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=7)

    plt.suptitle("Training Dynamics: Run J (ESMM-WC) vs Run AL (ESCM2-WC DR)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = FIG_DIR / "04_training_dynamics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("=== NB03 Figures ===")
    fig_auc_ece_comparison()
    fig_ieb_comparison()

    print("\n=== NB04 Figures ===")
    fig_ablation_ladder()
    fig_auc_vs_calibration()
    fig_cfr_lambda_ablation()
    fig_external_ps_impact()
    fig_training_dynamics()

    print("\nDone.")
