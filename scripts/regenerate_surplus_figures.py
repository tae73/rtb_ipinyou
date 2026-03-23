"""Regenerate calibration economic value + overbidding figures with retrained IEB values."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("results/figures")
CDF_DIR = Path("results/market_price_cdf")

# === Retrained IEB values (source of truth) ===
ACTUAL_CTR = 0.0008
CPC_TARGET = 200_000
V_TRUE = ACTUAL_CTR * CPC_TARGET  # 160 CPM

MODEL_SPECS = [
    ("ESCM²-WC(DR) AL", 0.073, "#2ca02c"),
    ("ESCM²-WC(DR)+ExtPS AW", 0.045, "#1f77b4"),
    ("LR CTR_all", 0.122, "#9467bd"),
    ("LGB CTR (biased)", 0.362, "#ff7f0e"),
    ("ESMM-WC J", 0.075, "#d62728"),
]


def find_optimal_bid(V, price_grid, cdf, n_points=5000):
    """Find b* = argmax_b (V - b) * F(b) subject to b <= V."""
    bid_range = np.linspace(1, min(V, price_grid.max()), n_points)
    f_b = np.interp(bid_range, price_grid, cdf)
    surplus = (V - bid_range) * f_b
    idx = np.argmax(surplus)
    return bid_range[idx], surplus[idx], f_b[idx]


def generate_calibration_economic_value():
    """Figure 18: 2-panel surplus comparison."""
    # Load KM CDF
    overall = np.load(CDF_DIR / "km_cdf_overall.npz")
    price_grid = overall["price_grid"]
    cdf = overall["cdf"]

    # Oracle
    b_oracle, s_oracle, _ = find_optimal_bid(V_TRUE, price_grid, cdf)
    print(f"Oracle: b*={b_oracle:.1f}, surplus={s_oracle:.2f} CPM")

    # Overall results
    overall_results = []
    for name, ieb, color in MODEL_SPECS:
        V_model = ACTUAL_CTR * (1 + ieb) * CPC_TARGET
        b_star, _, f_star = find_optimal_bid(V_model, price_grid, cdf)
        true_surplus = (V_TRUE - b_star) * f_star
        loss_pct = (s_oracle - true_surplus) / s_oracle * 100 if s_oracle > 0 else 0
        overall_results.append({
            "name": name, "ieb": ieb, "V": V_model, "b_star": b_star,
            "shade": b_star / V_model, "f_star": f_star,
            "true_surplus": true_surplus, "loss_pct": loss_pct, "color": color,
        })
        print(f"{name}: V={V_model:.1f}, b*={b_star:.1f}, "
              f"surplus={true_surplus:.2f} (loss={loss_pct:.1f}%)")

    # Exchange-conditional
    exchange_cdfs = {}
    for ex_id in [1, 2, 3]:
        ex_data = np.load(CDF_DIR / f"km_cdf_exchange_{ex_id}.npz")
        exchange_cdfs[ex_id] = (ex_data["price_grid"], ex_data["cdf"])

    print("\n=== Exchange-Conditional Surplus ===")
    ex_results = {}
    for ex_id, (ex_pg, ex_cdf) in sorted(exchange_cdfs.items()):
        b_or, s_or, _ = find_optimal_bid(V_TRUE, ex_pg, ex_cdf)
        ex_results[ex_id] = {"oracle_surplus": s_or, "models": {}}
        for name, ieb, color in MODEL_SPECS:
            V_m = ACTUAL_CTR * (1 + ieb) * CPC_TARGET
            b_m, _, f_m = find_optimal_bid(V_m, ex_pg, ex_cdf)
            ts = (V_TRUE - b_m) * f_m
            loss = (s_or - ts) / s_or * 100 if s_or > 0 else 0
            ex_results[ex_id]["models"][name] = {
                "b_star": b_m, "true_surplus": ts, "loss_pct": loss,
            }
        parts = [f"{n.split(' ')[0]}={ex_results[ex_id]['models'][n]['true_surplus']:.2f} "
                 f"(-{ex_results[ex_id]['models'][n]['loss_pct']:.0f}%)"
                 for n, _, _ in MODEL_SPECS]
        print(f"Ex{ex_id}: oracle={s_or:.2f} | " + " | ".join(parts))

    # === 2-Panel Figure ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Overall
    ax = axes[0]
    names_display = [r["name"].replace("ESCM²-WC(DR)", "ESCM²\nDR").replace("+ExtPS ", "+ExtPS\n")
                     for r in overall_results]
    colors = [r["color"] for r in overall_results]
    surpluses = [r["true_surplus"] for r in overall_results]

    x_pos = np.arange(len(overall_results))
    ax.bar(x_pos, surpluses, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(s_oracle, color="gray", linestyle="--", linewidth=1,
               label=f"Oracle: {s_oracle:.2f} CPM")

    for i, r in enumerate(overall_results):
        ax.text(i, r["true_surplus"] + 0.3,
                f'b*={r["b_star"]:.0f}\nIEB={r["ieb"]:.3f}',
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names_display, fontsize=8)
    ax.set_ylabel("Expected Surplus (CPM, true V basis)")
    ax.set_title(f"Overall: Optimal Bid Surplus (V_true={V_TRUE:.0f} CPM)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(surpluses) * 1.4)

    # Right: Exchange-conditional
    ax2 = axes[1]
    ex_ids = sorted(ex_results.keys())
    n_models = len(MODEL_SPECS)
    width = 0.15
    x_ex = np.arange(len(ex_ids))

    for j, (name, ieb, color) in enumerate(MODEL_SPECS):
        vals = [ex_results[ex]["models"][name]["true_surplus"] for ex in ex_ids]
        short_label = name.split(" ")[0] + " " + name.split(" ")[-1]
        ax2.bar(x_ex + j * width, vals, width, color=color, alpha=0.8,
                edgecolor="black", linewidth=0.5, label=short_label)

    oracle_vals = [ex_results[ex]["oracle_surplus"] for ex in ex_ids]
    ax2.scatter(x_ex + (n_models - 1) * width / 2, oracle_vals, marker="*", s=100,
                color="gold", edgecolor="black", zorder=5, label="Oracle")

    # Annotate max loss model per exchange
    for i, ex in enumerate(ex_ids):
        max_loss_name = max(ex_results[ex]["models"],
                           key=lambda n: ex_results[ex]["models"][n]["loss_pct"])
        loss = ex_results[ex]["models"][max_loss_name]["loss_pct"]
        if loss > 0.5:
            y_pos = ex_results[ex]["models"][max_loss_name]["true_surplus"]
            ax2.text(i + (n_models - 1) * width, y_pos + 0.5, f"-{loss:.0f}%",
                     ha="center", va="bottom", fontsize=8, color="red", fontweight="bold")

    ex_labels = {1: "Ex1\n(F300=69%)", 2: "Ex2\n(F300=29%)", 3: "Ex3\n(F300=12%)"}
    ax2.set_xticks(x_ex + (n_models - 1) * width / 2)
    ax2.set_xticklabels([ex_labels.get(ex, f"Ex{ex}") for ex in ex_ids], fontsize=9)
    ax2.set_ylabel("Expected Surplus (CPM, true V basis)")
    ax2.set_title("Exchange-Conditional: Miscalibration Impact")
    ax2.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    save_path = FIG_DIR / "05_calibration_economic_value.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {save_path}")


def generate_overbidding_cumulative():
    """Figure 17: Cumulative overbidding bar chart."""
    actual_ctr = 0.0008
    value_per_click = 10_000
    n_bids = 1_000_000

    ieb_values = [0.045, 0.073, 0.075, 0.122, 0.362]
    ieb_labels = ["AW (0.045)", "AL (0.073)", "J (0.075)",
                  "LR (0.122)", "LGB (0.362)"]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

    totals = [n_bids * actual_ctr * ieb * value_per_click for ieb in ieb_values]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(ieb_values))
    bars = ax.bar(x_pos, totals, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    for i, (bar, total) in enumerate(zip(bars, totals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20000,
                f"{total:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(ieb_labels, fontsize=10)
    ax.set_ylabel("Cumulative Overbid (CPM)")
    ax.set_title(f"Overbidding Cost by IEB ({n_bids / 1e6:.0f}M bids, "
                 f"value_per_click={value_per_click:,})")
    plt.tight_layout()

    save_path = FIG_DIR / "04_overbidding_simulation.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_overbidding_per_auction():
    """Figure 16: Per-auction overbidding 3-panel."""
    CPC_TARGET_SIM = 1000
    N_AUCTIONS = 10000
    np.random.seed(42)

    model_ieb = {
        "ESCM2-WC(DR)": 0.073,
        "LR CTR_all": 0.122,
        "LGB CTR": 0.362,
    }

    true_ctr = np.random.lognormal(mean=-7.5, sigma=1.0, size=N_AUCTIONS)
    true_ctr = np.clip(true_ctr, 1e-6, 0.05)
    market_prices = np.random.uniform(30, 120, size=N_AUCTIONS)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, (model_name, ieb) in enumerate(model_ieb.items()):
        ax = axes[ax_idx]
        bias_factor = 1 + ieb
        pred_ctr = true_ctr * bias_factor

        true_value = true_ctr * CPC_TARGET_SIM
        pred_value = pred_ctr * CPC_TARGET_SIM

        wins = pred_value >= market_prices
        win_rate = wins.mean()

        bid_error = (pred_value - true_value)
        roi = np.where(wins, true_value - market_prices, 0).sum() / np.where(wins, market_prices, 1).sum()

        ax.scatter(true_value, pred_value, alpha=0.1, s=1, color="C0")
        max_v = max(true_value.max(), pred_value.max())
        ax.plot([0, max_v], [0, max_v], "k--", alpha=0.5, label="Perfect")
        ax.plot([0, max_v], [0, max_v * bias_factor], "r-", alpha=0.7,
                label=f"IEB bias ({ieb:.1%})")

        ax.set_xlabel("True V(x) = CTR × CPC")
        ax.set_ylabel("Predicted V(x)")
        ax.set_title(f"{model_name}\nIEB={ieb:.3f}, Bid Error={ieb:.1%}")
        ax.legend(fontsize=7, loc="upper left")
        ax.text(0.95, 0.05, f"Win rate: {win_rate:.1%}\nROI: {roi:+.0%}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.suptitle(f"Overbidding Simulation: IEB → Bid Error → ROI Impact\n"
                 f"(N={N_AUCTIONS:,} auctions, CPC_target={CPC_TARGET_SIM} CPM)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_path = FIG_DIR / "03_overbidding_simulation.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print("=== Calibration Economic Value (Figure 18) ===")
    generate_calibration_economic_value()
    print("\n=== Overbidding Cumulative (Figure 17) ===")
    generate_overbidding_cumulative()
    print("\n=== Overbidding Per-Auction (Figure 16) ===")
    generate_overbidding_per_auction()
    print("\nDone.")
