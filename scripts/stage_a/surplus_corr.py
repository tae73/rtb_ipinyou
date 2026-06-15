"""Stage A — A4: does simulated bid SURPLUS track calibration (IEB) or AUC?

Redesign thesis under test:
  H1 (thesis):  surplus tracks calibration  -> corr(IEB, -surplus) strongly positive
                (i.e. worse IEB -> worse surplus)
  H0 (null/alt): surplus tracks ranking (AUC) -> corr(AUC, surplus) strongly positive

For each model:
  - winners pCTR -> V(x) = pCTR * CPC_target (value.py)
  - exchange-conditional optimal first-price bids on won-only test data (shading.py)
  - run_auction_simulation + compute_simulation_metrics (simulator.py)
  - calibration metrics: IEB, ECE-quantile, plus uniform-bin ECE, on winners pCTR vs y_click
  - AUCs: all_bids_auc / winners_only_auc from easy_negatives.json

We ALSO add mis-scaled variants (pCTR * k for k in {0.5,1,2}) of one neural model.
Monotone scaling leaves ranking (AUC) UNCHANGED but sweeps IEB and V(x) -> surplus.
This is the cleanest discriminator between H1 and H0.

Everything is numpy / pandas (no jax). Reuses src.bidding + src.metrics, no reimpl.
"""

import json
from typing import Dict, List, NamedTuple, Tuple

import numpy as np

from src.bidding.shading import (
    MarketCDF,
    ShadingConfig,
    exchange_conditional_shading,
    load_exchange_cdfs,
    load_market_cdf,
)
from src.bidding.simulator import (
    compute_simulation_metrics,
    run_auction_simulation,
)
from src.bidding.value import ValueConfig, compute_impression_values
from src.metrics.evaluation import compute_ece, compute_ieb

ROOT = "/home/mail-agent/project/rtb_ipinyou"
NPZ = f"{ROOT}/results/stage_a/test_predictions_all.npz"
EASY = f"{ROOT}/results/stage_a/easy_negatives.json"
CDF_DIR = f"{ROOT}/results/market_price_cdf"
OUT = f"{ROOT}/results/stage_a/surplus_corr.json"

CPC_TARGET = 200_000.0


# ---------------------------------------------------------------------------
# Quantile ECE (uniform-bin ECE collapses for ~0.001 CTR; quantile bins fix it)
# ---------------------------------------------------------------------------

def compute_ece_quantile(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Quantile-binned Expected Calibration Error.

    Bins by predicted-probability quantiles so each bin has ~equal mass — the
    right tool when predictions concentrate near 0 (CTR ~1e-3), where uniform
    [0,1] bins would dump everything into bin 0 and report ~0 trivially.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_pred)
    # quantile edges on the prediction distribution; dedup for ties
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_pred, qs))
    if len(edges) < 2:
        return 0.0
    # assign each point to a bin in [0, len(edges)-2]
    idx = np.clip(np.searchsorted(edges, y_pred, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        in_bin = idx == b
        m = int(in_bin.sum())
        if m == 0:
            continue
        conf = float(y_pred[in_bin].mean())
        acc = float(y_true[in_bin].mean())
        ece += (m / n) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# Correlations (Spearman + Pearson), pure numpy
# ---------------------------------------------------------------------------

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks with tie handling (matches scipy.stats.rankdata 'average')."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i + 1
        while j < len(sa) and sa[j] == sa[i]:
            j += 1
        avg = (i + 1 + j) / 2.0  # ranks are 1-based
        ranks[order[i:j]] = avg
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


class ModelPred(NamedTuple):
    name: str
    p_ctr: np.ndarray   # winners pCTR on won-only subset
    is_variant: bool     # mis-scaled variant flag


def main() -> None:
    d = np.load(NPZ)
    easy = json.load(open(EASY))

    y_win = d["y_win"].astype(bool)
    payprice = d["payprice"].astype(np.float64)
    valid = y_win & (payprice > 0)
    n_valid = int(valid.sum())

    market_prices = payprice[valid]
    clicks = d["y_click"][valid].astype(np.int32)
    adexchange = d["adexchange"][valid].astype(np.int64)

    overall_cdf = load_market_cdf(f"{CDF_DIR}/km_cdf_overall.npz")
    exchange_cdfs = load_exchange_cdfs(CDF_DIR)

    # ---- assemble model set: real models (winners pCTR) + mis-scaled variants ----
    real = {
        "esmmwc": d["esmmwc_p_ctr"],
        "escm2wc_dr": d["escm2wc_dr_p_ctr"],
        "escm2wc_dr_extps": d["escm2wc_dr_extps_p_ctr"],
        "lr_ctr_all": d["lr_ctr_all"],
        "lgb_ctr_all": d["lgb_ctr_all"],
    }

    preds: List[ModelPred] = []
    for name, arr in real.items():
        preds.append(ModelPred(name, arr[valid].astype(np.float64), False))

    # Mis-scaled variants of escm2wc_dr (best-calibrated 3-tower) to sweep IEB
    # at FIXED ranking. k in {0.5, 1, 2}; k=1 is identical to the real model but
    # kept as an explicit anchor row.
    base = d["escm2wc_dr_p_ctr"][valid].astype(np.float64)
    for k in (0.5, 1.0, 2.0):
        preds.append(ModelPred(f"escm2wc_dr_scale{k:g}", np.clip(base * k, 0.0, 1.0), True))

    cfg_ex = ShadingConfig(strategy="optimal", exchange_conditional=True)
    val_cfg = ValueConfig(goal_type="CPC", cpc_target=CPC_TARGET)

    per_model: Dict[str, Dict[str, float]] = {}
    for mp in preds:
        p = mp.p_ctr
        # V(x) = pCTR * CPC_target
        v = compute_impression_values(p, val_cfg).values
        # exchange-conditional optimal first-price shading
        shade = exchange_conditional_shading(v, adexchange, exchange_cdfs, overall_cdf, cfg_ex)
        bids = shade.bids
        res = run_auction_simulation(bids, market_prices, v, clicks, auction_type="first_price")
        sim = compute_simulation_metrics(res, v, market_prices, mp.name, cpc_target=CPC_TARGET)

        # REALIZED surplus = model-independent true value minus spend on the bids the
        # model actually places. V_true(x) = y_click * CPC_target (only realized
        # clicks have economic value). This is the honest economic quantity; the
        # simulator's `total_surplus` uses the model's OWN (possibly inflated) V(x),
        # so an over-predicting model can book phantom surplus. We report BOTH.
        won = res.wins.astype(bool)
        realized_value = float((clicks[won] * CPC_TARGET).sum())
        realized_surplus = realized_value - float(res.payments[won].sum())

        ieb = compute_ieb(clicks, p)
        ece_q = compute_ece_quantile(clicks, p, n_bins=10)
        ece_u = compute_ece(clicks, p, n_bins=10)

        # AUCs from easy_negatives (real models only); variants share base ranking
        if mp.name in easy:
            all_auc = easy[mp.name]["all_bids_auc"]
            won_auc = easy[mp.name]["winners_only_auc"]
        elif mp.is_variant:
            all_auc = easy["escm2wc_dr"]["all_bids_auc"]
            won_auc = easy["escm2wc_dr"]["winners_only_auc"]
        else:
            all_auc = float("nan")
            won_auc = float("nan")

        per_model[mp.name] = {
            "is_variant": mp.is_variant,
            "surplus": sim.total_surplus,
            "realized_surplus": realized_surplus,
            "surplus_per_win": sim.avg_surplus_per_win,
            "roi": sim.roi,
            "overpay": sim.overpayment_ratio,
            "n_wins": sim.n_wins,
            "win_rate": sim.win_rate,
            "total_clicks": sim.total_clicks,
            "total_spend": sim.total_spend,
            "mean_value": float(np.mean(v)),
            "mean_bid": float(np.mean(bids)),
            "all_bids_auc": all_auc,
            "winners_only_auc": won_auc,
            "ece_quantile": ece_q,
            "ece_uniform": ece_u,
            "ieb": ieb,
        }

    # ---- correlations ----
    # Full set (real + variants) and real-only set, to be honest about what the
    # variants are driving. AUC corr only over rows with a finite AUC.
    def _vec(keys, field):
        return np.array([per_model[k][field] for k in keys], float)

    real_keys = [k for k, vv in per_model.items() if not vv["is_variant"]]
    all_keys = list(per_model.keys())

    def _corr_block(keys: List[str]) -> Dict[str, Dict[str, float]]:
        surplus = _vec(keys, "surplus")
        rsurplus = _vec(keys, "realized_surplus")
        ieb = _vec(keys, "ieb")
        all_auc = _vec(keys, "all_bids_auc")
        won_auc = _vec(keys, "winners_only_auc")
        ece_q = _vec(keys, "ece_quantile")

        # finite masks for AUC (variants share base AUC, so finite here)
        def block(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 2:
                return {"pearson": float("nan"), "spearman": float("nan"), "n": int(m.sum())}
            return {
                "pearson": _pearson(a[m], b[m]),
                "spearman": _spearman(a[m], b[m]),
                "n": int(m.sum()),
            }

        return {
            "n_models": len(keys),
            # THESIS: worse calibration (higher IEB) -> lower surplus => corr(IEB,-surplus)>0
            "ieb_vs_neg_surplus": block(ieb, -surplus),
            "ieb_vs_surplus": block(ieb, surplus),
            "ece_quantile_vs_neg_surplus": block(ece_q, -surplus),
            # NULL: surplus tracks ranking
            "all_bids_auc_vs_surplus": block(all_auc, surplus),
            "winners_only_auc_vs_surplus": block(won_auc, surplus),
            # REALIZED (model-independent value) — the honest economic check
            "ieb_vs_neg_realized_surplus": block(ieb, -rsurplus),
            "all_bids_auc_vs_realized_surplus": block(all_auc, rsurplus),
            "winners_only_auc_vs_realized_surplus": block(won_auc, rsurplus),
        }

    correlations = {
        "real_plus_variants": _corr_block(all_keys),
        "real_only": _corr_block(real_keys),
    }

    out = {
        "n_won_only": n_valid,
        "n_clicks_won_only": int(clicks.sum()),
        "true_ctr_won_only": float(clicks.mean()),
        "cpc_target": CPC_TARGET,
        "shading": "exchange_conditional optimal first-price (CDFs ex1/2/3, overall fallback)",
        "value_def": "V(x) = winners_pCTR * CPC_target",
        "variant_note": "escm2wc_dr_scale{0.5,1,2}: pCTR*k, monotone -> AUC fixed, sweeps IEB",
        "per_model": per_model,
        "correlations": correlations,
    }

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)

    # ---- console report ----
    print(f"won-only n={n_valid:,}  clicks={int(clicks.sum()):,}  true_ctr={clicks.mean():.6f}\n")
    hdr = ("model", "var", "surplus(V)", "real_surp", "roi", "overpay",
           "all_auc", "won_auc", "ece_q", "ieb", "win_rate")
    print("{:<22}{:<4}{:>14}{:>14}{:>7}{:>9}{:>9}{:>9}{:>9}{:>8}{:>9}".format(*hdr))
    for k, vv in per_model.items():
        print("{:<22}{:<4}{:>14.3e}{:>14.3e}{:>7.3f}{:>9.3f}{:>9}{:>9}{:>9.4f}{:>8.4f}{:>9.4f}".format(
            k, "V" if vv["is_variant"] else "-",
            vv["surplus"], vv["realized_surplus"], vv["roi"], vv["overpay"],
            f"{vv['all_bids_auc']:.4f}" if np.isfinite(vv["all_bids_auc"]) else "nan",
            f"{vv['winners_only_auc']:.4f}" if np.isfinite(vv["winners_only_auc"]) else "nan",
            vv["ece_quantile"], vv["ieb"], vv["win_rate"],
        ))

    print("\n--- correlations (real + variants) ---")
    for k, b in correlations["real_plus_variants"].items():
        if isinstance(b, dict) and "pearson" in b:
            print(f"  {k:<32} pearson={b['pearson']:+.3f} spearman={b['spearman']:+.3f} (n={b['n']})")
    print("\n--- correlations (real only) ---")
    for k, b in correlations["real_only"].items():
        if isinstance(b, dict) and "pearson" in b:
            print(f"  {k:<32} pearson={b['pearson']:+.3f} spearman={b['spearman']:+.3f} (n={b['n']})")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
