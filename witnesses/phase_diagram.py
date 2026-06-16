"""Phase diagram — WHEN does win-selection-bias debiasing improve the BID (decision value)?

The flagship wedge is the COMPETITOR-MODEL-STRENGTH axis. In a controllable semi-synthetic RTB testbed
(ground-truth pCTR + OBSERVABLE lost inventory; market price calibrated to iPinYou observed stats), we
fix a strong corrected debiaser (GBM + IPW, ~the iPinYou neural model) and vary the BASELINE capacity it
competes against — {linear logistic ≈ LR, GBM ≈ LGB}. Sweeping selection strength γ and heterogeneity θ
we measure the debiasing EDGE = regret(baseline) − regret(debiaser) on FULL inventory.

Reproduces the iPinYou asymmetry controllably: edge over a WEAK (linear) baseline should be large; edge
over a STRONG (GBM) baseline should shrink / heterogenize — the real-world "robust vs LR, not vs LGB"
(I²=0.82) is the negative half of this diagram. Honest: we report where debiasing does NOT help.

Output: witnesses/phase_diagram.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "phase_diagram.json"
RNG = np.random.default_rng(0)
N, D, CPC = 120_000, 8, 3000.0           # CPC s.t. mean V ~ market median (~68)
BETA = RNG.normal(size=D); BETA /= np.linalg.norm(BETA)
PERP = RNG.normal(size=D); PERP -= (PERP @ BETA) * BETA; PERP /= np.linalg.norm(PERP)
MU, SIG = np.log(68.0), 0.70             # lognormal market price calibrated to iPinYou (median 68)


def make_pop(gamma, theta, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D))
    # nonlinear ground-truth pCTR so GBM > linear capacity is meaningful
    logit = -3.9 + 1.1 * (X @ BETA) + 0.7 * X[:, 0] * X[:, 1] + 0.6 * (X[:, 2] ** 2 - 1) + 0.5 * X[:, 3] * X[:, 4]
    pctr = 1 / (1 + np.exp(-logit))
    click = (rng.random(N) < pctr).astype(int)
    th = np.deg2rad(theta)
    z = X @ (np.cos(th) * BETA + np.sin(th) * PERP)
    z = (z - z.mean()) / z.std()
    mprice = np.exp(MU + SIG * rng.normal(size=N) + gamma * z)
    return X, pctr, click, mprice


def _clf(cap):
    if cap == "linear":
        return LogisticRegression(max_iter=300, C=1.0)
    return LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.05,
                          min_child_samples=80, subsample=0.8, colsample_bytree=0.8, verbose=-1)


def fit_pred(X, win, click, cap, debias, pwin):
    w = win == 1
    sw = None
    if debias:
        sw = 1.0 / np.clip(pwin[w], 0.05, 1.0)
        sw = sw / sw.mean()
    m = _clf(cap).fit(X[w], click[w], sample_weight=sw)
    return m.predict_proba(X)[:, 1]


def xfit_isotonic(p_w, y_w, p_full, k=4, seed=0):
    out = np.zeros_like(p_full)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    # fit iso on winner folds, apply to all (average of k maps for the full set)
    maps = []
    for tr, _ in kf.split(p_w):
        maps.append(IsotonicRegression(out_of_bounds="clip").fit(p_w[tr], y_w[tr]))
    return np.mean([mm.predict(p_full) for mm in maps], axis=0)


def surplus(phat, pctr, mprice, alpha=1.0):
    win2 = (phat * CPC * alpha) >= mprice
    return float(np.sum((pctr * CPC - mprice) * win2))


def one(gamma, theta, baseline_cap, seed):
    X, pctr, click, mprice = make_pop(gamma, theta, seed)
    b0 = np.quantile(mprice, 0.5)
    win = (b0 >= mprice).astype(int)
    pw = LGBMClassifier(n_estimators=150, num_leaves=31, min_child_samples=80, verbose=-1).fit(X, win).predict_proba(X)[:, 1]
    w = win == 1
    p_base = fit_pred(X, win, click, baseline_cap, False, pw)         # competitor baseline (biased)
    p_base_rec = xfit_isotonic(p_base[w], click[w], p_base)           # + recalibration (cheap fix)
    p_deb = fit_pred(X, win, click, "gbm", True, pw)                  # strong corrected debiaser
    s_or = surplus(pctr, pctr, mprice)
    reg = lambda p: (s_or - surplus(p, pctr, mprice)) / s_or if s_or > 1e-9 else np.nan
    return reg(p_base), reg(p_base_rec), reg(p_deb), float(win.mean())


def cell(gamma, theta, baseline_cap, seeds=3):
    R = np.array([one(gamma, theta, baseline_cap, 100 + int(gamma * 10) + theta + s) for s in range(seeds)])
    rb, rr, rd, wr = R.mean(0)
    return {"gamma": gamma, "theta": theta, "baseline": baseline_cap, "win_rate": round(float(wr), 3),
            "regret_baseline": round(float(rb), 4), "regret_baseline_recal": round(float(rr), 4),
            "regret_debiaser": round(float(rd), 4),
            "edge_vs_baseline_pp": round(float(rb - rd) * 100, 2),          # >0: debiasing beats this baseline
            "edge_vs_recal_pp": round(float(rr - rd) * 100, 2)}


def main():
    cells = []
    for cap in ("linear", "gbm"):
        for gamma in (0.4, 0.8, 1.2):
            for theta in (0, 60):
                cells.append(cell(gamma, theta, cap))
                c = cells[-1]
                print(f"  baseline={cap:6} γ={gamma} θ={theta:>2} | regret base={c['regret_baseline']:.4f} "
                      f"recal={c['regret_baseline_recal']:.4f} deb={c['regret_debiaser']:.4f} "
                      f"| edge vs base={c['edge_vs_baseline_pp']:+.2f}pp  vs recal={c['edge_vs_recal_pp']:+.2f}pp")
    lin = [c for c in cells if c["baseline"] == "linear"]
    gbm = [c for c in cells if c["baseline"] == "gbm"]
    summ = {
        "mean_edge_vs_linear_pp": round(float(np.mean([c["edge_vs_baseline_pp"] for c in lin])), 2),
        "mean_edge_vs_gbm_pp": round(float(np.mean([c["edge_vs_baseline_pp"] for c in gbm])), 2),
        "mean_edge_vs_recal_pp": round(float(np.mean([c["edge_vs_recal_pp"] for c in cells])), 2),
        "asymmetry_holds": None,  # filled below
    }
    summ["asymmetry_holds"] = bool(summ["mean_edge_vs_linear_pp"] > summ["mean_edge_vs_gbm_pp"] + 1.0)
    print(f"\nASYMMETRY: edge vs LINEAR={summ['mean_edge_vs_linear_pp']:+.2f}pp  "
          f"vs GBM={summ['mean_edge_vs_gbm_pp']:+.2f}pp  (debias>recal everywhere: {summ['mean_edge_vs_recal_pp']:+.2f}pp)")
    print("Reproduces iPinYou 'robust vs LR, not vs LGB':", summ["asymmetry_holds"])
    OUT.write_text(json.dumps({
        "_meta": {"dgp": "semi-synthetic; market price lognormal calibrated to iPinYou median 68",
                  "N": N, "CPC": CPC, "metric": "full-inventory decision-value regret (lower=better)",
                  "debiaser": "GBM + IPW (winners-only, win-propensity weighted)",
                  "axes": "baseline capacity {linear≈LR, gbm≈LGB} × selection strength γ × heterogeneity θ",
                  "anchor": "iPinYou fair split: debiasing robust vs LR, NOT robust vs LGB (I²=0.82)"},
        "cells": cells, "summary": summ}, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
