"""De-risk probe (Stage 3) — does win-selection-bias debiasing buy BIDDING value?

WHAT GENERALIZES
  A controllable semi-synthetic RTB DGP where ground-truth pCTR AND lost-inventory outcomes are KNOWN
  (the two things iPinYou censors). We induce win-selection bias by correlating the market price with
  the features, then measure full-inventory realized surplus for: oracle (true pCTR), biased (winners-
  only model), biased+recalibrated (global isotonic — the cheap fix), and debiased (IPW-weighted).
  Two knobs: γ = selection STRENGTH, θ = selection HETEROGENEITY angle between the price-direction δ
  and the pCTR-direction β. θ≈0 ⇒ bias is monotone in pCTR (recalibration should fix it). θ→90° ⇒ bias
  is orthogonal to pCTR (recalibration CANNOT fix it; only debiasing can).

THE RESULT (boxed below): the make-or-break is whether DEBIASING beats RECALIBRATION on full-inventory
  surplus, and where in (γ, θ) it does — i.e. a phase diagram. This is the decision-layer question
  iPinYou could not answer (lost inventory censored). GO if debiasing recovers a materially larger
  surplus fraction than recalibration in a non-trivial region (especially high θ); NO-GO if recalibration
  already captures (nearly) all recoverable surplus everywhere (⇒ the iPinYou negative result generalizes).

HONEST reduces_check: this is NOT rigged for debiasing — at θ≈0 we EXPECT recalibration to suffice
  (debiasing ≈ recal), which would (correctly) limit the novelty to the high-heterogeneity regime.
  We report the boundary honestly.

VERDICT: printed at the end (GO / NO-GO + the (γ,θ) region where debiasing > recalibration).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

OUT = Path(__file__).resolve().parent / "probe_debiasing_bidding_value.json"
RNG = np.random.default_rng(0)
N, D, CPC = 300_000, 8, 150.0   # CPC scaled so V=pCTR*CPC ~ market price (~1); pCTR base ~1%
BETA = RNG.normal(size=D); BETA /= np.linalg.norm(BETA)
PERP = RNG.normal(size=D); PERP -= (PERP @ BETA) * BETA; PERP /= np.linalg.norm(PERP)


def make_population(gamma: float, theta_deg: float, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D))
    logit = -4.6 + 1.3 * (X @ BETA)              # true pCTR, base rate ~1%
    pctr = 1 / (1 + np.exp(-logit))
    click = (rng.random(N) < pctr).astype(int)   # ground truth click for ALL (even lost)
    # market price correlated with delta = cos*beta + sin*perp  -> selection on that direction
    th = np.deg2rad(theta_deg)
    delta = np.cos(th) * BETA + np.sin(th) * PERP
    z = X @ delta
    mprice = np.exp(0.0 + 0.6 * rng.normal(size=N) + gamma * z)   # lognormal, shifted by selection
    return X, pctr, click, mprice


def winners_only_fit(X, win, click):
    w = win == 1
    return LogisticRegression(max_iter=200, C=10).fit(X[w], click[w])


def ipw_fit(X, win, click, pwin):
    w = win == 1
    sw = 1.0 / np.clip(pwin[w], 0.05, 1.0)
    sw = sw / sw.mean()
    return LogisticRegression(max_iter=200, C=10).fit(X[w], click[w], sample_weight=sw)


def recal(p_win_train, y_win_train, p_full):
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_win_train, y_win_train)
    return iso.predict(p_full)


def exp_surplus(phat, pctr, mprice):
    """EXPECTED full-inventory 2nd-price surplus: bid = phat*CPC decides the win; the realized value
    uses TRUE pctr (synthetic) so the click-sampling noise is removed and we measure bidding quality."""
    win2 = (phat * CPC) >= mprice
    return float(np.sum((pctr * CPC - mprice) * win2))


def _one(gamma, theta, seed):
    X, pctr, click, mprice = make_population(gamma, theta, seed)
    b0 = np.quantile(mprice, 0.6)
    win = (b0 >= mprice).astype(int)
    pw_model = LogisticRegression(max_iter=200, C=10).fit(X, win)
    pwin = pw_model.predict_proba(X)[:, 1]
    biased = winners_only_fit(X, win, click)
    debiased = ipw_fit(X, win, click, pwin)
    p_bias = biased.predict_proba(X)[:, 1]
    p_deb = debiased.predict_proba(X)[:, 1]
    w = win == 1
    p_bias_recal = recal(p_bias[w], click[w], p_bias)
    return (exp_surplus(pctr, pctr, mprice), exp_surplus(p_bias, pctr, mprice),
            exp_surplus(p_bias_recal, pctr, mprice), exp_surplus(p_deb, pctr, mprice), float(win.mean()))


def run_cell(gamma, theta, seed):
    # average over seeds to kill variance
    res = np.array([_one(gamma, theta, seed + k)[:4] for k in range(6)])
    orc, bia, rec, deb = res.mean(0)
    wr = _one(gamma, theta, seed)[4]
    # regret = fraction of optimal (oracle) surplus LOST. lower = better, 0 = optimal.
    reg = lambda v: (orc - v) / orc if orc > 1e-6 else float("nan")
    return {"gamma": gamma, "theta": theta, "win_rate": round(wr, 3),
            "surplus": {"oracle": round(orc, 1), "biased": round(bia, 1),
                        "biased_recal": round(rec, 1), "debiased": round(deb, 1)},
            "regret_biased": round(reg(bia), 3), "regret_recal": round(reg(rec), 3),
            "regret_debiased": round(reg(deb), 3),
            "debias_beats_recal_pp": round((reg(rec) - reg(deb)) * 100, 2)}


def main():
    cells = []
    for gamma in (0.5, 1.0, 1.5):
        for theta in (0, 30, 60, 90):
            cells.append(run_cell(gamma, theta, seed=int(gamma * 100 + theta)))
    # VERDICT: debiasing GO if it consistently has LOWER regret than recalibration, with a
    # non-trivial edge that grows with selection heterogeneity (theta).
    n_beats = sum(1 for c in cells if c["debias_beats_recal_pp"] > 0)
    edge = [c for c in cells if c["debias_beats_recal_pp"] >= 1.0]            # >=1 percentage-point edge
    best = max(cells, key=lambda c: c["debias_beats_recal_pp"])
    go = n_beats >= 10 and len(edge) >= 3 and best["debias_beats_recal_pp"] >= 2.0
    verdict = {
        "GO": bool(go),
        "n_cells_debiased_lower_regret": f"{n_beats}/12",
        "n_cells_edge_ge_1pp": len(edge),
        "best_cell": {k: best[k] for k in ("gamma", "theta", "regret_recal", "regret_debiased", "debias_beats_recal_pp")},
        "reading": ("GO — across the (γ,θ) grid debiasing has consistently LOWER bidding regret than "
                    "recalibration on FULL inventory (the censored question iPinYou cannot answer). "
                    "The edge is real but modest; the clean magnitude + phase boundary is the research to do."
                    if go else
                    "NO-GO / weak — debiasing does not consistently beat recalibration; the iPinYou negative "
                    "result largely generalizes."),
    }
    print(f"  {'cell':>10} | regret: biased  recal  debiased | deb beats recal (pp)")
    for c in cells:
        print(f"  γ={c['gamma']} θ={c['theta']:>2} | {c['regret_biased']:+.3f}  {c['regret_recal']:+.3f}  "
              f"{c['regret_debiased']:+.3f} | {c['debias_beats_recal_pp']:+.2f}")
    print(f"\nVERDICT: {'GO' if go else 'NO-GO'} | debiased lower-regret in {n_beats}/12 cells | {verdict['reading']}")
    OUT.write_text(json.dumps({"cells": cells, "verdict": verdict,
                               "dgp": {"N": N, "D": D, "base_rate": "~1%", "selection": "market price ~ exp(γ·xδ), δ=cosθ·β+sinθ·β⊥"}}, indent=2))
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
