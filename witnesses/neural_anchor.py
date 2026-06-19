"""Neural anchor — do C1/C2 survive on REAL iPinYou features + the REAL neural ESCM²-WC?

The phase diagram (Gaussian-feature, sklearn toy models) found: within-capacity debiasing helps a WEAK
model and grows with selection, but NOT a STRONG model (C1); and naive recalibration over-bids (C2). Two
residual critiques: the DGP was hand-tuned on Gaussian toys, and the real neural ESCM²-WC was never used.

This anchor closes both with an iPinYou-GROUNDED semi-synthetic:
  • REAL iPinYou feature vectors (not Gaussian) — the joint distribution and nonlinearity are the data's.
  • ground-truth pCTR p*(x) = LightGBM fit to REAL winner clicks (real feature→click shape), base rate
    rescaled to a learnable level → clicks observable for ALL rows (the lost inventory real data censors).
  • market price lognormal FIT to REAL winner payprices (median ≈ 68), with selection strength γ a knob.
Then the SAME within-capacity experiment across cap ∈ {linear(LR), gbm(LGB), NEURAL(ESCM²-WC, Flax)}.

Sharp prediction the phase diagram makes: the neural ESCM²-WC is a STRONG model, so C1 says its
within-capacity debiasing edge should be small (like GBM), while a weak LR should still benefit; and the
recalibration trap (C2) should bite regardless of capacity. We test exactly that, on real features.

Honest: p*(x) is a FITTED surrogate (real-feature shape, rescaled base rate), the market is fit to real
payprices, selection is synthesized; decision-value is unmeasurable on real iPinYou (the data ceiling),
so the testbed stays semi-synthetic. Output: witnesses/neural_anchor.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # good citizen on shared GPU
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "old"))  # so the model's internal `from src.…` resolves
from src.metrics.calibration import cross_fit_isotonic  # noqa: E402

DATA = ROOT / "data/ipinyou/prediction/features/train.parquet"
META = json.load(open(ROOT / "data/ipinyou/prediction/features/feature_metadata.json"))
OUT = HERE / "neural_anchor.json"

# ---- config -------------------------------------------------------------------------------------
CPC = 3000.0                 # value per click; set adaptively in main() so mean V ≈ market median
N_POOL = 800_000             # experiment population (real feature rows), fixed across seeds
N_PSTAR = 3_000_000          # real rows to fit the p*(x) surrogate (more = better weak signal)
BASE_RATE = 0.10             # rescaled synthetic click base rate (learnable; real winners-CTR ≈ 0.0007)
GAMMAS = (0.4, 0.8, 1.2)
SK_SEEDS, NN_SEEDS = 3, 2    # sklearn caps cheap → more seeds; neural is the expensive headline
ALPHAS = np.concatenate([np.linspace(0.05, 1.0, 20), np.linspace(1.1, 4.0, 30)])  # bid-shading grid

CAT = list(META["feature_info"]["categorical"])                 # 17 categorical features
NUM = [c for c in META["feature_info"]["numerical"] if c != "bidprice"]  # drop logged bid (leakage)
STR_CAT = ["slot_size_group", "region_group"]                   # string-coded categoricals
LOWCARD = ["region", "adexchange", "slotwidth", "slotheight", "slotvisibility", "slotformat",
           "advertiser", "hour", "weekday", "is_weekend", "is_peak_hour", "slot_size_group", "region_group"]


# ---- real data loading --------------------------------------------------------------------------
def _read_rows(n_rows, cols):
    pf = pq.ParquetFile(DATA)
    out, got = [], 0
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols).to_pandas()
        out.append(t); got += len(t)
        if got >= n_rows:
            break
    return pd.concat(out, ignore_index=True).iloc[:n_rows]


def load_real():
    cols = CAT + NUM + ["win", "click", "payprice"]
    df = _read_rows(max(N_PSTAR, N_POOL), cols)
    # encode string categoricals to int codes (stable factorize)
    for c in STR_CAT:
        df[c], _ = pd.factorize(df[c])
    # clip categorical codes to >=0 (some real cats use -1 for missing) — merges missing into category 0
    for c in CAT:
        df[c] = df[c].astype("int64").clip(lower=0)
    # z-score numericals with the dataset's committed stats
    mean, std = META["normalization_stats"]["mean"], META["normalization_stats"]["std"]
    for c in NUM:
        df[c] = (df[c].astype("float64") - mean[c]) / (std[c] + 1e-9)
    return df


# ---- p*(x) ground-truth surrogate + real market model -------------------------------------------
def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_pstar_and_market(df):
    """p*(x): LGB on REAL winner clicks (shape), intercept-shifted to BASE_RATE. Market: lognormal fit
    to REAL winner payprices. Returns (pstar_fn, MU, SIG)."""
    w = df["win"].values == 1
    Xc = _design_lgb(df)                       # numeric+categorical design for LGB
    yc = df["click"].values.astype(int)
    g = LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.03, min_child_samples=200,
                       subsample=0.8, colsample_bytree=0.8, verbose=-1)
    g.fit(Xc[w], yc[w], categorical_feature=CAT)
    raw_logit = _logit(g.predict_proba(Xc)[:, 1])              # over the full pool (shape only)
    # shift intercept so mean sigmoid(raw_logit + c) ≈ BASE_RATE (preserve shape/spread)
    lo, hi = -15.0, 15.0
    for _ in range(60):
        c = 0.5 * (lo + hi)
        if (1 / (1 + np.exp(-(raw_logit + c)))).mean() < BASE_RATE:
            lo = c
        else:
            hi = c
    shift = 0.5 * (lo + hi)
    pstar = 1 / (1 + np.exp(-(raw_logit + shift)))
    # market lognormal fit to REAL winner payprices (>0)
    pay = df.loc[w & (df["payprice"] > 0), "payprice"].values.astype("float64")
    MU, SIG = float(np.log(pay).mean()), float(np.log(pay).std())
    return pstar, MU, SIG, float(g.predict_proba(Xc)[:, 1].std())


# ---- feature design matrices per capacity -------------------------------------------------------
def _design_lgb(df):
    """DataFrame [categorical int | numericals] for LGB (native categorical via categorical_feature=CAT)."""
    out = pd.DataFrame({c: df[c].astype("int32") for c in CAT})
    for c in NUM:
        out[c] = df[c].astype("float32")
    return out


def _design_lr(df):
    """[numericals | one-hot of low-card categoricals] for the linear model."""
    blocks = [np.column_stack([df[c].values.astype("float64") for c in NUM])]
    for c in LOWCARD:
        v = df[c].values.astype(int)
        k = int(v.max()) + 1
        oh = np.zeros((len(v), k), dtype="float32"); oh[np.arange(len(v)), v] = 1.0
        blocks.append(oh)
    return np.column_stack(blocks)


def _design_neural(df):
    """dict of cat int32 arrays + num float32 arrays, plus feature_dims for embeddings."""
    x = {c: df[c].values.astype("int32") for c in CAT}
    x.update({c: df[c].values.astype("float32") for c in NUM})
    feature_dims = {c: int(df[c].max()) + 2 for c in CAT}
    feature_dims.update({c: 1 for c in NUM})
    return x, feature_dims


# ---- metric -------------------------------------------------------------------------------------
def surplus(phat, pctr, mprice):
    win2 = (phat * CPC) >= mprice
    return float(np.sum((pctr * CPC - mprice) * win2))


def surplus_shaded(phat, pctr, mprice):
    """BEST-CASE decision value under OPTIMAL linear bid-shading bid = α·p̂·CPC (best α on the eval set).
    HONEST CAVEAT: this is NOT a pure ranking metric — a single global α cannot fix a miscalibrated LEVEL,
    so it REWARDS SPREAD (a model with restored/over-restored spread can be shaded to good bids), and it
    flatters a high-variance model relative to truthful bidding. Report it ALONGSIDE the truthful-bid
    surplus (bid = p̂·CPC), never instead of it. The oracle (α=1) is 2nd-price optimal."""
    val = pctr * CPC - mprice
    bidunit = phat * CPC
    return float(max(val[(a * bidunit) >= mprice].sum() for a in ALPHAS))


def decompose(phat, pctr, mprice):
    bid = phat * CPC
    win = bid >= mprice
    tv = pctr * CPC
    unprof = win & (tv < mprice)
    return {"mean_bid": round(float(bid.mean()), 1), "n_wins": int(win.sum()),
            "won_surplus": round(float(np.sum((tv - mprice) * win)), 1),
            "unprofitable_win_share": round(float(unprof.sum() / max(win.sum(), 1)), 3)}


# ---- semi-synthetic population (real features, synthesized selection/clicks) ---------------------
def make_population(pstar, MU, SIG, gamma, seed):
    rng = np.random.default_rng(seed)
    z = _logit(pstar); z = (z - z.mean()) / (z.std() + 1e-9)     # selection aligned with true pCTR
    mprice = np.exp(MU + SIG * rng.normal(size=len(pstar)) + gamma * z)
    b0 = np.quantile(mprice, 0.5)
    win = (b0 >= mprice).astype(int)
    click = (rng.random(len(pstar)) < pstar).astype(int)
    return win, click, mprice


# ---- sklearn estimators (linear / gbm capacities) -----------------------------------------------
def win_propensity(design_lgb, win):
    m = LGBMClassifier(n_estimators=200, num_leaves=63, min_child_samples=100, verbose=-1)
    m.fit(design_lgb, win, categorical_feature=CAT)
    return m.predict_proba(design_lgb)[:, 1]


def fit_pred_sklearn(design, win, click, cap, debias, pwin):
    w = win == 1
    sw = None
    if debias:
        sw = 1.0 / np.clip(pwin[w], 0.05, 1.0); sw = sw / sw.mean()
    Xw = design[w] if cap == "linear" else design.loc[w]
    if cap == "linear":
        m = LogisticRegression(max_iter=200, C=1.0).fit(Xw, click[w], sample_weight=sw)
    else:
        m = LGBMClassifier(n_estimators=200, num_leaves=63, learning_rate=0.05, min_child_samples=100,
                           subsample=0.8, colsample_bytree=0.8, verbose=-1)
        m.fit(Xw, click[w], sample_weight=sw, categorical_feature=CAT)
    return m.predict_proba(design)[:, 1]


# ---- neural estimators (real Flax ESCM²-WC + matching-capacity biased tower) ---------------------
import jax                                                          # noqa: E402
import jax.numpy as jnp                                             # noqa: E402
from flax import nnx                                                # noqa: E402
import optax                                                        # noqa: E402
from src.models.escm2_wc import (ESCM2WC, ESCM2WCConfig,            # noqa: E402
                                 create_escm2wc_train_step, create_escm2wc_eval_step)
from src.models.base import MLP, EmbeddingLayer, binary_cross_entropy  # noqa: E402

EMBED, HIDDEN, BATCH, EPOCHS = 16, (128, 64), 4096, 8


class BiasedCTR(nnx.Module):
    """Single-tower CTR at matching capacity (embeddings + MLP), trained on winners-only."""
    def __init__(self, feature_dims, *, rngs):
        self.embedding = EmbeddingLayer(feature_dims, EMBED, rngs=rngs)
        self.mlp = MLP(HIDDEN, output_dim=1, dropout=0.2, input_dim=len(feature_dims) * EMBED, rngs=rngs)

    def __call__(self, x, training=True):
        return jax.nn.sigmoid(self.mlp(self.embedding(x), training=training)).squeeze(-1)


def _jx(x, idx):
    return {k: jnp.asarray(v[idx]) for k, v in x.items()}


def _epochs(n, idx, seed):
    rng = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        order = rng.permutation(idx)
        for i in range(0, len(order), BATCH):
            yield order[i:i + BATCH]


def predict_neural(fn, x):
    n = len(next(iter(x.values())))
    return np.concatenate([np.asarray(fn(_jx(x, np.arange(i, min(i + BATCH, n))))) for i in range(0, n, BATCH)])


def train_biased_neural(x, win, click, feature_dims, seed):
    model = BiasedCTR(feature_dims, rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    w_idx = np.where(win == 1)[0]

    @nnx.jit
    def step(model, opt, bx, by):
        def lf(m):
            return jnp.mean(binary_cross_entropy(m(bx, training=True), by))  # plain BCE → calibrated pCTR
        loss, grads = nnx.value_and_grad(lf)(model)
        opt.update(model, grads)
        return loss
    for b in _epochs(len(w_idx), w_idx, seed * 7 + 1):
        step(model, opt, _jx(x, b), jnp.asarray(click[b].astype("float32")))
    return predict_neural(lambda bx: model(bx, training=False), x)


def train_escm2wc_neural(x, win, click, feature_dims, seed, censor=True):
    """censor=True feeds click*win (real-iPinYou contract: click==0 on losers) — the ESCM²-WC joint-BCE
    term BCE(p_win·p_ctr, click) is written for censored click; feeding uncensored synthetic click inflates
    p_ctr (the overshoot bug). censor=False reproduces the prior (buggy) behavior for the probe comparison."""
    cfg = ESCM2WCConfig(feature_dims=feature_dims, embed_dim=EMBED, hidden_dims=HIDDEN,
                        win_hidden_dims=(64, 32), loss_type="dr", dropout=0.2)
    model = ESCM2WC(cfg, rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    train_step = create_escm2wc_train_step(cfg)
    eval_step = create_escm2wc_eval_step()
    click_feed = (click * win) if censor else click
    idx = np.arange(len(win))
    for b in _epochs(len(idx), idx, seed * 7 + 2):
        train_step(model, opt, {"x": _jx(x, b), "win": jnp.asarray(win[b].astype("float32")),
                                "click": jnp.asarray(click_feed[b].astype("float32"))})
    return predict_neural(lambda bx: eval_step(model, {"x": bx}).p_ctr, x)


# ---- post-hoc calibration (fix the over-bidding without re-introducing selection bias) ----------
from sklearn.isotonic import IsotonicRegression


def calibrate_naive(p, click, win):
    """Naive isotonic: fit p→click on WINNERS, apply to all. REINTRODUCES selection bias — winners are the
    low-pCTR sub-population (selection ∝ pCTR here), so the target click rate is biased low vs the marginal."""
    w = win == 1
    iso = IsotonicRegression(out_of_bounds="clip").fit(p[w], click[w])
    return iso.transform(p)


def calibrate_ipw(p, click, win, pwin):
    """SELECTION-AWARE: IPW-weighted isotonic. Fit p→click on winners with sample_weight=1/clip(P(win|x))
    (self-normalized) → Horvitz-Thompson re-weights winners back to the MARGINAL click rate, so calibration
    fixes the level WITHOUT undoing the debiasing. Apply to all rows."""
    w = win == 1
    sw = 1.0 / np.clip(pwin[w], 0.05, 1.0); sw = sw / sw.mean()
    iso = IsotonicRegression(out_of_bounds="clip").fit(p[w], click[w], sample_weight=sw)
    return iso.transform(p)


def ipw_ess(win, pwin):
    """Effective-sample-size ratio of the IPW weights on winners (overlap diagnostic; <0.1 = unreliable)."""
    sw = 1.0 / np.clip(pwin[win == 1], 0.05, 1.0)
    return float((sw.sum() ** 2) / (np.square(sw).sum() * len(sw) + 1e-12))


# ---- sweep --------------------------------------------------------------------------------------
import phase_diagram as P   # reuse helpers


def cell(payload, cap, gamma, seed):
    design_lgb, design_lr, x_neural, feature_dims, pstar, MU, SIG = payload
    win, click, mprice = make_population(pstar, MU, SIG, gamma, seed)
    pctr = pstar
    s_or = surplus(pctr, pctr, mprice)
    rsh = lambda p: (s_or - surplus_shaded(p, pctr, mprice)) / s_or if s_or > 1e-9 else float("nan")
    rtr = lambda p: (s_or - surplus(p, pctr, mprice)) / s_or if s_or > 1e-9 else float("nan")
    w = win == 1
    pwin = win_propensity(design_lgb, win)   # external LGB win-propensity (IPW debias + IPW calibration)
    if cap == "neural":
        p_bias = train_biased_neural(x_neural, win, click, feature_dims, seed)
        p_deb = train_escm2wc_neural(x_neural, win, click, feature_dims, seed)   # censored (the fix)
    else:
        design = design_lr if cap == "linear" else design_lgb
        p_bias = fit_pred_sklearn(design, win, click, cap, False, pwin)
        p_deb = fit_pred_sklearn(design, win, click, cap, True, pwin)
    p_rec = calibrate_naive(p_bias, click, win)            # biased + naive recal (C2 trap)
    p_deb_naive = calibrate_naive(p_deb, click, win)       # debiased + naive cal (reintroduces selection bias)
    p_deb_ipw = calibrate_ipw(p_deb, click, win, pwin)     # debiased + IPW cal (selection-aware fix)
    rec = lambda p: {"mean": round(float(p.mean()), 3), "std": round(float(p.std()), 3)}
    return {"capacity": cap, "gamma": gamma, "seed": seed, "win_rate": round(float(win.mean()), 3),
            "ess_ratio": round(ipw_ess(win, pwin), 3),
            # C1 — optimal-shaded decision-value regret (best-case; rewards spread): does debiasing improve the bid?
            "regret_biased": round(rsh(p_bias), 4), "regret_debiased": round(rsh(p_deb), 4),
            "edge_pp": round((rsh(p_bias) - rsh(p_deb)) * 100, 2),
            "shaded_edge_ipwcal_pp": round((rsh(p_bias) - rsh(p_deb_ipw)) * 100, 2),
            # mechanism — pCTR recovery + the calibration of the debiased model (raw / naive-cal / IPW-cal)
            "pctr_true": rec(pctr), "pctr_biased": rec(p_bias), "pctr_debiased": rec(p_deb),
            "pctr_debiased_naivecal": rec(p_deb_naive), "pctr_debiased_ipwcal": rec(p_deb_ipw),
            # TRUTHFUL-bid edge (PRIMARY, bid = p̂·CPC): raw debiased over-bids; does calibration fix it?
            "truthful_regret_biased": round(rtr(p_bias), 3), "truthful_regret_debiased": round(rtr(p_deb), 3),
            "truthful_edge_pp": round((rtr(p_bias) - rtr(p_deb)) * 100, 2),
            "truthful_edge_naivecal_pp": round((rtr(p_bias) - rtr(p_deb_naive)) * 100, 2),
            "truthful_edge_ipwcal_pp": round((rtr(p_bias) - rtr(p_deb_ipw)) * 100, 2),
            "recal_edge_pp": round((rtr(p_bias) - rtr(p_rec)) * 100, 2),
            "decompose": {"biased": _dec(p_bias, pctr, mprice), "recal": _dec(p_rec, pctr, mprice),
                          "debiased": _dec(p_deb, pctr, mprice), "debiased_ipwcal": _dec(p_deb_ipw, pctr, mprice),
                          "oracle_surplus": round(s_or, 1)}}


def _dec(phat, pctr, mprice):
    d = decompose(phat, pctr, mprice)
    return {"mean_bid": d["mean_bid"], "won_surplus": d["won_surplus"],
            "unprofitable_win_share": d["unprofitable_win_share"]}


def summarize(cells, base_rate, MU, SIG):
    caps = ("linear", "gbm", "neural")
    mean = lambda key, cap: round(float(np.mean([c[key] for c in cells if c["capacity"] == cap])), 2)
    by_gamma = lambda cap: {g: round(float(np.mean([c["edge_pp"] for c in cells
                                                    if c["capacity"] == cap and c["gamma"] == g])), 2) for g in GAMMAS}
    s = {f"shaded_edge_{cap}_pp": mean("edge_pp", cap) for cap in caps}      # best-case (optimal bid-shading)
    s.update({f"truthful_edge_{cap}_pp": mean("truthful_edge_pp", cap) for cap in caps})  # PRIMARY metric (bid=p̂·CPC)
    s.update({f"recal_edge_{cap}_pp": mean("recal_edge_pp", cap) for cap in caps})  # C2: truthful recal-trap edge
    bg = lambda key, cap: {g: round(float(np.mean([c[key] for c in cells if c["capacity"] == cap and c["gamma"] == g])), 2) for g in GAMMAS}
    s["shaded_edge_neural_by_gamma"] = bg("edge_pp", "neural")
    s["truthful_edge_neural_by_gamma"] = bg("truthful_edge_pp", "neural")
    s["edge_by_gamma"] = {cap: by_gamma(cap) for cap in caps}
    # pCTR recovery (neural mechanism): selection collapses the biased model; ESCM²-WC restores scale+spread
    nn = [c for c in cells if c["capacity"] == "neural"]
    rmean = lambda k, f: round(float(np.mean([c[k][f] for c in nn])), 3)
    s["pctr_recovery_neural"] = {
        "true_mean": rmean("pctr_true", "mean"), "biased_mean": rmean("pctr_biased", "mean"),
        "debiased_mean": rmean("pctr_debiased", "mean"), "true_std": rmean("pctr_true", "std"),
        "biased_std": rmean("pctr_biased", "std"), "debiased_std": rmean("pctr_debiased", "std")}
    s["capacity_gap_linear_to_neural_pp"] = round((mean("regret_biased", "linear") - mean("regret_biased", "neural")) * 100, 2)
    # falsifiable claims (HONEST: report metric-dependence — shaded vs truthful — not just the favorable metric)
    r = s["pctr_recovery_neural"]
    s["biased_pctr_collapsed"] = bool(r["biased_std"] < r["true_std"] and r["biased_mean"] < r["true_mean"])
    s["debiasing_restores_spread"] = bool(r["biased_std"] < 0.6 * r["true_std"] and r["debiased_std"] > 1.5 * r["biased_std"])
    s["debiasing_overshoots_level"] = bool(r["debiased_mean"] > r["true_mean"])      # ESCM²-WC over-restores the LEVEL
    s["shaded_edge_positive_neural"] = bool(s["shaded_edge_neural_pp"] > 1.0)        # best-case (optimal shading)
    s["truthful_edge_negative_neural"] = bool(s["truthful_edge_neural_pp"] < 0.0)    # PRIMARY metric: over-bids into losses
    s["edge_reverses_by_metric"] = bool(s["shaded_edge_neural_pp"] > 0 > s["truthful_edge_neural_pp"])  # the honest headline
    s["weak_selection_truthful_help"] = bool(s["truthful_edge_neural_by_gamma"][0.4] > 0)  # well-calibrated when selection weak
    s["recal_trap_holds_gbm"] = bool(s["recal_edge_gbm_pp"] < 0.0)           # truthful recal worse than biased (GBM)
    s["recal_trap_holds_neural"] = bool(s["recal_edge_neural_pp"] < 0.0)     # False (biased neural under-predicts)
    # --- censoring fix + post-hoc calibration of the debiased model ---
    s.update({f"truthful_edge_{cap}_ipwcal_pp": mean("truthful_edge_ipwcal_pp", cap) for cap in caps})
    s.update({f"truthful_edge_{cap}_naivecal_pp": mean("truthful_edge_naivecal_pp", cap) for cap in caps})
    s["truthful_edge_neural_ipwcal_by_gamma"] = bg("truthful_edge_ipwcal_pp", "neural")
    s["truthful_edge_neural_naivecal_by_gamma"] = bg("truthful_edge_naivecal_pp", "neural")
    s["pctr_recovery_neural"]["debiased_ipwcal_mean"] = rmean("pctr_debiased_ipwcal", "mean")
    s["pctr_recovery_neural"]["debiased_ipwcal_std"] = rmean("pctr_debiased_ipwcal", "std")
    s["mean_ess_neural"] = round(float(np.mean([c["ess_ratio"] for c in nn])), 3)
    s["frozen_prefix_truthful_edge_neural_pp"] = -47.22   # PRE-FIX (uncensored), for the correction delta
    # branch flags (honest, data-driven)
    s["censoring_fixes_overbidding"] = bool(s["truthful_edge_neural_pp"] > -5)         # vs frozen -47.2
    s["censoring_alone_truthful_nonneg"] = bool(s["truthful_edge_neural_pp"] > -1)     # censoring alone ≈ resolves over-bidding
    s["debiased_undershoots_level"] = bool(r["debiased_mean"] < r["true_mean"])        # censored ESCM²-WC now under-predicts
    s["ipw_calibration_helps_truthful"] = bool(s["truthful_edge_neural_ipwcal_pp"] > s["truthful_edge_neural_pp"])
    s["naive_calibration_helps_truthful"] = bool(s["truthful_edge_neural_naivecal_pp"] > s["truthful_edge_neural_pp"])
    s["calibration_does_not_fix_truthful"] = bool(not s["ipw_calibration_helps_truthful"])
    s["ipw_cal_hurts_strong_selection"] = bool(s["truthful_edge_neural_ipwcal_by_gamma"][1.2] < s["truthful_edge_neural_by_gamma"][1.2])
    s["base_rate"] = round(float(base_rate), 4)
    s["market_mu"], s["market_sig"] = round(MU, 3), round(SIG, 3)
    s["n_pool"], s["sk_seeds"], s["nn_seeds"] = N_POOL, SK_SEEDS, NN_SEEDS
    return s


def main():
    df = load_real()
    print(f"loaded {len(df):,} real rows | real win_rate {df['win'].mean():.3f} | real winners-CTR {df.loc[df['win']==1,'click'].mean():.5f}")
    pstar, MU, SIG, sig_raw = fit_pstar_and_market(df)
    global CPC
    CPC = float(np.exp(MU) / max(pstar.mean(), 1e-4))   # value scale s.t. mean V ≈ market median → MARGINAL decisions
    print(f"p*: base_rate {pstar.mean():.4f} std {pstar.std():.4f} (raw winner-signal std {sig_raw:.4f}) | "
          f"market MU {MU:.2f} SIG {SIG:.2f} | CPC {CPC:.0f} (mean V ≈ exp(MU)={np.exp(MU):.0f})")
    df = df.iloc[:N_POOL].reset_index(drop=True); pstar = pstar[:N_POOL]
    design_lgb, design_lr = _design_lgb(df), _design_lr(df)
    x_neural, feature_dims = _design_neural(df)
    payload = (design_lgb, design_lr, x_neural, feature_dims, pstar, MU, SIG)
    cells = []
    for cap in ("linear", "gbm", "neural"):
        for gamma in GAMMAS:
            for sd in range(NN_SEEDS if cap == "neural" else SK_SEEDS):
                c = cell(payload, cap, gamma, 100 + int(gamma * 10) + sd)
                cells.append(c)
                print(f"  {cap:6} γ={gamma} seed={sd} | DEBIAS edge {c['edge_pp']:+.1f}pp  recal_edge {c['recal_edge_pp']:+.1f}pp "
                      f"| regret bias={c['regret_biased']:.3f} deb={c['regret_debiased']:.3f} (win {c['win_rate']})")
    summ = summarize(cells, float(pstar.mean()), MU, SIG)
    print(f"\nCENSORING FIX: neural TRUTHFUL edge {summ['truthful_edge_neural_pp']:+.1f}pp "
          f"(was {summ['frozen_prefix_truthful_edge_neural_pp']:+.1f} uncensored) → over-bidding fixed: {summ['censoring_fixes_overbidding']}")
    print(f"  truthful by γ: {summ['truthful_edge_neural_by_gamma']}  | shaded {summ['shaded_edge_neural_pp']:+.1f}pp")
    print(f"  pCTR: true {summ['pctr_recovery_neural']['true_mean']} → biased {summ['pctr_recovery_neural']['biased_mean']} → "
          f"debiased {summ['pctr_recovery_neural']['debiased_mean']} (undershoots: {summ['debiased_undershoots_level']})")
    print(f"  CALIBRATION: +IPWcal {summ['truthful_edge_neural_ipwcal_pp']:+.1f} / +naivecal {summ['truthful_edge_neural_naivecal_pp']:+.1f}pp "
          f"→ calibration helps: {summ['ipw_calibration_helps_truthful']} | IPW-cal hurts strong-sel: {summ['ipw_cal_hurts_strong_selection']} (ESS {summ['mean_ess_neural']})")
    print(f"  LR/LGB truthful edge {summ['truthful_edge_linear_pp']:+.1f}/{summ['truthful_edge_gbm_pp']:+.1f} | GBM recal-trap {summ['recal_edge_gbm_pp']:+.1f}pp")
    OUT.write_text(json.dumps({
        "_meta": {"design": "iPinYou-GROUNDED semi-synthetic: REAL features + p*(x) fit to REAL winner clicks "
                            "(base rate rescaled) + market lognormal fit to REAL winner payprices; selection synthesized.",
                  "honest": "p*(x) is a FITTED surrogate (real-feature shape, rescaled base rate), NOT real CTR; "
                           "decision-value is unmeasurable on real iPinYou (data ceiling) so the testbed stays semi-synthetic.",
                  "real_data": "data/ipinyou/prediction/features/train.parquet (90.6M rows)",
                  "n_pool": N_POOL, "n_pstar_fit": N_PSTAR, "CPC": CPC, "base_rate": summ["base_rate"],
                  "capacities": "linear (LR, one-hot), gbm (LGB native cat), NEURAL (ESCM²-WC Flax, DR loss + matching biased tower)",
                  "edge_definition": "WITHIN-CAPACITY: regret(biased) − regret(debiased), same model class",
                  "metric_primary": "truthful 2nd-price decision-value regret (bid = p̂·CPC) — same as phase_diagram/recal_trap",
                  "metric_secondary": "best-case under OPTIMAL linear bid-shading (bid = α·p̂·CPC) — rewards spread; report alongside truthful, never instead",
                  "fix": "ESCM²-WC training now feeds CENSORED click (click*win), matching the real-iPinYou contract the "
                         "joint-BCE loss expects; the debiased output is then post-hoc calibrated with SELECTION-AWARE "
                         "IPW-weighted isotonic (vs naive isotonic which reintroduces the winners'-low-pCTR bias).",
                  "frozen_prefix_result": {"note": "PRE-FIX (commit 526ecef, UNCENSORED click) — kept for audit; superseded by this run",
                                           "debiased_pctr_mean": 0.127, "debiased_pctr_std": 0.198,
                                           "truthful_edge_neural_pp": -47.22, "shaded_edge_neural_pp": 23.52,
                                           "truthful_edge_neural_by_gamma": {"0.4": 7.73, "0.8": -29.23, "1.2": -120.17},
                                           "finding": "uncensored click inflated p_ctr (mean 0.083→0.127) → truthful over-bidding -47pp"}},
        "cells": cells, "summary": summ}, indent=2))
    print(f"wrote {OUT.name}")


def probe():
    """De-risk: does censoring reduce the overshoot (Q1), does IPW-cal push truthful edge ≥0 (Q2),
    does censoring ALONE fix it (Q3 kill-switch)? Small N, 2 γ, 1 seed (~2 min GPU1)."""
    global CPC
    df = _read_rows(300_000, CAT + NUM + ["win", "click", "payprice"])
    for c in STR_CAT:
        df[c], _ = pd.factorize(df[c])
    for c in CAT:
        df[c] = df[c].astype("int64").clip(lower=0)
    mean, std = META["normalization_stats"]["mean"], META["normalization_stats"]["std"]
    for c in NUM:
        df[c] = (df[c].astype("float64") - mean[c]) / (std[c] + 1e-9)
    pstar, MU, SIG, _ = fit_pstar_and_market(df)
    CPC = float(np.exp(MU) / max(pstar.mean(), 1e-4))
    df = df.iloc[:60_000].reset_index(drop=True); pstar = pstar[:60_000]
    x, fd = _design_neural(df); design_lgb = _design_lgb(df)
    print(f"PROBE  base_rate {pstar.mean():.3f}  CPC {CPC:.0f}  (true pCTR mean {pstar.mean():.3f} std {pstar.std():.3f})")
    for g in (0.8, 1.2):
        win, click, m = make_population(pstar, MU, SIG, g, 7)
        pwin = win_propensity(design_lgb, win)
        s_or = surplus(pstar, pstar, m); rtr = lambda p: (s_or - surplus(p, pstar, m)) / s_or
        pb = train_biased_neural(x, win, click, fd, 7)
        pu = train_escm2wc_neural(x, win, click, fd, 7, censor=False)   # prior (buggy) uncensored
        pc = train_escm2wc_neural(x, win, click, fd, 7, censor=True)    # the fix: censored
        pc_ipw = calibrate_ipw(pc, click, win, pwin)
        pc_naive = calibrate_naive(pc, click, win)
        e = lambda p: round((rtr(pb) - rtr(p)) * 100, 1)
        print(f"  γ={g} | pCTR mean(std): uncens {pu.mean():.3f}({pu.std():.3f}) cens {pc.mean():.3f}({pc.std():.3f}) "
              f"ipwcal {pc_ipw.mean():.3f}({pc_ipw.std():.3f}) | true {pstar.mean():.3f}({pstar.std():.3f})")
        print(f"        TRUTHFUL edge vs biased: uncens {e(pu):+.1f} | cens {e(pc):+.1f} | +IPWcal {e(pc_ipw):+.1f} | "
              f"+naivecal {e(pc_naive):+.1f} pp  (ESS {ipw_ess(win,pwin):.2f})")
    print("Q1 censoring reduces overshoot? compare cens vs uncens mean. Q2 IPW-cal truthful≥0? Q3 cens alone≥0?")


if __name__ == "__main__":
    import sys
    if "--probe" in sys.argv:
        probe()
    else:
        main()
