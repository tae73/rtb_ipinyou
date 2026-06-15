# Debiasing Win Selection Bias in Real-Time Bidding

**English** · [한국어](README.ko.md)

> Recovering unbiased click-through rates from auction-censored data with a doubly-robust
> 3-tower model — and an honest measurement of whether that actually buys better bids.

<p align="center">
  <img src="results/figures/portfolio/fig_surplus_forest.png" alt="Per-advertiser surplus forest plot: neural debiasing is robust vs a linear baseline but not vs a strong GBM" width="900">
</p>

---

## TL;DR (30 seconds)

In Real-Time Bidding (RTB) a bidder only sees a click when its bid **won** the auction. Training a
click model on winners alone is **selection-biased** — and biased pCTR means biased bids. This project
builds a **doubly-robust debiasing model** (`ESCM²-WC`) to recover unbiased pCTR and asks the only
question that matters for a bidder: **does it make better bidding decisions?**

The honest answer, after correcting two of our own mistakes:

- **Ranking — yes.** On a *fair* split, debiasing wins the bidding-relevant object (winners-only
  AUC **0.658 > LGB 0.632 > LR 0.554**).
- **Calibration — fully solved.** Cross-fit isotonic recalibration zeroes the global bias
  (IEB **0.597 → 0**) and a per-advertiser map closes the residual (**0.226 → 0.0006**), both
  rank-preserving.
- **Bidding value — split verdict.** Debiasing's realized-surplus gain is **robust over a linear LR
  baseline** (5/5 advertisers, cluster CI excludes 0) but **NOT robust over a strong GBM (LGB)** —
  that edge is advertiser-heterogeneous (**Cochran's Q I²=0.82**, significant on only 1 of 5
  advertisers, and it flips negative when that one advertiser is removed).

> **What this project is really about:** the discipline to tell a *real effect* from a *measurement
> artifact*. The first headline ("neural debiasing beats the baseline on AUC") was retracted as a
> split artifact — caught by our own root-cause audit — and re-cast around calibration and bidding
> surplus. Every number here is pinned to a committed artifact in
> [`docs/NUMBERS_LEDGER.md`](docs/NUMBERS_LEDGER.md).

---

## The problem — win selection bias

<p align="center">
  <img src="assets/funnel_selection_bias.svg" alt="Bid to Win to Click funnel: clicks are observed only on won auctions" width="820">
</p>

A bid passes through a **Bid → Win → Click** funnel: of **129.5M** bids only **30.6M** win the auction
(become impressions), and only on those do we ever observe a click (**~23K**, CTR ≈ 0.075%). Because
the click outcome of a *lost* bid is censored, the CTR you can measure on winners is not the population
CTR: `P(click | win) ≠ P(click)`. Feed that biased pCTR into a value estimate `V = pCTR × CPC` and your
bids are biased too. **Debiasing** aims to recover the unbiased pCTR a bidder should actually price on.

## The approach — `ESCM²-WC` (doubly-robust, 3-tower)

<p align="center">
  <img src="assets/escm2wc_architecture.svg" alt="ESCM2-WC three-tower architecture with a dual-purpose Win Tower" width="860">
</p>

Adapting [ESMM](https://arxiv.org/abs/1804.07931) / [ESCM²](https://arxiv.org/abs/2204.05125) from the
impression→click→conversion funnel to **bid→win→click**, the model shares an embedding+MLP trunk into
three towers:

| Tower | Predicts | Role |
|---|---|---|
| **Win Tower** | `P(win \| x)` | propensity for debiasing **and** win-rate model for bid shading (dual-purpose) |
| **CTR Tower** | `P(click \| win, x)` | the debiased pCTR, trained with doubly-robust (DR) weights |
| **Imputation Tower** | `δ̂` (CTR error) | the error model that makes the estimator *doubly robust* |

The **doubly-robust** correction (`w = win / P̂(win)`, clipped) is unbiased if *either* the propensity
*or* the outcome model is right; an **ESMM joint constraint** `P(click,win) = P(win)·P(click|win)` ties
the towers together. The ablation ladder is **Biased LGB → ESMM-WC → ESCM²-WC (IPW) → ESCM²-WC (DR)**.

## The honest research arc

<p align="center">
  <img src="assets/falsification_arc.svg" alt="Falsification-first research arc from retracted headline to honest verdict" width="960">
</p>

<p align="center">
  <img src="results/figures/portfolio/fig_artifact_vs_fair.png" alt="The original debiasing-loses headline was a split artifact" width="640">
</p>

The pre-redesign program chased prediction AUC and reported debiasing *losing* to logistic regression.
A [root-cause audit](docs/NUMBERS_LEDGER.md#c-the-retraction--the-original-negative-result-was-an-artifact)
showed that was an **evaluation artifact**: the train/test advertisers were disjoint, so "LR 0.714"
rode on a single unseen advertiser (exclude it and *every* model collapses to ≈0.499, chance). On a
**fair** per-advertiser temporal split the artifact disappears — LR drops to 0.554, LGB rises to
0.632 — and the defensible thesis becomes **calibration → bidding surplus**, not raw ranking.

---

## Results (5 minutes)

### 1 · Ranking — debiasing wins the bidding-relevant object

<p align="center">
  <img src="results/figures/portfolio/fig_ablation_auc.png" alt="Fair-split winners-only vs all-bids AUC" width="820">
</p>

On the fair split, neural leads **winners-only AUC** (`P(click | win)` — what a bidder ranks on:
**0.658** vs LGB 0.632, LR 0.554). It *trails* on all-bids AUC (LGB 0.720) where easy negatives
dominate — but that is not the object bidding uses.

### 2 · Calibration — global then per-advertiser, fully solved

<p align="center">
  <img src="results/figures/portfolio/fig_calibration_journey.png" alt="Calibration journey: global isotonic then per-advertiser segment maps" width="900">
</p>

The neural model under-predicts winners' pCTR (all 10 deciles low, IEB 0.597). **Cross-fit isotonic
recalibration** (K=5, leak-free, GPU 0) zeroes the global bias for all three models while preserving
ranking. A single global map can't fix per-advertiser bias (residual up to 0.226), so a **per-advertiser
isotonic map** drives the residual to **0.0006** — three orders of magnitude — and even nudges global
AUC up. Training-stage calibration was tested and is **negative** (no train-time knob calibrates without
collapsing ranking); cheap post-hoc isotonic is the answer.

### 3 · Bidding value — robust vs linear, not vs a strong GBM

<p align="center">
  <img src="results/figures/portfolio/fig_surplus_grid.png" alt="Decision value by bidding strategy, second-price" width="840">
</p>

Pricing the recalibrated pCTR into a **second-price** auction on actual paid prices (mean value
equalized across models, so differences are pure ranking + slice-calibration) gives realized surplus.
Under the 2p-optimal `truthful` strategy, neural beats **LR by +27.4M** (cluster CI [17.7M, 37.8M],
**excludes 0**) but beats **LGB by only +9.4M** (cluster CI [−11.1M, 40.7M], **contains 0**).

The forest plot at the top of this page resolves *why*: the neural−LGB edge is **advertiser-
heterogeneous** (Cochran's Q I²=0.82, p=0.0002), positive on 2/5 advertisers, CI-significant on only
one (3427, +13.9M), and **leave-one-advertiser-out flips the mean negative** (drop 3427 → −1.1M). The
neural−LR edge, by contrast, is positive on all 5. **Honest bottom line: debiasing improves bids over a
linear model; over a strong GBM the apparent gain is one advertiser, not a robust effect.**

### 4 · Full-inventory value — the won-only limit barely binds

<p align="center">
  <img src="results/figures/portfolio/fig_policy_value_decomp.png" alt="Full-inventory policy value is over 99 percent exact" width="780">
</p>

Won-only surplus is a censored estimand, so we project **full-inventory** value over all 19.4M test bids
under second-price. Because truthful bids sit *below* the logged flat bids, each policy re-wins an
*observed* subset: **≥99.26% of every model's value is exact** observed surplus (≤0.74% modeled). The
full-inventory gap is **consistent with the won-only result** — neural−LR +22.0M (CI excludes 0),
neural−LGB +9.7M (CI contains 0).

---

## Honest limitations

- **Won-only evaluation is a conservative lower bound.** Genuinely *lost* inventory (aggressive bids
  above the logged flat bids) is untestable here: the flat-bid logging makes the contextual market
  model `F(b|x)` unidentifiable (calibrates on only 13% of cells — a documented **NO-GO**).
- **Low cluster power.** The advertiser-cluster CIs rest on only **5 advertisers** (the entire
  shared-vocabulary population). The design's MDE (~11.5M) exceeds the observed per-advertiser mean
  (~1.9M), so a small homogeneous neural−LGB effect could not be detected even if it existed.
- **Heterogeneity, not just noise.** The neural−LGB spread is genuine between-advertiser heterogeneity
  (I²=0.82), so a single cluster mean is the wrong summary — reported as a distribution, not a point.

These are stated up front by design; the project's value is the *honest scoping*, not an inflated win.

---

## Repository map

```
rtb_ipinyou/
├── src/
│   ├── data/         parser.py · unifier.py            (bz2 logs → unified Parquet)
│   ├── features/     engineering.py                    (30 features, target encoding)
│   ├── models/       base.py · esmm_wc.py · escm2_wc.py (shared trunk + towers, DR/IPW loss)
│   ├── debiasing/    win_propensity.py · diagnostics.py (propensity, ESS/overlap)
│   ├── metrics/      calibration.py                     (cross_fit_isotonic, segment maps)
│   ├── bidding/      shading.py · simulator.py · policy_value.py (bid shading + surplus eval)
│   ├── causal/       cate.py · scm.py                   (CATE, DAG refutation)
│   └── distributed/  mesh.py · data_loader.py · checkpoint.py (JAX SPMD)
├── scripts/
│   ├── preprocess.py · build_features.py · train.py
│   ├── stage_a/      recalibrate · stage_b2_surplus · stage4_calibration · segment_calibration · policy_value · power_analysis
│   └── portfolio/    make_figures.py · make_diagrams.py   (this portfolio's figures/diagrams)
├── results/
│   ├── stage_a/      *.json ledgers + *_summary.md       (canonical, frozen)
│   └── figures/      analysis figures + portfolio/       (curated hero figures)
├── docs/             technical_report.md · evaluation_protocol.md · NUMBERS_LEDGER.md · GLOSSARY.md
└── assets/           *.svg concept diagrams (EN + .ko)
```

## Quick start — reproduce the portfolio figures

The portfolio figures and diagrams are generated **only from committed result JSONs** — no model
training or data access needed:

```bash
pip install -e ".[dev]"                       # or: pip install matplotlib numpy
python scripts/portfolio/make_figures.py      # → results/figures/portfolio/*.png
python scripts/portfolio/make_diagrams.py     # → assets/*.svg (+ .ko)
```

The full research pipeline (preprocess → features → train → evaluate) is documented in
[`docs/scripts_tutorial.md`](docs/scripts_tutorial.md).

## Deep dives

| Document | What it holds |
|---|---|
| [`docs/technical_report.md`](docs/technical_report.md) | Full method + results write-up (the 30-minute layer) |
| [`docs/NUMBERS_LEDGER.md`](docs/NUMBERS_LEDGER.md) | Every headline number → committed source, + corrections |
| [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) | Frozen, immutable evaluation contract |
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | Bilingual term reference (debiasing, IEB, I², DR/IPW, …) |
| [`results/stage_a/README.md`](results/stage_a/README.md) | Index of the machine-readable artifacts |

## Tech stack

JAX/Flax (neural towers, SPMD multi-GPU) · LightGBM (baselines + propensity) · scikit-learn (isotonic,
diagnostics) · Hydra + Typer (config/CLI) · matplotlib (figures). Python 3.12.

## Dataset & attribution

[**iPinYou** RTB dataset](http://contest.ipinyou.com/) (2013, seasons 2–3). The dataset is licensed by
iPinYou for research and is **not redistributed in this repository** (see `.gitignore`). Method
adapted from Ma et al., *ESMM* (SIGIR 2018) and Wang et al., *ESCM²* (SIGIR 2022).

## License

Code is released under the [MIT License](LICENSE). The iPinYou dataset is **not** covered by this
license and remains subject to iPinYou's terms.
