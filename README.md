# Debiasing Win Selection Bias in Real-Time Bidding

🇺🇸 **English** · [🇰🇷 한국어](README.ko.md)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![JAX](https://img.shields.io/badge/JAX%2FFlax-neural%20towers-EE4C2C)
![LightGBM](https://img.shields.io/badge/LightGBM-baselines-02A4D3)
![EconML](https://img.shields.io/badge/EconML%2FDoWhy-causal-7B42BC)
![Data](https://img.shields.io/badge/Data-iPinYou%20RTB%202013-005571)
![License](https://img.shields.io/badge/License-MIT-green)

> Recovering unbiased click-through rates from auction-censored data with a doubly-robust 3-tower model
> (`ESCM²-WC`) — and an **honest** measurement of whether that actually buys better bids. The headline is
> **methodological rigor + falsification**: every claim is pinned to a committed artifact, the first
> result was retracted as an artifact, and the verdict is reported as *robust vs a linear model, not
> robust vs a strong GBM*.

<p align="center">
  <img src="results/figures/portfolio/fig_surplus_forest.png" alt="Per-advertiser surplus forest: debiasing is robust vs a linear baseline, not vs a strong GBM" width="900">
</p>

---

### 🧭 Navigate by time

| ⏱️ 30 sec | 🔎 5 min | 🧪 30 min | ♻️ Reproduce |
|---|---|---|---|
| [TL;DR](#-tldr-30-seconds) · [Results at a glance](#-results-at-a-glance) | [Problem](#the-problem--win-selection-bias) · [Method](#the-approach--escmwc) · [Retraction](#the-honest-research-arc) · [Results](#results-5-minutes) | [Key insights](#-key-insights) · [Limitations](#-limitations--lessons) · [Appendix](#-appendix) · [`technical_report.md`](docs/technical_report.md) | [Quick start](#quick-start--reproduce-the-figures) · [Repo map](#repository-map) · [`NUMBERS_LEDGER.md`](docs/NUMBERS_LEDGER.md) |

---

## ⏱️ TL;DR (30 seconds)

In Real-Time Bidding a bidder only sees a click when its bid **won** the auction. Training a click model
on winners alone is **selection-biased** — and biased pCTR means biased bids. This project builds a
**doubly-robust debiasing model** (`ESCM²-WC`) to recover unbiased pCTR and asks the only question that
matters: **does it make better bidding decisions?**

- **Ranking — yes.** On a *fair* split, debiasing wins the bidding object (winners-only AUC **0.658 >
  LGB 0.632 > LR 0.554**); the full ablation ladder (LR → LGB → ESMM-WC → IPW → DR) is closed on one split.
- **Calibration — fully solved.** Cross-fit isotonic zeroes global bias (IEB **0.597 → 0**) and a
  per-advertiser map closes the residual (**0.226 → 0.0006**), both rank-preserving.
- **Bidding value — split verdict.** Realized-surplus gain is **robust over a linear LR** (5/5
  advertisers, CI excludes 0) but **NOT robust over a strong GBM** — that edge is advertiser-heterogeneous
  (**Cochran's Q I²=0.82**), significant on only 1 of 5 advertisers, and flips negative when that one is
  removed.

> **The story is honesty.** The first headline ("neural debiasing beats the baseline on AUC") was
> **retracted as a split artifact** — caught by our own root-cause audit — and re-cast around calibration
> and bidding surplus. Every number lives in [`docs/NUMBERS_LEDGER.md`](docs/NUMBERS_LEDGER.md).

## 🎯 Results at a glance

| # | Result | Value | Split | Status |
|---|---|---|---|---|
| 1 | Winners-only AUC (neural / LGB / LR) | **0.658** / 0.632 / 0.554 | fair | canonical |
| 2 | Global calibration (neural winners IEB) | 0.597 → **0.000** | fair | canonical |
| 3 | Per-advertiser residual IEB | 0.226 → **0.0006** | fair | canonical |
| 4 | Decision value vs **LR** (truthful 2p) | **+27.4M**, CI [17.7M, 37.8M] ✓ | fair | **robust** |
| 5 | Decision value vs **LGB** (truthful 2p) | +9.4M, CI [−11.1M, 40.7M] ✗ | fair | **not robust** (I²=0.82) |
| 6 | Full-inventory value V(π), neural | **4.39e8**, ≥99.26% exact | fair | canonical |
| 7 | Best bid-shading strategy | truthful, surplus **5.13e8** | fair | canonical |
| 8 | Budget pacing (WR-weighted vs uniform) | **+11–14%** surplus | fair | canonical |
| 9 | CATE bid-effect / SCM bid→surplus | τ_surplus +21 / −0.066 (robust refutation) | fair | **exploratory** |

---

## The problem — win selection bias

<p align="center">
  <img src="assets/funnel_selection_bias.svg" alt="Bid to Win to Click funnel: clicks observed only on won auctions" width="820">
</p>

A bid passes through a **Bid → Win → Click** funnel: of **129.5M** bids only **30.6M** win the auction
(become impressions, win rate ≈ 24%), and only on those is a click ever observed (**~23K**, CTR ≈ 0.075%).
Because the click outcome of a *lost* bid is censored, the CTR you can measure on winners is not the
population CTR: `P(click | win) ≠ P(click)`. Feed that biased pCTR into a value estimate `V = pCTR × CPC`
and your bids are biased too. **Debiasing** aims to recover the unbiased pCTR a bidder should price on.

The market it bids into is heavy-tailed and right-censored:

<p align="center">
  <img src="results/figures/portfolio/fig_market_cdf.png" alt="Kaplan-Meier market price CDF and exchange competition" width="840">
</p>

Market price median **68** / mean **78** CPM, floor binding **21%**; a Kaplan-Meier win-rate curve
(right-censored at max bid 300) and sharply different competition across exchanges (F(300) = 0.69 / 0.29
/ 0.12) drive the bid-shading model later.

## The approach — `ESCM²-WC`

<p align="center">
  <img src="assets/escm2wc_architecture.svg" alt="ESCM2-WC three-tower architecture with a dual-purpose Win Tower" width="960">
</p>

Adapting [ESMM](https://arxiv.org/abs/1804.07931) / [ESCM²](https://arxiv.org/abs/2204.05125) from the
impression→click→conversion funnel to **bid→win→click**, a shared embedding trunk feeds three towers:

| Tower | Predicts | Role |
|---|---|---|
| **Win Tower** | `P(win \| x)` | propensity for debiasing **and** win-rate model for bid shading (dual-purpose) |
| **CTR Tower** | `P(click \| win, x)` | the debiased pCTR, trained with doubly-robust (DR) weights |
| **Imputation Tower** | `δ̂` (CTR error) | the control variate that makes the estimator *doubly robust* |

The DR correction (`ŵ = win / P̂(win)`, clipped) is unbiased if *either* the propensity or the outcome
model is right; an **ESMM joint constraint** `P(click,win) = P(win)·P(click|win)` ties the towers. The
ablation ladder is **Biased LGB → ESMM-WC → ESCM²-WC (IPW) → ESCM²-WC (DR)**.

## The honest research arc

<p align="center">
  <img src="assets/falsification_arc.svg" alt="Falsification-first research arc" width="960">
</p>

The pre-redesign program chased prediction AUC and reported debiasing *losing* to logistic regression.
A root-cause audit showed that was an **evaluation artifact**: train/test advertisers were disjoint, so
"LR 0.714" rode on a single unseen advertiser (exclude it and *every* model collapses to ≈0.499, chance).
On a **fair** per-advertiser temporal split the artifact disappears (LR 0.714→0.554, LGB 0.479→0.632) and
the defensible thesis becomes **calibration → bidding surplus**, not raw ranking.

<p align="center">
  <img src="results/figures/portfolio/fig_artifact_vs_fair.png" alt="The original debiasing-loses headline was a split artifact" width="640">
</p>

---

## Results (5 minutes)

### 1 · Ranking & the ablation ladder

<p align="center">
  <img src="results/figures/portfolio/fig_ablation_ladder.png" alt="Fair-split ablation ladder winners-AUC and IEB" width="900">
</p>

On the fair split, every neural debiasing variant clusters at **~0.66 winners-only AUC** (all beat LR
0.554, at/above LGB 0.632); raw calibration varies wildly (ESMM-WC IEB −37.8) but **all rungs
recalibrate to IEB ≈ 0**. DR is the primary model for its calibration→decision pipeline, not for topping
AUC — an honest reading of a *clustered* ladder, not a clean monotone one.

### 2 · Calibration — global then per-advertiser, fully solved

<p align="center">
  <img src="results/figures/portfolio/fig_calibration_journey.png" alt="Calibration journey: global isotonic then per-advertiser maps" width="900">
</p>

Neural winners pCTR under-predicts (IEB 0.597, all deciles low). **Cross-fit isotonic** (K=5, leak-free)
zeroes the global bias rank-preservingly; a **per-advertiser** map drives the residual to **0.0006** (three
orders of magnitude) and even lifts global AUC. Training-stage calibration was tested and is **negative**
(no train-time knob calibrates without collapsing ranking); cheap post-hoc isotonic is the answer.

### 3 · Decision value — robust vs linear, not vs a strong GBM

<p align="center">
  <img src="results/figures/portfolio/fig_surplus_grid.png" alt="Decision value by strategy" width="840">
</p>

Pricing recalibrated pCTR into a **second-price** auction on actual paid prices (mean value equalized
across models) gives realized surplus. Under the 2p-optimal `truthful` strategy, neural beats **LR by
+27.4M** (cluster CI [17.7M, 37.8M], **excludes 0**) but beats **LGB by only +9.4M** (CI [−11.1M, 40.7M],
**contains 0**). The forest plot at the top of this page resolves *why*: the neural−LGB edge is
**advertiser-heterogeneous** (I²=0.82, p=0.0002), positive on 2/5, CI-significant on 1 (adv 3427, +13.9M),
and **leave-one-advertiser-out flips the mean negative** (−1.1M). **Honest bottom line: debiasing
improves bids over a linear model; over a strong GBM the apparent gain is one advertiser, not robust.**

### 4 · Full-inventory value — the won-only limit barely binds

<p align="center">
  <img src="results/figures/portfolio/fig_policy_value_decomp.png" alt="Full-inventory policy value is over 99 percent exact" width="780">
</p>

Projecting value over all 19.4M test bids under second-price: truthful bids sit *below* the logged flat
bids, so each policy re-wins an *observed* subset — **≥99.26% of every model's value is exact** (≤0.74%
modeled). The full-inventory gap is consistent with the won-only result (neural−LR +22.0M ✓, neural−LGB
+9.7M ✗).

### 5 · Bid-shading strategies

<p align="center">
  <img src="results/figures/portfolio/fig_bidding_strategies.png" alt="Bid-shading strategy comparison and alpha sweep" width="840">
</p>

On the fair split, `truthful` (the 2p-optimal) tops realized surplus at **5.13e8**; the linear α-sweep
traces the classic win-rate/cost tradeoff (surplus ↑, ROI ↓ as α grows). The Win Tower's win-rate model
(AUC ≈ 0.91) plus exchange-conditional market CDFs drive the optimal/dual-regime strategies.

### 6 · Budget pacing

<p align="center">
  <img src="results/figures/portfolio/fig_pacing.png" alt="Budget pacing on the fair split" width="820">
</p>

A PID budget-pacing controller over the 24-hour cycle: **WR-weighted hourly allocation lifts surplus
+11–14%** over uniform pacing across budget levels — smarter allocation to high-value hours, not just
even spend.

### 7 · Causal exploration (CATE + SCM/DAG) — *hypothesis-generating*

<p align="center">
  <img src="results/figures/portfolio/fig_cate.png" alt="CATE bid-effect contrast" width="780">
</p>
<p align="center">
  <img src="results/figures/portfolio/fig_scm.png" alt="SCM DAG bid effect and refutation tests" width="780">
</p>

Treating bid as a treatment, a naive within-advertiser contrast and a DoWhy backdoor estimate both find
**bid → surplus ≈ −0.066** (the overpayment effect; refutation tests all robust) and a *negative* volume
channel (τ_win < 0). These are **confounded and exploratory** — iPinYou's flat-bid logging and won-only
censoring put a credible causal estimate at the data ceiling (the documented P1 NO-GO). Reported honestly
as hypothesis-generating, not causal claims.

### 8 · Serving

A FastAPI bidder loads the LR pCTR model + exchange-conditional market CDFs and runs the full loop
(request → 30 features → pCTR → `V = pCTR × CPC` → bid shading → response) at **sub-100ms** with a
train/serve-skew guard. See `src/serving/app.py`.

---

## 💡 Key insights

- **A real effect vs a measurement artifact.** The retraction (and the heterogeneity analysis that
  replaced a misleading cluster mean) is the project's core skill — knowing when "debiasing wins" is real
  (vs LR) and when it is one advertiser (vs LGB).
- **Calibration ≠ ranking.** Raw calibration ranges from IEB −38 to +0.6 across models, yet all
  recalibrate to ≈0 and ranking is nearly invariant — post-hoc isotonic is the right, cheap tool.
- **The data has a ceiling.** Flat-bid logging makes lost-inventory value and bid-causal-effects
  unidentifiable; we report what's observable (≥99% of policy value) and label the rest exploratory.

## ⚠️ Limitations & Lessons

| Issue | Evidence | Mitigation / honest framing |
|---|---|---|
| Won-only surplus is a lower bound | lost inventory censored; F(b\|x) calibrates on 13% of cells (P1 **NO-GO**) | ≥99.26% of policy value is exact; aggressive-policy value untestable here |
| Low cluster power | 5 advertisers; MDE ~11.5M ≫ observed mean ~1.9M | report heterogeneity (I²=0.82), not a single cluster mean |
| neural−LGB not robust | positive 2/5, CI-sig 1/5; LOAO flips negative | stated as the honest verdict, not "debiasing wins" |
| CATE / SCM not identified | flat-bid + censored; τ_win counterintuitive | labeled **exploratory / hypothesis-generating** |
| Ablation not monotone in AUC | ESMM-WC 0.674 > DR 0.658 | DR primary for calibration/decision, not AUC; reported as-is |

## 🧪 Appendix

- **Ablation grid** (winners-AUC + IEB raw→recal for all 5 rungs) — [`NUMBERS_LEDGER.md §K`](docs/NUMBERS_LEDGER.md).
- **Sensitivity** — decision-value sign-stable across CPC sweep {1e5, 2e5, 4e5} and max-bid {300, 600}
  (`stage_b2_surplus.json`); bid-shading α-sweep and pacing budget-sweep in §L/§M.
- **Refutation tests** — SCM bid→{surplus,win} pass random-common-cause / placebo / data-subset (§O).
- **Market modeling** — KM CDFs, exchange-conditional, temporal drift KS=0.118, lognormal fit (§ market).
- Full evaluation contract: [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) (frozen).

---

## Repository map

```
rtb_ipinyou/
├── src/
│   ├── data/ · features/                      bz2 logs → unified Parquet, 30 features
│   ├── models/    base · esmm_wc · escm2_wc    shared trunk + Win/CTR/Imputation towers (DR/IPW)
│   ├── debiasing/ win_propensity · diagnostics propensity, ESS/overlap/positivity
│   ├── metrics/   calibration · cluster_inference  cross-fit isotonic, segment maps, Q/I²/MDE
│   ├── bidding/   shading · simulator · policy_value · pacing  shading + surplus eval + PID pacing
│   ├── causal/    cate · scm                   CATE, DAG refutation (exploratory)
│   ├── win_rate/  nonparametric · survival     Kaplan-Meier market CDFs
│   └── serving/   app.py                       FastAPI RTB bidder (<100ms)
├── scripts/
│   ├── train.py · preprocess.py · build_features.py
│   ├── stage_a/   recalibrate · stage_b2_surplus · power_analysis · policy_value ·
│   │              ablation_ladder · bidding_fair · pacing_fair · cate_fair · scm_fair
│   └── portfolio/ make_figures.py · make_diagrams.py
├── results/stage_a/  *.json ledgers + *_summary.md       (canonical, frozen)
├── results/figures/portfolio/  12 hero figures (regenerated from committed JSONs)
├── docs/   technical_report · evaluation_protocol · NUMBERS_LEDGER · GLOSSARY · archive/
└── assets/ *.svg concept diagrams (EN + .ko)
```

## Quick start — reproduce the figures

The portfolio figures/diagrams regenerate **from committed result JSONs** — no training or data access:

```bash
pip install -e ".[dev]"                       # or: pip install matplotlib numpy
python scripts/portfolio/make_figures.py      # → results/figures/portfolio/*.png  (12 figures)
python scripts/portfolio/make_diagrams.py     # → assets/*.svg (+ .ko)
```

The strengthened experiments rerun on the fair split (CPU, from committed predictions):
`python scripts/stage_a/{bidding_fair,pacing_fair,cate_fair,scm_fair,ablation_ladder}.py`. Full pipeline
(preprocess → features → train → evaluate): [`docs/scripts_tutorial.md`](docs/scripts_tutorial.md).

## Tech stack

JAX/Flax (neural towers, SPMD multi-GPU) · LightGBM (baselines + propensity) · scikit-learn (isotonic,
diagnostics) · EconML / DoWhy (causal) · Hydra + Typer (config/CLI) · matplotlib (figures). Python 3.12.

## Dataset & attribution

[**iPinYou** RTB dataset](http://contest.ipinyou.com/) (2013, seasons 2–3), licensed by iPinYou for
research and **not redistributed** here (see `.gitignore`). Method: Ma et al., *ESMM* (SIGIR 2018);
Wang et al., *ESCM²* (SIGIR 2022). Causal tooling: Athey & Wager causal forests; Chernozhukov et al. DML;
DoWhy.

## References

- Ma et al., *Entire Space Multi-Task Model (ESMM)*, SIGIR 2018.
- Wang et al., *ESCM²: Entire Space Counterfactual Multi-Task Model*, SIGIR 2022.
- Athey & Wager, *Generalized Random Forests*, 2018 · Chernozhukov et al., *Double/Debiased ML*, 2018.
- Zhang et al., *Real-Time Bidding Benchmarking with the iPinYou Dataset*.

## License

Code under the [MIT License](LICENSE). The iPinYou dataset is **not** covered and remains subject to
iPinYou's terms.
