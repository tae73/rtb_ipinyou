# When Does Debiasing Change the *Bid*?

🇺🇸 **English** · [🇰🇷 한국어](README.ko.md)

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-baselines-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-strong%20competitor-02A4D3)
![Method](https://img.shields.io/badge/contribution-characterization%2C%20not%20a%20method-7B42BC)
![Status](https://img.shields.io/badge/status-sketch%C2%B7synthetic--verified-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **The decision-layer question.** Win-selection-bias debiasing for RTB is a *solved method*
> (ESMM/ESCM², win-tower-as-propensity). The open question is **when it actually changes the bid** —
> decision value, not AUC — **and why naïvely recalibrating a biased model backfires on full inventory.**
> A controllable **semi-synthetic testbed** (ground-truth pCTR + *observable* lost inventory, calibrated
> to the real iPinYou market) draws the **regime phase diagram**. The wedge is the
> **competitor-model-strength** axis — and the real iPinYou result is its *negative half*.

<p align="center">
  <img src="witnesses/figures/fig_phase_diagram.png" alt="Phase diagram: debiasing's bidding edge grows over a weak linear competitor but vanishes against a strong GBM — the iPinYou anchor sits on the negative half" width="900">
</p>

---

### 🧭 Navigate by time

| ⏱️ 30 sec | 🔎 5 min | 🧪 30 min | ♻️ Reproduce |
|---|---|---|---|
| [TL;DR](#-tldr-30-seconds) · [Claims at a glance](#-claims-at-a-glance) | [The question](#the-question--decision-layer) · [C1 phase diagram](#c1--competitor-strength-governs-the-payoff) · [C2 recalibration trap](#c2--the-recalibration-trap) | [`methods.md`](methods.md) · [`review.md`](review.md) · [Honest scope](#-honest-scope) | [`MANIFEST.md`](MANIFEST.md) · [`repro/`](repro/) · [Foundation → `old/`](#foundation--the-ipinyou-study-old) |

---

## ⏱️ TL;DR (30 seconds)

A bidder only sees a click when its bid **won**. Train pCTR on winners → selection bias → biased bids.
Debiasing fixes the *model* — but the question that matters for an advertiser is **does it change the
bidding decision, and when?** iPinYou real data cannot answer this (flat-bid logging censors the lost
inventory). So we build a testbed where the lost inventory is **observable**, and sweep the regime.

- **C1 — Competitor strength governs the payoff.** Debiasing's bidding edge is **+24.2 pp** of decision
  regret over a *weak linear* competitor (growing with selection strength) but **−1.9 pp** over a
  *strong GBM* — it does **not** beat a strong baseline. This **reproduces and explains** the real iPinYou
  finding (debiasing robust vs LR, **not** vs LGB, I²=0.82): that result is the *negative half* of the diagram.
- **C2 — The recalibration trap.** Just recalibrating a biased model **inflates the bid** (137.8 → 193.5),
  wins *more unprofitable* inventory (share 0.495 → 0.509), and **lowers** surplus (4.3M → 3.3M). Principled
  DR debiasing fixes the *shape* without the level inflation and recovers **8.6M ≈ 88% of oracle**.

> **The contribution is a characterization, not a new method** — *real but thin*, so naming the regions
> where debiasing does **not** help is the whole point. Every number is pinned to a committed witness JSON
> ([`MANIFEST.md`](MANIFEST.md)); the harness in [`repro/`](repro/) re-asserts them.

## 🎯 Claims at a glance

| # | Claim | Headline | Witness | Status |
|---|---|---|---|---|
| C1 | Debiasing edge depends on competitor strength | **+24.2 pp** vs linear · **−1.9 pp** vs GBM | `witnesses/phase_diagram.json` | **confirmed** (synthetic) |
| C2 | Recalibration over-bids marginal inventory | surplus **4.3M → 3.3M** (recal) vs **8.6M** (DR) | `witnesses/recal_trap.json` | **confirmed** (synthetic) |
| — | De-risk probe (full-inventory regret) | debiasing lower regret in **12/12** cells | `witnesses/probe_debiasing_bidding_value.json` | **GO** |
| ⚓ | Real-world anchor (the negative half) | robust vs LR, **not** vs LGB, **I²=0.82** | [`old/`](old/) (iPinYou fair-split) | canonical |

---

## The question — decision layer

The debiasing *method* is **Foundation** (taken): win-tower-as-propensity (MTAE, CIKM'21),
ESMM/ESCM² estimators (SIGIR'18/'22), "debiasing helps bidding" on average (BGD, KDD'16). What no prior
work plots is **when** that correction changes the *bidding decision* — and against **how strong** a
competitor. That is the only open layer, and it is where the real iPinYou project hit its ceiling:
flat-bid logging censors the lost inventory, so the decisive counterfactual ("would a different bid have
won, and was it worth it?") is **unobservable** on 87% of cells.

So we move it into a **controllable semi-synthetic testbed** ([`methods.md`](methods.md)): features
x ∈ ℝ⁸ → **nonlinear** ground-truth pCTR; **market price = lognormal calibrated to the iPinYou median
(68 CPM)**; win-selection-bias knobs (strength γ, heterogeneity θ); and — the point — **click + price are
known for every bid, won or lost**. Metric = full-inventory **decision-value regret** `(S_oracle − S_model)/S_oracle`.

## C1 — Competitor strength governs the payoff

Each estimator is instantiated at two capacities — **linear (≈ LR)** and **GBM (≈ LGB)** — giving the
*competitor-strength axis*. The debiaser is fixed (GBM + IPW, win-propensity weighted ≈ the iPinYou
neural debiaser). `edge ≡ regret(baseline) − regret(debiaser)`.

| competitor baseline | mean edge | across γ = 0.4 → 1.2 |
|---|---|---|
| **linear (≈ LR)** | **+24.2 pp** | +16 → +45 pp (grows with selection strength) |
| **GBM (≈ LGB)** | **−1.9 pp** | ≈ 0 / slightly negative |

Debiasing robustly beats a *weak* linear competitor (more so as selection strengthens), but **cannot beat
a strong GBM**. This is exactly the iPinYou fair-split verdict — *robust vs LR, not robust vs LGB,
Cochran's Q I²=0.82* — reproduced and **mechanistically explained** in a controlled setting. The real
result is the diagram's **negative half**; the testbed supplies the controllable positive half.

## C2 — The recalibration trap

The tempting fix — recalibrate the biased model — is a **trap**. Strong selection + linear baseline
(`recal_trap.json:linear_strong`, γ=1.2):

<p align="center">
  <img src="witnesses/figures/fig_recal_trap.png" alt="Recalibration inflates the bid and wins more unprofitable inventory, lowering surplus; DR debiasing recovers ~88% of oracle" width="900">
</p>

| | biased | + recalibration | debiased (DR) | oracle |
|---|---|---|---|---|
| mean bid | 137.8 | **193.5** ↑ | 135.1 | — |
| unprofitable-win share | 0.495 | **0.509** | **0.219** | — |
| won surplus | 4.3M | **3.3M** ↓ | **8.6M** | 9.7M |

Recalibration raises the *level* of predictions, inflating the bid (137.8 → 193.5); it then wins
*marginal* inventory where true value < clearing price, so over half its wins are unprofitable and surplus
**drops** (4.3M → 3.3M). DR debiasing fixes the *shape* without level inflation and reaches **88% of
oracle** (8.6M / 9.7M). The trap is strong under *weak baseline / strong selection* and weak against a
strong GBM (`recal_trap.json:gbm_strong`) — consistent with C1.

---

## 🔬 Honest scope

This is `[sketch · synthetic-verified]`. Stated plainly ([`review.md`](review.md)):

- **Not a method.** No new estimator or identification result — those are Foundation. The contribution is
  a **characterization** of *when/why* debiasing changes the bid, on the **competitor-strength** axis.
- **Negative regions are reported, not hidden.** Debiasing does **not** beat a strong GBM; recalibration
  is not universally bad. The real iPinYou anchor (the negative half) carries equal weight.
- **Semi-synthetic.** Generalization is bounded by the phase diagram's *edges*; truthful-2nd-price centric.
- **Path to a full result:** ESCM²-WC neural anchor (GPU), an Open-Bandit-Dataset OPE sanity anchor, a
  small formal theorem for the recalibration over-bidding condition, and a strategy/budget axis.

## Foundation — the iPinYou study (`old/`)

The motivating prior work lives in [**`old/`**](old/): the full **win-selection-bias debiasing portfolio**
on real iPinYou RTB data — ESCM²-WC (doubly-robust 3-tower), the calibration story (global IEB 0.597 → 0;
per-advertiser residual → 0.0006), and the **honest research arc** (the first AUC headline retracted as a
split artifact). Its decisive, *honest* verdict — **debiasing's bidding value is robust vs a linear LR but
not vs a strong GBM (I²=0.82)** — is precisely the real-world **negative half** this testbed reproduces.
The data ceiling there (censored lost inventory) is *why* this study moves to an observable testbed.

## Repository map

```
README.md / README.ko.md   ← flagship front (EN / KO twin)
concept.md                 ← 1-pager: motivation → falsifiable claims
methods.md                 ← testbed · models · results (numbers = witness JSON)
review.md                  ← honest scoping · prior-art positioning · path to full paper
MANIFEST.md                ← canonical map: witness JSON → claims → figures
witnesses/                 ← the study
  phase_diagram.py + .json     C1 (competitor-strength sweep)
  recal_trap.py + .json        C2 (recalibration trap)
  probe_*.py + .json           de-risk probe (GO, 12/12)
  figures/make_figures.py      JSON → figures
repro/check.py             ← green harness: re-asserts C1 + C2 from the JSONs
old/                       ← Foundation: the iPinYou debiasing study (real-world anchor)
```

## ♻️ Reproduce

```bash
pip install -r requirements.txt
python witnesses/phase_diagram.py      # → witnesses/phase_diagram.json
python witnesses/recal_trap.py         # → witnesses/recal_trap.json
python witnesses/figures/make_figures.py
python repro/check.py                  # GREEN — re-asserts C1 + C2 from the canonical JSONs
```

Numbers everywhere are verbatim from the witness JSONs; the figures and tables regenerate from them.
See [`MANIFEST.md`](MANIFEST.md) for the claim → witness → figure map.
