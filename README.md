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
> to the real iPinYou market) draws the **regime phase diagram**, with the debiasing effect **isolated
> from raw model capacity**. The real iPinYou result is the diagram's *negative half*.

<p align="center">
  <img src="witnesses/figures/fig_phase_diagram.png" alt="Within-capacity phase diagram: IPW debiasing helps a weak linear model (grows with selection) but not a strong GBM; the big apparent edge is model capacity, not debiasing" width="900">
</p>

---

### 🧭 Navigate by time

| ⏱️ 30 sec | 🔎 5 min | 🧪 30 min | ♻️ Reproduce |
|---|---|---|---|
| [TL;DR](#-tldr-30-seconds) · [Claims at a glance](#-claims-at-a-glance) | [The question](#the-question--decision-layer) · [C1 phase diagram](#c1--competitor-strength-governs-the-payoff-within-capacity) · [C2 recalibration trap](#c2--the-recalibration-trap) | [`methods.md`](methods.md) · [`review.md`](review.md) · [Honest scope](#-honest-scope) | [`MANIFEST.md`](MANIFEST.md) · [`repro/`](repro/) · [Foundation → `old/`](#foundation--the-ipinyou-study-old) |

---

## ⏱️ TL;DR (30 seconds)

A bidder only sees a click when its bid **won**. Train pCTR on winners → selection bias → biased bids.
Debiasing fixes the *model* — but does it change the *bidding decision*, and when? iPinYou real data
cannot answer this (flat-bid logging censors the lost inventory). So we build a testbed where the lost
inventory is **observable**, and — crucially — we **hold model capacity fixed** so the debiasing effect
isn't confused with "a stronger model wins."

- **C1 — Competitor strength governs the payoff (within capacity).** Holding the model class fixed, IPW
  debiasing helps a **weak (linear ≈ LR)** model — mean **+4.4 pp** of decision-value regret, growing
  with selection strength (**+15.4 pp** at strong selection) — but does **not** help a **strong
  (GBM ≈ LGB)** model (**−1.9 pp**). This matches the real iPinYou sign (robust vs LR, **not** vs LGB,
  I²=0.82). The much larger **+26.3 pp** "edge" you'd see by comparing a GBM-debiaser to a linear baseline
  is **model capacity, not debiasing** — reported separately, not as a debiasing result.
- **C2 — The recalibration trap.** Just recalibrating a biased model **inflates the bid** (137.8 → 193.5
  in this cell; **+42.7%** mean across seeds), wins *more unprofitable* inventory, and **lowers** surplus (4.31M → 3.26M).
  Principled IPW debiasing avoids the level inflation and **recovers** surplus (5.92M). Robust 5/5 seeds.
- **★ Neural anchor — real features + the real model.** On an **iPinYou-grounded** semi-synthetic (real
  features + the real **ESCM²-WC**), a first pass showed the debiaser **over-bidding** (truthful edge
  **−47 pp**). Investigating *why* revealed it was mostly a **censoring wiring bug** in the testbed
  (uncensored synthetic click fed to a loss that expects censored click). **Fixed (`click·win`), the
  ESCM²-WC genuinely helps truthful bidding: +7.2 pp** (positive across γ), and generic **calibration**
  adds more (**+11 pp**). A 2×2 test (biased/debiased × naive/IPW) finds the **selection-aware IPW variant
  beats naive *only* at weak selection (good overlap); it loses at strong selection and ties net** — *calibration
  accuracy ≠ bidding value*. LR/LGB don't benefit (−4.8/−1.6). A 2nd self-correction — the over-bidding was
  a data-contract bug, not fundamental miscalibration.

> **The contribution is a characterization, not a new method** — *real but thin*, so naming the regions
> where debiasing does **not** help is the whole point. We even implemented a genuine **DR** estimator and
> report that **it did not beat plain IPW here** (−2.6 pp) rather than hide it. Every number is pinned to a
> committed witness JSON ([`MANIFEST.md`](MANIFEST.md)); [`repro/`](repro/) re-asserts them.

## 🎯 Claims at a glance

| # | Claim | Headline (within capacity) | Witness | Status |
|---|---|---|---|---|
| C1 | Debiasing edge depends on competitor strength | IPW **+4.4 pp** vs weak / **−1.9 pp** vs strong (capacity gap **+26.3 pp** shown separately) | `witnesses/phase_diagram.json` | **confirmed** (synthetic, 10 seeds) |
| C2 | Recalibration over-bids marginal inventory | surplus **4.31M → 3.26M** (recal) vs **5.92M** (IPW), 5/5 seeds | `witnesses/recal_trap.json` | **confirmed** (synthetic) |
| ★ | **Neural anchor** — real iPinYou features + the real **ESCM²-WC** | the −47pp over-bidding was a **censoring wiring bug**; fixed (`click·win`), the ESCM²-WC helps truthful bidding **+7.2 pp** (**+11** with calibration; selection-aware IPW beats naive only at weak selection — *accuracy ≠ bidding value*) | `witnesses/neural_anchor.json` | **modest positive** (corrected) |
| — | DR (genuine) did **not** beat IPW here | **−2.6 pp** within linear (reported, not hidden) | `witnesses/phase_diagram.json` | honest negative |
| ⚓ | Real-world anchor (the negative half) | robust vs LR, **not** vs LGB, **I²=0.82** | [`old/`](old/) (iPinYou fair-split) | canonical |

---

## The question — decision layer

The debiasing *method* is **Foundation** (taken): win-tower-as-propensity (MTAE, CIKM'21),
ESMM/ESCM² estimators (SIGIR'18/'22), "debiasing helps bidding" on average (BGD, KDD'16). What no prior
work plots is **when** that correction changes the *bidding decision*, against **how strong** a
competitor — and that is where the real iPinYou project hit its ceiling: flat-bid logging censors the lost
inventory, so the decisive counterfactual is **unobservable** on most cells.

So we move it into a **controllable semi-synthetic testbed** ([`methods.md`](methods.md)): features
x ∈ ℝ⁸ → **nonlinear** ground-truth pCTR; **market price = lognormal calibrated to the iPinYou median
(68 CPM)**; win-selection-bias knobs (strength γ, heterogeneity θ); and — the point — **click + price are
known for every bid, won or lost**. Metric = full-inventory **decision-value regret**.

## C1 — Competitor strength governs the payoff (within capacity)

To isolate debiasing from raw capacity we debias **inside a fixed model class** and compare
`regret(biased) − regret(debiased)` at the **same** capacity. The capacity gap (a GBM out-ranking an LR) is
reported *separately* — it is not debiasing.

| | within **linear** (≈ LR) | within **GBM** (≈ LGB) |
|---|---|---|
| **IPW debiasing edge** (mean, 10 seeds) | **+4.4 pp** | **−1.9 pp** |
| ↳ by selection strength γ = 0.4 → 0.8 → 1.2 | −0.8 → +5.0 → **+8.9 pp** | ≈ 0 / slightly − |
| ↳ strong-selection cell (γ=1.2, θ=0) | **+15.4 pp** | — |
| model **CAPACITY** gap (lin-biased − gbm-biased) | **+26.3 pp** — *NOT debiasing* | — |

Within a fixed model class, IPW debiasing helps a *weak* model **once selection is non-trivial** (≈0 / slightly
negative at weak selection) and grows with selection strength, but **cannot help a strong GBM** (−1.9 pp). That asymmetry has the **same sign** as the real iPinYou
fair-split verdict — *robust vs LR, not robust vs LGB, I²=0.82* — though the real mechanism (advertiser
heterogeneity) differs, so we claim sign-similarity, not mechanistic identity. The figure shows the
**+26.3 pp capacity gap as a separate greyed bar** so it is never mistaken for a debiasing effect.

## C2 — The recalibration trap

The tempting fix — recalibrate the biased model — is a **trap**. Strong selection + linear baseline
(`recal_trap.json:linear_strong`, γ=1.2):

<p align="center">
  <img src="witnesses/figures/fig_recal_trap.png" alt="Recalibration inflates the bid and wins more unprofitable inventory, lowering surplus; IPW debiasing recovers it" width="900">
</p>

| | biased | + recalibration | debiased (IPW) | oracle |
|---|---|---|---|---|
| mean bid | 137.8 | **193.5** ↑ | 154.8 | — |
| unprofitable-win share | 0.495 | **0.509** | 0.493 | — |
| won surplus | 4.31M | **3.26M** ↓ | **5.92M** | 9.74M |

Recalibration raises the *level* of predictions, inflating the bid (137.8 → 193.5 in this cell; **+42.7%**
mean across seeds); it then wins *marginal* inventory where true value < clearing price, so surplus **drops**
(4.31M → 3.26M). IPW debiasing fixes the *shape* without level inflation and **recovers** surplus
(5.92M; a linear model can't reach the nonlinear oracle 9.74M). Robust: across 5 seeds recal lowers
surplus **5/5** and IPW raises it **5/5**. The trap is weak against a strong GBM
(`recal_trap.json:gbm_strong`) — consistent with C1.

## ★ Neural anchor — real iPinYou features + the real ESCM²-WC

The synthetic results above use sklearn toy models on Gaussian features. This anchor closes both gaps with
an **iPinYou-grounded semi-synthetic**: **real feature vectors** (800K rows of the 90.6M-row parquet, 29
real features), ground-truth **p\*(x) fit to REAL winner clicks**, market **lognormal fit to REAL winner
payprices**, and the **real neural ESCM²-WC** (Flax, the project's actual engineering in [`old/src/`](old/src))
as the debiaser — across {LR, LGB, **neural ESCM²-WC**}. Primary metric = **truthful 2nd-price surplus**.

<p align="center">
  <img src="witnesses/figures/fig_neural_anchor.png" alt="The -47pp over-bidding was a censoring wiring bug; censoring click (click*win) fixes it (+7.2pp); a 2x2 test shows selection-aware IPW calibration beats naive only at weak selection (good ESS) and loses at strong selection" width="900">
</p>

> **2nd self-correction (a real bug, found by asking "can we fix it?").** A first pass showed the neural
> debiaser **over-bidding** (truthful edge **−47 pp**), framed as "restores ranking but overshoots
> calibration." Reading the ESCM²-WC loss revealed this was **mostly a wiring bug in the testbed**: I fed
> *uncensored* synthetic click, but the joint-BCE term `BCE(p_win·p_ctr, click)` is written for the
> real-iPinYou contract where **click is censored to 0 on losses**. Uncensored, loser clicks inflate p_ctr
> (mean → 0.127). Pre-fix numbers are frozen in `neural_anchor.json:_meta` for audit.

- **The fix — censor click (`click·win`).** The over-bidding disappears: debiased pCTR no longer overshoots
  (mean 0.083 → 0.064), and the **neural truthful edge becomes +7.2 pp** (was −47.2), positive in all six
  cells (by-γ 5.1 / 7.6 / 8.7). It's a genuine recovery, not conservative bidding — debiased bids *higher*
  than biased and wins *more* surplus. (*Caveat:* n=2 neural seeds/γ + mild p\* non-determinism — trust the
  **sign/direction**, magnitudes under-powered.)
- **Can we calibrate it further? And does selection-awareness pay?** A 2×2 test ({biased, debiased} ×
  {naive, IPW-weighted isotonic}). Generic calibration helps (+7.2 → **+11.1** IPW, **+11.2** naive). But
  the selection-aware **IPW beats naive *only at weak selection*** (γ=0.4, ESS 0.87: +1.2 pp) — there it
  correctly lifts the level toward the marginal (0.050 → 0.060 vs naive 0.054). At **strong selection**
  (ESS 0.63) its high-variance weights over-lift → **over-bid → it loses** (γ=1.2: −6.4 pp); net naive
  ties/wins. The driver is **overlap (ESS)**, and the lesson is **calibration accuracy ≠ bidding value** —
  a level-correct calibration over-bids near the margin (C2's trap at the calibration layer).
- **Honest answer to "what's the problem / can we fix it":** the over-bidding was largely a **data-contract
  bug** (censoring), not fundamental miscalibration; fixed, the ESCM²-WC helps (+7.2 pp). The intuitive
  "just calibrate it" helps a little, but the *principled* selection-aware variant has no net advantage —
  it only pays when overlap is good. LR/LGB don't benefit (−4.8 / −1.6).
- **C2 on real features.** The recal trap **reproduces for GBM** (−14.2 pp) but not LR/neural — capacity-dependent.

> Honest: iPinYou-**grounded** semi-synthetic — p\*(x) a fitted surrogate, market fit to real payprices,
> selection synthesized; decision-value unmeasurable on real iPinYou (data ceiling). Modest, positive,
> robust to the truthful metric. Details in [`methods.md`](methods.md) §6.

---

## 🔬 Honest scope

This is `[sketch · synthetic-verified]`. Stated plainly ([`review.md`](review.md)):

- **Direction-correction.** An earlier version headlined a "+24.2 pp debiasing edge vs linear" that
  **conflated capacity with debiasing**. This version isolates **within-capacity** debiasing (+4.4 / −1.9 pp)
  and reports the capacity gap (+26.3 pp) separately. We keep the correction visible rather than quietly
  restating it.
- **Not a method.** No new estimator or identification result — those are Foundation. The contribution is
  a **characterization** of *when/why* debiasing changes the bid, on the **competitor-strength** axis.
- **Negative regions are reported.** Debiasing does **not** beat a strong GBM; a genuine **DR did not beat
  IPW** here (−2.6 pp); recalibration is not universally bad. The real iPinYou anchor carries equal weight.
- **Semi-synthetic.** Generalization is bounded by the phase diagram's *edges*; truthful-2nd-price centric.
- **Path to a full result:** ESCM²-WC neural anchor (GPU), an Open-Bandit-Dataset OPE sanity anchor, a
  small formal theorem for the recalibration over-bidding condition, and a strategy/budget axis.

## Foundation — the iPinYou study (`old/`)

The motivating prior work lives in [**`old/`**](old/): the full **win-selection-bias debiasing portfolio**
on real iPinYou RTB data — ESCM²-WC (doubly-robust 3-tower), the calibration story (global IEB 0.597 → 0;
per-advertiser residual → 0.0006), and the **honest research arc** (the first AUC headline retracted as a
split artifact). Its decisive, *honest* verdict — **debiasing's bidding value is robust vs a linear LR but
not vs a strong GBM (I²=0.82)** — is precisely the real-world **negative half** this testbed reproduces by
sign. The data ceiling there (censored lost inventory) is *why* this study moves to an observable testbed.

## Repository map

```
README.md / README.ko.md   ← flagship front (EN / KO twin)
concept.md                 ← 1-pager: motivation → falsifiable claims
methods.md                 ← testbed · models · results (numbers = witness JSON)
review.md                  ← honest scoping · prior-art positioning · path to full paper
MANIFEST.md                ← canonical map: witness JSON → claims → figures
witnesses/                 ← the study
  phase_diagram.py + .json     C1 (within-capacity competitor-strength sweep, 10 seeds)
  recal_trap.py + .json        C2 (recalibration trap, 5-seed robustness)
  neural_anchor.py + .json     ★ real iPinYou features + real ESCM²-WC (Flax, imports old/src/)
  probe_*.py + .json           de-risk probe (GO)
  figures/make_figures.py      JSON → figures
repro/check.py             ← green harness: re-asserts C1 + C2 + neural anchor from the JSONs
old/                       ← Foundation: the iPinYou debiasing study (real-world anchor)
```

## ♻️ Reproduce

```bash
pip install -r requirements.txt
python witnesses/phase_diagram.py      # → witnesses/phase_diagram.json  (10 seeds)
python witnesses/recal_trap.py         # → witnesses/recal_trap.json
CUDA_VISIBLE_DEVICES=1 python witnesses/neural_anchor.py   # → neural_anchor.json (real features + ESCM²-WC, GPU)
python witnesses/figures/make_figures.py
python repro/check.py                  # GREEN — re-asserts C1 + C2 + neural anchor from the canonical JSONs
```

Numbers everywhere are verbatim from the witness JSONs; the figures and tables regenerate from them.
See [`MANIFEST.md`](MANIFEST.md) for the claim → witness → figure map.
