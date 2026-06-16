# RTB iPinYou — Canonical Evaluation Protocol (FROZEN)

**Status: FROZEN** · Established 2026-06-15 (Stage 3) · Authority: this document defines how models
are evaluated and compared from here on. Findings narrative: `docs/redesign_findings.md`.

This protocol formalizes the de-facto spec from `results/stage_a/phase1_findings.md` ("Implications
for downstream stages") plus the Stage B2 inference design. It exists because the pre-redesign
program drew wrong conclusions from the wrong split and the wrong metrics; freezing the protocol
prevents regression.

---

## 1. Canonical data split (FROZEN)

- **Use `data/ipinyou/prediction/features_fair/`** — a per-advertiser **temporal** split
  (0.70 / 0.15 / 0.15 within each advertiser's timeline) with **shared advertiser/creative
  vocabulary** across train/val/test.
- **Do NOT** evaluate on the original adversarial disjoint-advertiser split (it manufactured the
  retracted "LR beats neural" artifact; see `results/stage_a/rootcause_audit.md`).
- The legacy unified predictions `results/stage_a/test_predictions_all.npz` are **original-split**
  (diagnostic history only). Fair-split predictions:
  `results/models/escm2wc_dr_fair_posw/…` and `results/stage_a/{fair_baseline_preds,recalibrated_winners_preds}.npz`.

## 2. Ranking metrics — report TWO objects, never conflated

- **Winners-only AUC** = AUC of P(click \| win) on won impressions. **This is the object bidding
  consumes** and the primary ranking number.
- **All-bids AUC** = AUC of the P(click \| bid) product on all bids. Report separately; it is inflated
  by "easy negatives" (lost bids that never could click) and must not be compared against the
  winners-only number as if they were the same quantity.
- Reference (fair split): winners-only neural **0.658** > LGB 0.632 > LR 0.554.

## 3. Calibration metrics — quantile + slice, NOT global IEB

- **Primary:** **count-weighted quantile-ECE** + **per-slice signed bias** over `adexchange` and
  `advertiser` (`quantile_reliability`, `slice_calibration` in `src/metrics/calibration.py`).
- **Recalibration:** post-hoc **cross-fit isotonic** (`cross_fit_isotonic`, K-fold, leak-free
  out-of-fold). It is monotone → rank-preserving (assert winners-AUC drift < 5e-3).
- **Per-advertiser residual + segment recalibration.** Report per-advertiser residual IEB after the
  global map (a global monotone map cannot fix advertiser-specific level offsets). To close it, use
  **`segment_cross_fit_isotonic`** (a separate cross-fit isotonic per advertiser, global fallback for
  segments with < ~50 positives) — drives per-advertiser residual → ~0, preserves within-advertiser
  AUC (per-segment maps are monotone *within* a segment, NOT globally — global AUC may shift, usually
  *up*). Prefer it where each segment has enough clicks. (`scripts/stage_a/segment_calibration.py`.)
- **FORBIDDEN as a headline:** **global IEB** (`|mean(pred)−mean(true)|/mean(true)`). It is a single
  mean-cancellation scalar — a model can be 10/10-decile miscalibrated (under at the bottom, over at
  the top) yet report global IEB ≈ 0. Use it only as a coarse sanity scalar, never as the verdict.

## 4. Economic metric — realized won-only surplus + cluster CI (PRIMARY)

This is the primary metric for "is model A a better *bidder* than model B."

- **Definition:** V = pCTR × CPC_target (CPC_target = 200,000); shade to bids; run a won-only auction
  on the **actual observed payprice**; **realized surplus = Σ_{re-won}(click·CPC − payprice)**
  (`scripts/stage_a/stage_b2_surplus.py`, `surplus_corr.py:177-179`).
- **⚠️ Auction mechanism — use SECOND-PRICE.** iPinYou is second-price (you pay the market clearing
  price ≤ your bid). `stage_b2_surplus.py` historically ran a *first-price* auction (a bug); the
  corrected, mostly-observable **second-price** full-inventory projection is
  `scripts/stage_a/policy_value.py` / `src/bidding/policy_value.py` (`project_policy_value`). Under
  second-price, a policy's surplus is **exactly observable** wherever its bid ≤ the logged bid — so for
  conservative (truthful/shaded) policies the **won-only restriction barely binds** (≥99% of value is
  observed). Lost-inventory extrapolation needs a contextual market model `F(b|x)` that is **NOT
  identifiable** from iPinYou's 8 flat logged bid levels (P1 NO-GO) — do not claim lost-inventory value.
  Also exclude the ~1.2% won rows with payprice > bidprice (they violate second-price).
- **Model-independent value:** realized surplus uses the *actual* click, identical across models on
  identical rows; only *which* impressions a model re-wins (and what it pays) varies.
- **FORBIDDEN as a headline:** **`surplus(V)`** = Σ(V_model − payment) over wins, because it credits a
  model with its *own* (possibly inflated) V → over-prediction books phantom surplus. Report it only
  as a diagnostic.
- **Inference (mandatory):** realized surplus is driven by ~thousands of rare clicks → high variance.
  A point estimate is **not** a result. Report:
  - **Paired Poisson bootstrap CI** on the surplus *gap* between two models
    (`paired_bootstrap_surplus_gap`), and
  - the **conservative advertiser-cluster bootstrap CI** (`cluster_bootstrap_surplus_gap`) — the bound
    a decision-superiority claim must clear.
- **Comparison must equalize calibration first:** compare *recalibrated* models so mean V is equalized
  (assert `mean(V_recal)` equal across models to rtol 1e-2); then among-recal gaps isolate ranking +
  residual slice-calibration.
- **Robustness (mandatory):** report the gap under ≥3 shading strategies (`exchange_optimal`,
  `dual_regime` = headline, `truthful` = no-shading control) and a CPC sweep; the verdict must be
  **sign-consistent**.

## 5. Verdict bar

A decision-superiority claim (model A > model B as a bidder) is **SUPPORTED** iff: the among-recal
A−B **advertiser-cluster 95% CI excludes 0 under the SECOND-PRICE-optimal strategy (`truthful`)**, AND
sign-consistent across strategies, AND CPC-sweep sign-stable. Otherwise **NOT SUPPORTED** (acceptable).

- **⚠️ Use the SECOND-price-optimal strategy as headline** (`truthful`, not `dual_regime` — the latter
  is first-price-shaping; see §4 and `policy_value.py`).
- **Report heterogeneity, not just the cluster-mean.** With only ~5 advertisers the cluster-mean CI is
  low-power and can be dominated by one advertiser. Always report (`scripts/stage_a/power_analysis.py`,
  `src/metrics/cluster_inference.py`): **per-advertiser gap CIs** (which segments individually win),
  **Cochran's Q / I²** (homogeneous-underpowered vs *genuinely heterogeneous* — the decisive test), the
  **cluster-t MDE** (is the effect even detectable at this k?), and a **leave-one-advertiser-out** mean
  (single-advertiser leverage). A "wide mean-CI" is NOT automatically "underpowered" — if Q rejects
  homogeneity (e.g. neural−LGB: I²=0.82), the effect is heterogeneous and the **mean is the wrong
  estimand** (report fraction-of-advertisers-won instead).
- **Finer clustering (advertiser×day/hour) is FORBIDDEN** unless an intra-advertiser ICC≈0 gate passes
  (it doesn't here, ICC≈0.09) — it anti-conservatively manufactures significance. Advertiser is the
  cluster ceiling on iPinYou; **9-advertiser eval is a dead end** (disjoint-advertiser artifact).

## 6. Standing scope limitation

Won-only surplus scores only impressions the **original (non-debiased) policy won** — a cheap,
non-random slice. A debiasing model's value on **lost** inventory is unobservable offline (payprice is
right-censored). All surplus results are therefore a **conservative lower bound** on decision value,
not a full policy evaluation. State this caveat wherever surplus is reported.

---

## Quick reference

| dimension | use | do NOT use as headline |
|---|---|---|
| split | `features_fair/` (per-advertiser temporal, shared vocab) | original disjoint-advertiser split |
| ranking | winners-only AUC **and** all-bids AUC (separate) | a single conflated AUC |
| calibration | quantile-ECE + per-slice signed bias | global IEB |
| economic | realized won-only surplus + advertiser-cluster CI | `surplus(V)` (phantom) |
| inference | paired + cluster bootstrap on the gap | bare point estimate |
