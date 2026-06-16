# Numbers Ledger — single source of truth

**Purpose.** Every headline number used in the public portfolio (`README.md`, `README.ko.md`,
`docs/technical_report.md`, figures) is pinned here to its **committed artifact** in
`results/stage_a/*.json`. Authoring reads from this ledger, never re-derives. Built by adversarially
reconciling the JSON ledgers against the prose in `docs/redesign_findings.md`; where prose and the
committed JSON disagree, **the committed JSON wins** and the discrepancy is logged in §J.

> All `*e6` / `*e8` figures are realized second-price surplus in raw CPM·count units (CPC target =
> 200,000). "Winners-only AUC" = AUC of P(click | win) on the won subset — the object bidding uses.
> "IEB" = integrated expected bias (mean predicted − mean true, normalized); 0 = mean-calibrated.

---

## A. Dataset & fair split

| key | value | source |
|---|---|---|
| Full dataset (S2+S3) | 129.5M bids, 30.6M impressions, 23K clicks | project EDA / `CLAUDE.md` |
| Market price | median 70, mean 80 CPM; floor binding 32.24% | project EDA |
| Fair split | per-advertiser temporal 0.70 / 0.15 / 0.15, shared advertiser+creative vocab | `fair_baselines.json:split` |
| Feature count | 30 | `fair_baselines.json:n_features` |
| Split sizes | train 90,645,454 · val 19,424,024 · test 19,424,020 | `fair_baselines.json:sizes` |
| Test win rate | 0.2892 | `policy_value.json:_meta.winrate` |
| Test winners | **5,616,945** | `fair_baselines.json:test_won`, `recalibration.json` |
| Test clicks (among winners) | **4,534** | `fair_baselines.json:test_clicks` |
| Test winners CTR | 0.000803 | `fair_baselines.json:test_winners_ctr` |
| Won advertisers | {1458, 3358, 3386, 3427, 3476} (5 = the full shared-vocab population) | `fair_baselines.json:test_won_advertisers` |
| Surplus subset (payprice>0) | **5,616,873** winners (72 dropped), **4,512** clicks | `stage_b2_surplus.json:_meta`, `power_analysis.json:_meta` |
| CPC target | 200,000 | all surplus JSONs |

> **4,534 vs 4,512 is NOT a contradiction:** 4,534 = clicks on all test winners; 4,512 = clicks on the
> `payprice>0` subset used by surplus/power analysis (72 payprice=0 winner rows dropped). Likewise
> 5,616,945 (calibration) vs 5,616,873 (surplus) differ by those 72 rows.

## B. Fair-split ranking — winners-only AUC (the bidding object)

| model | winners-only AUC | all-bids AUC | source |
|---|---|---|---|
| **escm2wc_dr (neural)** | **0.658** | 0.615 | `fair_comparison.json`, `fair_baselines.json` |
| LGB (ctr_all) | 0.632 | 0.720 | `fair_baselines.json:LGB_ctr_all` |
| LR (ctr_all) | 0.554 | 0.685 | `fair_baselines.json:LR_ctr_all` |

- neural − LGB winners-AUC gap = **+0.026** (`fair_comparison.json:verdict`).
- **Honest nuance:** neural leads the *winners-only* object (what a bidder ranks on) but **trails on
  all-bids** (LGB 0.720 > LR 0.685 > neural 0.615). The old "LR 0.7687 all-bids" headline is gone.

## C. The retraction — the original negative result was an artifact

| key | value | source |
|---|---|---|
| LR winners-AUC: old adversarial split → fair split | 0.7144 → **0.5540** (Δ = +0.160 drop) | `fair_baselines.json:verdict` |
| LGB winners-AUC: old adversarial split → fair split | 0.4786 → **0.6321** (reversal) | `fair_baselines.json:verdict` |
| Decisive advertiser 2997 in fair test-won subset? | **No** (absent) | `fair_baselines.json:verdict` |
| confirmed_artifact | **true** | `fair_baselines.json:verdict` |
| Exclude advertiser 2997 (old split) → all models AUC | ≈ **0.499** (chance) | `rootcause_audit.md` |
| Oracle advertiser-base-rate AUC (old split) | 0.761 | `rootcause_audit.md` |
| Train∩test advertisers / creative-vocab overlap (old split) | ∅ / **0/55** | `rootcause_audit.md` |

## D. Global recalibration — cross-fit isotonic (K=5, leak-free, GPU 0, rank-preserving)

| model | winners IEB before → after | under-deciles | winners AUC before → after | source |
|---|---|---|---|---|
| **neural (escm2wc_dr)** | **0.597 → 0.000** | 10 → 0 | 0.658 → 0.656 | `recalibration.json:escm2wc_dr` |
| LR (ctr_all) | 0.435 → 0.000 | 3 → 0 | 0.554 → 0.558 | `recalibration.json:lr_ctr_all` |
| LGB (ctr_all) | 0.476 → 0.000 | 9 → 0 | 0.632 → 0.634 | `recalibration.json:lgb_ctr_all` |

- Neural per-advertiser residual IEB **after global** map: {1458: 0.007, 3358: 0.209, 3386: 0.155,
  3427: 0.085, 3476: **0.226**} → max **0.226** (a single global map cannot fix segment-level bias).
  Source: `recalibration.json:escm2wc_dr.per_advertiser_ieb_after`.

## E. Segment-aware (per-advertiser) calibration — last residual solved

| model | per-adv max residual IEB: raw → global → **segment** | global AUC: raw → global → segment | source |
|---|---|---|---|
| neural | 0.709 → 0.226 → **0.0006** | 0.658 → 0.656 → **0.666** | `segment_calibration.json:models.neural` |
| LR | 0.823 → 0.284 → **0.0003** | 0.554 → 0.558 → 0.577 | `segment_calibration.json:models.lr` |
| LGB | 0.738 → 0.272 → **0.0002** | 0.632 → 0.634 → 0.649 | `segment_calibration.json:models.lgb` |

- Bidding payoff (segment − global realized surplus, dual_regime): neural **+7.4e6** (cluster CI
  [−1.77M, 19.15M], ns); LR +6.1e6 (ns); **LGB +12.9e6 (cluster CI [8.85M, 17.33M] excludes 0 ✓)**.
- Verdict: `CALIBRATION_WIN_DECISION_NEUTRAL` (residual is a calibration nicety, not a won-only bidding
  lever for neural). Source: `segment_calibration.json:verdict` + `surplus_segment_vs_global`.

## F. Stage B2 — decision value (second-price, among-recal ranking contribution)

Mean V equalized across recal models ≈ **160.65 CPM** (`stage_b2_surplus.json:_meta.mean_V_recal_equalized`).
Source: `stage_b2_surplus.json:decomposition.ranking_contribution_among_recal`.

| strategy | neural − LGB | cluster CI | neural − LR | cluster CI |
|---|---|---|---|---|
| exchange_optimal | +1.80e7 | **[6.18M, 30.05M] ✓** | **+3.63e7** | **[25.79M, 43.41M] ✓** |
| dual_regime | +3.61e6 | [−16.51M, 31.44M] ✗ | −5.88e6 | [−12.41M, 0.64M] ✗ |
| **truthful (2p-optimal)** ★ | +9.38e6 | [−11.06M, 40.70M] ✗ | +2.74e7 | **[17.72M, 37.75M] ✓** |

- `thesis_supported = false` under the corrected second-price mechanism (`stage_b2_surplus.json:verdict`).
- Mechanism reversal (1p → 2p): best strategy 1p = `exchange_optimal`, 2p = `truthful`; `truthful`
  surplus 1p = −7.26e6 (neural, overpaying) → 2p = +4.40e8 (`stage_b2_surplus.json:mechanism_comparison`).
- ⚠️ **Ledger correction:** `redesign_findings.md` §5 lists `exchange_optimal` neural−LR as **+5.4e7**;
  the committed grid gives **+3.63e7** (neural recal 350,926,849 − LR recal 314,599,257). Use **+3.63e7**.

## G. Power / heterogeneity analysis — truthful 2p (the honest verdict)

Source: `power_analysis.json`.

**neural − LGB (NOT robust):**

| advertiser | gap | 95% CI | CI excl 0 | n_clicks |
|---|---|---|---|---|
| 3427 | **+13.93e6** | [7.63M, 20.44M] | **✓** | 1410 |
| 3386 | +0.22e6 | [−4.99M, 5.44M] | ✗ | 1012 |
| 3476 | −0.24e6 | [−3.08M, 2.57M] | ✗ | 506 |
| 1458 | −0.86e6 | [−4.25M, 2.56M] | ✗ | 1172 |
| 3358 | −3.68e6 | [−7.46M, 0.15M] | ✗ | 412 |

- Positive on **2/5**, CI-significant on **1/5** (only 3427). Cluster mean **+9.38e6**, CI
  **[−11.06M, 40.70M]**, p=0.6725 → contains 0.
- **Cochran's Q = 22.04** (df 4, **p = 0.0002**), **I² = 0.819**, τ² = 1.85e13 → genuinely heterogeneous.
- **LOAO:** dropping 3427 flips cluster mean to **−1.14e6**.
- MDE(80%) = **11.48e6** ≫ observed per-adv mean **1.88e6** → underpowered design.
- ICC(per-adv×hour) = 0.090 → finer clustering forbidden.

**neural − LR (robust):** positive **5/5**; CI-significant 2/5 (1458 +8.22e6 ✓, 3386 +7.86e6 ✓);
Q = 7.72 (p = 0.1024), I² = 0.482; cluster mean **+27.44e6**, CI **[17.72M, 37.75M] ✓**; sign-test p = 0.03125.

- Verdict (`power_analysis.json:verdict`): **neural_vs_lgb = NOT robust; neural_vs_lr = robust.**

## H. Full-inventory policy value — second-price truthful (escape won-only)

Source: `policy_value.json`.

| model | V(π) | V_exact | V_model | modeled share | mean bid |
|---|---|---|---|---|---|
| **neural** | **4.39e8** (438,993,941) | 4.37e8 | 2.09e6 | **0.48%** | 156.25 |
| LGB | 4.293e8 (429,307,359) | 4.278e8 | 1.52e6 | 0.35% | 132.39 |
| LR | 4.170e8 (416,964,049) | 4.139e8 | 3.08e6 | 0.74% | 142.04 |

- **≥99.26% of every model's value is EXACT** observed second-price surplus (max modeled share 0.74%).
- Full-inventory gap: neural−LR **+22.03e6** (cluster CI [5.47M, 39.82M] **✓**); neural−LGB **+9.69e6**
  (cluster CI [−16.36M, 46.80M] **✗**, p=0.659) — consistent with Stage B2 §F/§G.
- **P2 sanity:** logged-policy value = realized second-price surplus = **455,419,091** (exact match).
- **Data quirk:** 69,568 won rows (1.24%) have payprice > bidprice (violates 2p), correctly excluded.
- **P1 NO-GO:** contextual market model F(b|x) calibrates on only **13.3%** of logged (segment×bid)
  cells (`policy_value.json:P1_market_calibration.frac_in_wilson_ci` = 0.133) → lost-inventory
  extrapolation not credible (but ≤0.74% of value, so it barely binds).

## I. Training-stage calibration (Stage 4) — NEGATIVE (post-hoc isotonic wins)

Source: `stage4_calibration.json` + `redesign_findings.md` §6.

| run | lever | native winners IEB | winners AUC |
|---|---|---|---|
| fair baseline | dr-mse, joint 0.1, pos_weight 50 (inert) | 0.597 | 0.658 |
| A2 | relax joint → 0.03 (dr-mse) | 0.778 (worse) | 0.638 |
| B2 | dr-bce, pos_weight 20 | 728 (wild over) | 0.537 (ranking dead) |
| C | dr-bce, pos_weight 2 | 161 (wild over) | 0.519 (ranking dead) |

- No train-time knob calibrates without breaking ranking; for every model × strategy, native − (its own
  isotonic) realized-surplus CI excludes 0 on the **negative** side → post-hoc isotonic is the answer.
- Frozen val→test (A2, honest temporal-shift test): global IEB **0.057**.

---

## J. Honesty flags & corrections (adopted in all portfolio docs)

1. **Verdict framing.** `redesign_findings.md` §5 heading and §7 bottom line previously read "thesis
   SUPPORTED" — an **over-claim** contradicting their own body. The committed
   `stage_b2_surplus.json:verdict.thesis_supported = false` and `power_analysis.json:verdict` give the
   honest split verdict: **robust vs linear LR, NOT robust vs strong LGB (advertiser-heterogeneous,
   one-advertiser-driven).** Portfolio uses the honest verdict everywhere; the two stale headers in
   `redesign_findings.md` have since been corrected to "SPLIT" to match the doc's body and this ledger.
2. **exchange_optimal neural−LR = +3.63e7**, not the +5.4e7 in §5 prose (see §F). Committed grid wins.
3. **4,534 vs 4,512 clicks** and **5,616,945 vs 5,616,873 winners** are context differences
   (all winners vs payprice>0 subset), not contradictions (see §A).
4. **`CLAUDE.md` stale labels.** `src/bidding/shading.py`, `src/bidding/simulator.py`,
   `src/bidding/policy_value.py`, `src/causal/*` are marked "(planned)" but are **implemented** and were
   used to produce Stage B2 / policy-value results. Describe them as implemented; do not call planned.
5. **Scope limits to keep front-and-center:** won-only surplus is a conservative lower bound (lost
   inventory untestable, P1 NO-GO); the advertiser-cluster CI has only **5 clusters** (low power).
6. **Architecture (trained model).** The committed fair DR run (`escm2wc_dr_fair_posw`) used
   **embedding_dim = 16, no FM interaction** (17 categorical + 13 numerical → 16-dim each → trunk
   **z ∈ ℝ⁴⁸⁰**); towers Win 480→64→32→1, CTR/Imputation 480→128→64→1. (The YAML default `embed_dim=32`
   was not what was trained; the architecture diagram reflects the trained model.)

---

# Part 2 — Strengthened fair-split experiments (2026-06, all CPU/GPU re-runs on `features_fair`)

The breadth experiments were re-run on the **fair** split so they are canonical (bidding, pacing) or
honestly exploratory (CATE, SCM), replacing the original/unfair-split versions. Sources are committed
`results/stage_a/*.json`.

## K. Ablation ladder — fair split (`ablation_ladder.json`)
ESMM-WC + ESCM²-WC(IPW) retrained on the fair split (2026-06) to close the ladder; neural rungs share
CTR supervision (ctr_weight=1, pos_weight=50, joint=0.1, embed=16), only the debiasing mechanism varies.

| rung | mechanism | winners-only AUC | winners IEB raw → recal |
|---|---|---|---|
| LR (ctr_all) | biased linear | 0.554 | −0.436 → 0.000 |
| LGB (ctr_all) | biased GBM | 0.632 | 0.476 → 0.000 |
| ESMM-WC | ESMM joint (implicit) | **0.674** | −37.76 → 0.000 |
| ESCM²-WC (IPW) | inverse-propensity | 0.656 | −32.46 → 0.000 |
| ESCM²-WC (DR) | doubly robust (primary) | 0.658 | 0.597 → 0.000 |

- **Honest reading:** all neural variants cluster ~0.656–0.674 (all beat LR 0.554, at/above LGB 0.632);
  the differences among them are small. ESMM-WC edges raw AUC but at catastrophic raw calibration
  (IEB −37.8, the direct-BCE+pos_weight over-prediction). **Calibration recalibrates to ≈0 for every
  rung regardless of raw** (range −38 to +0.6). DR is primary for its decision-value/calibration
  pipeline, **not** because it maximizes AUC.

## L. Bid-shading strategies — fair split (`bidding_fair.json`)
Recalibrated neural pCTR, second-price, 5,616,873 winners. Replaces original-split `strategy_comparison_*`.

| strategy | realized 2p surplus (×1e8) |
|---|---|
| **truthful (2p-optimal)** | **5.13** |
| percentile | 5.13 |
| linear (α=0.8) | 5.07 |
| exchange_optimal | 4.54 |
| dual_regime | 3.49 |

- **Linear α-sweep:** surplus rises 3.88e8 → 5.13e8 as α 0.4→1.0; ROI falls 3.40 → 2.42 (win-rate/cost tradeoff).

## M. Budget pacing — fair split (`pacing_fair.json`)
PID pacing over the 24-hour cycle, truthful 2p. Replaces original-split `pacing_comparison.csv`.

- **WR-weighted hourly allocation beats uniform PID by +11.2% to +13.7%** surplus across budget
  fractions {0.2, 0.4, 0.6, 0.8} of unconstrained spend; both beat front-loaded "no pacing".

## N. Bid-effect contrast (CATE) — fair split, EXPLORATORY (`cate_fair.json`)
**Naive within-advertiser difference-in-means** (lowest vs highest logged bid level), canonical
advertisers; bid-varying carriers {3358, 3427, 3476}. **NOT confounding-adjusted** — a CausalForestDML
is neither identifiable nor tractable on flat-bid/won-only data (data ceiling). Hypothesis-generating only.

- τ_win **−0.333**, τ_payment(won) **−31.6**, τ_click **−0.0004**, τ_surplus **+21.0** CPM.
- Surplus decomposition: volume channel (NIE) **−22.7**, cost channel (NDE) **+43.7** (V̄ ≈ 68 CPM).
- The negative τ_win (more bid → fewer wins) is a confounding artifact (bid level ↔ context), matching
  the original analysis — illustrates the data ceiling, not a causal claim.

## O. SCM / DAG — fair split, EXPLORATORY (`scm_fair.json`)
DoWhy backdoor (linear regression) on 19.1M clean rows; **not an identified causal claim** (flat-bid +
censored lost inventory). Replaces original-split notebook 09b.

| effect | estimate | 95% CI | refutation |
|---|---|---|---|
| bid → surplus | **−0.0659** | [−0.0767, −0.0569] | **✓ robust** (random-cause Δ≈0%, placebo ≈0, subset stable) |
| bid → win | −0.00073 | [−0.00077, −0.00069] | **✓ robust** |

- Both estimates pass all three refutation tests (robustness diagnostics), but remain
  **hypothesis-generating** given the identification ceiling.
