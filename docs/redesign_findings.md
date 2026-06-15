# RTB iPinYou — Redesign Findings (Stage A → B2)

**Status: CURRENT** · Last updated 2026-06-15 · Supersedes the headline of
`prediction_report.md` / `prediction_report_summary.md` (pre-redesign, unfair split).

This is the single up-to-date account of what the project actually found after the 2026-06-14
redesign. Machine-readable backing lives in `results/stage_a/*.json`; the frozen evaluation
contract is `docs/evaluation_protocol.md`; per-artifact navigation is `results/stage_a/README.md`.

---

## TL;DR

- The old headline — *"neural debiasing beats baseline prediction AUC"* — was **discarded**. On the
  original split it was an **artifact** (adversarial disjoint-advertiser split + an unsupervised CTR
  tower), not a real result.
- The thesis was **re-cast**: debiasing's value = **better-calibrated / better-ordered pCTR →
  better first-price bidding decisions (higher realized surplus)**.
- On a **fair** per-advertiser temporal split the ranking artifact disappears: neural winners-CTR
  **AUC 0.658 > LGB 0.632 > LR 0.554**.
- Post-hoc **cross-fit isotonic recalibration** zeroes global winners-pCTR bias (**IEB 0.597→0** for
  the neural model; same for LR/LGB) while preserving ranking.
- **Stage B2 decision test (corrected to second-price + heterogeneity-resolved): split verdict.**
  Neural's recalibrated pCTR **robustly beats the linear LR** baseline (5/5 advertisers, CI excludes 0)
  but does **NOT robustly beat the strong LGB** baseline — that edge is **advertiser-heterogeneous**
  (Cochran's Q I²=0.82, p=0.0002), CI-significant on only 1/5 advertisers (3427), and flips negative
  when 3427 is dropped. The original "decisively SUPPORTED" was a **first-price-bug + strategy +
  low-power-mean artifact** (§5/§8). Honest net: debiasing helps bidding vs a linear model, not (robustly)
  vs a strong GBM.
- **Stage 4 (training-stage calibration): NEGATIVE — and that's good.** No train-time knob calibrates
  the model without either worsening under-prediction or collapsing ranking (even pos_weight=2 over-
  predicts wildly via DR-BCE). Post-hoc cross-fit isotonic stays the answer (cheap, rank-preserving,
  GPU 0, and a *better bidder* than any native model — CI-confirmed).
- **Segment-aware calibration (per-advertiser): the last residual SOLVED.** A separate isotonic map
  per advertiser drives per-advertiser residual IEB **0.226 → 0.0006** (ranking preserved, global AUC
  even rises); bidding payoff is small-positive & advertiser-concentrated (significant for LGB).
- **Honest limits:** won-only evaluation (cannot test lost inventory); the surplus advertiser-cluster
  CI has only 5 clusters (low power).

---

## 1. Why the headline changed (redesign, 2026-06-14)

The pre-redesign program chased prediction **AUC** and reported neural debiasing as *losing* to a
logistic baseline (LR all-bids AUC 0.7687, ESMM-WC 0.6905) — see the now-superseded
`prediction_report.md` and the 20-phase `performance_tuning.md`. A root-cause audit showed that
result was an evaluation artifact, and that the surviving, defensible value of debiasing is
**calibration → bidding surplus**, not raw ranking. GPU-hour (a paid personal server) became a
first-class constraint, so the redesign favors cheap, falsification-first probes.

## 2. Stage A — the negative result was an artifact

Two independent bugs manufactured the "debiasing loses" headline (`results/stage_a/rootcause_audit.md`):

1. **Adversarial disjoint-advertiser split** — train (S2) and test (S3) advertisers were disjoint, so
   "LR 0.71" rode on a single unseen advertiser (2997) plus an accidental monotone raw-ID encoding,
   not on genuine CTR skill.
2. **Unsupervised CTR tower** — saved runs trained with `ctr_weight ≈ 0`, so the neural CTR head was
   learned only indirectly through the win×ctr product → systematic 10/10-decile under-prediction.

Early `results/stage_a/{reliability_summary,phase1_findings}.md` diagnose this on the **original/unfair**
predictions; they also falsified two metric choices (global IEB = mean-cancellation; `surplus(V)` =
phantom surplus) — see `docs/evaluation_protocol.md`.

## 3. Fair split — the ranking artifact disappears

The canonical **fair split** `data/ipinyou/prediction/features_fair/` uses a per-advertiser temporal
split (0.70/0.15/0.15) with **shared advertiser/creative vocabulary**. Retrained with proper CTR
supervision (`ctr_weight=1.0`, `--ctr-pos-weight`):

| model (fair split) | winners-only AUC | all-bids AUC |
|---|---|---|
| **escm2wc_dr (neural)** | **0.658** | — |
| LGB (ctr_all) | 0.632 | 0.720 |
| LR (ctr_all) | 0.554 | 0.685 |

Winners-only AUC (P(click \| win) — the object bidding actually uses) is where debiasing wins. The old
"LR 0.7687 all-bids" headline is gone; on the fair split LGB > LR on all-bids, and neural leads the
winners object. (Test: 5.62M winners over advertisers {1458, 3358, 3386, 3427, 3476}, 4,534 clicks.)

## 4. Recalibration — global calibration fixed, cheaply

The remaining issue was *calibration*: neural winners pCTR under-predicted (IEB 0.597, all 10 deciles
under). A **cross-fit isotonic recalibration** (K=5, leak-free out-of-fold, GPU 0; `cross_fit_isotonic`
in `src/metrics/calibration.py`, probe `scripts/stage_a/recalibrate.py`) fixed it for all three models
— and isotonic is monotone, so **ranking is preserved**:

| model | winners IEB before→after | under-deciles | winners AUC (unchanged) |
|---|---|---|---|
| escm2wc_dr | **0.597 → 0.000** | 10 → 0 | 0.658 → 0.656 |
| lr_ctr_all | 0.435 → 0.000 | 3 → 0 | 0.554 → 0.558 |
| lgb_ctr_all | 0.476 → 0.000 | 9 → 0 | 0.632 → 0.634 |

**Caveat:** a single *global* map zeroes only the aggregate mean bias. **Per-advertiser residual
remains** (neural adv-3476 IEB 0.226, 3358 0.209) — the segment-level gap that global recalibration
cannot fix. Detail: `results/stage_a/recalibration_summary.md`.

## 5. Stage B2 — decision-level value (thesis SPLIT: robust vs LR, not vs LGB)

With global calibration leveled, do neural's recalibrated predictions yield **better bidding
decisions** than the recalibrated baselines? Method (`scripts/stage_a/stage_b2_surplus.py`, mirrors
`surplus_corr.py`): V = pCTR×CPC → bid shading → won-only auction on **actual payprice** → **realized
surplus** = Σ_{re-won}(click·CPC − payprice).

**Clean identification:** cross-fit isotonic pins every recal model's mean V to the empirical base
rate, so all three recal cells share **mean V ≈ 160.7 CPM**. Among-recal surplus differences are
therefore **pure ranking + residual slice-calibration** — the mean-bid-level confound is removed by
construction.

**⚠️ CORRECTED to SECOND-PRICE.** iPinYou is a second-price auction (winners pay the market clearing
price ≤ bid); the original Stage-B2 ran a **first-price** simulation (a bug found in §8). Under the
correct mechanism the **strategy ranking reverses** — `truthful` (bid = V) becomes the 2p-optimal
(you only pay the market price, so bidding high wins more without overpaying), while the shaded
`exchange_optimal`/`dual_regime` are suboptimal. Re-running under second-price:

**Among-recal neural − baseline realized-surplus gap, by strategy (second-price):**

| strategy | neural − lgb | advertiser-cluster CI | neural − lr | cluster CI |
|---|---|---|---|---|
| exchange_optimal | +1.81e7 | **[6.2M, 30.0M] — excludes 0** | +5.4e7 | excludes 0 |
| dual_regime | +3.6e6 | [-16.5M, 31.4M] — ns | −5.9e6 | [-12.4M, 0.6M] |
| **truthful (2p-optimal)** | +9.4e6 | [-11.1M, 40.7M] — **ns** | +2.74e7 | **[17.7M, 37.8M] — excludes 0** |

- **Under the correct mechanism the cluster-mean gap loses significance** (truthful neural−lgb CI
  contains 0; only `exchange_optimal` is significant — a multiplicity red flag). neural leads LGB on
  point across strategies (sign-consistent) but the mean is below the design's detectability — which
  the Stage-8 heterogeneity analysis (below) resolves precisely.
- **Mechanism reversal confirmed:** under first-price `truthful` surplus was *negative* (−7.3e6,
  overpaying the full bid); under second-price it is the **best** (+4.40e8). Second-price surplus ≥
  first-price for every cell (asserted invariant).
- **Resolution — neural vs LGB is NOT robust; it is advertiser-heterogeneous (Stage 8 power analysis,
  `scripts/stage_a/power_analysis.py`).** On the 2p-optimal truthful strategy the neural−LGB gap is
  positive on only **2/5** advertisers, **CI-significant on 1/5** (only adv 3427, +13.9M), and
  **Cochran's Q rejects homogeneity (I²=0.82, p=0.0002)** — so the spread is *genuine heterogeneity*,
  not a homogeneous effect buried in noise. **Leave-one-advertiser-out:** dropping 3427 flips the mean
  negative (−1.1M). Design MDE (~11.5M) ≫ the per-advertiser mean (1.9M). The edge does **not** track
  calibration (on adv 1458 neural is *better*-calibrated yet *loses* surplus). ⇒ **debiasing does NOT
  robustly beat the strong GBM on bidding decisions; its apparent edge is concentrated in one advertiser.**
- **vs LR (linear baseline): neural IS robust** — positive on **5/5** advertisers, cluster CI
  [17.7M, 37.8M] excludes 0. The machinery detects a real effect when one exists.
- **Mechanism reversal confirmed:** under first-price `truthful` surplus was *negative* (−7.3e6,
  overpaying the full bid); under second-price it is the **best** (+4.40e8). 2p surplus ≥ 1p per cell.
- **Bottom line (decision-value thesis, honest):** debiasing's bidding value over a **linear** model is
  robust; over a **strong GBM** it is **not robust** (advertiser-heterogeneous, one-advertiser-driven).
  The original "decisively SUPPORTED" was a first-price-bug + strategy + low-power-mean artifact. 9-advertiser
  eval is a dead end (disjoint-advertiser artifact); 5 advertisers is the cluster ceiling. 1p numbers
  retained in `stage_b2_surplus.json → mechanism_comparison`.

Detail: `results/stage_a/{stage_b2_surplus,power_analysis}_summary.md` + `.json` (canonical = second-price).

## 6. Stage 4 — training-stage calibration: NEGATIVE (post-hoc isotonic wins)

We tested whether the model can be calibrated **at training time** (so it needs no post-hoc map and,
ideally, closes the per-advertiser residual). Three retrains on the fair split
(`scripts/stage_a/stage4_calibration.py`, results in `results/stage_a/stage4_calibration.{json,_summary.md}`):

| run | lever | native winners IEB | winners AUC |
|---|---|---|---|
| fair baseline | dr-mse, joint 0.1, pos_weight 50 (inert) | 0.597 | 0.658 |
| A2 | relax squeeze (dr-mse, **joint 0.03**) | **0.778** (worse) | 0.638 |
| B2 | activate pos_weight (**dr-bce, pos_weight 20**) | **728** (wild over) | **0.537** (ranking dead) |
| C | gentle pos_weight (**dr-bce, pos_weight 2**) | **161** (wild over) | **0.519** (ranking dead) |

- **No native model is calibrated.** The two available train-time levers fail in *opposite*
  directions: relaxing the joint constraint worsens under-prediction; activating `pos_weight` (which
  requires switching DR-MSE→**DR-BCE**) over-corrects catastrophically at **every** value tested —
  even pos_weight=2 — **and** collapses winners-AUC to ~0.52. There is no sweet spot: DR-MSE gives
  good ranking but ignores pos_weight; DR-BCE enables pos_weight but destroys ranking.
- **Post-hoc isotonic is confirmed (and provably better).** Cross-fit isotonic drives global IEB→0
  for every model, rank-preserving. The surplus check is decisive: for every model × shading strategy,
  **native − (its own isotonic) realized-surplus CI excludes 0 on the negative side** — the
  un-recalibrated model is a strictly *worse* bidder than its post-hoc-isotonic version. So you do
  **not** retrain to calibrate; the cheap downstream isotonic step is correct and superior.
- **frozen val→test (4b, honest temporal-shift test):** on the least-degraded retrain (A2), a map
  fit on **val** winners and applied to test gives global IEB **0.057** (vs cross-fit's ~0) — it
  mostly generalizes across the S2→S3 shift, with a small temporal penalty. (The good fair-baseline
  predates val-prediction saving, so frozen could only run on the Stage 4 retrains — a caveat.)

## 7. Segment-aware (per-advertiser) calibration — the residual SOLVED

The one gap no global map could fix — per-advertiser residual IEB (~0.15–0.28 after global cross-fit,
frozen, or any training lever) — is closed by fitting a **separate cross-fit isotonic map per
advertiser** (`segment_cross_fit_isotonic`, `scripts/stage_a/segment_calibration.py`):

| model | per-adv max residual: raw → global → **segment** | within-adv AUC | global AUC |
|---|---|---|---|
| neural (escm2wc_dr) | 0.709 → 0.226 → **0.0006** | 0.658 → 0.657 (kept) | 0.656 → **0.666** (↑) |
| lr | 0.823 → 0.284 → **0.0003** | kept | 0.558 → 0.577 (↑) |
| lgb | 0.738 → 0.272 → **0.0002** | kept | 0.634 → 0.649 (↑) |

- **Calibration fully solved at every level.** Per-advertiser residual drops 3 orders of magnitude to
  ~0; within-advertiser ranking is preserved (monotone per-advertiser maps); and global AUC even
  **rises** (per-advertiser level alignment makes cross-advertiser comparison more accurate).
- **Bidding impact: small-positive, advertiser-concentrated.** Under dual_regime the segment−global
  realized-surplus gap is consistently positive (neural +7.4M, lr +6.1M, **lgb +12.9M**), **significant
  for LGB** (advertiser-cluster CI excludes 0) but positive-not-significant for neural/LR under the
  conservative **5-advertiser** cluster bootstrap (5 clusters ⇒ low power). The neural gain
  concentrates in the advertisers global mis-calibrated most (3386 +6.0M, 3358 +1.2M). So the
  per-advertiser residual is a real calibration defect that segment calibration removes, with a
  mild-positive (not decisively significant on won-only inventory for neural) bidding payoff.

### Standing limitation
- **won-only** surplus still cannot test value on *lost* inventory (censored payprice) → conservative
  lower bound; and the advertiser-cluster CI has only 5 clusters (low power for the surplus delta).

### Bottom line
Thesis **SPLIT** (Stage B2 — robust vs a linear LR, **not** robustly vs a strong LGB; see §5/§8). The
calibration story is now **complete**: post-hoc cross-fit isotonic
fixes the global level (rank-preserving, GPU 0); training-stage calibration is NEGATIVE (Stage 4, don't
fight it in training); and **per-advertiser segment calibration** closes the last residual (and nudges
surplus up, significantly so for LGB). Recommended pipeline = fair-split training (DR-MSE,
ranking-optimal) → post-hoc cross-fit isotonic, **per-advertiser where the segment has enough clicks**
→ bidding.

## 8. Full-inventory policy value (escape won-only) — second-price, mostly-observable

The won-only surplus is a biased estimand. We estimate the **full-inventory** value of a bidding
policy over all 19.4M fair-test bids via a **structural second-price projection** (`policy_value.py`,
`src/bidding/policy_value.py`). **Honest framing: this is NOT off-policy evaluation** — the logging
policy is deterministic over 8 flat bid levels → no propensity. Instead the surplus is **exactly
observable** wherever a policy's bid is determinable from the log; under second-price a policy bidding
`b` wins iff market ≤ `b`, and where `b ≤` the logged bid the outcome is fully observed (we saw the
market price = payprice on wins). Truthful second-price bids (`b = V = pCTR·CPC ≈ 160`) sit *below* the
logged flat bids (227–300), so **the policy re-wins an observed subset → won-only barely binds.**

**Result (truthful second-price, V(π) = V_exact + V_model):**

| model | V(π) | exact | modeled | modeled-share |
|---|---|---|---|---|
| neural | **4.39e8** | 4.37e8 | 2.1e6 | 0.5% |
| lgb | 4.29e8 | 4.28e8 | 1.5e6 | 0.4% |
| lr | 4.17e8 | 4.14e8 | 3.1e6 | 0.7% |

- **≥99.3% of every model's value is EXACT** observed second-price surplus — the headline does *not*
  depend on the market model. Full-inventory neural−lgb = **+9.7e6** (cluster CI [-16.4M, 46.8M],
  p=0.66) — neural leads on point, low-power 5-cluster CI contains 0, **consistent with Stage B2**.
- **The de-risk ladder worked:** **P2 passes** (the estimator reproduces realized second-price surplus
  on the logged policy, exactly); **P1 is a NO-GO** — the contextual market model `F(b|x)` calibrates
  on only 13% of logged (segment×bid) cells (the flat-bid/context confounding makes it unidentifiable),
  so **lost-inventory extrapolation is *not* credible** — but it barely matters here (≤0.7% modeled).
- **Two data-quality / methodology findings:** (i) **Stage-B2 ran a *first-price* auction on
  *second-price* data** (`stage_b2_surplus.py:93`) — **now FIXED**: `stage_b2_surplus.py` is corrected
  to second-price (§5), which weakened the headline (the neural-vs-strong-LGB edge lost significance
  under the correct mechanism — an honest over-claim correction); (ii) **1.24% of won rows have
  payprice > bidprice**, violating second-price (an iPinYou data quirk), correctly excluded.
- **Conclusion:** the won-only limitation is far less binding than feared *for conservative
  (truthful/shaded) policies* — their value is ~99.5% observable on second-price data. Genuine
  lost-inventory value (aggressive policies bidding above the logged flat bids) **remains untestable**
  on this data, because the flat-bid logging makes the required `F(b|x)` unidentifiable (P1 NO-GO).

---

## Artifacts
- Code: `scripts/stage_a/{recalibrate,stage_b2_surplus,stage4_calibration,segment_calibration,policy_value}.py`,
  `src/metrics/calibration.py` (`cross_fit_isotonic`/`fit_isotonic`/`segment_cross_fit_isotonic`),
  `src/bidding/simulator.py` (`paired_/cluster_bootstrap_surplus_gap`),
  `src/bidding/policy_value.py` (`project_policy_value`/`MarketModel`/`contextual_optimal_bid`).
- Results: `results/stage_a/{recalibration,stage_b2_surplus,fair_baselines,fair_comparison}.json`,
  the `*_summary.md` reports, and `recalibrated_winners_preds.npz`.
- Protocol: `docs/evaluation_protocol.md`. Index: `results/stage_a/README.md`. Tracking: `PLAN.md`.
