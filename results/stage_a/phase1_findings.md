# Stage A — Phase-1 Findings (synthesis + adversarial verification)

Scope: re-validate the data scenario and decompose the documented "LR baseline (all-bids AUC 0.7687)
beats neural debiasing" gap, then freeze a canonical eval spec. Falsification-first.

All numbers tagged: **[verified-recomputed]** = re-derived independently from the raw npz
(`results/stage_a/test_predictions_all.npz`) in this synthesis pass using the official
`src.metrics` functions; **[from-probe]** = taken from a Phase-1 probe artifact, not re-derived here;
**[needs-training]** = cannot be resolved with current artifacts, requires A2-full / A3 retraining.

---

## 0. Adversarial recomputation gate (did the probes over-state anything?)

Independently recomputed straight from the raw npz, bypassing the probe scripts, using
`_numpy_roc_auc`, `compute_ieb`, `quantile_reliability`:

| Quantity | Probe value | Re-derived value | |diff| | Tag |
|---|---|---|---|---|
| LR_ctr_all all-bids AUC | 0.7686519 | 0.7686519 | <1e-6 | [verified-recomputed] |
| LR_ctr_all winners-only AUC | 0.7143832 | 0.7143832 | <1e-6 | [verified-recomputed] |
| LR_ctr_all easy-neg gap | 0.0542687 | 0.0542687 | <1e-6 | [verified-recomputed] |
| escm2wc_dr all-bids AUC | 0.6850844 | 0.6850844 | <1e-6 | [verified-recomputed] |
| escm2wc_dr winners-only AUC | 0.5581832 | 0.5581832 | <1e-6 | [verified-recomputed] |
| escm2wc_dr winners-only quantile-ECE | 5.748e-04 | 5.748e-04 | <1e-7 | [verified-recomputed] |
| escm2wc_dr winners-only global IEB | 0.5431 | 0.5430 | <1e-3 | [verified-recomputed] |
| escm2wc_dr all-bids global IEB | 0.0726 | 0.0726 | <1e-4 | [verified-recomputed] |
| escm2wc_dr all-bids decile sign pattern | "bottom 7 −, top 3 +" | `-------+++` | exact | [verified-recomputed] |
| A4 corr(AUC, realized), real-only n=5 | Pearson 0.970 / Spear 1.000 | 0.970 / 1.000 | exact | [verified-recomputed] |
| A4 corr(IEB, −realized), real-only n=5 | Pearson 0.142 / Spear −0.300 | 0.142 / −0.300 | exact | [verified-recomputed] |
| A4 corr(IEB, −surplus(V)) full n=8 | Pearson 0.971 | 0.971 | exact | [verified-recomputed] |

Structural facts re-verified: n=19,424,025; n_win=4,234,318; clicks_total=4,482; **clicks on won=4,482,
clicks on lost=0** (clicks on lost auctions are structurally unobservable). The probes did NOT
over-state — every spot-checked headline reproduces to 6-7 sig figs. No discrepancy found between
probe artifacts and an independent recompute.

---

## (i) How much of the LR>neural AUC gap is EASY-NEGATIVES vs real skill?

**Verdict: mostly real winners-CTR skill, NOT the easy won/not-won contrast.**

| Model | all-bids AUC | winners-only AUC | easy-neg gap | Tag |
|---|---|---|---|---|
| **lr_ctr_all** | **0.7687** | **0.7144** | **+0.0543** | [verified-recomputed] |
| lgb_ctr_all | 0.5437 | 0.4786 | +0.0652 | [from-probe] |
| escm2wc_dr | 0.6851 | 0.5582 | +0.1269 | [verified-recomputed] |
| escm2wc_dr_extps | 0.6837 | 0.5602 | +0.1235 | [from-probe] |
| esmmwc | 0.6527 | 0.5234 | +0.1293 | [from-probe] |

- The LR headline 0.7687 is **not** an easy-negatives artifact. Restricting the *same* LR predictor
  to the won subset, AUC stays at **0.7144** — only **+0.054** (~7% of the headline) is attributable
  to the trivially-separable never-won bids. ~0.71/0.77 of LR's ranking power is genuine winners-CTR
  skill. [verified-recomputed]
- **Neural towers carry a 2.3-2.4x larger easy-neg gap (~0.124-0.129) and weak winners-only skill
  (0.52-0.56).** esmmwc's winners-only AUC (0.523) is barely above chance. On the like-for-like
  winners-CTR task the neural CTR towers genuinely underperform LR (0.714 vs 0.52-0.56) — this is a
  real skill gap, not an evaluation artifact. [verified-recomputed for escm2wc_dr; from-probe for others]
- **A distinct, larger component surfaced: the all-bids-vs-winners TRAINING-DATA effect.** The
  often-cited "LR winners-only ~0.32" is a *different object* — `lr_ctr`, the winners-**trained**
  model (test AUC 0.3216, anti-predictive). LR trained on the 19M-row all-bids population and evaluated
  on winners (0.7144) vastly outperforms LR trained only on the sparse won subset (0.3216). So the
  "LR baseline wins" story is primarily an **all-bids training-data** effect, only marginally an
  easy-negative evaluation effect. [from-probe — the 0.3216 figure is from `lr_ctr_result.json`,
  not re-derived here]

**Honest caveat:** "winners-only AUC" here uses each model's appropriate winners object (neural:
`p_ctr` = P(click|win); baselines: the single all-bids predictor restricted to won rows). It is *not*
the same as a model *trained* on winners. The decomposition cleanly isolates the **easy-negatives
component** (LR +0.054 vs neural ~0.126) but the larger **shift / train-population component** is only
partially characterized — fully isolating it is **[needs-training (A2 full / A3)]** (e.g. retrain LR
on winners-only with the all-bids feature pipeline; quantify the covariate shift between won and
all-bids feature distributions).

---

## (ii) Does surplus track CALIBRATION (IEB), not AUC? — A4

**Verdict: FALSIFIED once surplus is measured against model-independent realized value.
The "surplus tracks calibration" result holds ONLY for a self-referential bookkeeping quantity.**

Two surplus definitions, opposite conclusions:

| Surplus definition | Tracks | Pearson / Spearman (real-only n=5) | Tag |
|---|---|---|---|
| `surplus(V)` = Σ(model's own V − payment) | **IEB** (calibration) | corr(IEB, −surplus(V)) = +0.802 / +0.700 | [verified-recomputed] |
| `realized_surplus` = Σ(true_click·CPC) − spend | **AUC** (ranking) | corr(AUC, realized) = +0.970 / +1.000 | [verified-recomputed] |
| `realized_surplus` vs IEB | (no relation) | corr(IEB, −realized) = +0.142 / −0.300 | [verified-recomputed] |
| `surplus(V)` full n=8 (w/ scale variants) | IEB | corr(IEB, −surplus(V)) = +0.971 (Pearson) | [verified-recomputed] |

- `surplus(V)` is **circular**: it credits each model with its own V(x)=pCTR·CPC, so an over-predicting
  (high-IEB) model books phantom value it never realizes. The smoking gun: the monotone mis-scale
  `escm2wc_dr_scale2` has the *best* IEB (0.086) and *highest* surplus(V) (3.5e8) yet overbids hardest
  (win_rate 0.48, roi 2.13) — and its low IEB is a mean-matching accident (doubling a mean-pCTR that
  was ~half the truth), not distribution-wide calibration (its quantile-ECE and ranking are identical
  to scale1). [from-probe; sign and ordering verified-recomputed]
- **The model-independent `realized_surplus` reverses the story.** The economic winner is
  **lr_ctr_all** (realized 2.25e8, roi 7.21) — simultaneously the **highest-AUC (0.7687)** and the
  **worst-IEB (0.7673)** model. It wins by ranking clicks well and bidding gently (lowest mean bid,
  overpay 2.25), capturing real clicks cheaply despite terrible mean-calibration.
  **This single fact directly contradicts "surplus tracks calibration."** [verified-recomputed]

**Caveats (adversarial, must carry forward):**
- **N is tiny.** Real models n=5; n=8 only by adding 3 monotone scale-variants of one model that share
  its ranking — they inflate apparent N for the IEB↔surplus(V) claim without adding ranking variation,
  so the n=8 Pearson 0.971 is partly mechanical.
- **Leverage check (I re-ran it):** the probe warned "drop lr_ctr_all and the AUC↔realized signal
  weakens." That warning is **too pessimistic** — dropping lr_ctr_all gives Pearson **1.000** (n=4),
  and the 3 neural models alone give Pearson **0.999** over a narrow AUC range (0.653-0.685). So the
  AUC↔realized direction is *robust* to removing the anchor, not driven by it. [verified-recomputed]
  But the absolute AUC spread among non-LR models is small, so this is a *direction*, not a calibrated slope.
- realized_surplus rests on **4,482 clicks (~0.1% positives)** on the won-only subset (n=4,228,166);
  the V_true signal is sparse and won/lost selection is **not** corrected in the simulator. Treat
  magnitudes as directional. [from-probe]
- Quantile-ECE separates models; equal-width ECE collapses (all ~6e-4) at CTR≈1e-3 and is uninformative.

**Actionable for the freeze:** report **realized (model-independent) surplus alongside calibration,
NEVER surplus(V) alone** — the latter rewards overconfident pCTR by construction.

---

## (iii) Does the debiasing CALIBRATION headline survive quantile+slice scrutiny? — B1

**Verdict: NO. The "near-oracle calibration" headline is a mean-cancellation artifact.**

- **Documentation discrepancy [from-probe]:** docs cite Run AL WCTR IEB **0.014**, but the *retrained*
  `escm2wc_dr` artifact's all-bids global IEB is **0.0726** [verified-recomputed] — 5x worse;
  `escm2wc_dr_extps` is 0.0454. The headline is fragile before any binning.
- **All-bids view (the object behind the headline):** for all three neural models the equal-frequency
  deciles split by sign — bottom 7 under-predict, top 3 over-predict. I independently reproduced the
  exact sign pattern `-------+++` for escm2wc_dr. [verified-recomputed] Count-weighted quantile-ECE
  (1.3e-4) is ~8-13x the equal-width ECE (~1e-5, itself ~0 only because every score collapses into the
  first equal-width bin). Opposite-sign deciles cancel into the small global mean. [from-probe + decile
  pattern verified-recomputed]
- **Slice decomposition (all-bids) [from-probe]:** worst slice |bias| = 3.4x base rate, **sign flips
  across levels** (exchange 0 under-predicts ~0.42x; exchanges 1-2 over-predict ~2.7-3.5x; advertiser
  2259 over-predicts ~5.6x; hourly bias flips, 14 hours +, 10 hours −). Heterogeneous opposite-sign
  biases average to the small global IEB — exactly the structure that masks region-level overbidding.
- **Winners-only object — the P(click|win) that actually feeds V(x) in bidding:** grossly miscalibrated,
  global IEB **0.52-0.54** [verified-recomputed: escm2wc_dr 0.543], **under-predicting in 10/10 deciles**
  (monotone shrinkage toward 0), worst slice 3.5-3.6x base rate. The calibration the bidding chain relies
  on is NOT the well-behaved all-bids object the headline implies. [from-probe + global IEB verified]
- Baselines are no better per-decile (LR worst decile 2.0x, LGB top decile over-predicts 3.24x). [from-probe]

**Actionable for the freeze:** drop global IEB as the primary calibration metric; replace with
**count-weighted quantile-ECE + per-slice signed-bias (max |bias| + a sign-flip/heterogeneity flag)**.
The IEB→overbidding→surplus chain in the bidding docs must be re-derived against decile/slice
calibration, because the global-mean IEB masks the per-region over-prediction that would actually
cause overbidding.

---

## Implications for downstream stages

### Stage A4 — canonical eval-spec freeze
1. **Report TWO ranking objects, never one:** all-bids AUC *and* winners-only AUC (P(click|win) restricted
   to won), so the easy-negatives component (gap) is always visible. The single all-bids AUC is not a
   like-for-like model comparison. [verified-recomputed basis]
2. **Calibration metric = count-weighted quantile-ECE + per-slice signed-bias (max |bias|, sign-flip flag).**
   Demote global IEB to a secondary/diagnostic number — it is a mean-cancellation artifact here. [B1]
3. **Calibrate/report on BOTH objects** — all-bids product *and* winners-only P(click|win) — because the
   winners-only object (IEB 0.52-0.54) is the one the bidder consumes, and it is far worse than the
   all-bids headline. [verified-recomputed]
4. **Economic metric = realized (model-independent) surplus, reported with spend/roi/overpay/win_rate;
   surplus(V) is banned as a standalone headline.** Carry the n=5 / 4,482-click caveat in the spec so no
   future reader treats the A4 correlations as precise. [A4, verified-recomputed]

### Stage B2 — debiasing-vs-recalibration ablation (the central test)
- The data make the **null hypothesis sharp and plausible:** a high-AUC, badly-calibrated ranker
  (lr_ctr_all: AUC 0.7687, IEB 0.7673) is the **best realized bidder**. So a cheap **post-hoc recalibration
  of a strong ranker** (e.g. isotonic/Platt on lr_ctr_all or lgb, on the winners-only object) is the
  must-beat baseline. If ESCM²-WC debiasing cannot beat "best ranker + recalibration" on *realized surplus*
  and *quantile/slice calibration*, the debiasing machinery is not earning its complexity. [implication
  from verified A4 + B1]
- B2 must evaluate on **realized surplus + quantile/slice calibration**, NOT global IEB or surplus(V),
  or it will reward exactly the mean-cancellation / phantom-value artifacts identified above.
- The winners-only object is where debiasing must prove itself: all neural towers under-predict in 10/10
  deciles there (IEB ~0.53), so "ESCM²-WC produces unbiased P(click|win)" is currently **false on this
  artifact** and is the precise claim B2 should test. [verified-recomputed]
- **Open items requiring training, not resolvable now [needs-training (A2 full / A3)]:** (a) full
  shift/task-difficulty/easy-negative decomposition — isolate covariate shift (won vs all-bids feature
  dist.) from training-population effect by retraining LR winners-only under the all-bids pipeline;
  (b) the 0.014 vs 0.0726 IEB documentation gap — confirm whether the canonical artifact or the doc is
  stale by a controlled retrain.
