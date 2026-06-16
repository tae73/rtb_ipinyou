# Root-Cause Audit — Why is debiasing + calibration underperforming?

**Researcher's question:** Is the negative debiasing/calibration result a GENUINE finding, or a CODE / DATA-STRUCTURE / MODELING bug-artifact?

**Scope:** Read-only audit. No source edits, no training. Synthesis of three independent audits (code / data / modeling) reconciled against an independent recomputation from `results/stage_a/test_predictions_all.npz`, the saved configs in `results/models/*_result.json`, the experiment grid (`results/models/{escm2wc_*,esmmwc_*}/`), and the feature parquets.

---

## TL;DR VERDICT (confidence: HIGH on the diagnosis, MEDIUM on the cure)

The negative result is **dominantly an EVALUATION-PROTOCOL / DATA-STRUCTURE ARTIFACT, not a genuine "debiasing fails" finding, and not primarily a loss-weight bug.** Two mechanisms, in priority order:

1. **PRIMARY — adversarial split.** Train and test advertiser/creative vocabularies are **100% disjoint** (train = Season-2 campaigns {1458,3358,3386,3427,3476}; test = Season-3 campaigns {2259,2261,2821,2997}; creative_hash overlap 0/55). The entire winners-CTR ranking signal at test is the *between-advertiser base rate*, and one unseen advertiser (2997, CTR 0.0044 ≈ 10x, 14.7% of won test rows) carries essentially all of it. **Remove 2997 and every model — neural AND LR — collapses to AUC ≈ 0.50.** The famous "LR 0.71 vs neural 0.56" gap is one unseen advertiser plus LR's accidental monotone extrapolation on the raw integer `advertiser` ID. This is independently re-confirmed below.

2. **SECONDARY (real but smaller) — calibration shrinkage from unweighted BCE at a 0.1% base rate + product-loss coupling.** The monotone 10/10-decile under-prediction of winners `p_ctr` is a genuine calibration symptom, but it is present **on the in-distribution val set too** (val ctr_ieb 0.51-0.67 from epoch 1), it is **decoupled from ranking** (LR under-predicts *more* — ratio 0.23 vs neural 0.46 — yet ranks far better), and it is **invariant to every loss-weight knob tried** in the grid. So loss weights are a contributor to absolute miscalibration, NOT the cause of the low test AUC.

The CODE audit's headline claim (ctr_weight∈{0,0.01} is the root cause) is **partially falsified**: the val→test AUC gap is invariant to ctr_weight (0.0/0.01/0.1) across 30+ grid runs. The DATA and MODELING audits are corroborated.

---

## (1) Ranked root-cause table

Tags: `[confirmed-from-data]` = re-derived from npz/parquet here; `[confirmed-from-code]` = read in source; `[hypothesis]` = plausible, not yet isolated.

| # | Root cause | Tag | Evidence (independently re-derived) | Impact on the negative result |
|---|---|---|---|---|
| **R1** | **Train/test advertiser & creative vocabularies 100% disjoint (S2→S3 temporal split)** | `[confirmed-from-data]` | `train∩test advertisers = ∅` (train {1458,3358,3386,3427,3476} vs test {2259,2261,2821,2997}); `creative_hash` overlap **0/55**; train season={2}, test season={3}. Source: `split_temporal` engineering.py:658. | **DOMINANT.** Neural/LGB identity embeddings for all 4 test advertisers are at/near init → no learned signal. This is the whole story behind near-chance neural winners-AUC. |
| **R2** | **Single unseen high-CTR advertiser (2997) is the entire test ranking signal** | `[confirmed-from-data]` | Oracle "advertiser-base-rate as score" winners-AUC = **0.761**. **Excluding 2997**, winners-AUC: escm2wc_dr 0.4988, esmmwc 0.4988, extps 0.4967, **LR 0.4986**, LGB 0.5497 — all ≈ chance. With 2997: LR 0.714, escm2wc_dr 0.558. | **DOMINANT.** The headline 0.71-vs-0.55 gap is 100% attributable to 2997. Within every advertiser, click is ~unpredictable for ALL models (AUC 0.48-0.56). |
| **R3** | **LR's "win" is accidental: raw integer `advertiser` ID monotone-extrapolates to the max-ID unseen advertiser** | `[confirmed-from-data]` | Raw `advertiser`-ID-as-score winners-AUC = **0.759** ≈ LR's 0.714; corr(LR_score, advertiser)=0.298; 2997 is the max ID (2997) → high score by luck of encoding. `esmmwc_scalar_r` (use_scalar_input=True = LR's exact representation, embedding removed) gets test AUC **0.401**, WORSE than embeddings. | HIGH (explanatory). LR is not "better-modeled"; it rides one fragile encoding coincidence. Falsifies "scalar/ordinality is the fix." |
| **R4** | **slotvisibility & slotformat 100% NULL at test (2 of 17 categoricals dead at inference)** | `[confirmed-from-data]` | null%: train 0.0 / val 18.0 / **test 100.0** for both. Filled with 0 by `_to_numpy_dtypes` engineering.py:812-828; 0 is also the dominant real train code. | HIGH on ceiling, but ORTHOGONAL to the AUC collapse (LR uses same features). Caps achievable test AUC; not the calibration-failure cause. |
| **R5** | **Unweighted BCE at 0.1% base rate → winners `p_ctr` shrinks toward base rate (monotone under-prediction)** | `[confirmed-from-code]` + `[confirmed-from-data]` | `binary_cross_entropy` base.py:400-419 has no `pos_weight`. Winners pred/true ratio: escm2wc_dr 0.457, esmmwc 0.460, extps 0.481. **Present on val from epoch 1** (val ctr_ieb 0.51 @ ep1, escm2wc_dr training_history). `p_ctr` capped at max 0.0155. | MEDIUM. Real calibration effect; the actual mechanism behind "10/10 deciles under-predict." But decoupled from ranking and from the test-AUC story. |
| **R6** | **Product-loss coupling `p_click_bid = p_win·p_ctr` + over-confident `p_win` deflates `p_ctr`** | `[confirmed-from-code]` + `[confirmed-from-data]` | escm2_wc.py:225 / esmm_wc.py:176; joint BCE escm2_wc.py:342. `p_click_bid` mean 0.000214 ≈ true all-bids 0.000231 (joint well-calibrated to ITS target); reconstructed `p_win` mean ≈ 0.41 vs true 0.218 → product forces `p_ctr` down. | MEDIUM. Amplifies R5's shrinkage when joint_weight ≫ ctr_weight. Contributor to absolute miscalibration. |
| **R7** | **ctr_weight ∈ {0.0, 0.01}, joint_weight=1.0 in saved runs (deliberate override, not YAML default 1/1/1)** | `[confirmed-from-code]` | Saved configs: esmmwc ctr_weight=0.0; escm2wc_dr/extps ctr_weight=0.01; all win_weight=0.01, joint_weight=1.0. YAML default is 1/1/1 (configs/model/*.yaml). Total loss esmm_wc.py:228-232. | **LOW–MEDIUM, and DEMOTED.** Grid shows test AUC is INVARIANT to ctr_weight 0.0→0.01→0.1 (0.553→0.553→0.477) and to win_weight 0.01→0.1→1.0; `esmmwc_lw_j` hit ctr_ieb=0.005 yet AUC 0.624. So this affects calibration absolute level, NOT the negative-result AUC. The CODE audit over-weighted this. |
| **R8** | **Early stopping on `ctr_auc` (pure ranking, winners-only), patience=5, best_epoch=5** | `[confirmed-from-code]` | es_metric=ctr_auc, patience=5, best_epoch=5/50 in all saved runs; val_loss=-ctr_auc (train.py:1885-1897). | LOW. ES on a ranking metric is insensitive to calibration, but val ctr_auc legitimately peaks ~0.71 by ep5 (training_history) — it is NOT stopping prematurely on ranking. Mild concern, not a driver. |
| **R9** | **DR-MSE objective has a downward pull on `p_ctr` at CTR≈0.1%** | `[confirmed-from-code]` + `[confirmed-from-data]` | escm2_wc.py:282-305: clipped (0.05–1) self-normalized weights, residual `(click−p_ctr)`; for click=0 majority the target pulls `p_ctr→0`; `delta_hat` is stop_gradient (line 299). | LOW. Ruled out as a driver: IPW vs DR vs ESMM all land 0.52-0.56 test; ESMM (no imputation) is even worse. Estimator choice does not move the result. |
| **R10** | **adexchange: value 4 unseen in train (train {1,2,3}), 9.2% of test + 10.5% null at test** | `[confirmed-from-data]` | npz adexchange min=0 max=4; test null 10.5%. | LOW. Untrained embedding row + null-fill on a minor structural feature. |
| **R11** | **region/city `-1` sentinel clipped to 0 (collides with region 0)** | `[confirmed-from-code]` | EmbeddingLayer `jnp.clip(x,0,max_idx)` base.py:222,229; ~4070 train rows. | NEGLIGIBLE (0.004%). |
| **R12** | **Dead `object_cols` branch in `load_feature_splits` (StringDtype cols are `str`, not `object`)** | `[confirmed-from-code]` | engineering.py:855 branch yields `object_cols=[]`; slot_size_group/region_group are saved via train.py's separate `is_string_dtype` loop (train.py:1521) → correctly int-coded {1..6}/{1..4}. | LATENT BUG, not active. Concern that these columns are zeroed is REFUTED for the WC path. Fix to avoid future silent breakage. |
| **R13** | **`season` constant per split (train S2, test S3)** | `[confirmed-from-data]` | nuniq(season)=1 per split. | LOW now (dead feature); leak hazard if val (mixed S2+S3) is used for tuning. |

---

## (2) THE single most likely explanation for the monotone 10/10 under-prediction of winners `p_ctr`

**It is two effects stacked, with the calibration symptom (R5) and the ranking failure (R1/R2) having DIFFERENT causes — they must not be conflated.**

- **The monotone shrinkage itself (every decile pred < true) is R5+R6:** an unweighted BCE/joint objective at a 0.106% winners base rate, where the product loss `p_win·p_ctr` (R6) is pinned to the all-bids click rate while `p_win` runs ~1.87x hot, mechanically squeezing `p_ctr` low. **Decisive proof this is a real model property, not a test-shift artifact: the same under-prediction appears on the in-distribution VAL set from epoch 1** (escm2wc_dr training_history: val ctr_ieb 0.509 @ep1, 0.670 @ep2). So the calibration bias is intrinsic to the objective + base rate.

- **BUT this is NOT why the comparison looks bad / why neural loses to LR.** Calibration and ranking are decoupled here: **LR under-predicts MORE severely** (winners pred/true ratio 0.233 vs neural 0.457) yet ranks far better (AUC 0.714 vs 0.558). The bad *ranking* (the actual "underperformance" the researcher sees) is R1/R2 — disjoint advertiser vocab + the single unseen 2997 axis — not the shrinkage.

**One-line answer:** the 10/10 under-prediction is a genuine, objective-induced calibration shrinkage (unweighted BCE at 0.1% base rate, amplified by the product coupling), but it is a red herring for the headline negative result; the *performance* gap is the adversarial disjoint-advertiser split.

---

## (3) DECISIVE PROBE PLAN (cheap, retraining-light, ordered by power-to-flip-the-verdict)

Each probe states exact setup and the GO (→ implementation artifact, debiasing might work) / NO-GO (→ genuine finding) that flips the verdict.

### PROBE A — Shared-vocabulary re-split (HIGHEST POWER; the one experiment most likely to flip the verdict)
- **Setup:** Re-split so every test advertiser & creative appears in train: either (i) random/grouped split stratified by `advertiser`, or (ii) fold S3 into train and hold out a random slice. Retrain ONLY escm2wc_dr (current config is fine) + re-fit LR/LGB. Compare winners-CTR AUC and IEB.
- **GO (artifact):** Neural winners-AUC and LR winners-AUC **converge** (both toward the within-advertiser ceiling ~0.55-0.60, gap < 0.05) and neural IEB drops sharply. → The negative result was the split; debiasing machinery is sound. **This is the predicted outcome.**
- **NO-GO (genuine):** Neural still trails LR by ≥0.10 on winners-AUC under a shared-vocab split. → Something in the neural/debiasing pipeline genuinely underperforms LR.
- **Cost:** 1 retrain (~30 min) + 2 sklearn/LGB fits.

### PROBE B — Within-advertiser & exclude-2997 stratified metrics (ZERO retrain; <5 min) — ALREADY DONE HERE
- **Setup:** From the existing npz, report winners-CTR AUC (a) per advertiser, (b) with 2997 excluded, (c) oracle advertiser-base-rate AUC.
- **Result (computed):** exclude-2997 → all models ≈ 0.499 (LR 0.4986, escm2wc_dr 0.4988); oracle base-rate = 0.761; raw-advertiser-ID = 0.759.
- **GO (artifact) — CONFIRMED:** All models collapse to chance without 2997 ⇒ the headline gap is one unseen advertiser, not model quality. **Verdict already flipped toward artifact by this probe.**

### PROBE C — pos-weighted / focal winners-CTR BCE at full ctr_weight (calibration isolation)
- **Setup:** Retrain escm2wc_dr with `ctr_weight=1.0, joint_weight∈{0.1,0}` and a `pos_weight≈1/base_rate` (or focal) on the winners-CTR BCE in base.py:400. Re-measure winners pred/true ratio per decile.
- **GO (calibration is fixable):** decile ratios move from {0.06…0.80} toward ~1.0 and global winners-IEB drops well below 0.5 — **without necessarily improving AUC** (expected, since ranking is R1/R2-bound).
- **NO-GO:** ratios stay <0.5 even with pos-weighting at full ctr_weight ⇒ a deeper objective bug (e.g., DR-MSE pull R9). 
- **Cost:** 1 retrain. **Purpose: isolate R5/R6/R7 (calibration) from R1/R2 (ranking). It should fix calibration but NOT the AUC — that asymmetry itself is diagnostic.**

### PROBE D — Frequency/target-encode advertiser & creative instead of identity embeddings (generalization mechanism)
- **Setup:** Replace `advertiser`/`creative_hash` identity embeddings with their leakage-safe frequency/target encodings (already present as `*_freq` features), keeping the temporal split. Retrain escm2wc_dr. (`esmmwc_nofreq_u` removed freq and got AUC 0.412 — the inverse experiment; this is the complement.)
- **GO (artifact, fixable):** neural winners-AUC rises toward LR's even on the disjoint split ⇒ the only thing missing was a transferable encoding of identity; debiasing is sound.
- **NO-GO:** no improvement ⇒ identity is genuinely non-transferable across seasons (supports "genuine, but evaluation-design-driven").
- **Cost:** feature swap + 1 retrain.

### PROBE E — Drop dead features, confirm ceiling (orthogonality check)
- **Setup:** Drop slotvisibility/slotformat (100% null at test) and handle adexchange=4/null explicitly; retrain. 
- **GO/Insight:** small AUC change ⇒ confirms R4 is a ceiling issue, orthogonal to the collapse (expected). Large change ⇒ dead features were actively misleading.
- **Cost:** trivial, 1 retrain. Low priority (diagnostic, won't flip the verdict).

**Minimal decisive set:** B (done — already flips toward artifact) + A (confirms with a clean split) + C (proves calibration is independently fixable). D/E are confirmatory.

---

## (4) Preliminary honest verdict

**The debiasing/calibration "underperformance" is NOT a genuine finding about ESCM²-WC/DR. It is an artifact of the evaluation protocol (adversarial disjoint-advertiser temporal split) plus a separate, real, but secondary calibration shrinkage from an unweighted objective at a 0.1% base rate.**

**Confidence by claim:**
- R1/R2 (disjoint split + single-advertiser signal is the cause of the AUC gap): **HIGH (~0.9).** Independently re-derived: exclude-2997 → all models ≈ 0.499; train∩test advertisers = ∅; creative overlap 0/55; oracle base-rate AUC 0.761 ≈ raw-advertiser-ID 0.759 ≈ LR 0.714.
- R3 (LR's edge is an encoding coincidence, not better modeling): **HIGH (~0.85).** scalar-input neural (LR's representation) is worse (0.401); LR rides the max-ID monotone extrapolation.
- R5/R6 (monotone under-prediction is real objective-induced shrinkage, present on val too): **HIGH (~0.85).** But its impact on the headline result: **LOW** — calibration is decoupled from the ranking gap.
- R7 (loss weights are the primary cause — the CODE audit's headline): **LOW (~0.15). Partially falsified** — test AUC invariant to ctr_weight/win_weight across 30+ grid runs.
- Cure confidence (a shared-vocab split + transferable encoding + pos-weighted BCE will turn the result positive): **MEDIUM (~0.6).** Probe B already shows within-advertiser CTR is near-unpredictable for everyone (AUC ~0.5), so the most likely *honest* post-fix conclusion is: **"winning-impression CTR is near-unpredictable from these features once you control for advertiser; the debiasing/calibration machinery is sound, and the prior negative comparison was an adversarial-split artifact"** — i.e. the headline flips from "debiasing fails" to "the benchmark was mis-designed," not necessarily to "debiasing wins big."

**Adversarial honesty in both directions:**
- Against manufacturing a bug: the loss-weight "bug" (R7) is real config-wise but does **not** explain the negative result; do not overclaim it. R9 (DR pull), R11 (sentinel), R12 (dead branch) are not drivers.
- Against excusing a real one: R5/R6 calibration shrinkage is a genuine model defect (under-predicts even in-distribution on val) and the unweighted BCE + product coupling should be fixed regardless of the split. Probe C must show it is independently fixable before declaring the pipeline "sound."

**Bottom line for the researcher:** Do not conclude anything about ESCM²-WC vs LR from the current numbers. Re-run on a shared-advertiser/creative split (Probe A) and report within-advertiser + exclude-2997 stratified metrics (Probe B, already showing all models ≈ chance). Separately fix the calibration objective (Probe C: pos-weighted/focal winners-CTR BCE at non-trivial ctr_weight) to address the monotone under-prediction. The negative result, as it stands, is an implementation/evaluation artifact.

---

### Independently recomputed evidence appendix
```
TEST won composition:   2259 ctr .00039 (7.8%) | 2261 .00030 (32.4%) | 2821 .00061 (45.2%) | 2997 .00444 (14.7%)
winners-CTR AUC (all won):     escm2wc_dr .5582  esmmwc .5234  extps .5602  LR .7144  LGB .4786
winners-CTR AUC (excl 2997):   escm2wc_dr .4988  esmmwc .4988  extps .4967  LR .4986  LGB .5497
oracle advertiser-base-rate AUC = .7614 ;  raw-advertiser-ID-as-score AUC = .7594 ;  corr(LR,adv)=.298
winners pred/true ratio:       escm2wc_dr .457  esmmwc .460  extps .481  LR .233  LGB .304   (true P(click|win)=.001058)
p_ctr won vs lost:             escm2wc_dr .000484 vs .000426  (tower barely win-conditions)
reconstructed p_win mean .41 vs true .218 ; p_click_bid mean .000214 ≈ true all-bids .000231
val ctr_auc trajectory (escm2wc_dr): ep1 .589 -> ep2 .627 -> ep3 .667 -> peak .709 @ep5 ; test_ctr_biased_auc .558  (val→test gap .15, same metric)
val ctr_ieb from ep1: .509 (under-prediction present IN-DISTRIBUTION, not just at test)
grid: test_ctr_biased_auc invariant to ctr_weight 0.0/.01/.1 = .553/.553/.477 ; esmmwc_scalar_r(scalar)= .401 ; esmmwc_lw_j ctr_ieb .005 but AUC .624
train∩test advertisers = {} ; creative_hash overlap 0/55 ; train season {2} test {3}
nulls: slotvisibility/slotformat 0%/18%/100% (train/val/test) ; adexchange test 10.5% ; filled 0 by _to_numpy_dtypes
```

Key files: `src/features/engineering.py:658` (`split_temporal` — artifact source), `:812-828` (`_to_numpy_dtypes` null→0 fill), `:855` (dead object_cols branch); `scripts/train.py:1521` (string→int auto-encode), `:1885-1897` (es_metric=ctr_auc), `:1989-2012` (raw p_ctr eval/save, no post-proc); `src/models/escm2_wc.py:225` (product), `:282-305` (DR-MSE weights), `:342` (joint BCE); `src/models/esmm_wc.py:176,224,228-232`; `src/models/base.py:222,229` (clip index), `:400-419` (unweighted BCE); `src/distributed/data_loader.py:148`; data `data/ipinyou/prediction/features/{train,val,test}.parquet`; predictions `results/stage_a/test_predictions_all.npz`; configs `results/models/{esmmwc,escm2wc_dr,escm2wc_dr_extps}_result.json` + grid `results/models/{escm2wc_*,esmmwc_*}/`.
