# Stage A — B1: Reliability under quantile + slice scrutiny

n=19,424,025 | n_win=4,234,318 | clicks(all)=4,482 (all on won rows; base rate 2.307e-04) | winners click base rate 1.058e-03.

**Question.** The calibration headline (ESCM2-WC(DR) WCTR IEB ~0.014 in docs; ~0.073 retrained / ~0.045 ext-PS) is a SINGLE GLOBAL mean-bias ratio. Does it survive equal-FREQUENCY deciles + per-slice decomposition, or is it a mean-cancellation artifact?

- `global_IEB` = |mean(pred)-mean(true)|/mean(true) (legacy, the headline).
- `eqw_ECE` = legacy 10 equal-WIDTH bins (resolution ~0 at this base rate).
- `q_ECE` = count-weighted 10 equal-FREQUENCY deciles (the upgrade).
- `maxDecile|bias|` = worst single decile |mean_pred-mean_true|; `(xBR)` = that as a multiple of the base rate.
- `maxSlice|bias|` = worst level over adexchange/advertiser/hour.


## All-bids view (the object whose global IEB is the headline)

| Model | global_IEB | eqw_ECE | q_ECE | maxDecile\|bias\| (xBR) | maxSlice\|bias\| (adx / adv / hour) |
|---|---|---|---|---|---|
| esmmwc | 0.0750 | 1.7e-05 | 1.379e-04 | 4.046e-04 (1.8x) | 9.37e-04 / 9.37e-04 / 3.01e-04 |
| escm2wc_dr | 0.0726 | 1.7e-05 | 1.315e-04 | 2.951e-04 (1.3x) | 7.86e-04 / 7.86e-04 / 2.68e-04 |
| escm2wc_dr_extps | 0.0454 | 1.0e-05 | 1.386e-04 | 3.309e-04 (1.4x) | 7.73e-04 / 7.73e-04 / 2.76e-04 |
| lr_ctr_all | 0.1216 | 2.8e-05 | 9.597e-05 | 4.641e-04 (2.0x) | 8.85e-04 / 8.85e-04 / 2.48e-04 |
| lgb_ctr_all | 0.1347 | 9.2e-05 | 1.353e-04 | 8.321e-04 (3.6x) | 1.03e-03 / 1.03e-03 / 8.74e-04 |

## Winners-only view (the P(click|win) object that feeds bidding V(x))

| Model | global_IEB | eqw_ECE | q_ECE | maxDecile\|bias\| (xBR) | maxSlice\|bias\| (adx / adv / hour) |
|---|---|---|---|---|---|
| esmmwc | 0.5400 | 5.7e-04 | 6.818e-04 | 1.235e-03 (1.2x) | 3.92e-03 / 3.92e-03 / 1.97e-03 |
| escm2wc_dr | 0.5430 | 5.7e-04 | 5.748e-04 | 1.033e-03 (1.0x) | 3.77e-03 / 3.77e-03 / 1.98e-03 |
| escm2wc_dr_extps | 0.5186 | 5.5e-04 | 5.490e-04 | 1.132e-03 (1.1x) | 3.75e-03 / 3.75e-03 / 1.94e-03 |
| lr_ctr_all | 0.7672 | 8.1e-04 | 8.120e-04 | 3.022e-03 (2.9x) | 3.95e-03 / 3.95e-03 / 2.15e-03 |
| lgb_ctr_all | 0.6961 | 9.2e-04 | 7.933e-04 | 1.957e-03 (1.8x) | 4.10e-03 / 4.10e-03 / 2.19e-03 |

## Decile detail — escm2wc_dr winners-only (does 'near-oracle' hold per decile?)

| decile | mean_pred | mean_true | pred/true | count | signed bias |
|---|---|---|---|---|---|
| 0 | 3.098e-05 | 5.337e-04 | 0.06x | 423,432 | -5.028e-04 |
| 1 | 7.711e-05 | 1.110e-03 | 0.07x | 423,432 | -1.033e-03 |
| 2 | 1.394e-04 | 1.124e-03 | 0.12x | 423,432 | -9.848e-04 |
| 3 | 2.206e-04 | 1.086e-03 | 0.20x | 423,432 | -8.658e-04 |
| 4 | 3.156e-04 | 8.502e-04 | 0.37x | 423,436 | -5.346e-04 |
| 5 | 4.163e-04 | 8.974e-04 | 0.46x | 423,428 | -4.812e-04 |
| 6 | 5.339e-04 | 8.738e-04 | 0.61x | 423,432 | -3.399e-04 |
| 7 | 6.773e-04 | 8.596e-04 | 0.79x | 423,430 | -1.823e-04 |
| 8 | 8.857e-04 | 1.332e-03 | 0.66x | 423,432 | -4.463e-04 |
| 9 | 1.540e-03 | 1.918e-03 | 0.80x | 423,432 | -3.773e-04 |

## Verdict

**Verdict: the 'near-oracle calibration' headline does NOT survive binned/slice scrutiny — the small global IEB is a mean-cancellation artifact.**

### All-bids deciles (the object whose global IEB is the headline)

- **escm2wc_dr.** Global IEB 0.073 ("near-oracle"), yet the equal-frequency deciles SPLIT BY SIGN: 7 lower deciles UNDER-predict (ratio as low as ~0.0x) and 3 top deciles OVER-predict (ratio up to ~1.5-1.9x). Worst single decile |bias| = 2.95e-04 = 1.3x the 2.31e-04 base rate; quantile-ECE 1.32e-04 is ~8x the legacy equal-width ECE (1.7e-05). The opposite-sign deciles cancel in the single global mean.
- **escm2wc_dr_extps.** Global IEB 0.045 ("near-oracle"), yet the equal-frequency deciles SPLIT BY SIGN: 7 lower deciles UNDER-predict (ratio as low as ~0.0x) and 3 top deciles OVER-predict (ratio up to ~1.5-1.9x). Worst single decile |bias| = 3.31e-04 = 1.4x the 2.31e-04 base rate; quantile-ECE 1.39e-04 is ~13x the legacy equal-width ECE (1.0e-05). The opposite-sign deciles cancel in the single global mean.

### Slice decomposition (all-bids)

- **escm2wc_dr.** Worst slice = adexchange with |bias| = 7.86e-04 = 3.4x base rate. Across adexchange the bias FLIPS SIGN (2 levels over-predict, 3 under-predict): e.g. exchange 0 / advertiser 2997 under-predicts (~0.42x) while exchanges 1-2 over-predict (~2.7-3.5x). Hourly bias likewise flips (early-morning hours over-predict ~3x, evening peak hours under-predict ~0.5x). These cancel into the small global IEB.
- **escm2wc_dr_extps.** Worst slice = adexchange with |bias| = 7.73e-04 = 3.4x base rate. Across adexchange the bias FLIPS SIGN (2 levels over-predict, 3 under-predict): e.g. exchange 0 / advertiser 2997 under-predicts (~0.42x) while exchanges 1-2 over-predict (~2.7-3.5x). Hourly bias likewise flips (early-morning hours over-predict ~3x, evening peak hours under-predict ~0.5x). These cancel into the small global IEB.

### Winners-only object (the P(click|win) that feeds bidding V(x))

- **escm2wc_dr.** This object (not the all-bids product) is what drives V(x); its global IEB is 0.54 and it UNDER-predicts in 10/10 deciles (monotone shrinkage toward 0, ratios ~0.06x->0.85x). Worst slice |bias| = 3.77e-03 = 3.6x the 1.06e-03 winners base rate. Far from near-oracle.
- **escm2wc_dr_extps.** This object (not the all-bids product) is what drives V(x); its global IEB is 0.52 and it UNDER-predicts in 10/10 deciles (monotone shrinkage toward 0, ratios ~0.06x->0.85x). Worst slice |bias| = 3.75e-03 = 3.5x the 1.06e-03 winners base rate. Far from near-oracle.

**Bottom line.** For all three neural models the small all-bids IEB (0.045-0.075) coexists with (a) deciles that split into a systematic under-predicting lower half and over-predicting upper half, and (b) exchange/advertiser/hour slices whose signed bias FLIPS direction and reaches 3-4x the base rate. The global mean cancels these out, so IEB reports near-oracle calibration that is not present at any usable resolution. The baselines are no better calibrated per-decile (LR/LGB worst decile 2.0x / 3.6x base rate). For bidding, the relevant object is the winners-only P(click|win), which is grossly under-calibrated (global IEB 0.52-0.54, monotone shrinkage). RECOMMENDATION for the frozen eval spec: replace global IEB with the count-weighted quantile-ECE + per-slice signed-bias (max & sign-flip flag) as the primary calibration metric; the IEB->surplus chain in the bidding docs must be re-derived against decile/slice calibration, not the global mean.
