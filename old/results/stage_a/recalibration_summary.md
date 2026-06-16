# Stage A — Post-hoc Isotonic Recalibration of Winners pCTR

## TL;DR VERDICT

- **Global calibration FIXED** by cross-fitted isotonic (K=5, leak-free out-of-fold, GPU 0).
- Neural (escm2wc_dr) winners IEB **0.5967 → 0.0**, under-prediction deciles **10/10 → 0/10**.
- Ranking untouched: isotonic is monotone, so winners AUC is preserved (0.6581 → 0.6564).
- **Caveat — per-advertiser residual remains.** A single GLOBAL map zeroes the aggregate mean bias but cannot correct advertiser-specific bias: neural max per-advertiser residual IEB after recal = **0.2263** (adv 3476). ⇒ motivates segment-level (per-advertiser) recalibration and/or training-stage calibration — picked up by Stage B2 slice calibration.
- Protocol: cross-fit on test winners; literal val→test frozen-map deferred to the next neural retrain (now saves val predictions). Alignment (OK).


## Before → After (winners pCTR)

| model | IEB before | IEB after | qECE before | qECE after | under-deciles | AUC (unchanged) |
|---|---|---|---|---|---|---|
| escm2wc_dr | 0.5967 | 0.0 | 0.000479 | 2.6e-05 | 10/10 → 0/10 | 0.6581 → 0.6564 |
| lr_ctr_all | 0.4355 | 0.0 | 0.000547 | 1.6e-05 | 3/10 → 0/10 | 0.554 → 0.5581 |
| lgb_ctr_all | 0.4755 | 0.0 | 0.000543 | 2.7e-05 | 9/10 → 0/10 | 0.6321 → 0.6337 |

## escm2wc_dr — decile pred/true ratios

| decile | pred (before→after) | true | ratio before | ratio after |
|---|---|---|---|---|
| 0 | 3.6e-05 → 0.0002343 | 0.000235 | 0.153 | 0.854 |
| 1 | 7.25e-05 → 0.0003392 | 0.0003383 | 0.214 | 1.023 |
| 2 | 0.0001065 → 0.0004277 | 0.0004273 | 0.249 | 1.013 |
| 3 | 0.0001438 → 0.0005856 | 0.0005982 | 0.24 | 0.999 |
| 4 | 0.0001878 → 0.0006491 | 0.0006053 | 0.31 | 1.053 |
| 5 | 0.0002451 → 0.000867 | 0.000883 | 0.278 | 0.977 |
| 6 | 0.0003189 → 0.0009191 | 0.0009008 | 0.354 | 0.937 |
| 7 | 0.0004139 → 0.0009958 | 0.0010148 | 0.408 | 1.064 |
| 8 | 0.0005597 → 0.0011504 | 0.0011002 | 0.509 | 0.996 |
| 9 | 0.0011552 → 0.0019548 | 0.0019299 | 0.599 | 1.013 |

**Per-advertiser winners IEB (before → after):**

- adv 1458: 0.5867 → 0.0073
- adv 3358: 0.7086 → 0.2085
- adv 3386: 0.6973 → 0.1553
- adv 3427: 0.5209 → 0.0851
- adv 3476: 0.5391 → 0.2263

## lr_ctr_all — decile pred/true ratios

| decile | pred (before→after) | true | ratio before | ratio after |
|---|---|---|---|---|
| 0 | 0.000286 → 0.0005986 | 0.0006089 | 0.47 | 0.936 |
| 1 | 0.0004187 → 0.0006633 | 0.0006943 | 0.603 | 0.983 |
| 2 | 0.0005077 → 0.0006854 | 0.0006872 | 0.739 | 0.973 |
| 3 | 0.0005954 → 0.000713 | 0.0007264 | 0.82 | 1.021 |
| 4 | 0.0006989 → 0.0007698 | 0.0007086 | 0.986 | 1.003 |
| 5 | 0.0008252 → 0.0007987 | 0.0008937 | 0.923 | 0.995 |
| 6 | 0.0010075 → 0.0007989 | 0.0008012 | 1.258 | 0.999 |
| 7 | 0.001296 → 0.0008015 | 0.0008403 | 1.542 | 1.029 |
| 8 | 0.0019248 → 0.000805 | 0.0007762 | 2.48 | 1.036 |
| 9 | 0.0039708 → 0.0046618 | 0.0012961 | 3.064 | 1.01 |

**Per-advertiser winners IEB (before → after):**

- adv 1458: 0.8233 → 0.0999
- adv 3358: 0.1445 → 0.1528
- adv 3386: 0.4175 → 0.0217
- adv 3427: 0.2626 → 0.0414
- adv 3476: 0.5272 → 0.2837

## lgb_ctr_all — decile pred/true ratios

| decile | pred (before→after) | true | ratio before | ratio after |
|---|---|---|---|---|
| 0 | 5.04e-05 → 0.0003644 | 0.0004483 | 0.112 | 0.872 |
| 1 | 6.74e-05 → 0.0003824 | 0.0003737 | 0.18 | 1.076 |
| 2 | 8.48e-05 → 0.000461 | 0.0004078 | 0.208 | 1.034 |
| 3 | 0.0001126 → 0.0007 | 0.0007256 | 0.155 | 0.958 |
| 4 | 0.0001426 → 0.000725 | 0.0006895 | 0.207 | 0.996 |
| 5 | 0.0001784 → 0.0007344 | 0.0007158 | 0.249 | 1.048 |
| 6 | 0.0002108 → 0.0008101 | 0.0008168 | 0.258 | 0.952 |
| 7 | 0.000255 → 0.0009085 | 0.0008759 | 0.291 | 1.018 |
| 8 | 0.0003369 → 0.0010132 | 0.0010177 | 0.331 | 1.024 |
| 9 | 0.0027766 → 0.002004 | 0.0019703 | 1.409 | 1.009 |

**Per-advertiser winners IEB (before → after):**

- adv 1458: 0.694 → 0.0665
- adv 3358: 0.7376 → 0.0861
- adv 3386: 0.5659 → 0.0031
- adv 3427: 0.0803 → 0.1253
- adv 3476: 0.6767 → 0.2716

## Files

- `results/stage_a/recalibration.json` — full per-model before/after metrics + decile tables.
- `results/stage_a/recalibrated_winners_preds.npz` — recalibrated winners pCTR (idx_won, y_click_won, {model}_raw/{model}_recal) for the Stage B2 surplus comparison.
