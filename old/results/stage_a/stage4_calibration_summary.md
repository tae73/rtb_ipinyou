# Stage 4 — Training-stage calibration (native / frozen val→test / cross-fit)

## TL;DR VERDICT

- **NEGATIVE (acceptable):** no training run beats the global-isotonic per-advertiser ceiling (0.226) without losing ranking/over-predicting. Post-hoc cross-fit isotonic remains the recommended calibration path (rank-preserving, leak-free, 1-line, GPU 0). Both levers fail in opposite directions: relax-joint (A2) → worse under-prediction (IEB 0.78); activate-pos_weight via DR-BCE (B2 pw20, C pw2) → wild over-prediction (IEB 161–729) **and** ranking collapse (AUC 0.66→0.52–0.54), with no sweet spot even at pos_weight=2.

- Ceiling to beat = global-isotonic max hard-advertiser (3358/3476) residual **0.226**; ranking floor AUC **0.653**.


## Calibration by regime (winners pCTR)

| run | regime | IEB | winners AUC | qECE | max hard-adv resid |
|---|---|---|---|---|---|
| fair_baseline | native | 0.5967 | 0.6581 | 0.000479 | 0.7086 |
| fair_baseline | crossfit | 0.0 | 0.6564 | 2.6e-05 | 0.2263 |
| A2_relax_jw003 | native | 0.7776 | 0.638 | 0.000625 | 0.8521 |
| A2_relax_jw003 | frozen_val2test | 0.0566 | 0.6351 | 7.4e-05 | 0.2877 |
| A2_relax_jw003 | crossfit | 0.0 | 0.6388 | 2.1e-05 | 0.2474 |
| B2_pw20 | native | 728.6422 | 0.5368 | 0.585306 | 837.3274 |
| B2_pw20 | frozen_val2test | 0.0251 | 0.5199 | 2e-05 | 0.3016 |
| B2_pw20 | crossfit | 0.0 | 0.5531 | 2e-05 | 0.231 |
| C_pw2 | native | 160.7902 | 0.5194 | 0.12916 | 89.0076 |
| C_pw2 | frozen_val2test | 0.0473 | 0.5279 | 3.8e-05 | 0.2272 |
| C_pw2 | crossfit | 0.0 | 0.5384 | 2.3e-05 | 0.1471 |

## Per-advertiser residual IEB (native / frozen / cross-fit)

| run | regime | 1458 | 3358 | 3386 | 3427 | 3476 |
|---|---|---|---|---|---|---|
| fair_baseline | native | 0.5867 | 0.7086 | 0.6973 | 0.5209 | 0.5391 |
| fair_baseline | crossfit | 0.0073 | 0.2085 | 0.1553 | 0.0851 | 0.2263 |
| A2_relax_jw003 | native | 0.7536 | 0.8521 | 0.8059 | 0.7631 | 0.7567 |
| A2_relax_jw003 | frozen_val2test | 0.0182 | 0.2877 | 0.1621 | 0.0227 | 0.1591 |
| A2_relax_jw003 | crossfit | 0.0215 | 0.2307 | 0.0982 | 0.0314 | 0.2474 |
| B2_pw20 | native | 554.2195 | 781.9677 | 593.8399 | 915.7902 | 837.3274 |
| B2_pw20 | frozen_val2test | 0.0859 | 0.0656 | 0.0754 | 0.117 | 0.3016 |
| B2_pw20 | crossfit | 0.1799 | 0.0068 | 0.1176 | 0.149 | 0.231 |
| C_pw2 | native | 41.8617 | 89.0076 | 50.4395 | 403.2431 | 39.7924 |
| C_pw2 | frozen_val2test | 0.0463 | 0.0257 | 0.056 | 0.156 | 0.2272 |
| C_pw2 | crossfit | 0.0969 | 0.0402 | 0.1183 | 0.1245 | 0.1471 |

## Surplus: native vs own cross-fit isotonic (does native still need post-hoc?)

| run | strategy | native surplus | iso surplus | native−iso | cluster CI95 | p>0 | meanV nat/iso |
|---|---|---|---|---|---|---|---|
| fair_baseline | exchange_optimal | 1.442e+08 | 2.648e+08 | -1.205e+08 | [-191895972.2, -61565833.3] | 0.0 | 64.79/160.65 |
| fair_baseline | dual_regime | 1.243e+08 | 2.060e+08 | -8.174e+07 | [-152625872.9, -21623342.4] | 0.0 | 64.79/160.65 |
| fair_baseline | truthful | 1.752e+08 | -7.260e+06 | 1.825e+08 | [56111236.7, 310133149.4] | 1.0 | 64.79/160.65 |
| A2_relax_jw003 | exchange_optimal | 6.321e+07 | 2.302e+08 | -1.670e+08 | [-233881232.3, -107491250.2] | 0.0 | 35.72/160.66 |
| A2_relax_jw003 | dual_regime | 5.632e+07 | 1.905e+08 | -1.342e+08 | [-241292698.5, -38471204.8] | 0.0 | 35.72/160.66 |
| A2_relax_jw003 | truthful | 1.100e+08 | -1.005e+07 | 1.201e+08 | [42086685.8, 200337253.5] | 1.0 | 35.72/160.66 |
| B2_pw20 | exchange_optimal | -7.113e+08 | 1.669e+08 | -8.781e+08 | [-1180121080.8, -553695777.3] | 0.0 | 117221.72/160.66 |
| B2_pw20 | dual_regime | -7.827e+08 | 1.327e+08 | -9.153e+08 | [-1200920757.8, -579971924.8] | 0.0 | 117221.72/160.66 |
| B2_pw20 | truthful | -7.827e+08 | -7.859e+07 | -7.041e+08 | [-929568038.6, -444493606.6] | 0.0 | 117221.72/160.66 |
| C_pw2 | exchange_optimal | -4.811e+08 | 1.623e+08 | -6.434e+08 | [-931409535.7, -377685533.5] | 0.0 | 25992.87/160.66 |
| C_pw2 | dual_regime | -5.565e+08 | 1.327e+08 | -6.892e+08 | [-966167964.7, -412148247.4] | 0.0 | 25992.87/160.66 |
| C_pw2 | truthful | -7.481e+08 | -7.922e+07 | -6.689e+08 | [-898634132.2, -430359127.0] | 0.0 | 25992.87/160.66 |

*Reading: if native−iso ≈ 0 (CI contains 0) the native model bids as well as its post-hoc isotonic → training calibration is non-inferior; if native ≪ iso, native still under-bids (under-prediction not fixed at train time) and post-hoc isotonic stays necessary.*


## Files
- `results/stage_a/stage4_calibration.json` — full metrics + per-adv + surplus CIs.
