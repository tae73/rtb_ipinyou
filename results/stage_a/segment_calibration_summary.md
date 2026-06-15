# Stage 5 — Segment-aware (per-advertiser) calibration

## TL;DR VERDICT

- **[CALIBRATION_WIN_DECISION_NEUTRAL]** Per-advertiser calibration drives neural max residual 0.2263->0.0006 (fixes the one gap no global map could), BUT realized-surplus gap vs global is decision-neutral (cluster CI [-1766966.8, 19149874.2] contains 0) — the residual was a calibration nicety, not a bidding lever on won-only inventory.

- Global-isotonic per-advertiser ceiling to beat = **0.226** (neural). Segment map: 0.2263 → **0.0006**.
- **Fuller read (beyond the binary label):** the segment−global surplus gap is **consistently positive
  under dual_regime** for all 3 models (neural +7.4M, lr +6.1M, **lgb +12.9M, cluster CI excludes 0**),
  concentrated in the advertisers global mis-calibrated most (neural: 3386, 3358). Significant for LGB;
  positive-but-not-significant for neural/LR under the conservative **5-advertiser** cluster bootstrap
  (5 clusters = very low power). Bonus: per-advertiser level alignment **raises global AUC** (neural
  0.656→0.666). So: calibration fully solved; bidding impact small-positive & advertiser-concentrated.


## Per-advertiser residual IEB (raw / global / segment), by model

| model | regime | 1458 | 3358 | 3386 | 3427 | 3476 | max | within-adv AUC | global AUC |
|---|---|---|---|---|---|---|---|---|---|
| neural | raw | 0.5867 | 0.7086 | 0.6973 | 0.5209 | 0.5391 | 0.7086 | 0.66 | 0.6581 |
| neural | global | 0.0073 | 0.2085 | 0.1553 | 0.0851 | 0.2263 | 0.2263 | 0.658 | 0.6564 |
| neural | segment | 0.0 | 0.0006 | 0.0001 | 0.0 | 0.0 | 0.0006 | 0.6568 | 0.6655 |
| lr | raw | 0.8233 | 0.1445 | 0.4175 | 0.2626 | 0.5272 | 0.8233 | 0.5455 | 0.554 |
| lr | global | 0.0999 | 0.1528 | 0.0217 | 0.0414 | 0.2837 | 0.2837 | 0.5491 | 0.5581 |
| lr | segment | 0.0 | 0.0003 | 0.0001 | 0.0 | 0.0001 | 0.0003 | 0.5467 | 0.5766 |
| lgb | raw | 0.694 | 0.7376 | 0.5659 | 0.0803 | 0.6767 | 0.7376 | 0.6226 | 0.6321 |
| lgb | global | 0.0665 | 0.0861 | 0.0031 | 0.1253 | 0.2716 | 0.2716 | 0.6245 | 0.6337 |
| lgb | segment | 0.0002 | 0.0 | 0.0001 | 0.0001 | 0.0002 | 0.0002 | 0.6265 | 0.649 |

## Surplus: segment − global (does fixing the residual help bidding?)

| model | strategy | global surplus | segment surplus | seg−glob | cluster CI95 | p>0 |
|---|---|---|---|---|---|---|
| neural | exchange_optimal | 2.648e+08 | 2.600e+08 | -4.709e+06 | [-12428644.6, 2932538.6] | 0.132 |
| neural | dual_regime | 2.060e+08 | 2.134e+08 | 7.382e+06 | [-1766966.8, 19149874.2] | 0.9357 |
| neural | truthful | -7.260e+06 | 5.829e+06 | 1.309e+07 | [-43465299.3, 69643548.0] | 0.6777 |
| lr | exchange_optimal | 2.099e+08 | 2.092e+08 | -7.418e+05 | [-13229896.0, 11746333.2] | 0.408 |
| lr | dual_regime | 1.818e+08 | 1.879e+08 | 6.057e+06 | [-987230.1, 13331133.5] | 0.9527 |
| lr | truthful | -1.984e+07 | -1.264e+07 | 7.199e+06 | [-53310539.5, 66200509.0] | 0.6087 |
| lgb | exchange_optimal | 2.404e+08 | 2.522e+08 | 1.184e+07 | [-4577816.3, 32740928.2] | 0.9033 |
| lgb | dual_regime | 1.882e+08 | 2.011e+08 | 1.287e+07 | [8849746.8, 17326328.5] | 1.0 |
| lgb | truthful | -1.405e+07 | -3.519e+06 | 1.053e+07 | [-32081428.5, 63713271.3] | 0.6397 |

## Neural per-advertiser surplus delta (segment − global, dual_regime)

| advertiser | seg−glob surplus | (residual was: global → segment) |
|---|---|---|
| 3386 | 6.003e+06 | 0.1553 → 0.0001 |
| 3358 | 1.225e+06 | 0.2085 → 0.0006 |
| 3427 | 1.173e+06 | 0.0851 → 0.0 |
| 3476 | -3.109e+04 | 0.2263 → 0.0 |
| 1458 | -9.868e+05 | 0.0073 → 0.0 |

*Within-advertiser AUC is the ranking check (monotone per-advertiser maps preserve it); global AUC may shift because per-advertiser maps are not globally monotone — expected.*

## Files
- `results/stage_a/segment_calibration.json`, `segment_recalibrated_winners_preds.npz`.
