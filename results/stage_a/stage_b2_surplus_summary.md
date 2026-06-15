# Stage B2 — Decision-level value of debiasing (SECOND-PRICE, realized surplus + slice calibration)

## TL;DR VERDICT

- **Auction = SECOND-PRICE (corrected from a first-price bug).** Winners pay the market clearing price (≤ bid), not their bid. Strategy ranking reversed: **2p-optimal = `truthful`** (1p-best was `exchange_optimal`).
- **Thesis NOT SUPPORTED.** Among-recal (mean V equalized ⇒ pure ranking + residual slice-calibration), headline **truthful** neural−lgb realized-surplus gap = **9.378e+06** (paired CI [-814891.7, 19527747.4], p=0.9644; advertiser-cluster CI [-11055312.0, 40701557.0], p=0.6725). (dual_regime gap 3.608e+06, cluster CI [-16506328.1, 31440325.0].)
- Sign-consistent across 3 strategies: **True**; CPC-sweep sign-stable: **True**.
- Rule: SUPPORTED iff neural-lgb advertiser-cluster 95% CI excludes 0 under the 2p-optimal strategy (truthful) AND sign-consistent across 3 strategies AND CPC-sweep sign-stable.
- **Scope:** won-only surplus cannot test debiasing's value on LOST inventory (censored payprice); conservative lower bound. Full-inventory view: policy_value.py.
- Setup: winners kept 5,616,873 (clicks 4,512), CPC=200000, mean V_recal equalized = {'neural': 160.7, 'lr': 160.7, 'lgb': 160.7}.


## Mechanism comparison: second-price (correct) vs first-price (old bug), recal surplus

| strategy | model | surplus 2p (correct) | surplus 1p (buggy) | neural−lgb 1p |
|---|---|---|---|---|
| exchange_optimal | neural | 3.509e+08 | 2.648e+08 | 2.437e+07 |
| exchange_optimal | lr | 3.146e+08 | 2.099e+08 |  |
| exchange_optimal | lgb | 3.329e+08 | 2.404e+08 |  |
| dual_regime | neural | 2.663e+08 | 2.060e+08 | 1.786e+07 |
| dual_regime | lr | 2.722e+08 | 1.818e+08 |  |
| dual_regime | lgb | 2.627e+08 | 1.882e+08 |  |
| truthful | neural | 4.398e+08 | -7.260e+06 | 6.787e+06 |
| truthful | lr | 4.123e+08 | -1.984e+07 |  |
| truthful | lgb | 4.304e+08 | -1.405e+07 |  |

*Under second-price you pay the market price, so surplus is higher and `truthful` is optimal (vs `exchange_optimal` under first-price). The old first-price headline (neural−lgb dual_regime ≈ +1.79e7) is superseded by the second-price numbers below.*


## Among-recal ranking gap (neural − baseline), by strategy [SECOND-PRICE]

| strategy | neural−lgb point | paired CI95 | p>0 | cluster CI95 | p>0 |  neural−lr point |
|---|---|---|---|---|---|---|
| exchange_optimal | 1.805e+07 | [8349542.2, 27666603.2] | 0.9996 | [6178801.0, 30045905.0] | 0.9995 | 3.633e+07 |
| dual_regime | 3.608e+06 | [-4620616.3, 12177088.7] | 0.8074 | [-16506328.1, 31440325.0] | 0.5897 | -5.885e+06 |
| truthful | 9.378e+06 | [-814891.7, 19527747.4] | 0.9644 | [-11055312.0, 40701557.0] | 0.6725 | 2.744e+07 |

## Realized surplus grid (raw → recal), by strategy

| strategy | model | raw surplus | recal surplus | Δ_cal | recal win_rate | recal clicks |
|---|---|---|---|---|---|---|
| exchange_optimal | neural | 1.612e+08 | 3.509e+08 | 1.897e+08 | 0.5987 | 2648 |
| exchange_optimal | lr | 2.603e+08 | 3.146e+08 | 5.431e+07 | 0.6765 | 2525 |
| exchange_optimal | lgb | 8.412e+07 | 3.329e+08 | 2.488e+08 | 0.618 | 2561 |
| dual_regime | neural | 1.363e+08 | 2.663e+08 | 1.299e+08 | 0.4932 | 2038 |
| dual_regime | lr | 2.685e+08 | 2.722e+08 | 3.711e+06 | 0.5758 | 2160 |
| dual_regime | lgb | 7.545e+07 | 2.627e+08 | 1.872e+08 | 0.5388 | 2070 |
| truthful | neural | 2.726e+08 | 4.398e+08 | 1.671e+08 | 0.8392 | 3744 |
| truthful | lr | 4.096e+08 | 4.123e+08 | 2.779e+06 | 0.8868 | 3620 |
| truthful | lgb | 1.529e+08 | 4.304e+08 | 2.775e+08 | 0.8723 | 3731 |

## Per-advertiser (dual_regime, recal): surplus gap vs residual calibration

| advertiser | neural−lgb surplus | neural resid IEB | lgb resid IEB |
|---|---|---|---|
| 3427 | 1.106e+07 | 0.0851 | 0.1253 |
| 3358 | 8.944e+05 | 0.2085 | 0.0861 |
| 3476 | -6.225e+05 | 0.2262 | 0.2715 |
| 3386 | -2.645e+06 | 0.1553 | 0.0031 |
| 1458 | -5.083e+06 | 0.0073 | 0.0665 |

*Reading: if neural's surplus edge concentrates in HIGH-residual-IEB advertisers, that is decision-level evidence for the slice-calibration mechanism global recal can't fix; if it sits in low-residual advertisers, the edge is pure ranking.*


## Sensitivity (dual_regime, among-recal neural−lgb)

| knob | value | neural−lgb point | p>0 |
|---|---|---|---|
| CPC | 1e+05 | 4.615e+06 | 0.962 |
| CPC | 2e+05 | 3.608e+06 | 0.8165 |
| CPC | 4e+05 | 2.113e+07 | 0.994 |
| max_bid | 300 | 3.608e+06 | 0.8165 |
| max_bid | 600 | 2.783e+06 | 0.754 |

## Files
- `/home/mail-agent/project/rtb_ipinyou/results/stage_a/stage_b2_surplus.json` — full grid, decomposition, bootstrap CIs, calibration, sensitivity.
