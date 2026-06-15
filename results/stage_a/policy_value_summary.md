# Stage 6 — Full-inventory policy-value projection (escape won-only)

## TL;DR

- Full-inventory second-price truthful-bidding value: neural−lgb = 9.687e+06 (cluster CI [-16364172.9, 46797983.0], p=0.659). MOSTLY OBSERVABLE: ≤0.7% of value is extrapolated (the rest is EXACT second-price surplus). P2 sanity holds (estimator reproduces realized surplus on the logged policy).

- **Honest framing:** structural second-price policy-value projection (NOT OPE — deterministic flat logging). V(π) = **V_exact** (observed) + **V_model** (extrapolated increment where the policy bids above the logged flat bid). Max modeled share = **0.7%**.

- **P2 (recover observable):** logged-policy value 4.554e+08 == realized 2nd-price surplus on valid wins 4.554e+08 → match=True. Data quirk: **69,568 (1.2%) won rows have payprice>bidprice** (violates second-price) and are correctly excluded. (Separately: Stage-B2 ran *first-price* on second-price data — corrected here.)

- **P1 (market model):** F(b|x) within Wilson CI on **13%** of logged (segment×bid) cells (≤300 only; says nothing about b>300).


## Truthful (second-price) full-inventory value, by model

| model | V(π) total | V_exact | V_model | modeled % | rows extrapolated % | mean bid |
|---|---|---|---|---|---|---|
| neural | 4.390e+08 | 4.369e+08 | 2.086e+06 | 0.5% | 3.695% | 156.25 |
| lr | 4.170e+08 | 4.139e+08 | 3.079e+06 | 0.7% | 1.601% | 142.04 |
| lgb | 4.293e+08 | 4.278e+08 | 1.516e+06 | 0.4% | 3.087% | 132.39 |

## Full-inventory value gap (neural − baseline)

| gap | point | paired CI95 | p>0 | cluster CI95 | p>0 |
|---|---|---|---|---|---|
| neural_minus_lr | 2.203e+07 | [12378183.5, 31754360.1] | 1.0 | [5474837.2, 39818970.1] | 0.9997 |
| neural_minus_lgb | 9.687e+06 | [-689951.4, 19708424.6] | 0.966 | [-16364172.9, 46797983.0] | 0.659 |

*Most of V(π) is EXACT second-price surplus (observed market prices on re-won inventory); the modeled increment is the censored region a value-driven policy wins by bidding above the logged flat bid. Won-only barely binds for truthful bids ≈ V < logged 227–300.*


## Files
- `results/stage_a/policy_value.json`
