# Stage 8 — neural-vs-LGB heterogeneity / power analysis (truthful 2p-optimal)

## TL;DR VERDICT

- **neural vs LGB (strong GBM): NOT robust — advertiser-heterogeneous: positive on 2/5 (CI-significant on only 1/5), heterogeneity CONFIRMED (I²=0.819, Q p=0.0002); cluster-mean CI [-11055312.0, 40701557.0] contains 0; MDE 1.15e+07 vs observed mean 1.88e+06; leave-one-out shows single-advertiser leverage.**

- **neural vs LR (linear): robust (cluster CI [17721422.0, 37751797.0], positive on 5/5)**

- Finer clustering: forbidden (ICC>0).

- Framing: 5 advertisers = entire fair shared-vocab split (population). Cluster-mean CI = sanity bound, not headline. 9-advertiser eval = dead end (disjoint-advertiser artifact).


## neural - lgb: per-advertiser (R1) + mechanism

| advertiser | gap | CI95 | excl 0 | clicks | neural resid IEB | LGB/LR resid IEB |
|---|---|---|---|---|---|---|
| 3427 | 1.393e+07 | [7630597.4, 20440034.0] | True | 1410 | 0.0851 | 0.1253 |
| 3386 | 2.202e+05 | [-4992726.8, 5437442.6] | False | 1012 | 0.1553 | 0.0031 |
| 3476 | -2.364e+05 | [-3078603.4, 2568618.4] | False | 506 | 0.2262 | 0.2715 |
| 1458 | -8.587e+05 | [-4252250.7, 2564036.3] | False | 1172 | 0.0073 | 0.0665 |
| 3358 | -3.680e+06 | [-7460973.9, 152262.3] | False | 412 | 0.2085 | 0.0861 |

- **R2 sign test:** 2/5 positive, p=0.8125 (n=5 ⇒ near-zero power; footnote only).
- **R3 heterogeneity:** Q=22.04 (df=4, p=0.0002), I²=0.819, τ²=1.85e+13 ⇒ **HETEROGENEOUS**.
- **R4 MDE:** mean 1.876e+06, SD_between 6.908e+06, MDE(80%) 1.148e+07 ⇒ observed AT/BELOW MDE; ~106.5 clusters needed (homog. approx).
- **R6 leave-one-advertiser-out means:** [2559087.8, 3264352.5, 2289357.5, -1138630.0, 2403509.2] (single-advertiser leverage).
- **cluster-mean sanity:** 9.378e+06, CI [-11055312.0, 40701557.0], excludes 0 = False.

## neural - lr: per-advertiser (R1) + mechanism

| advertiser | gap | CI95 | excl 0 | clicks | neural resid IEB | LGB/LR resid IEB |
|---|---|---|---|---|---|---|
| 1458 | 8.217e+06 | [4258929.2, 12344511.6] | True | 1172 | 0.0073 | 0.0999 |
| 3386 | 7.860e+06 | [2849813.2, 12844578.9] | True | 1012 | 0.1553 | 0.0217 |
| 3427 | 5.597e+06 | [-133901.6, 11339121.4] | False | 1410 | 0.0851 | 0.0414 |
| 3358 | 3.296e+06 | [-256386.7, 6880454.8] | False | 412 | 0.2085 | 0.1528 |
| 3476 | 2.465e+06 | [-196471.9, 5226891.4] | False | 506 | 0.2262 | 0.2836 |

- **R2 sign test:** 5/5 positive, p=0.03125 (n=5 ⇒ near-zero power; footnote only).
- **R3 heterogeneity:** Q=7.72 (df=4, p=0.1024), I²=0.482, τ²=3.77e+12 ⇒ **cannot reject homogeneity (Q underpowered)**.
- **R4 MDE:** mean 5.487e+06, SD_between 2.599e+06, MDE(80%) 4.322e+06 ⇒ observed above MDE; ~1.8 clusters needed (homog. approx).
- **R6 leave-one-advertiser-out means:** [4804615.8, 6034991.2, 4893972.2, 5459603.5, 6242600.2] (single-advertiser leverage).
- **cluster-mean sanity:** 2.744e+07, CI [17721422.0, 37751797.0], excludes 0 = True.

## Files
- `results/stage_a/power_analysis.json`
