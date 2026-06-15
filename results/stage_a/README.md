# Stage A / B2 artifacts — index

Falsification-first arc: **diagnose on the original (unfair) split → fix on the fair split → prove
decision value**. Read the consolidated narrative in `docs/redesign_findings.md` and the frozen
evaluation contract in `docs/evaluation_protocol.md` first; this index just maps the raw artifacts.

> **Split label matters.** Early diagnosis artifacts use the **original/unfair** disjoint-advertiser
> split; the fix-and-prove artifacts use the **fair** per-advertiser-temporal split
> (`data/ipinyou/prediction/features_fair/`). Don't compare numbers across split labels.

## Narrative (markdown reports)

| order | file | date | split | one-line verdict |
|---|---|---|---|---|
| 1 | `reliability_summary.md` | 06-14 | original/unfair | global IEB = mean-cancellation; winners object grossly miscalibrated (10/10 deciles) |
| 2 | `phase1_findings.md` | 06-14 | original/unfair | verification + de-facto eval spec; falsified `surplus(V)` and global IEB as headlines |
| 3 | `rootcause_audit.md` | 06-14 | original/unfair | "debiasing loses" = **artifact** (adversarial split + unsupervised CTR tower) |
| 4 | `recalibration_summary.md` | 06-15 | **fair** | cross-fit isotonic: winners IEB 0.597→0 (all 3 models), AUC preserved; per-advertiser residual remains |
| 5 | `stage_b2_surplus_summary.md` | 06-15→06-16 | **fair** | decision test, **SECOND-PRICE (corrected from first-price bug)**: thesis PARTIALLY supported — neural>LR significant, neural>LGB point-only (2p-optimal truthful CI contains 0); strategy rank reversed (truthful=2p-optimal). `mechanism_comparison` retains 1p |
| 6 | `stage4_calibration_summary.md` | 06-15 | **fair** | training-stage calibration **NEGATIVE**: no train-time knob calibrates without breaking ranking; post-hoc isotonic wins (CI-confirmed) |
| 7 | `segment_calibration_summary.md` | 06-15 | **fair** | per-advertiser calibration **SOLVES** the residual (0.226→0.0006); bidding payoff small-positive (sig. for LGB), advertiser-concentrated |
| 8 | `policy_value_summary.md` | 06-16 | **fair** | full-inventory **second-price** policy value (escape won-only): ≥99.3% EXACT; found stage_b2 first-price bug; P1 NO-GO (F(b\|x) unidentifiable from flat bids) |
| 9 | `power_analysis_summary.md` | 06-16 | **fair** | neural-vs-LGB **heterogeneity/power resolution**: neural>LGB **NOT robust** (heterogeneous, I²=0.82, 1/5 CI-sig, 3427-driven); neural>LR **robust** (5/5); ICC>0 forbids finer clustering |

## Machine-readable (JSON)

| file | split | holds |
|---|---|---|
| `fair_baselines.json` | fair | LR/LGB winners + all-bids AUC, split metadata, won advertisers |
| `fair_comparison.json` | fair | neural vs LR/LGB winners AUC + decile ratios |
| `recalibration.json` | fair | per-model isotonic before/after IEB, deciles, per-advertiser residual |
| `stage_b2_surplus.json` | fair | 6-cell × 3-strategy surplus grid, bootstrap CIs, sensitivity, verdict |
| `stage4_calibration.json` | fair | per-model native/frozen-val→test/cross-fit calibration + per-adv + native-vs-iso surplus CIs |
| `segment_calibration.json` | fair | per-model raw/global/segment per-advertiser residual + within/global AUC + segment-vs-global surplus CIs |
| `policy_value.json` | fair | full-inventory second-price V(π) per model (V_exact/V_model split), P0/P1/P2 probe results, neural−baseline cluster CIs |
| `power_analysis.json` | fair | neural−LGB/LR per-advertiser CIs, Cochran's Q/I²/τ², cluster-t MDE, ICC gate, leave-one-advertiser-out, heterogeneity verdict |
| `reliability.json` | original/unfair | quantile/slice calibration, decile tables, all 5 models |
| `surplus_corr.json` | original/unfair | surplus vs IEB/AUC correlations + scale variants |
| `easy_negatives.json` | original/unfair | all-bids vs winners-only AUC (easy-negative gap) |

## Predictions / logs

| file | split | holds |
|---|---|---|
| `recalibrated_winners_preds.npz` | fair | winners `idx_won`, `y_click_won`, per-model `_raw`/`_recal` pCTR (Stage B2 input) |
| `fair_baseline_preds.npz` | fair | LR/LGB test (+ now val) predictions, labels, advertiser |
| `test_predictions_all.npz` | original/unfair | legacy unified test predictions (diagnostic history only) |
| `fair_baselines_run.log`, `stage_b2_run.log` | — | run logs |
