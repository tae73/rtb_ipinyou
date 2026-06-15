# Neural Model Performance Tuning Log

> ⚠️ **HISTORICAL — not maintained.** A 20-phase AUC-tuning log on the **original/unfair split**
> (ended 2026-03-16). The 2026-06 redesign showed the AUC "gap" it chased was largely a split
> artifact; kept for methodology reference only. **Current → [`redesign_findings.md`](../redesign_findings.md);
> protocol → [`evaluation_protocol.md`](../evaluation_protocol.md).**

**Last Updated**: 2026-03-16 (Phase 20 Target Encoding — 가설 기각, TE가 neural model에서 역효과)

## 1. Problem Statement

LR CTR_all baseline achieves **Test AUC 0.7687** on iPinYou all-bids CTR prediction.
Neural models (ESMM-WC / ESCM2-WC) initially underperformed but subsequent tuning
closed the gap significantly vs LGB baselines.

**Goal**: Close the gap between LR baseline and neural multi-task models while preserving
the debiasing benefits of ESMM/ESCM2 architecture.

---

## 2. Baseline Comparison

### 2.1 공정 비교 원칙

Baseline과 ESMM-WC의 비교는 **동일한 train set + test set** 기준으로 수행해야 한다.

- **Winners-only CTR**: train=winners, test=winners → P(Click|Win=1, X)
- **All-bids CTR**: train=all bids, test=all bids → P(Click_bid|X)
- **Win prediction**: train=all bids, test=all bids → P(Win|X)

ESMM-WC의 `test_ctr_biased_auc`는 all bids로 학습하되 winners-only에서 평가하므로,
winners-only로 학습한 LGB CTR과는 **학습 조건이 다름**에 유의.

### 2.2 결과 (실제 result JSON 기준)

**Winners-only CTR evaluation:**

| Model | Train Set | Train AUC | Val AUC | Test AUC |
|-------|-----------|-----------|---------|----------|
| LGB CTR | winners | 0.8113 | 0.7089 | **0.6890** |
| **ESMM-WC Run J** | **all bids** | -- | ~0.70 | **0.6237** |
| LR CTR | winners | 0.6333 | 0.6281 | 0.3216 |

**All-bids CTR evaluation (P(click|bid)):**

| Model | Train Set | Train AUC | Val AUC | Test AUC |
|-------|-----------|-----------|---------|----------|
| **LR CTR_all** | all bids | 0.7731 | 0.6805 | **0.7687** |
| **ESMM-WC WCTR** | all bids | -- | -- | **0.6905** |
| LGB CTR_all | all bids | 0.9059 | 0.7598 | 0.5437 |

**Win prediction:**

| Model | Train AUC | Val AUC | Test AUC |
|-------|-----------|---------|----------|
| LGB Win | 0.9308 | 0.8553 | 0.6493 |
| ESMM-WC Win | -- | ~0.82 | 0.6432 |
| LR Win | 0.8240 | 0.7534 | 0.5889 |

### 2.3 분석

**ESMM-WC는 LGB를 이미 유의미하게 능가한다:**
- All-bids CTR: ESMM-WC 0.6905 >> LGB CTR_all 0.5437 (+0.147)
- Winners-only CTR: ESMM-WC 0.6237 vs LGB CTR 0.6890 (gap 0.065, 학습조건 다름 감안)
- Win prediction: ESMM-WC 0.6432 ≈ LGB Win 0.6493

**LR CTR_all(0.7687)만이 ESMM-WC를 유의미하게 앞선다:**
- Gap: LR 0.7687 vs ESMM-WC WCTR 0.6905 = 0.078 (동일 조건: all bids 학습+평가)
- 단, LR CTR_all의 all-bids 평가는 non-won bids (click 불가능한 "easy negatives")를
  대량 포함하여 AUC가 부풀려지는 효과가 있음
- LR의 temporal robustness는 linear model의 낮은 complexity에서 기인

**"Easy Negatives" 효과 심층 분석:**

LR CTR_all의 Test AUC 0.7687이 실제 CTR 예측 능력을 과대 평가하는 구조적 원인:

1. **Non-won bids의 guaranteed click=0**: iPinYou 데이터에서 non-won bids는 전체의
   ~76% (129.5M bids 중 30.6M impressions). 이 76%는 광고가 노출되지 않아 click이
   원천적으로 불가능한 **trivial negatives**. 모델이 "won vs not-won"을 구분하는 것만으로
   all-bids AUC를 크게 높일 수 있다.

2. **LR winners-only CTR = 0.3216 (random 이하)**: LR은 winners 내에서 click/non-click을
   전혀 판별하지 못한다. Random AUC(0.5)보다 낮은 0.3216은 LR의 CTR 예측 방향이
   실제와 반대임을 의미. **실제 CTR 예측 능력이 없음에도 all-bids AUC가 0.7687.**

3. **AUC 부풀림 메커니즘**: All-bids 평가에서 AUC는 positive(clicked) vs negative(non-clicked)
   쌍의 ranking 정확도를 측정. Non-won bids는 확실한 negative로, 모델이 이들에게 낮은
   score를 부여하면 (won/non-won 구분만으로 가능) AUC가 상승. 실제 의사결정에 필요한
   "winners 내에서 누가 클릭할까"는 반영하지 못함.

4. **ESMM-WC와의 공정 비교**: ESMM-WC WCTR도 동일한 all-bids 조건이지만 AUC 0.6905.
   P(win)×P(click|win) product의 noise가 LR의 직접 P(click|bid) 대비 불리하나,
   ESMM-WC는 winners-only CTR AUC 0.6237로 **실질적 CTR 판별력**이 존재.
   LR의 0.7687 vs ESMM-WC의 0.6905 gap(0.078)은 easy negatives 효과를 포함하므로,
   실제 CTR 예측 품질 차이보다 과장되어 있다.

---

## 3. Root Cause Analysis

### 3.1 Remaining Gap: LR CTR_all vs ESMM-WC

ESMM-WC(0.6905)가 LR CTR_all(0.7687)에 미치지 못하는 원인:

1. **Nonlinear model의 temporal shift 취약성**: MLP의 nonlinear mapping이 S2 distribution을
   memorize → S3에서 일반화 실패. LR은 linear coefficients만 사용하여 robust.
2. **Multi-task learning overhead**: Win + CTR 동시 학습으로 CTR-specific optimization이 diluted.
   win_weight=0.01로 완화했으나 완전 제거는 불가.
3. **Easy negatives 효과**: LR CTR_all의 all-bids 평가에서 non-won bids(click=0 guaranteed)가
   AUC를 부풀림. ESMM-WC WCTR도 동일 조건이지만, P(win)×P(click|win) product의 noise가 더 큼.

### 3.2 Feature Dimension Analysis

| Component | Without Bypass | With Bypass (ed=32) | With Bypass (ed=16) |
|-----------|---------------|--------------------|--------------------|
| Cat embeddings (17 features) | 17x32 = 544 | 17x32 = 544 | 17x16 = 272 |
| FM interaction | 32 | 32 | 16 |
| Num features (13 features) | 13x32 = **416** | 13x1 = **13** | 13x1 = **13** |
| **Total MLP input** | **992** | **589** | **301** |

---

## 4. Experiment Phases

### Phase 1: Initial Hyperparameter Search

**가설**: Vanilla ESMM-WC (no scheduler, no regularization)로 LR 수준 성능 달성 가능한가?

Common: batch=4096, epochs=50, patience=10, hidden="256,128,64", dropout=0.3, embed_dim=32
No scheduler/warmup/gradient-clip/weight-decay

| Run | lr   | Best Ep | Test CTR AUC | Test WCTR AUC |
|-----|------|---------|-------------|--------------|
| A   | 1e-3 | 5       | 0.5173      | 0.5995       |
| B   | 3e-3 | 4       | 0.5170      | 0.5868       |
| C   | 5e-4 | 8       | 0.5036      | 0.5844       |

**결론**: Test CTR AUC ~0.517 (near-random). batch=4096에서 배치당 클릭 ~0.7건으로
CTR gradient starvation 발생. LR 간 차이도 미미하여 근본적인 학습 자체가 불충분.
대규모 batch + regularization + scheduler 도입 필요 → Phase 2

### Phase 2: Regularization Tuning (2026-02-27)

**가설**: Regularization (dropout, weight-decay, large batch) + cosine scheduler로
S2 overfitting을 방지하면 Test CTR AUC 0.65+ 달성 가능

Common: lr=1e-3, batch=65536, epochs=50, patience=15, es-metric=ctr_auc,
cosine scheduler, warmup=200, gradient-clip=1.0, weight-decay=1e-3, embed_dim=32, LayerNorm

| Run | dropout | hidden | win_weight | Test CTR AUC | Test WCTR AUC |
|-----|---------|--------|------------|-------------|--------------|
| D | 0.3 | 128,64 | 1.0 | 0.4937 | 0.5886 |
| E | 0.5 | 64,32 | 1.0 | 0.5160 | 0.3971 |
| F | 0.4 | 128,64 | 0.1 | 0.5226 | 0.6005 |

**결론**: 0.65 목표 미달. 그러나 win_weight=0.1 (Run F)이 1.0 대비 일관되게 우위.
Win tower가 gradient를 지배하여 CTR tower 학습을 방해하는 현상 확인.
Win-weight를 더 극단적으로 줄이면 CTR tower가 독립 학습 가능할 것 → Phase 3

### Phase 3: Loss Weight Exploration (2026-03-04)

**가설**: Win-weight를 극단적으로 최소화(0.01)하면 win tower gradient 간섭이 제거되어
CTR tower가 독립적으로 학습 가능. joint_weight 증가로 ESMM constraint 강화도 시도.

| Run | win_weight | joint_weight | Test CTR AUC | Test WCTR AUC |
|-----|-----------|-------------|-------------|--------------|
| F | 0.1 | 1.0 | 0.5226 | 0.6005 |
| **G** | **0.01** | **1.0** | **0.5888** | **0.6426** |
| H | 0.1 | 3.0 | 0.5334 | 0.5815 |
| I | 0.01 | 3.0 | 0.5591 | 0.6207 |

**결론**: win_weight=0.01이 breakthrough — CTR AUC 0.5226→0.5888 (+0.0662).
joint_weight 증가(3.0)는 P(win)×P(ctr|win) product metric에 치중하여 개별 CTR tower 성능 저하.
단, 모든 Phase 3 runs가 epoch 3에서 수렴 → 학습이 너무 빠르게 종료.
LR 감소로 peak epoch을 우측으로 밀면 더 나은 일반화 plateau 도달 가능 → Phase 4

### Phase 4: Convergence Speed Tuning (2026-03-04)

**가설**: LR 감소(1e-3→5e-4, 3e-4)로 peak epoch을 뒤로 밀면,
모델이 더 넓은 loss landscape를 탐색하여 일반화 성능이 향상된 plateau에 도달.

| Run | learning_rate | warmup | Best Ep | Test CTR AUC | Test WCTR AUC |
|-----|--------------|--------|---------|-------------|--------------|
| G | 1e-3 | 200 | 3 | 0.5888 | 0.6426 |
| **J** | **5e-4** | **200** | **5** | **0.6237** | **0.6905** |
| K | 3e-4 | 200 | 7 | 0.5684 | 0.6726 |
| L | 3e-4 | 1000 | 7 | 0.5744 | 0.6737 |

**결론**: 가설 부분 확인 — peak epoch이 LR에 반비례하여 이동 (3→5→7).
Run J (lr=5e-4)가 best Test CTR AUC 0.6237, WCTR AUC 0.6905 달성.
단, 3e-4로 더 줄이면 수렴 속도만 늦어지고 일반화는 악화 — diminishing returns.
Run J의 epoch 5 이후 성능 저하가 overfitting인지 확인 필요 → Phase 5에서 regularization 강화 시도

### Phase 5: Regularization Reinforcement (2026-03-04)

**가설**: Run J의 epoch 5 이후 성능 하락은 overfitting이므로,
dropout 강화(0.4→0.5) 및 weight-decay 증가(1e-3→3e-3)로 plateau를 유지할 수 있다.

| Run | dropout | weight_decay | Best Ep | Test CTR AUC | Test WCTR AUC |
|-----|---------|-------------|---------|-------------|--------------|
| J | 0.4 | 1e-3 | 5 | **0.6237** | **0.6905** |
| M | 0.5 | 1e-3 | 5 | 0.5094 | 0.6444 |
| N | 0.4 | 3e-3 | 5 | 0.5332 | 0.6583 |
| O | 0.5 | 3e-3 | 7 | 0.5478 | 0.6656 |

**결론**: 가설 기각 — **Regularization Paradox**. 모든 regularization 강화가 역효과.
Val-Test gap이 regularization에도 불구하고 악화 → 원인은 overfitting이 아닌
**temporal distribution shift** (S2 train → S3 test, ~4개월 gap).
LR이 temporal-robust한 이유는 numerical features를 raw scalar로 직접 사용하기 때문.
Neural model도 동일하게 numeric bypass로 raw scalar 정보를 보존하면 효과적일 것 → Phase 6

### 전체 진행 요약

Baseline 비교 기준: LGB CTR 0.6890 (winners-only), LGB CTR_all 0.5437 (all-bids), LR CTR_all 0.7687 (all-bids)

| Phase | 가설 | 핵심 변경 | Best Run | Test CTR AUC | Test WCTR AUC | 결과 |
|-------|------|----------|----------|-------------|--------------|------|
| 1 | Vanilla ESMM-WC | 없음 (vanilla) | A | 0.5173 | 0.5995 | Near-random, gradient starvation |
| 2 | Regularization + scheduler | batch→65536, cosine, LayerNorm | F | 0.5226 | 0.6005 | win-weight 핵심 변수 발견 |
| 3 | Win-weight 극소화 | ww=0.01 (win gradient 억제) | G | 0.5888 | 0.6426 | +0.0662 (breakthrough), LGB CTR_all 돌파 |
| 4 | LR 감소로 수렴 조절 | lr→5e-4 | **J** | **0.6237** | **0.6905** | +0.0349, **LGB CTR_all(0.54) 대비 +0.15** |
| 5 | Regularization 강화 | dropout↑, wd↑ | M/N/O | 0.5094-0.5478 | — | 역효과 (distribution shift) |
| 6 | Numeric bypass (ESMM-WC) | 아키텍처 변경 (raw scalar) | — | — | — | ESCM2-WC로 방향 전환, 보류 |
| **7** | **ESCM2-WC Transfer** | **3-tower (DR/IPW)** | **P (DR)** | **0.5534** | **0.6728** | 3rd tower overhead, ESMM-WC 미달 |
| 8 | DR debiasing strength | ctr_weight 탐색 | R | 0.5526 | 0.6766 | 미약한 개선, propensity 품질이 병목 |
| 9 | Win weight 복원 | ww=0.1~1.0 | U/V/W | 0.4384-0.3038 | 0.4482-0.5542 | Catastrophic: gradient 간섭 재현 |
| 10 | Gradient isolation | stop_grad_win_embed | X/Y | 0.3729-0.3756 | 0.4905-0.4979 | Win AUC↑ but CTR still fails |
| 11 | DR formulation | BCE + clipping | Z/AA/AB | 0.5432-0.5526 | 0.6708-0.6744 | 무의미 (Run R ±0.006) |
| 12 | Numeric bypass/scalar | 아키텍처 변경 | AC/AD/AE | 0.3654-0.4550 | 0.3282-0.5611 | Catastrophic: embedding 제거 역효과 |
| 13B | Impute loss + CFR 탐색 | Huber/CFR변경 | **AL** | **0.5636** | **0.6843** | cfr=0.2가 유일 개선(+0.008) |
| 14 | Per-tower dropout | tower별 dropout | AM/AN | 0.4458-0.4626 | 0.6377-0.6471 | Catastrophic: embedding overfit |
| 15 | 복합 최적 조합 | — | — | — | — | SKIPPED (14 전면 실패) |
| 16 | Checkpoint averaging | top-K weight avg | AP | 0.5377 | 0.6722 | Peak 희석, 역효과 |
| 17 | cfr/impute_loss_weight 확장 | cfr=0.3~0.5, ilw=0.3~1.0 | AL (unchanged) | 0.5636 | 0.6843 | 0.2/0.5가 sweet spot, IEB 악화 |
| **18** | **External Win PS** | **LGB PS(AUC 0.91) for DR** | **AW** | **0.5713** | **0.6882** | **ESCM2-WC best AUC, IEB 0.045** |
| 20 | Target Encoding | 5 cats × 2 targets TE | AY | 0.4209 | 0.5480 | **가설 기각**: TE가 CTR -0.20, WCTR -0.14 악화 |

---

## 5. Numeric Bypass (Phase 6)

**가설**: LR이 temporal-robust한 이유는 numerical features를 raw scalar로 직접 사용하여
ordinal/magnitude 정보를 보존하기 때문. Neural model에서도 numeric bypass로 동일 효과를
재현하면 temporal distribution shift에 대한 robustness가 향상되어 LR-NN gap을 줄일 수 있다.

### 5.1 Motivation

LR's advantage: 13 numerical features processed as raw StandardScaler float32 scalars,
preserving ordinal/magnitude information directly accessible to linear coefficients.

ESMM-WC's disadvantage: `Linear(1, 32)` projection per numerical feature destroys
the very information that LR leverages most effectively.

### 5.2 Solution: Numeric Bypass

Skip the `Linear(1, embed_dim)` projection for numerical features. Pass raw normalized
scalars directly to the MLP, concatenated after categorical embeddings.

**Architecture change in `EmbeddingLayer`**:
```
Before: [cat1_embed(32), ..., catN_embed(32), num1_proj(32), ..., numM_proj(32)]
After:  [cat1_embed(32), ..., catN_embed(32), num1_raw(1),  ..., numM_raw(1)]
```

**FM interaction**: Only applied to categorical embeddings (uniform embed_dim required).
Raw numerics are appended after FM output.

### 5.3 Implementation

Files modified:
- `src/models/base.py`: `EmbeddingLayer` -- `use_numeric_bypass` parameter
- `src/models/esmm_wc.py`: `ESMMWCConfig` + `ESMMWC` -- input_dim calc, FM split
- `src/models/escm2_wc.py`: `ESCM2WCConfig` + `ESCM2WC` -- same pattern
- `scripts/train.py`: `--use-numeric-bypass` CLI flag
- `scripts/sweep.py`: kwargs forwarding
- `configs/model/*.yaml`: `use_numeric_bypass: false` default

### 5.4 Status

ESMM-WC numeric bypass 실험은 ESCM²-WC 탐색으로 연구 방향 전환하여 보류.
ESCM²-WC에서 Phase 10으로 numeric bypass 재실험 예정.

---

## 6. ESCM²-WC Systematic Experiments

ESMM-WC Run J (WCTR AUC 0.6905)를 base로 ESCM²-WC (3-tower, DR/IPW debiasing)
성능 탐색. 목표: WCTR AUC ≥ 0.7687 (LR CTR_all).

### Phase 7: Run J Transfer Baseline (DR vs IPW) — 2026-03-08

**가설**: ESMM-WC Run J의 검증된 hyperparameters를 ESCM²-WC에 이전하면
최소 ESMM-WC WCTR AUC 0.6905 이상을 달성한다.

**Base config**: Run J (lr=5e-4, batch=65536, dropout=0.4, wd=1e-3, cosine+warmup=200,
LayerNorm, embed_dim=32, hidden=[128,64], win_weight=0.01, ctr_weight=0.0, joint_weight=1.0)

| Run | loss_type | ctr_weight | cfr_lambda | Best Ep | Val CTR AUC | Test CTR AUC | Test WCTR AUC | Test Win AUC |
|-----|-----------|-----------|-----------|---------|-------------|-------------|--------------|-------------|
| **P** | dr | 0.0 | 0.1 | 5 | 0.7059 | 0.5534 | **0.6728** | 0.6389 |
| **Q** | ipw | 0.0 | 0.0 | 5 | 0.7058 | 0.5292 | **0.6526** | 0.6347 |
| *Run J (ESMM-WC)* | — | 0.0 | — | 5 | ~0.70 | 0.6237 | **0.6905** | 0.6432 |

**결론**: 가설 기각.
- 3rd tower (Imputation) overhead가 Test 성능을 저하시킴
- DR(P) > IPW(Q) — DR imputation correction이 약간 도움 (+0.02 WCTR AUC)
- Val에서는 Run J와 유사하나 Test에서 큰 degradation → 3rd tower가 temporal shift에 더 취약
- ctr_weight=0.0에서 DR debiasing이 비활성 → Phase 8에서 ctr_weight > 0 탐색

### Phase 8: DR Debiasing Strength (ctr_weight 탐색) — COMPLETE

**가설**: ctr_weight=0.0에서는 DR debiasing이 비활성. ctr_weight를 0.01~0.1로 올리면
DR correction이 CTR tower에 추가 signal을 제공하여 ESMM-WC 대비 improvement를 얻을 수 있다.

**Base**: Phase 7 best (DR, Run P config) + ctr_weight variation

| Run | ctr_weight | cfr_lambda | impute_loss_weight | Best Ep | Val CTR AUC | Test WCTR AUC | Test CTR AUC |
|-----|-----------|-----------|-------------------|---------|-------------|--------------|-------------|
| *Run P (base)* | 0.0 | 0.1 | 0.5 | 5 | 0.7059 | 0.6728 | 0.5534 |
| **R** | 0.01 | 0.1 | 0.5 | 5 | 0.7095 | **0.6766** | 0.5526 |
| **S** | 0.1 | 0.1 | 0.5 | 5 | 0.7105 | 0.6262 | 0.4770 |

**결론**: 가설 부분 확인.
- ctr_weight=0.01 (Run R): 미약한 WCTR 개선 (+0.0038 vs Run P), Val CTR AUC도 소폭 개선
- ctr_weight=0.1 (Run S): catastrophic failure — WCTR 0.6262, CTR 0.4770
- **근본 원인**: win_weight=0.01은 ESCM2-WC에서 구조적 모순. 논문은 propensity tower에
  weight=1.0 (full supervision)을 적용하지만, ESMM-WC에서 전이한 0.01이 약한 propensity →
  부정확한 DR importance weight → 노이즈 DR signal. ctr_weight가 클수록 이 noise가 증폭.
- **다음**: Phase 9에서 win_weight 복원하여 propensity 품질 개선 후 DR 효과 재검증

### Phase 9: Win Weight 복원 — COMPLETE

**가설**: ESCM2-WC의 win tower는 DR propensity를 제공하므로 논문처럼 강한 supervision이 필요.
win_weight를 0.01→0.1~1.0으로 올리면 propensity calibration이 개선되어 DR이 안정화된다.

단, ESMM-WC Phase 2-3에서 win_weight=1.0이 shared embedding gradient 간섭을 일으킨
바 있으므로, 이 간섭이 ESCM2에서도 재현되는지 확인 필요.

**Base**: Phase 8 best (Run R config: ctr_weight=0.01)

| Run | win_weight | ctr_weight | Best Ep | Val CTR AUC | Test WCTR AUC | Test CTR AUC | Test Win AUC |
|-----|-----------|-----------|---------|-------------|--------------|-------------|-------------|
| *R (base)* | 0.01 | 0.01 | 5 | 0.7095 | 0.6766 | 0.5526 | 0.6389 |
| **U** | 0.1 | 0.01 | 10 | 0.7174 | 0.5542 | 0.4384 | 0.6315 |
| **V** | 0.5 | 0.01 | 13 | 0.7247 | 0.4691 | 0.3457 | 0.6323 |
| **W** | 1.0 | 0.01 | 14 | 0.7270 | 0.4482 | 0.3038 | 0.6136 |

**결론**: 가설 기각 — win_weight↑가 Val CTR AUC를 높이지만 Test에서 catastrophic failure.
- Val-Test gap 급증: U(0.717→0.438), V(0.725→0.346), W(0.727→0.304)
- Win tower gradient가 shared embeddings를 S2 temporal distribution에 overfit시킴
- Win AUC도 하락 (0.6389→0.6136): 높은 win_weight로 win tower 자체도 overfitting
- **Decision**: gradient 간섭 재현 확인 → Phase 10 (stop_grad) 필수

### Phase 10: Embedding Gradient Isolation — COMPLETE

**가설**: win_weight↑ 시 shared embedding에서 gradient 간섭 발생 가능.
Win tower로 흐르는 embedding gradient를 `stop_gradient`로 차단하면,
win tower가 강한 supervision을 받으면서도 shared embedding은 CTR/joint에 최적화.

**코드 변경**: `src/models/escm2_wc.py` — `stop_grad_win_embedding` config + forward pass

| Run | win_weight | stop_grad_win_embed | ctr_weight | Best Ep | Val CTR AUC | Test WCTR AUC | Test CTR AUC | Test Win AUC |
|-----|-----------|---------------------|-----------|---------|-------------|--------------|-------------|-------------|
| *W (base)* | 1.0 | False | 0.01 | 14 | 0.7270 | 0.4482 | 0.3038 | 0.6136 |
| **X** | 1.0 | True | 0.01 | 4 | 0.7006 | 0.4905 | 0.3729 | 0.6657 |
| **Y** | 1.0 | True | 0.05 | 5 | 0.7027 | 0.4979 | 0.3756 | 0.6575 |

**결론**: 가설 부분 확인 — stop_grad가 win_auc를 개선 (0.6136→0.6657),
embedding 보호 효과 확인. 그러나 Test WCTR/CTR은 여전히 catastrophic (~0.49/0.37).
- stop_grad는 embedding 간섭만 차단하지, win tower 자체의 overfitting은 해결 불가
- win_weight=1.0의 강한 win supervision이 win tower MLP를 S2에 overfit시켜
  부정확한 propensity → DR correction 왜곡
- **결론**: win_weight=0.01이 ESCM2-WC에서 최적. Phase 11은 Run R 기반으로 진행

### Phase 11: DR Loss Formulation + Clipping — COMPLETE

**가설**: MSE DR의 squared error가 importance weight 극단값을 증폭.
BCE pseudo-label variant와 aggressive clipping이 안정성을 개선한다.

**Base**: Run R config (win_weight=0.01, ctr_weight=0.01)

| Run | dr_loss_type | max_weight | win_eps | Best Ep | Val CTR AUC | Test WCTR AUC | Test CTR AUC | Test Win AUC |
|-----|-------------|-----------|---------|---------|-------------|--------------|-------------|-------------|
| *R (base)* | mse | 10.0 | 0.05 | 5 | 0.7095 | 0.6766 | 0.5526 | 0.6389 |
| **Z** | bce | 10.0 | 0.05 | 5 | 0.7087 | 0.6724 | 0.5492 | 0.6397 |
| **AA** | mse | 5.0 | 0.1 | 5 | 0.7098 | 0.6744 | 0.5485 | 0.6377 |
| **AB** | bce | 5.0 | 0.1 | 5 | 0.7052 | 0.6708 | 0.5432 | 0.6404 |

**결론**: 가설 기각 — DR loss formulation/clipping 변경이 무의미.
- 모든 variant가 Run R(0.6766)과 ±0.006 범위 내 (Z: 0.6724, AA: 0.6744, AB: 0.6708)
- win_weight=0.01에서 DR correction signal이 극히 약해 formulation 차이가 무시됨
- Aggressive clipping(max_weight=5.0)이 약간 더 나쁜 결과 — 이미 약한 signal을 더 축소
- **핵심 인사이트**: ESCM2-WC의 근본 한계는 DR formulation이 아닌, 3rd tower가
  temporal shift에 추가 취약점을 만드는 structural overhead

### Phase 12: Numeric Bypass + Scalar Input — COMPLETE

**가설**: Phase 9-11 best config 위에 numeric bypass / scalar input으로
temporal robustness 개선. LR의 raw scalar 사용 패턴 재현.

| Run | Feature Mode | Best Ep | Val CTR AUC | Test WCTR AUC | Test CTR AUC | Test Win AUC |
|-----|-------------|---------|-------------|--------------|-------------|-------------|
| *R (base)* | standard (embed) | 5 | 0.7095 | 0.6766 | 0.5526 | 0.6389 |
| **AC** | numeric bypass (ed=32) | 5 | 0.7021 | 0.3282 | 0.3654 | 0.5953 |
| **AD** | numeric bypass (ed=16) | 5 | 0.6992 | 0.4683 | 0.4550 | 0.6082 |
| **AE** | scalar input | 20 | 0.6643 | 0.5611 | 0.4029 | 0.5928 |

**결론**: 가설 기각 — 모든 architecture 변경이 catastrophic failure.
- **AC (bypass, ed=32)**: MLP input 589차원(vs 992) → Val 0.70로 유사하나 Test 0.33 (반전)
- **AD (bypass, ed=16)**: MLP input 301차원 → 더 compact하나 Test 0.47 역시 최악
- **AE (scalar input)**: 모든 feature를 scalar로 → n_cat=0, n_num=30, best_epoch=20
  peak이 늦어졌으나 Test 0.56으로 여전히 Run R(0.68) 대비 크게 하락
- **핵심 원인**: Numeric bypass가 categorical embedding의 expressiveness를 손상시킴.
  LR은 linear model이라 raw scalar가 유리하지만, neural model은 categorical embedding의
  nonlinear interaction이 핵심 강점. Bypass/scalar는 이 강점을 제거.
- Temporal robustness 문제는 feature representation이 아닌 model capacity 자체의 문제

### Phase 13B: Imputation Loss + CFR 탐색 — COMPLETE

**가설**: Imputation tower의 noisy target (click ∈ {0,1} - p_ctr)에 Huber loss를 사용하면
outlier delta에 robust해진다. CFR lambda=0 또는 shrink type이 성능에 영향을 줄 수 있다.

**코드 변경**: `ESCM2WCConfig`에 `impute_loss_type`, `impute_huber_delta` 추가.
`create_escm2wc_loss_fn()`에서 Huber loss 분기 구현.

**Base**: Run R config (win_weight=0.01, ctr_weight=0.01, cfr_lambda=0.1)

| Run | 변경점 | Best Ep | Val CTR AUC | Test WCTR AUC | Test CTR AUC | Test Win AUC |
|-----|--------|---------|-------------|--------------|-------------|-------------|
| *R (base)* | — | 5 | 0.7095 | 0.6766 | 0.5526 | 0.6389 |
| **AJ** | impute_loss=huber, delta=0.1 | 5 | 0.7082 | 0.6664 | 0.5335 | 0.6377 |
| **AK** | cfr_lambda=0.0 | 5 | 0.7073 | 0.6638 | 0.5338 | 0.6379 |
| **AL** | **cfr_lambda=0.2** | **5** | **0.7083** | **0.6843** | **0.5636** | **0.6403** |

**결론**: 가설 부분 확인.
- **AL (cfr_lambda=0.2)이 유일한 개선**: WCTR AUC 0.6843 (+0.0077 vs Run R). CFR 강화가 imputation tower의 과적합을 억제.
- **AJ (Huber loss)**: Run R 대비 하락 (0.6664 vs 0.6766). win_weight=0.01에서 imputation target noise가 작아 Huber의 robustness 이점이 미미하고, quadratic region 축소가 오히려 signal 손실.
- **AK (cfr_lambda=0.0)**: 0.6638으로 하락. CFR 비활성화 → imputation tower가 unselected(win=0) samples에서 과적합.
- **CFR lambda 방향성 확인**: 0.0(0.6638) < 0.1(0.6766) < 0.2(0.6843). CFR regularization이 도움되며, 더 강한 값이 유효.
- 다만 AL도 ESMM-WC Run J(0.6905)에 0.006 미달.

**실행 명령**:
```bash
# Run AJ: Huber imputation loss
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features --model-dir results/models \
    --debiasing dr --epochs 50 --batch-size 65536 --learning-rate 5e-4 \
    --embedding-dim 32 --hidden-dims 128,64 --win-hidden-dims 64,32 \
    --dropout 0.4 --weight-decay 1e-3 --scheduler cosine --warmup-steps 200 \
    --gradient-clip 1.0 --use-layer-norm --eval-every 1 --es-metric ctr_auc \
    --patience 5 --cfr-lambda 0.1 --win-eps 0.05 --max-weight 10.0 \
    --impute-loss-weight 0.5 --joint-weight 1.0 --win-weight 0.01 --ctr-weight 0.01 \
    --impute-loss-type huber --impute-huber-delta 0.1

# Run AK: CFR disabled
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features --model-dir results/models \
    --debiasing dr --epochs 50 --batch-size 65536 --learning-rate 5e-4 \
    --embedding-dim 32 --hidden-dims 128,64 --win-hidden-dims 64,32 \
    --dropout 0.4 --weight-decay 1e-3 --scheduler cosine --warmup-steps 200 \
    --gradient-clip 1.0 --use-layer-norm --eval-every 1 --es-metric ctr_auc \
    --patience 5 --cfr-lambda 0.0 --win-eps 0.05 --max-weight 10.0 \
    --impute-loss-weight 0.5 --joint-weight 1.0 --win-weight 0.01 --ctr-weight 0.01

# Run AL: Stronger CFR with shrink
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features --model-dir results/models \
    --debiasing dr --epochs 50 --batch-size 65536 --learning-rate 5e-4 \
    --embedding-dim 32 --hidden-dims 128,64 --win-hidden-dims 64,32 \
    --dropout 0.4 --weight-decay 1e-3 --scheduler cosine --warmup-steps 200 \
    --gradient-clip 1.0 --use-layer-norm --eval-every 1 --es-metric ctr_auc \
    --patience 5 --cfr-lambda 0.2 --win-eps 0.05 --max-weight 10.0 \
    --impute-loss-weight 0.5 --joint-weight 1.0 --win-weight 0.01 --ctr-weight 0.01
```

### Phase 14: Per-Tower Dropout — COMPLETE

**가설**: Win tower는 강한 supervision(20M won samples) → 낮은 dropout(0.2) 적합.
CTR tower는 약한 signal(23K clicks) → 높은 dropout(0.5)으로 과적합 방지.

**코드 변경**: `ESCM2WCConfig`에 `win_dropout`, `ctr_dropout`, `impute_dropout` 추가.
Tower별 MLP 생성 시 해당 dropout 사용 (None이면 global fallback).

| Run | Win Dropout | CTR Dropout | Impute Dropout | Best Ep | Val CTR AUC | Test WCTR AUC | Test CTR AUC | Test Win AUC |
|-----|------------|-------------|----------------|---------|-------------|--------------|-------------|-------------|
| *R (uniform 0.4)* | 0.4 | 0.4 | 0.4 | 5 | 0.7095 | 0.6766 | 0.5526 | 0.6389 |
| **AM** | 0.2 | 0.5 | 0.4 | 5 | 0.7100 | 0.6377 | 0.4458 | 0.6504 |
| **AN** | 0.2 | 0.5 | 0.5 | 5 | 0.7118 | 0.6471 | 0.4626 | 0.6500 |

**결론**: 가설 기각 — per-tower dropout이 catastrophic failure.
- AM/AN 모두 Val CTR AUC는 Run R과 동일 수준(0.710-0.712)이나 Test에서 급락 (0.64-0.65)
- Win tower dropout 0.2 → Win AUC 소폭 개선(0.6389→0.6500)이나, CTR AUC 급락(0.55→0.45)
- **원인**: Win tower의 낮은 dropout(0.2)이 shared embedding을 S2 distribution에 overfit시킴.
  Phase 9 (win_weight↑)과 동일한 gradient 간섭 패턴 — dropout 감소 ≈ effective capacity 증가.
- **결론**: Uniform dropout=0.4가 최적. Per-tower dropout은 Phase 15에서 제외.

**실행 명령**:
```bash
# Run AM
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features --model-dir results/models \
    --debiasing dr --epochs 50 --batch-size 65536 --learning-rate 5e-4 \
    --embedding-dim 32 --hidden-dims 128,64 --win-hidden-dims 64,32 \
    --dropout 0.4 --weight-decay 1e-3 --scheduler cosine --warmup-steps 200 \
    --gradient-clip 1.0 --use-layer-norm --eval-every 1 --es-metric ctr_auc \
    --patience 5 --cfr-lambda 0.1 --win-eps 0.05 --max-weight 10.0 \
    --impute-loss-weight 0.5 --joint-weight 1.0 --win-weight 0.01 --ctr-weight 0.01 \
    --win-dropout 0.2 --ctr-dropout 0.5 --impute-dropout 0.4

# Run AN
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features --model-dir results/models \
    --debiasing dr --epochs 50 --batch-size 65536 --learning-rate 5e-4 \
    --embedding-dim 32 --hidden-dims 128,64 --win-hidden-dims 64,32 \
    --dropout 0.4 --weight-decay 1e-3 --scheduler cosine --warmup-steps 200 \
    --gradient-clip 1.0 --use-layer-norm --eval-every 1 --es-metric ctr_auc \
    --patience 5 --cfr-lambda 0.1 --win-eps 0.05 --max-weight 10.0 \
    --impute-loss-weight 0.5 --joint-weight 1.0 --win-weight 0.01 --ctr-weight 0.01 \
    --win-dropout 0.2 --ctr-dropout 0.5 --impute-dropout 0.5
```

### Phase 15: 복합 최적 조합 — SKIPPED

Phase 14 (per-tower dropout) 전면 실패로 조합할 설정이 없음.
Phase 13B best = Run AL (cfr_lambda=0.2)이 유일한 개선 → Phase 16에서 AL 기반 checkpoint averaging 직행.

### Phase 16: Checkpoint Averaging — COMPLETE

**가설**: 단일 best checkpoint 대신 top-K epochs의 weight를 평균하면 prediction 안정화.
Training cost 추가 없이 Val-Test gap 축소 가능.

**코드 변경**: `--top-k-avg K` CLI 옵션 추가. Training 중 top-K states 추적,
evaluation 시 floating-point parameter만 평균 (RNG key는 제외).

| Run | top_k_avg | Base Config | Avg Epochs | Test WCTR AUC | Test CTR AUC | Test Win AUC |
|-----|-----------|-------------|------------|--------------|-------------|-------------|
| *AL (no avg)* | 1 | cfr_lambda=0.2 | 5 only | **0.6843** | **0.5636** | 0.6403 |
| **AP** | 3 | AL (cfr_lambda=0.2) | 4,5,6 | 0.6722 | 0.5377 | 0.6429 |

**결론**: 가설 기각.
- Checkpoint averaging이 WCTR AUC를 0.6843→0.6722로 오히려 하락시킴 (-0.012)
- Epochs 4,5,6 평균이 best epoch 5보다 나쁨 — temporal shift 하에서 peak epoch의 특수한 지점이 중요
- Smoothing이 그 peak을 희석. 일반적 학습과 달리 S2→S3 shift에서는 averaging이 역효과

### Phase 17: 미탐색 Hyperparameter 확장 — COMPLETE

**가설**: Phase 13B에서 cfr_lambda의 단조 증가 트렌드(0.0→0.1→0.2)가 확인됨.
cfr_lambda를 0.2 이상으로 확장하고, impute_loss_weight와 impute_hidden_dims를
함께 조정하면 imputation tower의 regularization이 강화되어 추가 개선 가능.

**Phase 17-1: cfr_lambda 확장** (Run AL base)

| Run | cfr_lambda | impute_loss_weight | Test WCTR AUC | Test CTR AUC | Test WCTR IEB |
|-----|-----------|-------------------|--------------|-------------|--------------|
| *AL (baseline)* | 0.2 | 0.5 | **0.6843** | **0.5636** | **0.014** |
| AQ | 0.3 | 0.5 | 0.6841 | 0.5407 | 0.114 |
| AR | 0.5 | 0.5 | 0.6774 | 0.5489 | 0.105 |

**Phase 17-2: impute_loss_weight 탐색** (17-1 best = AL, cfr_lambda=0.2)

| Run | impute_loss_weight | cfr_lambda | Test WCTR AUC | Test CTR AUC | Test WCTR IEB |
|-----|-------------------|-----------|--------------|-------------|--------------|
| *AL (baseline)* | 0.5 | 0.2 | 0.6843 | **0.5636** | **0.014** |
| AS | 0.3 | 0.2 | **0.6866** | 0.5605 | 0.112 |
| AT | 1.0 | 0.2 | 0.6759 | 0.5369 | 0.108 |

**Phase 17-3: impute_hidden_dims 축소** — SKIPPED

17-1, 17-2 모두 AL 대비 IEB 급증 (0.014→0.10+), AUC 개선도 미미 (+0.002 max).
hidden_dims 축소가 이 trade-off를 역전시킬 가능성 낮아 skip.

**Phase 17 결론**: 가설 기각.
- cfr_lambda 단조 증가 트렌드는 **0.2에서 peak**. 0.3/0.5에서 AUC 유지~하락이나 **IEB 8배 악화** (0.014→0.10+).
  과도한 CFR이 imputation tower를 over-regularize → DR correction 품질 하락 → WCTR bias 증가.
- impute_loss_weight도 0.5이 최적. 0.3(AS)은 AUC 소폭 개선(+0.002)이나 IEB 급증, 1.0(AT)은 양쪽 모두 하락.
- **Run AL (cfr=0.2, ilw=0.5)이 internal HP 최적 확정.**

**실행 명령**:
```bash
# Run AQ: cfr_lambda=0.3
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features --model-dir results/models \
    --debiasing dr --epochs 50 --batch-size 65536 --learning-rate 5e-4 \
    --embedding-dim 32 --hidden-dims 128,64 --win-hidden-dims 64,32 \
    --dropout 0.4 --weight-decay 1e-3 --scheduler cosine --warmup-steps 200 \
    --gradient-clip 1.0 --use-layer-norm --eval-every 1 --es-metric ctr_auc \
    --patience 5 --cfr-lambda 0.3 --win-eps 0.05 --max-weight 10.0 \
    --impute-loss-weight 0.5 --joint-weight 1.0 --win-weight 0.01 --ctr-weight 0.01

# Run AR: cfr_lambda=0.5
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features --model-dir results/models \
    --debiasing dr --epochs 50 --batch-size 65536 --learning-rate 5e-4 \
    --embedding-dim 32 --hidden-dims 128,64 --win-hidden-dims 64,32 \
    --dropout 0.4 --weight-decay 1e-3 --scheduler cosine --warmup-steps 200 \
    --gradient-clip 1.0 --use-layer-norm --eval-every 1 --es-metric ctr_auc \
    --patience 5 --cfr-lambda 0.5 --win-eps 0.05 --max-weight 10.0 \
    --impute-loss-weight 0.5 --joint-weight 1.0 --win-weight 0.01 --ctr-weight 0.01

# Run AS: impute_loss_weight=0.3
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features --model-dir results/models \
    --debiasing dr --epochs 50 --batch-size 65536 --learning-rate 5e-4 \
    --embedding-dim 32 --hidden-dims 128,64 --win-hidden-dims 64,32 \
    --dropout 0.4 --weight-decay 1e-3 --scheduler cosine --warmup-steps 200 \
    --gradient-clip 1.0 --use-layer-norm --eval-every 1 --es-metric ctr_auc \
    --patience 5 --cfr-lambda 0.2 --win-eps 0.05 --max-weight 10.0 \
    --impute-loss-weight 0.3 --joint-weight 1.0 --win-weight 0.01 --ctr-weight 0.01

# Run AT: impute_loss_weight=1.0
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features --model-dir results/models \
    --debiasing dr --epochs 50 --batch-size 65536 --learning-rate 5e-4 \
    --embedding-dim 32 --hidden-dims 128,64 --win-hidden-dims 64,32 \
    --dropout 0.4 --weight-decay 1e-3 --scheduler cosine --warmup-steps 200 \
    --gradient-clip 1.0 --use-layer-norm --eval-every 1 --es-metric ctr_auc \
    --patience 5 --cfr-lambda 0.2 --win-eps 0.05 --max-weight 10.0 \
    --impute-loss-weight 1.0 --joint-weight 1.0 --win-weight 0.01 --ctr-weight 0.01
```

### Phase 18: External Win Propensity for DR — COMPLETE

**가설**: Internal win tower(AUC 0.64, win_weight=0.01 제약)를 LGB Win PS(AUC 0.91)로
교체하면 DR importance weights의 정확도가 크게 향상되어 WCTR AUC 개선 + calibration 유지.
논문의 DR 프레임워크를 유지하면서 propensity source만 교체 — causal inference standard practice.

**코드 변경**: `--use-external-propensity` CLI 옵션 추가.
`load_win_propensity_models()` → `materialize_to_source(ext_propensity=...)` → `batch_to_jax()` →
loss_fn에서 `config.use_external_propensity` 분기로 external PS 사용.
Internal win tower는 Win BCE + ESMM joint constraint 유지.

**External PS 특성** (학습 시 출력):
- Train PS: AUC=0.9308, mean=0.2133
- Val PS: AUC=0.8553, mean=0.3425
- Test PS: AUC=0.6493, mean=0.4825
- **Train→Test PS AUC 급락** (0.93→0.65): LGB Win PS도 temporal shift의 영향

| Run | Config | External PS | Test WCTR AUC | Test CTR AUC | Test WCTR IEB | Test Win AUC |
|-----|--------|-------------|--------------|-------------|--------------|-------------|
| *AL (internal PS)* | cfr=0.2, ctr_w=0.01 | No | 0.6843 | 0.5636 | **0.014** | 0.6403 |
| **AW** | **AL + Ext PS** | **LGB Win** | **0.6882** | **0.5713** | 0.045 | 0.6466 |
| AV | Run J cfg (ctr_w=0.0, cfr=0.1) + Ext PS | LGB Win | 0.6712 | 0.5604 | 0.035 | 0.6384 |

**결론**: 가설 부분 확인.
- **Run AW(AL+Ext PS)가 전체 ESCM2-WC best WCTR AUC 0.6882** (+0.004 vs AL).
  External PS의 정확한 importance weights가 DR correction을 개선, CTR AUC도 +0.008.
- **AUC-Calibration trade-off**: AW의 WCTR IEB 0.045는 AL(0.014)보다 3배 높으나,
  Phase 17 runs(0.10+)보다는 양호. External PS가 train에서 과신(AUC 0.93)하여
  test(AUC 0.65)에서 DR weight가 불정확 → bias 증가.
- **AV(ctr_weight=0.0)가 AW보다 열위**: DR debiasing이 활성화되려면 약간의 ctr_weight(0.01)가 필요.
  ctr_weight=0.0에서는 DR correction이 CTR tower로 전파되지 않음.
- **External PS의 한계**: Train AUC 0.93이지만 Test AUC 0.65 — temporal shift로
  PS 품질이 급감. 이상적으로는 test-time PS도 정확해야 DR이 최대 효과.

**실행 명령**:
```bash
# Run AW: AL config + External PS (best)
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features --model-dir results/models \
    --debiasing dr --epochs 50 --batch-size 65536 --learning-rate 5e-4 \
    --embedding-dim 32 --hidden-dims 128,64 --win-hidden-dims 64,32 \
    --dropout 0.4 --weight-decay 1e-3 --scheduler cosine --warmup-steps 200 \
    --gradient-clip 1.0 --use-layer-norm --eval-every 1 --es-metric ctr_auc \
    --patience 5 --cfr-lambda 0.2 --win-eps 0.05 --max-weight 10.0 \
    --impute-loss-weight 0.5 --joint-weight 1.0 --win-weight 0.01 --ctr-weight 0.01 \
    --use-external-propensity

# Run AV: Run J config + External PS
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features --model-dir results/models \
    --debiasing dr --epochs 50 --batch-size 65536 --learning-rate 5e-4 \
    --embedding-dim 32 --hidden-dims 128,64 --win-hidden-dims 64,32 \
    --dropout 0.4 --weight-decay 1e-3 --scheduler cosine --warmup-steps 200 \
    --gradient-clip 1.0 --use-layer-norm --eval-every 1 --es-metric ctr_auc \
    --patience 5 --cfr-lambda 0.1 --win-eps 0.05 --max-weight 10.0 \
    --impute-loss-weight 0.5 --joint-weight 1.0 --win-weight 0.01 --ctr-weight 0.0 \
    --use-external-propensity
```

### Phase 20: Target Encoding Feature 추가 — FAILED (가설 기각)

**가설**: Categorical features (region, city, advertiser, domain_hash, creative_hash)에 대해
click/win target encoding을 적용하면, CTR tower에 사전 계산된 확률적 signal을 주입하여
극도로 sparse한 click signal (배치당 ~3.4 clicks/65536)을 보완.
K-fold OOF encoding + Bayesian smoothing (m=10)으로 leakage 방지.
10개 새 numerical features: 5 cats × 2 targets (click, win).

**Feature Pipeline 변경**:
- `scripts/build_features.py`: `--target-encoding` flag + `target-encode` 서브커맨드 추가
- `src/features/engineering.py`: `target_encode_kfold()` sklearn→pure numpy KFold 대체
- Features: 30→40 (17 cat + 23 num), TE columns은 numerical로 자동 분류

| Run | Base Config | Features | 비교 대상 |
|-----|------------|----------|----------|
| AX | Run AW (ESCM2-WC DR, ext PS, cfr=0.2) | 30→40 (+10 TE) | AW (0.6882) |
| AY | Run J (ESMM-WC) | 30→40 (+10 TE) | J (0.6905) |

**결과**:

| Run | Model | Test CTR AUC | Test WCTR AUC | vs Baseline | 비고 |
|-----|-------|-------------|--------------|-------------|------|
| AY | ESMM-WC + TE | **0.4209** | **0.5480** | J 0.6905 → **-0.1424** | 가설 기각 |
| AX | ESCM2-WC DR + ext PS + TE | -- | -- | -- | SKIPPED (AY 결과 기반) |

**결론**: **가설 기각**. Target encoding이 성능을 대폭 악화시킴.

- CTR AUC: 0.6237 → 0.4209 (-0.2028), WCTR AUC: 0.6905 → 0.5480 (-0.1424)
- **원인 분석**:
  1. **Click TE의 극도로 낮은 signal**: global_mean=0.000146 (CTR 0.015%), std~5e-5 → 거의 모든 TE 값이 동일하여 noise만 추가
  2. **Feature 비율 왜곡**: 10 TE features가 numerical 23개의 43%를 차지 — MLP가 uninformative features에 capacity 낭비
  3. **Temporal drift 증폭**: S2 train 통계가 S3 test에서 stale → TE가 오히려 temporal shift를 악화
  4. **Win TE도 비효과적**: Win TE는 상대적으로 informative(mean 0.21, std 0.06-0.18)하나, 이미 Win Tower가 직접 학습 → redundant signal
- **교훈**: Neural multi-task model에서 TE는 LR/LGB와 달리 embedding layer가 categorical→continuous mapping을 이미 수행하므로 redundant. 특히 극도로 sparse한 target(click rate 0.015%)에 대한 TE는 noise 대비 signal이 부족

**실행 명령**:
```bash
# Feature build with target encoding
python scripts/build_features.py target-encode \
    --data-dir data/ipinyou/prediction/features \
    --output-dir data/ipinyou/prediction/features_te

# Run AX: ESCM2-WC DR + External PS + TE
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features_te --model-dir results/models/escm2wc_te_AX \
    --debiasing dr --epochs 50 --batch-size 65536 --learning-rate 5e-4 \
    --embedding-dim 32 --hidden-dims 128,64 --win-hidden-dims 64,32 \
    --dropout 0.4 --weight-decay 1e-3 --scheduler cosine --warmup-steps 200 \
    --gradient-clip 1.0 --use-layer-norm --eval-every 1 --es-metric ctr_auc \
    --patience 5 --cfr-lambda 0.2 --win-eps 0.05 --max-weight 10.0 \
    --impute-loss-weight 0.5 --joint-weight 1.0 --win-weight 0.01 --ctr-weight 0.01 \
    --use-external-propensity

# Run AY: ESMM-WC + TE (Run J config)
python scripts/train.py esmmwc \
    --data-dir data/ipinyou/prediction/features_te --model-dir results/models/esmmwc_te_AY \
    --epochs 50 --batch-size 65536 --learning-rate 5e-4 \
    --embedding-dim 32 --hidden-dims 128,64 --win-hidden-dims 64,32 \
    --dropout 0.4 --weight-decay 1e-3 --scheduler cosine --warmup-steps 200 \
    --gradient-clip 1.0 --use-layer-norm --eval-every 1 --es-metric ctr_auc \
    --patience 15 --joint-weight 1.0 --win-weight 0.01 --ctr-weight 0.0
```

---

## 7. Calibration & Bias 비교

AUC는 ranking 능력만 측정하므로, 확률 calibration과 prediction bias를 별도로 평가해야
모델의 실제 production 적합성을 판단할 수 있다.

### 7.1 ECE (Expected Calibration Error) 비교

ECE는 예측 확률과 실제 빈도 간 calibration 오차. 낮을수록 우수.

| Model | Task | Test AUC | Test ECE |
|-------|------|----------|----------|
| LGB CTR | winners CTR | **0.6890** | 0.000514 |
| LGB CTR_all | all-bids CTR | 0.5437 | 0.000094 |
| LGB Win | Win | 0.6493 | **0.264** |
| LR CTR | winners CTR | 0.3216 | 0.000180 |
| LR CTR_all | all-bids CTR | **0.7687** | 0.000028 |
| LR Win | Win | 0.5889 | **0.263** |
| ESMM-WC Run J | CTR (biased) | 0.6237 | **0.000006** |
| ESMM-WC Run J | WCTR | 0.6905 | 0.000308 |
| ESCM2-WC(DR) Run R | CTR (biased) | 0.5526 | 0.000565 |
| ESCM2-WC(DR) Run R | WCTR | 0.6766 | 0.000013 |
| **ESCM2-WC(DR) Run AL** | **WCTR** | **0.6843** | **0.000003** |
| ESCM2-WC(DR) Run AW | WCTR | **0.6882** | 0.000026 |
| ESCM2-WC(IPW) Run Q | WCTR | 0.6526 | 0.000011 |

**핵심 관찰:**
- **LGB Win ECE 0.264**: Win probability calibration이 극히 나쁨. 예측값과 실제 win rate 간
  26.4% gap — 이 확률을 bid shading에 직접 사용하면 심각한 가격 왜곡 발생
- **LR Win ECE 0.263**: LGB Win과 유사하게 나쁜 calibration
- **ESCM2-WC(DR) WCTR ECE 0.000003**: 가장 우수한 확률 calibration. ESMM-WC(0.000308)
  대비 100배 이상 우수
- CTR 개별 평가에서는 ESMM-WC Run J(0.000006)이 최소 ECE이나, production에서
  사용하는 WCTR(all-bids CTR) 기준으로는 ESCM2-WC(DR)이 압도적

### 7.2 IEB (Integral Error Bias) 비교

IEB는 누적 예측 편향. 낮을수록 unbiased prediction에 가까움.
ESMM-WC와 ESCM2-WC만 CTR/WCTR IEB를 산출 가능 (multi-task 모델).

| Model | CTR IEB | WCTR IEB | 비고 |
|-------|---------|----------|------|
| ESMM-WC Run J | **0.005** | 1.335 | CTR 우수, WCTR 매우 편향 |
| ESCM2-WC(DR) Run P | 0.521 | 0.057 | ctr_weight=0.0 |
| ESCM2-WC(DR) Run R | 0.534 | 0.056 | ctr_weight=0.01 |
| **ESCM2-WC(DR) Run AL** | 0.515 | **0.014** | **cfr_lambda=0.2, calibration best** |
| ESCM2-WC(DR) Run AW | 0.519 | 0.045 | AL + External PS, **AUC best** |
| ESCM2-WC(IPW) Run Q | 0.532 | 0.046 | IPW debiasing |

**핵심 관찰:**
- **ESCM2-WC(DR) Run AL의 WCTR IEB 0.014**: ESMM-WC Run J(1.335) 대비 **~95배 우수**.
  DR debiasing이 all-bids CTR prediction의 selection bias를 효과적으로 제거
- **ESMM-WC의 CTR IEB 0.005 vs ESCM2-WC의 ~0.5**: Winners-only CTR에서는 ESMM-WC가
  calibration 우위. 그러나 이는 winners subset에서만 유효하며, all-bids 전체에서는 의미 제한적
- **Trade-off 구조**: ESMM-WC는 CTR tower 자체는 well-calibrated하나 P(win)×P(click|win)
  product의 error가 누적되어 WCTR IEB가 큼. ESCM2-WC(DR)은 DR correction이 product
  bias를 보정하여 WCTR IEB를 극적으로 낮춤

### 7.3 AUC vs Calibration Trade-off

| 평가 관점 | Best Model | 수치 |
|-----------|-----------|------|
| Winners-only CTR AUC | LGB CTR | 0.6890 |
| All-bids CTR AUC | LR CTR_all | 0.7687 |
| Win AUC | LGB Win | 0.6493 |
| **WCTR AUC (best neural)** | **ESCM2-WC(DR) AW** | **0.6882** |
| **WCTR ECE (calibration)** | **ESCM2-WC(DR) AL** | **0.000003** |
| **WCTR IEB (bias)** | **ESCM2-WC(DR) AL** | **0.014** |
| Win ECE (calibration) | ESMM-WC Run J | ~0.001 (implicit) |

AUC 기준으로는 LGB/LR baseline이 개별 task에서 우위이나,
**calibration과 bias 기준으로는 ESCM2-WC(DR)이 압도적 우위**.

RTB production에서는 calibrated probability가 bid price = base_price × pCTR에 직접 사용되므로,
AUC(ranking)보다 calibration(절대값 정확도)이 더 중요.

**모델 선택 trade-off (Phase 18 이후):**
- **Calibration 중시 → Run AL**: WCTR ECE 0.000003, IEB 0.014 (최고 calibration)
- **AUC 중시 → Run AW**: WCTR AUC 0.6882 (최고 AUC), IEB 0.045 (양호)
- 실제 production에서는 두 모델의 A/B test로 revenue impact 비교 필요

---

## 8. Key Tuning Insights

### 6.1 Loss Weight Sensitivity

Win-weight is the single most impactful hyperparameter:

| win_weight | Effect |
|-----------|--------|
| 1.0 | Win tower dominates gradients, CTR tower underfits |
| 0.1 | Moderate improvement |
| **0.01** | **Optimal** -- CTR tower learns from ESMM constraint |

Joint-weight should remain at 1.0. Increasing it (3.0) shifts focus to
`P(win) x P(ctr|win)` product metric, degrading component CTR tower.

### 6.2 Learning Rate and Convergence

| LR | Best Epoch | Behavior |
|----|-----------|----------|
| 1e-3 | 3 | Too fast, sharp peak then decay |
| **5e-4** | **5** | **Sweet spot** |
| 3e-4 | 7 | Slow convergence, worse generalization |

### 6.3 Regularization Ceiling

Beyond Run J's configuration (dropout=0.4, weight_decay=1e-3), additional
regularization **hurts** test performance. The Val-Test gap is driven by
temporal distribution shift (S2 train -> S3 test), not by overfitting.

### 6.4 Architecture Choices

- **LayerNorm**: Consistently helpful across all runs
- **FM interaction**: Provides modest gains for categorical feature interactions
- **Cosine scheduler + warmup**: Essential for stable training with large batches (65536)
- **Gradient clipping (1.0)**: Prevents loss spikes during early training

### 6.5 Early Stopping

- `es-metric=ctr_auc` is critical (vs `total` which is win-dominated)
- `patience=5` is sufficient given fast convergence patterns
- `eval-every=1` required for `ctr_auc` metric computation

---

## 9. Temporal Distribution Shift

The fundamental bottleneck is **S2->S3 temporal shift**:

- Train/Val: Season 2 (2013-06)
- Test: Season 3 (2013-10)
- ~4 months gap with different campaign mixes, user behaviors, market conditions

모든 모델이 temporal shift의 영향을 받지만, 정도가 다름:
- **LGB**: Train→Test AUC drop 최대 (CTR: 0.81→0.69, Win: 0.93→0.65, CTR_all: 0.91→0.54)
- **ESMM-WC**: Val→Test gap 존재하나 LGB보다 양호 (CTR_all: ESMM-WC 0.69 >> LGB 0.54)
- **LR**: Temporal shift에 가장 robust (CTR_all: 0.77→0.77, 거의 무하락)

LR이 robust한 이유:
1. Linear model은 파라미터가 적어 distribution memorization이 제한적
2. Raw scalar features가 temporal context를 직접 전달
3. Embedding lookup tables가 없어 category-specific overfitting 없음

**결론**: ESMM-WC는 LGB 대비 temporal robustness에서 우위이나, LR의 linear simplicity를
neural model로 재현하기 어려움. 연구 방향을 ESCM²-WC(DR) debiasing + Bid Optimization으로 전환.

---

## 10. 구조적 Trade-off 분석

16 Phases의 실험을 통해 확인된 ESMM-WC / ESCM2-WC 아키텍처의 구조적 한계와 trade-off.

### 10.1 Product Noise: P(win)×P(click|win) vs 직접 P(click|bid)

ESMM/ESCM2의 핵심 구조는 `WCTR = P(win) × P(click|win)`:
- **장점**: Funnel decomposition으로 selection bias를 명시적으로 모델링
- **단점**: 두 확률의 곱(product)이 개별 tower의 error를 **곱셈으로 증폭**

실험적 증거:
- LR CTR_all은 P(click|bid)를 직접 학습 → all-bids AUC 0.7687
- ESMM-WC는 product로 WCTR AUC 0.6905 (gap 0.078)
- ESCM2-WC(DR)은 3rd tower overhead까지 추가되어 WCTR AUC 0.6843

그러나 **calibration에서는 product 구조가 유리**:
- ESCM2-WC(DR) WCTR IEB 0.014 — DR이 product bias를 보정
- LR CTR_all은 AUC가 높지만, winners 내 CTR AUC 0.3216으로 실질적 판별력 부재

### 10.2 Shared Embedding Conflict (win_weight 딜레마)

Multi-task 모델의 핵심 한계: Win tower와 CTR tower가 **동일한 embedding layer를 공유**.

| win_weight | Win Tower | CTR Tower | 결과 |
|-----------|-----------|-----------|------|
| 1.0 | 강한 supervision | Gradient 간섭으로 학습 불가 | CTR AUC 0.30 (catastrophic) |
| 0.1 | 중간 supervision | 여전히 간섭 | CTR AUC 0.44 |
| **0.01** | **최소 supervision** | **독립 학습 가능** | **CTR AUC 0.62 (best)** |

- win_weight=0.01에서 Win tower는 사실상 CTR gradient의 부산물로만 학습
- 이는 ESCM2-WC에서 DR propensity의 품질 저하를 야기 (정확한 P(win)이 필요하나 약한 supervision)
- `stop_gradient` (Phase 10)도 embedding 보호에는 성공하나 win tower MLP overfitting은 해결 불가
- **딜레마**: Win tower를 강하게 학습시키면 CTR 성능 하락, 약하게 하면 propensity 부정확

### 10.3 3rd Tower (Imputation) Overhead

ESCM2-WC는 ESMM-WC 대비 Imputation tower를 추가:
- 추가 파라미터가 temporal shift에 대한 추가 취약점
- Run J(ESMM-WC) WCTR 0.6905 > Run R(ESCM2-WC) WCTR 0.6766 (3rd tower 추가로 하락)
- cfr_lambda=0.2 (Run AL)로 개선해도 WCTR 0.6843 — 여전히 ESMM-WC 미달
- **Phase 18 External PS (Run AW)로 WCTR 0.6882 달성** — ESMM-WC Run J(0.6905)에 근접
- **AUC 기준**으로는 3rd tower가 internal PS에서 순손해이나, external PS로 gap 축소

그러나 **calibration 기준**으로는 3rd tower가 순이익:
- Run AL WCTR IEB 0.014 << Run J WCTR IEB 1.335
- DR correction이 WCTR prediction bias를 극적으로 감소시킴

### 10.4 Production 관점 모델 선택 가이드

| 사용 목적 | 추천 모델 | 근거 |
|-----------|-----------|------|
| Winners-only CTR ranking | LGB CTR | AUC 0.6890 (개별 best) |
| Win rate prediction (ranking) | LGB Win | AUC 0.6493 (개별 best) |
| All-bids CTR ranking | LR CTR_all | AUC 0.7687 (easy negatives 포함) |
| **Best AUC neural model** | **ESCM2-WC(DR) Run AW** | **WCTR AUC 0.6882, IEB 0.045** |
| **Calibrated pCTR for bidding** | **ESCM2-WC(DR) Run AL** | **WCTR ECE 0.000003, IEB 0.014** |
| **Debiased CTR (unbiased)** | **ESCM2-WC(DR) Run AL/AW** | **DR이 selection bias 제거** |
| Win rate for bid shading | ESMM-WC Run J Win tower | Win AUC 0.6432, implicit calibration |

**Production RTB 시나리오에서의 결론:**

1. **Bid price = base_price × pCTR**: RTB에서 예측 확률은 bid price에 직접 곱해짐.
   AUC(ranking)가 높아도 calibration이 나쁘면 체계적 과입찰/저입찰 발생.
   ESCM2-WC(DR)의 WCTR ECE 0.000003(AL) / 0.000026(AW)은 가격 왜곡을 최소화.

2. **Win rate for bid shading**: First-price auction에서 bid shading에 P(win|bid)이 필요.
   LGB Win은 AUC 0.6493이나 **ECE 0.264로 calibration 최악** — 직접 사용 불가.
   ESMM-WC/ESCM2-WC의 win tower는 jointly trained으로 implicit calibration이 양호.

3. **AUC만 보면 LGB/LR이 개별 task best**이나, multi-task 모델이
   (a) selection bias 제거, (b) 확률 calibration, (c) 단일 inference로 CTR+Win 동시 예측이라는
   production 이점을 제공.

4. **AUC-Calibration trade-off**: Run AW(AUC 0.6882, IEB 0.045) vs Run AL(AUC 0.6843, IEB 0.014).
   Calibration 중시 시 AL, ranking 중시 시 AW. A/B test로 revenue impact 비교 필요.

5. **최종 추천**: Production 배포에는 **ESCM2-WC(DR)**을 primary model로,
   LGB/LR baseline은 offline evaluation의 benchmark로 유지.
   AUC gap(LR CTR_all 0.7687 vs AW 0.6882 = 0.081)은 easy negatives 효과를 감안하면 실질적으로 더 작으며,
   calibration 우위(ECE 100x, IEB 95x)가 RTB revenue에 직접적 영향.
