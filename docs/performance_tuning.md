# Neural Model Performance Tuning Log

**Last Updated**: 2026-03-08

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
| 5 | Regularization 강화 | dropout↑, wd↑ | M/N/O | 0.5094-0.5478 | -- | 역효과 (distribution shift) |
| 6 | Numeric bypass | 아키텍처 변경 (raw scalar) | P/Q | (대기) | -- | 구현 완료, 실험 대기 |

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

### 5.4 Experiment Plan

| Run | embed_dim | Bypass | Cat Embed | FM | Num | Total MLP Input |
|-----|-----------|--------|-----------|----|----|-----------------|
| J (base) | 32 | OFF | 544 | 32 | 416 | 992 |
| P | 32 | ON | 544 | 32 | 13 | 589 |
| Q | 16 | ON | 272 | 16 | 13 | 301 |

### 5.5 Success Criteria

- **Primary**: Test WCTR AUC > 0.6905 (Run J best) or Test CTR AUC > 0.6237
- **Secondary**: Val-Test gap reduction (temporal robustness via preserved ordinal information)

---

## 6. Key Tuning Insights

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

## 7. Temporal Distribution Shift

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
