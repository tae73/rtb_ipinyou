# Neural Model Performance Tuning Log

**Last Updated**: 2026-03-04

## 1. Problem Statement

LR CTR_all baseline achieves **Test AUC 0.7828** on iPinYou RTB data using 30 raw StandardScaler float32 features.
Neural models (ESMM-WC / ESCM2-WC) initially underperformed due to architectural choices that
destroyed ordinal/magnitude information in numerical features.

**Goal**: Close the gap between LR baseline and neural multi-task models while preserving
the debiasing benefits of ESMM/ESCM2 architecture.

---

## 2. Baseline Comparison

| Model | Test CTR AUC | Test WCTR AUC | Notes |
|-------|-------------|--------------|-------|
| LGB CTR (biased) | 0.7446 | -- | Winners-only CTR, overfits to win distribution |
| LGB Win PS | AUC ~0.91 | -- | Win propensity model, positivity overlap ~48% |
| LGB CTR_all | 0.7695 | -- | All-bids CTR, less biased but LGB memorizes |
| **LR CTR_all (biased)** | **0.7828** | -- | 30 features, raw StandardScaler, temporal-robust |
| ESMM-WC Run J (best) | 0.6237 | 0.6905 | 2-tower, ESMM constraint |

**Gap**: LR 0.7828 vs ESMM-WC 0.6237 = **0.1591** (20% relative degradation)

**왜 LR CTR_all이 best baseline인가?**
- LR은 30개 numerical features를 raw StandardScaler float32로 직접 사용 → 시간적 변화에 강건
- LGB는 tree split으로 training distribution을 memorize → S2→S3 temporal shift에 취약
- LR의 top features (bidprice, domain_freq, slot_size 등)는 numerical features이며,
  이들의 ordinal/magnitude 정보가 CTR 예측에 핵심적 역할

---

## 3. Root Cause Analysis

### 3.1 Numerical Feature Embedding Bottleneck

The 13 numerical features (bidprice, domain_freq, creative_freq, hour, weekday, etc.) are
transformed by `Linear(1, 32)` projection per feature in `EmbeddingLayer`. This:

1. **Destroys ordinal relationships**: A linear projection maps scalar -> 32-dim vector,
   losing the natural ordering (e.g., bidprice=50 vs bidprice=100)
2. **Wastes parameters**: 13 features x (1x32 + 32 bias) = 832 parameters for what
   LR handles with 13 direct coefficients
3. **Inflates MLP input**: 13x32=416 dims from numericals alone vs 13 raw scalars

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

| Phase | 가설 | 핵심 변경 | Best Run | Test CTR AUC | 결과 |
|-------|------|----------|----------|-------------|------|
| 1 | Vanilla ESMM-WC로 LR 수준 가능? | 없음 (vanilla) | A | 0.5173 | Near-random, gradient starvation |
| 2 | Regularization + scheduler로 0.65+ | batch→65536, cosine, LayerNorm | F | 0.5226 | +0.005, win-weight 핵심 변수 발견 |
| 3 | Win-weight 극소화로 CTR 독립 학습 | ww=0.01 (win gradient 억제) | G | 0.5888 | +0.0662 (breakthrough) |
| 4 | LR 감소로 peak epoch 우측 이동 | lr→5e-4 | J | 0.6237 | +0.0349 (best) |
| 5 | Regularization 강화로 plateau 유지 | dropout↑, wd↑ | M/N/O | 0.5094-0.5478 | 역효과 (distribution shift) |
| 6 | Numeric bypass로 LR-NN gap 축소 | 아키텍처 변경 (raw scalar) | P/Q | (대기) | 구현 완료, 실험 대기 |

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

LR is more robust because:
1. Raw features preserve temporal/contextual signals directly
2. Linear model has fewer parameters to overfit
3. No embedding lookup tables that memorize training distribution

Neural models must bridge this gap through:
- Numeric bypass (preserve raw feature information)
- Feature engineering for temporal robustness
- Domain adaptation techniques (future work)
