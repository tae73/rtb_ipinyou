# SP1: 예측 모델 + Win Selection Bias Debiasing

> **현행 결과:** fair-split 재학습·recalibration·Stage B2 결과와 정본 지표 정의는
> [`../redesign_findings.md`](../redesign_findings.md) · [`../evaluation_protocol.md`](../evaluation_protocol.md) 참조.

---

## 개요

| 항목 | 내용 |
|------|------|
| **목적** | Bid→Win→Click 퍼널에서 unbiased pCTR 예측 |
| **선행 조건** | SP0 (데이터 준비) 완료 |
| **후속 단계** | SP2 (Win Rate), SP3 (Bid Optimization), SP5 (Serving) |
| **핵심 산출물** | ESMM-WC, ESCM²-WC(DR) debiased pCTR 모델 |

**CVR Tower Pivot (EDA 2.2.1):**
- Branding (5/10 advertisers): CVR=0 → CTCVR 의미 없음
- Retargeting (3/10): CVR 28-53%, click→conv ~1초 → retargeting artifact
- 전체 CTCVR 0.0061% → CVR near-trivial
- **Pivot**: Impression→Click→Conversion 대신 **Bid→Win→Click** 퍼널 적용

---

## Part A: 데이터셋 구조 (Bid→Win→Click)

### A-1. 퍼널 구조

```
전체 Bid 기회 → Win (낙찰) → Click
                ↑
          Win Selection Bias
        (129.5M bids → 30.6M won)

데이터셋:
- D_all: 전체 입찰 (129.5M bids, S2: 106.6M + S3: 22.9M)
- D_won: Win=1 샘플 (30.6M impressions, WR 23.67%)
- D_clicked: Click=1 샘플 (23,058 clicks, CTR 0.0752%)
```

### A-2. pCTR 모델 (Biased Baseline)

```python
# pCTR Biased Baseline: D_won에서만 학습
# 문제: Win Selection Bias — P(Click|Win=1, X) ≠ P(Click|X)
train_won = train[train['win'] == 1]
val_won = val[val['win'] == 1]

X_train = train_won[feature_cols]
y_train = train_won['click']

pctr_model = lgb.train(params, train_data, ...)
# AUC ~0.7-0.75, but biased (overestimates CTR by ~6.7%)
```

### A-3. CVR Near-Trivial (EDA 2.2.1 Finding)

```
기존 Click→Conversion 퍼널의 한계:
- 전체 CTCVR = 0.0061% (Conversion 1,860건 / 30.6M impressions)
- Imp→Click: ~12초, Click→Conv: ~1초, 100% click-through attribution
- Branding 5/10: CVR = 0 (전환 자체가 없음)
- Retargeting 3/10: CVR 28-53% (click→conv ~1초, retargeting artifact)
- pCVR 모델 학습은 실질적으로 3개 advertiser의 retargeting 패턴만 학습

→ CTCVR 예측이 아닌 CTR 예측이 핵심 value signal
→ V(x) = debiased_pCTR × CPC_target (CPC campaign)
```

### A-4. Calibration의 중요성

```
RTB에서 Calibration이 중요한 이유:

bid = debiased_pCTR × CPC_target × shade × pace

- pCTR이 체계적으로 과대추정 (+6.7%) → 입찰가 과다 → ROI 하락
- pCTR이 체계적으로 과소추정 → 입찰가 과소 → Win rate 하락

해결: ESMM-WC / ESCM²-WC(DR) 로 unbiased pCTR 획득
```

---

## Part B: Win Selection Bias 이해

### B-1. Win Selection Bias 구조

```
D_all (전체 129.5M bids)
    ↓ P(Win=1 | X, bid)
D_won (Win=1, 30.6M)             ← Win Selection Bias
    ↓ P(Click=1 | Win=1, X)
D_clicked (Click=1, 23K)

문제:
- pCTR: D_won에서 학습 → P(Click | Win=1, X) ≠ P(Click | X)
- Win=1 샘플: 입찰가 높은 경쟁 낮은 세그먼트에서 over-sampled
- → CTR +6.7% 과대추정 (notebook 02 진단)
```

### B-2. 왜 Debiasing이 필요한가?

```
Win Selection Bias의 영향:

1. CTR 과대추정 (+6.7%)
   - Won impressions: 경쟁이 낮은 세그먼트에서 over-sampled
   - 이 세그먼트의 CTR이 실제보다 높게 추정됨
   - Subgroup별 -10%~+18% 편향 (Simpson's Paradox)

2. 입찰 최적화 왜곡
   - Biased CTR로 입찰 → 실제 가치 높은 유저 저평가
   - ROI 하락

3. 모델 일반화 실패
   - Train (Won=1) 분포 ≠ Serving (전체 bids) 분포
   - Covariate shift로 성능 저하
```

### B-3. IPW vs Doubly Robust 비교

| 항목 | IPW | Doubly Robust |
|------|-----|---------------|
| **공식** | w × y | w × (y - μ) + μ |
| **일관성 조건** | Propensity 정확 | Propensity OR Outcome 정확 |
| **Variance** | 높음 (weight 변동) | 낮음 (μ가 변동 흡수) |
| **Extreme weight** | 민감 | 덜 민감 |
| **iPinYou 현황** | PS AUC 0.91, overlap 46% → 위험 | DR + ESMM constraint → 완화 |
| **권장** | Ablation용 | **Production 권장** |

### B-4. Positivity Diagnosis (from notebook 02)

| 진단 항목 | LGB (clean) | 해석 |
|-----------|-------------|------|
| PS AUC | ~0.91 | 높은 분리도 |
| Overlap [0.1, 0.9] | ~46% | 약절반만 overlap |
| ESS ratio | ~7% | IPW weight 불안정 |
| CTR Overestimation | +6.7% | 교정 필요 |

**IPW 단독의 위험성과 완화:**
1. **DR은 doubly robust**: propensity OR imputation 중 하나만 맞아도 unbiased
2. **Self-normalized weights + clipping**: weight explosion 방지
3. **ESMM constraint**: propensity 의존 없는 추가 signal (joint BCE)
4. **Imputation tower 학습 데이터**: 30.6M won samples (기존 23K clicks보다 1300x 풍부)

---

## Part C: Debiasing 방법론 (ESMM-WC + ESCM²-WC)

### C-1. Win Propensity 모델 (Dual Purpose)

```python
# Win Propensity: P(Win|X, bid)
# Dual purpose:
#   (a) CTR debiasing propensity (ESCM²-WC 내부 or 외부 LGB)
#   (b) Bid shading win rate model (SP3)

from src.debiasing.win_propensity import WinPropensityModel

# External LGB Win PS (cross-fitted, calibrated)
wp_model = WinPropensityModel()
wp_model.fit(X, bid_price, y_win)  # AUC ~0.91
propensity = wp_model.predict(X, bid_price)
```

### C-2. ESMM-WC (2-tower, ESMM Constraint Only)

```python
# ESMM-WC: Ablation baseline (no DR/IPW)
# 구조:
#   Win Tower: P(Win|X, bid)
#   CTR Tower: P(Click|X, Win)
#   Joint: P(Click_bid) = P(Win) × P(Click|Win)

from src.models.esmm_wc import ESMMWCConfig, ESMMWC

config = ESMMWCConfig(
    feature_dims=feature_dims,
    embed_dim=16,
    hidden_dims=(128, 64),
    win_hidden_dims=(64, 32),
)
model = ESMMWC(config, rngs=rngs)

# Losses:
# 1. Win BCE: BCE(p_win, win) — all 129.5M bids
# 2. CTR BCE: win × BCE(p_ctr, click) — won samples only
# 3. Joint BCE: BCE(p_click_bid, click) — all bids (ESMM constraint)
#
# No DR/IPW, no imputation tower
# Implicit debiasing via ESMM constraint on joint prediction
```

### C-3. ESCM²-WC(DR) (3-tower, Primary Model)

```python
# ESCM²-WC(DR): Primary debiasing model
# 구조:
#   Win Tower: P(Win|X, bid) — propensity
#   CTR Tower: P(Click|X, Win) — debiased via DR
#   Imputation Tower: delta_hat — CTR error prediction
#   Joint: P(Click_bid) = P(Win) × P(Click|Win)

from src.models.escm2_wc import ESCM2WCConfig, ESCM2WC

config = ESCM2WCConfig(
    feature_dims=feature_dims,
    embed_dim=16,
    hidden_dims=(128, 64),
    win_hidden_dims=(64, 32),
    loss_type="dr",     # 'dr' or 'ipw'
    win_eps=0.05,       # propensity clipping
    max_weight=10.0,
    normalize_weights=True,
    cfr_lambda=0.1,
)
model = ESCM2WC(config, rngs=rngs)

# Losses:
# 1. Win BCE: BCE(p_win, win) — all 129.5M bids
# 2. CTR DR: delta_hat + (win/P(Win)) × ((click - p_ctr) - delta_hat)
# 3. Joint BCE: BCE(p_click_bid, click) — all bids (ESMM constraint)
# 4. Imputation: won × MSE(y_impute, click - p_ctr) — supervised on won
# 5. CFR: unselected (win=0) samples regularization
```

### C-4. ESCM²-WC 학습 (JAX/Flax)

```python
import optax
from flax import nnx

# Model + optimizer
rngs = nnx.Rngs(0)
model = ESCM2WC(config, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

# Training step
train_step = create_escm2wc_train_step(config)

for epoch in range(50):
    for batch in data_loader:
        # batch = {"x": features, "win": ..., "click": ...}
        # NOTE: ALL bids, no win==1 filtering
        loss = train_step(model, optimizer, batch)
```

```bash
# CLI training
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features \
    --model-dir results/models \
    --debiasing dr \
    --epochs 50 \
    --batch-size 4096
```

---

## Part D: 실무 고려사항

### D-1. Weight 분포 모니터링

```python
from src.models.escm2 import compute_weight_statistics

# Monitor IPW weights during training
stats = compute_weight_statistics(weights)
# Check ESS ratio > 0.05, weight_max < max_weight
```

### D-2. 65M Rows 메모리 관리

```
대규모 데이터 처리:
- Batch training (4096) → GPU 메모리 관리
- 필요시 data generator (streaming from Parquet)
- CTR 극단적 sparsity: 23,058/129,493,498 = 0.0178%
  → Focal loss 또는 negative sampling 검토 가능
```

### D-3. Gradient Clipping

```python
# DR weights로 인한 gradient 폭발 방지
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3),
)
```

### D-4. Exchange-conditional Win Propensity (EDA-driven)

```
EDA Finding:
- Exchange별 win rate: 13.24% ~ 55.55% (극심한 차이)
- Exchange별 floor 메커니즘 상이 (Ex1: no floor, Ex2: moderate, Ex3: active)

고려사항:
- Win Tower에 exchange interaction feature 추가 검토 (exchange × bid 등)
- Exchange별 win 메커니즘 차이를 반영하면 propensity 정확도 향상 기대

Note: 원논문(ESMM/ESCM²) 구조는 shared embedding + shared input.
Tower-specific feature selection은 future ablation으로만 검토.
현재는 exchange를 shared input에 포함하되, interaction term은 optional.
```

### D-5. Floor-binding-aware Propensity Clipping (EDA-driven)

```
EDA Finding:
- 32.24% of won bids에서 payprice ≈ floor price (floor-binding)
- Floor-bound 시 P(Win) ≈ 1 가능 (floor만 넘으면 낙찰)
- → P(Win) 분포에 mass point 생성

고려사항:
- win_eps (propensity clipping lower bound) 조정 시 floor-binding 비율 고려
- Floor-bound 샘플의 propensity는 클리핑 하한보다 높을 수 있음
- is_floor_binding feature 추가로 모델이 두 regime을 명시적으로 구분
```

### D-6. Domain Concentration 처리 (EDA-driven)

```
EDA Finding:
- 108K unique domains, top 50 ≈ 60% traffic
- Top 238 (0.2%) = 80% of bids
- 희소 domain은 모델 학습에 noise

처리 방안:
- domain_group feature: top-N domains 개별 유지, 나머지 `other` 그룹
- N = 50 (≈60% traffic) 또는 N = 100 (≈70% traffic) 권장
- Embedding-based 모델에서는 hashing trick으로 차원 축소 가능
```

---

## Part E: Unbiased Evaluation

### E-1. 평가 메트릭

| Metric | 대상 | 설명 |
|--------|------|------|
| Win AUC | All bids | Win Tower 성능 (≈ LGB PS AUC ~0.91) |
| CTR AUC (biased) | Won only | CTR Tower 성능 (biased sample) |
| CTR AUC (IPW-unbiased) | All bids | IPW-weighted AUC (unbiased) |
| Joint AUC | All bids | P(Click_bid) = P(Win) × P(Click|Win) |
| ECE | Won only | Calibration error |
| IEB | All bids | Integrated estimation bias |

### E-2. Ablation Study 설계

```python
# Ablation: Biased → ESMM-WC → ESCM²-WC(DR)
experiments = [
    {'name': 'Biased Baseline (LGB)', 'model': 'baseline'},
    {'name': 'ESMM-WC', 'model': 'esmmwc'},
    {'name': 'ESCM²-WC (IPW)', 'model': 'escm2wc', 'debiasing': 'ipw'},
    {'name': 'ESCM²-WC (DR)', 'model': 'escm2wc', 'debiasing': 'dr'},
    {'name': '+External Win PS', 'model': 'escm2wc', 'debiasing': 'dr', 'ext_ps': True},
]

# Expected: sequential improvement in IEB, ECE
# ESCM²-WC(DR) = primary contribution
```

---

## Part F: 예상 결과

| 모델 | Win AUC | CTR AUC (biased) | Joint AUC | ECE | 역할 |
|------|---------|------------------|-----------|-----|------|
| Biased Baseline (LGB) | N/A | 0.730 | N/A | 0.082 | Baseline |
| **ESMM-WC** | ~0.91 | ~0.745 | ~0.75 | ~0.065 | ESMM constraint |
| ESCM²-WC (IPW) | ~0.91 | ~0.760 | ~0.77 | ~0.055 | + IPW debiasing |
| **ESCM²-WC (DR)** | ~0.91 | **~0.775** | **~0.79** | **~0.045** | **Primary model** |
| +External Win PS | ~0.91 | ~0.778 | ~0.79 | ~0.044 | External PS variant |

---

## 산출물

| 산출물 | 경로 | 설명 |
|--------|------|------|
| pCTR 모델 (biased baseline) | `results/models/lgb_ctr.txt` | LightGBM baseline |
| ESMM-WC 모델 | `results/models/esmmwc_result.json` | 2-tower ablation baseline |
| ESCM²-WC(DR) 모델 | `results/models/escm2wc_dr_result.json` | Primary debiased model |
| Win Propensity 모델 | `results/models/win_propensity.pkl` | External PS (optional) |
| 성능 비교 리포트 | `notebooks/04_prediction_debiasing.ipynb` | Ablation 결과 |
| 학습 코드 | `src/models/esmm_wc.py`, `src/models/escm2_wc.py` | 모델 + Loss + Training |

---

## 참고 문헌

1. Ma, X., et al. (2018). "ESMM: Entire Space Multi-Task Model." SIGIR.
2. Wang, X., et al. (2022). "ESCM²: Entire Space Counterfactual Multi-Task Model." SIGIR.
3. Chernozhukov, V., et al. (2018). "Double/Debiased Machine Learning." Econometrics Journal.
