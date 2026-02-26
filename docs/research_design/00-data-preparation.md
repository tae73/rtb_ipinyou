# SP0: 데이터 준비 & EDA

---

## 개요

| 항목 | 내용 |
|------|------|
| **목적** | iPinYou 데이터셋 구축, 기초 탐색, Selection Bias 사전 진단 |
| **선행 조건** | 없음 |
| **후속 단계** | SP1 (Prediction Models) |
| **핵심 산출물** | 전처리 데이터, Selection Bias 진단 리포트 |

---

## Part A: 데이터 구축

### A-1. 데이터 획득

```bash
# make-ipinyou-data 실행
git clone https://github.com/wnzhang/make-ipinyou-data
cd make-ipinyou-data
./make-ipinyou-data.sh
```

### A-2. 데이터 파싱 및 통합

```
데이터 파일 구조:
├── bid.*.txt        # 전체 입찰 로그
├── imp.*.txt        # 노출 로그 (win=1)
├── clk.*.txt        # 클릭 로그
└── conv.*.txt       # 전환 로그

통합 방법:
1. bidid 기준으로 모든 로그 조인
2. win = 1 if bidid in imp.txt else 0
3. click = 1 if bidid in clk.txt else 0
4. conversion = 1 if bidid in conv.txt else 0
```

### A-3. 데이터 구조

```python
# 통합 데이터셋 스키마
D_all = {
    # 식별자
    'bidid': str,           # 입찰 ID
    'campaign': int,        # 캠페인 ID (2259, 2261, 2821, 2997, ...)

    # 컨텍스트
    'timestamp': int,       # Unix timestamp
    'hour': int,            # 0-23
    'weekday': int,         # 0-6
    'exchange': str,        # 'tanx', 'youku', etc.
    'slot_width': int,
    'slot_height': int,
    'slot_visibility': int,
    'slot_format': int,

    # 사용자
    'usertag': str,         # 사용자 태그 (comma-separated)
    'region': str,          # 지역 코드
    'city': str,

    # 입찰/가격
    'bid_price': float,     # 입찰가
    'floor_price': float,   # 최저가
    'market_price': float,  # 낙찰가 (win=1일 때만 의미)

    # 결과
    'win': int,             # 0 or 1
    'click': int,           # 0 or 1 (win=1일 때만 의미)
    'conversion': int       # 0 or 1 (click=1일 때만 의미)
}
```

### A-4. Train/Test Split 전략

```python
# 전략 1: S2=Train, S3=Test (권장 — Temporal Split)
# 근거:
# (1) Debiased 모델이 분포 변화 하에서도 robust함을 입증
# (2) S2→S3 KS=0.1294가 자연 drift baseline → MLOps drift threshold 설정 근거
# (3) 향후 retraining trigger 기준 수립
train = data[data['season'] == 2]         # S2: 106.6M bids
test  = data[data['season'] == 3]         # S3: 22.9M bids

# 전략 2: S2 내부 Temporal Split (ablation/튜닝용)
train_inner = s2[s2['day'] <= 14]         # Day 1-14 (~70%)
val         = s2[s2['day'].between(15, 17)] # Day 15-17 (~15%)
test_inner  = s2[s2['day'] >= 18]         # Day 18-21 (~15%)

# 중요: 동일 사용자가 train/test에 걸치는 것은 허용
# (시점 기준 분할이므로)
```

---

## Part B: 기초 EDA

### B-1. 캠페인별 통계

| Advertiser | Bids | Wins | Clicks | Conv | WR | CTR | CVR | Taxonomy |
|------------|------|------|--------|------|----|-----|-----|----------|
| 1458 | 29.4M | 6.15M | 4,900 | 2 | 20.93% | 0.08% | 0.04% | Branding |
| 3386 | 28.2M | 5.67M | 4,158 | 0 | 20.12% | 0.07% | 0% | Branding |
| 3427 | 28.1M | 5.16M | 3,834 | 0 | 18.40% | 0.07% | 0% | Branding |
| 3476 | 13.4M | 3.94M | 2,050 | 52 | 29.32% | 0.05% | 2.54% | Mixed |
| 2821 | 10.6M | 2.61M | 1,646 | 874 | 24.67% | 0.06% | 53.10% | Retargeting |
| 3358 | 7.5M | 3.46M | 2,734 | 754 | 46.06% | 0.08% | 27.58% | Retargeting |
| 2259 | 6.0M | 1.66M | 558 | 178 | 27.80% | 0.03% | 31.90% | Retargeting |
| 2261 | 4.3M | 1.37M | 414 | 0 | 31.72% | 0.03% | 0% | Branding |
| 2997 | 2.0M | 623K | 2,764 | 0 | 30.59% | 0.44% | 0% | Branding |
| **Total** | **129.5M** | **30.6M** | **23,058** | **1,860** | **23.67%** | **0.0752%** | **8.07%** | |

```python
# 캠페인별 기초 통계
campaign_stats = data.groupby('campaign').agg({
    'bidid': 'count',
    'win': 'sum',
    'click': 'sum',
    'conversion': 'sum',
    'bid_price': 'mean',
    'market_price': lambda x: x[data['win']==1].mean()
})
campaign_stats['win_rate'] = campaign_stats['win'] / campaign_stats['bidid']
campaign_stats['ctr'] = campaign_stats['click'] / campaign_stats['win']
campaign_stats['cvr'] = campaign_stats['conversion'] / campaign_stats['click']
```

### B-2. Win Rate 분포

```python
# 전체 win rate
overall_win_rate = data['win'].mean()

# 시간대별 win rate
hourly_win_rate = data.groupby('hour')['win'].mean()

# 거래소별 win rate
exchange_win_rate = data.groupby('exchange')['win'].mean()

# Bid price 구간별 win rate
data['bid_bucket'] = pd.cut(data['bid_price'], bins=20)
bid_win_rate = data.groupby('bid_bucket')['win'].mean()
```

### B-3. Bid vs Market Price 분포

```python
# Win=1인 경우만 market_price 의미 있음
won_data = data[data['win'] == 1]

# Bid-Market spread
won_data['spread'] = won_data['bid_price'] - won_data['market_price']
print(won_data['spread'].describe())

# 시각화
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(data['bid_price'], bins=50, alpha=0.7)
axes[0].set_title('Bid Price Distribution')

axes[1].hist(won_data['market_price'], bins=50, alpha=0.7)
axes[1].set_title('Market Price Distribution (Win=1)')

axes[2].scatter(won_data['bid_price'], won_data['market_price'], alpha=0.1)
axes[2].plot([0, 300], [0, 300], 'r--')
axes[2].set_title('Bid vs Market Price')
```

### B-4. CTR/CVR 분포

```python
# 캠페인별 CTR/CVR
campaign_ctr = data[data['win']==1].groupby('campaign')['click'].mean()
campaign_cvr = data[data['click']==1].groupby('campaign')['conversion'].mean()

# 시간대별 CTR
hourly_ctr = data[data['win']==1].groupby('hour')['click'].mean()

# 슬롯 크기별 CTR
slot_ctr = data[data['win']==1].groupby(['slot_width', 'slot_height'])['click'].mean()
```

### B-5. Data Quality & Anomalies (EDA Findings)

| 항목 | Finding | 영향 |
|------|---------|------|
| **IVT Screening** | 76 zero-win domains (7.16M bids, 5.5%), 741 zero-click domains | Win Tower 학습 시 noise, bid 자원 낭비 |
| **Pop-up (slotformat==5)** | CTR 0.86% (standard 대비 11.4x) | Misclick artifact → CTR Tower 오염 |
| **Visibility 255** | Exchange 1에서만 출현하는 sentinel value | 일반 numeric feature로 처리 시 왜곡 |
| **Temporal Drift** | S2→S3 market price KS=0.1294, CDF rightward shift | 모델 일반화 검증 필요 |
| **Geographic Concentration** | 36 regions, top 17 (47.2%) = 80% of bids | 희소 region 처리 필요 |
| **Domain Concentration** | 108K domains, top 238 (0.2%) = 80% of bids | 희소 domain grouping 필요 |
| **Floor Binding** | 32.24% of won bids에서 payprice ≈ floor price | Win probability에 mass point 생성 |
| **Hourly Win Rate** | U-shape: 새벽 43.06%, 오후 8.59% | 시간대별 경쟁 강도 극심한 차이 |
| **Exchange Win Rate** | 13.24% ~ 55.55% | Exchange별 floor 메커니즘 상이 |

### B-6. Data Quality Pipeline 권장사항 (SP0 → SP1 연결)

EDA findings를 기반으로 전처리 파이프라인에 다음 데이터 품질 처리를 권장:

| Feature/Flag | 정의 | 활용 |
|-------------|------|------|
| `domain_quality_tier` | zero_win / zero_click / normal (76 + 741 domains) | Win Tower: zero-win domain 제외/다운샘플링 |
| `is_popup` | slotformat == 5 | CTR Tower에서 제외 또는 별도 처리 (misclick artifact) |
| `is_visibility_unknown` | slotvisibility == 255 binary indicator | Exchange 1 sentinel value → 단순 binary indicator |
| `is_floor_binding` | payprice ≈ slotprice (won bids, 32.24%) | Dual-regime shading (SP3): floor-bound vs competitive |
| `domain_group` | Top-N domains + `other` | Domain concentration 처리 (top 50 ≈ 60% traffic) |
| `is_weekend` | weekday >= 5 | EDA 확인: 7-level weekday보다 Win/CTR 설명력 충분 |

---

## Part C: Selection Bias 사전 진단

> **핵심**: Win Selection Bias가 실제로 얼마나 심각한지 데이터로 진단

### C-1. Covariate Distribution Shift 분석

#### 시간에 따른 Win=1 샘플의 Covariate 분포 변화

```python
# Day 1-7 vs Day 15-21의 Win=1 샘플 비교
early_won = data[(data['day'] <= 7) & (data['win'] == 1)]
late_won = data[(data['day'] >= 15) & (data['win'] == 1)]

# 각 covariate의 분포 비교
from scipy.stats import ks_2samp

covariates = ['hour', 'exchange', 'region', 'slot_width', 'slot_height']
temporal_shift = {}

for cov in covariates:
    if data[cov].dtype == 'object':
        # Categorical: Chi-squared test
        early_dist = early_won[cov].value_counts(normalize=True)
        late_dist = late_won[cov].value_counts(normalize=True)
        # 간단히 TV distance 계산
        tv_distance = 0.5 * (early_dist - late_dist.reindex(early_dist.index, fill_value=0)).abs().sum()
        temporal_shift[cov] = {'type': 'categorical', 'tv_distance': tv_distance}
    else:
        # Numerical: KS test
        stat, pvalue = ks_2samp(early_won[cov], late_won[cov])
        temporal_shift[cov] = {'type': 'numerical', 'ks_stat': stat, 'pvalue': pvalue}
```

#### Win=1 vs Win=0 Covariate 비교

```python
# 각 feature에 대해 Win=1 vs Win=0 분포 비교
won = data[data['win'] == 1]
lost = data[data['win'] == 0]

covariate_shift = {}
for cov in ['hour', 'bid_price', 'floor_price', 'region']:
    if data[cov].dtype in ['int64', 'float64']:
        stat, pvalue = ks_2samp(won[cov].dropna(), lost[cov].dropna())
        covariate_shift[cov] = {'ks_stat': stat, 'pvalue': pvalue}

        # 시각화
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(won[cov], bins=50, alpha=0.5, density=True, label='Win=1')
        ax.hist(lost[cov], bins=50, alpha=0.5, density=True, label='Win=0')
        ax.legend()
        ax.set_title(f'{cov}: Win=1 vs Win=0 (KS={stat:.3f})')
```

#### 주요 검토 변수

| 변수 | 검토 이유 | 예상 bias 방향 |
|------|----------|---------------|
| **hour** | 시간대별 경쟁 강도 차이 | 경쟁 낮은 시간대 Win↑ |
| **exchange** | 거래소별 win rate 차이 | 특정 거래소 편향 |
| **region** | 지역별 경쟁 차이 | 특정 지역 편향 |
| **usertag** | 고가치 유저 편향 | 고가치 유저 Win↓ (경쟁↑) |
| **slot_size** | 광고 슬롯별 경쟁 차이 | 인기 슬롯 Win↓ |

### C-2. Win Rate 결정 요인 분석

```python
# Bid price와 Win의 관계
# 명확한 threshold가 존재하는지?

import statsmodels.api as sm

# 로지스틱 회귀로 win 결정 요인 분석
X = data[['bid_price', 'floor_price', 'hour', 'is_weekend']]
X = sm.add_constant(X)
y = data['win']

logit_model = sm.Logit(y, X).fit()
print(logit_model.summary())

# Bid/Floor 비율과 Win의 관계
data['bid_floor_ratio'] = data['bid_price'] / (data['floor_price'] + 1)
data['bid_floor_bucket'] = pd.cut(data['bid_floor_ratio'], bins=20)
ratio_win_rate = data.groupby('bid_floor_bucket')['win'].mean()
```

```python
# Market price 분포의 시간적 변화
for hour in [10, 14, 20]:
    hour_data = data[(data['hour'] == hour) & (data['win'] == 1)]
    print(f"Hour {hour}: Market price mean={hour_data['market_price'].mean():.2f}, "
          f"std={hour_data['market_price'].std():.2f}")
```

### C-3. Selection Bias로 인한 문제점 진단

| 문제 | 진단 방법 | 진단 코드 |
|------|----------|----------|
| **CTR 과대추정** | 전체 bid의 예상 CTR vs Win=1의 CTR | 아래 참조 |
| **고가치 유저 편향** | usertag 분포 비교 | 아래 참조 |
| **시간대 편향** | hour별 win rate vs CTR 상관관계 | 아래 참조 |
| **Covariate shift** | 시간별 covariate 분포 변화 | C-1 참조 |

```python
# CTR 과대추정 진단
# Naive CTR (Win=1만): 실제로 관측된 CTR
naive_ctr = data[data['win'] == 1]['click'].mean()

# 간단한 IPW CTR 추정 (bias 정량화)
# Step 1: Win propensity 추정
from sklearn.linear_model import LogisticRegression

X_prop = data[['bid_price', 'floor_price', 'hour']]
y_prop = data['win']
prop_model = LogisticRegression().fit(X_prop, y_prop)
p_win = prop_model.predict_proba(X_prop)[:, 1]

# Step 2: IPW로 전체 공간 CTR 추정 (Win=1 샘플만 사용)
won_mask = data['win'] == 1
ipw_weights = 1 / p_win[won_mask]
ipw_weights = np.clip(ipw_weights, 0.1, 10)  # Clipping
ipw_ctr = np.average(data.loc[won_mask, 'click'], weights=ipw_weights)

print(f"Naive CTR (Win=1 only): {naive_ctr:.4f}")
print(f"IPW-adjusted CTR: {ipw_ctr:.4f}")
print(f"Estimated bias: {naive_ctr - ipw_ctr:.4f}")
```

```python
# 고가치 유저 편향 진단
# Usertag 분포 비교 (Win=1 vs All)
def parse_usertags(tag_str):
    if pd.isna(tag_str):
        return []
    return [int(t) for t in tag_str.split(',') if t]

all_tags = data['usertag'].apply(parse_usertags).explode()
won_tags = data[data['win']==1]['usertag'].apply(parse_usertags).explode()

all_tag_dist = all_tags.value_counts(normalize=True)
won_tag_dist = won_tags.value_counts(normalize=True)

# 차이가 큰 태그 식별
tag_diff = (won_tag_dist - all_tag_dist.reindex(won_tag_dist.index, fill_value=0)).abs()
print("Most biased usertags:")
print(tag_diff.sort_values(ascending=False).head(10))
```

```python
# 시간대 편향 진단
# Win rate와 CTR의 시간대별 상관관계
hourly_stats = data.groupby('hour').agg({
    'win': 'mean',
    'click': lambda x: x[data.loc[x.index, 'win']==1].mean()
}).rename(columns={'win': 'win_rate', 'click': 'ctr'})

correlation = hourly_stats['win_rate'].corr(hourly_stats['ctr'])
print(f"Hourly win_rate-CTR correlation: {correlation:.3f}")

# 만약 음의 상관관계라면: 경쟁 낮은 시간 = Win 많음 = 낮은 CTR
# → Selection bias 존재 가능성
```

### C-4. Positivity 가정 검증

```
Positivity: P(Win=1 | X=x) > 0 for all x in support

진단:
- 각 covariate stratum별 win rate 계산
- Win rate = 0인 stratum 존재 여부
- Extreme propensity (win rate < 1% 또는 > 99%) 비율
```

```python
# Positivity 진단
# Stratified win rate 계산

# 예: hour × exchange stratum
strata_win_rate = data.groupby(['hour', 'exchange']).agg({
    'win': ['mean', 'count']
}).reset_index()
strata_win_rate.columns = ['hour', 'exchange', 'win_rate', 'count']

# 문제가 되는 stratum 식별
zero_win_strata = strata_win_rate[strata_win_rate['win_rate'] == 0]
extreme_win_strata = strata_win_rate[
    (strata_win_rate['win_rate'] < 0.01) | (strata_win_rate['win_rate'] > 0.99)
]

print(f"Zero win rate strata: {len(zero_win_strata)}")
print(f"Extreme win rate strata (<1% or >99%): {len(extreme_win_strata)}")
print(f"Total samples in extreme strata: {extreme_win_strata['count'].sum()}")

# Propensity 분포 확인
print("\nPropensity distribution:")
print(f"Win rate < 1%: {(p_win < 0.01).mean()*100:.1f}% of samples")
print(f"Win rate 1-5%: {((p_win >= 0.01) & (p_win < 0.05)).mean()*100:.1f}%")
print(f"Win rate 5-30%: {((p_win >= 0.05) & (p_win < 0.30)).mean()*100:.1f}%")
print(f"Win rate 30-50%: {((p_win >= 0.30) & (p_win < 0.50)).mean()*100:.1f}%")
print(f"Win rate > 50%: {(p_win >= 0.50).mean()*100:.1f}%")
```

**EDA 실제 결과 (LightGBM, clean — usertag leakage 제거 후):**

| 진단 항목 | 값 | 해석 |
|-----------|-----|------|
| Win PS AUC | ~0.91 | 높은 분리도 → strong selection |
| Overlap [0.1, 0.9] | ~46% | 약 절반만 common support |
| ESS ratio | ~7% | IPW weight 매우 불안정 |
| CTR Overestimation | +6.7% | Naive CTR가 실제보다 높게 추정 |

→ **IPW 단독 위험** → DR doubly robust + ESMM joint constraint + 30.6M won samples imputation 필수

### C-5. Bias 정량화 시도

```python
# Selection Bias 정량화
# Naive vs IPW estimator 차이

def estimate_selection_bias(data, outcome_col, propensity):
    """
    Selection Bias = E[Y | Win=1] - E[Y] (전체 공간)

    IPW로 전체 공간 기대값 추정
    """
    won_mask = data['win'] == 1

    # Naive: Win=1 샘플의 평균
    naive = data.loc[won_mask, outcome_col].mean()

    # IPW: 전체 공간 추정
    weights = 1 / propensity[won_mask]
    weights = np.clip(weights, 0.1, 10)  # Stabilization
    ipw = np.average(data.loc[won_mask, outcome_col], weights=weights)

    # Bias 추정
    bias = naive - ipw
    relative_bias = bias / naive if naive > 0 else 0

    return {
        'naive': naive,
        'ipw': ipw,
        'bias': bias,
        'relative_bias_pct': relative_bias * 100
    }

# CTR bias
ctr_bias = estimate_selection_bias(data, 'click', p_win)
print(f"CTR Selection Bias:")
print(f"  Naive CTR: {ctr_bias['naive']:.4f}")
print(f"  IPW CTR: {ctr_bias['ipw']:.4f}")
print(f"  Bias: {ctr_bias['bias']:.4f} ({ctr_bias['relative_bias_pct']:.1f}%)")
```

### C-6. Selection Bias 진단 요약 템플릿

```markdown
## Selection Bias 진단 결과

### 1. Covariate Shift 심각도
| Covariate | Metric | Value | 판정 |
|-----------|--------|-------|------|
| bid_floor_ratio | Cohen's d | 0.83 | **심각** — auction competition 주도 |
| adexchange | Cohen's d | -0.86 | **심각** — exchange별 win mechanism 상이 |
| hour | KS stat | ~0.05 | 경미 |
| region | KS stat | ~0.03 | 경미 |
| slot_size | KS stat | ~0.04 | 경미 |

### 2. Positivity 위반
- Overlap [0.1, 0.9]: **~46%** (약 절반만 common support)
- ESS ratio: **~7%** (IPW weight 매우 불안정)
- 주요 위반 영역: 높은 bid_floor_ratio (경쟁 약한 구간), 특정 exchange

### 3. Bias 정량화
| Metric | Naive (Win=1 only) | IPW-adjusted | Bias |
|--------|-------------------|--------------|------|
| CTR | 0.0752% | ~0.0705% | **+6.7%** overestimation |
| Subgroup CTR bias | — | — | -10% ~ +18% (Simpson's Paradox) |

### 4. 권장 Debiasing 전략
- [ ] ~~IPW: Propensity 기반 reweighting~~ (단독 위험 — ESS 7%)
- [x] **DR: Doubly Robust (IPW + Imputation)** → ESCM²-WC(DR) primary
- [x] **ESMM constraint**: Joint BCE로 propensity 의존 없는 추가 signal
- [x] Clipping: win_eps=0.05, max_weight=10.0
- [x] Self-normalized weights
```

---

## Part D: Feature Engineering

> **상세 레퍼런스**: [`docs/feature_dictionary.md`](../feature_dictionary.md) — 각 feature의 what/why/how, 설계 결정, leakage 경고 등

**Implementation**: `src/features/engineering.py` :: `engineer_features()`, `src/features/usertag.py`, `scripts/build_features.py`

### D-1. Time Features (7개)

`add_time_features()` — hour, minute, weekday, is_weekend, is_peak_hour, hour_sin, hour_cos

- **Why**: RTB 경쟁 강도와 사용자 행동이 시간대별로 다름 (EDA 결과 win rate 8.6%~43.1% 변동)
- **Cyclical encoding**: `sin(2*pi*hour/24)`, `cos(2*pi*hour/24)` — hour 23↔0 거리 보존
- **is_peak_hour**: 오전 7-9시, 저녁 17-20시 (출퇴근 러시아워)

### D-2. Slot Features (4개 + passthrough)

`add_slot_features()` — slot_area, slot_area_log, slot_aspect_ratio, slot_size_group

- **Why**: 광고 슬롯 크기/형태가 CTR과 경쟁 수준에 직접 영향
- **IAB 표준 크기 매핑**: `_SLOT_SIZE_MAP` 딕셔너리 기반 7개 그룹 (leaderboard, medium_rectangle, skyscraper, square, mobile, banner, other)
- **Vectorized merge**: `_SLOT_SIZE_LOOKUP` DataFrame으로 pre-build하여 130M row 성능 최적화 (Python tuple 생성 회피)
- **Log transform**: `slot_area` → `log1p` 변환 (right-skewed 분포 정규화)

### D-3. Region Features (3개+)

`add_region_features()`, `compute_region_stats()` — region_freq, region_group, region_ctr (optional)

- **Why**: 36개 지역별 경쟁 환경과 사용자 특성 차이 (상위 17개가 80% 차지)
- **Frequency encoding**: Count 기반 (target encoding 대신 leakage 방지)
- **Quantile grouping**: `pd.qcut(q=4)` → low/medium/high/very_high
- **region_ctr**: Training set에서 region별 historical CTR 계산, val/test에 merge (결측 = global mean)

### D-4. Competition Features (3개+)

`add_competition_features()`, `compute_market_stats()` — bid_floor_ratio, market_price_avg, market_price_std (optional)

- **Why**: 입찰 경쟁 강도가 win probability의 핵심 결정 요인
- **bid_floor_ratio**: `bidprice / slotprice` (입찰 공격성), [0, 100] clipping
- **Market stats**: Won bids의 `(adexchange, slot_size_group)`별 payprice 평균/표준편차, training set에서만 계산
- **Group key 변경**: 초기 설계(`hour x exchange`)에서 `(adexchange, slot_size_group)`으로 — 시장가는 슬롯 크기에 더 강하게 의존

### D-5. Usertag Features (sparse multi-hot)

`src/features/usertag.py` :: `build_vocab()`, `encode_multihot_sparse()`

- **Why**: 사용자 관심사 taxonomy → CTR/CVR 예측의 핵심 신호
- **Top-100 vocab**: Training set에서 min_count=10 이상 출현한 상위 100개 태그
- **Sparse CSR matrix**: `scipy.sparse` 형태로 별도 `.npz` 파일 저장 (메모리 효율)
- **Leakage 경고**: `n_tags`/`has_tags`는 win label과 상관 (bid log usertag null → win=1만 tags 보유) → 제거됨 (2026-02-12)

---

## Part E: iPinYou 데이터셋 한계

### E-1. Auction Type
- **iPinYou**: Second-price auction (낙찰가 = 차점 입찰가)
- **현재 시장**: First-price auction이 주류
- **영향**: Bid shading 전략 필요 (SP3에서 다룸)

### E-2. 데이터 연도
- **iPinYou**: 2013년 10월 데이터
- **현재**: 10년 이상 경과
- **영향**: 사용자 행동, 광고 형식, 경쟁 환경 변화

### E-3. User ID
- **iPinYou**: Cookie 기반 usertag
- **현재**: Privacy-preserving (Topics API, Privacy Sandbox)
- **영향**: User-level targeting 제한

### E-4. Win=0 데이터
- **iPinYou**: Win=0에서도 market_price 제공 (연구용)
- **실제 DSP**: Win=0은 대부분 market_price 미제공
- **영향**: Propensity 추정 난이도 증가

### E-5. Selection Bias 특성
- **연구용 데이터셋**: Bias가 실제보다 덜 심각할 수 있음
- **실제 DSP**: Win rate 5-10%로 더 극단적인 bias

---

## Part F: 이후 단계로의 시사점

### Selection Bias 진단 결과에 따른 권장사항

| 진단 결과 | 권장 Debiasing 전략 |
|----------|-------------------|
| KS stat > 0.1 | DR 필수, 적극적 clipping |
| Extreme propensity > 10% | Stabilized weights, Cross-fitting |
| Bias > 10% | DR-ESCM² 권장 |
| Positivity 위반 심각 | Trimming 또는 Stratification |

### Propensity 모델 설계 시 고려사항

1. **Feature 선택**: Bid price 반드시 포함
2. **Calibration**: IPW weight에 직접 영향
3. **Cross-fitting**: Overfitting 방지
4. **Monitoring**: Weight 분포 지속 모니터링

---

## 산출물

| 산출물 | 경로 | 설명 |
|--------|------|------|
| 전처리 데이터셋 | `data/processed/unified_dataset.parquet` | 통합 데이터 |
| Train/Val/Test | `data/processed/{train,val,test}.parquet` | 분할 데이터 |
| EDA 노트북 | `notebooks/00_eda.ipynb` | 기초 EDA |
| Selection Bias 진단 | `notebooks/00_selection_bias_diagnosis.ipynb` | Bias 진단 |
| Feature 정의서 | `docs/feature_dictionary.md` | Feature 설명 |
| Bias 진단 리포트 | `reports/selection_bias_report.md` | 진단 결과 |
