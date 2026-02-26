# SP2: Win Rate 분석

---

## 개요

| 항목 | 내용 |
|------|------|
| **목적** | Win Rate 곡선 추정, 입찰 탄력성 분석, Market Price 분포 추정 |
| **선행 조건** | SP0 (데이터 준비), SP1 (기본 예측 모델) |
| **후속 단계** | SP3 (Bid Optimization), SP5 (Serving) |
| **핵심 산출물** | Win Rate 곡선, 탄력성 추정치, Market Price 분포 |

---

## Part A: Win Rate 곡선 추정

### A-1. 비모수적 추정

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def nonparametric_win_rate(data, bid_col='bid_price', win_col='win',
                           n_bins=50, min_samples=100):
    """
    비모수적 Win Rate 추정

    WinRate(b) = #wins(b±δ) / #bids(b±δ)
    """
    # Bid price 구간 생성
    bins = np.linspace(data[bid_col].min(), data[bid_col].quantile(0.99), n_bins)
    data['bid_bucket'] = pd.cut(data[bid_col], bins=bins)

    # 구간별 win rate 계산
    win_rate = data.groupby('bid_bucket', observed=True).agg({
        win_col: ['mean', 'sum', 'count']
    })
    win_rate.columns = ['win_rate', 'wins', 'total']

    # 샘플 수 충분한 구간만 사용
    win_rate = win_rate[win_rate['total'] >= min_samples]

    # Confidence interval (Wilson score)
    win_rate['ci_lower'], win_rate['ci_upper'] = wilson_ci(
        win_rate['wins'], win_rate['total']
    )

    return win_rate


def wilson_ci(successes, total, z=1.96):
    """Wilson score confidence interval"""
    p = successes / total
    n = total

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator

    return center - margin, center + margin


# 전체 데이터에 대한 Win Rate 곡선
win_rate_curve = nonparametric_win_rate(data)

# 시각화
fig, ax = plt.subplots(figsize=(10, 6))
bid_midpoints = win_rate_curve.index.map(lambda x: x.mid)
ax.plot(bid_midpoints, win_rate_curve['win_rate'], 'b-', label='Win Rate')
ax.fill_between(bid_midpoints, win_rate_curve['ci_lower'],
                win_rate_curve['ci_upper'], alpha=0.2)
ax.set_xlabel('Bid Price')
ax.set_ylabel('Win Rate')
ax.set_title('Win Rate Curve (Non-parametric)')
ax.legend()
```

### A-2. 로지스틱 회귀 추정

```python
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

def logistic_win_rate(data, features=['log_bid', 'log_floor', 'hour']):
    """
    로지스틱 회귀 기반 Win Rate 모델

    P(win) = Λ(β₀ + β₁·log(bid) + γ'X)
    """
    # 데이터 준비
    data['log_bid'] = np.log(data['bid_price'] + 1)
    data['log_floor'] = np.log(data['floor_price'] + 1)

    X = data[features]
    X = sm.add_constant(X)
    y = data['win']

    # 로지스틱 회귀 적합
    model = sm.Logit(y, X).fit(disp=0)

    return model


# 모델 적합
win_model = logistic_win_rate(data)
print(win_model.summary())

# 계수 해석
# β_log_bid: log(bid) 1 증가 (bid ~2.7배) → 낙찰 log-odds β_log_bid 증가
print("\nCoefficient Interpretation:")
for param, coef in win_model.params.items():
    if param == 'log_bid':
        print(f"  {param}: bid 1% 증가 → win odds {np.exp(coef/100)-1:.4f}% 증가")
    else:
        print(f"  {param}: {coef:.4f}")
```

### A-3. 세그먼트별 Win Rate 곡선

```python
def segment_win_rate_curves(data, segment_col, **kwargs):
    """세그먼트별 Win Rate 곡선"""
    segments = data[segment_col].unique()
    curves = {}

    for seg in segments:
        seg_data = data[data[segment_col] == seg]
        if len(seg_data) > 1000:  # 충분한 샘플
            curves[seg] = nonparametric_win_rate(seg_data, **kwargs)

    return curves

# 시간대별
hourly_curves = segment_win_rate_curves(data, 'hour')

# 거래소별
exchange_curves = segment_win_rate_curves(data, 'exchange')

# 캠페인별
campaign_curves = segment_win_rate_curves(data, 'campaign')

# 시각화: 시간대별 Win Rate 곡선 비교
fig, ax = plt.subplots(figsize=(12, 6))
for hour in [10, 14, 20]:
    if hour in hourly_curves:
        curve = hourly_curves[hour]
        bid_midpoints = curve.index.map(lambda x: x.mid)
        ax.plot(bid_midpoints, curve['win_rate'], label=f'Hour {hour}')

ax.set_xlabel('Bid Price')
ax.set_ylabel('Win Rate')
ax.set_title('Win Rate by Hour')
ax.legend()
```

---

## Part B: 입찰 탄력성 분석

### B-1. 탄력성 정의

```
탄력성 (Elasticity):
ε = ∂log(P(win)) / ∂log(bid)
  = (∂P/P) / (∂b/b)
  = bid의 1% 변화에 대한 win probability의 % 변화

해석:
- ε > 1: 탄력적 (bid 변화에 민감)
- ε < 1: 비탄력적 (bid 변화에 둔감)
- ε = 0: 완전 비탄력적 (bid 무관)
```

### B-2. 탄력성 추정

```python
def estimate_elasticity(win_model, data):
    """
    로지스틱 모델에서 탄력성 추정

    로지스틱 모델: P(win) = Λ(β₀ + β₁·log(bid) + ...)
    탄력성: ε = β₁ × (1 - P(win))

    Note: 로지스틱에서 marginal effect는 P(1-P)에 비례
    """
    data['log_bid'] = np.log(data['bid_price'] + 1)
    data['log_floor'] = np.log(data['floor_price'] + 1)

    X = data[['log_bid', 'log_floor', 'hour']]
    X = sm.add_constant(X)

    # Win probability
    p_win = win_model.predict(X)

    # 탄력성
    beta_log_bid = win_model.params['log_bid']
    elasticity = beta_log_bid * (1 - p_win)

    return elasticity


# 전체 탄력성
elasticity = estimate_elasticity(win_model, data)
print(f"Mean elasticity: {elasticity.mean():.4f}")
print(f"Std elasticity: {elasticity.std():.4f}")

# 세그먼트별 탄력성
data['elasticity'] = elasticity
segment_elasticity = data.groupby('campaign')['elasticity'].agg(['mean', 'std'])
print("\nElasticity by Campaign:")
print(segment_elasticity)
```

### B-3. 탄력성 기반 인사이트

```python
# 탄력성 분포 분석
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 전체 분포
axes[0].hist(elasticity, bins=50, alpha=0.7)
axes[0].axvline(elasticity.mean(), color='r', linestyle='--')
axes[0].set_title('Elasticity Distribution')
axes[0].set_xlabel('Elasticity')

# Win probability별 탄력성
data['win_prob'] = win_model.predict(X)
data['win_prob_bucket'] = pd.cut(data['win_prob'], bins=10)
elasticity_by_prob = data.groupby('win_prob_bucket', observed=True)['elasticity'].mean()
axes[1].bar(range(len(elasticity_by_prob)), elasticity_by_prob.values)
axes[1].set_title('Elasticity by Win Probability')
axes[1].set_xlabel('Win Probability Bucket')

# 시간대별 탄력성
hourly_elasticity = data.groupby('hour')['elasticity'].mean()
axes[2].plot(hourly_elasticity.index, hourly_elasticity.values, 'o-')
axes[2].set_title('Elasticity by Hour')
axes[2].set_xlabel('Hour')

plt.tight_layout()

# 비즈니스 인사이트
print("\n=== Business Insights ===")
print(f"1. 평균 탄력성 {elasticity.mean():.2f}: bid 1% 증가 → win rate {elasticity.mean():.2f}% 증가")

low_elasticity_segments = data.groupby('campaign')['elasticity'].mean().sort_values().head(3)
print(f"\n2. 낮은 탄력성 캠페인 (bid 증가 효과 적음):")
for camp, elast in low_elasticity_segments.items():
    print(f"   Campaign {camp}: ε = {elast:.3f}")

high_elasticity_hours = data.groupby('hour')['elasticity'].mean().sort_values(ascending=False).head(5)
print(f"\n3. 높은 탄력성 시간대 (bid 증가 효과 큼):")
for hour, elast in high_elasticity_hours.items():
    print(f"   Hour {hour}: ε = {elast:.3f}")
```

---

## Part C: Market Price 분포 추정

### C-0. EDA 기반 Market Price 실측치 (신규)

| Metric | 전체 (S2+S3) | 비고 |
|--------|-------------|------|
| **Market price median** | 70 CPM | Won bids |
| **Market price mean** | 80 CPM | Right-skewed |
| **Market price P90** | 177 CPM | |
| **Market price P95** | 219 CPM | |
| **Floor price median** | 40 CPM | |
| **Floor price mean** | 45.47 CPM | |
| **Floor binding** | 32.24% | payprice ≈ floor |
| **Overpayment** | ~76% | (bid - market) / market |

**Exchange별 특성:**

| Exchange | Win Rate | Floor 특성 | Price 분포 |
|----------|----------|-----------|-----------|
| Ex1 | Low (~13%) | No floor | 순수 경쟁, 변동 큼 |
| Ex2 | Medium | Moderate | Floor 위 경쟁 |
| Ex3 | High (~56%) | Active floor | Pseudo first-price, floor 근처 집중 |

**Exchange-conditional CDF 권장:**
- 단일 CDF 대신 exchange-conditional CDF 추정 (exchange별 price 분포 상이)
- Ex3: Floor-truncated 분포 모델 (floor 근처 mass point 처리)
- `estimate_market_price_survival()`에 exchange 기준 분리 적용

### C-1. 문제 정의

```
Market Price 추정의 어려움:

- Win=1: market_price 관측 (실제 낙찰가)
- Win=0: market_price 미관측 (bid < market_price라는 것만 알 수 있음)

→ Censored data 문제!

접근 방법:
1. Win=1 데이터만 사용 (biased)
2. Survival Analysis 기반 추정 (unbiased)
3. Parametric 분포 가정 후 MLE
```

### C-2. Survival Analysis 기반 추정

```python
from lifelines import KaplanMeierFitter
from lifelines import WeibullFitter

def estimate_market_price_survival(data):
    """
    Survival Analysis로 Market Price 분포 추정

    설정:
    - "Event time" = market_price (Win=1) 또는 bid_price (Win=0, censored)
    - "Event" = Win indicator
    - Survival S(p) = P(market_price > p) = 1 - CDF(p)
    - P(win | bid=b) = 1 - S(b) = CDF(b)
    """
    # Duration: 관측된 가격 (market_price if win, bid_price if lose)
    durations = np.where(
        data['win'] == 1,
        data['market_price'],
        data['bid_price']
    )

    # Event: win indicator
    events = data['win']

    # Kaplan-Meier 추정
    km = KaplanMeierFitter()
    km.fit(durations, event_observed=events)

    return km


# Market Price 분포 추정
km_model = estimate_market_price_survival(data)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Survival function
km_model.plot_survival_function(ax=axes[0])
axes[0].set_title('Market Price Survival Function\nS(p) = P(market_price > p)')
axes[0].set_xlabel('Price')
axes[0].set_ylabel('P(market_price > p)')

# CDF (= Win Rate given bid)
bid_range = np.linspace(0, data['bid_price'].quantile(0.99), 100)
win_prob = 1 - km_model.survival_function_at_times(bid_range).values.flatten()
axes[1].plot(bid_range, win_prob)
axes[1].set_title('P(Win | bid) = CDF of Market Price')
axes[1].set_xlabel('Bid Price')
axes[1].set_ylabel('P(Win)')

plt.tight_layout()
```

### C-3. 세그먼트별 Market Price 분포

```python
def segment_market_price_dist(data, segment_col):
    """세그먼트별 Market Price 분포"""
    segments = data[segment_col].unique()
    distributions = {}

    for seg in segments:
        seg_data = data[data[segment_col] == seg]
        if len(seg_data) > 1000:
            km = estimate_market_price_survival(seg_data)

            # 주요 통계량 추출
            # Median market price (survival = 0.5인 점)
            try:
                median = km.median_survival_time_
            except:
                median = np.nan

            distributions[seg] = {
                'model': km,
                'median': median,
                'n_samples': len(seg_data),
                'win_rate': seg_data['win'].mean()
            }

    return distributions


# 시간대별 Market Price 분포
hourly_dist = segment_market_price_dist(data, 'hour')

# 결과 요약
print("Market Price by Hour:")
for hour, dist in sorted(hourly_dist.items()):
    print(f"  Hour {hour}: median={dist['median']:.1f}, "
          f"win_rate={dist['win_rate']:.3f}")

# 거래소별
exchange_dist = segment_market_price_dist(data, 'exchange')
print("\nMarket Price by Exchange:")
for ex, dist in exchange_dist.items():
    print(f"  {ex}: median={dist['median']:.1f}, win_rate={dist['win_rate']:.3f}")
```

### C-4. Parametric 분포 추정

```python
from lifelines import WeibullFitter, LogNormalFitter, ExponentialFitter
from scipy import stats

def parametric_market_price(data, distribution='weibull'):
    """
    Parametric 분포로 Market Price 추정

    옵션:
    - weibull: Weibull 분포 (유연함)
    - lognormal: Log-normal 분포 (가격 데이터에 적합)
    - exponential: 지수 분포 (단순)
    """
    durations = np.where(
        data['win'] == 1,
        data['market_price'],
        data['bid_price']
    )
    events = data['win']

    if distribution == 'weibull':
        fitter = WeibullFitter()
    elif distribution == 'lognormal':
        fitter = LogNormalFitter()
    elif distribution == 'exponential':
        fitter = ExponentialFitter()

    fitter.fit(durations, event_observed=events)

    return fitter


# 분포 비교
distributions = {}
for dist_name in ['weibull', 'lognormal', 'exponential']:
    try:
        fitter = parametric_market_price(data, dist_name)
        distributions[dist_name] = {
            'model': fitter,
            'aic': fitter.AIC_,
            'bic': fitter.BIC_
        }
    except:
        pass

# 최적 분포 선택 (AIC 기준)
best_dist = min(distributions.items(), key=lambda x: x[1]['aic'])
print(f"Best distribution: {best_dist[0]} (AIC={best_dist[1]['aic']:.2f})")
```

---

## Part D: 실무적 고려사항

### D-1. Win=0 샘플의 Market Price 미제공 시 대안

```
실제 DSP 환경:
- Win=0 샘플: bid price만 알고 market price 모름
- Win=1 샘플: market price (낙찰가) 알 수 있음

iPinYou 데이터셋의 특수성:
- Win=0에서도 market price 제공 (연구용이라 가능)
- 실제 환경에서는 불가능

대안 접근법:
1. Win=1 데이터만으로 분포 추정 (biased)
2. Censored regression: Win=0은 bid가 lower bound
3. Survival analysis: 위에서 다룬 방법
4. 거래소별 aggregate 통계 활용
```

### D-2. 시간대/거래소별 분리 추정

```python
class MarketPriceEstimator:
    """
    세그먼트별 Market Price 분포 관리

    Serving 시 해당 세그먼트의 분포 파라미터 사용
    """

    def __init__(self):
        self.distributions = {}

    def fit(self, data, segment_cols=['hour', 'exchange']):
        """세그먼트별 분포 학습"""
        for (hour, exchange), group in data.groupby(segment_cols):
            if len(group) > 500:
                key = f"{hour}_{exchange}"
                self.distributions[key] = self._fit_single(group)

        # 전체 데이터로 fallback 분포
        self.distributions['_default'] = self._fit_single(data)

    def _fit_single(self, data):
        """단일 세그먼트 분포 추정"""
        won_data = data[data['win'] == 1]
        return {
            'mean': won_data['market_price'].mean(),
            'std': won_data['market_price'].std(),
            'median': won_data['market_price'].median(),
            'p25': won_data['market_price'].quantile(0.25),
            'p75': won_data['market_price'].quantile(0.75)
        }

    def get_distribution(self, hour, exchange):
        """세그먼트별 분포 조회"""
        key = f"{hour}_{exchange}"
        return self.distributions.get(key, self.distributions['_default'])

    def predict_win_prob(self, bid, hour, exchange):
        """
        Bid에 대한 Win probability 예측

        간단한 근사: CDF(bid) ≈ P(market_price < bid)
        정규 분포 가정
        """
        dist = self.get_distribution(hour, exchange)
        # 정규 분포 CDF
        z = (bid - dist['mean']) / dist['std']
        return stats.norm.cdf(z)
```

### D-3. Hourly Win Rate as Modeling Dimension (EDA-driven, 신규)

```
EDA Finding:
- Hourly win rate: U-shape, 8.59% (오후) ~ 43.06% (새벽)
- Win rate 곡선이 hour 축으로도 크게 변동

권장:
- Win rate 곡선을 hour 축으로도 추정: WR(bid, hour)
- hour × bid 2D win rate surface 검토
- segment_win_rate_curves(data, 'hour')를 Serving lookup에 hour dimension 추가

시간대별 win rate surface 활용:
1. Bid shading: 시간대별 최적 shade factor 차등 적용
2. Budget pacing: 경쟁 강도(≈1/WR) 기반 pacing weight
3. Win Tower validation: 예측 win rate vs 실측 hourly win rate 비교
```

### D-4. Win Rate 곡선 Serving

```python
# Serving을 위한 Win Rate 조회 테이블 생성
def create_win_rate_lookup(data, bid_bins=100, segments=['hour', 'exchange']):
    """
    Bid price × Segment → Win Rate 조회 테이블

    Serving 시 실시간 계산 대신 테이블 조회
    """
    # Bid price 구간화
    bid_range = np.linspace(0, data['bid_price'].quantile(0.99), bid_bins)

    lookup = {}
    for segment_values, group in data.groupby(segments):
        key = '_'.join(map(str, segment_values))

        # 해당 세그먼트의 win rate 곡선
        win_rates = []
        for i in range(len(bid_range) - 1):
            mask = (group['bid_price'] >= bid_range[i]) & \
                   (group['bid_price'] < bid_range[i+1])
            if mask.sum() > 0:
                win_rates.append(group.loc[mask, 'win'].mean())
            else:
                win_rates.append(np.nan)

        lookup[key] = {
            'bid_bins': bid_range[:-1],
            'win_rates': np.array(win_rates)
        }

    return lookup


# 조회 함수
def lookup_win_rate(lookup, bid, hour, exchange):
    """Win Rate 조회"""
    key = f"{hour}_{exchange}"
    if key not in lookup:
        key = '_default'

    table = lookup[key]
    idx = np.searchsorted(table['bid_bins'], bid) - 1
    idx = max(0, min(idx, len(table['win_rates']) - 1))

    return table['win_rates'][idx]
```

---

## Part E: 분석 결과 활용

### E-1. 입찰 전략 수립 인풋

```
Win Rate 분석 → SP3 (Bid Optimization) 인풋:

1. Market Price 분포 → Bid Shading 전략
   - 중앙값 대비 입찰가 조정
   - 세그먼트별 shading factor

2. 탄력성 → 입찰가 민감도 분석
   - 탄력성 높은 세그먼트: bid 조정 효과 큼
   - 탄력성 낮은 세그먼트: bid 외 요인 중요

3. Win Rate 곡선 → ROI 최적화
   - Target win rate 설정
   - 해당 win rate 달성 위한 bid 계산

4. Temporal Drift 시사점 (EDA)
   - S2→S3 market price KS=0.1294 (CDF rightward shift)
   - Win rate 곡선의 시간 안정성 검증 필요 (S2 vs S3 비교)
   - Drift 보정: S3 데이터로 win rate lookup 갱신 or 온라인 업데이트
```

### E-2. Feature Store 저장

```python
# Redis에 저장할 Market Price 통계
def prepare_market_price_features(estimator, feature_store):
    """Market Price 통계를 Feature Store에 저장"""
    for key, dist in estimator.distributions.items():
        if key == '_default':
            continue

        hour, exchange = key.split('_')
        feature_store.set(
            f"market_dist:{hour}:{exchange}",
            {
                'median': dist['median'],
                'mean': dist['mean'],
                'p25': dist['p25'],
                'p75': dist['p75']
            }
        )
```

### E-3. S2 vs S3 Win Rate Curve 안정성 검증 (신규)

```
EDA Finding:
- S2→S3 temporal drift: KS D=0.1294 (market price)
- Market price CDF의 rightward shift (S3에서 경쟁 심화)

검증 방법:
1. S2, S3 각각에 대해 win rate 곡선 추정
2. 동일 bid price에서의 win rate 차이 계산
3. 차이가 유의하면 drift 보정 필요

보정 방안:
- Recalibration: S3 데이터로 win rate lookup 갱신
- Online update: Streaming 방식으로 분포 파라미터 업데이트
- Robust shading: S2/S3 평균 분포 사용 (보수적)

MLOps 연결:
- KS D=0.1294를 자연 drift baseline으로 활용
- SP5 Drift Monitoring threshold: D>0.10 warning, D>0.15 critical
```

---

## 산출물

| 산출물 | 경로 | 설명 |
|--------|------|------|
| Win Rate 곡선 | `reports/figures/02_win_rate_curves.png` | 세그먼트별 곡선 |
| 탄력성 분석 | `reports/02_elasticity.md` | 탄력성 추정치 |
| Market Price 분포 | `models/market_price_estimator.pkl` | 분포 추정 결과 |
| Win Rate Lookup | `data/processed/win_rate_lookup.pkl` | Serving용 테이블 |
| 분석 노트북 | `notebooks/02_win_rate_analysis.ipynb` | 분석 코드 |

---

## 핵심 요약

1. **Win Rate 곡선**: 비모수적 + 로지스틱 모델로 추정
2. **탄력성**: bid 변화에 대한 win rate 민감도 분석
3. **Market Price**: Survival analysis로 censored data 처리
4. **세그먼트별 분리**: 시간대, 거래소별 다른 경쟁 환경 반영
5. **Serving 준비**: Lookup 테이블로 실시간 조회 지원
