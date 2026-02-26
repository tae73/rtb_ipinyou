# SP3: 입찰 최적화

---

## 개요

| 항목 | 내용 |
|------|------|
| **목적** | First-price 입찰 전략 수립 및 Budget Pacing |
| **선행 조건** | SP1 (예측 모델), SP2 (Win Rate 분석) |
| **후속 단계** | SP4 (Causal Analysis), SP5 (Serving) |
| **핵심 산출물** | FirstPriceBidder, BudgetPacer, 시뮬레이션 결과 |

---

## SP1 → SP3 Connection

SP1의 산출물이 SP3 입찰 최적화의 핵심 입력:

```
SP1 산출물                              SP3 활용
─────────────                           ─────────
1. Debiased pCTR (ESCM²-WC(DR))       → V(x) = debiased_pCTR(x) × CPC_target
   ⚠ Biased prediction 사용 시          → 입찰가 왜곡 → ROI 저하
                                         (see 02_selection_bias_diagnosis: +6.7% overestimation)
   ⚠ CVR near-trivial (EDA 2.2.1)      → pCVR 무의미 → CPC campaign 기반

2. Win Tower (AUC ~0.91)              → Market price CDF 추정 → Bid shading factor
   ESCM²-WC의 Win Tower dual purpose:   → shade(x) = F_market(b) 기반 최적 입찰
   (a) CTR debiasing propensity
   (b) bid shading win rate model

3. 전체 입찰 공식:
   bid(x) = V(x) × shade(x) × pace(t)
           = [debiased_pCTR × CPC_target]     (CPC campaign)
             × [Win Tower market CDF shading]
             × [PID pacing multiplier]

   Optional (CPA campaign):
   bid(x) = debiased_pCTR × pCVR_naive × conv_value × shade × pace
```

**Bid→Win→Click Reframe 이후 변화:**
- Win Tower = ESCM²-WC의 내장 win rate model → 별도 학습 불필요
- V(x) = debiased_pCTR × CPC_target (CVR near-trivial → CPC 기반 단순화)
- Win PS AUC ~0.91 (clean, usertag leakage 제거) → shading 품질

**Market Price Context (EDA Findings):**
- Market price: median 70, mean 80, P90: 177, P95: 219 CPM
- Floor price: median 40, mean 45.47, binding 32.24%
- iPinYou flat-bid → ~76% overpayment (bid vs market price)
- Exchange별 floor 메커니즘 상이: Ex1 no floor, Ex2 moderate, Ex3 active floor

---

## Part A: Value 기반 입찰

### A-1. Value 계산

```python
def calculate_value(pctr, conv_value, goal_type='CPC', pcvr=None):
    """
    Bid의 기대 가치 계산

    Primary: V(x) = debiased_pCTR × CPC_target

    ⚠ 반드시 Debiased pCTR (ESCM²-WC(DR) 출력) 사용:
    - Biased pCTR → V(x) 왜곡 → +6.7% 과대 입찰
    - Debiased → unbiased V(x) → 정확한 ROI 최적화
    - see: 02_selection_bias_diagnosis, SP1 → SP3 Connection

    Campaign Goal별 Value 정의:
    - CPC: V = pCTR × CPC_target (primary, CVR near-trivial)
    - CPA: V = pCTR × pCVR_naive × CPA_target (optional, retargeting only)
    - CPM: V = CPM_target / 1000 (브랜딩)
    """
    if goal_type == 'CPC':
        # 클릭 가치 기반 (primary — CVR near-trivial)
        return pctr * conv_value
    elif goal_type == 'CPA':
        # 전환 가치 기반 (optional, retargeting only)
        if pcvr is None:
            raise ValueError("pcvr required for CPA goal")
        return pctr * pcvr * conv_value
    elif goal_type == 'CPM':
        # 노출 가치 기반 (브랜딩)
        return conv_value / 1000
    else:
        raise ValueError(f"Unknown goal type: {goal_type}")


# 예시
pctr = 0.01        # 1% CTR
pcvr = 0.05        # 5% CVR (click → conversion)
target_cpa = 50    # $50 per conversion

value = calculate_value(pctr, pcvr, target_cpa, 'CPA')
print(f"Value per impression: ${value:.4f}")
# 0.01 * 0.05 * 50 = $0.025
```

### A-2. 캠페인 목표별 Value 정의

| Goal | Value 계산 | 의미 | 비고 |
|------|-----------|------|------|
| **CPC** | pCTR × CPC_target | 클릭 기대 가치 | **Primary (CVR near-trivial)** |
| **CPA** | pCTR × pCVR × CPA_target | 전환 기대 가치 | Optional (retargeting only) |
| **ROAS** | pCTR × pCVR × expected_revenue | 수익 기대 가치 | Optional |
| **CPM** | fixed_cpm / 1000 | 고정 노출 가치 | Branding |

---

## Part B: First-Price Bid Shading

### B-1. Second-price vs First-price 차이

```
Second-price Auction (과거):
- 입찰: bid = value (truthful bidding이 최적)
- 지불: 2등 입찰가 (market_price)
- 전략: 솔직하게 가치만큼 입찰

First-price Auction (현재):
- 입찰: bid < value (shading 필요)
- 지불: 자신의 입찰가 그대로
- 전략: value보다 낮게 입찰해서 이윤 확보

First-price에서 truthful bidding의 문제:
- bid = value로 낙찰 시, 이윤 = 0
- 모든 거래에서 마진 없음
- → Bid Shading 필수
```

### B-2. Bid Shading 전략

```python
def calculate_shading_factor(value, market_dist, strategy='median'):
    """
    First-price Bid Shading Factor 계산

    전략 옵션:
    1. median: market price 중앙값 기준 shading
    2. percentile: target percentile 기준
    3. optimal: 최적 입찰 공식 (수치적 해)
    """
    if strategy == 'median':
        # 간단한 휴리스틱: value와 median market price 비교
        median_market = market_dist.get('median', 70)

        if value > median_market * 2:
            # Value가 매우 높으면 적극적 shading
            shade = 0.65
        elif value > median_market * 1.5:
            shade = 0.75
        elif value > median_market:
            shade = 0.85
        else:
            # Value가 낮으면 보수적 shading (win 확보 중요)
            shade = 0.95

        return shade

    elif strategy == 'percentile':
        # Target win rate에 해당하는 percentile 사용
        p75 = market_dist.get('p75', 150)
        shade = min(p75 / value, 0.95) if value > 0 else 0.9
        return shade

    elif strategy == 'optimal':
        # 최적 입찰 공식 사용 (아래 참조)
        return optimal_shading(value, market_dist)


def optimal_shading(value, market_dist):
    """
    최적 First-price 입찰

    이론:
    max_b E[utility] = (value - b) × P(win | b)

    FOC: P(win | b*) = (value - b*) × f(b*)
         where f는 market price의 PDF

    해석:
    - 입찰가 올리면 win prob 증가, 마진 감소
    - 최적점에서 두 효과 균형

    근사 (log-normal 가정):
    b* ≈ value × (1 - 1/(1 + σ²/μ))
    """
    mu = market_dist.get('mean', 80)
    sigma = market_dist.get('std', 40)

    if sigma == 0 or mu == 0:
        return 0.85  # fallback

    # Log-normal 근사
    coefficient_of_variation = sigma / mu
    shade = 1 - 1 / (1 + coefficient_of_variation ** 2)
    shade = max(0.5, min(shade, 0.95))  # 범위 제한

    return shade
```

### B-3. 수치적 최적 입찰 공식

```python
from scipy.optimize import minimize_scalar
from scipy.stats import norm

def optimal_bid_numerical(value, market_cdf, market_pdf, bounds=(0, None)):
    """
    수치적으로 최적 입찰가 계산

    max_b E[utility] = (v - b) × F(b)
    where F = CDF of market price

    FOC: F(b*) = (v - b*) × f(b*)
    """
    if bounds[1] is None:
        bounds = (0, value)

    def neg_expected_utility(b):
        """음의 기대 효용 (최소화용)"""
        if b <= 0 or b >= value:
            return 0
        win_prob = market_cdf(b)
        margin = value - b
        return -(margin * win_prob)

    result = minimize_scalar(
        neg_expected_utility,
        bounds=bounds,
        method='bounded'
    )

    return result.x


# 예시: 정규 분포 가정
def normal_market_price(mean=100, std=30):
    """정규 분포 market price"""
    def cdf(b):
        return norm.cdf(b, loc=mean, scale=std)

    def pdf(b):
        return norm.pdf(b, loc=mean, scale=std)

    return cdf, pdf

# 최적 입찰 계산
cdf, pdf = normal_market_price(mean=100, std=30)
value = 150

optimal_bid = optimal_bid_numerical(value, cdf, pdf)
shade_factor = optimal_bid / value

print(f"Value: {value}")
print(f"Optimal bid: {optimal_bid:.2f}")
print(f"Shading factor: {shade_factor:.3f}")
print(f"Expected win prob: {cdf(optimal_bid):.3f}")
print(f"Expected margin: {value - optimal_bid:.2f}")
```

### B-4. iPinYou가 Second-price인 한계

```
iPinYou 데이터셋의 제약:
- 2013년 데이터: Second-price auction
- 현재 시장: First-price가 주류

접근 방법:
1. Second-price 데이터로 Win Rate 곡선 추정
2. First-price 시뮬레이션 환경 구축
3. Bid shading 전략 테스트

주의:
- Second-price 데이터의 market_price ≠ First-price의 winning bid
- 경쟁 행태가 다를 수 있음
- 결과는 참고용으로만 활용

Limitation으로 명시 필요

**Exchange별 Floor 메커니즘 (EDA 확인):**
- Exchange 1: Floor price 거의 없음 → 순수 경쟁 환경
- Exchange 2: Moderate floor → floor 위 경쟁
- Exchange 3: Active floor (pseudo first-price 특성) → floor가 실질적 최저 지불가
- 32.24% floor binding → 시뮬레이션 시 exchange-conditional 처리 필요
```

### B-5. Dual-regime Shading (EDA-driven)

```
EDA Finding:
- 32.24% of won bids에서 payprice ≈ floor price (floor-binding)
- Floor-bound regime vs Competitive regime → 서로 다른 shading 전략 필요

Dual-regime 설계:
1. Floor-bound regime (is_floor_binding = 1):
   - Market price ≈ floor → shade to floor
   - bid = max(floor, value × 0.95)  (overpayment 최소화)
   - 불필요한 초과 입찰 방지

2. Competitive regime (is_floor_binding = 0):
   - Market price > floor → standard shading
   - bid = value × shade(market_dist)
   - 기존 optimal shading 적용

활용 feature: `is_floor_binding` (SP0 Data Quality Pipeline)
```

### B-6. Overpayment Diagnostic Metric (EDA-driven)

```
iPinYou Baseline:
- Flat-bid 전략 → 평균 ~76% overpayment
- overpayment_ratio = (bid - market_price) / market_price

Metric 정의:
overpayment_ratio = (bid - payprice) / payprice  (won bids)

비교 시나리오:
| 전략 | Overpayment Ratio |
|------|-------------------|
| Flat-bid (iPinYou 현행) | ~76% (baseline) |
| Value-based (no shading) | ~40-50% (추정) |
| **Shaded (dual-regime)** | **~15-25% (목표)** |

→ Bid shading 효과를 overpayment 감소로 정량화
```

---

## Part C: Budget Pacing

### C-1. Pacing의 필요성

```
Budget Pacing이 필요한 이유:

1. 예산 조기 소진 방지
   - 오전에 모든 예산 소진 → 오후 기회 손실

2. 트래픽 패턴 활용
   - 시간대별 CTR/CVR 다름
   - 최적 시간대에 예산 집중

3. 경쟁 환경 대응
   - 경쟁 높은 시간대 → 입찰 자제
   - 경쟁 낮은 시간대 → 적극 입찰

4. 학습 데이터 균형
   - 다양한 시간대/세그먼트 데이터 확보
```

### C-2. PID Controller 방식

```python
class PIDPacer:
    """
    PID Controller 기반 Budget Pacing

    목표: 예산 소진 속도를 이상적 속도에 맞춤

    PID 공식:
    u(t) = Kp × e(t) + Ki × ∫e(τ)dτ + Kd × de/dt

    여기서:
    - e(t) = ideal_spent - actual_spent (오차)
    - u(t) → pacing multiplier
    """

    def __init__(self, daily_budget, Kp=0.5, Ki=0.1, Kd=0.1,
                 multiplier_range=(0.5, 2.0)):
        self.daily_budget = daily_budget
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.multiplier_range = multiplier_range

        self.spent = 0
        self.integral_error = 0
        self.prev_error = 0
        self.history = []

    def get_pacing_multiplier(self, current_hour, total_hours=24):
        """
        현재 시점의 Pacing Multiplier 계산

        Returns:
            multiplier: 입찰가에 곱할 계수 (1.0 = 정상 속도)
        """
        # 이상적 소진율
        progress_ratio = current_hour / total_hours
        ideal_spent = self.daily_budget * progress_ratio

        # 오차 (양수 = 과소 소진, 음수 = 과다 소진)
        error = ideal_spent - self.spent

        # PID 계산
        self.integral_error += error
        derivative_error = error - self.prev_error
        self.prev_error = error

        pid_output = (
            self.Kp * error +
            self.Ki * self.integral_error +
            self.Kd * derivative_error
        )

        # Multiplier 계산 (정규화)
        multiplier = 1.0 + pid_output / max(self.daily_budget * 0.1, 1)
        multiplier = max(self.multiplier_range[0],
                        min(self.multiplier_range[1], multiplier))

        self.history.append({
            'hour': current_hour,
            'spent': self.spent,
            'ideal': ideal_spent,
            'error': error,
            'multiplier': multiplier
        })

        return multiplier

    def record_spend(self, amount):
        """지출 기록"""
        self.spent += amount

    def reset(self):
        """일일 리셋"""
        self.spent = 0
        self.integral_error = 0
        self.prev_error = 0
        self.history = []


# 사용 예시
pacer = PIDPacer(daily_budget=100000)

for hour in range(24):
    multiplier = pacer.get_pacing_multiplier(hour)
    print(f"Hour {hour}: multiplier={multiplier:.3f}, spent={pacer.spent:.0f}")

    # 시뮬레이션: 해당 시간의 지출
    hourly_spend = 4000 * multiplier * (1 + 0.2 * np.random.randn())
    pacer.record_spend(hourly_spend)
```

### C-3. Throttling 방식

```python
class ThrottlingPacer:
    """
    Throttling 기반 Budget Pacing

    원리:
    - 예산 초과 속도 시 → 일부 입찰 기회 건너뜀 (throttle)
    - 예산 부족 속도 시 → 모든 입찰 기회 참여

    장점: 입찰가 자체는 변경하지 않음 (shading과 분리)
    """

    def __init__(self, daily_budget, total_hours=24):
        self.daily_budget = daily_budget
        self.total_hours = total_hours
        self.spent = 0
        self.current_hour = 0

    def should_bid(self):
        """
        현재 입찰 기회에 참여할지 결정

        Returns:
            bool: True면 입찰, False면 skip
        """
        if self.current_hour >= self.total_hours:
            return self.spent < self.daily_budget

        # 남은 시간
        remaining_hours = self.total_hours - self.current_hour

        # 남은 예산
        remaining_budget = self.daily_budget - self.spent

        # 이상적 시간당 지출
        ideal_hourly = remaining_budget / max(remaining_hours, 1)

        # 현재 속도 추정 (최근 1시간 평균)
        current_rate = self._estimate_current_rate()

        if current_rate > ideal_hourly * 1.2:
            # 과다 지출 → 확률적 throttling
            throttle_prob = 1 - (ideal_hourly / current_rate)
            return np.random.random() > throttle_prob
        else:
            return True

    def _estimate_current_rate(self):
        """현재 지출 속도 추정"""
        if self.current_hour == 0:
            return self.daily_budget / self.total_hours
        return self.spent / self.current_hour

    def record_spend(self, amount):
        self.spent += amount

    def advance_hour(self):
        self.current_hour += 1
```

### C-4. 시간대별 예산 배분

```python
class HourlyBudgetAllocator:
    """
    시간대별 차등 예산 배분

    기반:
    - 시간대별 CTR/CVR 분석 (SP0 EDA)
    - 시간대별 경쟁 강도 (SP2 Win Rate)
    - 시간대별 전환 가치
    """

    def __init__(self, daily_budget, hourly_weights=None):
        self.daily_budget = daily_budget

        if hourly_weights is None:
            # 기본 가중치 (비즈니스 시간 중심)
            self.hourly_weights = self._default_weights()
        else:
            self.hourly_weights = hourly_weights

        # 정규화
        total_weight = sum(self.hourly_weights.values())
        self.hourly_budgets = {
            h: daily_budget * w / total_weight
            for h, w in self.hourly_weights.items()
        }

    def _default_weights(self):
        """
        기본 시간대별 가중치

        EDA U-shape hourly win rate 반영:
        - 새벽 (0-5): WR 30-43% (경쟁↓) → 높은 가중치 (효율적 예산 사용)
        - 오전 (6-11): WR 15-25% (경쟁↑) → 중간 가중치
        - 오후 (12-17): WR 8-15% (경쟁 최고) → 낮은 가중치
        - 저녁 (18-23): WR 15-30% (경쟁↓) → 중간-높은 가중치
        """
        weights = {}
        for h in range(24):
            if 0 <= h < 6:       # 새벽
                weights[h] = 0.3
            elif 6 <= h < 9:     # 출근
                weights[h] = 0.8
            elif 9 <= h < 12:    # 오전
                weights[h] = 1.2
            elif 12 <= h < 14:   # 점심
                weights[h] = 1.0
            elif 14 <= h < 18:   # 오후
                weights[h] = 1.3
            elif 18 <= h < 21:   # 저녁
                weights[h] = 1.5
            else:                # 밤
                weights[h] = 0.7
        return weights

    def get_hourly_budget(self, hour):
        """특정 시간의 예산"""
        return self.hourly_budgets.get(hour, self.daily_budget / 24)

    @classmethod
    def from_data(cls, data, daily_budget, value_col='conversion'):
        """
        데이터 기반 시간대별 가중치 계산

        가중치 = 시간대별 conversion rate × volume
        """
        hourly_stats = data.groupby('hour').agg({
            value_col: 'sum',
            'win': 'count'
        })
        hourly_stats['value_rate'] = hourly_stats[value_col] / hourly_stats['win']
        hourly_stats['weight'] = hourly_stats['value_rate'] * hourly_stats['win']

        # 정규화
        hourly_stats['weight'] = hourly_stats['weight'] / hourly_stats['weight'].sum()

        weights = hourly_stats['weight'].to_dict()
        return cls(daily_budget, weights)
```

### C-5. Hour-adaptive Budget Pacing (EDA-driven)

```
EDA Finding:
- Hourly win rate: U-shape (새벽 43.06% → 오후 8.59%)
- 경쟁 낮은 시간대 (새벽)에 동일 예산으로 더 많은 impressions 확보 가능

Hour-adaptive 전략:
1. 균등 pacing (baseline): 시간당 동일 예산 배분
2. WR-weighted pacing: hourly win rate prior 반영
   - 경쟁 낮은 시간대 (새벽): 예산 집중 → 높은 win rate × 낮은 cost
   - 경쟁 높은 시간대 (오후): 예산 절약 → 불필요한 overpayment 방지

PID Pacing 통합:
- PID Controller의 ideal_hourly 계산에 hourly win rate weight 반영
- ideal_spent = daily_budget × Σ(w_h / Σw) for h <= current_hour
- w_h = EDA 기반 시간대별 가중치 (win rate inverse proxy)

기대 효과:
- 동일 예산으로 win rate 5-10% 향상 (시뮬레이션 검증 필요)
- Overpayment 감소 (경쟁 높은 시간대 회피)
```

---

## Part D: 통합 입찰 함수

### D-1. 통합 Bidder 클래스

```python
class FirstPriceBidder:
    """
    통합 First-price Bidder

    bid(x) = V(x) × shade(x) × pace(t)

    구성:
    - pCTR 모델: value 계산 (ESCM²-WC(DR) debiased 출력 사용)
    - Market price 분포: shading factor (Win Tower 활용)
    - Budget pacer: pacing multiplier

    Win Tower dual purpose (ESCM²-WC):
    1. CTR debiasing: win propensity → DR/IPW weights
    2. Bid optimization: P(Win|X, bid) → market price CDF → shade(x)
       - Win PS AUC ~0.91 (clean)
       - shade(x) = optimal_bid(V(x), F_market) / V(x)
    """

    def __init__(self, pctr_model, market_estimator, pacer,
                 campaign_config, pcvr_model=None):
        self.pctr_model = pctr_model
        self.pcvr_model = pcvr_model  # Optional (CPA campaigns)
        self.market_estimator = market_estimator
        self.pacer = pacer
        self.campaign_config = campaign_config

    def bid(self, features, current_hour, floor_price):
        """
        입찰가 계산

        Args:
            features: 입찰 feature vector
            current_hour: 현재 시간
            floor_price: 최저 입찰가

        Returns:
            bid_price: 최종 입찰가
            debug_info: 디버깅 정보
        """
        debug_info = {}

        # 1. Value 계산
        pctr = self.pctr_model.predict(features)[0]
        conv_value = self.campaign_config.get('target_cpc', 10)
        goal_type = self.campaign_config.get('goal_type', 'CPC')

        pcvr = self.pcvr_model.predict(features)[0] if self.pcvr_model else None
        value = calculate_value(pctr, conv_value, goal_type, pcvr=pcvr)

        debug_info['pctr'] = pctr
        debug_info['pcvr'] = pcvr
        debug_info['value'] = value

        # 2. Bid Shading
        exchange = features.get('exchange', 'default')
        market_dist = self.market_estimator.get_distribution(current_hour, exchange)
        shade = calculate_shading_factor(value, market_dist, strategy='median')

        debug_info['shade'] = shade
        debug_info['market_median'] = market_dist.get('median', 0)

        # 3. Budget Pacing
        pace = self.pacer.get_pacing_multiplier(current_hour)

        debug_info['pace'] = pace

        # 4. 최종 입찰가
        bid = value * shade * pace

        # 5. 제약 조건
        bid = max(bid, floor_price)  # Floor 이상
        bid = max(bid, 0)            # 음수 방지

        # 예산 체크 (선택적)
        remaining_budget = self.pacer.daily_budget - self.pacer.spent
        if bid > remaining_budget:
            bid = 0  # 예산 부족 시 입찰 포기

        debug_info['final_bid'] = bid

        return bid, debug_info

    def on_win(self, winning_price):
        """낙찰 시 예산 차감"""
        self.pacer.record_spend(winning_price)
```

### D-2. 입찰 파이프라인

```python
class BiddingPipeline:
    """
    전체 입찰 파이프라인

    Flow:
    1. Bid request 수신
    2. Feature 조회/구성
    3. 입찰가 계산
    4. 제약 조건 검증
    5. Bid response 반환
    """

    def __init__(self, feature_store, bidder, config):
        self.feature_store = feature_store
        self.bidder = bidder
        self.config = config

    def process_bid_request(self, bid_request):
        """
        Bid request 처리

        Args:
            bid_request: {
                'bid_id': str,
                'user_id': str,
                'campaign_id': str,
                'hour': int,
                'exchange': str,
                'floor_price': float,
                ...
            }

        Returns:
            bid_response or None
        """
        # 1. Feature 조회
        user_features = self.feature_store.get_user_features(
            bid_request['user_id']
        )
        campaign_features = self.feature_store.get_campaign_features(
            bid_request['campaign_id']
        )

        # 2. Feature vector 구성
        features = {
            **bid_request,
            **user_features,
            **campaign_features
        }

        # 3. 입찰가 계산
        bid_price, debug_info = self.bidder.bid(
            features,
            bid_request['hour'],
            bid_request['floor_price']
        )

        # 4. 입찰 여부 결정
        if bid_price <= 0:
            return None

        # 5. Response 생성
        return {
            'bid_id': bid_request['bid_id'],
            'bid_price': bid_price,
            'campaign_id': bid_request['campaign_id'],
            'debug': debug_info
        }
```

---

## Part E: 실무적 문제

### E-1. 경쟁 반응 미반영 문제

```
시뮬레이션의 한계:

1. 정적 경쟁 가정
   - Market price 분포가 고정이라고 가정
   - 실제: 경쟁사도 전략 조정

2. 전략적 상호작용
   - 우리가 bid 올리면 → 경쟁사도 대응 → 균형점 이동
   - Game theory 필요

3. 시장 변화
   - 새로운 경쟁자 진입
   - 계절적 변동
   - 경제 상황 변화

해결책:
- 시뮬레이션은 참고용으로만 활용
- A/B 테스트로 실제 효과 검증
- 지속적 모니터링 및 조정
```

### E-2. Market Price 추정 불확실성

```
Market Price 추정의 불확실성:

1. Win=0 데이터 없음 (실제 환경)
   - iPinYou는 연구용으로 제공
   - 실제 DSP는 lose notification 제한적

2. 분포 변화
   - 시간에 따라 market price 분포 변화
   - 실시간 업데이트 필요

3. 세그먼트별 차이
   - 시간대, 거래소, 지역별로 다른 분포
   - 충분한 데이터 필요

해결책:
- Robust shading (보수적 추정)
- 온라인 학습 (streaming update)
- Fallback to default
```

### E-3. Budget 소진 패턴의 현실성

```
Budget Pacing의 현실적 고려:

1. 불규칙한 입찰 기회
   - QPS 변동
   - 특정 시간대 집중

2. 예측 불가능한 전환
   - 대형 전환 발생 시 예산 급감
   - 클릭 후 며칠 뒤 전환

3. 실시간 예산 업데이트
   - Win notification 지연
   - Consistency 문제

해결책:
- 보수적 예산 관리
- 버퍼 확보
- Eventual consistency 허용
```

---

## Part F: 시뮬레이션

### F-1. 입찰 시뮬레이션

```python
class BiddingSimulator:
    """입찰 전략 시뮬레이션"""

    def __init__(self, data, bidder, market_estimator):
        self.data = data
        self.bidder = bidder
        self.market_estimator = market_estimator

    def simulate(self, strategy='proposed'):
        """
        전략 시뮬레이션

        Args:
            strategy: 'proposed', 'truthful', 'fixed'

        Returns:
            results: 시뮬레이션 결과
        """
        results = {
            'bids': [],
            'wins': [],
            'clicks': [],
            'conversions': [],
            'spend': [],
            'revenue': []
        }

        for idx, row in self.data.iterrows():
            # 입찰가 계산
            if strategy == 'proposed':
                bid, _ = self.bidder.bid(row, row['hour'], row['floor_price'])
            elif strategy == 'truthful':
                # Truthful: value 그대로 입찰
                bid = row['value'] if 'value' in row else row['bid_price']
            elif strategy == 'fixed':
                # Fixed: 고정 입찰가
                bid = 100

            results['bids'].append(bid)

            # Win 여부 (market price 기준)
            # Note: iPinYou는 second-price이므로 시뮬레이션용
            win = bid >= row.get('market_price', row['floor_price'])
            results['wins'].append(win)

            if win:
                # First-price: bid 그대로 지불
                spend = bid
                results['spend'].append(spend)

                # Click & Conversion
                click = row.get('click', 0)
                conv = row.get('conversion', 0)
                results['clicks'].append(click)
                results['conversions'].append(conv)

                # Revenue (conversion value)
                if conv:
                    results['revenue'].append(row.get('conv_value', 50))
                else:
                    results['revenue'].append(0)
            else:
                results['spend'].append(0)
                results['clicks'].append(0)
                results['conversions'].append(0)
                results['revenue'].append(0)

        return pd.DataFrame(results)

    def compare_strategies(self, strategies=['proposed', 'truthful', 'fixed']):
        """전략 비교"""
        comparison = {}

        for strategy in strategies:
            results = self.simulate(strategy)

            comparison[strategy] = {
                'total_bids': len(results),
                'wins': results['wins'].sum(),
                'win_rate': results['wins'].mean(),
                'clicks': results['clicks'].sum(),
                'conversions': results['conversions'].sum(),
                'total_spend': results['spend'].sum(),
                'total_revenue': results['revenue'].sum(),
                'roi': results['revenue'].sum() / max(results['spend'].sum(), 1)
            }

        return pd.DataFrame(comparison).T
```

### F-2. 예상 시뮬레이션 결과

| 전략 | Win Rate | Conversions | Spend | ROI |
|------|----------|-------------|-------|-----|
| Fixed (100) | 25% | 1,000 | 500K | 1.0 |
| Truthful | 35% | 1,200 | 700K | 0.86 |
| **Proposed (DR+Shading)** | 28% | 1,150 | 450K | **1.28** |

---

## 산출물

| 산출물 | 경로 | 설명 |
|--------|------|------|
| FirstPriceBidder | `src/bidding/first_price_bidder.py` | 통합 입찰 클래스 |
| BudgetPacer | `src/bidding/budget_pacer.py` | PID/Throttling Pacer |
| BidShading | `src/bidding/bid_shading.py` | Shading 전략 |
| 시뮬레이션 결과 | `reports/03_simulation_results.md` | ROI 비교 |
| 분석 노트북 | `notebooks/03_bid_optimization.ipynb` | 실험 코드 |

---

## 핵심 요약

1. **Value 계산**: V(x) = debiased_pCTR × CPC_target (primary, CVR near-trivial)
2. **Bid Shading**: Win Tower P(Win|X, bid) → market price CDF → shade (0.7~0.95)
3. **Budget Pacing**: PID Controller로 예산 속도 조절
4. **통합 공식**: bid = V × shade × pace
5. **제약 조건**: floor_price, budget limit
6. **Win Tower dual purpose**: debiasing propensity + bid shading win rate
7. **Limitation**: 경쟁 반응 미반영, iPinYou = Second-price
