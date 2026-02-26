# SP4: CATE / SCM / 정책 시뮬레이션

---

## 개요

| 항목 | 내용 |
|------|------|
| **목적** | 인과 분석 및 정책 평가 |
| **선행 조건** | SP1 (예측 모델), SP3 (입찰 최적화) |
| **후속 단계** | SP5 (Serving) |
| **핵심 산출물** | CATE 분석, DAG, 정책 시뮬레이션 결과 |

---

## Part A: CATE 분석

### A-1. 문제 정의

```
CATE (Conditional Average Treatment Effect):
τ(x) = E[Y(1) - Y(0) | X = x]
     = 처치 효과가 특성 X에 따라 어떻게 달라지는가?

RTB에서의 적용:
- Treatment T: log(bid_price)
- Outcome Y: win, click (conversion은 near-trivial → 제외 권장)
- Covariates X: user, context, campaign features
- CATE: 세그먼트별 입찰 효과 차이

EDA 수치:
- Win rate: 23.67% (hourly 8.59%~43.06%, exchange 13.24%~55.55%)
- CTR: 0.0752%, CVR: 8.07% (of clicks)
- 9 advertisers: Branding 5, Retargeting 3, Mixed 1

질문:
"어떤 세그먼트에서 bid 증가가 더 효과적인가?"
```

### A-2. CausalForestDML 설정

```python
from econml.dml import CausalForestDML
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

# 데이터 준비
# Y: 결과 변수 (win, click, conversion)
# T: 처치 변수 (log_bid)
# X: 이질성 변수 (CATE가 이에 따라 달라짐)
# W: 교란 변수 (T와 Y 모두에 영향)

Y = data['win'].values  # 또는 'click', 'conversion'
T = np.log(data['bid_price'].values + 1)  # 연속 처치
X = data[['campaign', 'hour', 'exchange', 'region']].values  # 이질성
W = data[['floor_price', 'slot_area', 'avg_market_price']].values  # 교란

# CausalForestDML 설정
forest = CausalForestDML(
    model_y=LGBMRegressor(n_estimators=100, num_leaves=31, verbose=-1),
    model_t=LGBMRegressor(n_estimators=100, num_leaves=31, verbose=-1),
    discrete_treatment=False,  # 연속 처치
    n_estimators=1000,
    honest=True,  # Honest splitting
    inference=True,  # 신뢰구간 계산
    random_state=42
)

# 학습
forest.fit(Y=Y, T=T, X=X, W=W)
```

### A-3. CATE 추출 및 분석

```python
# CATE 추정
cate = forest.effect(X)  # 각 샘플의 CATE
cate_ci = forest.effect_interval(X, alpha=0.05)  # 95% CI

# ATE (Average Treatment Effect)
ate = forest.ate(X)
ate_ci = forest.ate_interval(X, alpha=0.05)

print(f"ATE: {ate:.4f}")
print(f"95% CI: [{ate_ci[0]:.4f}, {ate_ci[1]:.4f}]")
print(f"해석: log(bid) 1 증가 → win rate {ate*100:.2f}%p 증가")

# CATE 분포
print(f"\nCATE Distribution:")
print(f"  Mean: {cate.mean():.4f}")
print(f"  Std: {cate.std():.4f}")
print(f"  Min: {cate.min():.4f}")
print(f"  Max: {cate.max():.4f}")
```

### A-4. 세그먼트별 CATE 분석

```python
# 데이터에 CATE 추가
data['cate'] = cate

# 캠페인별 CATE
campaign_cate = data.groupby('campaign')['cate'].agg(['mean', 'std', 'count'])
print("\nCATE by Campaign:")
print(campaign_cate.sort_values('mean', ascending=False))

# 시간대별 CATE
hourly_cate = data.groupby('hour')['cate'].mean()

# 시각화
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# CATE 분포
axes[0].hist(cate, bins=50, alpha=0.7)
axes[0].axvline(cate.mean(), color='r', linestyle='--', label='Mean')
axes[0].set_title('CATE Distribution')
axes[0].set_xlabel('CATE')
axes[0].legend()

# 캠페인별 CATE
campaign_cate['mean'].plot(kind='bar', ax=axes[1])
axes[1].set_title('CATE by Campaign')
axes[1].set_xlabel('Campaign')
axes[1].set_ylabel('Mean CATE')

# 시간대별 CATE
axes[2].plot(hourly_cate.index, hourly_cate.values, 'o-')
axes[2].set_title('CATE by Hour')
axes[2].set_xlabel('Hour')
axes[2].set_ylabel('Mean CATE')

plt.tight_layout()
```

### A-5. 변수 중요도 분석

```python
# Feature importances (which features explain CATE heterogeneity?)
importance = forest.feature_importances_

feature_names = ['campaign', 'hour', 'exchange', 'region']
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\nFeature Importance for CATE Heterogeneity:")
print(importance_df)

# 시각화
plt.figure(figsize=(8, 4))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance')
plt.title('Which features explain treatment effect heterogeneity?')
plt.tight_layout()
```

### A-6. 실무 적용의 한계

```
CATE 분석의 한계:

1. 연속 처치 해석 어려움
   - T = log(bid) (연속)
   - τ(x) = ∂Y/∂T | X = x
   - "bid 1% 증가의 효과"가 비선형일 수 있음

2. Unconfoundedness 가정
   - 모든 교란변수 통제했다고 가정
   - 관측되지 않은 교란변수 존재 가능
   - 예: 경쟁 강도, 광고 품질

3. Production 적용 복잡성
   - CATE → 입찰 전략 변환 필요
   - Lookup table 방식: segment별 CATE 기반 bid 조정

4. 계산 비용
   - 학습: 수 시간
   - 추론: segment당 평균 CATE 사용 (실시간 불가)

5. Positivity 고려사항 (02_selection_bias_diagnosis):
   - Win→Click CATE: Win PS AUC=0.91 (clean, leakage 제거) → positivity violation (overlap ~46%) → CATE 불안정
   - Bid→Win CATE: bid variation 풍부 (연속 변수) → overlap 양호 → 추정 가능
   - 권장: bid→win CATE에 집중, win→click CATE는 참고용으로만 활용

권장:
- 연구/분석 목적으로 활용
- 세그먼트별 전략 차별화 인사이트 도출
- Production은 단순화된 규칙 적용
- bid→win CATE에 집중 (positivity 양호)
```

### A-7. Attribution-aware CATE (EDA-driven, 신규)

```
EDA Finding:
- CVR near-trivial: CTCVR 0.0061%
- Retargeting: click→conv ~1초 (retargeting artifact, 100% click-through)
- Branding: CVR=0 (5/10 advertisers)

Attribution 한계:
- Conversion은 retargeting artifact로 confounded
- Click→Conv attribution window가 ~1초 → 사실상 retargeting tracking pixel
- 인과적으로 "bid가 conversion에 미치는 영향"을 추정하기 어려움

CATE Outcome 권장:
- Primary: Y = Click (win→click CATE, 모든 advertiser 공통)
- Secondary: Y = Win (bid→win CATE, positivity 양호)
- Exclude: Y = Conversion (retargeting artifact로 confounded)
```

### A-8. Advertiser-stratified CATE (EDA-driven, 신규)

```
EDA Finding:
- Branding (5): CTR 0.03-0.44%, CVR=0, WR 18-31%
- Retargeting (3): CTR 0.03-0.08%, CVR 28-53%, WR 25-46%
- Mixed (1): CTR 0.05%, CVR 2.54%, WR 29%

Pooled CATE의 위험:
- Advertiser taxonomy별 CTR/CVR/WR 패턴이 완전히 다름
- Pooled CATE는 Simpson's Paradox 위험 (subgroup별 -10%~+18% 편향)

권장:
- Advertiser taxonomy별 CATE 분리 추정:
  1. Branding CATE: bid→win, win→click (CVR 무시)
  2. Retargeting CATE: bid→win, win→click (CVR는 참고만)
  3. Mixed CATE: Branding 기준 적용
- CausalForestDML의 X에 advertiser taxonomy 반드시 포함
- 또는 advertiser별 별도 모델 학습
```

---

## Part B: SCM & Counterfactual

### B-1. DAG 명세 (DoWhy)

```python
import dowhy
from dowhy import CausalModel

# DAG 정의
graph = """
digraph {
    // Exogenous
    U_context [label="U_context", observed="no"];
    U_user [label="U_user", observed="no"];
    U_market [label="U_market", observed="no"];

    // Observed variables
    campaign [label="Campaign"];
    context [label="Context (hour, site)"];
    user [label="User (tag, region)"];
    pctr [label="pCTR"];
    pcvr [label="pCVR"];
    bid [label="Bid Price"];
    floor [label="Floor Price"];
    market_price [label="Market Price"];
    win [label="Win"];
    click [label="Click"];
    conversion [label="Conversion"];

    // Causal relationships
    U_context -> context;
    U_user -> user;
    U_market -> market_price;

    campaign -> pctr;
    context -> pctr;
    user -> pctr;

    campaign -> pcvr;
    user -> pcvr;

    pctr -> bid;
    pcvr -> bid;
    campaign -> bid;

    bid -> win;
    market_price -> win;
    floor -> win;
    context -> market_price;

    win -> click;
    user -> click;
    context -> click;

    click -> conversion;
    user -> conversion;
}
"""

# Causal Model 생성
model = CausalModel(
    data=data,
    treatment='bid',
    outcome='win',
    graph=graph
)

# DAG 시각화
model.view_model()
```

### B-2. 식별 및 추정

```python
# 인과 효과 식별
identified_estimand = model.identify_effect(
    proceed_when_unidentifiable=True
)
print(identified_estimand)

# 추정 방법 선택
estimate_methods = [
    "backdoor.linear_regression",
    "backdoor.propensity_score_matching",
    "backdoor.propensity_score_weighting"
]

results = {}
for method in estimate_methods:
    try:
        estimate = model.estimate_effect(
            identified_estimand,
            method_name=method
        )
        results[method] = {
            'estimate': estimate.value,
            'ci': estimate.get_confidence_intervals()
        }
    except Exception as e:
        results[method] = {'error': str(e)}

# 결과 비교
print("\nCausal Effect Estimates:")
for method, result in results.items():
    if 'estimate' in result:
        print(f"  {method}: {result['estimate']:.4f}")
```

### B-3. 반박 테스트 (Refutation)

```python
# 반박 테스트: 결과의 robust함 확인

# 1. Random common cause
refute_random = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="random_common_cause",
    num_simulations=100
)
print("\nRefutation - Random Common Cause:")
print(refute_random)

# 2. Placebo treatment
refute_placebo = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="placebo_treatment_refuter",
    placebo_type="permute"
)
print("\nRefutation - Placebo Treatment:")
print(refute_placebo)

# 3. Data subset
refute_subset = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="data_subset_refuter",
    subset_fraction=0.8,
    num_simulations=10
)
print("\nRefutation - Data Subset:")
print(refute_subset)
```

### B-4. 반사실 추론 (Counterfactual)

```python
# 개별 반사실: "이 bid가 X였다면 win했을까?"
def individual_counterfactual(model, data_point, treatment_value):
    """
    개별 샘플에 대한 반사실 추론

    주의: 개별 반사실은 불확실성이 매우 큼
    """
    # 원래 결과
    original_outcome = data_point['win']

    # 반사실 설정
    cf_data = data_point.copy()
    cf_data['bid'] = treatment_value

    # 반사실 결과 추정 (간단한 근사)
    # 실제로는 structural equation 필요
    pass


# 집합 반사실: "모든 bid를 10% 올리면?"
def aggregate_counterfactual(data, model, bid_multiplier=1.1):
    """
    집합 반사실 시뮬레이션

    모든 bid를 일괄 변경했을 때의 결과 예측
    """
    # 원래 bid
    original_bids = data['bid_price'].values
    original_wins = data['win'].mean()

    # 반사실 bid
    cf_bids = original_bids * bid_multiplier

    # Win rate 예측 (Win Rate 모델 사용)
    # 가정: 다른 조건 동일
    cf_wins = predict_win_rate(cf_bids, data)

    return {
        'original_win_rate': original_wins,
        'counterfactual_win_rate': cf_wins.mean(),
        'bid_change': bid_multiplier - 1,
        'win_rate_change': cf_wins.mean() - original_wins
    }
```

### B-5. 실무 적용의 심각한 한계

```
SCM/Counterfactual의 한계 (Production 적용 어려움):

1. 개별 반사실의 불확실성
   - "이 사용자가 다른 bid를 받았다면?"
   - 근본적으로 관측 불가능한 반사실
   - 모델 의존성 높음

2. DAG 명세의 주관성
   - 변수 간 인과관계 가정 필요
   - 전문가 판단에 의존
   - 틀린 DAG → 틀린 결과

3. 계산 복잡성
   - 실시간 serving 불가
   - Batch 분석용으로만 활용

4. 해석의 어려움
   - 비즈니스 의사결정으로 연결 어려움
   - "So what?" 질문에 답하기 어려움

권장:
- 포트폴리오/논문용으로만 활용
- Production에는 단순한 시뮬레이션 사용
- DAG는 인과 관계 문서화 목적
```

---

## Part C: 정책 시뮬레이션 (Production 필수)

### C-1. 입찰 전략 비교 시뮬레이션

```python
class PolicySimulator:
    """
    입찰 정책 시뮬레이션

    다양한 전략을 historical 데이터에 적용하여 비교
    """

    def __init__(self, data, win_rate_model, pctr_model, pcvr_model):
        self.data = data.copy()
        self.win_rate_model = win_rate_model
        self.pctr_model = pctr_model
        self.pcvr_model = pcvr_model

    def simulate_policy(self, policy, **kwargs):
        """
        정책 시뮬레이션

        Args:
            policy: 'fixed', 'pCTR_proportional', 'value_based', 'cate_optimized'
        """
        results = []

        for idx, row in self.data.iterrows():
            # 정책별 bid 계산
            bid = self._compute_bid(row, policy, **kwargs)

            # Win 확률 예측
            win_prob = self._predict_win(row, bid)

            # 기대 결과 계산
            expected_win = win_prob
            expected_click = win_prob * row.get('pctr', 0.01)
            expected_conv = expected_click * row.get('pcvr', 0.05)

            # First-price: 지불 = bid (win한 경우)
            expected_spend = win_prob * bid

            results.append({
                'bid': bid,
                'win_prob': win_prob,
                'expected_win': expected_win,
                'expected_click': expected_click,
                'expected_conv': expected_conv,
                'expected_spend': expected_spend
            })

        return pd.DataFrame(results)

    def _compute_bid(self, row, policy, **kwargs):
        """정책별 bid 계산"""
        if policy == 'fixed':
            return kwargs.get('fixed_bid', 100)

        elif policy == 'pCTR_proportional':
            # bid = base * pCTR
            base = kwargs.get('base_bid', 1000)
            pctr = row.get('pctr', self.pctr_model.predict(row)[0])
            return base * pctr

        elif policy == 'value_based':
            # bid = value * shade
            # V(x) = debiased_pCTR × CPC_target (CVR near-trivial)
            pctr = row.get('pctr', 0.01)
            cpc_target = kwargs.get('cpc_target', 10)
            shade = kwargs.get('shade', 0.85)
            value = pctr * cpc_target
            return value * shade

        elif policy == 'cate_optimized':
            # bid = value * (1 + cate_adjustment)
            base_bid = self._compute_bid(row, 'value_based', **kwargs)
            cate = row.get('cate', 0)
            cate_adj = np.clip(cate * kwargs.get('cate_scale', 1), -0.3, 0.3)
            return base_bid * (1 + cate_adj)

        else:
            raise ValueError(f"Unknown policy: {policy}")

    def _predict_win(self, row, bid):
        """Win 확률 예측"""
        features = row.to_dict()
        features['bid_price'] = bid
        return self.win_rate_model.predict(features)

    def compare_policies(self, policies_config):
        """다중 정책 비교"""
        comparison = {}

        for name, config in policies_config.items():
            policy = config.pop('policy')
            results = self.simulate_policy(policy, **config)

            comparison[name] = {
                'total_bids': len(results),
                'expected_wins': results['expected_win'].sum(),
                'expected_clicks': results['expected_click'].sum(),
                'expected_conversions': results['expected_conv'].sum(),
                'total_spend': results['expected_spend'].sum(),
                'avg_bid': results['bid'].mean(),
                'win_rate': results['win_prob'].mean()
            }

            # ROI 계산
            conv_value = config.get('conv_value', 50)
            comparison[name]['expected_revenue'] = \
                comparison[name]['expected_conversions'] * conv_value
            comparison[name]['roi'] = \
                comparison[name]['expected_revenue'] / \
                max(comparison[name]['total_spend'], 1)

        return pd.DataFrame(comparison).T
```

### C-1a. Debiased vs Biased pCTR 정책 비교 (신규)

```
핵심 시뮬레이션: Debiasing의 경제적 가치 정량화

비교:
1. Biased policy: V(x) = biased_pCTR × CPC_target
   - CTR overestimation +6.7% → 입찰가 6.7% 과다
   - 고가치 세그먼트 저평가, 저가치 세그먼트 과평가

2. Debiased policy: V(x) = debiased_pCTR × CPC_target
   - ESCM²-WC(DR) 출력 → unbiased value

Expected outcome:
- Debiased: 동일 예산에서 ROI 향상 (정확한 V(x))
- Biased: 예산 낭비 (과대 입찰) + 기회 손실 (과소 입찰)
- 차이 = debiasing의 경제적 가치
```

### C-1b. Floor-aware vs Floor-naive Shading 비교 (신규)

```
비교:
1. Floor-naive: 모든 bid에 동일한 shade factor 적용
   - shade = optimal_shading(value, market_dist)

2. Floor-aware (dual-regime):
   - Floor-bound: shade to floor (overpayment 최소화)
   - Competitive: standard shading

Expected outcome:
- Floor-aware → overpayment 감소 (76% baseline → 15-25% 목표)
- 특히 Ex3 (active floor) 환경에서 효과 극대화
```

### C-1c. Advertiser Taxonomy별 최적 정책 탐색 (신규)

```
Advertiser별 최적 bid multiplier/shade 탐색:

1. Branding (1458, 3386, 3427, 2261, 2997):
   - 목표: Reach 최대화 (Win count)
   - 전략: 보수적 shading + 넓은 pacing
   - CTR 0.03-0.44%, WR 18-31%

2. Retargeting (2821, 3358, 2259):
   - 목표: ROI 최적화 (CPC or CPA)
   - 전략: 적극적 shading + CTR 기반 value bidding
   - CTR 0.03-0.08%, CVR 28-53%, WR 25-46%

3. Mixed (3476):
   - Branding 기준 적용 (CVR 2.54%로 낮음)
```

### C-2. 예산 시나리오 분석

```python
class BudgetScenarioAnalyzer:
    """예산 시나리오 분석"""

    def __init__(self, simulator, base_budget):
        self.simulator = simulator
        self.base_budget = base_budget

    def analyze_scenarios(self, budget_multipliers=[0.8, 1.0, 1.2, 1.5],
                          hourly_allocation='uniform'):
        """
        다양한 예산 수준에서 최적 전략 분석

        hourly_allocation: 'uniform' (균등) or 'adaptive' (EDA hourly WR 기반)
        """
        scenarios = {}

        for mult in budget_multipliers:
            budget = self.base_budget * mult

            # 해당 예산으로 최적화 시뮬레이션
            results = self._optimize_for_budget(budget)

            scenarios[f'{int(mult*100)}%'] = {
                'budget': budget,
                'expected_conversions': results['conversions'],
                'expected_revenue': results['revenue'],
                'roi': results['roi'],
                'optimal_shade': results['optimal_shade']
            }

        return pd.DataFrame(scenarios).T

    def _optimize_for_budget(self, budget):
        """주어진 예산에서 최적 전략 찾기"""
        best_roi = 0
        best_results = None
        best_shade = 0.85

        for shade in np.arange(0.5, 1.0, 0.05):
            results = self.simulator.simulate_policy(
                'value_based',
                shade=shade,
                conv_value=50
            )

            # 예산 제약 적용
            cumsum = results['expected_spend'].cumsum()
            mask = cumsum <= budget
            results_constrained = results[mask]

            if len(results_constrained) > 0:
                revenue = results_constrained['expected_conv'].sum() * 50
                spend = results_constrained['expected_spend'].sum()
                roi = revenue / max(spend, 1)

                if roi > best_roi:
                    best_roi = roi
                    best_shade = shade
                    best_results = {
                        'conversions': results_constrained['expected_conv'].sum(),
                        'revenue': revenue,
                        'spend': spend,
                        'roi': roi,
                        'optimal_shade': shade
                    }

        return best_results
```

### C-3. 시뮬레이션의 한계

```
정책 시뮬레이션의 한계:

1. 경쟁 반응 미반영
   - 우리가 bid 전략 변경 → 경쟁사 대응
   - 시뮬레이션은 "다른 모든 조건 동일" 가정
   - 실제 효과는 달라질 수 있음

2. Historical data 분포 이동
   - 과거 데이터로 미래 예측
   - 시장 환경 변화 미반영

3. 예산 소진 패턴
   - 시뮬레이션: 순차적 처리
   - 실제: 동시 다발적 입찰 기회

4. 모델 불확실성
   - Win rate, CTR, CVR 모델 오차
   - 오차 전파 (error propagation)

5. Overpayment Diagnostic (추가 metric)
   - overpayment_ratio = (bid - payprice) / payprice
   - iPinYou flat-bid baseline: ~76%
   - 시뮬레이션 결과에 Flat-bid / Shaded / Value-based 각각의 overpayment 비교

해결책:
- 시뮬레이션은 방향성 지표로만 활용
- A/B 테스트로 실제 효과 검증
- 민감도 분석 수행
```

---

## Part D: 방법론별 Production 적용 가능성

### D-1. 요약 테이블

| 방법론 | 학술적 가치 | 실무 적용성 | 계산 복잡도 | 권장 |
|--------|------------|------------|------------|------|
| **CATE** | ★★★★★ | ★★★★☆ | 중간 | Production 권장 |
| **SCM/CF** | ★★★★★ | ★★☆☆☆ | 높음 | 포트폴리오용 |
| **정책 시뮬레이션** | ★★★★☆ | ★★★★★ | 낮음 | **Production 필수** |

### D-2. Production 적용 가이드

```
CATE 적용:
1. Segment별 평균 CATE 계산 (batch)
2. CATE 기반 bid 조정 규칙 수립
   - High CATE segment: bid 증가 효과적 → 적극 입찰
   - Low CATE segment: bid 증가 효과 적음 → 보수적 입찰
3. Lookup table로 serving

SCM 적용:
- Production에 직접 적용하지 않음
- 연구/분석 목적으로만 사용
- DAG 문서화, 인과 관계 이해 목적

정책 시뮬레이션 적용:
1. 정기적 (주/월) 시뮬레이션 실행
2. 최적 파라미터 도출 (shade, pace 등)
3. A/B 테스트로 검증
4. Production 적용
```

---

## 산출물

| 산출물 | 경로 | 설명 |
|--------|------|------|
| CausalForest 모델 | `models/causal_forest.pkl` | 학습된 모델 |
| CATE 분석 리포트 | `reports/04_cate_analysis.md` | 이질성 분석 |
| DAG 명세 | `docs/causal_dag.gv` | Graphviz 파일 |
| 정책 시뮬레이션 | `reports/04_policy_simulation.md` | 전략 비교 결과 |
| 분석 노트북 | `notebooks/04_causal_analysis.ipynb` | 실험 코드 |

---

## 핵심 요약

1. **CATE**: 세그먼트별 입찰 효과 이질성 분석 → Production 활용 가능
2. **SCM**: DAG 기반 인과 구조 명세 → 문서화/이해 목적
3. **Counterfactual**: 반사실 추론 → 학술 목적 (Production 어려움)
4. **정책 시뮬레이션**: 전략 비교 → **Production 필수**
5. **한계**: 경쟁 반응 미반영, 모델 불확실성, 시장 변화
