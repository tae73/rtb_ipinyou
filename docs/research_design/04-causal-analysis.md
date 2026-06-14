# SP4: CATE / SCM / 정책 시뮬레이션

---

## 개요

| 항목 | 내용 |
|------|------|
| **목적** | Multi-outcome 인과 분석 (Surplus 중심) 및 정책 평가 |
| **선행 조건** | SP1 (예측 모델 + Win Tower), SP3 (입찰 최적화 + 시뮬레이션) |
| **후속 단계** | SP5 (Serving) |
| **핵심 산출물** | Multi-outcome CATE, Surplus 분해, DAG, 정책 시뮬레이션 |

**핵심 재설계 (2026-04):**
1. **Surplus를 primary CATE outcome으로**: 기존 Y=Win/Click → Y=Surplus + 분해
2. **V(x)는 CATE outcome 불가**: bid 무관 (∂V/∂bid = 0), 순환 논리
3. **Multi-outcome 분해**: τ_win × τ_pay × τ_click → τ_surplus 합성으로 point mass 문제 해결
4. **Win을 mediator로**: NIE(volume channel) vs NDE(cost channel) 분해
5. **SCM**: DAG에 surplus/payment 추가, `run_auction_simulation()` 기반 model-based counterfactual

---

## Part A: Multi-Outcome CATE 분석

### A-1. 문제 재정의: Surplus 중심 Multi-Outcome CATE

```
기존 설계:
  T = log(bid), Y = Win (primary), Y = Click (secondary)
  → Surplus는 NB07 시뮬레이션에서만 사용, 인과 프레임워크 밖

재설계:
  T = log(bid_price)
  Y = {Win, Payment, Click, Surplus}  ← 4개 outcome 동시 추정
  X = advertiser_taxonomy, hour, adexchange, region  ← 이질성 변수
  W = slotprice, slot_area, domain_freq  ← 교란 변수

핵심 질문 변경:
  기존: "어떤 세그먼트에서 bid 증가가 더 효과적인가?" (Win 기준)
  재설계: "어떤 세그먼트에서 bid 증가가 Surplus를 늘리는가?" (경제적 가치 기준)

Surplus 공식 (first-price):
  Surplus(bid, x) = [V(x) - bid] × 1{bid ≥ market_price}
  where V(x) = debiased_pCTR(x) × CPC_target

bid 증가의 두 채널:
  1. Volume channel: bid ↑ → P(win) ↑ → surplus 획득 기회 증가
  2. Cost channel: bid ↑ → payment ↑ → surplus per win 감소

EDA 수치:
  - Win rate: 23.67% (hourly 8.59%~43.06%, exchange 13.24%~55.55%)
  - CTR: 0.0752%, CVR: 8.07% (of clicks) — CVR near-trivial → 제외
  - 9 advertisers: Branding 5, Retargeting 3, Mixed 1
  - Market price: median 68, mean 78 CPM; Floor binding 32.24%
  - iPinYou flat-bid surplus: -805M (massive overpayment 10.14×)
```

### A-2. V(x)가 CATE Outcome이 될 수 없는 이유

```
V(x) = debiased_pCTR(x) × CPC_target

문제 1: bid에 의존하지 않음
  V(x)는 features X에 의해 결정되며, bid가 달라져도 V(x)는 변하지 않음.
  τ(x) = ∂V(x)/∂log(bid) = 0  ← CATE가 trivially zero

문제 2: 모델 예측값이지 관측된 outcome이 아님
  V(x)는 ESCM²-WC(DR) 모델의 출력. 이를 outcome으로 쓰면:
  "모델 예측이 bid에 따라 어떻게 달라지는가?" → 순환 논리

문제 3: 이미 해결된 문제
  V(x)의 heterogeneity (어떤 impression이 가치 있는가?)는
  pCTR 모델이 이미 답한 문제. CATE를 쓸 필요가 없음.

V(x)의 올바른 역할:
  - CATE outcome이 아닌 surplus 계산의 상수
  - τ_surplus(x) ≈ V(x) × τ_win(x) - τ_pay(x)에서 V(x)는 고정된 가중치
  - Surplus 분해에서 Volume channel의 스케일 결정
```

### A-3. Multi-Outcome CausalForestDML 설정

```python
from econml.dml import CausalForestDML
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from src.bidding.value import compute_impression_values, ValueConfig

# ── 데이터 준비 ──

# Treatment: 연속 처치
T = np.log(df['bidprice'].values + 1)

# Heterogeneity variables: CATE가 이에 따라 달라짐
X = df[['advertiser_taxonomy', 'hour', 'adexchange', 'region']].values

# Confounders: T와 Y 모두에 영향
W = df[['slotprice', 'slot_area', 'domain_freq']].values

# V(x) for surplus computation
values = compute_impression_values(predictions['p_ctr']).values

# ── Outcome 1: Win (full sample, binary) ──
# τ_win(x) = ∂P(win)/∂log(bid) | X=x — "bid 탄성도"
Y_win = df['win'].values

# ── Outcome 2: Payment (winners-only, continuous) ──
# τ_pay(x) = ∂E[payprice|win=1]/∂log(bid) | X=x — "비용 탄성도"
# Point mass at 0 회피: winners-only로 필터링
win_mask = df['win'].values == 1
Y_pay = df.loc[win_mask, 'payprice'].values
T_pay, X_pay, W_pay = T[win_mask], X[win_mask], W[win_mask]

# ── Outcome 3: Click (full sample, binary) ──
# τ_click(x) = ∂P(click)/∂log(bid) | X=x — "가치 탄성도"
# 예상: 작음 (bid는 click에 직접 영향 미미)
Y_click = df['click'].values

# ── Outcome 4: Surplus (full sample, continuous — 검증용) ──
# 76% lost bids에서 surplus=0 (point mass)
# 직접 추정 후 분해 결과와 비교하여 검증
Y_surplus = (values - df['payprice'].values) * df['win'].values

# ── CausalForestDML 공통 설정 ──
def make_forest():
    return CausalForestDML(
        model_y=LGBMRegressor(n_estimators=200, num_leaves=31, verbose=-1),
        model_t=LGBMRegressor(n_estimators=200, num_leaves=31, verbose=-1),
        discrete_treatment=False,  # 연속 처치
        n_estimators=1000,
        honest=True,               # Honest splitting → valid CI
        inference=True,
        random_state=42,
    )

# ── 4개 outcome 순차 추정 ──
forest_win = make_forest()
forest_win.fit(Y=Y_win, T=T, X=X, W=W)
tau_win = forest_win.effect(X).flatten()
ate_win = forest_win.ate(X)
print(f"Win ATE: {ate_win:.4f} — log(bid) 1 증가 → win rate {ate_win*100:.2f}%p 증가")

forest_pay = make_forest()
forest_pay.fit(Y=Y_pay, T=T_pay, X=X_pay, W=W_pay)
tau_pay = forest_pay.effect(X_pay).flatten()
ate_pay = forest_pay.ate(X_pay)
print(f"Payment ATE: {ate_pay:.1f} CPM — winners-only, first-price 기계적 관계 포함")

forest_click = make_forest()
forest_click.fit(Y=Y_click, T=T, X=X, W=W)
tau_click = forest_click.effect(X).flatten()
ate_click = forest_click.ate(X)
print(f"Click ATE: {ate_click:.6f} — 예상: bid→click 직접 효과 미미")

forest_surplus = make_forest()
forest_surplus.fit(Y=Y_surplus, T=T, X=X, W=W)
tau_surplus_direct = forest_surplus.effect(X).flatten()
ate_surplus = forest_surplus.ate(X)
print(f"Surplus ATE: {ate_surplus:.2f} CPM — bid 증가의 순경제 효과")
```

### A-4. Surplus CATE 분해 및 검증

```
이론적 분해:
  Surplus = (V - payment) × win = V × win - payment × win

  ∂Surplus/∂log(bid) = V × ∂win/∂log(bid) - ∂(payment×win)/∂log(bid)
  
  τ_surplus(x) ≈ V(x) × τ_win(x) - τ_pay_full(x)

  where τ_pay_full = ∂E[payprice × win]/∂log(bid) on full sample

근사:
  winners-only τ_pay를 full sample로 확장 시:
  τ_surplus(x) ≈ V(x) × τ_win(x) - τ_pay(x)  (winners 부분만)

검증 목적:
  - 직접 추정 (forest_surplus) vs 분해 추정의 상관관계
  - 높으면: 분해가 유효, 개별 component 해석 가능
  - 낮으면: 상호작용 term이 크거나 winners-only collider bias 존재
```

```python
# ── 분해 합성 ──
# winners-only τ_pay를 full sample 인덱스에 매핑
tau_pay_full = np.zeros_like(tau_win)
tau_pay_full[win_mask] = tau_pay

tau_surplus_decomposed = values * tau_win - tau_pay_full

# ── 검증 ──
# winners-only에서만 비교 (non-winners는 둘 다 ≈ 0이라 trivial correlation)
direct_won = tau_surplus_direct[win_mask]
decomposed_won = tau_surplus_decomposed[win_mask]

corr = np.corrcoef(direct_won, decomposed_won)[0, 1]
mae = np.mean(np.abs(direct_won - decomposed_won))
bias = np.mean(decomposed_won - direct_won)

print(f"Decomposition validation (winners-only):")
print(f"  Correlation: {corr:.3f} (기대: > 0.5)")
print(f"  MAE: {mae:.2f} CPM")
print(f"  Bias: {bias:.2f} CPM")
```

### A-5. Mediation Analysis: Volume vs Cost Channel

```
Surplus에 대한 bid 효과의 두 경로:

  Bid ─── [Volume] ──→ Win ──→ Surplus
    └──── [Cost] ────→ Payment ──→ Surplus (negative)

NIE (Natural Indirect Effect, Volume channel):
  bid ↑ → P(win) ↑ → 더 많은 impression 획득 → surplus 증가
  NIE = E[V(x) × τ_win(x)]

NDE (Natural Direct Effect, Cost channel):
  bid ↑ → payment ↑ (first-price) → surplus per win 감소
  NDE = Total - NIE

해석:
  NIE > |NDE|: bid 증가가 전체적으로 이득 (더 많이 이겨서 보상)
  NIE < |NDE|: bid 증가가 전체적으로 손해 (overpayment이 지배)

주의:
  이 분해는 Pearl의 formal NDE/NIE가 아닌 algebraic 분해.
  Surplus = V×win - payment×win 이 항등식이므로 인과 가정 추가 불필요.
  세그먼트별로 NIE/NDE 비율이 달라지는 것이 핵심 인사이트.
```

```python
# ── Mediation 분해 ──

# NIE (Volume channel): V(x) × τ_win(x) 평균
nie = np.mean(values * tau_win)

# Total effect: 직접 추정한 surplus CATE 평균
total = np.mean(tau_surplus_direct)

# NDE (Cost channel): Total - NIE
nde = total - nie

mediation_proportion = nie / total if abs(total) > 1e-10 else float('nan')

print(f"Total Effect: {total:.2f} CPM/unit log(bid)")
print(f"NIE (Volume):  {nie:.2f} CPM ({mediation_proportion*100:.1f}%)")
print(f"NDE (Cost):    {nde:.2f} CPM ({(1-mediation_proportion)*100:.1f}%)")

# ── 세그먼트별 Mediation ──
# Advertiser taxonomy별 NIE/NDE 분해
for taxonomy in ['branding', 'retargeting', 'mixed']:
    mask = df['advertiser_taxonomy'] == taxonomy
    seg_nie = np.mean(values[mask] * tau_win[mask])
    seg_total = np.mean(tau_surplus_direct[mask])
    seg_nde = seg_total - seg_nie
    print(f"\n{taxonomy}:")
    print(f"  NIE={seg_nie:.2f}, NDE={seg_nde:.2f}, Total={seg_total:.2f}")
```

### A-6. 세그먼트별 CATE + Feature Importance

```python
# ── Feature Importance: 어떤 변수가 CATE heterogeneity를 설명하는가? ──
feature_names = ['advertiser_taxonomy', 'hour', 'adexchange', 'region']

for name, forest in [('win', forest_win), ('surplus', forest_surplus)]:
    importance = forest.feature_importances_
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
    }).sort_values('importance', ascending=False)
    print(f"\n{name} CATE Feature Importance:")
    print(imp_df.to_string(index=False))

# ── 세그먼트별 CATE 분석 ──
# Exchange × Hour heatmap
segment_df = df[['adexchange', 'hour']].copy()
segment_df['tau_win'] = tau_win
segment_df['tau_surplus'] = tau_surplus_direct

# Exchange별
print("\nτ_surplus by Exchange:")
print(segment_df.groupby('adexchange')['tau_surplus'].agg(['mean', 'std', 'count']))

# Hour별
print("\nτ_surplus by Hour:")
print(segment_df.groupby('hour')['tau_surplus'].agg(['mean', 'std']))

# 시각화: heatmap, CATE 분포, feature importance bar chart
```

### A-7. Advertiser-Stratified CATE

```
EDA Finding:
  - Branding (5): CTR 0.03-0.44%, CVR=0, WR 18-31%
  - Retargeting (3): CTR 0.03-0.08%, CVR 28-53%, WR 25-46%
  - Mixed (1): CTR 0.05%, CVR 2.54%, WR 29%

Pooled CATE의 위험:
  - Advertiser taxonomy별 CTR/CVR/WR 패턴이 완전히 다름
  - Pooled CATE는 Simpson's Paradox 위험 (subgroup별 -10%~+18% 편향)

접근:
  1. advertiser_taxonomy를 X에 포함하여 pooled 추정 (기본)
  2. Taxonomy별 별도 CausalForestDML 학습 (검증용)
  3. 두 접근의 CATE ranking 비교 — 일치하면 pooled 채택
```

```python
ADVERTISER_TAXONOMY = {
    "retargeting": [2821, 3358, 2259],
    "branding": [1458, 3386, 3427, 2261, 2997],
    "mixed": [3476],
}

# Taxonomy별 별도 추정
for taxonomy, adv_ids in ADVERTISER_TAXONOMY.items():
    mask = df['advertiser'].isin(adv_ids)
    if mask.sum() < 10_000:
        continue
    
    forest_seg = make_forest()
    forest_seg.fit(
        Y=Y_surplus[mask], T=T[mask], 
        X=X[mask], W=W[mask],
    )
    ate_seg = forest_seg.ate(X[mask])
    print(f"{taxonomy}: Surplus ATE = {ate_seg:.2f} CPM (n={mask.sum():,})")
```

### A-8. CATE → Bidding Insight

```
CATE 결과의 실무 활용:

1. 고-τ_surplus 세그먼트 식별
   - τ_surplus(x) > 0: bid 증가가 순이익 증가 → 적극 입찰
   - τ_surplus(x) < 0: bid 증가가 순이익 감소 → 보수적 입찰

2. bid 조정 규칙 (offline, segment-level)
   bid_adj(x) = 1 + γ × normalize(τ_surplus(x))
   where γ는 조정 강도 (보수적: 0.1, 적극적: 0.3)

3. NB07 결과와 cross-reference
   - CATE에서 고-surplus 세그먼트 = NB07에서 dual-regime이 잘 작동하는 세그먼트?
   - Exchange 1 (median 153 CPM) vs Exchange 3 (median > 300 CPM) CATE 비교
   - 일치하면: CATE가 시뮬레이션 결과를 인과적으로 뒷받침

4. Production 적용
   - Segment-level CATE → lookup table (batch 계산)
   - bid(x) = V(x) × shade(x) × pace(t) × bid_adj(x)
   - bid_adj(x)가 CATE 기반 새 factor
```

### A-9. 한계 및 주의사항

```
1. Point Mass at 0 (Surplus)
   - Lost bids 76.3%에서 surplus=0
   - 직접 surplus CATE는 zero-inflated → 분해 접근 권장
   - 분해 결과와 직접 결과 비교로 robustness 확인

2. Payment의 Collider Bias
   - winners-only 조건부 추정 시 win이 collider:
     bid → win ← market_price → payment
   - win=1로 조건부하면 bid-market_price 사이 spurious association
   - 완화: W에 slotprice (market competition proxy) 포함
   - 문서화: collider bias 가능성 명시

3. First-Price Mechanical Relationship
   - First-price에서 payment = bid (if win)
   - τ_pay는 산술 항등식에 가까움 (인과 해석 주의)
   - 의미 있는 인과 질문: "bid ↑ → 더 비싼 impression도 이기면서
     average payment은 어떻게 변하는가?" (composition effect)

4. Unconfoundedness 가정
   - 관측되지 않은 교란: 경쟁 강도, 광고 품질
   - iPinYou flat-bid 특성상 bid variation 제한적 (6개 discrete price)
   - Sensitivity analysis 필요

5. Positivity
   - Bid→Win: 양호 (연속 bid, 풍부한 variation)
   - Win→Click: 위험 (PS AUC 0.91, overlap ~46%)
   - → bid→win CATE에 집중, win→click은 참고용

6. Computational Cost
   - 19.4M rows → CausalForestDML 직접 적용 어려움
   - subsample_n=2M (탐색), 5M (최종)으로 관리
   - Subsample stability 검증 필수 (1M, 2M, 5M에서 ATE 수렴 확인)

7. iPinYou Bid Structure
   - iPinYou는 6개 discrete flat-bid price만 사용 (227~300 CPM)
   - Treatment variation이 제한적 → CATE 해상도 한계
   - Cross-advertiser pooling으로 variation 확보 가능
```

---

## Part B: SCM & Counterfactual

### B-1. DAG 명세 (DoWhy)

```python
import dowhy
from dowhy import CausalModel

# DAG 정의 — Surplus를 최종 outcome으로, 2-channel 구조 반영
graph = """
digraph {
    // Exogenous (unobserved)
    U_market [label="U_market", observed="no"];
    U_user [label="U_user", observed="no"];

    // Observed variables
    context [label="Context\\n(hour, exchange, domain)"];
    user [label="User\\n(region, usertag)"];
    campaign [label="Campaign\\n(advertiser)"];
    floor [label="Floor Price"];
    market_price [label="Market Price"];
    bid [label="Bid Price"];
    win [label="Win"];
    payment [label="Payment"];
    click [label="Click"];
    value [label="V(x)"];
    surplus [label="Surplus"];

    // Exogenous → Observed
    U_market -> market_price;
    U_user -> user;

    // Context → Market
    context -> market_price;

    // Bidding decision
    campaign -> bid;
    context -> bid;
    user -> bid;

    // Auction outcome
    bid -> win;
    market_price -> win;
    floor -> win;
    context -> floor;

    // Two channels: Bid → Surplus
    // Volume channel: bid → win → surplus
    bid -> payment;       // First-price: payment = bid if win
    win -> payment;       // Payment only occurs if won

    // Click & Value
    win -> click;
    user -> click;
    context -> click;

    click -> value;       // V(x) = pCTR(x) × CPC_target
    user -> value;
    context -> value;
    campaign -> value;

    // Surplus composition
    value -> surplus;     // Surplus = V - payment (if win)
    payment -> surplus;
    win -> surplus;
}
"""

# ── Causal Model 생성 ──
# Treatment: bid → Outcome: surplus (primary)
model_surplus = CausalModel(
    data=data,
    treatment='bid',
    outcome='surplus',
    graph=graph,
)

# Treatment: bid → Outcome: win (secondary)
model_win = CausalModel(
    data=data,
    treatment='bid',
    outcome='win',
    graph=graph,
)

# DAG 시각화
model_surplus.view_model()
```

### B-2. 식별 및 추정

```python
# ── Surplus에 대한 인과 효과 식별 ──
identified_estimand = model_surplus.identify_effect(
    proceed_when_unidentifiable=True
)
print(identified_estimand)
# Backdoor adjustment set: {context, user, campaign, floor}

# ── 추정 방법 비교 ──
estimate_methods = [
    "backdoor.linear_regression",
    "backdoor.propensity_score_weighting",
]

results = {}
for method in estimate_methods:
    try:
        estimate = model_surplus.estimate_effect(
            identified_estimand,
            method_name=method,
        )
        results[method] = {
            'estimate': estimate.value,
            'ci': estimate.get_confidence_intervals(),
        }
    except Exception as e:
        results[method] = {'error': str(e)}

print("\nCausal Effect of Bid on Surplus:")
for method, result in results.items():
    if 'estimate' in result:
        print(f"  {method}: {result['estimate']:.4f}")

# Win에 대한 인과 효과도 동일하게 추정 (비교용)
```

### B-3. 반박 테스트 (Refutation)

```python
# 추정 결과의 robustness 확인

# 1. Random common cause: 랜덤 교란변수 추가해도 결과 안정적인가?
refute_random = model_surplus.refute_estimate(
    identified_estimand,
    estimate,
    method_name="random_common_cause",
    num_simulations=100,
)
print("Random Common Cause:", refute_random)
# 기대: estimate 변화 < 10%

# 2. Placebo treatment: treatment를 무작위 치환하면 효과가 사라지는가?
refute_placebo = model_surplus.refute_estimate(
    identified_estimand,
    estimate,
    method_name="placebo_treatment_refuter",
    placebo_type="permute",
)
print("Placebo Treatment:", refute_placebo)
# 기대: effect ≈ 0

# 3. Data subset: 80% 서브셋에서도 결과 안정적인가?
refute_subset = model_surplus.refute_estimate(
    identified_estimand,
    estimate,
    method_name="data_subset_refuter",
    subset_fraction=0.8,
    num_simulations=10,
)
print("Data Subset:", refute_subset)
# 기대: CV < 0.2
```

### B-4. Model-Based Counterfactual

```
기존 설계: structural equation 기반 개별 counterfactual (구현 어려움)
재설계: SP3 run_auction_simulation() 활용 model-based counterfactual

장점:
  - 이미 검증된 시뮬레이션 엔진 재사용
  - KM CDF 기반 win probability → 현실적 counterfactual
  - Surplus 직접 계산 → 경제적 해석 용이

한계:
  - Competitive response 미반영 (partial equilibrium)
  - Model uncertainty 전파 미계산
```

```python
from src.bidding.simulator import run_auction_simulation, compute_simulation_metrics
from src.bidding.value import compute_impression_values

# ── 데이터 준비 (won-only: payprice 관측) ──
won_mask = test_df['win'] == 1
market_prices = test_df.loc[won_mask, 'payprice'].values
clicks = test_df.loc[won_mask, 'click'].values
original_bids = test_df.loc[won_mask, 'bidprice'].values.astype(float)

# V(x) from debiased pCTR
values = compute_impression_values(predictions['p_ctr'][won_mask]).values

# ── Original baseline ──
original_result = run_auction_simulation(
    bids=original_bids,
    market_prices=market_prices,
    values=values,
    clicks=clicks,
    auction_type="first_price",
)
original_metrics = compute_simulation_metrics(
    original_result, values, market_prices, "original",
)

# ── Counterfactual scenarios ──
scenarios = {
    "bid -20%": 0.8,
    "bid -10%": 0.9,
    "bid +10%": 1.1,
    "bid +20%": 1.2,
}

cf_results = {}
for name, multiplier in scenarios.items():
    cf_bids = original_bids * multiplier
    cf_result = run_auction_simulation(
        bids=cf_bids,
        market_prices=market_prices,
        values=values,
        clicks=clicks,
        auction_type="first_price",
    )
    cf_metrics = compute_simulation_metrics(
        cf_result, values, market_prices, name,
    )
    
    cf_results[name] = {
        'win_rate': cf_metrics.win_rate,
        'total_surplus': cf_metrics.total_surplus,
        'overpayment_ratio': cf_metrics.overpayment_ratio,
        'delta_surplus': cf_metrics.total_surplus - original_metrics.total_surplus,
        'delta_win_rate': cf_metrics.win_rate - original_metrics.win_rate,
    }

print("\nCounterfactual Analysis:")
print(pd.DataFrame(cf_results).T)

# ── 개별 counterfactual case studies ──
# 특정 impression 선택: 다양한 특성 (high-V, low-V, floor-bound 등)
case_indices = [0, 1000, 50000]  # 예시
for idx in case_indices:
    v = values[idx]
    mp = market_prices[idx]
    orig_bid = original_bids[idx]
    print(f"\nCase {idx}: V={v:.1f}, market={mp:.0f}, bid={orig_bid:.0f}")
    for mult in [0.8, 0.9, 1.1]:
        cf_bid = orig_bid * mult
        cf_win = 1 if cf_bid >= mp else 0
        cf_surplus = (v - cf_bid) * cf_win if cf_win else 0
        print(f"  bid×{mult}: bid={cf_bid:.0f}, win={cf_win}, surplus={cf_surplus:.1f}")
```

### B-5. 실무 적용의 한계

```
SCM/Counterfactual의 한계:

1. 개별 반사실의 불확실성
   - "이 impression에 bid=150을 걸었다면?" → 근본적으로 관측 불가
   - Model-based counterfactual은 partial equilibrium 가정
   - 경쟁자 반응, 시장 변화 미반영

2. DAG 명세의 주관성
   - 변수 간 인과관계 가정 필요 (전문가 판단 의존)
   - Surplus 노드의 구조: V(x) → surplus 경로가 모델 의존적
   - 틀린 DAG → 틀린 identification

3. 계산 복잡성
   - dowhy 추정: 500K 서브셋에서도 수 분
   - 실시간 serving 불가 → batch 분석용

4. First-price Surplus 특수성
   - payment = bid (if win) → surplus = V - bid (if win)
   - bid를 높이면 기계적으로 surplus 감소 (winning 제외)
   - 의미 있는 counterfactual: win 여부가 바뀌는 marginal case

권장:
  - CATE는 segment-level 의사결정에 활용 (Part A)
  - SCM은 인과 구조 이해/문서화 + robustness 검증 목적
  - Production은 model-based counterfactual (simulation) 중심
```

---

## Part C: 정책 시뮬레이션 (Production 필수)

### C-1. 입찰 전략 비교 시뮬레이션

```python
class PolicySimulator:
    """
    입찰 정책 시뮬레이션

    다양한 전략을 historical 데이터에 적용하여 비교.
    SP3에서 src/bidding/simulator.py로 구현 완료:
      - run_auction_simulation(), compare_strategies()
      - 8개 표준 전략: flat, truthful, linear, optimal, dual-regime 등
    
    이 섹션은 CATE-informed 전략 추가를 위한 확장 설계.
    """

    def __init__(self, data, win_rate_model, pctr_model):
        self.data = data.copy()
        self.win_rate_model = win_rate_model
        self.pctr_model = pctr_model

    def simulate_policy(self, policy, **kwargs):
        """정책 시뮬레이션"""
        results = []

        for idx, row in self.data.iterrows():
            bid = self._compute_bid(row, policy, **kwargs)
            win_prob = self._predict_win(row, bid)
            expected_win = win_prob
            expected_click = win_prob * row.get('pctr', 0.01)
            expected_spend = win_prob * bid

            results.append({
                'bid': bid,
                'win_prob': win_prob,
                'expected_win': expected_win,
                'expected_click': expected_click,
                'expected_spend': expected_spend,
            })

        return pd.DataFrame(results)

    def _compute_bid(self, row, policy, **kwargs):
        """정책별 bid 계산"""
        if policy == 'fixed':
            return kwargs.get('fixed_bid', 100)

        elif policy == 'pCTR_proportional':
            base = kwargs.get('base_bid', 1000)
            pctr = row.get('pctr', self.pctr_model.predict(row)[0])
            return base * pctr

        elif policy == 'value_based':
            pctr = row.get('pctr', 0.01)
            cpc_target = kwargs.get('cpc_target', 200_000)
            shade = kwargs.get('shade', 0.85)
            value = pctr * cpc_target
            return value * shade

        elif policy == 'cate_optimized':
            # CATE 기반 bid 조정 (Part A 결과 활용)
            # bid = V(x) × shade(x) × (1 + γ × τ_surplus(x))
            base_bid = self._compute_bid(row, 'value_based', **kwargs)
            tau_surplus = row.get('tau_surplus', 0)
            gamma = kwargs.get('gamma', 0.2)
            cate_adj = np.clip(gamma * tau_surplus, -0.3, 0.3)
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
                'total_spend': results['expected_spend'].sum(),
                'avg_bid': results['bid'].mean(),
                'win_rate': results['win_prob'].mean(),
            }

            cpc_target = config.get('cpc_target', 200_000)
            comparison[name]['expected_revenue'] = \
                comparison[name]['expected_clicks'] * cpc_target
            comparison[name]['roi'] = \
                comparison[name]['expected_revenue'] / \
                max(comparison[name]['total_spend'], 1)

        return pd.DataFrame(comparison).T
```

### C-1a. Debiased vs Biased pCTR 정책 비교

```
핵심 시뮬레이션: Debiasing의 경제적 가치 정량화

비교:
1. Biased policy: V(x) = biased_pCTR × CPC_target
   - CTR overestimation +4.57% (LGB) → 입찰가 과다
   - IEB 0.362 → 25.9× overbidding (NB05 Section 11)

2. Debiased policy: V(x) = debiased_pCTR × CPC_target
   - ESCM²-WC(DR) IEB 0.014 → near-oracle V(x)

3. CATE-optimized policy: 2번 + τ_surplus 기반 세그먼트 조정
   - 고-surplus 세그먼트 적극, 저-surplus 세그먼트 보수

Expected outcome:
  - Debiased >> Biased (NB05에서 이미 확인: -14.7% surplus loss 감소)
  - CATE-optimized ≥ Debiased (세그먼트 최적화 추가 이득)
```

### C-1b. Floor-aware vs Floor-naive Shading 비교

```
비교:
1. Floor-naive: 모든 bid에 동일한 shade factor 적용
2. Floor-aware (dual-regime): floor-bound vs competitive 분리

Expected outcome:
  - Floor-aware → overpayment 감소 (NB07: dual-regime surplus +128M, best)
  - 특히 Ex3 (active floor) 환경에서 효과 극대화
  - 32.24% floor-binding impressions에서 just-above-floor bidding
```

### C-1c. Advertiser Taxonomy별 최적 정책 탐색

```
Advertiser별 최적 bid multiplier/shade 탐색:

1. Branding (1458, 3386, 3427, 2261, 2997):
   - 목표: Reach 최대화 (Win count)
   - 전략: 보수적 shading + 넓은 pacing
   - CATE insight: τ_win 중심 최적화

2. Retargeting (2821, 3358, 2259):
   - 목표: ROI 최적화 (CPC)
   - 전략: 적극적 shading + τ_surplus 기반 bidding
   - CATE insight: τ_surplus > 0 세그먼트만 적극 입찰

3. Mixed (3476):
   - Branding 기준 적용
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
        """다양한 예산 수준에서 최적 전략 분석"""
        scenarios = {}

        for mult in budget_multipliers:
            budget = self.base_budget * mult
            results = self._optimize_for_budget(budget)

            scenarios[f'{int(mult*100)}%'] = {
                'budget': budget,
                'expected_conversions': results['conversions'],
                'expected_revenue': results['revenue'],
                'roi': results['roi'],
                'optimal_shade': results['optimal_shade'],
            }

        return pd.DataFrame(scenarios).T

    def _optimize_for_budget(self, budget):
        """주어진 예산에서 최적 전략 찾기"""
        best_roi = 0
        best_results = None

        for shade in np.arange(0.5, 1.0, 0.05):
            results = self.simulator.simulate_policy(
                'value_based', shade=shade,
            )
            cumsum = results['expected_spend'].cumsum()
            mask = cumsum <= budget
            results_constrained = results[mask]

            if len(results_constrained) > 0:
                revenue = results_constrained['expected_click'].sum() * 200_000
                spend = results_constrained['expected_spend'].sum()
                roi = revenue / max(spend, 1)

                if roi > best_roi:
                    best_roi = roi
                    best_results = {
                        'conversions': results_constrained['expected_click'].sum(),
                        'revenue': revenue,
                        'spend': spend,
                        'roi': roi,
                        'optimal_shade': shade,
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
   - S2→S3 temporal drift (KS=0.118)
   - 과거 데이터로 미래 예측의 한계

3. 예산 소진 패턴
   - 시뮬레이션: 순차적 처리
   - 실제: 동시 다발적 입찰 기회

4. 모델 불확실성
   - Win rate, CTR 모델 오차 → V(x) 오차 → surplus 오차
   - IEB가 0에 가까울수록 안정적 (ESCM²-WC(DR) IEB 0.014)

5. Overpayment Diagnostic
   - overpayment_ratio = (bid - payprice) / payprice
   - iPinYou flat-bid baseline: 10.14× → dual-regime: 2.77×

해결책:
  - 시뮬레이션은 방향성 지표로만 활용
  - A/B 테스트로 실제 효과 검증
  - 민감도 분석 수행 (NB07 Section 10)
```

---

## Part D: 방법론별 Production 적용 가능성

### D-1. 요약 테이블

| 방법론 | 학술적 가치 | 실무 적용성 | 계산 복잡도 | 권장 |
|--------|------------|------------|------------|------|
| **Multi-Outcome CATE** | ★★★★★ | ★★★★☆ | 중간 | **Production 권장** |
| **Surplus 분해 + Mediation** | ★★★★★ | ★★★☆☆ | 중간 | 분석/인사이트 |
| **SCM/Counterfactual** | ★★★★☆ | ★★☆☆☆ | 높음 | 포트폴리오용 |
| **정책 시뮬레이션** | ★★★★☆ | ★★★★★ | 낮음 | **Production 필수** |

### D-2. Production 적용 가이드

```
Multi-Outcome CATE 적용:
1. Segment별 τ_surplus 계산 (batch, weekly/monthly)
2. τ_surplus > 0 세그먼트: 적극 입찰 (bid_adj > 1)
3. τ_surplus < 0 세그먼트: 보수적 입찰 (bid_adj < 1)
4. Lookup table로 serving: segment → bid_adj factor
5. bid(x) = V(x) × shade(x) × pace(t) × bid_adj(x)

Surplus 분해 적용:
- NIE/NDE 비율로 전략 방향 결정
  - NIE 지배: 더 많이 이기는 게 중요 → reach 우선
  - NDE 지배: 비용 절감이 중요 → shade 강화
- Advertiser taxonomy별 다른 전략

SCM 적용:
- Production에 직접 적용하지 않음
- DAG 문서화: 인과 구조 이해, 신규 feature 추가 시 참조
- Refutation test: 모델 robustness 점검

정책 시뮬레이션 적용:
1. 정기적 (주/월) 시뮬레이션 실행
2. 최적 파라미터 도출 (shade, pace, bid_adj)
3. A/B 테스트로 검증
4. Production 적용
```

---

## 산출물

| 산출물 | 경로 | 설명 |
|--------|------|------|
| CATE 분석 모듈 | `src/causal/cate.py` | Multi-outcome CATE + mediation + segment |
| SCM 모듈 | `src/causal/scm.py` | DAG + dowhy + counterfactual |
| CATE 노트북 | `notebooks/09a_cate_analysis.ipynb` | ~10 sections, multi-outcome CATE 분석 |
| SCM 노트북 | `notebooks/09b_scm_dag.ipynb` | ~7 sections, DAG + counterfactual |
| CATE Figures | `results/figures/09a_*.png` | CATE 분포, mediation, segment heatmap |
| SCM Figures | `results/figures/09b_*.png` | DAG, counterfactual comparison |

---

## 핵심 요약

1. **Multi-Outcome CATE**: Surplus 중심 4-outcome 추정 (win, payment, click, surplus) → 분해 검증
2. **V(x)는 CATE outcome 불가**: bid 무관, 모델 출력 순환. Surplus 계산의 상수로만 사용
3. **Surplus 분해**: τ_surplus ≈ V(x)·τ_win - τ_pay → point mass 문제 해결 + 메커니즘 해석
4. **Mediation**: NIE(volume: 더 많이 이겨서) vs NDE(cost: 덜 내서) → 전략 방향 결정
5. **SCM**: DAG 문서화 + model-based counterfactual (`run_auction_simulation()` 재활용)
6. **정책 시뮬레이션**: CATE-informed bid 조정 (τ_surplus 기반 bid_adj) → Production 연결
7. **한계**: point mass, collider bias, unconfoundedness, iPinYou flat-bid 제한, temporal drift
