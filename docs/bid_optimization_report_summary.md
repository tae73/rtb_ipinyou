# First-Price Bid Optimization for RTB — iPinYou

## Executive Summary

본 프로젝트는 iPinYou RTB 데이터셋의 **4,228,166건 won impression**에 대해 First-price 입찰 최적화를 수행하였다. SP1에서 달성한 **debiased pCTR** (ESCM²-WC(DR), IEB 0.014)을 기반으로 impression value를 계산하고, **Kaplan-Meier CDF** 기반 bid shading과 **PID budget pacing**을 결합하여 `bid(x) = V(x) × shade(x) × pace(t)` 모듈형 입찰 시스템을 구현하였다. iPinYou flat-bid의 surplus **-805M** CPM에서 dual-regime shading의 **+128M** CPM으로 **933M CPM 개선**을 달성하며, win rate를 100%에서 38%로 줄이는 대가로 ROI를 0.74에서 3.26으로 **4.4배** 향상시켰다.

### Key Results

| 핵심 성과 | 수치 |
|----------|------|
| iPinYou Flat Surplus | -805.1M CPM (ROI 0.74, Overpayment 10.14×) |
| **Dual-Regime Surplus** | **+127.7M CPM (ROI 3.26, Overpayment 2.77×)** |
| Surplus 개선 | **+933M CPM** (negative → positive) |
| Floor-Binding 활용 | Won impression의 50.7%가 floor 가격에 binding |
| Exchange 이질성 | F(300) = 11.9%~68.8%, 경쟁도 2~8× 차이 |
| Debiasing 연결 | IEB 0.014 → near-oracle; IEB 0.362 → 25.9× overbid |
| Budget Pacing | PID 99%+ utilization, WR-weighted 시간대 배분 |

| 분석 단계 | 핵심 발견 | 비즈니스 임팩트 |
|----------|----------|----------------|
| Value 계산 | V(x) mean 96.9 CPM, 52.6% > market median | Impression-level 가치 차별화 |
| Bid Shading | Dual-regime: floor-aware + CDF-optimal | Surplus 933M CPM 개선 |
| Exchange 분석 | F(300) 11.9%~68.8%, 경쟁도 2~8× 차이 | Exchange-conditional shading +1.8M gain |
| Budget Pacing | Hourly WR U-shape (8.6%~43%) | 경쟁 낮은 시간대 집중 투자 |

### Approach & Technical Highlights

| 단계 | 방법론 | 핵심 기술 | Output |
|------|--------|----------|--------|
| Value | V(x) = pCTR × CPC_target | ESCM²-WC(DR) debiased pCTR | Impression value 분포 |
| Shading | (V−b) × F(b) 최대화 | KM CDF, grid search, dual-regime | Optimal bid per impression |
| Exchange | Exchange-conditional CDF | Exchange별 KM survival | 차별화된 shading factor |
| Pacing | PID controller | Kp/Ki/Kd feedback, WR-weighted | Hourly multiplier |

**프로젝트 차별점:**
- **End-to-End Pipeline**: SP1 debiased pCTR → V(x) → shade(x) → pace(t) 모듈형 설계
- **EDA-Driven Innovation**: Floor-binding 발견 (50.7%) → dual-regime 전략 도출
- **Calibration-Revenue Chain**: IEB → overbidding → surplus까지 end-to-end 정량화

---

## Motivation & Framework

First-price auction에서 DSP의 핵심 과제는 "이 impression에 얼마를 지불할 것인가"를 실시간으로 결정하는 것이다. iPinYou의 flat-bid 전략 (277/294 CPM)은 market price 중앙값 68 CPM 대비 **4배 이상 overpayment**이 발생하며, surplus가 **-805M CPM** (적자)이다. Bid shading은 이 문제를 해결한다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BID OPTIMIZATION FRAMEWORK                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PROBLEM: First-Price Overpayment                                           │
│  ═════════════════════════════════                                           │
│                                                                             │
│  iPinYou Flat Bid (277/294 CPM) vs Market Median (68 CPM)                   │
│  → 10.14× overpayment, Surplus -805M CPM, ROI 0.74                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SOLUTION: bid(x) = V(x) × shade(x) × pace(t)                              │
│  ═══════════════════════════════════════════════                              │
│                                                                             │
│  Step 1: V(x) = debiased_pCTR × CPC_target    [SP1 Output]                 │
│  Step 2: shade(x) = b*/V, argmax (V−b)×F(b)   [KM CDF from SP2]           │
│  Step 3: pace(t) = PID multiplier              [Budget control]             │
│                                                                             │
│  Output: Surplus +128M CPM, ROI 3.26, Overpayment 2.77×                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VALUE CHAIN: Debiasing → Value → Shading → Surplus                         │
│  ═════════════════════════════════════════════════════                        │
│                                                                             │
│  SP1 (IEB 0.014) → V(x) accurate → Optimal shade → Near-oracle surplus     │
│  SP1 (IEB 0.362) → V(x) inflated → 25.9× overbid → Surplus 손실           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**왜 Bid Optimization인가?**

| 측면 | iPinYou Flat Bid | Optimized Bidding |
|------|-----------------|-------------------|
| 입찰 방식 | 277/294 CPM 고정 | V(x) × shade(x) × pace(t) |
| Win Rate | 100% (전수 낙찰) | 37.9% (가치 있는 impression만) |
| Surplus | -805M CPM (적자) | **+128M CPM** (흑자) |
| Overpayment | 10.14× | **2.77×** |
| ROI | 0.74 | **3.26** |
| 예산 효율 | 무차별 지출 | PID + WR-weighted 시간대 배분 |

---

## Key Insights: 핵심 발견

### 1. Win Rate vs Surplus Trade-off

| 전략 | Win Rate | Surplus (M CPM) | ROI | 해석 |
|------|----------|-----------------|-----|------|
| iPinYou Flat | **100%** | **-805.1** | 0.74 | 전수 낙찰, 대규모 적자 |
| Truthful (V=bid) | 51.8% | +17.7 | 1.56 | 마진 거의 없음 |
| **Dual-Regime** | 37.9% | **+127.7** | **3.26** | 가치 있는 impression만 낙찰 |

Win rate를 100%에서 38%로 줄이면서 surplus를 933M CPM 개선. **"모든 impression을 이기는 것"보다 "이길 가치가 있는 impression만 이기는 것"이 압도적으로 효율적**.

### 2. Floor-Binding Discovery → Dual-Regime Innovation

Won impression의 **50.7%**에서 market price가 floor price에 근접 (floor-binding). 이 EDA 발견을 전략에 반영하여 floor-bound impression에서는 floor 바로 위에 입찰 (overpayment 최소화), competitive impression에서는 CDF 기반 optimal shading을 적용. 결과적으로 **최고 surplus (+127.7M)과 최저 overpayment (2.77×)**을 동시 달성.

### 3. Debiasing → Bidding 인과 체인 (SP1 → SP3 연결)

| 모델 | IEB | Overbid 배수 | Surplus 영향 |
|------|-----|-------------|-------------|
| ESCM²-WC(DR) | **0.014** | 1.0× (oracle) | Near-oracle surplus |
| LR CTR_all | 0.122 | 8.7× | -2.2% surplus 손실 |
| LGB CTR (biased) | 0.362 | 25.9× | -2.2% surplus 손실 |
| ESMM-WC | 1.335 | 95× | -14.7% surplus 손실 |

SP1에서 달성한 **IEB 0.014 calibration이 SP3 bidding surplus를 결정**한다. AUC가 높아도 calibration이 나쁘면 V(x)를 과대추정하여 overbidding이 발생하며, **AUC best ≠ Bidding best**.

---

## Methodology

### Analysis Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                   BID OPTIMIZATION PIPELINE                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  [SP1: ESCM²-WC(DR)]      [SP2: KM CDF]     [EDA: Floor]     │
│     debiased pCTR             F(b)             50.7% binding   │
│         │                      │                    │          │
│         ▼                      ▼                    ▼          │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────┐      │
│  │ V(x) = pCTR  │   │ Optimal Bid  │   │ Dual-Regime   │      │
│  │  × CPC_target│   │ argmax       │   │ Floor vs      │      │
│  │              │   │ (V−b)×F(b)   │   │ Competitive   │      │
│  └──────┬───────┘   └──────┬───────┘   └───────┬───────┘      │
│         │                  │                    │              │
│         └──────────┬───────┘────────────────────┘              │
│                    ▼                                           │
│         ┌──────────────────┐                                   │
│         │ bid(x) = V×shade │                                   │
│         │     × pace(t)    │                                   │
│         └────────┬─────────┘                                   │
│                  ▼                                              │
│         [PID Budget Pacing]                                    │
│         → Hourly multiplier                                    │
│         → WR-weighted allocation                               │
│                  ▼                                              │
│         [Offline Auction Simulation]                           │
│         → 4.23M won impressions                                │
│         → 7 strategies compared                                │
└────────────────────────────────────────────────────────────────┘
```

### Bid Shading Theory

First-price auction에서 입찰자의 기대 이윤(expected surplus):

```
Expected Surplus = (V − bid) × P(Win | bid) = (V − b) × F(b)
```

최적 bid는 마진(V−b)과 낙찰 확률 F(b)의 곱을 최대화하는 지점이다:

```
b* = argmax_b  (V − b) × F(b),   min_bid ≤ b ≤ V
```

F(b)는 **Kaplan-Meier** 추정으로 구한다 — won impression은 event, lost impression은 right-censored 관측. 1,000개 bid 후보 grid search로 argmax 계산 (~6초 / 4.2M impressions).

> 상세 수식 및 KM CDF 도출: [bid_optimization_report.md §3](bid_optimization_report.md#3-방법론-bid-shading) 참조

### 비교 전략

| # | 전략 | 설명 | 출처 |
|---|------|------|------|
| 1 | iPinYou Flat | 277/294 CPM 고정 | Baseline |
| 2 | Truthful | bid = V(x) | 경제 이론 |
| 3 | Linear α=0.8 | bid = 0.8 × V(x) | Ou et al. Eq.5 |
| 4 | Linear α=0.6 | bid = 0.6 × V(x) | 민감도 분석 |
| 5 | Optimal KM | argmax (V−b) × F_km(b) | Ou et al. Sec 6.2.1 |
| 6 | Optimal Exchange | Exchange별 CDF 적용 | 본 프로젝트 |
| 7 | **Dual-Regime** | Floor-aware + competitive | 본 프로젝트 (EDA-driven) |

### Dual-Regime 설계

```
if floor-bound (slotprice > 0 and slotprice > V × 0.1):
    bid = floor_price × 1.05    ← floor 바로 위 (overpayment 최소화)
else:
    bid = optimal_bid(V, CDF)   ← 표준 distribution-based shading
```

- **Floor-bound regime (50.7%)**: Market price ≈ floor → floor 위에 최소 입찰
- **Competitive regime (49.3%)**: Floor 없거나 미미 → KM CDF 기반 optimal shading

### Budget Pacing

**PID Controller** (Ou et al., 2024, Sec 5.1.2):

```
error(t) = ideal_spent(t) − actual_spent(t)
ϕ = Kp × error + Ki × Σerror + Kd × Δerror/Δt
multiplier = clip(1 + ϕ / norm, [0.3, 2.0])
```

**WR-weighted allocation**: 시간대별 win rate U-shape (새벽 43% → 오후 8.6%)를 활용하여 경쟁이 낮은 시간대에 예산 집중.

---

## Results Summary

### 핵심 전략 비교

| 전략 | Win Rate | Clicks | Surplus (M CPM) | Overpayment | Avg CPC (K) | ROI |
|------|----------|--------|-----------------|-------------|-------------|-----|
| **iPinYou Flat** | **100.0%** | **4,482** | **-805.1** | 10.14× | 271.0 | 0.74 |
| Truthful | 51.8% | 2,266 | +17.7 | 5.79× | 127.8 | 1.56 |
| Linear α=0.8 | 45.1% | 1,980 | +63.7 | 5.25× | 108.4 | 1.84 |
| Linear α=0.6 | 37.1% | 1,702 | +100.1 | 4.58× | 83.9 | 2.38 |
| Optimal KM | 33.2% | 1,480 | +119.0 | 3.44× | 59.5 | **3.36** |
| Optimal Exchange | 33.7% | 1,486 | +120.8 | 3.38× | 61.7 | 3.24 |
| **Dual-Regime** | 37.9% | 1,608 | **+127.7** | **2.77×** | 61.3 | 3.26 |

![Strategy Comparison](../results/figures/07_strategy_comparison.png)
*7개 bidding 전략의 핵심 지표 비교. Dual-regime이 가장 높은 surplus와 가장 낮은 overpayment을 달성한다.*

### Exchange-Conditional 분석

| Exchange | F(300) | Median | 경쟁 수준 | 최적 전략 |
|----------|--------|--------|---------|----------|
| Exchange 1 | 68.8% | 153 CPM | 약 | 공격적 shading 가능 |
| Exchange 2 | 29.1% | ∞ | 중간 | 중간 수준 shading |
| Exchange 3 | 11.9% | ∞ | 강 | 보수적, 높은 bid 필요 |
| Exchange 4 | N/A | N/A | N/A | Overall CDF fallback |

Exchange 1에서 300 CPM 입찰로 68.8% 낙찰되지만, Exchange 3에서는 11.9%에 불과하다. Exchange-conditional shading은 전략 5 (Optimal KM) 대비 **+1.8M CPM surplus gain**.

![Exchange Shading](../results/figures/07_exchange_shading.png)
*Exchange별 KM CDF 비교와 exchange-conditional optimal bid. 경쟁도에 따라 2~8배 shading 차이.*

### Dual-Regime 상세

Floor-bound impression (50.7%)에서 floor 바로 위에 입찰하여 overpayment 최소화:

- Optimal KM 대비: Win rate +4.7%p, Surplus +8.7M, Overpayment −0.67
- **Highest surplus + Lowest overpayment 동시 달성**

![Dual Regime](../results/figures/07_dual_regime.png)
*Floor-bound vs Competitive regime의 market price 분포. 50.7%의 floor-binding이 dual-regime 전략의 근거.*

### Alpha 민감도 & Pareto Frontier

| α | Win Rate | Clicks | Surplus (M CPM) | ROI |
|---|----------|--------|-----------------|-----|
| 0.4 | 27.4% | 1,332 | 117.7 | **3.44** |
| 0.5 | 32.6% | 1,516 | 112.4 | 2.77 |
| 0.6 | 37.1% | 1,702 | 100.1 | 2.38 |
| 1.0 | 51.8% | 2,266 | 17.7 | 1.56 |

Pareto optimal zone: α = 0.4~0.5. 그러나 Optimal KM (ROI 3.36, surplus 119M)이 linear α=0.4 (ROI 3.44, surplus 118M)보다 clicks 148개 더 확보하며 Pareto frontier에서 dominant.

![Pareto Frontier](../results/figures/07_pareto_frontier.png)
*Clicks vs Surplus Pareto frontier. Optimal KM과 Dual-Regime이 frontier 상에서 dominant.*

### Debiasing의 경제적 영향

Calibration error (IEB)가 V(x)를 왜곡하여 overbidding으로 이어지는 인과 체인:

![Debiasing Impact](../results/figures/07_debiasing_impact.png)
*IEB가 bidding surplus에 미치는 영향. IEB 증가 시 V(x) 과대추정 → surplus 급격 감소.*

**핵심**: SP1에서 달성한 ESCM²-WC(DR)의 IEB 0.014가 near-oracle surplus를 보장한다. Biased LGB baseline (IEB 0.362)은 25.9배 더 많은 overbidding을 유발한다.

### Advertiser별 분석

| Advertiser | 유형 | Dual-Regime Surplus (M) | Dual-Regime ROI | 특징 |
|------------|------|------------------------|-----------------|------|
| 2259 | Retargeting | 7.3 | 0.59 | 높은 CTR |
| 2261 | Branding | 35.1 | 0.90 | 대량 volume |
| 2821 | Retargeting | 47.8 | 1.42 | **최대 절대 surplus** |
| 2997 | High-value | 37.4 | **9.98** | **최고 ROI** |

Advertiser 2997은 높은 CTR로 V(x)가 크고 shading 여지가 많아 ROI 9.98. Advertiser 2821은 volume이 커서 절대 surplus 최대.

![Advertiser Comparison](../results/figures/07_advertiser_comparison.png)
*Advertiser별 bidding 전략 비교. 각 advertiser의 특성에 따라 최적 전략의 효과가 상이.*

### Budget Pacing

시간대별 win rate는 뚜렷한 **U-shape 패턴**: 새벽 ~43% (경쟁 약) → 오후 ~8.6% (경쟁 최고). WR-weighted pacing은 이 패턴을 활용하여 경쟁 낮은 시간대에 예산 집중.

![Hourly Traffic](../results/figures/08_hourly_traffic_pattern.png)
*시간대별 bid volume, win rate, CTR. Win rate의 U-shape 패턴이 WR-weighted pacing의 근거.*

| Daily Budget | Wins | Clicks | Surplus (K CPM) | Utilization | ROI |
|-------------|------|--------|-----------------|-------------|-----|
| 50K CPM | 504 | 0 | 48.3 | 99.8% | — |
| 200K CPM | 2,131 | 0 | 186.6 | 99.9% | — |
| 1M CPM | 11,782 | 2 | 838.9 | 100.0% | 0.40 |
| **Unlimited** | **1,402,972** | **1,480** | **118,988.4** | **100%** | **3.36** |

PID controller는 모든 budget level에서 **99%+ 예산 활용률**. 예산 증가에 따른 수확 체감 패턴 관찰.

![Budget Sensitivity](../results/figures/08_budget_sensitivity.png)
*Budget level별 clicks, surplus, ROI 변화. 예산 증가에 따른 diminishing returns.*

---

## Production Considerations

**Production 입찰 공식:**

```
bid(x) = V(x) × shade(x) × pace(t)
         ────   ────────   ────────
         ① 가치   ② 최적 할인  ③ 예산 조절
```

| Module | Source | Core Function | Latency |
|--------|--------|--------------|---------|
| Value | `src/bidding/value.py` | `compute_impression_values()` | <1ms |
| Shading | `src/bidding/shading.py` | `dual_regime_shading()` | <5ms |
| Pacing | `src/bidding/pacing.py` | `compute_pid_multiplier()` | <1ms |
| Simulator | `src/bidding/simulator.py` | `run_auction_simulation()` | Offline only |

**추천 Production 설정:**

| 구성 요소 | 선택 | 근거 |
|----------|------|------|
| CTR 모델 | ESCM²-WC(DR)+ExtPS | IEB 0.045, near-oracle surplus |
| Shading | Dual-Regime | Highest surplus, lowest overpayment |
| Pacing | PID + WR-weighted | 99%+ utilization, 시간대 최적화 |

3개 모듈이 독립적이므로 각각 개별 개선/교체 가능 (modular design).

---

## Limitations & Lessons Learned

| 한계 | 증거 | 완화책 |
|------|------|--------|
| **정적 경쟁 가정** | Offline sim, 경쟁자 반응 미반영 | Game-theoretic equilibrium 분석 |
| **KM CDF Heavy Censoring** | F(500) = 21.3%, right tail 불확실 | max_bid = 300 CPM cap |
| **Won-Only Simulation** | Lost bids 76% market_price 미관측 | KM sampling 기반 full sim |
| **Second-Price 데이터** | iPinYou 2013 = SP auction | FP 결과는 근사값 |
| **제한된 Advertiser** | 4개 test advertiser, 2013 중국 | 외부 데이터 검증 필요 |

### 교훈

> "iPinYou flat-bid의 -805M CPM surplus에서 dual-regime의 +128M CPM으로의 전환은,
> SP1 debiasing (IEB 0.014) → accurate V(x) → optimal shade(x) → surplus maximization이라는
> **end-to-end value chain**의 실현이다. Win rate를 62%p 줄이는 대가로 surplus를 933M CPM 개선한
> 이 결과는 '모든 impression을 이기는 것'보다 '이길 가치가 있는 impression만 이기는 것'이
> 압도적으로 효율적임을 보여준다."

### 향후 방향

1. **RL 기반 동적 bidding**: Multi-agent RL로 경쟁자 반응 모델링 (Ou et al., 2024, Sec 5.2)
2. **Online Evaluation**: Off-policy evaluation / A/B testing 프레임워크
3. **Budget-Constrained Optimization**: Lagrangian dual method로 예산 제약 하 최적 bidding
4. **Production Serving**: FastAPI + ONNX Runtime 실시간 추론 (<50ms P95)

---

## Technical Reports

상세 방법론, 실험 설계, 부록은 다음을 참조한다:

- **[Bid Optimization Report (Full)](bid_optimization_report.md)**: 전체 리포트 (7개 전략 상세, advertiser별 결과, KM CDF 통계)
- **[Prediction Report Summary](prediction_report_summary.md)**: SP1 Debiasing 요약 (IEB → overbidding → surplus chain)
- **[Performance Tuning Log](performance_tuning.md)**: 20단계 성능 튜닝 상세 로그

---

## Notebooks

| Notebook | 분석 내용 |
|----------|----------|
| `07_bid_optimization` | Value 분포, bid shading 이론, 7 전략 시뮬레이션, exchange 분석, dual-regime |
| `08_budget_pacing` | PID controller, WR-weighted allocation, budget sensitivity |

---

## Implementation Modules

| 모듈 | 경로 | 핵심 함수 | 역할 |
|------|------|----------|------|
| Value | `src/bidding/value.py` | `compute_impression_values()` | V(x) = pCTR × CPC_target |
| Shading | `src/bidding/shading.py` | `optimal_bid_vectorized()`, `dual_regime_shading()` | b* = argmax (V−b)×F(b) |
| Pacing | `src/bidding/pacing.py` | `compute_pid_multiplier()`, `simulate_pacing()` | PID budget control |
| Simulator | `src/bidding/simulator.py` | `run_auction_simulation()`, `compare_strategies()` | Offline auction engine |

---

## Data Source

**iPinYou RTB Dataset** — 2013년 중국 DSP의 실제 RTB 로그

| 항목 | 값 |
|------|-----|
| 총 Bids | 129.5M (Season 2+3) |
| Impressions (Won) | 30.6M (Win Rate 23.67%) |
| **Test Set** | 19.4M bids, 4.23M won, 4,482 clicks |
| iPinYou 입찰 방식 | Flat-bid: 277/294 CPM (2종류) |
| Market Price | Median 68, Mean 78, P90 166 CPM |
| Floor Price | Median 40 CPM, 32.24% floor binding |
| Ad Exchange | 4개 (Exchange 1~4) |
| Test Advertisers | 4개 (2259, 2261, 2821, 2997) |
| CPC Target | 200,000 CPM/click |
