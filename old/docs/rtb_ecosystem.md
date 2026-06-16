# RTB 생태계 & 경매 메커니즘

iPinYou RTB 프로젝트의 도메인 배경지식. Second-price에서 first-price 경매로의 전환이 이 프로젝트의 bid shading 모듈(SP3)이 존재하는 근본적 이유이다.

---

## 1. RTB 생태계 참여자

### 1-1. 주요 참여자와 역할

| 참여자 | 역할 | 예시 |
|--------|------|------|
| **Publisher** | 웹페이지/앱의 광고 지면(inventory) 보유 | 뉴스 사이트, 블로그 |
| **Ad Server (DFP/GAM)** | Publisher 측 광고 관리 및 우선순위 결정 | Google Ad Manager |
| **SSP** (Supply-Side Platform) | Publisher의 inventory를 경매에 제출 | Google AdX, Rubicon |
| **Ad Exchange** | 실시간 경매 수행, 입찰 중개 | Google AdX, OpenX |
| **DSP** (Demand-Side Platform) | Advertiser를 대신하여 실시간 입찰 수행 | **iPinYou**, The Trade Desk |
| **Advertiser** | 광고 집행 의뢰, 예산/타겟/목표 설정 | 이커머스, 브랜드 |
| **DMP** (Data Management Platform) | 사용자 데이터 수집/세분화, 타겟팅 지원 | Oracle BlueKai, Lotame |

### 1-2. 생태계 구조

```
Advertiser ──▶ DSP (iPinYou) ──▶ Ad Exchange ◀── SSP ◀── Publisher
    │              │                   │            │         │
    │         입찰 전략           실시간 경매      지면 제출   광고 지면
    │         pCTR 예측           낙찰 결정       floor price   보유
    │         bid shading         win/lose 통보    설정
    │              │                   │
    └── 예산/목표 ─┘                   │
                                       │
                DMP ──────────────────▶ │
              (사용자 데이터,            │
               타겟팅 세그먼트)          │
                                       ▼
                                    User에게
                                    광고 노출
```

### 1-3. iPinYou 데이터셋과의 매핑

iPinYou는 **DSP** (Demand-Side Platform) 역할:

| 생태계 개체 | iPinYou 데이터 필드 | 설명 |
|-------------|---------------------|------|
| DSP 자체 | iPinYou 시스템 | 입찰 로그를 생성한 주체 |
| Advertiser | `advertiser` (ID) | 9개 캠페인의 광고주 |
| Ad Exchange | `adexchange` (1~5) | 경매를 중개한 거래소 |
| Publisher | `domain` (해시) | 광고 지면을 제공한 퍼블리셔 |
| User | `userid`, `usertag` | 광고 노출 대상 사용자 |
| 광고 슬롯 | `slotwidth`, `slotheight`, `slotvisibility`, `slotformat` | 지면의 물리적 특성 |

---

## 2. 하나의 입찰이 처리되는 흐름

### 2-1. End-to-End 흐름

```
[1] User 페이지 방문
     │
     ▼
[2] Publisher Ad Server: 광고 슬롯 탐지, SSP/Exchange에 경매 요청
     │
     ▼
[3] Ad Exchange → DSP들에게 Bid Request 전송 (JSON/Protobuf)
     │            ┌──────────────────────────────────────┐
     │            │ Bid Request 포함 정보:                │
     │            │  - user_id, user_agent, ip            │
     │            │  - 슬롯 크기/위치/도메인              │
     │            │  - floor_price (최저 입찰가)           │
     │            └──────────────────────────────────────┘
     ▼
[4] DSP (iPinYou) 내부 처리 ──── ⏱ 전체 100ms 이내
     │  ├─ User lookup (DMP/쿠키 기반)
     │  ├─ Feature 구성 (시간, 슬롯, 지역, 경쟁 등)
     │  ├─ pCTR 예측 (모델 추론)
     │  ├─ Value 계산: V(x) = pCTR × CPC_target
     │  ├─ Bid shading: shade(x)        ← SP3
     │  └─ 최종 입찰가: bid = V × shade × pace
     │
     ▼
[5] Bid Response → Ad Exchange
     │
     ▼
[6] Ad Exchange: 경매 수행 → 낙찰자 결정
     │  ├─ 낙찰 (Win)  → 광고 노출 → 클릭? → 전환?
     │  └─ 패찰 (Lose) → 기회 손실
     │
     ▼
[7] Win Notification → DSP에 결과 통보 (지불가 포함)
```

### 2-2. iPinYou 로그와의 매핑

| 흐름 단계 | iPinYou 로그 | 데이터 | 관측 조건 |
|-----------|-------------|--------|-----------|
| [3] Bid Request 수신 | `bid.*.txt` | 전체 입찰 로그 (129.5M) | 항상 관측 |
| [6] 낙찰 (Win) | `imp.*.txt` | 노출 로그 (30.6M) | `win=1`일 때만 |
| [6→] 클릭 | `clk.*.txt` | 클릭 로그 (23K) | `win=1 & click=1` |
| [6→] 전환 | `conv.*.txt` | 전환 로그 (1.86K) | `win=1 & click=1 & conversion=1` |

**핵심 관측**: Lost bids(`win=0`)에서는 `payprice`, `click`, `conversion`이 관측 불가능. 이것이 **Win Selection Bias**의 원인이며, 이 프로젝트의 ESCM²-WC(DR) 모델이 해결하려는 문제.

### 2-3. 실시간 처리 요구사항

| 구간 | 제한 시간 | 설명 |
|------|-----------|------|
| Bid Request → Response | **100ms** | DSP가 응답해야 하는 최대 시간 |
| DSP 내부 처리 | **<50ms** (목표) | Feature 조회 + 모델 추론 + 입찰 결정 |
| Network latency | ~30-50ms | Exchange ↔ DSP 간 왕복 |

→ `mlops/serving/app.py`의 FastAPI + ONNX Runtime 설계 근거 (SP5)

---

## 3. Second-Price Auction

### 3-1. Vickrey (1961) 메커니즘

Second-price sealed-bid auction (Vickrey auction):
- 최고 입찰자가 낙찰
- **지불가 = 2등 입찰가** (자신의 입찰가가 아님)

```
예시:
  DSP A 입찰: 300 CPM  ← 낙찰
  DSP B 입찰: 250 CPM  ← 2등
  DSP C 입찰: 200 CPM

  결과:
    낙찰자: DSP A
    지불가: 250 CPM (2등 입찰가)
    Surplus: 300 - 250 = 50 CPM (DSP A의 이윤)
```

### 3-2. Truthful Bidding이 Dominant Strategy인 이유

Second-price에서는 자신의 진정한 가치(value)대로 입찰하는 것이 최적:

```
DSP의 가치: v = 300 CPM

Case 1: Truthful (bid = v = 300)
  - 2등이 250이면 → 낙찰, 지불 250, 이윤 50 ✓
  - 2등이 350이면 → 패찰, 이윤 0 (적절한 결과) ✓

Case 2: Underbid (bid = 200 < v)
  - 2등이 250이면 → 패찰 (이윤 50을 놓침) ✗
  - 낙찰 기회 감소, 지불가는 어차피 2등가이므로 절약 효과 없음

Case 3: Overbid (bid = 400 > v)
  - 2등이 350이면 → 낙찰, 지불 350, 이윤 -50 (손해) ✗
  - 가치 이상을 지불할 위험
```

**결론**: 어떤 전략에서든 `bid = v`가 weakly dominant. 입찰가는 낙찰 여부만 결정하고, 지불가에는 영향 없음.

### 3-3. iPinYou 데이터에서의 bidprice vs payprice

| 필드 | 의미 | iPinYou 컬럼 |
|------|------|-------------|
| `bidprice` | DSP(iPinYou)의 입찰가 | `bidprice` (CPM) |
| `payprice` | 실제 지불가 (= 2등 입찰가) | `payprice` (CPM, win=1에서만 관측) |
| Surplus | DSP의 이윤 | `bidprice - payprice` (≥ 0) |

```
Second-price 특성 (iPinYou 데이터):
  payprice ≤ bidprice   (항상 성립)
  평균 surplus > 0       (DSP에 유리)
```

### 3-4. iPinYou가 Second-Price인 이유

iPinYou 데이터셋은 **2013년** 수집. 이 시점에는 디스플레이 광고 시장이 거의 전적으로 second-price auction을 사용:

- Google DoubleClick Ad Exchange: second-price (2019년까지)
- OpenX, Rubicon 등 주요 거래소: second-price
- 업계 표준이 Vickrey auction 기반

→ `payprice < bidprice` 관계가 데이터에 일관되게 나타남

---

## 4. Waterfall → Header Bidding 전환 (2015~)

### 4-1. 기존 Waterfall 구조

2015년 이전 Publisher의 광고 판매 방식:

```
Publisher Ad Server (DFP)
     │
     ▼
[Priority 1] Direct sold (보장 계약)
     │ 안 팔리면 ↓
     ▼
[Priority 2] Ad Exchange A (highest historical CPM)
     │ 안 팔리면 ↓
     ▼
[Priority 3] Ad Exchange B (2nd highest)
     │ 안 팔리면 ↓
     ▼
[Priority 4] Ad Network C (remnant)
     │ 안 팔리면 ↓
     ▼
[Fallback] House ad / 빈 슬롯
```

**Waterfall의 한계:**
- **순차 호출**: 상위 Exchange가 응답할 때까지 하위 Exchange는 기회 없음
- **과거 CPM 기반 우선순위**: 실시간 수요를 반영하지 못함
- **수익 손실**: Exchange B가 더 높은 가격을 제시할 수 있어도 기회를 얻지 못함
- **Latency 누적**: 순차 호출로 인한 페이지 로딩 지연

### 4-2. Header Bidding의 등장 (2015~)

Publisher의 `<header>` 태그에 JavaScript를 삽입하여 **여러 Exchange에 동시 호출**:

```
Publisher 페이지 로드
     │
     ▼
Header Bidding Wrapper (Prebid.js 등)
     │
     ├──▶ Ad Exchange A ──▶ bid: 280 CPM
     ├──▶ Ad Exchange B ──▶ bid: 310 CPM  ← 최고가
     ├──▶ Ad Exchange C ──▶ bid: 250 CPM
     │         (병렬 호출)
     ▼
최고 bid (310) → DFP에 key-value로 전달
     │
     ▼
DFP: Header Bidding 최고가 vs Direct sold 비교 → 최종 결정
```

**Header Bidding의 장점:**
- **병렬 경쟁**: 모든 Exchange가 동시에 경쟁 → Publisher 수익 증가 (평균 +30~50%)
- **실시간 가격 발견**: 과거 CPM이 아닌 실시간 입찰가 기반
- **투명성**: Publisher가 각 Exchange의 입찰가를 직접 확인

### 4-3. Header Bidding이 Second-Price의 Incentive Compatibility를 깨뜨린 이유

Header Bidding 도입 이후, second-price auction의 이론적 전제가 무너짐:

**문제 1: 2단계 경매 중첩**

```
[1단계] 각 Exchange 내부: Second-price auction
         Exchange A 내부 낙찰가: 280 CPM (2등: 260)
         Exchange B 내부 낙찰가: 310 CPM (2등: 290)

[2단계] Header Bidding: Exchange 간 First-price 비교
         Exchange A 제출: 280 CPM
         Exchange B 제출: 310 CPM  ← 최종 낙찰

         문제: Exchange B는 310을 지불 (1단계의 2등가 290이 아님)
         → 2단계에서 first-price 로직이 적용됨
```

내부적으로 second-price이지만, Exchange 간 경쟁은 사실상 first-price처럼 동작. DSP 입장에서는 "어느 Exchange를 통해 입찰하느냐"에 따라 최종 지불가가 달라지는 비효율 발생.

**문제 2: Exchange의 bid manipulation**

Exchange들이 경쟁에서 이기기 위해 다양한 조작을 수행:

| 조작 방식 | 설명 | 결과 |
|-----------|------|------|
| Soft floor | 공식 floor 외에 동적 최저가 부과 | payprice 인위적 상승 |
| Bid inflation | DSP 입찰가를 Exchange가 상향 조정 | 2단계에서 유리하지만 DSP에 초과 청구 |
| Last-look | 자사 DSP에게 최고 입찰가 정보 공유 | 공정 경쟁 훼손 |

**문제 3: Truthful bidding이 더 이상 최적이 아닌 구조**

```
이론적 전제 (단일 second-price):
  입찰가 → 낙찰 여부만 결정
  지불가 = 2등 입찰가 (나의 입찰과 무관)
  → Truthful bidding 최적

Header Bidding 환경 (2단계 중첩):
  입찰가 → 낙찰 여부 + 실질 지불가에 영향
  Exchange가 입찰가를 조작 가능
  → Truthful bidding이 과잉 지불 유발
  → DSP들이 전략적으로 underbid 시작
```

---

## 5. First-Price Auction 전환 (2017~2019)

### 5-1. 전환 타임라인

| 시기 | 이벤트 |
|------|--------|
| 2017 | Index Exchange, first-price 전환 시작 |
| 2018 | AppNexus, Rubicon 등 주요 Exchange 전환 |
| 2019.03 | **Google AdX first-price 전환** (시장 결정적 전환점) |
| 2019 하반기 | 업계 대부분 first-price로 전환 완료 |

### 5-2. 전환 동기

| 동기 | 설명 |
|------|------|
| **투명성** | "입찰한 만큼 지불" — 중간 조작 여지 제거 |
| **단순성** | 2단계 경매 중첩 문제 해소, 가격 결정 로직 명확 |
| **공정성** | Exchange의 bid manipulation 인센티브 제거 |
| **예측 가능성** | DSP가 지불가를 정확히 예측 가능 |

### 5-3. DSP에 새로운 과제: Bid Shading 필요성

```
Second-price (과거):
  bid = value = 300 CPM (truthful)
  payprice = 250 CPM (2등가)
  surplus = 50 CPM ← 자동으로 확보

First-price (현재):
  bid = value = 300 CPM (truthful bid)
  payprice = 300 CPM (= 자신의 입찰가)
  surplus = 0 CPM ← 모든 이윤 상실!

  → Bid Shading 필수:
  bid = value × shade = 300 × 0.8 = 240 CPM
  payprice = 240 CPM
  surplus = 60 CPM (shading으로 이윤 확보)
  단, win probability 감소 → 최적 shade factor 필요
```

**iPinYou 데이터에서의 Overpayment 실태:**
- iPinYou는 flat-bid 전략 사용 → 평균 ~76% overpayment (bid vs market price)
- Market price median 70 CPM, mean 80 CPM에 비해 과도한 입찰
- Bid shading이 없을 경우의 비효율성을 정량적으로 보여주는 baseline

**Bid Shading의 핵심 트레이드오프:**

```
max_b  E[utility] = (value - b) × P(win | b)
                      ─────────   ──────────
                       margin      win prob

  b ↑ → win prob ↑, margin ↓
  b ↓ → win prob ↓, margin ↑

  최적 b*: 두 효과의 균형점 (FOC로 결정)
```

### 5-4. 이 프로젝트의 SP3 Bid Shading 모듈의 존재 이유

First-price 전환으로 인해 DSP에게 필수가 된 bid shading을 구현하는 것이 SP3의 핵심:

```
SP1 → SP3 연결:

1. ESCM²-WC(DR) → Debiased pCTR
   → V(x) = debiased_pCTR × CPC_target

2. Win Tower (AUC ~0.91) → P(Win | X, bid)
   → Market price CDF 추정
   → shade(x) = optimal_bid / V(x)

3. 최종 입찰:
   bid(x) = V(x) × shade(x) × pace(t)
```

Win Tower가 **이중 역할**을 수행하는 구조:

| 역할 | 용도 | SP |
|------|------|-----|
| Debiasing propensity | CTR 예측의 selection bias 보정 (DR/IPW weights) | SP1 |
| Win rate model | Market price CDF 추정 → bid shading factor 계산 | SP3 |

→ `src/bidding/shading.py`, `src/models/escm2_wc.py` Win Tower 참조

---

## 6. iPinYou 데이터셋의 위치

### 6-1. Second-Price 시대 데이터의 한계

iPinYou (2013)는 second-price 경매 데이터이므로, first-price 최적화 연구에 직접 사용 시 한계:

| 항목 | Second-price (iPinYou) | First-price (현재 시장) | 영향 |
|------|----------------------|----------------------|------|
| 지불가 | `payprice` = 2등 입찰가 | `payprice` = 자신의 bid | 가격 분포 상이 |
| 입찰 행태 | Truthful bidding 경향 | Strategic bidding (shading) | 입찰 분포 상이 |
| Surplus | `bidprice - payprice > 0` | Surplus = 0 (shading 전) | 경제성 평가 불가 |
| Market price | 2등 입찰가로 관측 가능 | 경쟁 입찰가 비공개 | 분포 추정 방법 상이 |

### 6-2. 그럼에도 iPinYou를 사용하는 이유

| 이유 | 설명 |
|------|------|
| **Win=0 데이터 포함** | Lost bids의 market price가 관측 가능 — 실무에서는 거의 불가능 |
| **Selection Bias 연구** | Bid→Win→Click 퍼널의 selection bias를 연구하기에 이상적 |
| **규모** | 129.5M bids, 30.6M impressions — 통계적으로 충분 |
| **학술 표준** | RTB 연구의 de facto benchmark dataset |

### 6-3. First-Price 시뮬레이션 접근법

iPinYou의 second-price 데이터를 first-price 환경으로 변환하는 전략:

```
Second-price 데이터에서 추출 가능:
  - Win=0 bids의 market price → 경쟁 분포 추정
  - Win=1의 payprice → 2등 입찰가 분포 (market price proxy)

First-price 시뮬레이션:
  1. Market price 분포 추정 (Win Tower 활용)
  2. 가상 first-price 경매 환경 구축
  3. Bid shading 전략 테스트
  4. ROI 비교 (truthful vs shaded)

⚠ 주의:
  - Second-price의 입찰 행태 ≠ First-price의 입찰 행태
  - 경쟁자 반응(game-theoretic effects)이 반영되지 않음
  - 결과는 참고용, 실무 배포 전 A/B 테스트 필수
```

→ `docs/research_design/03-bid-optimization.md` Part B (Bid Shading) 및 Part F (시뮬레이션) 참조

---

## 참고 문헌

- Vickrey, W. (1961). "Counterspeculation, Auctions, and Competitive Sealed Tenders." Journal of Finance.
- Zhang, W., et al. (2014). "Optimal Real-Time Bidding for Display Advertising." KDD.
- Yuan, S., et al. (2013). "Real-Time Bidding for Online Advertising: Measurement and Analysis." ADKDD.
- Despotakis, S., et al. (2021). "First-Price Auctions in Online Display Advertising." Journal of Marketing Research.
