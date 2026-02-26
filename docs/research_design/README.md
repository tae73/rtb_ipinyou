# iPinYou RTB 프로젝트 Research Design

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **프로젝트명** | iPinYou RTB 광고 입찰의 인과적 효과 분석 및 최적화 |
| **목표** | Win Selection Bias를 고려한 unbiased pCTR 예측 및 최적 입찰 전략 수립 |
| **핵심 방법론** | ESMM-WC / ESCM²-WC(DR) (Bid→Win→Click debiasing), Win Tower dual purpose (debiasing + bid optimization), CATE 분석 |
| **데이터셋** | iPinYou RTB Dataset (2013) |

---

## 문서 구조

```
docs/
├── README.md                    ← 현재 문서 (프로젝트 개요)
├── 00-data-preparation.md       # SP0: 데이터 준비 & EDA + Selection Bias 진단
├── 01-prediction-models.md      # SP1: pCTR/pCVR 예측 + Debiasing
├── 02-win-rate-analysis.md      # SP2: Win Rate 분석
├── 03-bid-optimization.md       # SP3: 입찰 최적화
├── 04-causal-analysis.md        # SP4: CATE / SCM / 정책 시뮬레이션
├── 05-serving.md                # SP5: 실시간 Serving
└── old/                         # 아카이브된 기존 문서
```

---

## 서브 프로젝트 의존성 (DAG)

```
SP0 (Data Prep)
    │
    ▼
SP1 (Prediction + Debiasing)
    │
    ├───────┬───────┐
    ▼       ▼       ▼
SP2      SP3      SP4
(Win)   (Bid)   (Causal)
    │       │       │
    └───┬───┘       │
        ▼           │
      SP5 ◄─────────┘
   (Serving)
```

---

## 서브 프로젝트 요약

### SP0: 데이터 준비 & EDA
- **목적**: 데이터셋 구축 및 Selection Bias 사전 진단
- **핵심 내용**: 데이터 파싱, Feature Engineering, Covariate Shift 분석
- **산출물**: 전처리 데이터, Selection Bias 진단 리포트
- **문서**: [[00-data-preparation]]

### SP1: 예측 모델 + Debiasing
- **목적**: Bid→Win→Click 퍼널에서 unbiased pCTR 예측 (CVR near-trivial → Win→Click에 집중)
- **핵심 내용**: ESMM-WC (2-tower, ESMM constraint) + ESCM²-WC(DR) (3-tower, DR debiasing), Win Tower dual purpose (debiasing + bid optimization)
- **Positivity 진단**: Win PS AUC ~0.91, overlap ~46% → IPW 단독 위험 → DR + ESMM constraint으로 완화
- **산출물**: Debiased pCTR 모델, Ablation 비교 리포트
- **문서**: [[01-prediction-models]]

### SP2: Win Rate 분석
- **목적**: Win Rate 곡선 추정 및 입찰 탄력성 분석
- **핵심 내용**: 비모수적 추정, Market Price 분포, Survival Analysis
- **산출물**: Win Rate 곡선, 탄력성 분석
- **문서**: [[02-win-rate-analysis]]

### SP3: 입찰 최적화
- **목적**: First-price 입찰 전략 및 Budget Pacing
- **핵심 내용**: Bid Shading, PID Pacing, 통합 입찰 함수
- **산출물**: FirstPriceBidder, BudgetPacer
- **문서**: [[03-bid-optimization]]

### SP4: 인과 분석
- **목적**: CATE 분석 및 정책 시뮬레이션
- **핵심 내용**: CausalForestDML, SCM, 정책 비교
- **산출물**: CATE 분석, DAG, 시뮬레이션 결과
- **문서**: [[04-causal-analysis]]

### SP5: 실시간 Serving
- **목적**: ESCM²(DR) 및 입찰 모델의 Production 배포
- **핵심 내용**: Feature Store, Model Serving, Monitoring
- **산출물**: RTBBidder, Feature Store, Monitoring
- **문서**: [[05-serving]]

---

## 데이터셋 정보 및 한계

### iPinYou 데이터셋 개요

| 항목 | 값 |
|------|-----|
| 기간 | 2013년 6월, 10월 (Season 2+3) |
| 전체 bid | 129.5M (S2: 106.6M + S3: 22.9M) |
| Won impressions | 30.6M (win rate 23.67%) |
| Clicks | 23,058 (CTR 0.0752%) |
| Conversions | 1,860 (CVR 8.07% of clicks, CTCVR 0.0061%) |
| 캠페인 수 | 9 (Branding 5, Retargeting 3, Mixed 1) |
| Market price | Median 70, Mean 80 CPM |
| Floor binding | 32.24% |

### iPinYou 데이터셋의 한계

| 항목 | iPinYou (2013) | 현재 시장 (2024+) | 영향 |
|------|----------------|-------------------|------|
| **Auction** | Second-price | First-price | Bid shading 필요 |
| **User ID** | Cookie | Privacy-preserving | Targeting 제한 |
| **Win=0 데이터** | Market price 포함 | 대부분 미제공 | Propensity 추정 어려움 |
| **Conversion** | 단일 정의 | Multi-touch | Attribution 복잡 |
| **데이터 연도** | 2013년 | - | 10년 이상 경과 |
| **Data Quality** | IVT 76 zero-win domains (5.5%), pop-up 11.4x CTR, visibility 255 sentinel | 현대 Ad Verification 기술 | 전처리 필터 필요 |

### 실무 적용 시 추가 고려사항

1. **경쟁 반응**: 시뮬레이션은 경쟁사 대응을 가정하지 않음
2. **시장 변화**: 시간에 따른 competition intensity 변화
3. **Privacy**: Cookie deprecation, Topics API 등
4. **Attribution**: Multi-touch, View-through conversion

---

## 권장 진행 순서

### Phase 1: MVP (필수)
1. **SP0**: 데이터 준비 + Selection Bias 진단
2. **SP1 (Basic)**: 기본 pCTR/pCVR 모델
3. **SP2**: Win Rate 분석
4. **SP3**: 기본 입찰 최적화

### Phase 2: Debiasing + Bid Optimization (권장)
1. **SP1 (Full)**: ESCM²(DR) 구현 + ablation (click-stage primary, two-stage secondary)
2. **SP3** ⬆️: Bid optimization (win propensity → market price → bid shading, elevated priority)
3. **Ablation Study**: Biased → ESCM²(DR) → +win-stage (marginal check)

### Phase 3: Advanced (선택)
1. **SP4**: CATE 분석 + SCM
2. **SP5**: Serving 구현

---

## 코드 구조

```
rtb_ipinyou/
├── data/ipinyou/
│   ├── raw/ipinyou/            # 원본 데이터 (.gitignore)
│   └── prediction/             # unified/, features/
├── notebooks/
│   ├── 00_*.ipynb             # SP0 노트북
│   ├── 01_*.ipynb             # SP1 노트북
│   └── ...
├── src/
│   ├── data/                   # parser.py, unifier.py
│   ├── features/               # engineering.py, usertag.py
│   ├── models/                 # base.py, esmm_wc.py, escm2_wc.py
│   ├── debiasing/              # win_propensity.py, diagnostics.py
│   ├── bidding/                # shading.py (planned)
│   ├── causal/                 # policy_simulator.py (planned)
│   └── config.py
├── scripts/                    # CLI entry points
├── results/                    # Models, figures, tables
├── mlops/                      # Serving infrastructure
└── docs/                       # 연구 설계 문서
```

---

## 참고 문헌

1. Zhang, W., et al. (2014). "Optimal Real-Time Bidding for Display Advertising." KDD.
2. Ma, X., et al. (2018). "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate." SIGIR.
3. Wang, X., et al. (2022). "ESCM²: Entire Space Counterfactual Multi-Task Model for Post-Click Conversion Rate Estimation." SIGIR.
4. Ren, K., et al. (2016). "Bid-aware Gradient Descent for Unbiased Learning with Censored Data in Display Advertising." KDD.
5. Chernozhukov, V., et al. (2018). "Double/Debiased Machine Learning for Treatment and Structural Parameters." Econometrics Journal.

---

## 관련 링크

- **프로젝트 Index**: [[index]]
- **MOC**: [[MOC-Pricing]]
