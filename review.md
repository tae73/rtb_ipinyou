# Honest scoping & positioning

`refute-by-default`. 이 노트는 주장을 띄우지 않고 *깎는다*.

## 1. What is (and is NOT) the contribution

- **NOT a method.** win-tower-as-propensity, ESMM/ESCM² IPW/DR, "debiasing이 입찰을 돕는다"는 전부
  **Foundation(선점)** 이다. 새 추정량/식별을 주장하지 않는다.
- **IS a characterization (decision layer).** "win-selection-bias 보정이 *언제* 입찰 의사결정 가치를
  바꾸는가"를, **competitor-model-strength**를 주축으로 한 regime phase diagram + **관측 가능 반사실**로
  특성화한다. 이 결과는 **real but thin** — 그래서 음성 영역을 명시하는 정직함이 본질이다.

## 2. Layered-novelty status (선점 확인 결과)

| Layer | 상태 | 근거(대표 선행) |
|---|---|---|
| Identification | **TAKEN** | MTAE (CIKM'21), Breaking Determinism (WSDM'26) — win/bid-model-as-propensity |
| Estimation | **TAKEN** | ESMM (SIGIR'18), ESCM² (SIGIR'22) — ESCM²-WC는 라벨 재인덱싱 |
| Inference | **TAKEN** | AuctionGym (KDD'23) — semi-synthetic 경매 + DR off-policy + 조건 sweep |
| **DECISION** | **OPEN (narrow)** | *언제* 디바이어싱이 입찰 가치를 바꾸는지, regime phase diagram으로 그린 것은 없음 |

## 3. Differentiate against (위협 순) — 반드시 지킬 경계

1. **BGD (Zhang & Wang, KDD'16)** — 가장 위험. 이미 win-censoring 보정이 입찰 개선(−9.3% eCPC)을 보임.
   → "디바이어싱이 입찰을 돕는다"를 **우리 기여로 주장 금지.** 우리는 *평균/1-regime*이 아니라 *언제/regime별,
   의사결정 가치로* + competitor-strength 축.
2. **AuctionGym (Jeunen et al., KDD'23)** — "AuctionGym + knob"으로 읽힐 위험. → 그들은 *estimator class*
   sweep; 우리는 *win-selection-bias-on-pCTR knob + debiasing payoff phase diagram*. 새 시뮬레이터 주장 금지.
3. **Breaking Determinism (WSDM'26)** — win-model-as-propensity + semi-synthetic + 실 A/B. → **OPE 방법으로
   pitch 금지**(레인 점유). 우리는 OPE 추정량이 아니라 특성화.
4. **MTAE (CIKM'21)** — win-tower-debiases-CTR(iPinYou 포함) 원조. → win-tower 이중활용 novelty 주장 불가; 인용.
5. **Calibration Matters (NeurIPS'22) + SPO+** — "정확도≠의사결정 가치" 동기 선점. → *발견*이 아니라 instantiate하는 *scaffold*.

## 4. Unique wedge (방어 가능한 단 하나)

**competitor-model-strength 축.** 어떤 선행도 "디바이어싱의 *입찰* 가치 ↔ 경쟁 모델 강도"를 plot하지 않았고,
우리는 그 **음성 절반을 실제 데이터(iPinYou: robust vs LR, NOT vs LGB, I²=0.82)** 로, **양성 절반을 통제
testbed**로 동시에 갖는다. 이 비대칭의 재현+설명이 wedge다.

## 5. Limitations (정직)
- semi-synthetic. DGP·시장모델·전략 선택이 결과를 좌우 — 일반화는 phase diagram의 *경계*에 한정.
- recalibration-trap은 **strong-selection / weak-baseline**에서 강하고, 강한 GBM baseline에선 약함 — 보편 법칙 아님.
- truthful 2nd-price 중심. 다른 전략/예산 제약에서 부호가 달라질 수 있음(probe에서 관측됨) — 미탐색 축.
- 디바이어싱은 *강한 GBM*을 못 이긴다 — 이건 결함이 아니라 **결과의 일부**(정직 보고).

## 6. Path to a full result (현재는 워크숍/숏 수준)
- **ESCM²-WC neural anchor**(GPU): 합성 phase diagram이 실제 신경망 디바이어로도 유지되는지.
- **Open Bandit Dataset(ZOZO)**: logged-propensity 실데이터 OPE sanity anchor (iPinYou는 P1 NO-GO).
- **small theory**: recalibration이 과입찰하는 조건의 분석적 진술(선택편향 분포 × truthful 임계).
- **strategy/budget 축** 추가 → phase diagram 완성.
