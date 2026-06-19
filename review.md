# Honest scoping & positioning

`refute-by-default`. 이 노트는 주장을 띄우지 않고 *깎는다*.

> **0. Direction-correction (정직 기록).** 초기 버전은 "디바이어싱 edge **+24.2pp** vs linear"를 헤드라인으로
> 썼다. 이는 **용량과 디바이어싱의 confound**였다(debiaser=GBM, baseline=linear → "GBM이 LR을 이김"이 섞임).
> 적대 리뷰에서 적발 → **within-capacity**(같은 모델 클래스 안에서 biased vs debiased)로 교정. 정직한 수치:
> IPW **+4.4pp**(linear, 강한 selection에서 +15.4pp) / **−1.9pp**(GBM); 용량 gap **+26.3pp**는 별도 보고
> (디바이어싱 아님). 또 진짜 **DR**을 구현했더니 이 testbed에선 **IPW를 못 이김**(−2.6pp) — 숨기지 않고 보고.

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

**within-capacity competitor-model-strength 축.** 어떤 선행도 "디바이어싱의 *입찰* 가치 ↔ 경쟁 모델 강도"를,
**용량을 통제한 채** plot하지 않았다. 우리는 그 **음성 절반을 실제 데이터(iPinYou: robust vs LR, NOT vs LGB,
I²=0.82)** 로, **양성 절반을 통제 testbed(within-capacity IPW +4.4pp linear vs −1.9pp GBM)** 로 갖는다.
**부호가 같음**을 보일 뿐 메커니즘 동일은 주장하지 않는다(실제는 광고주 이질성, 합성은 capacity-saturation).
wedge의 정직함 = 큰 겉보기 숫자(+26.3pp)가 디바이어싱이 아니라 **용량**임을 figure에서 분리해 보이는 것.

## 5. Limitations (정직)
- **capacity confound (교정 완료, §0).** 초기 +24.2pp는 용량 혼동 → within-capacity로 교정, 용량 gap 별도 명시.
- semi-synthetic. DGP·시장모델·전략 선택이 결과를 좌우 — 일반화는 phase diagram의 *경계*에 한정.
- within-capacity 디바이어싱 효과는 **작다**(linear +4.4pp 평균; 약한 selection에선 ~0/음수). 강한 selection에서만 큼.
- recalibration-trap은 **strong-selection / weak-baseline**에서 강하고, 강한 GBM baseline에선 약함 — 보편 법칙 아님.
- recalibration이 global cross-fit isotonic이라 off-support 외삽 약점 포함 — shift-aware baseline은 미탐색(§6).
- truthful 2nd-price 중심. 다른 전략/예산 제약에서 부호가 달라질 수 있음(probe에서 관측됨) — 미탐색 축.
- 디바이어싱은 *강한 GBM*을 못 이기고, 진짜 **DR도 IPW를 못 이김** — 결함이 아니라 **결과의 일부**(정직 보고).

## 5.5 Neural anchor — done, **cautionary** (실 feature + 실 모델)
- **ESCM²-WC neural anchor 완료**(`witnesses/neural_anchor.py`, GPU): iPinYou-grounded semi-synthetic
  (실 feature + p\*=실 winner click fit + market=실 payprice fit)에서 **실 신경망 ESCM²-WC**로 재검정.
- **방향교정(정직 기록):** 첫 패스 "+23.5pp 신경망 디바이어싱 edge"는 **metric 의존**이었다. 적대 리뷰가 적발 →
  ESCM²-WC가 붕괴 분산은 복원(0.056→0.198)하나 **레벨 overshoot**(mean 0.083→0.127, 1.5×) → **primary
  truthful-bid metric에선 과입찰로 음의 surplus**(neural truthful edge **−47pp** 평균; γ별 +7.7→−29→−120).
  +23.5pp는 **optimal bid-shading서만**(이 metric은 ranking이 아니라 *spread*를 보상 = 과대평가 위험).
- **honest 결론:** 디바이어싱은 *ranking*을 복원하나 over-restore(overshoot)하면 truthful bidding서 과입찰 —
  **디바이어 자체에서 재현된 C2 recalibration trap**(잘못된 레벨 → 과입찰 → 손실). 약selection선 well-calibrated→도움(+7.7pp).
- **honest 경계:** semi-synthetic(p\* surrogate·합성 selection), **truthful surplus=primary, optimal-shading은
  best-case(spread 보상)·함께 보고**, decision-value는 실데이터서 측정 불가(천장). C2 trap은 GBM서만 재현.

## 6. Path to a full result (현재는 워크숍/숏 수준)
- **real logged-propensity OPE**: Open Bandit Dataset(ZOZO) 등에서 합성 의존 제거.
- **shift-aware calibration baseline**: C2를 "naive isotonic이 나쁨"이 아니라 "shift를 무시한 recal이 나쁨"으로
  격상 (propensity-weighted calibration 대조군).
- **small theory**: recalibration이 과입찰하는 조건의 분석적 진술(선택편향 분포 × truthful 임계).
- **strategy/budget 축** 추가 → phase diagram 완성.
