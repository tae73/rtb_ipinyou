# Concept (1-pager)

**Motivation.** RTB에서 bidder는 *이긴* 입찰에서만 click을 본다 → winners-only로 학습한 pCTR은
selection-biased → 편향된 bid. 디바이어싱 *방법*은 이미 풀렸다(ESMM/ESCM²). 풀리지 않은 것: **그게
*언제* 입찰 의사결정에 실제로 도움이 되는가?** iPinYou 실데이터는 flat-bid·검열로 이 질문에 답할 수 없었다
(lost inventory를 못 봄).

**Mechanism.** 통제 가능한 semi-synthetic RTB testbed(실 iPinYou 시장통계 calibration; **ground-truth
pCTR + 관측 가능한 lost inventory**)에서, **모델 용량을 고정한 채**(within-capacity) 디바이어싱 효과만
분리해 측정한다 — 경쟁 baseline의 강도(linear↔GBM)·selection 강도·이질성을 축으로.

**Falsifiable claims.**
- **C1.** within-capacity IPW 디바이어싱 payoff는 *competitor-model-strength*에 의존 — 약한(linear) 모델은
  돕고(selection이 셀수록 커짐), 강한(GBM) 모델은 사라진다. (반증: 두 경우 edge가 같으면 기각.)
- **C2.** selection-biased 모델의 단순 recalibration은 truthful 입찰을 과입찰시켜 surplus를 *낮춘다*
  (recalibration trap). (반증: recal이 base보다 항상 같거나 높은 surplus면 기각.)

**Status (정직).** de-risk probe GO → within-capacity phase diagram + recal-trap 확인
(`witnesses/{phase_diagram,recal_trap}.json`, 10 seeds).
- **C1 확인:** IPW edge **+4.4pp**(linear, 강한 selection에서 +15.4pp) vs **−1.9pp**(GBM). 거대한
  **+26.3pp**는 디바이어싱이 아니라 **모델 용량**(별도 보고). 더 정교한 **DR은 IPW를 못 이김**(−2.6pp, 정직 보고).
- **C2 확인:** surplus **4.3M→3.3M**(recal) vs **5.9M**(IPW), 5/5 seeds robust.
- **★ Neural anchor (실 feature + 실 ESCM²-WC):** 첫 패스 −47pp 과입찰은 대부분 **censoring 배선 버그**
  (uncensored 합성 click을 censored-click 기대 loss에 먹임 → p_ctr 부풀림). **수정(`click·win`) 후 ESCM²-WC가
  truthful bidding을 진짜로 도움 +7.2pp**(γ 양수; n=2 seed/γ+p* non-determinism라 부호/방향만 신뢰). 일반
  calibration이 +11pp. 2×2 테스트: **selection-aware IPW는 약selection(overlap 좋을 때)에만 naive를 이기고
  강selection선 짐 → net 무승부**; 핵심 변수=overlap(ESS), 교훈=**calibration 정확도 ≠ 입찰 가치**. LR/LGB는
  음수(−4.8/−1.6). pCTR overshoot 사라짐(0.127→0.064). **2차 자기교정 — 과입찰은 data-contract 버그였지 근본
  miscalibration 아님; 직관적 "calibrate"는 작은 lift, 정교한 selection-aware는 overlap 좋을 때만 값을 함.**
- 실세계 anchor = iPinYou (robust vs LR, NOT vs LGB, I²=0.82) = phase diagram의 음성 절반과 **부호 동일**
  (메커니즘은 다름).

**Next:** [`methods.md`](methods.md) (결과), [`review.md`](review.md) (정직 scoping·선점), [`old/`](old/) (foundation).
