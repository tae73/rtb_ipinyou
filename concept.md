# Concept (1-pager)

**Motivation.** RTB에서 bidder는 *이긴* 입찰에서만 click을 본다 → winners-only로 학습한 pCTR은
selection-biased → 편향된 bid. 디바이어싱 *방법*은 이미 풀렸다(ESMM/ESCM²). 풀리지 않은 것: **그게
*언제* 입찰 의사결정에 실제로 도움이 되는가?** iPinYou 실데이터는 flat-bid·검열로 이 질문에 답할 수 없었다
(lost inventory를 못 봄).

**Mechanism.** 통제 가능한 semi-synthetic RTB testbed(실 iPinYou 시장통계 calibration; **ground-truth
pCTR + 관측 가능한 lost inventory**)에서, 경쟁 baseline의 강도(linear↔GBM)·selection 강도·이질성을 축으로
**입찰 의사결정 가치(surplus)의 디바이어싱 payoff**를 측정한다.

**Falsifiable claims.**
- **C1.** 디바이어싱 payoff는 *competitor-model-strength*에 의존 — 약한(linear) baseline 대비 robust,
  강한(GBM) baseline 대비 사라짐. (반증: 두 경우 edge가 같으면 기각.)
- **C2.** selection-biased 모델의 단순 recalibration은 truthful 입찰을 과입찰시켜 surplus를 *낮춘다*
  (recalibration trap). (반증: recal이 base보다 항상 같거나 높은 surplus면 기각.)

**Status.** de-risk probe GO(`witnesses/probe_*.json`) → phase diagram + recal-trap 확인
(`witnesses/{phase_diagram,recal_trap}.json`). C1·C2 모두 **확인**. 실세계 anchor = iPinYou
(robust vs LR, NOT vs LGB, I²=0.82) = phase diagram의 음성 절반.

**Next:** [`methods.md`](methods.md) (결과), [`review.md`](review.md) (정직 scoping·선점), [`old/`](old/) (foundation).
