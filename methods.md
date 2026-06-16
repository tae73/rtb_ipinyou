# Methods & Results — When does win-selection-bias debiasing improve the bid?

**Status:** `[sketch · 합성검증]` (semi-synthetic, observable-counterfactual). Numbers are verbatim from
`witnesses/phase_diagram.json` and `witnesses/recal_trap.json`. Real-world anchor = the iPinYou study in
[`old/`](old/).

## 1. Question & claim

RTB debiasing의 *방법*(ESMM/ESCM², win-tower-as-propensity)은 이미 확립돼 있다 (Foundation). 열려 있는 것은
**decision layer**의 질문이다:

> **win-selection-bias 보정은 *언제* 입찰(decision value / surplus)을 바꾸는가 — 그리고 단순
> recalibration은 왜 full inventory에서 역효과인가?**

핵심 주장 (falsifiable):
- **(C1) Competitor-strength governs the payoff.** 디바이어싱의 입찰 가치는 *경쟁 baseline의 강도*에
  의존한다 — **약한(linear) baseline 대비 robust, 강한(GBM) baseline 대비 사라진다.**
- **(C2) Recalibration trap.** selection-biased 모델을 단순 recalibration하면 truthful 입찰가가 부풀려져
  *marginal inventory*(true value < clearing price)를 더 따내며 surplus가 **하락**한다; DR 디바이어싱이 원칙적 해법.

## 2. Testbed (controllable, observable counterfactuals)

`witnesses/phase_diagram.py` (`make_pop`). iPinYou가 검열하는 두 가지(ground-truth pCTR, lost-inventory
결과)를 **관측 가능**하게 만든 semi-synthetic DGP:
- features x ∈ ℝ⁸; **nonlinear** ground-truth pCTR `σ(−3.9 + 1.1·xβ + 0.7 x₀x₁ + 0.6(x₂²−1) + 0.5 x₃x₄)`
  (base rate ~2%) → GBM이 linear보다 의미 있게 강함.
- **market price = lognormal, iPinYou 관측 통계로 calibration** (median 68). win-selection-bias:
  `m = exp(μ + σN + γ·z)`, `z = x·(cosθ·β + sinθ·β⊥)` → 강도 γ, 이질성 θ.
- click ~ Bernoulli(pCTR) — **모든 입찰에 대해 관측**(패찰 포함). 가치/잉여는 **expected**로 평가
  (`(pctr·CPC − m)·1[bid≥m]`) → click-sampling noise 제거.
- metric = **full-inventory decision-value regret** `(S_oracle − S_model)/S_oracle` (낮을수록 좋음).

## 3. Models — the competitor-strength axis

각 추정량을 두 capacity로: **{linear (LogisticRegression ≈ LR), GBM (LightGBM ≈ LGB)}**.
- oracle (true pCTR) · biased (winners-only) · biased+recalibration (cross-fit isotonic) · debiased
  (winners-only + **IPW**, win-propensity 가중). debiaser는 강한 GBM+IPW로 고정(≈ iPinYou neural).
- edge ≡ `regret(baseline) − regret(debiaser)` (>0 = 디바이어싱이 그 baseline을 이김).

## 4. Result — the phase diagram

<p align="center"><img src="witnesses/figures/fig_phase_diagram.png" width="900"></p>

평균 디바이어싱 edge (`phase_diagram.json:summary`):

| competitor baseline | mean edge | γ=0.4 → 1.2 |
|---|---|---|
| **linear (≈ LR)** | **+24.2 pp** | +16 → +45 pp (강도와 함께 증가) |
| **GBM (≈ LGB)** | **−1.9 pp** | 거의 0 / 약간 음수 |

→ **C1 확인.** 디바이어싱은 약한 linear baseline을 robust하게 이기지만(강도가 셀수록 더), 강한 GBM
baseline은 **이기지 못한다.** 이는 iPinYou fair-split 결과(*robust vs LR, NOT robust vs LGB, I²=0.82*)를
**통제된 환경에서 재현·설명**한다 — 그 실제 결과가 이 phase diagram의 *음성 절반*이다.

## 5. Result — the recalibration trap

<p align="center"><img src="witnesses/figures/fig_recal_trap.png" width="900"></p>

강한 selection + linear baseline (`recal_trap.json:linear_strong`):

| | biased | + recalibration | debiased (DR) | oracle |
|---|---|---|---|---|
| mean bid | 137.8 | **193.5** ↑ | 135.1 | — |
| unprofitable-win share | 0.495 | **0.509** | **0.219** | — |
| won surplus | 4.3M | **3.3M** ↓ | **8.6M** | 9.7M |

→ **C2 확인.** recalibration은 레벨을 올려 입찰가를 부풀리고(137.8→193.5), 절반 이상의 낙찰이
*unprofitable*(true value < price)이 되어 surplus가 **하락**(4.3M→3.3M). DR 디바이어싱은 일괄 레벨
인플레이션 없이 shape를 고쳐 oracle(9.7M)의 88%(8.6M)에 도달. (강한 GBM baseline에선 trap이 약함 —
`recal_trap.json:gbm_strong`.)

## 6. Honest scope
- `[sketch·합성검증]` — semi-synthetic. 결론은 *언제/왜*의 **특성화**이지 새 방법이 아니다.
- 음성 영역을 명시한다(GBM baseline 대비 디바이어싱은 도움이 안 됨). 실제 iPinYou anchor가 이를 뒷받침.
- 한계·선점·확장 경로는 [`review.md`](review.md). 정본 수치는 `witnesses/*.json`, 재현은 [`repro/`](repro/).
