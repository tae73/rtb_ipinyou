# Methods & Results — When does win-selection-bias debiasing improve the bid?

**Status:** `[sketch · 합성검증]` (semi-synthetic, observable-counterfactual). Numbers are verbatim from
`witnesses/phase_diagram.json` and `witnesses/recal_trap.json` (10 seeds). Real-world anchor = the iPinYou
study in [`old/`](old/).

> **Direction-correction (정직 기록).** An earlier version of this study reported a headline
> "debiasing edge **+24.2 pp** vs a linear competitor." That number **conflated model capacity with
> debiasing** — the debiaser was a GBM and the baseline was linear, so it mixed "a GBM out-ranks an LR"
> with "debiasing helps." This version **isolates debiasing within a fixed model class** and reports the
> capacity gap *separately*. The honest debiasing effect is much smaller (and negative in places); the
> capacity gap (**+26.3 pp**) is reported as what it is — model class, **not** debiasing.

## 1. Question & claim

RTB debiasing의 *방법*(ESMM/ESCM², win-tower-as-propensity)은 이미 확립돼 있다 (Foundation). 열려 있는 것은
**decision layer**의 질문이다:

> **win-selection-bias 보정은 *언제* 입찰(decision value / surplus)을 바꾸는가 — 그리고 단순
> recalibration은 왜 full inventory에서 역효과인가?**

핵심 주장 (falsifiable):
- **(C1) Competitor-strength governs the payoff (within capacity).** 모델 용량을 고정했을 때, IPW
  디바이어싱의 입찰 가치는 *경쟁 baseline의 강도*에 의존한다 — **약한(linear) 모델은 robust하게 돕고
  selection이 셀수록 커지지만, 강한(GBM) 모델은 돕지 못한다.**
- **(C2) Recalibration trap.** selection-biased 모델을 단순 recalibration하면 truthful 입찰가가 부풀려져
  *marginal inventory*(true value < clearing price)를 더 따내며 surplus가 **하락**한다; 원칙적 IPW
  디바이어싱은 레벨 인플레이션 없이 shape를 고친다.

## 2. Testbed (controllable, observable counterfactuals)

`witnesses/phase_diagram.py` (`make_pop`). iPinYou가 검열하는 두 가지(ground-truth pCTR, lost-inventory
결과)를 **관측 가능**하게 만든 semi-synthetic DGP:
- features x ∈ ℝ⁸; **nonlinear** ground-truth pCTR `σ(−3.9 + 1.1·xβ + 0.7 x₀x₁ + 0.6(x₂²−1) + 0.5 x₃x₄)`
  (base rate ~2%) → GBM이 linear보다 의미 있게 강함 (이 비선형성이 capacity gap의 원천 — 정직히 명시).
- **market price = lognormal, iPinYou 관측 통계로 calibration** (median 68). win-selection-bias:
  `m = exp(μ + σN + γ·z)`, `z = x·(cosθ·β + sinθ·β⊥)` → 강도 γ, 이질성 θ.
- click ~ Bernoulli(pCTR) — **모든 입찰에 대해 관측**(패찰 포함). 가치/잉여는 **expected**로 평가
  (`(pctr·CPC − m)·1[bid≥m]`) → click-sampling noise 제거.
- metric = **full-inventory decision-value regret** `(S_oracle − S_model)/S_oracle` (낮을수록 좋음).

## 3. Models — isolating debiasing from capacity

각 추정량을 **같은 capacity 안에서** 비교한다 (그래야 용량이 상쇄된다): **{linear (LogisticRegression ≈ LR),
GBM (LightGBM ≈ LGB)}**. 각 cap에서:
- biased (winners-only) · biased+recalibration (cross-fit isotonic) · **IPW** (winners-only + win-propensity
  weighting, *primary*) · **DR** (imputation + IPW pseudo-label, ESCM²-style, *secondary*).
- **within-capacity edge** ≡ `regret(biased cap) − regret(debiased cap)` — 같은 모델 클래스이므로 **용량이
  상쇄되고 디바이어싱 효과만 남는다.**
- **capacity gap** ≡ `regret(linear-biased) − regret(gbm-biased)` — "GBM이 LR을 이긴다", **디바이어싱 아님.**
  별도 보고.

## 4. Result — the within-capacity phase diagram

<p align="center"><img src="witnesses/figures/fig_phase_diagram.png" width="900"></p>

`phase_diagram.json:summary` (10 seeds, primary = IPW):

| | within **linear** (≈ LR) | within **GBM** (≈ LGB) |
|---|---|---|
| **IPW debiasing edge** (mean) | **+4.4 pp** | **−1.9 pp** |
| ↳ by selection strength γ=0.4 → 0.8 → 1.2 | −0.8 → +5.0 → **+8.9 pp** | (≈0 / slightly −) |
| ↳ strong-selection cell (γ=1.2, θ=0) | **+15.4 pp** | — |
| DR debiasing edge (secondary) | **−2.6 pp** (did **not** beat IPW) | −0.3 pp |
| **model CAPACITY gap** (lin-biased − gbm-biased) | **+26.3 pp** — *NOT debiasing* | — |

→ **C1 확인 (정직 버전).** 용량을 고정하면, IPW 디바이어싱은 **약한 linear 모델을 돕고**(selection이 셀수록
커져 강한 selection에서 +15.4pp), **강한 GBM 모델은 돕지 못한다**(−1.9pp). 이 비대칭은 iPinYou fair-split
결과(*robust vs LR, NOT robust vs LGB, I²=0.82*)와 **부호가 같다** — 단 실제 메커니즘(광고주 이질성)은 다르므로
"같은 방향"이라 하고 "메커니즘 동일"이라 주장하지 않는다.
→ **정직한 음성:** 더 정교한 **DR**(진짜 imputation+IPW)을 구현했으나 이 testbed에서 **IPW를 못 이겼다**(−2.6pp).
숨기지 않고 보고한다. 그리고 거대한 capacity gap(+26.3pp)은 디바이어싱이 아니라 모델 클래스 효과다.

## 5. Result — the recalibration trap

<p align="center"><img src="witnesses/figures/fig_recal_trap.png" width="900"></p>

강한 selection + linear baseline (`recal_trap.json:linear_strong`, γ=1.2):

| | biased | + recalibration | debiased (IPW) | oracle |
|---|---|---|---|---|
| mean bid | 137.8 | **193.5** ↑ | 154.8 | — |
| unprofitable-win share | 0.495 | **0.509** | 0.493 | — |
| won surplus | 4.31M | **3.26M** ↓ | **5.92M** | 9.74M |

→ **C2 확인.** recalibration은 레벨을 올려 입찰가를 부풀리고(137.8→193.5), 낙찰의 더 많은 비율이
*unprofitable*(true value < price)이 되어 surplus가 **하락**(4.31M→3.26M). IPW 디바이어싱은 일괄 레벨
인플레이션 없이 shape를 고쳐 surplus를 **회복**(5.92M; oracle 9.74M의 61%, linear는 비선형 pCTR을 완전히 못
맞춤). **Robust:** 5 seeds 모두에서 recal은 surplus를 낮추고(5/5) IPW는 높였으며(5/5), recal 입찰 인플레는
평균 **+42.7%**. 강한 GBM baseline에선 trap이 약하다(`recal_trap.json:gbm_strong`: recal 8.96M→8.69M) — C1과 일관.

## 6. Honest scope
- `[sketch·합성검증]` — semi-synthetic. 결론은 *언제/왜*의 **특성화**이지 새 방법이 아니다.
- 헤드라인을 capacity-confound에서 **within-capacity**로 교정했고, capacity gap을 명시한다.
- 음성 영역을 보고한다: 디바이어싱은 강한 GBM을 못 이기고, DR은 IPW를 못 이겼다. 실 iPinYou anchor가 부호를 뒷받침.
- 한계·선점·확장 경로는 [`review.md`](review.md). 정본 수치는 `witnesses/*.json`, 재현은 [`repro/`](repro/).
