# 디바이어싱은 *언제* 입찰을 바꾸는가?

[🇺🇸 English](README.md) · 🇰🇷 **한국어**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-baselines-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-strong%20competitor-02A4D3)
![Method](https://img.shields.io/badge/contribution-characterization%2C%20not%20a%20method-7B42BC)
![Status](https://img.shields.io/badge/status-sketch%C2%B7synthetic--verified-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **Decision-layer 질문.** RTB의 win-selection-bias 디바이어싱은 이미 *풀린 방법*이다
> (ESMM/ESCM², win-tower-as-propensity). 열린 질문은 **그게 *언제* 입찰을 실제로 바꾸는가** — AUC가 아니라
> 의사결정 가치로 — **그리고 편향 모델을 단순 recalibration하면 왜 full inventory에서 역효과인가**이다.
> 통제 가능한 **semi-synthetic testbed**(ground-truth pCTR + *관측 가능한* lost inventory, 실 iPinYou 시장
> 통계로 calibration)로 **regime phase diagram**을 그린다. wedge는 **competitor-model-strength** 축이고,
> 실제 iPinYou 결과가 그 *음성 절반*이다.

<p align="center">
  <img src="witnesses/figures/fig_phase_diagram.png" alt="Phase diagram: 약한 linear 경쟁자 대비 디바이어싱 edge는 커지지만 강한 GBM 대비 사라진다 — iPinYou anchor가 음성 절반에 위치" width="900">
</p>

---

### 🧭 시간으로 탐색

| ⏱️ 30초 | 🔎 5분 | 🧪 30분 | ♻️ 재현 |
|---|---|---|---|
| [TL;DR](#-tldr-30초) · [주장 한눈에](#-주장-한눈에) | [질문](#질문--decision-layer) · [C1 phase diagram](#c1--경쟁자-강도가-payoff를-지배) · [C2 recalibration trap](#c2--recalibration-trap) | [`methods.md`](methods.md) · [`review.md`](review.md) · [정직한 scope](#-정직한-scope) | [`MANIFEST.md`](MANIFEST.md) · [`repro/`](repro/) · [Foundation → `old/`](#foundation--ipinyou-연구-old) |

---

## ⏱️ TL;DR (30초)

bidder는 *이긴* 입찰에서만 click을 본다. winners로만 pCTR을 학습 → selection bias → 편향된 입찰.
디바이어싱은 *모델*을 고친다 — 그러나 광고주에게 중요한 질문은 **그게 입찰 의사결정을 바꾸는가, 그리고
언제인가**이다. iPinYou 실데이터는 이 질문에 답할 수 없다(flat-bid 로깅이 lost inventory를 검열). 그래서
lost inventory가 **관측 가능한** testbed를 만들어 regime을 sweep한다.

- **C1 — 경쟁자 강도가 payoff를 지배.** 디바이어싱의 입찰 edge는 *약한 linear* 경쟁자 대비 의사결정 regret
  **+24.2 pp**(selection이 강할수록 증가)이지만 *강한 GBM* 대비 **−1.9 pp** — 강한 baseline은 **못 이긴다.**
  이는 실제 iPinYou 결과(디바이어싱 robust vs LR, **NOT** vs LGB, I²=0.82)를 **재현·설명**한다: 그 결과가
  이 diagram의 *음성 절반*이다.
- **C2 — Recalibration trap.** 편향 모델을 그냥 recalibration하면 **입찰가가 부풀고**(137.8 → 193.5),
  *unprofitable* inventory를 더 따내(share 0.495 → 0.509) surplus가 **하락**(4.3M → 3.3M)한다. 원칙적인 DR
  디바이어싱은 레벨 인플레 없이 *shape*를 고쳐 **8.6M ≈ oracle의 88%**를 회복한다.

> **기여는 새 방법이 아니라 characterization** — *real but thin* — 이라서, 디바이어싱이 **도움이 안 되는**
> 영역을 명시하는 것이 본질이다. 모든 수치는 committed witness JSON에 고정([`MANIFEST.md`](MANIFEST.md)),
> [`repro/`](repro/) harness가 재확인한다.

## 🎯 주장 한눈에

| # | 주장 | 헤드라인 | Witness | 상태 |
|---|---|---|---|---|
| C1 | 디바이어싱 edge는 경쟁자 강도에 의존 | **+24.2 pp** vs linear · **−1.9 pp** vs GBM | `witnesses/phase_diagram.json` | **확인** (합성) |
| C2 | recalibration이 marginal inventory를 과입찰 | surplus **4.3M → 3.3M**(recal) vs **8.6M**(DR) | `witnesses/recal_trap.json` | **확인** (합성) |
| — | de-risk probe (full-inventory regret) | **12/12** 셀에서 디바이어싱 regret 낮음 | `witnesses/probe_debiasing_bidding_value.json` | **GO** |
| ⚓ | 실세계 anchor (음성 절반) | robust vs LR, **NOT** vs LGB, **I²=0.82** | [`old/`](old/) (iPinYou fair-split) | canonical |

---

## 질문 — decision layer

디바이어싱 *방법*은 **Foundation**(선점)이다: win-tower-as-propensity (MTAE, CIKM'21),
ESMM/ESCM² 추정량 (SIGIR'18/'22), "디바이어싱이 평균적으로 입찰을 돕는다" (BGD, KDD'16). 어떤 선행도 그
보정이 **언제** 입찰 *의사결정*을 바꾸는지, 그리고 **얼마나 강한** 경쟁자 대비인지를 plot하지 않았다.
이것이 유일하게 열린 층이고, 실제 iPinYou 프로젝트가 천장에 부딪힌 지점이다: flat-bid 로깅이 lost inventory를
검열해, 결정적 반사실("다른 입찰가였다면 이겼을까, 그게 가치 있었나?")이 셀의 87%에서 **관측 불가**다.

그래서 이를 **통제 가능한 semi-synthetic testbed**([`methods.md`](methods.md))로 옮긴다: features
x ∈ ℝ⁸ → **nonlinear** ground-truth pCTR; **market price = iPinYou median(68 CPM)에 calibration한 lognormal**;
win-selection-bias knob (강도 γ, 이질성 θ); 그리고 — 핵심 — **click + price가 모든 입찰(낙찰·패찰)에 대해
관측된다.** metric = full-inventory **decision-value regret** `(S_oracle − S_model)/S_oracle`.

## C1 — 경쟁자 강도가 payoff를 지배

각 추정량을 두 capacity로: **linear (≈ LR)**, **GBM (≈ LGB)** — 이것이 *competitor-strength 축*이다.
debiaser는 고정(GBM + IPW, win-propensity 가중 ≈ iPinYou neural debiaser).
`edge ≡ regret(baseline) − regret(debiaser)`.

| 경쟁 baseline | 평균 edge | γ = 0.4 → 1.2 |
|---|---|---|
| **linear (≈ LR)** | **+24.2 pp** | +16 → +45 pp (selection 강도와 함께 증가) |
| **GBM (≈ LGB)** | **−1.9 pp** | ≈ 0 / 약간 음수 |

디바이어싱은 *약한* linear 경쟁자를 robust하게 이기지만(selection이 셀수록 더), **강한 GBM은 못 이긴다.**
이것이 정확히 iPinYou fair-split 결론 — *robust vs LR, NOT robust vs LGB, Cochran Q I²=0.82* — 을
통제된 환경에서 **재현·메커니즘적으로 설명**한 것이다. 실제 결과는 diagram의 **음성 절반**이고, testbed가
통제 가능한 양성 절반을 공급한다.

## C2 — Recalibration trap

솔깃한 해법 — 편향 모델을 recalibration — 은 **함정**이다. 강한 selection + linear baseline
(`recal_trap.json:linear_strong`, γ=1.2):

<p align="center">
  <img src="witnesses/figures/fig_recal_trap.png" alt="recalibration이 입찰가를 부풀려 unprofitable inventory를 더 따내 surplus가 하락; DR 디바이어싱은 oracle의 ~88% 회복" width="900">
</p>

| | biased | + recalibration | debiased (DR) | oracle |
|---|---|---|---|---|
| mean bid | 137.8 | **193.5** ↑ | 135.1 | — |
| unprofitable-win share | 0.495 | **0.509** | **0.219** | — |
| won surplus | 4.3M | **3.3M** ↓ | **8.6M** | 9.7M |

recalibration은 예측의 *레벨*을 올려 입찰가를 부풀리고(137.8 → 193.5), true value < clearing price인
*marginal* inventory를 따내 낙찰의 절반 이상이 unprofitable이 되며 surplus가 **하락**(4.3M → 3.3M)한다.
DR 디바이어싱은 레벨 인플레 없이 *shape*를 고쳐 **oracle의 88%**(8.6M / 9.7M)에 도달한다. trap은
*약한 baseline / 강한 selection*에서 강하고 강한 GBM 대비(`recal_trap.json:gbm_strong`)에선 약하다 — C1과 일관.

---

## 🔬 정직한 scope

이 연구는 `[sketch · 합성검증]`이다. 솔직히 말하면([`review.md`](review.md)):

- **방법이 아니다.** 새 추정량/식별 결과가 아니다 — 그것들은 Foundation. 기여는 디바이어싱이 *언제/왜* 입찰을
  바꾸는지를 **competitor-strength** 축에서 **특성화**한 것이다.
- **음성 영역을 숨기지 않고 보고한다.** 디바이어싱은 강한 GBM을 **못 이기고**, recalibration이 보편적으로
  나쁜 것도 아니다. 실제 iPinYou anchor(음성 절반)도 동등한 비중으로 싣는다.
- **Semi-synthetic.** 일반화는 phase diagram의 *경계*에 한정, truthful 2nd-price 중심.
- **Full result로 가는 경로:** ESCM²-WC neural anchor (GPU), Open-Bandit-Dataset OPE sanity anchor,
  recalibration 과입찰 조건의 작은 정리, strategy/budget 축.

## Foundation — iPinYou 연구 (`old/`)

동기가 된 선행 연구는 [**`old/`**](old/)에 있다: 실 iPinYou RTB 데이터의 **win-selection-bias 디바이어싱
포트폴리오** 전체 — ESCM²-WC (doubly-robust 3-tower), calibration 스토리(global IEB 0.597 → 0; 광고주별
잔차 → 0.0006), 그리고 **정직한 연구 arc**(첫 AUC 헤드라인을 split artifact로 retraction). 그 결정적이고
*정직한* 결론 — **디바이어싱의 입찰 가치는 linear LR 대비 robust지만 강한 GBM 대비 NOT robust (I²=0.82)** —
가 바로 이 testbed가 재현하는 실세계 **음성 절반**이다. 거기서의 데이터 천장(검열된 lost inventory)이 이
연구를 관측 가능한 testbed로 옮긴 *이유*다.

## 저장소 맵

```
README.md / README.ko.md   ← flagship front (EN / KO twin)
concept.md                 ← 1-pager: 동기 → 검증 가능 주장
methods.md                 ← testbed · models · results (수치 = witness JSON)
review.md                  ← 정직한 scoping · 선행연구 위치 · full paper 경로
MANIFEST.md                ← canonical 맵: witness JSON → 주장 → figure
witnesses/                 ← 본 연구
  phase_diagram.py + .json     C1 (competitor-strength sweep)
  recal_trap.py + .json        C2 (recalibration trap)
  probe_*.py + .json           de-risk probe (GO, 12/12)
  figures/make_figures.py      JSON → figure
repro/check.py             ← green harness: JSON에서 C1 + C2 재확인
old/                       ← Foundation: iPinYou 디바이어싱 연구 (실세계 anchor)
```

## ♻️ 재현

```bash
pip install -r requirements.txt
python witnesses/phase_diagram.py      # → witnesses/phase_diagram.json
python witnesses/recal_trap.py         # → witnesses/recal_trap.json
python witnesses/figures/make_figures.py
python repro/check.py                  # GREEN — canonical JSON에서 C1 + C2 재확인
```

모든 수치는 witness JSON에서 verbatim, figure·표는 거기서 재생성된다. 주장 → witness → figure 맵은
[`MANIFEST.md`](MANIFEST.md) 참조.
