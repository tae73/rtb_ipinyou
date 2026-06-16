# Glossary / 용어집

Single source of truth for terminology across the English and Korean docs. Numbers are illustrative of
this project's results; authoritative values live in [`NUMBERS_LEDGER.md`](NUMBERS_LEDGER.md).

영어/한국어 문서 전반의 용어 단일 기준. 표기는 **영어 용어 — 한국어 용어**, 정의는 양 언어 병기.

## Core problem / 핵심 문제

| Term — 용어 | Definition (EN) | 정의 (KO) |
|---|---|---|
| **Win selection bias** — 승리 선택 편향 | Clicks are observed only on bids that won the auction, so `P(click \| win) ≠ P(click)`. | 낙찰된 입찰에서만 클릭이 관측되어 `P(click \| win) ≠ P(click)` 이 되는 편향. |
| **Bid → Win → Click funnel** — 입찰→낙찰→클릭 퍼널 | The cascade a bid passes through; only wins become impressions, only impressions can click. | 입찰이 거치는 단계; 낙찰만 노출이 되고, 노출만 클릭될 수 있다. |
| **Censored / lost inventory** — 검열된/패찰 인벤토리 | Outcomes of lost bids are unobserved (payprice/click censored). | 패찰 입찰의 결과(지불가격·클릭)는 관측되지 않음. |

## Models & debiasing / 모델·디바이어싱

| Term — 용어 | Definition (EN) | 정의 (KO) |
|---|---|---|
| **ESMM-WC** | 2-tower model (Win + CTR) with an ESMM joint constraint for implicit debiasing. | Win·CTR 2-tower에 ESMM 결합 제약을 둔 암묵적 디바이어싱 모델. |
| **ESCM²-WC** | 3-tower model (Win + CTR + Imputation) with explicit DR/IPW debiasing. | Win·CTR·Imputation 3-tower에 명시적 DR/IPW 디바이어싱을 둔 모델. |
| **Propensity** — 성향점수 | `P(win \| x)`, the probability a bid wins; the debiasing weight's denominator. | `P(win \| x)`, 입찰의 낙찰 확률; 디바이어싱 가중치의 분모. |
| **IPW** (inverse propensity weighting) — 역성향가중 | Reweight observed (won) samples by `win / P(win)` to undo selection. | 관측(낙찰) 표본을 `win / P(win)`로 재가중해 선택을 보정. |
| **DR** (doubly robust) — 이중 강건 | Combines IPW with an imputation model; unbiased if *either* is correct. | IPW와 대체(imputation) 모델 결합; 둘 중 하나만 맞아도 비편향. |
| **Win / CTR / Imputation Tower** — Win/CTR/Imputation 타워 | The three heads of ESCM²-WC: propensity, debiased pCTR, CTR-error model. | ESCM²-WC의 세 헤드: 성향점수, 비편향 pCTR, CTR 오차 모델. |
| **Dual-purpose Win Tower** — Win 타워 이중 활용 | The same Win Tower serves as (a) debiasing propensity and (b) bid-shading win-rate model. | 동일 Win 타워를 (a) 디바이어싱 성향점수, (b) 비드 셰이딩 낙찰률 모델로 함께 사용. |

## Calibration / 캘리브레이션

| Term — 용어 | Definition (EN) | 정의 (KO) |
|---|---|---|
| **IEB** (integrated expected bias) | Normalized mean(pred) − mean(true); 0 = mean-calibrated. | 정규화된 평균(pred) − 평균(true); 0이면 평균 캘리브레이션. |
| **ECE** (expected calibration error) | Binned gap between confidence and accuracy. | 신뢰도와 정확도의 구간별 차이. |
| **Cross-fit isotonic recalibration** — 교차적합 등위 재보정 | Leak-free (K-fold out-of-fold) monotone map that fixes calibration while preserving ranking. | 누수 없는(K-fold OOF) 단조 매핑으로, 순위를 보존하며 캘리브레이션을 교정. |
| **Segment (per-advertiser) calibration** — 세그먼트(광고주별) 보정 | A separate isotonic map per advertiser, closing per-segment residual bias. | 광고주별 별도 등위 매핑으로 세그먼트 잔차 편향을 제거. |

## Ranking & bidding / 랭킹·입찰

| Term — 용어 | Definition (EN) | 정의 (KO) |
|---|---|---|
| **Winners-only AUC** — 낙찰자 한정 AUC | AUC of `P(click \| win)` on won samples — the object a bidder ranks on. | 낙찰 표본에서 `P(click \| win)`의 AUC — 입찰자가 실제로 정렬하는 대상. |
| **All-bids AUC** — 전체입찰 AUC | AUC over all bids; inflated by "easy negatives" (lost bids), not the bidding object. | 전체 입찰에 대한 AUC; 패찰(easy negative)로 부풀려지며 입찰 대상이 아님. |
| **Second-price auction** — 2차가격 경매 | Winner pays the market clearing price (≤ its bid); truthful bidding is optimal. | 낙찰자가 시장 청산가(≤ 입찰가)를 지불; 진실 입찰이 최적. |
| **Realized surplus** — 실현 잉여 | `Σ (click·CPC − payprice)` over re-won impressions; the decision-level metric. | 재낙찰 노출에 대한 `Σ (click·CPC − payprice)`; 의사결정 수준 지표. |
| **Bid shading** — 비드 셰이딩 | Bidding below value to trade win rate against cost; uses the win-rate model. | 낙찰률과 비용을 절충하려 가치보다 낮게 입찰; 낙찰률 모델을 사용. |
| **Truthful bidding** — 진실 입찰 | Bidding `b = V` (the 2p-optimal strategy here). | `b = V`로 입찰 (본 연구의 2차가격 최적 전략). |

## Inference & power / 추론·검정력

| Term — 용어 | Definition (EN) | 정의 (KO) |
|---|---|---|
| **Advertiser-cluster bootstrap** — 광고주 클러스터 부트스트랩 | Resamples *advertisers* (not rows) for honest CIs under within-advertiser correlation. | 행이 아닌 *광고주*를 재표집해 광고주 내 상관 하에서 정직한 CI 산출. |
| **Cochran's Q / I²** — 코크란 Q / I² | Heterogeneity test / fraction of variance that is between-cluster (I²=0.82 ⇒ heterogeneous). | 이질성 검정 / 클러스터 간 분산 비율 (I²=0.82 ⇒ 이질적). |
| **LOAO** (leave-one-advertiser-out) — 광고주 1개 제외 | Refit the cluster mean dropping each advertiser; reveals single-advertiser leverage. | 광고주를 하나씩 빼고 클러스터 평균 재계산; 단일 광고주 레버리지 노출. |
| **MDE** (minimum detectable effect) — 최소검출효과 | Smallest effect the design could detect at 80% power (~11.5M here ≫ observed mean). | 80% 검정력에서 검출 가능한 최소 효과 (여기선 ~11.5M ≫ 관측 평균). |
| **ESS / overlap (positivity)** — 유효표본수 / 중첩(양의확률) | Effective sample size after weighting / propensity common support; low ESS ⇒ prefer DR over IPW. | 가중 후 유효표본수 / 성향점수 공통지지; ESS 낮으면 IPW보다 DR 선호. |
| **Fair split** — 공정 split | Per-advertiser temporal split with shared advertiser/creative vocabulary (vs the artifact-prone disjoint split). | 광고주/크리에이티브 어휘를 공유하는 광고주별 시간 분할 (아티팩트를 유발하는 분리 split과 대비). |
