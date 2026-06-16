# RTB iPinYou Project Progress Tracking

**Last Updated**: 2026-06-16 (**PIVOT → decision-layer 연구로 repo 전면 재front**)

> ## 🧭 현재 방향 (2026-06-16 pivot)
> iPinYou 디바이어싱 포트폴리오는 데이터 천장(flat-bid·검열 lost-inventory·5광고주)으로 "특별한 긍정 결과"를
> 못 냈다(정직 결론: 선형 LR 대비 robust, 강 GBM 대비 NOT robust, I²=0.82). 그래서 **decision-layer**로 피벗:
> *win-selection-bias 디바이어싱은 **언제** 입찰(decision value)을 바꾸는가 — 그리고 단순 recalibration은 왜
> full inventory에서 역효과인가?* 통제 가능한 **semi-synthetic testbed**(관측 가능한 lost-inventory)로
> regime **phase diagram**을 그렸고 wedge는 **competitor-model-strength** 축이다. iPinYou 실결과는 그 *음성 절반*.
> - **C1**(competitor-strength governs payoff): edge **+24.2pp** vs linear, **−1.9pp** vs GBM — `witnesses/phase_diagram.json`
> - **C2**(recalibration trap): surplus **4.3M→3.3M**(recal) vs **8.6M**(DR) — `witnesses/recal_trap.json`
> - 새 front = `README.md`/`README.ko.md` + `concept/methods/review/MANIFEST.md` + `witnesses/` + `repro/`.
> - **과거 iPinYou 포트폴리오 전체 → `old/`** (Foundation·실세계 anchor). 아래 로그는 그 foundation의 기록.

---

## 🔄 전면 재설계 (2026-06-14~) — falsification-first, cost-conscious

**중단 사유 재진단:** ① **성능 이슈 = 헤드라인 반증.** 20-phase 실험(performance_tuning.md)이 사실상 de-risk probe였고, neural debiasing(ESMM/ESCM²-WC DR)은 예측 AUC에서 LR baseline(0.7687)을 못 이김. 지배 요인 = **temporal shift S2→S3**. 단 **calibration→bidding surplus** 가치는 생존. ② **자원 문제 = 금전 비용**(개인 구매 새 서버). → **GPU-hour 최소화**가 1급 제약.

**Redesign north star:** debiasing+**calibration** 최우선 → **데이터 시나리오 재검증 선행** → **temporal shift 정면돌파**. 범위 = 예측+bidding 핵심 + serving 데모(production 가능성 시연). 상세 계획: `~/.claude/plans/generic-prancing-aho.md`.

**진행 상황:**
- [x] **Step 0a 환경 (conda env `rtb_ipinyou`, python 3.12, uv 설치):** JAX[cuda12] 0.10.1 **2× L40S 인식**, flax 0.12.7 등 + serving/dev extras, editable 패키지. **`PYTHONNOUSERSITE=1`**(conda config var)로 `~/.local`(169 pkg, sklearn 1.8.0 등) shadowing 차단 → 격리 복구. **`scikit-learn==1.6.1`**(저장 아티팩트와 일치 + econml 호환). `uv.lock`(154 pkg, 전 extra resolve) + `requirements.lock`(105). 구 `.venv` 제거.
- [x] **Foundation(scaffolding):** `tests/` 신규 **74개 테스트 전부 green**(models·metrics·**calibration 회귀**·bidding·**data_loader batch invariant**·mesh·**serving**), `Makefile`, `pyproject.toml`(lean core 14 + extras nn-gpu/nn-cpu/serving/causal/tracking/dev + pytest 설정), `src/distributed/mesh.py` **`pick_devices()`**(GPU 경합 대응 — 점유 GPU 회피), `scripts/verify_data.py`(데이터 무결성+MANIFEST, 업로드 완료 시 실행 대기).
- [x] **Serving 데모(E4, production 가능성 시연):** `src/serving/app.py` FastAPI — 실모델 `lr_ctr_all.joblib`(StandardScaler+LR, **train/serve skew 0** 검증) → `compute_impression_values` → exchange-conditional CDF dual-regime shading → `POST /bid`. **P95 1.3ms**(<100ms). adversarial verify가 저가치 impression `bid>value` 버그 발견·수정 + 회귀 테스트. `mlops/serving/{Dockerfile,README.md}`.
- [x] **재현성 해결:** sklearn 1.6.1 핀으로 아티팩트 version warning 제거 + econml/uv.lock 충돌 해소(154 pkg lock 생성). 74 테스트 green, serving version warning 없음. `Makefile`은 conda env(+`PYTHONNOUSERSITE=1`) 기준으로 갱신.
- [ ] **Step 0b 데이터 검증:** 데이터 업로드 진행 중(rsync) → 완료 시 `verify_data.py` → MANIFEST freeze (HARD gate).
- [x] **Stage A 데이터 재검증 — 핵심 발견(2026-06-14):** Step 0b 무결성 freeze(MANIFEST) 완료. Phase 1(재학습-0 probe)이 "debiasing 부정 결과"를 냈으나, **근본원인 감사**로 그 부정이 **아티팩트**임을 규명: (i) **adversarial disjoint-advertiser split**(train S2 ∩ test S3 광고주 = ∅; "LR 0.71"은 미관측 광고주 2997 1개 + raw-ID 인코딩 우연), (ii) **CTR tower 미supervision**(saved runs ctr_weight≈0 → p_ctr가 product로만 간접 학습 → 10/10 decile under-predict). `results/stage_a/rootcause_audit.md`.
- [x] **Probe A+C 결정 실험(fair split):** per-advertiser temporal split(`features_fair/`, 9광고주·100% creative 공유) 구축 + calibration fix(pos_weight/focal BCE, `--ctr-pos-weight`). 공정-split 재학습(escm2wc_dr, ctr_weight=1.0, pos_weight=50, batch 65536, GPU1) 결과:
  - **Ranking: 부정=아티팩트 확정.** neural winners-CTR AUC **0.558→0.658**(2경로 교차검증), **LR 0.554·LGB 0.632 모두 제침**. ⇒ debiasing은 작동; 막힘은 split+학습구성 버그였음.
  - **Calibration: 별개 미해결 이슈.** winners pCTR 여전히 under-predict(IEB 0.597, decile 0.15–0.60); pos_weight=50은 ranking만 회복. p_win 과대(product-coupling)가 잔존 → 다음 수정 대상(higher pos_weight/p_win 보정/post-hoc isotonic).
- [x] **Stage A step 1 — Post-hoc isotonic recalibration(2026-06-15):** winners pCTR을 **cross-fitted isotonic**(K=5, leak-free out-of-fold, **GPU 0**)으로 보정 — neural+LR+LGB 동일 프로토콜. `src/metrics/calibration.py`에 `fit_isotonic`/`cross_fit_isotonic` 추가(기존 `quantile_reliability`/`compute_ieb` 재사용), probe `scripts/stage_a/recalibrate.py` → `results/stage_a/recalibration.{json,summary.md}` + `recalibrated_winners_preds.npz`. 결과(IEB before→after, under-deciles, AUC 불변):
  - escm2wc_dr **0.597→0.000**, under **10→0**, AUC 0.658→0.656 (monotone=rank-preserving 확인)
  - lr_ctr_all 0.435→0.000 (under 3→0), lgb_ctr_all 0.476→0.000 (under 9→0)
  - **Caveat:** 단일 GLOBAL map은 집계 mean bias만 0으로; **per-advertiser 잔차 존재**(neural adv 3476 IEB 0.226, 3358 0.209) → segment-level/training-stage calibration 필요(Stage B2 slice calibration에서 처리). 흥미로운 부수발견: baseline은 bottom-under/top-over(LR 3.06×, LGB 1.41×) 형태로 neural의 균일 under와 다른 miscalibration shape — isotonic이 둘 다 교정.
  - **Plumbing(향후 frozen-map val→test 공짜化):** `train.py`가 `{model}_val_predictions.npz` 저장, `fair_baselines.py`가 `lr_p_val/lgb_p_val/y_win_val/y_click_val` 저장 → 다음 step-4 재학습에서 추가 GPU 0으로 val→test 검증 가능.
- [x] **Stage B2 — Decision-level value(realized surplus + slice calibration, 2026-06-15) [⚠️ first-price 버그, 2026-06-16 second-price로 교정 ↓Stage 7]:** recalibrated-neural vs recalibrated-LR/LGB를 **realized(모델 독립) bid surplus**(won-only, 실제 payprice)로 비교. **핵심 식별성:** cross-fit isotonic이 세 recal 모델 mean V를 **동일(160.7 CPM)**로 고정 → among-recal surplus 차이는 **순수 ranking + 잔차 slice-calibration**(mean-bid 교란 제거). 코드: `src/bidding/simulator.py`에 tested `paired_bootstrap_surplus_gap`/`cluster_bootstrap_surplus_gap`(hybrid exact-clicked/CLT-bulk) 추가, probe `scripts/stage_a/stage_b2_surplus.py` → `stage_b2_surplus.{json,_summary.md}`. **결과 = 가설 SUPPORTED:**
  - headline **dual_regime** neural−lgb realized surplus **+1.79e7**, advertiser-cluster 95% CI **[8.4M, 27.1M] (0 제외, p=1.0)**; neural−lr **+2.42e7** CI [13.1M,36.5M]. neural surplus(2.06e8) > lgb(1.88e8) > lr(1.82e8) — winners-AUC 순서(0.658>0.632>0.554)와 일치.
  - **3-strategy sign-consistent**(exchange_optimal +2.44e7, dual_regime +1.79e7, truthful +0.68e7); **CPC sweep(1e5/2e5/4e5)·max_bid(300/600) sign-stable**. (단 truthful control은 cluster CI가 0 횡단 — no-shading에선 유의성↓, 부호만 유지.)
  - **메커니즘 = 주로 RANKING.** neural이 5개 광고주 전부 surplus 우위. 잔차 slice-IEB가 더 좋은 곳(3427·3476)에선 calibration도 기여하나, 더 나쁜 곳(3358·3386)에서도 ranking으로 우위 → slice-calibration이 아니라 ordering이 주동인. post-recal slice max-bias는 모델 간 혼재(neural adv 1.9e-4 vs lgb 1.6e-4).
  - **Δ_cal(raw→recal):** shading 전략에선 recalibration이 surplus를 대폭↑(under-predict 모델의 과소입찰 해소) → Stage A 보정이 의사결정상 유의미했음 확인. **Limitation:** won-only는 LOST 인벤토리에서의 debiasing 가치를 검정 불가(payprice censored) → 보수적 하한.
- [x] **Stage 3 — eval-protocol freeze + 문서정리(2026-06-15):** 정본 평가 프로토콜을 `docs/evaluation_protocol.md`로 freeze(canonical fair split; winners-only·all-bids AUC 분리; primary = realized surplus + among-recal cluster-CI + slice calibration; global IEB/surplus(V) 금지; won-only scope 한계). 흩어진 redesign narrative를 `docs/redesign_findings.md`로 consolidate, `results/stage_a/README.md` index 추가. Stale 문서(prediction_report*·performance_tuning·bid_optimization_report*)에 superseded/historical banner(본문 전면개정은 Stage 4 이후로 보류). README·research_design 포인터 추가.
- [x] **Stage 4 — 학습단계 calibration: NEGATIVE(2026-06-15, 가설엔 무해).** fair 설정에 train-time 레버만 바꿔 3회 재학습(`scripts/stage_a/stage4_calibration.py` harness로 native/frozen-val→test/cross-fit + per-adv + surplus 평가):
  - A2(relax squeeze, joint 0.1→0.03): native IEB **0.597→0.778**(under 악화), AUC 0.638.
  - B2(activate pos_weight, dr-bce pw20): native IEB **728**, AUC **0.537**(ranking 붕괴) — 폭주.
  - C(dr-bce pw2, sweet-spot probe): native IEB **161**, AUC **0.519** — pos_weight=2도 폭주. **스윗스팟 없음.**
  - **근본:** DR-MSE는 ranking 좋지만 pos_weight 무시; pos_weight 켜려면 DR-BCE 필요한데 DR-BCE+pos_weight는 어떤 값에서도 over-predict+ranking 파괴. **어떤 native 모델도 calibrate 안 됨.**
  - **post-hoc isotonic 우월 확정(경제적 증명):** 모든 모델×전략에서 `native−(자기 isotonic)` realized-surplus CI가 0을 음수쪽으로 제외 → native가 자기 isotonic보다 *더 나쁜 입찰자*. 재학습 불필요, cross-fit isotonic(rank-preserving, GPU 0)이 정답.
  - **frozen val→test(4b):** 최소손상 A2에서 val-map→test global IEB **0.057**(cross-fit ~0 대비 소폭 temporal penalty) — 대체로 generalize. 단 fair baseline은 val예측 미저장이라 frozen 불가(caveat).
  - **남은 단일 레버 = per-advertiser 잔차**: global map(post-hoc·frozen·train-time 전부)이 hard-adv(3358/3476) 잔차 ~0.23–0.29를 못 내림 → **segment-aware(광고주별) calibration**만이 미해결 미래과제. 산출물 `results/stage_a/stage4_calibration.{json,_summary.md}`.
- [x] **Stage 5 — Segment-aware(per-advertiser) calibration: 잔차 SOLVED(2026-06-15).** 광고주별 cross-fit isotonic(`segment_cross_fit_isotonic` + global fallback)로 global map이 못 고치던 per-advertiser 잔차를 제거: neural max **0.226→0.0006**, lr 0.284→0.0003, lgb 0.272→0.0002. **within-advertiser AUC 보존**(monotone per-seg), **global AUC는 오히려 상승**(neural 0.656→0.666 — 광고주간 level 정렬로 cross-adv 비교 정확↑). **입찰 영향 = 소폭 양(+), 광고주 집중:** dual_regime segment−global surplus neural +7.4M·lr +6.1M·**lgb +12.9M(cluster CI 0 제외, 유의)**; neural/lr은 양수지만 5-광고주 cluster bootstrap(저power)에서 0 미제외. neural 이득은 global 최악 광고주(3386 +6.0M, 3358 +1.2M)에 집중. 코드 `scripts/stage_a/segment_calibration.py`, 산출물 `results/stage_a/segment_calibration.{json,_summary.md}` + `segment_recalibrated_winners_preds.npz`. **⇒ calibration 스토리 완결**(global level=post-hoc isotonic, train-time=NEGATIVE, segment=잔차 해소).
- [x] **Stage 6 — Full-inventory policy value(won-only 탈출, 2026-06-16):** 전체 19.4M bid에서 정책 가치를 **second-price 구조적 projection**으로 추정(`src/bidding/policy_value.py` `project_policy_value`/`MarketModel`, probe `scripts/stage_a/policy_value.py`). **핵심:** OPE 아님(deterministic 8-level flat 로깅 → propensity 퇴화); 대신 second-price에서 정책 bid가 logged bid 이하인 곳은 surplus가 **정확히 관측 가능**. truthful bid(=V≈160) < logged flat(227–300)라 정책은 관측된 win의 부분집합을 재획득 → **won-only가 거의 안 binding**. 결과: truthful V(π) neural **4.39e8** > lgb 4.29e8 > lr 4.17e8 (각 **≥99.3% EXACT**, modeled ≤0.7%); neural−lgb +9.7e6, cluster CI [-16.4M,46.8M] p=0.66 — Stage B2와 일관. **de-risk ladder 작동:** P2 통과(logged 정책이 realized 2p surplus 정확 재현), **P1 NO-GO**(F(b|x)가 flat-bid 교란으로 logged cell의 13%만 calibrate → lost-inventory 외삽 신뢰불가, 단 ≤0.7%만 modeled라 headline 무관). **부수 발견:** ① `stage_b2_surplus.py:93`이 second-price 데이터에 **first-price 시뮬(버그)** → 본 projection이 교정본(결론 불변); ② won row의 **1.24%가 payprice>bidprice**(second-price 위배, iPinYou data quirk) 정확 제외. 산출물 `results/stage_a/policy_value.{json,_summary.md}`.
- [x] **Stage 7 — Stage-B2 second-price 재계산(2026-06-16, 정직성 교정):** `stage_b2_surplus.py`를 **second-price**로 교정(winner는 시장청산가=payprice 지불, bid 아님). **전략 순위 반전 확정:** 2p-optimal=**truthful**(1p-best는 exchange_optimal); truthful은 1p surplus 음수(-7.3e6)였으나 2p에선 최고(+4.40e8). 셀별 2p≥1p 단언 통과; P2 sanity(logged 정책 2p == realized 2p)통과. **결과 = thesis 부분 지지(원 "decisively SUPPORTED"는 first-price 버그+dual_regime 선택으로 과장됐었음):** neural이 LGB를 **point에선 전 전략 우위**(sign-consistent, CPC-stable) + **LR은 유의하게 제침**(truthful CI [17.7M,37.8M]); 단 **강baseline LGB 대비 2p-optimal(truthful)에서 유의성 없음**(CI [-11.1M,40.7M], 5광고주 저power), exchange_optimal에서만 유의([6.2M,30.0M]). 1p 수치는 `mechanism_comparison`에 보존. 산출물 갱신 `results/stage_a/stage_b2_surplus.{json,_summary.md}`(canonical=second-price).
- [x] **Stage 8 — neural-vs-LGB heterogeneity/power 분석(2026-06-16, 결판):** Stage-7의 미해결 질문(neural>LGB가 underpowered인지 null인지)을 claim ladder(R1~R6)로 결판. 신규 tested `src/metrics/cluster_inference.py`(Cochran's Q/I²/τ², cluster-t MDE, sign test, LOAO) + probe `scripts/stage_a/power_analysis.py`. **결과 = neural>LGB는 NOT robust(heterogeneous, underpowered 아님):** truthful 2p에서 광고주 **2/5만 양(+), CI-유의 1/5(adv 3427만 +13.9M)**; **Cochran's Q가 homogeneity 기각(I²=0.82, p=0.0002)** → 진짜 heterogeneity; **LOAO: 3427 빼면 평균 음수(-1.1M)** → 단일광고주 leverage; MDE(~11.5M) ≫ per-adv mean(1.9M); ICC=0.09>0 → finer clustering 금지(근거 제시); 메커니즘 불일치(adv 1458은 neural이 더 calibrated인데 surplus 패배). **neural>LR은 robust(5/5, CI [17.7M,37.8M])**. **정직한 결론: debiasing의 입찰 가치는 선형 baseline 대비는 robust, 강 GBM 대비는 not-robust(단일광고주 집중).** 9-advertiser는 dead end(disjoint artifact) 확정. 산출물 `results/stage_a/power_analysis.{json,_summary.md}`.
- [x] **Portfolio v2 — dunnhumby급 전면 고도화 + 실험 강화(2026-06-16, push 9c816d0):** breadth 실험을 **fair split로 강화**. ① **ablation ladder 완성**(ESMM-WC + ESCM²-WC IPW를 fair split GPU 재학습): winners-AUC LR .554→LGB .632→ESMM-WC **.674**→IPW .656→DR .658(군집형; 전 rung recal IEB≈0). ② **bidding/pacing canonical**(`bidding_fair.json` truthful 5.13e8 best + α-sweep; `pacing_fair.json` WR-weighted +11~14%) — 원본 unfair-split CSV 대체. ③ **CATE/SCM exploratory**(`scm_fair.json` bid→surplus −0.066 refutation robust; `cate_fair.json` naive contrast τ_surplus +21/τ_win −0.33 confounded). hero figure **12개**(`scripts/portfolio/make_figures.py`), 아키텍처 SVG dim 교정(embed16/no-FM/trunk480=학습모델), funnel/arc shadow-polish, 영/한 dunnhumby급 README(badges·time-TOC·results-at-a-glance·실험별 섹션+표+figure·Limitations ledger·Appendix·인용) + technical_report·NUMBERS_LEDGER(Part 2) 확장. 정직 verdict 불변(선형 LR robust, 강 GBM not robust). env note: conda 없음→jax 0.10.1 stack pip설치, GPU1.
- [x] **Portfolio & 문서화(2026-06-16):** 정직성-우선 공개 포트폴리오 구축. **영어 주 `README.md` + 한글 twin `README.ko.md`**(수치 drift 0, 4-에이전트 적대검증 확인) + `docs/technical_report.md`(30분 레이어) + `docs/GLOSSARY.md`(이중언어) + `docs/NUMBERS_LEDGER.md`(committed JSON 단일 진실표). **Ledger 적대대조로 교정 3건:** ① §5 `exchange_optimal` neural−LR **+5.4e7→+3.63e7**(committed grid 기준), ② `redesign_findings.md` §5/§7 `SUPPORTED` 헤더 → **`SPLIT`**(본문·ledger와 자기일관화), ③ full-inventory exact 비율 **≥99.3%→≥99.26%**(LR 99.26%가 하한). **신규 hero figure 6종**(`scripts/portfolio/make_figures.py` → `results/figures/portfolio/`: surplus-forest·calibration-journey·ablation-auc·surplus-grid·policy-value-decomp·artifact-vs-fair, **committed JSON에서만 생성 — 모델 재학습 없음**) + **개념 SVG 3종 EN/KO**(`scripts/portfolio/make_diagrams.py` → `assets/`, cairosvg 렌더검증). **Publish-prep:** `LICENSE`(MIT + iPinYou 데이터 별도고지), `.gitignore` 보강(`*.joblib`), 폐기 docs 5종 → `docs/archive/`(redirect README + 링크 교정). 공개 헤드라인 = **방법론적 엄밀성 + 정직한 반증**(선형 LR 대비 robust, 강 GBM 대비 not-robust).
- [ ] **(남은 트랙) 미래 연구:** temporal/online calibration · risk-aware bidding · serving 데모. (통계력/lost-inventory는 데이터 구조상 천장 확정 — 더 못 함; 포트폴리오 정리는 ✅ 완료 ↑.)

> 아래는 재설계 이전(2026-04-01까지)의 진행 이력 — 보존. 재설계로 헤드라인 주장("debiasing이 baseline AUC를 이긴다")은 폐기되고 calibration→surplus로 재배치됨.

---

## Project Overview

iPinYou RTB 데이터를 활용한 **Selection Bias Debiasing + First-price Bid Optimization** 연구 프로젝트.

**핵심 혁신:**
1. ESMM-WC / ESCM²-WC(DR) for Bid→Win→Click debiasing: unbiased pCTR (win selection bias 해결)
2. Win Tower dual purpose: (a) CTR debiasing propensity, (b) bid shading win rate model
3. Ablation: Biased baseline → ESMM-WC → ESCM²-WC(IPW) → ESCM²-WC(DR) 순차적 개선
4. Production Serving (<100ms latency)

**핵심 인사이트:**
1. **CVR Tower Pivot (2026-02-17)**: EDA 2.2.1에서 CVR 예측이 near-trivial 확인 (Train(S2) 기준 Branding 3개 CVR=0, Retargeting 1개(3358) CVR 27% retargeting artifact). Bid→Win→Click 퍼널로 reframe
2. **Bid→Win→Click reframe 근거**: 129.5M 전체 bids 활용 (vs 30.6M impressions), Click 23K (vs Conversion 1,860), Win PS AUC 0.91 진단 완료, CTR = 핵심 value signal
3. **Win bias**: Clean PS AUC ~0.91, +4.57% CTR overestimation (LGB), positivity violation (overlap ~48%) → IPW 단독 위험 → DR + ESMM constraint로 완화
4. **Advertiser Taxonomy**: Retargeting(2821,3358,2259) vs Branding(1458,3386,3427,2261,2997) vs Mixed(3476) — advertiser stratification 필수
5. **Usertag Label Leakage**: `has_tags`/`n_tags` 소스 제거 완료 (2026-02-12)

---

## Progress Summary

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: MVP | ✅ Complete | 100% |
| Phase 2: Debiasing | 🔄 In Progress | 95% |
| Phase 3: Causal & Serving | 🔄 In Progress | 20% |

---

## Phase 1: MVP (SP0 + SP1 Basic) ✅ Complete

### Task 1.1: 데이터 파이프라인 (SP0) ✅ Complete
- [x] `src/data/parser.py`: bid/imp/clk/conv 파싱 (Tab-separated, bz2)
- [x] `src/data/unifier.py`: bidid 기준 조인, win/click/conversion 라벨링
- [x] `src/config.py`: NamedTuple 기반 설정 (DataConfig, FeatureConfig 등)
- [x] Parquet 저장 (season/day 파티션) - `save_to_parquet()`, `load_from_parquet()`
- [x] `scripts/preprocess.py`: CLI for data preprocessing
- [x] `notebooks/00_data_preparation.ipynb`: Data pipeline demo

### Task 1.2: EDA & Selection Bias 진단 (SP0) ✅ Complete (EDA 전문가 리뷰 반영 2026-02-10)
- [x] `notebooks/01_eda_analysis.ipynb`: CTR/CVR 분포, 캠페인별 통계, **Floor Price/Ad Format/Geographic/Publisher 분석 추가**
  - **Section 12 추가 (전문가 리뷰 반영)**: Market Price CDF, Temporal Stability, Floor Binding, IVT Screening, Adv×Exchange, Competition Intensity, Day-of-Week
  - **Section 2.2 추가 (HIGH 우선순위 보완)**: Conversion Attribution Analysis — View-through vs Click-through 분리, Raw timestamp 기반 attribution window 분석, ESCM² CVR tower 설계 시사점
- [x] `notebooks/02_selection_bias_diagnosis.ipynb`: **전면 재작성 (2026-02-13)** — Two-stage selection bias (Win + Click) 통합 진단
  - **Part 0**: Setup + Advertiser Taxonomy (retargeting/branding/mixed 분류, EDA 01 기반)
  - **Part 1**: Win Selection Bias — covariate shift, LR/LGB win PS, CTR distortion (+2.58%), advertiser-stratified bias, positivity diagnostics
  - **Part 2**: Click Selection Bias — click covariate shift, click PS (is_unbalanced=True), CVR distortion (retargeting only), advertiser-stratified click bias
  - **Part 3**: Two-Stage Integration — Win×Click PS correlation, comprehensive summary table, advertiser-specific debiasing recommendations
  - `src/debiasing/diagnostics.py`: 기존 제네릭 함수 재사용 (win/click 모두 동일 API)
  - `src/debiasing/win_propensity.py`: click PS에도 재사용 (is_unbalanced LGB params)
  - **Strategic Conclusions 5가지**: (1) Win bias = auction structure, (2) Click bias = ESCM² 필수, (3) Advertiser stratification, (4) Branding→CTR only / Retargeting→CVR critical, (5) Two-stage IPW→DR 선호
- [x] `docs/scripts_tutorial.md`: Script usage guide

### Task 1.3: Baseline 모델 (SP1 Basic) ✅ Complete
- [x] `src/features/engineering.py`: Feature engineering functions
- [x] `src/features/usertag.py`: Usertag multi-hot encoding
- [x] `scripts/build_features.py`: Feature engineering CLI
- [x] `scripts/train.py`: Model training CLI (baseline + ESCM2), **`--use-usertag` 플래그 추가**
- [x] `notebooks/03_prediction.ipynb`: Unified prediction comparison (Baseline LGB/LR + ESMM-WC + ESCM2-WC, AUC/ECE/IEB). Section 5 고도화 + Section 8+ 구조 정비: 기존 Section 8-12를 Appendix A-C로 재구성, Section 12(중복) 삭제, Section 5 cross-reference 연결
- [x] `src/metrics/evaluation.py`: Core metric functions (compute_ece, compute_ieb, compute_metrics, EvalMetrics)
- [x] `src/metrics/result_loader.py`: JSON result loading + UnifiedMetrics normalization
- [x] `src/metrics/comparison.py`: Comparison table generation + highlight_best
- [x] `src/metrics/diagnostics_plot.py`: 3-panel prediction diagnostics (Calibration + ROC + Score Distribution)
- [x] `scripts/train.py`: Per-sample test predictions `.npz` 저장 (재학습 시 자동 생성)

### Task 1.4: Feature Ablation (Partially Complete)
- [x] `domain` feature 추가: 108K domains → hash encoding (10K buckets) + frequency encoding (freq + freq_log)
- [x] `creative` feature 추가: hash encoding (5K buckets) + frequency encoding (freq + freq_log)
- [x] Target encoding 적용: `target_encode_kfold` 파이프라인 통합 (`--target-encoding` flag + `target-encode` 서브커맨드), Phase 20 실험 진행 중
- [ ] EDA에서 S2→S3 creative overlap 비율 확인 (temporal drift 주요 원인 가능성)

---

## Phase 2: Debiasing Ablation (SP1 Full + SP2 + SP3)

### Task 2.1: ESMM-WC + ESCM²-WC 구현 ✅ Complete
- [x] `src/models/base.py`: Shared layers (MLP, EmbeddingLayer, FeatureInteraction) + loss utilities (binary_cross_entropy, counterfactual_risk, etc.)
- [x] `src/models/esmm_wc.py`: ESMM-WC 2-tower (Win + CTR), ESMM constraint only
- [x] `src/models/escm2_wc.py`: ESCM²-WC 3-tower (Win + CTR + Imputation), DR/IPW debiasing
- [x] `scripts/train.py`: `esmmwc`, `escm2wc` CLI commands 추가
- [x] `src/models/escm2_rtb.py`: 삭제 (CVR tower 포함 4-tower → Bid→Win→Click reframe으로 불필요)

### Task 2.2: Win Propensity (분리 모델) ✅ Complete
- [x] `src/debiasing/win_propensity.py`: LightGBM + Isotonic Calibration (external PS option)

### Task 2.2.5: Distributed Training Infrastructure ✅ Complete
- [x] `src/distributed/mesh.py`: JAX SPMD Mesh 생성, device sharding utilities
- [x] `src/distributed/data_loader.py`: grain 기반 DataLoader (RTBDataSource, materialize_to_source, batch_to_jax)
- [x] `src/distributed/train_state.py`: LR schedule (warmup + cosine/linear decay), optimizer factory (AdamW + gradient clipping)
- [x] `src/distributed/checkpoint.py`: orbax 기반 checkpoint save/restore with metadata
- [x] `src/distributed/__init__.py`: Public API exports
- [x] `scripts/train.py`: `--distributed`, `--scheduler`, `--warmup-steps`, `--gradient-clip`, `--num-devices`, `--resume-from` CLI args
- [x] `configs/distributed/*.yaml`: DistributedConfig YAML configs
- [x] `pyproject.toml`: grain, orbax-checkpoint dependencies

### Task 2.2.6: W&B Sweep Integration ✅ Complete
- [x] `scripts/train.py`: Debiasing hyperparameters CLI 노출 (dropout, cfr_lambda, win_eps, max_weight, loss weights)
- [x] `scripts/train.py`: `wandb_run` pass-through — sweep agent가 pre-initialized run 전달 시 double init 방지
- [x] `scripts/sweep.py`: **NEW** — Typer CLI: `create` (YAML→wandb.sweep) + `agent` (wandb.agent→_train_wc_model)
- [x] `experiments/sweep_escm2wc.yaml`: **NEW** — Bayes sweep config (15 params: architecture + training + debiasing)
- [x] `experiments/sweep_esmmwc.yaml`: **NEW** — Bayes sweep config (7 params: architecture + training only)

### Task 2.2.7: ESCM²-WC(DR) Performance Improvement ✅ Complete
- [x] `src/distributed/data_loader.py`: RTBDataSource, NumpyBatchIterator, materialize_to_source, batch_to_jax에 ext_propensity 전파
- [x] `src/models/escm2_wc.py`: ESCM2WCConfig에 use_external_propensity + loss_fn propensity 분기
- [x] `scripts/train.py`: `--use-external-propensity`, `--external-ps-model-dir` CLI flags + external PS 로드/주입
- [x] `configs/model/escm2wc_dr.yaml`: use_external_propensity 필드 추가
- [x] `docs/performance_tuning.md`: Phase 17/18 실험 결과 기록
- [x] Phase 17-1: cfr_lambda={0.3, 0.5} — AQ(0.6841), AR(0.6774), AL(0.2) 최적 확정
- [x] Phase 17-2: impute_loss_weight={0.3, 1.0} — AS(0.6866), AT(0.6759), AL(0.5) 최적. IEB 급증
- [x] Phase 17-3: SKIPPED (17-1/2 모두 AL 대비 IEB 악화)
- [x] Phase 18: External Win PS — **Run AW WCTR AUC 0.6882 (ESCM2-WC best)**, AV(0.6712)

### Task 2.2.8: Target Encoding Feature 추가 (Phase 20) ❌ 가설 기각
- [x] `src/features/engineering.py`: `target_encode_kfold()` sklearn KFold → pure numpy KFold 대체 (numpy 호환성)
- [x] `scripts/build_features.py`: `--target-encoding/--no-target-encoding` CLI 플래그 + `target-encode` 서브커맨드 추가
- [x] Feature rebuild: 5 cats × 2 targets = 10 TE features (numerical), 30→40 features 확인
- [x] Run AY: ESMM-WC + TE → **WCTR AUC 0.5480 (J 0.6905 대비 -0.1424)** — 가설 기각
- [x] Run AX: SKIPPED (AY 결과 기반, TE가 neural model에서 역효과)
- [x] `docs/performance_tuning.md`: Phase 20 결과 기록 완료

### Task 2.3: Ablation Study
- [x] `notebooks/04_prediction_debiasing.ipynb` — DR 메커니즘 이론 + CFR/ExtPS/IPW-DR ablation + Negative results
- [ ] `experiments/configs/prediction_debiasing.yaml`

**Ablation Structure (Bid→Win→Click reframe):**
| Model | Win Debiasing | Description | File |
|-------|---------------|-------------|------|
| Biased Baseline (LGB) | None | CTR on winners only | `scripts/train.py` baseline |
| **ESMM-WC** | Implicit (ESMM) | 2-tower, ESMM constraint only | `esmm_wc.py` |
| ESCM²-WC (IPW) | IPW | 3-tower, IPW debiasing | `escm2_wc.py` (loss_type=ipw) |
| **ESCM²-WC (DR)** | **DR (primary)** | 3-tower, DR debiasing | `escm2_wc.py` (loss_type=dr) |
| +External Win PS | DR (external PS) | DR with LGB propensity | `escm2_wc.py` + `win_propensity.py` |

- Key metrics: Win AUC, CTR AUC (biased + IPW-unbiased), Joint AUC, ECE, IEB
- Expected: Biased → ESMM-WC → ESCM²-WC(DR) 순차적 개선

### Task 2.3.5: Win Rate 분석 (SP2) — SP3 입력 ✅ Complete
- [x] `src/win_rate/nonparametric.py`: Wilson CI, empirical win rate curve, market price stats, serving lookup
- [x] `src/win_rate/survival.py`: KM CDF, parametric fit (Weibull/LogNormal/Exponential), segment CDF
- [x] `notebooks/05_win_rate_market_price.ipynb`: NB05+NB06 통합 — 11 sections, 9 figures
- [x] `pyproject.toml`: lifelines>=0.29.0 의존성 추가
- [x] `results/market_price_cdf/`: KM CDF export (overall + exchange-conditional .npz + summary.json)
- [x] Section 11 (NEW): Calibration 경제적 가치 분석 — IEB → V(x) → optimal bid → surplus 비교, exchange-conditional surplus loss
- [x] Section 11 확장: LR CTR_all (IEB 0.122) + LGB CTR (IEB 0.362) baseline 추가 → 5-model surplus 비교, Figure 14 재생성

**SP2 → SP3 연결:**
- Market price CDF 추정 → bid shading 입력
- Win Tower (SP1) + market price CDF (SP2) → bid shading shade(x) (SP3)
- **Calibration 경제적 가치**: ESCM²-WC(DR) AL (IEB 0.014) → oracle-matching surplus, LR CTR_all (IEB 0.122) → 2.2% 손실 (8.7× overbid), LGB CTR (IEB 0.362) → 2.2% 손실 (25.9× overbid), ESMM-WC J (IEB 1.335) → 14.7% 손실 (95× overbid). AUC best (LR 0.7687) ≠ bidding best

### Task 2.4: Bid Optimization (SP3) ✅ Complete
- [x] `src/bidding/value.py`: ValueConfig, ValueResult, compute_impression_values (CPC/CPA/CPM)
- [x] `src/bidding/shading.py`: Distribution-based optimal shading (argmax (V-b)×F(b)), linear, percentile, dual-regime (floor-aware), exchange-conditional
- [x] `src/bidding/pacing.py`: PID controller, throttling, WR-weighted hourly budget allocation
- [x] `src/bidding/simulator.py`: Offline auction simulation engine, 8-strategy comparison, won-only simulation framework
- [x] `src/config.py`: BiddingConfig 확장 (goal_type, cpc_target=200K, shading_strategy, PID params, auction_type)
- [x] `configs/bidding/default.yaml`: YAML config 업데이트
- [x] `notebooks/07_bid_optimization.ipynb`: 12 sections, 11 figures, 7 CSV. Core result: iPinYou flat-bid surplus -805M → dual_regime surplus +128M
- [x] `notebooks/08_budget_pacing.ipynb`: 8 sections, 7 figures. PID pacing + WR-weighted allocation + budget sensitivity

**SP1 → SP3 연결 (Bid→Win→Click reframe 이후):**
- V(x) = debiased_pCTR(x) × CPC_target (CVR near-trivial → CPC 기반 단순화)
- Win Tower dual purpose: (a) CTR debiasing propensity, (b) market price CDF → bid shading
- 전체 공식: bid(x) = V(x) × shade(x) × pace(t)

**Core Simulation Results (Won-only, First-price, ESCM²-WC(DR)):**
| Strategy | Win Rate | Clicks | Total Surplus | Overpayment | ROI |
|----------|----------|--------|---------------|-------------|-----|
| iPinYou flat | 100% | 4,482 | -805M | 10.14× | 0.74 |
| Truthful (bid=V) | 51.8% | 2,266 | +17.7M | 5.79× | 1.56 |
| Linear α=0.6 | 37.1% | 1,702 | +100M | 4.58× | 2.38 |
| Optimal KM | 33.2% | 1,480 | +119M | 3.44× | 3.36 |
| **Dual-regime** | **37.9%** | **1,608** | **+128M** | **2.77×** | **3.26** |

**Key Insight**: Dual-regime (floor-aware) shading achieves highest surplus by bidding just above floor for floor-bound impressions (50.7%) and optimal shading for competitive impressions.

---

## Phase 3: Causal & Serving (SP4 + SP5)

### Task 3.1: Multi-Outcome CATE Analysis (SP4 Part A) 🔄 In Progress
- [x] `src/causal/cate.py`: Multi-outcome CATE (win, payment, click, surplus) + mediation + segment
- [x] `src/causal/__init__.py`: Public API exports
- [x] `docs/research_design/04-causal-analysis.md`: 전면 재작성 (Surplus 중심, V(x) 불가 명시, 분해+mediation)
- [ ] `notebooks/09a_cate_analysis.ipynb`: ~10 sections (CATE 분포, 분해 검증, mediation, segment heatmap)

**재설계 (2026-04):**
- Treatment: T = log(bid_price), Outcome: Surplus (primary) + Win/Payment/Click 분해
- V(x) CATE outcome 불가 (bid 무관, ∂V/∂bid = 0, 순환 논리)
- Surplus point mass at 0 해결: Multi-outcome 분해 (τ_surplus ≈ V·τ_win - τ_pay)
- Mediation: NIE (volume: V(x)×τ_win) vs NDE (cost: Total - NIE)
- Advertiser-stratified CATE (Simpson's Paradox 방지)

### Task 3.2: SCM & DAG (SP4 Part B) 🔄 In Progress
- [x] `src/causal/scm.py`: DAG + DoWhy estimation + refutation + model-based counterfactual
- [ ] `notebooks/09b_scm_dag.ipynb`: ~7 sections (DAG 시각화, estimation, refutation, counterfactual scenarios)

**재설계 (2026-04):**
- DAG에 surplus, payment 노드 추가 (2-channel: volume + cost)
- Model-based counterfactual: `run_auction_simulation()` 재활용 (structural equation 대신)
- Refutation: random_common_cause, placebo_treatment, data_subset

### Task 3.3: Policy Simulation (SP4 Part C) — SP3에서 대부분 구현 완료
- [x] `src/bidding/simulator.py`: 8-strategy offline auction simulation (SP3 완료)
- [x] `src/bidding/shading.py`: Optimal/dual-regime/exchange-conditional shading (SP3 완료)
- [x] `src/bidding/pacing.py`: PID controller + WR-weighted allocation (SP3 완료)
- [ ] CATE-informed bidding strategy 연동 (τ_surplus 기반 bid_adj)
- [ ] `notebooks/10_policy_simulation.ipynb`

### Task 3.4: Serving (SP5)
- [ ] `configs/feast/`: Feast feature definitions
- [ ] `src/serving/feast_store.py`
- [ ] `mlops/serving/export.py`: JAX → ONNX 변환
- [ ] `mlops/serving/app.py`: FastAPI 서비스
- [ ] `mlops/monitoring/rtb_monitor.py`
- [ ] `notebooks/11_serving_integration.ipynb`
- [ ] `notebooks/12_evaluation.ipynb`

---

## Implemented Components

### Core Modules
| Module | Path | Description |
|--------|------|-------------|
| Data Parser | `src/data/parser.py` | bid/imp/clk/conv bz2 파싱 + parallel |
| Data Unifier | `src/data/unifier.py` | bidid 기준 조인, 라벨링, Parquet I/O |
| Config | `src/config.py` | NamedTuple 기반 설정 |
| Config Utils | `src/config_utils.py` | Hydra Compose API bridge (YAML → NamedTuple) |
| **Ray Utils** | `src/ray_utils.py` | Ray 기반 병렬 처리 유틸리티 |
| Feature Engineering | `src/features/engineering.py` | Time, slot, region, competition features + parallel |
| Usertag Encoding | `src/features/usertag.py` | Multi-hot, hashing, vocabulary + parallel |
| Base Layers | `src/models/base.py` | MLP, EmbeddingLayer, FM, loss utilities (BCE, CFR, ESS) |
| ESMM-WC | `src/models/esmm_wc.py` | Bid→Win→Click 2-tower (ESMM constraint) |
| ESCM²-WC | `src/models/escm2_wc.py` | Bid→Win→Click 3-tower (DR/IPW debiasing) |
| Win Propensity | `src/debiasing/win_propensity.py` | LightGBM + Calibration (external PS) |
| **CATE** | `src/causal/cate.py` | Multi-outcome CATE (CausalForestDML) + mediation + segment |
| **SCM** | `src/causal/scm.py` | DAG + DoWhy estimation + refutation + model-based counterfactual |
| **Bid Value** | `src/bidding/value.py` | V(x) = pCTR × CPC_target (CPC/CPA/CPM) |
| **Bid Shading** | `src/bidding/shading.py` | Optimal, linear, percentile, dual-regime, exchange-conditional |
| **Budget Pacing** | `src/bidding/pacing.py` | PID controller, throttling, WR-weighted allocation |
| **Auction Simulator** | `src/bidding/simulator.py` | Offline auction simulation, 8-strategy comparison framework |

### CLI Scripts
| Script | Path | Description |
|--------|------|-------------|
| Preprocess | `scripts/preprocess.py` | Raw → Unified Parquet (`--workers`, `--config-dir`) |
| Build Features | `scripts/build_features.py` | Feature engineering + Split (`--workers`, `--config-dir`) |
| Train | `scripts/train.py` | Baseline + ESMM-WC + ESCM²-WC training (`--config-dir`) |
| Sweep | `scripts/sweep.py` | W&B Sweep: `create` + `agent` commands (Bayes HP optimization) |

### Notebooks
| Notebook | Status | Description |
|----------|--------|-------------|
| 00_data_preparation | ✅ | Data pipeline demo |
| 01_eda_analysis | ✅ | EDA + Floor Price/Ad Format/Geographic/Publisher 분석 + Conversion Attribution. Slow usertag cells (Ray parsing, chunked agg) 제거 — summary markdown만 유지 |
| 02_selection_bias_diagnosis | ✅ | Bias quantification |
| 03_prediction_baseline | ✅ | LightGBM baseline + 3-panel diagnostics (LGB CTR, LR CTR_all) |
| 04_prediction_debiasing | ✅ | DR 메커니즘 이론 + Component ablation (CFR, ExtPS) + Negative results + AUC-Calibration trade-off + Section 11 Neural diagnostics |
| 05_win_rate_market_price | ✅ | NB05+NB06 통합 — 11 sections, 9 figures, KM CDF export, Calibration 경제적 가치 분석 |
| 07_bid_optimization | ✅ | **SP3 Core** — 12 sections, 11 figures: Value analysis, bid shading theory, 8-strategy simulation, debiasing impact, sensitivity analysis |
| 08_budget_pacing | ✅ | PID pacing + WR-weighted allocation + budget sensitivity + full pipeline comparison |

### Documentation
| Doc | Path | Description |
|-----|------|-------------|
| Scripts Tutorial | `docs/scripts_tutorial.md` | CLI usage guide + Ray parallelization |
| Feature Dictionary | `docs/feature_dictionary.md` | Feature별 what/why/how 상세 레퍼런스 |
| RTB Ecosystem | `docs/rtb_ecosystem.md` | RTB 생태계 구조 & 경매 메커니즘 (SP3 도메인 배경) |
| Research Design | `docs/research_design/` | SP0-SP5 research docs |
| **Prediction Report** | `docs/prediction_report.md` | **포트폴리오 수준 연구 보고서 (778줄, 18 figures, 6 sections + 부록)** |
| **Bid Optimization Report** | `docs/bid_optimization_report.md` | **SP3 방법론 + 결과 보고서 (619줄, 15 figures, 7 sections + 부록)** |
| **Bid Optimization Summary** | `docs/bid_optimization_report_summary.md` | **SP3 요약 보고서 (412줄, 10 figures) — prediction_report_summary.md 대응** |

### iPinYou Data Schema
- **Bid Log**: 21 columns (bidid, timestamp, ipinyouid, useragent, ip, region, city, adexchange, domain, url, urlid, slotid, slotwidth, slotheight, slotvisibility, slotformat, slotprice, creative, bidprice, advertiser, usertag)
- **Imp/Clk/Conv Log**: 24 columns (bidid, timestamp, logtype, + bid columns + payprice, keypageurl)
- **Seasons**: 1st (2013.03), 2nd (2013.06), 3rd (2013.10)

---

## Key Findings

### EDA Analysis (from notebook 01)

**Dataset Scale (S2+S3):**
- **Total**: 129,493,498 bids (S2: 106,578,660 + S3: 22,914,838), 9 advertisers
- **Funnel**: 129.5M bids → 30.6M impressions (WR 23.67%) → 23,058 clicks (CTR 0.0752%) → 1,860 conversions (CVR 8.07%)
- **Advertiser Taxonomy**: Branding 5 (1458, 3386, 3427, 2261, 2997), Retargeting 3 (2821, 3358, 2259), Mixed 1 (3476)

**Win Rate Patterns:**
- **Overall**: 23.67%, varies dramatically by hour (8.59%~43.06%, U-shape) and exchange (13.24%~55.55%)
- **Hourly U-shape**: 새벽 high WR (경쟁↓) → 오후 low WR (경쟁↑) → trade-off for budget pacing
- **Exchange heterogeneity**: Ex1 no floor, Ex2 moderate, Ex3 active floor → exchange-conditional modeling 필요

**Market Price & Competition:**
- **Market price**: Median 70, Mean 80, P90: 177, P95: 219 CPM
- **Floor price**: Median 40, Mean 45.47 CPM, **32.24% floor binding** (payprice ≈ floor)
- **Overpayment**: iPinYou flat-bid → ~76% overpayment vs market price
- **Bid-Pay Spread**: Mean ~192 CPM, significant overbidding

**Data Quality:**
- **IVT**: 76 zero-win domains (7.16M bids, 5.5%), 741 zero-click domains
- **Pop-up**: slotformat==5 CTR 0.86% (11.4x standard) → misclick artifact
- **Visibility 255**: Exchange 1 sentinel value → `is_visibility_unknown` indicator 필요
- **Temporal Drift**: S2→S3 KS=0.1294, CDF rightward shift → drift monitoring baseline

**Geographic & Publisher:**
- **Geographic Concentration**: 36 regions, top 17 (47.2%) account for 80% of bids
- **Publisher Concentration**: 108K domains, top 238 (0.2%) account for 80% of bids; top 50 ≈ 60% traffic
- **Ad Format**: Win rate and CTR differ across formats; format x visibility interaction exists; pop-up 11.4x CTR

**Attribution & Conversion:**
- **CVR near-trivial**: WCTR 0.015%, Branding 3개(train) CVR=0, Retargeting 1개(train: 3358) CVR 27%
- **Click→Conv**: ~1초 (retargeting artifact), Imp→Click: ~12초, 100% click-through attribution
- **Usertag**: Top-100 tags cover most traffic; CTR Tower에서 안전 사용, Win Tower 금지 (leakage)

### Conversion Attribution Analysis (from notebook 01, Section 2.2, 2026-02-11)
- **View-through vs Click-through**: Unified data 기반 `conversion=1 & click=0` 비율 확인
- **Attribution Window**: Raw conv/clk timestamp 기반 imp→click, click→conv, imp→conv 시간 분포 분석
- **ESCM² CVR Tower 시사점**: View-through 존재 시 click-through만 사용, advertiser별 전환 정의 차이 확인
- **Figure**: `results/figures/01_eda_conversion_attribution.png`

### Bid Optimization Analysis (from notebooks 07-08, 2026-03-24)

**Won-only First-Price Simulation (4.23M impressions):**
- **iPinYou flat-bid baseline**: 100% WR, surplus -805M (massive overpayment 10.14×), ROI 0.74
- **Optimal KM shading**: 33.2% WR, surplus +119M, overpayment 3.44×, ROI 3.36
- **Dual-regime (best surplus)**: 37.9% WR, surplus +128M, overpayment 2.77× (lowest), ROI 3.26
- **Exchange-conditional**: Exchange 1 (median 153) allows aggressive shading; Exchange 2/3 require conservative bids
- **Debiasing impact**: ESCM²-WC(DR) IEB 0.014 → near-oracle V(x), IEB 0.5 → surplus halved
- **Budget pacing**: PID controller with WR-weighted hourly allocation improves efficiency
- **Key insight**: Floor-aware dual-regime shading outperforms pure optimal shading (50.7% floor-bound impressions bid just above floor)

### Two-Stage Selection Bias Analysis (from notebook 02, rewritten 2026-02-13)

**Win Stage (Bids → Impressions):**
- **Clean Win PS AUC**: ~0.91 (LightGBM, 기존 0.993은 usertag leakage)
- **CTR Bias**: +4.57% overestimation (LGB), LR은 -1.12% (모델 의존적)
- **Covariate Shift Drivers**: `bid_floor_ratio`(d=0.83, Large), `slotprice`(d=-0.69, Medium) — numerical features only (KS test)
- **Subgroup Heterogeneity**: Exchange별 -8.4%~+10.3% (Simpson's Paradox), Advertiser별 -10%~+10.5%
- **Positivity**: Overlap [0.1,0.9] 47.8%, ESS ratio 9.66% → IPW 제한적

**Click Stage (Impressions → Clicks):**
- **Click PS**: CLI 학습 LGB 로드 (`load_click_propensity_models()`), AUC 0.7275 (CTR 0.07%, 1:1451 ratio)
- **CVR Distortion**: Retargeting advertisers only (branding has CVR=0)
- **Advertiser Dependency**: Branding → CTR debiasing만, Retargeting → CVR debiasing critical

**Advertiser Taxonomy:**
- Retargeting (2821, 3358, 2259): CVR 28-53%, click→conv ~1초 → ESCM²(DR) full pipeline
- Branding (1458, 3386, 3427, 2261, 2997): CVR=0 → ESMM/PLE CTR tower only
- Mixed (3476): Conservative ESCM²(DR)

**Usertag Leakage (historical)**: `has_tags`/`n_tags` 소스 제거 완료 (2026-02-12)

### Baseline Model Performance (from result JSONs, n_estimators=300, min_child_samples=50)
> ⚠️ **PRE-FAIR-SPLIT (superseded).** All AUC numbers below are from the **original/unfair**
> disjoint-advertiser split (e.g. "LR 0.7687") — the redesign showed these to be an artifact. Current
> fair-split numbers (neural winners 0.658 > LGB 0.632 > LR 0.554) and the calibration→surplus result
> are in `docs/redesign_findings.md`; protocol in `docs/evaluation_protocol.md`.
- **LGB CTR (biased)**: Train AUC 0.8113 → Test AUC 0.6890 (20 trees, winners-only)
- **LR CTR_all (best all-bids)**: Train AUC 0.7731 → Test AUC 0.7687 — temporal shift에 robust, all-bids 최고
- **LR CTR (winners-only)**: Train AUC 0.6333 → Test AUC 0.3216 — LR은 winners-only에서 열위
- **LGB Win**: Train AUC 0.9308 → Test AUC 0.6493 (300 trees)
- **LGB CTR_all**: Train AUC 0.9059 → Test AUC 0.5437 (39 trees, severe overfitting)
- **ESMM-WC Run J (best)**: Test CTR AUC 0.6237 (winners), Test WCTR AUC 0.6905 (all-bids) — **LGB CTR_all(0.54) 대비 +0.15 우위**
- **공정 비교**: All-bids 기준 LR(0.7687) > ESMM-WC(0.6905) > LGB(0.5437). Winners-only 기준 LGB(0.6890) > ESMM-WC(0.6237) > LR(0.3216)
- **CTR Calibration**: Uniform ECE misleading (resolution=0) — quantile calibration으로 재측정 (LGB CTR ECE 0.06, LR CTR_all 분석 추가)
- **Temporal Shift**: S2→S3 전 LGB 모델에서 심각한 Train→Test AUC 하락 (CTR: 0.81→0.69, Win: 0.93→0.65, CTR_all: 0.91→0.54). LR CTR_all만 유일하게 robust (0.77→0.77)
- **LR CTR_all feature structure**: Top features `adexchange`(-0.61), `weekday`(-0.32), `hour`(+0.25) — contextual/temporal 지배. `bidprice`(#12), `slotprice`(#24 ≈0) 비지배적. CTR과 5+ sign reversals (Spearman ρ=0.409)
- **LR coefficients 검증**: top features 시각화 추가 — bidprice/slotprice 위치 직접 확인
- **LGB overfitting**: 0.007% positive rate에서 순차적 residual fitting → noise memorization. Early stopping 무효
- **Bid shading**: LR CTR_all 기반 V(x) 분석 추가 (well-calibrated, AUC 0.78)
- **LR CTR vs CTR_all 격차 근본 원인**: 동일 13K positives, 다른 negatives — CTR_all의 71.6M lost bids가 "easy negatives"로 작용. Top features sign reversal (adexchange #1 양측 반대 방향). Contextual features (adexchange, temporal) anchor → task separability 결정적
- **Limitation**: CTR baseline은 winners only에서 학습 (win selection bias) — flat-bid → 저경쟁 인벤토리 과대표

### Notebook 04: DR 메커니즘 + Ablation Study (2026-03-17)
- **DR Unbiasedness**: Toy simulation (10K samples, 100 reps) — Naive bias +0.06, IPW/DR ~0 (DR lower variance)
- **ESMM-WC vs ESCM2-WC(DR)**: J(WCTR AUC 0.6905, IEB 1.335) vs AL(0.6843, IEB 0.014) — AUC -0.006 but **IEB 95x 개선**
- **CFR Lambda Ablation**: 0.0(0.6638) < 0.1(0.6766) < **0.2(0.6843, IEB 0.014)** > 0.3(0.6841, IEB 0.114) > 0.5(0.6774, IEB 0.105). cfr=0.2 sweet spot
- **External PS**: AW(AL+ExtPS) WCTR AUC 0.6882 (+0.004 vs AL), IEB 0.045 (+0.031) — AUC-Calibration trade-off
- **Negative Results**: Per-tower dropout(AM, -0.047 AUC), Checkpoint averaging(AP, -0.012), Huber imputation(AJ, -0.018)
- **Figures**: 8 figures saved to `results/figures/04_*.png`

### Neural 모델 재학습 + Diagnostics (2026-03-22)
- **ESMM-WC (Run J 재학습)**: Test WCTR AUC 0.6527, IEB 0.075. Val CTR AUC 0.706 (best epoch 5/10)
- **ESCM²-WC(DR) (Run AL 재학습)**: Test WCTR AUC 0.6851, IEB 0.073. Val CTR AUC 0.709 (best epoch 5/10)
- **ESCM²-WC(DR)+ExtPS (Run AW 재학습)**: Test WCTR AUC 0.6837, IEB 0.045. Val CTR AUC 0.709 (best epoch 5/10)
- **ExtPS suffix 수정**: `train.py` model_name에 `_extps` suffix 추가 — ExtPS 결과 파일 분리
- **Diagnostics figures**: 4개 3-panel plots 생성 (LR CTR_all AUC 0.769, Neural 3개)
- **prediction_report.md**: Figure 11-14 추가 (ESCM2-DR, ESMM-WC, ExtPS, LR CTR_all)

### Notebook 05: Win Rate Analysis & Market Price CDF (2026-03-17)
- **Flat Bidding**: iPinYou uses only 6 discrete bid prices (227–300 CPM). Cross-advertiser pooling required for win rate analysis.
- **KM Market Price CDF**: True median > 300 CPM (unidentifiable, S(300)=0.79). F(300) ≈ 21.3% ≈ overall WR — KM well-calibrated.
- **Exchange-Conditional**: Ex1 F(300)=68.8% (median 153 CPM), Ex2 F(300)=29.1%, Ex3 F(300)=11.9% — exchange-conditional shading essential
- **Market Price Stats (winners)**: Mean 78, Median 68, P90 166, P95 214 CPM. Floor binding 20.8%.
- **Parametric Fit**: LogNormal best AIC/BIC (mu=7.93, sigma=2.52). Weibull 2nd, Exponential worst.
- **Temporal Drift**: S2→S3 KS=0.118 on pay prices — market shifted rightward, re-estimation needed.
- **SP3 Shade Demo**: V(x)=200 → optimal bid=89 CPM (shade=44.6%), surplus=17.2 CPM.
- **Calibration Economic Value (5-model)**: AL(IEB 0.014) → oracle surplus, AW(0.045) → ~0%, LR CTR_all(0.122) → -2.2%, LGB CTR(0.362) → -2.2%, ESMM-WC J(1.335) → -14.7%. Ex1: LGB -8.2%, J -42%. AUC best (LR 0.7687) ≠ bidding best.
- **Modules**: `src/win_rate/nonparametric.py` (Wilson CI, empirical curves, lookup), `src/win_rate/survival.py` (KM, parametric, segment CDF)
- **Figures**: 9 figures `results/figures/05_*.png` (05_calibration_economic_value.png 5-model 재생성), CDF export `results/market_price_cdf/`

### Notebook 03 Refactor + LGB Hyperparameter 개선 (2026-02-23)
- **`scripts/train.py`**: LGB params 개선 — `n_estimators=300`, `min_child_samples=50`, `subsample=0.8`, `feature_fraction=0.8`, `early_stopping=30`
- **`docs/scripts_tutorial.md`**: `--n-estimators` default 100→300 업데이트
- **Notebook 03 신규 섹션**: (1) Temporal degradation table, (2) Why LGB CTR_all fails 분석, (3) LR CTR_all won-subset 재평가, (4) Adaptive calibration (quantile bins), (5) Bid shading V(x) perspective
- **Cell 0**: Key Findings 4개 bullet 추가
- **Cell 17 → Summary**: 7개 Key Observations로 전면 재작성
- **Feature Ablation**: PLAN.md Task 1.4 추가 (domain, creative, target encoding, S2→S3 overlap)

### Notebook 03 Senior DS Review + 수치 동기화 (2026-02-23)
- **P0 수치 동기화**: Cell 8 테이블 (LGB 3 tasks 수치 갱신), Cell 23 Summary (trees/AUC 갱신), Cell 0 Key Finding #4, Cell 19 markdown (V(x) 해석 전면 재작성)
- **P1 분석 보강**: (1) LR coefficients 검증 셀 추가 (Cell 13), (2) LR CTR_all calibration 셀 추가 (Cell 20), (3) LR CTR_all pCTR + V(x) 분석 셀 추가 (Cell 23)
- **P2 코드/해석 수정**: (1) Cell 22 `dir()` 기반 → `y_true is not None` 단순화, (2) `scripts/train.py:170` ECE boundary `>` → `>=` 수정, (3) Cell 8 temporal split 구조 명시

### Notebook 03 재학습 결과 반영 (2026-02-24)
- **LGB CTR 재학습**: 1 tree → 11 trees, Train AUC 0.6814→0.7507, Test AUC 0.4469→0.4965
- **LGB CTR_all 재학습**: 1 tree → 29 trees, Train AUC 0.8154→0.8772, Test AUC 0.4541→0.5299
- **Notebook 03 수치 교체**: Cell 0 (Key Finding #4), Cell 5, Cell 8 (Temporal Table), Cell 11 (LGB failure 분석), Cell 21 (Bid Shading), Cell 26 (Summary)
- **`scale_pos_weight` 참조 완전 제거**: 노트북 + PLAN.md에서 모든 spw 관련 서술 삭제
- **서사 전환**: spw 역효과 → LGB overfitting/temporal drift 한계 + selection bias → ESMM-WC 동기

### Feature Engineering: domain + creative Hash/Frequency Encoding (2026-02-25)
- **`src/features/engineering.py`**: `hash_encode()` (MD5, deterministic, unique-value map), `add_high_card_features()` (hash + freq + freq_log)
- **Features added**: `domain_hash`(10K buckets), `creative_hash`(5K buckets), `domain_freq`, `domain_freq_log`, `creative_freq`, `creative_freq_log`
- **Total features**: 24 → 30 (categorical 15→17, numerical 9→13)
- **Neural input dim**: 24×32=768 → 30×32=960 (+24%)
- **Config**: `configs/features/default.yaml` hash_encoding section, CLI `--domain-buckets`/`--creative-buckets`
- **train.py 변경 불필요**: `feature_dims = max+2` 로직이 hash features에 자동 대응 (`n_buckets+2 ≥ n_buckets+1`)
- **Feature rebuild 완료** (2026-02-25): 129.5M rows, 24분 소요. Categorical 17, Numerical 13, Total 30 확인. Z-score stats 13 numerical features 포함
- **Split**: Train 90.6M (WR 21.0%, CTR 0.070%) | Val 19.4M (WR 38.0%, CTR 0.072%) | Test 19.4M (WR 21.8%, CTR 0.106%)

### Distributed Training Infrastructure (2026-02-25)
- **`src/distributed/mesh.py`**: JAX SPMD Mesh 생성, `create_mesh()` + `get_data_sharding()` for multi-device data parallelism
- **`src/distributed/data_loader.py`**: grain `DataLoader` 기반 — `RTBDataSource` (Parquet→dict), `materialize_to_source()` (feature split), `batch_to_jax()` (numpy→jax array + optional sharding)
- **`src/distributed/train_state.py`**: `create_optimizer()` (AdamW + gradient clipping + warmup/cosine/linear LR schedule), `create_train_state()` (Flax TrainState factory)
- **`src/distributed/checkpoint.py`**: orbax `CheckpointManager` 래퍼 — save/restore with step metadata, automatic cleanup
- **Invariants**: `batch_to_jax()` 출력 = `{"x": Dict[str, Array], "win": Array, "click": Array}` (single/multi-device 동일), CLI args ↔ DistributedConfig 1:1 대응
- **Dependencies**: `grain>=0.2.14`, `orbax-checkpoint>=0.4.0` (pyproject.toml)

### Neural Model Smoke Test (2026-02-25)
- **목적**: ESMM-WC / ESCM²-WC 코드가 한 번도 실행된 적 없어 GPU 서버 학습 전 파이프라인 작동 검증
- **설정**: 200K train samples, 40K val/test, 2 epochs, batch_size=1024 (macOS CPU)
- **`--max-samples` flag 추가**: `scripts/train.py` esmmwc/escm2wc 커맨드 + `_train_wc_model()` — Parquet 로드 후 `head(N)` 서브샘플링
- **Flax 0.12 API 호환성 수정** (Bug Fix):
  - `nnx.Optimizer(model, tx)` → `nnx.Optimizer(model, tx, wrt=nnx.Param)` (Flax 0.11.0+ required `wrt` arg)
  - `optimizer.update(grads)` → `optimizer.update(model, grads)` (Flax 0.11.0+ `update()` signature change)
  - 영향 파일: `scripts/train.py`, `src/models/esmm_wc.py`, `src/models/escm2_wc.py`
- **결과** (200K samples, 2 epochs — 값 자체는 소량 데이터이므로 의미 없음):

| Model | Training Time | Win AUC | CTR AUC (biased) | WCTR AUC | CTR IEB | WCTR IEB |
|-------|-------------|---------|-------------------|----------|---------|----------|
| ESMM-WC | 25.5s | 0.5736 | 0.4308 | 0.2490 | 3.056 | 1.521 |
| ESCM²-WC(IPW) | 26.3s | 0.5673 | 0.3835 | 0.3022 | 1.585 | 0.261 |
| ESCM²-WC(DR) | 25.8s | 0.5798 | 0.6286 | 0.4066 | 10.620 | 8.587 |

- **성공 기준 충족**: (1) No RuntimeError/OOM, (2) No NaN/Inf loss, (3) 3개 결과 JSON 생성, (4) 전 metric 계산 완료
- **알려진 제한**: `src/distributed/checkpoint.py`의 `nnx.Optimizer` 사용도 Flax 0.12 호환성 수정 필요 (distributed 모드에서만 영향, 로컬 smoke test 범위 밖)

### Synthetic Data Sanity Test (2026-02-26)
- **목적**: Smoke test에서 AUC 0.43~0.63으로 낮았으나, 극저 CTR(0.07%) + 2 epochs 때문. 모델 아키텍처/학습 로직 자체 검증을 위해 쉬운 합성 데이터로 AUC 0.8+ 달성 확인
- **`scripts/generate_synthetic.py`**: 합성 데이터 생성 스크립트 (기존 파이프라인 포맷 완전 일치)
- **합성 데이터 설계**:
  - 규모: Train 50K, Val 10K, Test 10K (빠른 iteration)
  - 30개 feature (17 categorical + 13 numerical), 실제 데이터와 동일 구조
  - **Win signal** (bidprice, slotprice, adexchange 기반): Bayes-optimal AUC ~0.82
  - **Click signal** (slot_area, region_freq, hour_sin/cos, advertiser, slotformat 기반): Bayes-optimal AUC ~0.84
  - **핵심 설계 원칙**: Win/Click signal은 orthogonal features 사용 (겹치면 selection bias로 CTR AUC 저하)
  - Win rate ~35%, Click rate ~29% of winners (실제보다 높게 — 학습 용이)
- **학습 설정**: 50 epochs, batch_size=2048, lr=0.0005, early stopping patience=10
- **결과**:

| Model | Win AUC | CTR AUC | WCTR AUC | Time |
|-------|---------|---------|----------|------|
| ESMM-WC | **0.8090** | **0.8343** | 0.8352 | 74s |
| ESCM²-WC(IPW) | **0.8076** | **0.8250** | 0.8308 | 87s |
| ESCM²-WC(DR) | **0.8090** | **0.8371** | 0.8397 | 87s |

- **성공 기준 충족**: Win AUC ≥ 0.8, CTR AUC ≥ 0.8 (3개 모델 모두)
- **핵심 발견**:
  1. 모델 아키텍처/학습 로직에 버그 없음 확인
  2. Win/Click signal이 orthogonal해야 CTR AUC가 높음 (bidprice를 click signal에 넣으면 won-only 평가에서 selection bias 발생)
  3. 낮은 learning rate (0.0005)가 win tower 과적합 방지에 효과적
  4. DR 모델이 CTR AUC 최고 (0.8371) — DR debiasing 효과 확인

### W&B Sweep Integration (2026-02-26)
- **문제**: Sweep agent가 `wandb.init()`으로 sweep-linked run 생성 → `_train_wc_model()`이 다시 `wandb.init()` 호출 → double init으로 첫 run이 끊김
- **해결**: `wandb_run` parameter를 `_train_wc_model()`에 추가, `_wandb_run_provided` flag로 init/finish lifecycle 관리
- **Debiasing HP CLI 노출**: `--dropout`, `--cfr-lambda`, `--win-eps`, `--max-weight`, `--win-weight`, `--ctr-weight`, `--joint-weight`, `--impute-loss-weight` (escm2wc), `--dropout` (esmmwc)
- **Sweep config**: Bayes optimization + Hyperband early termination (ESCM²-WC: 15 params, ESMM-WC: 7 params)
- **파일**: `scripts/sweep.py` (create/agent CLI), `experiments/sweep_escm2wc.yaml`, `experiments/sweep_esmmwc.yaml`

### Validation Loss Dropout Bug Fix (2026-02-27)
- **증상**: ESMM-WC validation loss가 epoch마다 단조 증가 (0.5712 → 0.5874 → 0.5925 → 0.5933)
- **원인**: `create_esmm_wc_loss_fn()`/`create_escm2wc_loss_fn()`의 inner `loss_fn`이 `training=True` 하드코딩 → validation에서도 dropout=0.3 활성 → P_click = P_win × P_ctr에서 noise 증폭
- **수정**: `loss_fn` 시그니처에 `training: bool = True` parameter 추가, validation 호출부에서 `training=False` 전달
- **파일**: `src/models/esmm_wc.py:185`, `src/models/escm2_wc.py:228`, `scripts/train.py:1601`

### ESMM-WC Val Loss 단조 증가 — Optimizer 분리 + Loss Weight CLI (2026-02-27)
- **증상**: Dropout 버그 수정 후에도 val loss 단조 증가 (0.5695 → 0.5836)
- **원인 A (Win tower overfitting)**: Single-device에서 Adam(lr=1e-3), weight decay·schedule·clipping 모두 없음 — `distributed` guard가 advanced optimizer를 차단
- **원인 B (CTR tower 미학습)**: batch당 click ≈ 0.6 (BS=4096, CTR 0.07%) → 55% 배치에 click 0개 → gradient ≈ 0. `esmmwc` CLI에 `--ctr-weight` 없어 조정 불가
- **Fix A**: `_train_wc_model()` optimizer 분기에서 `distributed` guard 제거 → `weight_decay > 0` 포함 시 항상 `create_optimizer()` 사용. `lr_schedule_fn`도 동일 수정. `--weight-decay` CLI 양쪽 추가 (기본값 1e-5)
- **Fix B**: `esmmwc()` CLI에 `--win-weight`, `--ctr-weight`, `--joint-weight` 추가, `ESMMWCConfig` 생성 시 전달
- **W&B**: `weight_decay` + loss weights를 base wandb_config에 추가 (esmmwc에서도 기록)
- **호환성**: `--weight-decay 0`으로 기존 vanilla Adam 동작 동일. `--scheduler cosine` 단독(non-distributed)도 정상 동작
- **파일**: `scripts/train.py` (optimizer 분리, CLI 추가, config 전달, W&B config)

### ESMM-WC 성능 분석 및 Fix 1+2 (2026-02-27)
- **문제 분석**: `esmmwc_result.json`에서 val→test 성능 급락 발견 (win_auc 0.837→0.583, ctr_biased_auc 0.699→0.298)
- **Root Causes**: (1) Best model 미저장 — epoch 10 (overfitted) 모델로 평가, best는 epoch 2, (2) Win loss가 total의 87% 지배 — CTR 최적화 무관, (3) CTR class imbalance (0.075%) 미대응, (4) Temporal distribution shift (S2→S3), (5) max_samples로 가장 오래된 데이터만 선택
- **Fix 1 (Best Model Save/Restore)**:
  - `scripts/train.py`: `best_state = None` 초기화, `nnx.split(model)` on best val loss, `nnx.update(model, best_state)` before evaluation
  - Best epoch 모델로 평가하여 overfitting 방지
- **Fix 2 (Loss Weight — 논문 충실 수정)**:
  - 원래: `ctr_weight=20, joint_weight=20` — 논문에 없는 term 과도한 가중
  - ESMM (Ma et al., 2018): `L = L_CTR + L_CTCVR` — ctr_loss(won-only 직접감독)는 논문에 없음
  - ESCM² (Wang et al., 2022): `L = L_CTR + L_CTCVR + λ_c · L_CVR^{DR}` — λ_c ∈ [0, 0.1]
  - **esmmwc**: `--ctr-weight` 20→**0.0** (논문에 없는 term 제거), `--joint-weight` 20→**1.0**
  - **escm2wc**: `--ctr-weight` 20→**0.1** (λ_c 상한), `--joint-weight` 20→**1.0**
  - 파일: `scripts/train.py` (CLI defaults 4곳)
- **Early Stopping patience 10→5**: `max_patience = 10` → `max_patience = 5` (line 1588) — overfitting 조기 차단

### ESMM-WC 성능 개선 — Early Stopping + CLI 확장 (2026-03-03)
- **Root Cause Analysis**: (1) 배치당 click 0.7개 (batch_size=4096, CTR 0.017%) → CTR gradient 소실, (2) early stopping이 `val_total`(win 지배) 기준 → CTR 미수렴 상태에서 학습 종료, (3) `patience=5` hardcoded → 유연성 부족
- **`scripts/train.py` 변경**:
  - `--es-metric` CLI 추가 (esmmwc/escm2wc): `total` (기존), `joint` (CTR-relevant), `ctr_auc` (직접 AUC 최적화). default=`joint`
  - `--patience` CLI 추가: hardcoded `max_patience=5` → 파라미터화 (default=10)
  - `--use-layer-norm/--no-use-layer-norm` CLI 추가: `ESMMWCConfig`/`ESCM2WCConfig`에 전달
  - Early stopping 로직: val metrics 계산을 early stopping 전으로 이동 (`ctr_auc` 사용 시 매 epoch eval 강제)
  - W&B config 및 result JSON에 `es_metric`, `patience`, `use_layer_norm` 기록
- **`configs/model/esmmwc.yaml`**: `use_layer_norm: false` → `true`
- **`experiments/sweep_esmmwc.yaml`**: metric `val_total` → `val_joint`, batch_size `[2048,4096,8192]` → `[32768,65536]`, `es_metric`/`patience`/`weight_decay`/`scheduler`/`warmup_steps`/`gradient_clip`/`use_layer_norm` search space 추가
- **권장 실험 커맨드**: `--batch-size 65536 --es-metric joint --patience 15 --epochs 50 --scheduler cosine --warmup-steps 200 --gradient-clip 1.0 --weight-decay 1e-4 --dropout 0.2 --embedding-dim 32 --hidden-dims "256,128,64" --use-layer-norm`
- **기대 효과**: batch 65536에서 배치당 click 0.7→11개, joint 기준 early stopping으로 CTR tower 수렴 보장

### ESCM2-WC Config/Sweep YAML 동기화 (2026-03-03)
- **`configs/model/escm2wc_dr.yaml`**: `use_layer_norm: false` → `true` (ESMM-WC와 동일 default)
- **`configs/model/escm2wc_ipw.yaml`**: `use_layer_norm: false` → `true` (ESMM-WC와 동일 default)
- **`experiments/sweep_escm2wc.yaml`**: ESMM-WC sweep 패턴과 동일하게 업데이트 (ESCM2 고유 debiasing params 유지)
  - metric: `val_total` → `val_joint`
  - embedding_dim: `[8,16,32]` → `[16,32]`, hidden_dims: 소형 제거, dropout max: 0.5→0.3
  - batch_size: `[2048,4096,8192]` → `[32768,65536]`, lr: `[1e-4,1e-2]` → `[5e-4,5e-3]`
  - **신규 params**: `use_layer_norm`, `es_metric`(joint fixed), `patience`, `weight_decay`, `scheduler`, `warmup_steps`, `gradient_clip`
  - **유지**: debiasing, cfr_lambda, win_eps, max_weight, impute_loss_weight, win/ctr/joint_weight

### ESMM-WC Phase 2 Regularization 실험 (2026-03-03)
- **목적**: Phase 1 LR 탐색에서 S2 overfitting 확인 (val CTR AUC ~0.71 but test CTR AUC 0.37~0.52). Regularization 강화로 일반화 개선 시도
- **공통 설정**: LR=1e-3, batch_size=65536, epochs=50, patience=15, es-metric=ctr_auc, scheduler=cosine, warmup=200, gradient-clip=1.0, weight-decay=1e-3, embedding-dim=32, use-layer-norm
- **Phase 1 baseline**: Run C (lr=3e-3) Test CTR AUC 0.517, LR baseline Test CTR AUC 0.783

| Run | Config | Best Ep | Test Win AUC | Test CTR AUC | Test WCTR AUC | Test CTR IEB |
|-----|--------|---------|-------------|-------------|--------------|-------------|
| D | dropout=0.3, hidden=128,64 | 13 | 0.6199 | 0.4937 | 0.5886 | 0.3972 |
| E | dropout=0.5, hidden=64,32 | 13 | 0.5834 | **0.5160** | 0.3971 | 0.4469 |
| F | dropout=0.4, hidden=128,64, win-weight=0.1 | 5 | 0.6369 | **0.5226** | **0.6005** | 0.6617 |

- **결과 분석**:
  - Run F (win-weight=0.1)가 **Test CTR AUC 0.5226** 달성 — Phase 1 best (0.517) 소폭 개선
  - Run F의 win-weight 축소가 CTR tower 학습에 긍정적 (best epoch 5 = 빠른 수렴, overfitting 전 중단)
  - Run E (극단적 capacity 축소)는 CTR AUC 0.516으로 Phase 1 best와 동일 수준 — capacity가 아닌 다른 병목
  - Run D (moderate reg)는 CTR AUC 0.494 — regularization만으로는 S2→S3 temporal drift 해결 불가
  - **성공 기준 미달**: 0.517 → 0.5226 (소폭 개선), 목표 0.65 미달
  - **핵심 인사이트**: Win tower gradient 지배 완화(win-weight=0.1)가 가장 효과적. 그러나 S2→S3 temporal drift가 근본 병목 — regularization만으로 해결 불가. LR baseline(0.783)과의 격차는 NN의 overfitting이 아닌 temporal distribution shift가 주요 원인

### ESMM-WC Phase 3 Loss Weight 탐색 실험 (2026-03-04)
- **목적**: Phase 2 Run F (win-weight=0.1, test CTR AUC 0.5226) 기반, win-weight 추가 축소 + joint-weight 증가 효과 탐색
- **공통 설정**: LR=1e-3, batch_size=65536, epochs=50, patience=15, es-metric=ctr_auc, scheduler=cosine, warmup=200, gradient-clip=1.0, weight-decay=1e-3, dropout=0.4, embedding-dim=32, hidden-dims="128,64", use-layer-norm

| Run | win_weight | joint_weight | Best Ep | Test Win AUC | Test CTR AUC | Test WCTR AUC |
|-----|-----------|-------------|---------|-------------|-------------|--------------|
| F(base) | 0.1 | 1.0 | 5 | 0.6369 | 0.5226 | 0.6005 |
| **G** | **0.01** | **1.0** | **3** | **0.6518** | **0.5888** | **0.6426** |
| H | 0.1 | 3.0 | 3 | 0.6331 | 0.5334 | 0.5815 |
| I | 0.01 | 3.0 | 3 | 0.6283 | 0.5591 | 0.6207 |

- **결과 분석**:
  - **Run G (ww=0.01, jw=1.0)가 Phase 전체 Best**: Test CTR AUC **0.5888** (+0.0662 vs Run F)
  - Win-weight 극소화(0.01)가 가장 효과적: G(0.5888) > I(0.5591) >> H(0.5334) > F(0.5226)
  - Joint-weight 증가(3.0)는 오히려 해로움: G(jw=1.0, 0.5888) > I(jw=3.0, 0.5591), F(jw=1.0, 0.5226) < H(jw=3.0, 0.5334)
  - Win-weight가 dominant factor: ww=0.01 두 run (G, I)이 ww=0.1 두 run (F, H) 보다 일관 우수
  - **핵심 인사이트**: Win tower gradient를 거의 끄고 (ww=0.01) joint constraint(P(win)×P(ctr|win))만으로 학습 시, CTR tower가 win tower 간섭 없이 ESMM debiasing signal을 효과적으로 흡수. Joint-weight 과도한 증가는 loss landscape를 왜곡하여 역효과
  - 모든 Phase 3 runs가 early stop epoch 3 — 빠른 수렴 확인
  - **성공 기준 달성**: 0.5888 > 0.5226 (Phase 2 best)

### ESMM-WC Phase 4 수렴 속도 조절 실험 (2026-03-04)
- **목적**: Phase 3 Run G (best epoch=3)의 빠른 수렴 문제 해결 — LR 감소 + warmup 조절로 peak epoch 후방 이동 및 일반화 개선
- **공통 설정**: Run G 기반, batch_size=65536, patience=15, es-metric=ctr_auc, scheduler=cosine, gradient-clip=1.0, weight-decay=1e-3, dropout=0.4, embedding-dim=32, hidden-dims="128,64", use-layer-norm, win-weight=0.01

| Run | learning_rate | warmup_steps | Best Ep | Val CTR AUC | Test CTR AUC | Test WCTR AUC |
|-----|--------------|-------------|---------|-------------|-------------|--------------|
| G(base) | 1e-3 | 200 | 3 | 0.7059 | 0.5888 | 0.6426 |
| J | 5e-4 | 200 | 5 | 0.7055 | 0.6237 | 0.6905 |
| K | 3e-4 | 200 | 7 | 0.7089 | 0.5684 | 0.6726 |
| L | 3e-4 | 1000 | 7 | 0.7106 | 0.5744 | 0.6737 |

- **결과 분석**:
  - **Secondary 성공**: LR 감소로 peak epoch 후방 이동 확인 — G(ep3) → J(ep5) → K/L(ep7). 가설 검증 완료
  - **Primary 미달**: Test CTR AUC 0.5888 (Run G) 넘지 못함. Run J(0.6237)만 WCTR에서 우수하나 CTR에서는 여전히 미달
  - **Val-Test 괴리 심화**: Val CTR AUC는 K(0.7089), L(0.7106)이 G(0.7059)보다 높지만, Test에서는 K(0.5684), L(0.5744) < G(0.5888)
  - **핵심 인사이트**: LR 감소가 수렴을 늦추고 Val AUC peak을 살짝 올리지만, test 일반화로 이어지지 않음. Val-Test distribution shift가 근본 원인일 가능성
  - Run J(lr=5e-4)가 Test WCTR AUC 0.6905로 전체 best WCTR — 느린 수렴이 WCTR에는 도움
  - **ESMM-WC best는 여전히 Run G** (Test CTR AUC 0.5888). Phase 5에서는 ESCM²-WC(DR)로 전환하여 DR debiasing 효과 확인 필요

### ESMM-WC Phase 5 Regularization 강화 실험 (2026-03-04)
- **목적**: Phase 4 Run J (best epoch=5) 후 단조 하락 overfitting 문제 — dropout/weight-decay 강화로 일반화 개선 시도
- **공통 설정**: Run J 기반, lr=5e-4, warmup=200, batch_size=65536, patience=5, es-metric=ctr_auc, scheduler=cosine, gradient-clip=1.0, embedding-dim=32, hidden-dims="128,64", use-layer-norm, win-weight=0.01

| Run | dropout | weight_decay | Best Ep | Val CTR AUC | Test CTR AUC | Test WCTR AUC |
|-----|---------|-------------|---------|-------------|-------------|--------------|
| J(base) | 0.4 | 1e-3 | 5 | 0.7055 | 0.6237 | 0.6905 |
| M | 0.5 | 1e-3 | 5 | 0.7052 | 0.5094 | 0.6444 |
| N | 0.4 | 3e-3 | 5 | 0.7053 | 0.5332 | 0.6583 |
| O | 0.5 | 3e-3 | 7 | 0.7058 | 0.5478 | 0.6656 |

- **결과 분석**:
  - **Primary 실패**: 전 run에서 Test CTR AUC 하락 — M(0.5094), N(0.5332), O(0.5478) vs J(0.6237)
  - **Secondary 부분 성공**: Run O에서 best epoch 5→7로 후방 이동 (두 regularization 시너지)
  - **Val-Test 괴리 심화**: Val CTR AUC 거의 동일 (~0.705) but Test CTR 큰 차이 → regularization이 아닌 distribution shift가 근본 원인
  - **핵심 인사이트**: 강한 regularization이 오히려 test 성능 악화. 모델이 val에서 학습한 패턴이 test(S3)에서 유효하지 않음
  - **ESMM-WC 최종 best: Run J** (Test CTR AUC 0.6237, Test WCTR AUC 0.6905) 유지
  - **결론**: ESMM-WC 튜닝 한계 도달. ESCM²-WC(DR)로 전환하여 DR debiasing 효과 확인 필요

### ESCM2-WC Phase 7-12 실험 결과 (2026-03-08~09)

**결론: ESCM2-WC(DR)이 ESMM-WC Run J(0.6905)를 돌파하지 못함.**

- Phase 7 (Transfer): DR(0.6728), IPW(0.6526) — 3rd tower overhead로 ESMM-WC 미달
- Phase 8 (ctr_weight): Run R(0.6766) — ctr_weight=0.01 미약한 개선, 0.1은 catastrophic
- Phase 9 (win_weight↑): 0.1~1.0 모두 Val-Test gap 급증, catastrophic failure
- Phase 10 (stop_grad): Win AUC↑ but CTR 여전히 실패 — win tower 자체 overfitting
- Phase 11 (DR formulation): BCE/MSE, clipping 변경 무의미 (Run R ±0.006)
- Phase 12 (numeric bypass/scalar): 모든 architecture 변경이 catastrophic (0.33~0.56)

**핵심 인사이트:**
1. ESCM2-WC의 3rd tower(Imputation)가 temporal shift에 추가 취약점 생성
2. win_weight=0.01이 ESCM2-WC에서도 최적 — 논문과 상반 (iPinYou의 temporal shift 특성)
3. DR debiasing이 win_weight=0.01에서는 사실상 비활성 (weak propensity → noise DR signal)
4. Numeric bypass/scalar은 categorical embedding의 expressiveness를 손상
5. **ESMM-WC Run J(WCTR 0.6905)가 best neural model — ESCM2는 overhead만 추가**

### Phase 13B-16 실험 완료 (2026-03-10)

**결과: ESCM2-WC(DR)는 ESMM-WC Run J(0.6905) 돌파 실패**

- [x] `src/models/escm2_wc.py`: `impute_loss_type`/`impute_huber_delta` (Huber loss), per-tower dropout
- [x] `scripts/train.py`: CLI 옵션 (`--impute-loss-type`, `--impute-huber-delta`, `--win-dropout`, `--ctr-dropout`, `--impute-dropout`, `--top-k-avg`), result JSON 기록, top-K checkpoint averaging
- [x] `configs/model/escm2wc_dr.yaml`, `escm2wc_ipw.yaml`: 새 필드 추가
- [x] `docs/performance_tuning.md`: Phase 13B-16 실험 결과 기록
- [x] Phase 13B: AJ(Huber 0.6664), AK(cfr=0 0.6638), **AL(cfr=0.2 0.6843 best)**
- [x] Phase 14: AM(0.6377), AN(0.6471) — per-tower dropout catastrophic
- [x] Phase 15: SKIPPED (14 전면 실패)
- [x] Phase 16: AP(AL+top-k-avg=3 → 0.6722) — checkpoint avg 역효과

**구조적 한계 확인:**
1. win_weight=0.01 → propensity 부정확 → DR debiasing 효과 제한
2. win_weight↑ → shared embedding overfit → CTR 붕괴 (Phase 9)
3. 3rd tower(Imputation) overhead가 temporal shift에 추가 취약점
4. ESCM2-WC best(AL 0.6843) < ESMM-WC(J 0.6905) — 일관적 열위

### Phase 17-18 실험 완료 (2026-03-16)

**Phase 17: cfr_lambda/impute_loss_weight 확장 — AL(0.2/0.5) 최적 확정**
- Phase 17-1: AQ(cfr=0.3, WCTR 0.6841), AR(cfr=0.5, 0.6774) — AUC 유지~하락, IEB 8배 악화 (0.014→0.10+)
- Phase 17-2: AS(ilw=0.3, 0.6866), AT(ilw=1.0, 0.6759) — AS AUC 소폭 개선이나 IEB 급증
- Phase 17-3: SKIPPED (17-1/2 모두 AL 대비 IEB 악화, 추가 탐색 불필요)

**Phase 18: External Win PS — Run AW WCTR AUC 0.6882 (ESCM2-WC best AUC)**
- AW(AL+Ext PS): WCTR AUC **0.6882** (+0.004 vs AL), CTR AUC 0.5713, IEB 0.045
- AV(Run J cfg+Ext PS): WCTR AUC 0.6712 — ctr_weight=0.0이 DR 전파를 차단
- External PS Train AUC 0.93 → Test AUC 0.65: temporal shift로 PS 품질 급감

**최종 모델 선택:**
- **Calibration best**: Run AL (WCTR ECE 0.000003, IEB 0.014) — bid price calibration 중시
- **AUC best**: Run AW (WCTR AUC 0.6882, IEB 0.045) — ranking 중시
- ESMM-WC Run J(WCTR 0.6905)에 근접하면서 calibration 압도적 우위 (IEB 95x 개선)

### Scripts Tutorial Update (2026-02-26)
- **문서 갱신**: `docs/scripts_tutorial.md` — 전 CLI flags, training output, W&B, Sweep, Distributed training 문서화
- **esmmwc**: 8→21 flags (13 추가: learning-rate, dropout, eval-every, quiet, distributed×6, wandb×3)
- **escm2wc**: 8→28 flags (20 추가: esmmwc 누락분 + cfr-lambda, win-eps, max-weight, win/ctr/joint/impute-loss-weight)
- **새 섹션**: Training Output & Metrics (loss components, val metrics 의미, result JSON 구조)
- **새 섹션**: W&B Experiment Tracking (--use-wandb, epoch/final logging)
- **새 섹션**: W&B Sweep (sweep.py create/agent, sweep config, end-to-end 예시)
- **새 섹션**: Distributed Training (--distributed, scheduler, checkpoint resume)
- **sweep.py**: Overview 테이블 + sweep 섹션에 전 CLI flags 문서화
- **wandb**: Missing Dependencies에 추가, Model Outputs에 sweep/ 디렉토리 추가

### Ray Parallelization (2026-02-06)
- **Implemented**: File-level, partition-level, batch-level parallelism
- **Expected Speedup**: 3-5x on multi-core machines
- **Modules Updated**: `parser.py`, `engineering.py`, `usertag.py`
- **CLI Flag**: `--workers N` for parallel processing
- **Fallback**: Graceful degradation to sequential if Ray unavailable

### EDA Notebook Optimization (2026-02-16)
- **Usertag parsing**: `parse_usertag_series_parallel()` — Ray partition-level parallelism for 129.5M rows
- **Tag stats**: `compute_tag_stats_parallel()` — Ray map-reduce (Counter per partition → merge)
- **Groupby parallelism**: `parallel_groupbys()` — ThreadPoolExecutor for 6 independent groupbys (GIL-releasing C-level agg)
- **Weekday parsing**: `parse_weekday_from_timestamp()` — unique-value map (~30 dates vs 129.5M rows)
- **New modules**: `src/eda/parallel_ops.py`, `src/eda/__init__.py`
- **Cells updated**: notebook 01 cells 18, 47, 48, 61

### CVR Tower Pivot → Bid→Win→Click Reframe (2026-02-17)

**근거 (EDA 2.2.1):**
- Branding (train 기준 3개): CVR=0 → conversion prediction 불가
- Retargeting (train 기준 1개: 3358): CVR 27%, click→conv ~1초 → retargeting artifact
- 전체 WCTR 0.015%: CTR 자체가 극히 낮음

**Pivot: Bid→Win→Click 퍼널:**
- 129.5M 전체 bids 활용 (vs 30.6M impressions)
- Click 23K (vs Conversion 1,860, 12x 더 많음)
- Win PS AUC 0.91 (진단 완료) → DR + ESMM constraint으로 positivity 완화
- V(x) = debiased_pCTR × CPC (vs debiased_pCTR × pCVR × conv_value where pCVR 무의미)

**구조 변경:**
- `escm2_rtb.py` 삭제 (4-tower, CVR tower 포함)
- `esmm_wc.py` 신규 (2-tower: Win + CTR, ESMM constraint)
- `escm2_wc.py` 신규 (3-tower: Win + CTR + Imputation, DR/IPW)

### Strategic Conclusions (updated 2026-02-17)

1. **Win bias = auction structure** — +4.57% CTR overest. (LGB), positivity violation (overlap 47.8%)
2. **DR + ESMM constraint mitigates positivity** — Doubly robust + joint BCE + imputation tower (19.0M won samples, train)
3. **Advertiser stratification is essential** — Branding(CTR only) vs Retargeting(CVR critical)
4. **Win Tower dual purpose** — (a) CTR debiasing propensity, (b) bid shading win rate model
5. **CTR = 핵심 value signal** — CVR near-trivial → V(x) = debiased_pCTR × CPC

### Notebook 02 LGB 학습 속도 최적화 (2026-02-20)
- **Root Cause**: `LGBMClassifier` + numpy float64 → 메모리 폭발 (~32GB, swap 발생) → 2시간+ 소요
- **Fix**: `lgb.train()` native API + `lgb.Dataset(DataFrame)` + `categorical_feature=` native categorical
- **Cell 10 (Win PS)**: LabelEncoder 제거, numpy 변환 제거, val_df eval set 사용, `learning_rate=0.1, num_boost_round=100`
- **Cell 23 (Click PS)**: 동일 패턴, `is_unbalanced=True`, `num_boost_round=150`
- **결과**: ~32GB → ~8-10GB 메모리, 2시간+ → 수 분, AUC ~0.91 유지
- **캐시**: 기존 캐시 무효화 (파라미터 변경), 재생성 후 ~1초 재실행

### Target Encoding 추가 — LR + LGB 적용 (2026-02-21)
- **`src/features/engineering.py`**: `TargetEncodingResult` NamedTuple + `target_encode_kfold()` 함수 추가
  - K-fold OOF encoding (train leakage 방지), full-train stats for val
  - Bayesian smoothing: `(count * mean + m * global_mean) / (count + m)`
- **`src/features/__init__.py`**: `target_encode_kfold`, `TargetEncodingResult` export 추가
- **Notebook 02 cell-3**: Win TE 계산 (`target_encode_kfold(df, val_df, cat_features, 'win')`)
- **Notebook 02 cell-10**: LR → `num_features + te_win_features` (raw cat 제거), LGB → `all_features + te_win_features` (TE additive), `num_threads: 0`, cache `v2`
- **Notebook 02 cell-11**: TE color (`#9b59b6`) 추가
- **Notebook 02 new cell-20**: Click TE 계산 (`target_encode_kfold(df_winners, val_winners_full, cat_features, 'click')`)
- **Notebook 02 cell-24**: Click PS → `lgb_features_click = all_features + te_click_features`, val TE 첨부, `num_threads: 0`, cache `v2`
- **Notebook 02 cell-25**: Click importance chart TE color 추가

### Win PS CLI 학습 + 노트북 로드 전환 (2026-02-21)
- **`scripts/train.py`**: 모델 파일명 `baseline_*` → `lgb_*` 통일, `--include-lr/--no-include-lr` 플래그 추가 (전 task LR 학습)
- **`src/debiasing/win_propensity.py`**: `WinPropensityLoadedResult` NamedTuple + `load_win_propensity_models()` 함수 추가 (CLI 학습 모델 로드 → predict → metrics)
- **Notebook 02 Cell 10**: 인-노트북 학습 (~1시간+) → CLI 모델 로드 + 추론 (~수 초)로 전면 재작성, 캐시 v3→v4
- **Notebook 02 Cell 2**: 불필요 import 제거 (SGDClassifier, StandardScaler, fit_win_propensity 등)
- **Notebook 02 Cell 13/16/32**: `propensity_lr`/`auc_lr` None guard 추가 (LR 미학습 시 graceful degradation)
- **Notebook 03**: `baseline_ctr/win/ctr_all` → `lgb_ctr/win/ctr_all` 참조 갱신
- **Docs**: `scripts_tutorial.md` 출력 테이블 갱신, `01-prediction-models.md` 경로 갱신

### Click PS CLI 학습 + 노트북 로더 전환 (2026-02-22)
- **`scripts/train.py`**: `is_unbalanced` dead code 제거 (LightGBM이 인식 안 하는 misspelling으로 silently ignored됨; 정규 `is_unbalance`는 AUC 0.73→0.57 degradation 유발). 1줄 주석으로 교체
- **`src/debiasing/win_propensity.py`**: `ClickPropensityLoadedResult` NamedTuple + `load_click_propensity_models()` 함수 추가 (Win PS 로더와 동일 패턴)
- **`src/debiasing/__init__.py`**: 새 함수/타입 export 추가
- **Notebook 02 Cell 23**: 인-노트북 LGB fitting (67줄) → `load_click_propensity_models()` 로더 (~15줄)로 교체 (Win PS cell 10 패턴 통일)
- **Notebook 02 Cell 2**: 불필요 import 8개 제거 (`lgb`, `roc_auc_score`, `brier_score_loss`, `calibration_curve`, `LabelEncoder`, `WinPropensityResult`, `load_from_parquet`, `compute_dataset_stats`)
- **CLI 재학습**: `lgb_ctr.txt` AUC 0.7275 (train), 24 features, early stopping round 6
- **발견**: `is_unbalanced`는 LightGBM에서 인식 안 됨 (정규명: `is_unbalance` 's' 없음). 정규 파라미터 사용 시 1:1451 극단적 class weight → early stopping round 1에서 중단 (AUC 0.57). 노트북의 AUC 0.73은 `is_unbalanced` 무시 상태에서 달성된 것

### Notebook 02 마크다운 수치 동기화 + CTCVR→WCTR 전환 (2026-02-22)
- **수치 업데이트**: Funnel 45M→90.6M, 9.5M→19.0M, 6,630→13,260 (train set 기준)
- **CTR Bias**: +6.7%→+2.58%→+4.57% (LGB), LR -1.12% (모델 의존적)
- **Positivity**: overlap ~46%→56.8%→47.8%, ESS ~7%→13.3%→9.66%, won samples 15M→19.0M
- **Exchange bias 범위**: -10%~+18%→-8.4%~+10.3%
- **용어 전환**: CTCVR→WCTR (P(Win)×P(Click|Win)), 프로젝트 전체 일관성
- **Covariate shift 해석 수정**: Cell 9 — adexchange/slotformat(categorical) 제거, bid_floor_ratio(d=0.83)/slotprice(d=-0.69) 중심. Cell 22 — 모든 click shift Negligible로 전면 재작성
- **Advertiser 수 수정**: temporal split으로 train≈S2 기준 5개 광고주 명시
- **CLAUDE.md/PLAN.md 동기화**: positivity 수치 + WCTR 반영

### Notebook 02 Click PS Miscalibration Root Cause 분석 추가 (2026-02-23)
- **Cell 29 (code)**: 2.2c Root Cause Analysis — (1) `scale_pos_weight`=38.1 → mean pred 112x inflation (0.0774 vs 0.0007), (2) isotonic calibration 미적용 (ECE 0.3974→0.0000), (3) class sparsity (13,110/19.0M)
- **Cell 30 (markdown)**: Root Cause & Implications 요약표 + Isotonic이 ECE 0으로 만드는 이유 (calibration vs discrimination 구분) + Win PS vs Click PS 교훈
- **Cell 34 (markdown)**: Click-Stage Summary point 4 — 실제 수치 반영 (112x inflation, ECE 0→0.0000, AUC 불변)
- **Figure**: `02_bias_click_miscalibration_rootcause.png` — PS distribution, calibration before/after isotonic, per-bin positive count
- **핵심 발견**: Isotonic으로 ECE 완전 보정(0.0000) 가능하나, discrimination(AUC 0.68-0.73) 불변 → click-stage IPW 부적절 결론 불변

### Notebook 02 수치 통일 + IEB 연계 + Calibration/Sensitivity 추가 (2026-02-23)
- **수치 불일치 통일**: +2.58% → +4.57% (cells 0,17,29,33), 56.8% → 47.8%, 13.3% → 9.66% (코드 출력과 일치)
- **Spearman correlation 버그 수정**: Cell 30 — 독립 random sampling → paired index sampling (pairing 보존)
- **Win PS Calibration Curve 추가**: `02_bias_win_calibration.png` — Uniform + Quantile bins reliability diagram, ECE 0.0099 확인
- **Click PS Calibration Curve 추가**: `02_bias_click_calibration.png` — Click PS ECE 0.3974 miscalibration 시각화, Win PS vs Click PS 비교
- **Propensity Trimming Sensitivity**: 4개 clip_range별 IPW CTR bias + ESS ratio 비교표, bias 방향(+) 안정성 확인
- **IEB 연계**: Cell 31 summary table에 IEB 섹션 추가, Cell 33에 IEB bridge paragraph — `|naive_ctr - ipw_ctr| / ipw_ctr = 4.57%` = theoretical IEB lower bound
- **Cell 29 ECE 서술**: Click PS ECE 0.3974 → click-stage IPW 부적절 근거 명시, Win vs Click 비교표에 PS ECE 행 추가
- **Import 정리**: `propensity_sensitivity` 제거, `calibration_curve` 추가
- **CLAUDE.md/PLAN.md 동기화**: overlap ~57%→~48%, ESS ~13%→~10%

### Notebook 02 Win Selection Bias 메커니즘 설명 보강 (2026-02-22)
- **Cell 0 (Executive Summary)**: Key Finding #1에 메커니즘 추가 — "낙찰 샘플이 저경쟁 인벤토리에 편중 → naive CTR이 특정 옥션 구조에 편향되어 전체 입찰 모수에 대한 정확한 pCTR 추정 불가"
- **Cell 12 (Win Propensity Interpretation)**: 단순 결론 → "Why Win Selection Bias is the Primary Target" 섹션으로 교체. Flat-bid → 저경쟁 인벤토리 과대 대표 → 옥션 구조에 편향된 CTR → 전체 모수 교정 필요 인과 논리 체인
- **Cell 33 (Strategic Conclusions #1)**: 메커니즘 한 문장 추가 — flat-bid 편중 → 옥션 구조 편향 CTR → pCTR 부정확 → 입찰 최적화 성능 저하
- **논리 체인**: Cell 7-9(covariate shift, `bid_floor_ratio` d=0.83) → Cell 12(해석 + 메커니즘) → Cell 33(전략적 결론) 일관 연결

### LR Baseline 전 task 확장 (2026-02-21)
- **`scripts/train.py`**: `--include-lr` 조건을 `task == "win"` → 전 task(ctr, win, ctr_all)로 일반화, 모델명/경로를 `lr_{task}` 동적화
- **Notebook 03**: Cell 0(모델 목록 LR 언급), Cell 5(훈련 커맨드 LR 포함), Cell 6(LR 3종 result 로드), Cell 7(LGB-LR 페어 표시, `n_estimators` 조건부), Cell 15(ablation 테이블 LR 행 추가), Cell 17(summary 테이블 LR 3종 추가)
- **`docs/scripts_tutorial.md`**: `--include-lr` 설명에서 "(win task only)" 제거, Model Outputs 테이블에 `lr_ctr`/`lr_ctr_all` 행 추가

### Target Encoding K-fold 벡터화 (2026-02-21)
- **Root Cause**: `target_encode_kfold()` Python for-loop (lines 121-123, 138-140) — 5 folds × 15 features × ~72M rows = ~6.75B iterations → ~31분
- **Fix**: `pd.Series.groupby(level=0).agg(['sum','count'])` + `.map(smoothed).fillna(global_mean)` 벡터화
- **4곳 변경**: (1) fold stats, (2) OOF 적용, (3) full stats, (4) val 적용 — 모두 list comprehension/for-loop → pandas vectorized ops
- **Expected**: ~31분 → ~40초 (~50x speedup), TE 결과 값 동일

### Bug Fixes (2026-02-20)
- **LGB fold 병렬화 CPU 미활용**: `parallel_folds=-1` 시 `effective_workers=os.cpu_count()` 계산 → 실제 fold 수(5)보다 과대 추정 → `per_model_jobs=1`(단일 스레드 LGB). `min(len(folds), effective_workers)` 적용으로 5fold×3thread=~94% CPU 활용.

### Bug Fixes (2026-02-06)
- **Arrow overflow**: `split_temporal()` used `df.sort_values().reset_index()` on 129M rows → Arrow string offset overflow. Fixed with `np.argsort` + `iloc` to avoid full DataFrame sort.
- **Usertag vocab**: All usertag values null in unified parquet → vocab_size=1 misleading. Added null-check with warning before vocab building.
- **Merge safety**: Added row-count assertions to region and competition merges to catch silent row duplication from duplicate keys.
- **Region CTR**: `.replace(0, 1)` → `.clip(lower=1)` for Arrow-backed int64 compatibility.

---

## Pipeline Usage

### End-to-End Pipeline (Sequential)
```bash
# 1. Preprocess
python scripts/preprocess.py unify \
    --raw-dir data/ipinyou/raw/ipinyou \
    --output-dir data/ipinyou/prediction/unified \
    --seasons 2,3

# 2. Build features
python scripts/build_features.py build \
    --data-dir data/ipinyou/prediction/unified \
    --output-dir data/ipinyou/prediction/features

# 3. Train baseline
python scripts/train.py baseline \
    --data-dir data/ipinyou/prediction/features \
    --model-dir results/models \
    --task ctr

# 4. Evaluate
python scripts/train.py evaluate \
    --model-dir results/models \
    --data-dir data/ipinyou/prediction/features
```

### End-to-End Pipeline (Parallel with Ray)
```bash
# 1. Preprocess (file-level parallelism, 3-5x faster)
python scripts/preprocess.py unify \
    --raw-dir data/ipinyou/raw/ipinyou \
    --output-dir data/ipinyou/prediction/unified \
    --seasons 2,3 \
    --workers 8

# 2. Build features (partition-level parallelism)
python scripts/build_features.py build \
    --data-dir data/ipinyou/prediction/unified \
    --output-dir data/ipinyou/prediction/features \
    --workers 8

# 3. Train ESMM-WC (2-tower, ESMM constraint only)
python scripts/train.py esmmwc \
    --data-dir data/ipinyou/prediction/features \
    --model-dir results/models

# 4. Train ESCM2-WC with DR debiasing (primary model)
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features \
    --model-dir results/models \
    --debiasing dr

# 5. Evaluate
python scripts/train.py evaluate \
    --model-dir results/models \
    --data-dir data/ipinyou/prediction/features
```

---

## Next Steps

1. ~~Complete data pipeline (Task 1.1)~~ → ✅ Complete
2. ~~Implement ESCM2 and two-stage debiasing~~ → ✅ Complete
3. ~~Create Phase 1 notebooks~~ → ✅ Complete
4. ~~Add Ray parallelization to data pipeline~~ → ✅ Complete (2026-02-06)
5. ~~Run EDA and selection bias diagnosis on real data~~ → ✅ Complete (2026-02-09)
6. ~~ESMM-WC + ESCM²-WC 구현~~ → ✅ Complete (2026-02-17, Bid→Win→Click reframe)
7. ~~Implement ablation study~~ (Task 2.3) — Phase 1-18 완료. ESMM-WC Run J (WCTR 0.6905) AUC best overall, ESCM2-WC(DR) Run AW (WCTR 0.6882) ESCM2 best AUC + External PS, Run AL (WCTR 0.6843) calibration best (IEB 0.014). `docs/performance_tuning.md` Section 7, 10 업데이트
8. ~~통합 평가 모듈 생성~~ → ✅ Complete (2026-03-13~16, `src/metrics/` 모듈 + `notebooks/03_prediction.ipynb` 재구성. Section 5 고도화 + Section 8-12 → Appendix A-C 재구성, 중복 섹션 삭제)
9. ~~Win rate analysis~~ (Task 2.3.5) → ✅ Complete (2026-03-17, NB05+NB06 통합, KM CDF + parametric fit + exchange-conditional, `src/win_rate/` 모듈)
10. ~~Neural 모델 재학습 + Diagnostics Figure 생성~~ → ✅ Complete (2026-03-22)
    - [x] ESMM-WC 재학습 (Run J config) → `esmmwc_test_predictions.npz` (Test WCTR AUC 0.6527, IEB 0.075)
    - [x] ESCM²-WC(DR) 재학습 (Run AL config) → `escm2wc_dr_test_predictions.npz` (Test WCTR AUC 0.6851, IEB 0.073)
    - [x] ESCM²-WC(DR)+ExtPS 재학습 (Run AW config) → `escm2wc_dr_extps_test_predictions.npz` (Test WCTR AUC 0.6837, IEB 0.045)
    - [x] LR CTR_all 3-panel diagnostics figure 생성 → `03_lr_ctr_all_diagnostics.png` (AUC 0.7687)
    - [x] Neural model diagnostics figures 생성 → `04_esmmwc_diagnostics.png`, `04_escm2wc_dr_diagnostics.png`, `04_escm2wc_dr_extps_diagnostics.png`
    - [x] `docs/prediction_report.md` Figure 11-14 경로 확인 및 추가 완료
11. **Bid optimization** (Task 2.4) — V(x) = debiased_pCTR × CPC, Win Tower → bid shading
12. ~~Research design docs 업데이트~~ → ✅ Complete (Bid→Win→Click reframe 2026-02-17, EDA findings 반영 2026-02-19)
13. ~~Distributed training infrastructure~~ → ✅ Complete (2026-02-25, JAX SPMD + grain DataLoader + orbax checkpoint)
14. ~~Neural model smoke test~~ → ✅ Complete (2026-02-25, 3 models × 200K × 2 epochs, Flax 0.12 API fix)
15. ~~Synthetic data sanity test~~ → ✅ Complete (2026-02-26, 3 models × 50K × 50 epochs, Win/CTR AUC ≥ 0.8)

---

## Dependencies

```bash
# Core
jax>=0.4.20, flax>=0.8.0, optax>=0.1.7
lightgbm>=4.0.0, scikit-learn>=1.3.0

# CLI + Config
typer>=0.9.0
hydra-core>=1.3.0, omegaconf>=2.3.0

# Data
pandas>=2.0, pyarrow>=14.0

# Parallel Processing (optional, for --workers flag)
ray>=2.9.0

# Serving (Phase 3)
fastapi>=0.104.0, onnxruntime>=1.16.0, feast>=0.35.0

# Causal (Phase 3)
econml>=0.15.0, dowhy>=0.11.0
```
