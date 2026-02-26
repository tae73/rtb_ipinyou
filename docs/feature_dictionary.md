# Feature Dictionary

iPinYou RTB 데이터셋의 feature engineering 파이프라인과 모든 feature의 상세 정의.

---

## Pipeline Overview

```
Raw bz2 logs
    │
    ▼
Unified Parquet (bidid 기준 조인, win/click/conv 라벨링)
    │  src/data/parser.py, src/data/unifier.py
    │  scripts/preprocess.py
    │
    ▼
Feature Engineering (vectorized, 전체 데이터)
    │  src/features/engineering.py :: engineer_features()
    │
    ├─ Temporal Split (70/15/15)
    │  src/features/engineering.py :: split_temporal()
    │
    ├─ Training-Only Statistics 계산
    │  compute_region_stats(), compute_market_stats()
    │
    ├─ Statistics → Val/Test merge (결측 = global mean)
    │
    ├─ Usertag Vocabulary (training set 기준)
    │  src/features/usertag.py :: build_vocab()
    │
    ├─ Sparse Multi-hot Encoding (per-split .npz)
    │  src/features/usertag.py :: encode_multihot_sparse()
    │
    ▼
Train/Val/Test Parquet + feature_metadata.json
    scripts/build_features.py :: build()
```

---

## Feature Groups

### 1. Time Features (7 features)

**Why**: RTB 경쟁 강도와 사용자 행동이 시간대별로 크게 다름. 출퇴근 시간 경쟁 심화, 심야 시간 경쟁 완화 등.

**Source**: `src/features/engineering.py` :: `add_time_features()`

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `hour` | Int64 | 0-23 | 시간 (timestamp 문자열 8:10 자리 파싱) |
| `minute` | Int64 | 0-59 | 분 (timestamp 문자열 10:12 자리 파싱) |
| `weekday` | Int64 | 0-6 | 요일 (0=Mon, 6=Sun, `pd.to_datetime` + `dt.dayofweek`) |
| `is_weekend` | int | 0/1 | 주말 여부 (weekday >= 5) |
| `is_peak_hour` | int | 0/1 | 피크 시간 여부: 오전 7-9시, 저녁 17-20시 |
| `hour_sin` | float | [-1, 1] | Cyclical encoding: `sin(2*pi*hour/24)` |
| `hour_cos` | float | [-1, 1] | Cyclical encoding: `cos(2*pi*hour/24)` |

**Design Decisions**:
- **Cyclical encoding**: hour 23과 hour 0의 거리가 1이 되도록 sin/cos 변환. 트리 모델은 hour 정수 자체를 사용하지만, 신경망 입력에서는 cyclical encoding이 연속적 주기성을 보존.
- **is_peak_hour 정의**: 오전 러시(7-9), 저녁 러시(17-20). EDA 결과 이 시간대에 경쟁 강도(bid volume)와 win rate가 유의미하게 변동.
- **is_weekend 우선 권장**: EDA 확인 결과, 7-level weekday 대비 binary is_weekend이 Win/CTR 패턴 설명력 충분. 모델 입력에서 is_weekend 우선 사용 권장 (weekday는 보조).

---

### 2. Slot Features (4 features + 2 passthrough)

**Why**: 광고 슬롯의 크기와 형태가 CTR, 경쟁 수준, 시각적 노출도에 직접 영향. IAB 표준 크기별로 경쟁 패턴이 상이.

**Source**: `src/features/engineering.py` :: `add_slot_features()`

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `slot_area` | Int64 | >0 | 슬롯 면적 (`slotwidth * slotheight`) |
| `slot_area_log` | float | >0 | Log-transformed 면적 (`np.log1p(slot_area)`) |
| `slot_aspect_ratio` | float | >0 | 가로세로비 (`slotwidth / slotheight`, 0 방지) |
| `slot_size_group` | str | 7 categories | IAB 표준 크기 그룹 |
| `slotvisibility` | Int64 | passthrough | 슬롯 가시성 (above/below fold) |
| `slotformat` | Int64 | passthrough | 슬롯 포맷 (fixed, pop, etc.) |

**`slot_size_group` Categories** (IAB Standard Ad Units):

| Group | Sizes (w x h) | Description |
|-------|---------------|-------------|
| `leaderboard` | 728x90, 970x90, 970x250 | 페이지 상단 가로 배너 |
| `medium_rectangle` | 300x250, 336x280 | 본문 내 직사각형 (가장 보편적) |
| `skyscraper` | 300x600, 160x600, 120x600 | 사이드바 세로 배너 |
| `square` | 250x250, 200x200 | 정사각형 |
| `mobile` | 320x50, 320x100, 300x50 | 모바일 배너 |
| `banner` | 468x60 | 클래식 배너 |
| `other` | 기타 | 비표준 크기 |

**Design Decisions**:
- **IAB 매핑**: `_SLOT_SIZE_MAP` 딕셔너리로 표준 크기를 카테고리로 매핑. `_SLOT_SIZE_LOOKUP` DataFrame으로 pre-build하여 vectorized merge 수행 (130M row에서 Python tuple 생성 회피).
- **Log transform**: `slot_area`는 right-skewed 분포 → `log1p` 변환으로 정규분포에 근사.
- **Aspect ratio**: 0으로 나누기 방지를 위해 `slotheight.replace(0, 1)`.
- **slotvisibility 255**: Exchange 1에서만 출현하는 sentinel value (unknown visibility). `is_visibility_unknown` binary indicator 추가 권장 (Data Quality Features 참조).

---

### 3. Region Features (3+ features)

**Why**: 지역별로 경쟁 환경, 사용자 인구통계, 광고주 타겟팅 전략이 다름. 36개 지역 중 상위 17개(47.2%)가 전체 bid의 80% 차지 — 심한 geographic concentration.

**Source**: `src/features/engineering.py` :: `add_region_features()`, `compute_region_stats()`

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `region_freq` | int | >0 | Frequency encoding (해당 region의 전체 출현 횟수) |
| `region_group` | str | 4 categories | Quantile 기반 빈도 그룹 (`low`/`medium`/`high`/`very_high`) |
| `region_ctr` | float | [0, 1] | Historical CTR by region (training set 통계, optional) |

**Training-Only Statistics** (`compute_region_stats()`):
- Training set에서 `region`별 `n_bids`, `n_wins`, `n_clicks` 집계
- `region_win_rate = n_wins / n_bids`
- `region_ctr = n_clicks / n_wins.clip(lower=1)`
- Val/Test에 merge, 결측치는 전체 평균으로 대체

**Design Decisions**:
- **Frequency encoding**: Target encoding 대신 count 기반 인코딩으로 label leakage 방지.
- **Quantile grouping**: `pd.qcut(q=4)` 사용, 고유값 부족 시 `"medium"` fallback.
- **Merge safety**: `assert len(df) == n_before` row count 검증으로 duplicate key merge 방지.

---

### 4. Competition Features (3+ features)

**Why**: 입찰 경쟁 강도가 win probability의 핵심 결정 요인. `bid_floor_ratio`는 입찰 공격성, market price 통계는 해당 세그먼트의 경쟁 수준을 반영.

**Source**: `src/features/engineering.py` :: `add_competition_features()`, `compute_market_stats()`

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `bid_floor_ratio` | float | [0, 100] | 입찰 공격성 (`bidprice / slotprice`, clipped) |
| `market_price_avg` | float | >0 | 세그먼트별 평균 시장가 (training set, optional) |
| `market_price_std` | float | >=0 | 세그먼트별 시장가 표준편차 (training set, optional) |

**Training-Only Statistics** (`compute_market_stats()`):
- **Won bids만 사용** (`win == 1`): payprice는 낙찰된 입찰에서만 관측 가능
- Group key: `(adexchange, slot_size_group)` — 거래소와 슬롯 크기 조합별 시장가 분포
- Val/Test에 merge, 결측치는 전체 평균으로 대체

**Design Decisions**:
- **Clipping**: `bid_floor_ratio`를 [0, 100]으로 clip하여 `slotprice=0`에 의한 극단값 방지.
- **Group key 선택**: 초기 설계(`hour x exchange`)에서 `(adexchange, slot_size_group)`으로 변경. 시장가는 슬롯 크기에 더 강하게 의존하며, hour는 이미 time features에 포함.
- **Won bids only**: Lost bids에서는 market price가 관측 불가능하므로, 통계 계산에 won bids만 사용하는 것이 올바른 접근.

---

### 4.5 Data Quality Features (EDA-driven, 권장)

**Why**: EDA에서 발견된 데이터 품질 이슈를 feature로 인코딩하여 모델이 anomaly를 명시적으로 처리할 수 있게 함. 전처리 파이프라인(SP0 → SP1)에서 추가 권장.

**Source**: `src/features/engineering.py` (planned addition)

| Feature | Type | Definition | EDA Finding | 활용 |
|---------|------|-----------|-------------|------|
| `domain_quality_tier` | str | zero_win / zero_click / normal | 76 zero-win domains (7.16M bids, 5.5%), 741 zero-click domains | Win Tower: zero-win domain 제외/다운샘플링 |
| `is_popup` | int | slotformat == 5 | CTR 0.86% (standard 대비 11.4x) → misclick artifact | CTR Tower에서 제외 또는 별도 처리 |
| `is_visibility_unknown` | int | slotvisibility == 255 | Exchange 1에서만 출현하는 sentinel value | Binary indicator (exchange-conditional 불필요) |
| `is_floor_binding` | int | payprice ≈ slotprice (won bids) | 32.24% of won bids → P(Win) 분포에 mass point | Dual-regime shading (SP3): floor-bound vs competitive |
| `domain_group` | str | Top-N domains + `other` | Top 50 domains ≈ 60% traffic, 108K total | Domain concentration 처리, 희소 domain 통합 |

**Design Decisions**:
- **domain_quality_tier**: 3-level categorical. Win Tower 학습 시 zero-win domain은 win=0 기회가 없어 bias 유발 가능 → 제외 또는 다운샘플링 권장.
- **is_popup**: Binary indicator. Pop-up (slotformat==5)의 CTR 0.86%는 misclick artifact로 추정. CTR Tower에서 제외하거나 별도 모델 적용 권장.
- **is_visibility_unknown**: Exchange 1에서만 255 출현 → 단순 binary indicator로 충분 (exchange-conditional 불필요).
- **is_floor_binding**: Won bids에서만 계산 가능. SP3 bid shading에서 floor-bound regime (shade to floor) vs competitive regime (standard shading) 구분에 활용.
- **domain_group**: Top-50 또는 Top-100 domains를 개별 유지, 나머지를 `other`로 통합. Domain concentration이 심하므로(top 238 = 80%) 효과적인 차원 축소.

---

### 5. Usertag Features (sparse multi-hot)

**Why**: 사용자 관심사 taxonomy가 CTR/CVR 예측의 핵심 신호. iPinYou usertag는 콤마 구분 정수 ID로, 사용자의 관심 카테고리를 나타냄.

**Source**: `src/features/usertag.py`, `scripts/build_features.py`

| Feature | Type | Shape | Description |
|---------|------|-------|-------------|
| `tag_0` ... `tag_N` | float32 | (n_samples, vocab.n_tags) | Multi-hot sparse encoding (.npz) |

**Vocabulary** (`build_vocab()`):
- **Top-100**: 가장 빈번한 상위 100개 태그 선택
- **min_count=10**: 최소 10회 이상 출현한 태그만 포함
- **Index 0**: Unknown/OOV 태그용 (reserve)
- **Training set 기준**: Vocabulary는 training set에서만 구축하여 data leakage 방지

**Encoding** (`encode_multihot_sparse()`):
- **Sparse CSR Matrix**: `scipy.sparse.csr_matrix` 형태로 저장 (메모리 효율)
- **Per-split 저장**: `{train,val,test}_usertag.npz`로 분할별 별도 저장
- **Coverage**: top-100 vocab가 전체 tag 출현의 대부분을 커버 (coverage 메트릭 출력)

**Encoding Modes** (`scripts/build_features.py --usertag-encoding`):

| Mode | Output | Description |
|------|--------|-------------|
| `summary` (default) | vocab JSON만 | Vocabulary 구축만, DataFrame에 feature 추가 안 함 |
| `sparse` | per-split .npz | Multi-hot sparse CSR matrix |
| `hashing` | per-split .npz | Feature hashing (vocab 불필요, 차원 축소) |

**Leakage Warning**:

> `n_tags` (사용자 태그 수), `has_tags` (태그 존재 여부) feature는 **win label과 상관관계**가 있어 제거됨 (2026-02-12).
>
> **원인**: Bid log의 usertag 필드는 null이 가능하나, impression/click/conversion log에서 usertag가 항상 존재 → `has_tags=1`이 `win=1`과 강하게 상관 → win propensity 모델에 spurious signal 제공.
>
> **영향**: Win propensity AUC가 0.993 → ~0.91로 하락 (leakage 제거 후 clean AUC).

**Usertag 전면 제외 정책 (2026-02-19)**:

> 현재 모든 모델(ESMM-WC, ESCM²-WC)에서 usertag를 **사용하지 않음**.
> 이유: Shared embedding 구조에서 usertag가 Win Tower에 leakage를 발생시킴 (bid log usertag null 패턴이 win label과 상관).
> Tower-specific feature selection이 구현되지 않은 상태에서는 전면 제외가 안전.
>
> `scripts/build_features.py`의 `--no-usertag` 기본값이 `True`로 설정됨.
> Usertag 모듈 (`src/features/usertag.py`)은 라이브러리 코드로 유지 — 향후 tower-specific feature selection 구현 시 재활성화 가능.

**Tower-specific Usage (향후 계획)**:
- **CTR Tower**: Usertag multi-hot 안전하게 사용 가능 (CTR 예측에 유효한 signal)
- **Win Tower**: Usertag 사용 **금지** — bid log usertag null 패턴이 win label과 상관 → leakage 원인
- **현재 구조**: Shared embedding + shared input → tower-specific feature selection 미구현 → 전면 제외

---

### 6. High-Cardinality Features (6 features)

**Why**: `domain` (108K unique) and `creative` (advertiser-specific creative IDs) are high-cardinality categorical columns that contain publisher and ad creative signals. Direct embedding would create oversized tables (108K vocab). Hash encoding provides fixed-size buckets with unseen value handling; frequency encoding captures popularity as a numerical signal.

**Source**: `src/features/engineering.py` :: `hash_encode()`, `add_high_card_features()`

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `domain_hash` | int32 | [0, 10000] | Publisher domain, hash-encoded. `hashlib.md5(domain) % 10000 + 1`, 0=NaN/unknown |
| `creative_hash` | int32 | [0, 5000] | Creative ID, hash-encoded. `hashlib.md5(creative) % 5000 + 1`, 0=NaN/unknown |
| `domain_freq` | int64 | >=0 | Domain frequency (bid count). `value_counts()` mapping |
| `domain_freq_log` | float32 | >=0 | Log-transformed domain frequency. `log(1 + domain_freq)` — power law correction |
| `creative_freq` | int64 | >=0 | Creative frequency (bid count). `value_counts()` mapping |
| `creative_freq_log` | float32 | >=0 | Log-transformed creative frequency. `log(1 + creative_freq)` — power law correction |

**Hash Encoding Design**:
- **MD5 hash**: `hashlib.md5` is deterministic across Python versions and runs (unlike `hash()`)
- **Unique-value mapping**: Hash computed only for unique values (108K), then vectorized `.map()` over 129M rows
- **Bucket 0 reserved**: NaN/unknown values → bucket 0. Hashed values → [1, n_buckets]
- **Vocab size**: `n_buckets + 1` (0 through n_buckets). `feature_dims` via `max() + 2` = `n_buckets + 2` (1 unused slot, harmless)
- **Collision**: 108K/10K ≈ 10.8 avg collision per bucket for domain. Collision information loss is partially compensated by embedding learning

**Frequency Encoding Design**:
- `freq`: Raw count from `value_counts()` — useful for tree models (LGB)
- `freq_log`: `np.log1p(freq)` — corrects power law skew for neural models (Z-score normalized)
- Both provided: tree models benefit from raw counts, neural models from log-transformed

**Configuration** (`configs/features/default.yaml`):
```yaml
hash_encoding:
  domain_buckets: 10000
  creative_buckets: 5000
  hash_seed: 42
```

CLI override: `--domain-buckets 10000 --creative-buckets 5000`

---

## Passthrough / ID Features

Feature engineering pipeline에서 변환하지 않고 그대로 전달되는 컬럼:

| Feature | Type | Description | 용도 |
|---------|------|-------------|------|
| `adexchange` | Int64 | 광고 거래소 ID | 카테고리 feature, market stats group key |
| `domain` | str | 퍼블리셔 도메인 (해시) | 카테고리 feature |
| `advertiser` | Int64 | 광고주 ID | Stratification key (retargeting/branding) |
| `region` | Int64 | 지역 코드 | region_freq 계산 원본 |
| `city` | Int64 | 도시 코드 | 카테고리 feature |
| `slotvisibility` | Int64 | 슬롯 가시성 | Above/below fold |
| `slotformat` | Int64 | 슬롯 포맷 | Fixed/pop 등 |
| `bidprice` | Int64 | 입찰가 (CPM) | Competition feature 원본 |
| `slotprice` | Int64 | Floor price (CPM) | Competition feature 원본 |

---

## Label Columns

| Column | Type | Description | 관측 조건 |
|--------|------|-------------|-----------|
| `win` | int | 낙찰 여부 | 모든 bid에서 관측 |
| `click` | int | 클릭 여부 | win=1인 bid에서만 의미 |
| `conversion` | int | 전환 여부 | click=1인 bid에서만 의미 (retargeting만) |
| `payprice` | Int64 | 실제 지불가 (CPM) | win=1인 bid에서만 관측 |

---

## Data Splitting

**Method**: Temporal split (시간순 정렬 후 비율 분할)

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | 모델 학습 + 통계 계산 |
| Val | 15% | 하이퍼파라미터 튜닝 |
| Test | 15% | 최종 평가 |

**Implementation**: `split_temporal()` in `src/features/engineering.py`
- `np.argsort(timestamp)` + `iloc` 사용 (Arrow string offset overflow 방지, 129M rows)
- `string[pyarrow]` → `large_string[pyarrow]` 64-bit offset 변환
- 시간순 정렬이므로 미래 정보 누출(data leakage) 방지

---

## Training-Only Statistics Summary

| Statistic | Function | Group Key | Merge Target | Missing Strategy |
|-----------|----------|-----------|--------------|------------------|
| `region_stats` | `compute_region_stats()` | `region` | val, test | global mean |
| `market_stats` | `compute_market_stats()` | `(adexchange, slot_size_group)` | val, test | global mean |

두 통계 모두 **training set에서만 계산**하여 val/test에 left-merge. 이는 실제 배포 환경에서 미래 데이터를 사용하지 않는 것과 동일.

---

## File Outputs

```
data/ipinyou/prediction/features/
├── train.parquet              # Training set (engineered features)
├── val.parquet                # Validation set
├── test.parquet               # Test set
├── feature_metadata.json      # Feature names, types, groups
├── train_usertag.npz          # Sparse usertag matrix (optional)
├── val_usertag.npz
├── test_usertag.npz
├── vocab/
│   └── usertag_vocab.json     # Tag-to-index mapping, counts
└── stats/
    ├── region_stats.parquet   # Training region statistics
    └── market_stats.parquet   # Training market price statistics
```
