# 데이터 셋업 가이드 (Data Setup)

이 프로젝트는 iPinYou RTB 데이터셋과 그로부터 생성되는 파생 산출물을 모두 `data/ipinyou/` 아래에
둔다. `data/`는 **git-ignored**(`.gitignore`의 `data/`, `*.parquet`, `*.npz`, `*.bz2`, `*.tar`)이므로
저장소에는 데이터가 들어 있지 않다. 이 문서는 **어떤 데이터가 어떤 디렉토리에 어떤 형태로 필요하고,
어떻게 다시 받는지**를 설명한다.

데이터는 두 부류다.

- **RAW** — iPinYou 원본 로그(bz2). 약 5.9GB. 재취득은 make-ipinyou-data 재다운로드 또는 외장/서버
  보관본 복사.
- **파생(Derived)** — 파이프라인(`scripts/`)이 RAW로부터 생성. 재실행으로 재생성 가능하며, 원격
  서버에도 백업되어 있다.

> **현재 상태(2026-06)**: 로컬 디스크 절약을 위해 **RAW는 외장 드라이브
> `/Volumes/samsungj2/data/rtb_ipinyou/`로 이동**했고, **파생 데이터는 로컬에서 삭제**했다. 아래
> "되받기" 절차로 언제든 복원할 수 있다.

데이터 취득·전처리·EDA의 상세 배경은 [docs/research_design/00-data-preparation.md](research_design/00-data-preparation.md)
참고. 스크립트 사용법은 [docs/scripts_tutorial.md](scripts_tutorial.md) 참고.

---

## 디렉토리 레이아웃 · 형태 · 출처

모든 경로는 CLI(`--raw-dir`, `--data-dir`, `--output-dir`) 또는 `configs/data/ipinyou.yaml` /
`src/config.py`의 `DataConfig`로 변경 가능하다. 아래는 기본(default) 경로 기준이다.

### RAW

| 디렉토리 | 형태 | 용량(약) | 출처 |
|---|---|---|---|
| `data/ipinyou/raw/ipinyou/{training,testing}{1st,2nd,3rd}/` | bid/imp/clk/conv 로그 `*.txt.bz2` (Tab-separated) | 5.9G | make-ipinyou-data (GitHub) 재다운로드 / 외장·서버 보관본 |

> Season 1(2013.03, usertag 없음) / Season 2(2013.06) / Season 3(2013.10). 파이프라인은 기본 S2+S3 사용.

### 파생 (Derived)

| 디렉토리 | 형태 | 용량(약) | 생성 스크립트 | 서버 백업 |
|---|---|---|---|---|
| `data/ipinyou/prediction/unified/` | Parquet (season/day Hive 파티션, bidid 조인 + win/click/conv 라벨) | 10G | `scripts/preprocess.py unify` | ✓ |
| `data/ipinyou/prediction/features/` | Parquet (`train`/`val`/`test`) + `stats/`(region·market) + `vocab/`(usertag) | 5.0G | `scripts/build_features.py build` | ✓ |
| `data/ipinyou/prediction/synthetic/` | Parquet (seed=42 합성, 모델 sanity check) | 5.9M | `scripts/generate_synthetic.py generate` | ✓ |
| `data/ipinyou/processed/` | (현재 비어있음, 미사용) | 0 | — | — |

> 루트의 `features.tar`(~117M)는 코드에서 참조되지 않는 orphaned 아티팩트로, 이 가이드의 데이터
> 파이프라인과 무관하다.

---

## 되받기 (Restore)

### RAW 복원

1. **외장 보관본 복사** (가장 빠름):
   ```bash
   rsync -a /Volumes/samsungj2/data/rtb_ipinyou/ipinyou/raw/ data/ipinyou/raw/
   ```
2. **외장에 둔 채 사용** (복사 없이 경로만 지정):
   ```bash
   python scripts/preprocess.py unify \
     --raw-dir /Volumes/samsungj2/data/rtb_ipinyou/ipinyou/raw/ipinyou \
     --output-dir data/ipinyou/prediction/unified --seasons 2,3
   ```
3. **서버에서 pull** (`rtb` 서버에 백업):
   ```bash
   rsync -avz -e 'ssh -p 5040' \
     mail-agent@3.38.195.121:'project/rtb_ipinyou/data/ipinyou/raw/' \
     data/ipinyou/raw/
   ```
4. **처음부터**: make-ipinyou-data(GitHub) 절차로 원본 iPinYou 데이터셋 재구성
   (상세: `docs/research_design/00-data-preparation.md` Part A).

### 파생 데이터 복원

**옵션 A — 원격 서버에서 pull** (전체 백업되어 있어 빠름):
```bash
rsync -avz -e 'ssh -p 5040' \
  mail-agent@3.38.195.121:'project/rtb_ipinyou/data/ipinyou/prediction/' \
  data/ipinyou/prediction/
```

**옵션 B — 파이프라인으로 재생성** (`CLAUDE.md` Pipeline Usage 1→2):
```bash
python scripts/preprocess.py unify \
  --raw-dir data/ipinyou/raw/ipinyou \
  --output-dir data/ipinyou/prediction/unified --seasons 2,3
python scripts/build_features.py build \
  --data-dir data/ipinyou/prediction/unified \
  --output-dir data/ipinyou/prediction/features
```

> macOS 기본 `rsync`는 `openrsync`(2.6.9 호환)라 `--info=progress2` 같은 rsync 3.x 옵션을
> 지원하지 않는다. `-a` 또는 `-av`를 사용한다.

---

## 디스크 사용 요약

| 부류 | 로컬 보관 위치 | 용량(약) |
|---|---|---|
| RAW | 외장 `/Volumes/samsungj2/data/rtb_ipinyou/` (+ `rtb` 서버) | ~5.9G |
| 파생 | 삭제됨 (서버 백업 또는 재생성) | ~15G |

RAW를 외장으로 옮기고 파생을 삭제하면 로컬에서 약 **21GB**를 확보한다.
