# RTB iPinYou: Selection Bias Debiasing + First-price Bid Optimization

## Overview

This project addresses **Win Selection Bias** in Real-Time Bidding (RTB) through the **Bid→Win→Click** funnel:

- **Win Bias**: Only won bids become impressions → P(Win|X, bid)
- **Debiased CTR**: P(Click|X, Win) corrected via DR/IPW using Win Tower propensity

We implement a debiasing ablation:
1. **Biased Baseline** (LightGBM): CTR on winners only
2. **ESMM-WC**: 2-tower (Win + CTR), ESMM joint constraint only
3. **ESCM²-WC(DR)**: 3-tower (Win + CTR + Imputation), Doubly Robust debiasing

## Project Structure

```
rtb_ipinyou/
├── src/
│   ├── data/           # Data parsing and unification
│   ├── features/       # Feature engineering, usertag encoding
│   ├── models/         # base.py (shared layers), esmm_wc.py, escm2_wc.py
│   ├── debiasing/      # Win propensity estimation, diagnostics
│   ├── config.py       # NamedTuple configurations
│   └── config_utils.py # Hydra Compose API bridge (YAML → NamedTuple)
├── scripts/            # CLI entry points (preprocess, build_features, train)
├── configs/            # Hydra YAML config groups
├── notebooks/          # Analysis notebooks
├── docs/               # Research design docs, tutorials
├── mlops/              # Tracking, serving, monitoring
└── experiments/        # Ablation studies
```

## Key Components

### Data Pipeline (`src/data/`)
- `parser.py`: Parse iPinYou bid/imp/clk/conv logs (bz2 format)
- `unifier.py`: Join logs by bidid, create win/click/conversion labels

### Models (`src/models/`)
- `base.py`: Shared layers (MLP, EmbeddingLayer, FeatureInteraction, loss utilities)
- `esmm_wc.py`: ESMM-WC 2-tower (Win + CTR, ESMM constraint)
- `escm2_wc.py`: ESCM²-WC 3-tower (Win + CTR + Imputation, DR/IPW debiasing)

### Debiasing (`src/debiasing/`)
- `win_propensity.py`: LightGBM + Isotonic calibration (external PS option)
- `diagnostics.py`: Positivity diagnostics, covariate shift analysis

## Quick Start

```python
from src.data.parser import parse_season_logs
from src.data.unifier import create_unified_dataset

# Parse data
data_dir = "data/ipinyou/raw/ipinyou"
df = create_unified_dataset(data_dir, season="2nd")

# Check stats
print(f"Win rate: {df['win'].mean():.4%}")
print(f"CTR: {df['click'].sum() / df['win'].sum():.4%}")
```

## Config Management

All scripts support optional Hydra config via `--config-dir` and `--overrides`:

```bash
# 기존 방식 (변경 없음)
python scripts/train.py escm2wc --data-dir ... --model-dir ...

# YAML config 사용
python scripts/train.py escm2wc --data-dir ... --model-dir ... \
    --config-dir configs

# Override로 모델/파라미터 변경
python scripts/train.py escm2wc --data-dir ... --model-dir ... \
    --config-dir configs --overrides "model=escm2wc_ipw,training.batch_size=2048"
```

See [docs/scripts_tutorial.md](docs/scripts_tutorial.md) for full config reference.

## Installation

```bash
pip install -e ".[dev]"
```

## Dataset

iPinYou RTB Dataset (2013):
- Season 1: 2013.03 (no usertag)
- Season 2: 2013.06 (with usertag)
- Season 3: 2013.10 (with usertag)

데이터 디렉토리 레이아웃·형태·취득/복원 방법은 [docs/data_setup.md](docs/data_setup.md) 참고
(`data/`는 git-ignored이며 RAW는 외장 드라이브에 보관).

## References

- Wang et al., "ESCM²: Entire Space Counterfactual Multi-Task Model for Post-Click Conversion Rate Estimation" (SIGIR 2022)
- Ma et al., "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate" (SIGIR 2018)
- Zhang et al., "Real-Time Bidding Benchmarking with iPinYou Dataset"
