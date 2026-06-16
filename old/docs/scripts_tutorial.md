# RTB iPinYou Scripts Tutorial

This guide explains how to use the CLI scripts for data processing and model training.

## Overview

The project uses Typer-based CLI scripts with optional Hydra config management for reproducible ML pipelines:

| Script | Purpose |
|--------|---------|
| `preprocess.py` | Parse raw bz2 logs → Unified Parquet |
| `build_features.py` | Feature engineering + Train/Val/Test split |
| `train.py` | Model training (Baseline CTR/Win/CTR_all, ESMM-WC, ESCM²-WC) and evaluation |

## Quick Start

```bash
# 1. Preprocess raw data
python scripts/preprocess.py unify \
    --raw-dir data/ipinyou/raw/ipinyou \
    --output-dir data/ipinyou/prediction/unified \
    --seasons 2,3

# 2. Build features (usertag excluded by default — leakage prevention)
python scripts/build_features.py build \
    --data-dir data/ipinyou/prediction/unified \
    --output-dir data/ipinyou/prediction/features

# 3. Train baseline model (biased CTR on winners only)
python scripts/train.py baseline \
    --data-dir data/ipinyou/prediction/features \
    --model-dir results/models \
    --task ctr

# 4. Train ESMM-WC (2-tower, ESMM constraint)
python scripts/train.py esmmwc \
    --data-dir data/ipinyou/prediction/features \
    --model-dir results/models

# 5. Train ESCM2-WC with DR debiasing (primary model)
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features \
    --model-dir results/models \
    --debiasing dr

# 6. Evaluate models
python scripts/train.py evaluate \
    --model-dir results/models \
    --data-dir data/ipinyou/prediction/features
```

## Parallel Processing with Ray

All data processing scripts support parallel execution via Ray for 3-5x speedup on multi-core machines.

### Quick Start (Parallel)

```bash
# Preprocess with 8 workers
python scripts/preprocess.py unify \
    --raw-dir data/ipinyou/raw/ipinyou \
    --output-dir data/ipinyou/prediction/unified \
    --seasons 2,3 \
    --workers 8

# Build features with 8 workers
python scripts/build_features.py build \
    --data-dir data/ipinyou/prediction/unified \
    --output-dir data/ipinyou/prediction/features \
    --workers 8
```

### Installation

```bash
pip install ray
```

### How It Works

| Component | Parallelization Strategy | Module |
|-----------|--------------------------|--------|
| File Parsing | File-level parallelism (23 files per season) | `src/data/parser.py` |
| Feature Engineering | DataFrame partition parallelism | `src/features/engineering.py` |
| Usertag Encoding | Batch parallelism (10K samples/batch) | `src/features/usertag.py` |
| Vocabulary Building | Map-reduce (count per partition) | `src/features/usertag.py` |

### Fallback Behavior

- If Ray is not installed, scripts fall back to sequential processing
- If `--workers` is not specified, scripts use sequential processing
- No code changes needed for sequential execution

## Config Management with Hydra

All scripts support optional Hydra config loading via `--config-dir` and `--overrides`. This allows centralized YAML-based configuration while preserving full CLI backward compatibility.

### Architecture

- **Hydra Compose API** (not `@hydra.main()`) — compatible with Typer CLI
- **`src/config_utils.py`** is the sole hydra/omegaconf import point
- **`src/` modules** receive NamedTuple configs only (no OmegaConf dependency)
- **Precedence**: CLI args > `--overrides` > YAML defaults > NamedTuple defaults

### Config Directory Structure

```
configs/
├── config.yaml              # Master defaults list
├── data/
│   └── ipinyou.yaml         # DataConfig
├── features/
│   └── default.yaml         # FeatureConfig + split settings
├── training/
│   └── default.yaml         # TrainingConfig + OptimizerConfig
├── model/
│   ├── esmmwc.yaml          # ESMM-WC (2-tower)
│   ├── escm2wc_dr.yaml      # ESCM2-WC DR (primary)
│   ├── escm2wc_ipw.yaml     # ESCM2-WC IPW
│   └── baseline_lgb.yaml    # LightGBM baseline
├── debiasing/
│   └── win_propensity.yaml  # WinPropensityConfig
├── serving/
│   └── default.yaml         # ServingConfig
├── bidding/
│   └── default.yaml         # BiddingConfig
└── ray/
    └── default.yaml         # RayConfig
```

### Usage Examples

```bash
# Existing CLI (unchanged, no Hydra needed)
python scripts/train.py escm2wc --data-dir ... --model-dir ... --debiasing dr

# Load defaults from YAML config
python scripts/train.py escm2wc --data-dir ... --model-dir ... \
    --config-dir configs

# Switch model config group via override
python scripts/train.py escm2wc --data-dir ... --model-dir ... \
    --config-dir configs --overrides "model=escm2wc_ipw"

# Override individual values
python scripts/train.py escm2wc --data-dir ... --model-dir ... \
    --config-dir configs --overrides "training.batch_size=2048,model.dropout=0.5"

# Combine group switch + value override
python scripts/train.py escm2wc --data-dir ... --model-dir ... \
    --config-dir configs --overrides "model=escm2wc_ipw,model.cfr_lambda=0.2"
```

### Programmatic Usage

```python
from src.config_utils import load_config, build_escm2wc_config

# Load with defaults
cfg = load_config()

# Load with overrides
cfg = load_config(overrides=["model=escm2wc_ipw", "training.batch_size=2048"])

# Convert to NamedTuple (for src/ modules)
feature_dims = {"weekday": 8, "hour": 25, "bidprice": -1}
config = build_escm2wc_config(cfg, feature_dims)
```

### Missing Dependencies

```bash
pip install hydra-core omegaconf  # Required for --config-dir / --overrides
```

## Scripts Reference

### preprocess.py

Parse and unify iPinYou RTB log files.

#### Commands

**`unify`**: Parse raw logs and create unified Parquet dataset

```bash
python scripts/preprocess.py unify \
    --raw-dir <RAW_DATA_PATH> \
    --output-dir <OUTPUT_PATH> \
    --seasons 2,3 \
    --max-rows <N>  # Optional: for testing
```

| Option | Description | Default |
|--------|-------------|---------|
| `--raw-dir` | Raw data directory (contains training2nd, training3rd) | Required |
| `--output-dir` | Output directory for Parquet files | Required |
| `--seasons` | Seasons to process (1, 2, 3) | 2, 3 |
| `--max-rows` | Max rows per file (for testing) | None |
| `--workers` | Number of parallel workers (Ray) | None (sequential) |
| `--quiet` | Suppress output | False |
| `--config-dir` | Hydra config directory (e.g., `configs`) | None |
| `--overrides` / `-O` | Hydra overrides, comma-separated | None |

**`validate`**: Validate unified dataset

```bash
python scripts/preprocess.py validate \
    --data-dir data/ipinyou/prediction/unified
```

**`stats`**: Show dataset statistics

```bash
python scripts/preprocess.py stats \
    --data-dir data/ipinyou/prediction/unified \
    --by-season  # Optional: breakdown by season
    --by-day     # Optional: breakdown by day
```

**`sample`**: Sample rows for inspection

```bash
python scripts/preprocess.py sample \
    --data-dir data/ipinyou/prediction/unified \
    --n 100 \
    --output sample.csv
```

### build_features.py

Feature engineering and data splitting.

#### Commands

**`build`**: Create features and train/val/test splits

```bash
python scripts/build_features.py build \
    --data-dir data/ipinyou/prediction/unified \
    --output-dir data/ipinyou/prediction/features \
    --split-method temporal \
    --train-ratio 0.7 \
    --val-ratio 0.15
```

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir` | Unified data directory | Required |
| `--output-dir` | Output directory | Required |
| `--split-method` | 'temporal' or 'by_days' | temporal |
| `--train-ratio` | Training set ratio | 0.7 |
| `--val-ratio` | Validation set ratio | 0.15 |
| `--train-days` | Training days (for by_days split) | None |
| `--usertag-encoding` | 'summary', 'sparse', 'hashing' (leakage warning) | summary |
| `--usertag-top-n` | Top N usertags in vocabulary | 100 |
| `--usertag-min-count` | Minimum occurrence count for a usertag | 10 |
| `--no-usertag` | Skip usertag encoding (default: True) | True |
| `--workers` | Number of parallel workers (Ray) | None (sequential) |
| `--config-dir` | Hydra config directory (e.g., `configs`) | None |
| `--overrides` / `-O` | Hydra overrides, comma-separated | None |

**`info`**: Show feature dataset information

```bash
python scripts/build_features.py info \
    --data-dir data/ipinyou/prediction/features
```

**`vocab`**: Build usertag vocabulary

```bash
python scripts/build_features.py vocab \
    --data-dir data/ipinyou/prediction/unified \
    --output-dir data/ipinyou/prediction/features/vocab \
    --top-n 100 \
    --min-count 10
```

**`stats`**: Show detailed split statistics

```bash
python scripts/build_features.py stats \
    --data-dir data/ipinyou/prediction/features \
    --split train  # train, val, or test
```

### train.py

Model training and evaluation for Bid→Win→Click pipeline.

#### Commands

**`baseline`**: Train LightGBM baseline

```bash
python scripts/train.py baseline \
    --data-dir data/ipinyou/prediction/features \
    --model-dir results/models \
    --task ctr \
    --n-estimators 300 \
    --learning-rate 0.1 \
    --max-depth 6
```

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir` | Feature data directory | Required |
| `--model-dir` | Model output directory | Required |
| `--task` | 'ctr', 'win', or 'ctr_all' | ctr |
| `--n-estimators` | Number of trees | 300 |
| `--learning-rate` | Learning rate | 0.1 |
| `--max-depth` | Tree max depth | 6 |
| `--include-lr/--no-include-lr` | Also train LR baseline | True |
| `--config-dir` | Hydra config directory (e.g., `configs`) | None |
| `--overrides` / `-O` | Hydra overrides, comma-separated | None |

**Baseline task options:**

| Task | Data | Target | Description |
|------|------|--------|-------------|
| `ctr` | win=1 only | click | Biased CTR baseline — P(Click\|Win=1, X) |
| `win` | ALL bids | win | Win Tower baseline — P(Win\|X, bid) |
| `ctr_all` | ALL bids | click | Population-level WCTR baseline — P(Click_bid\|X) |

**`esmmwc`**: Train ESMM-WC model (2-tower, ESMM constraint)

```bash
python scripts/train.py esmmwc \
    --data-dir data/ipinyou/prediction/features \
    --model-dir results/models \
    --epochs 50 \
    --batch-size 4096
```

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs` | Training epochs | 50 |
| `--batch-size`, `-b` | Batch size | 4096 |
| `--learning-rate`, `-lr` | Learning rate | 0.001 |
| `--embedding-dim` | Embedding dimension | 16 |
| `--hidden-dims` | CTR tower MLP dimensions | 128,64 |
| `--win-hidden-dims` | Win tower MLP dimensions | 64,32 |
| `--dropout` | Dropout rate | 0.3 |
| `--win-weight` | Win loss weight | 1.0 |
| `--ctr-weight` | CTR loss weight (0.0 per Ma et al. 2018 — no direct supervision) | 0.0 |
| `--joint-weight` | Joint (ESMM) loss weight | 1.0 |
| `--weight-decay` | AdamW weight decay (0 = vanilla Adam) | 1e-5 |
| `--scheduler` | LR scheduler: constant, cosine, linear | constant |
| `--warmup-steps` | LR warmup steps | 0 |
| `--gradient-clip` | Gradient clipping norm (0 = disabled) | 0.0 |
| `--eval-every` | Val AUC/ECE/IEB metrics every N epochs | 5 |
| `--max-samples` | Limit training samples (for smoke testing) | None (all) |
| `--quiet`, `-q` | Suppress progress output | False |
| `--distributed` | Enable distributed SPMD training | False |
| `--num-devices` | Number of devices (None = auto-detect) | None |
| `--resume-from` | Checkpoint path to resume from | None |
| `--use-wandb` | Enable W&B logging | False |
| `--wandb-project` | W&B project name | rtb-ipinyou |
| `--wandb-run-name` | W&B run name (auto-generated if None) | None |
| `--config-dir` | Hydra config directory (e.g., `configs`) | None |
| `--overrides` / `-O` | Hydra overrides, comma-separated | None |

**`escm2wc`**: Train ESCM2-WC model (3-tower, DR/IPW debiasing)

```bash
python scripts/train.py escm2wc \
    --data-dir data/ipinyou/prediction/features \
    --model-dir results/models \
    --debiasing dr \
    --epochs 50
```

| Option | Description | Default |
|--------|-------------|---------|
| `--debiasing` | Debiasing method: ipw or dr | dr |
| `--epochs` | Training epochs | 50 |
| `--batch-size`, `-b` | Batch size | 4096 |
| `--learning-rate`, `-lr` | Learning rate | 0.001 |
| `--embedding-dim` | Embedding dimension | 16 |
| `--hidden-dims` | CTR tower MLP dimensions | 128,64 |
| `--win-hidden-dims` | Win tower MLP dimensions | 64,32 |
| `--dropout` | Dropout rate | 0.3 |
| `--cfr-lambda` | CFR regularization weight | 0.1 |
| `--win-eps` | Win propensity clipping floor | 0.05 |
| `--max-weight` | IPW/DR weight clipping ceiling | 10.0 |
| `--win-weight` | Win loss weight | 1.0 |
| `--ctr-weight` | CTR loss weight λ_c (ESCM² range [0, 0.1]) | 0.1 |
| `--joint-weight` | Joint loss weight | 1.0 |
| `--impute-loss-weight` | Imputation loss weight (DR) | 0.5 |
| `--dr-loss-type` | DR loss variant: 'mse' (paper) or 'bce' (pseudo-label) | mse |
| `--stop-grad-win-embedding` | Stop gradient from win tower to shared embedding | False |
| `--es-metric` | Early stopping metric: 'total', 'joint', or 'ctr_auc' | joint |
| `--patience` | Early stopping patience (epochs without improvement) | 10 |
| `--use-layer-norm` | Use LayerNorm in MLP towers | False |
| `--use-numeric-bypass` | Pass raw numerical features to MLP (skip embed projection) | False |
| `--use-scalar-input` | Treat ALL features as scalar floats (like LR) | False |
| `--exclude-features` | Comma-separated feature names to exclude | None |
| `--weight-decay` | AdamW weight decay (0 = vanilla Adam) | 1e-5 |
| `--scheduler` | LR scheduler: constant, cosine, linear | constant |
| `--warmup-steps` | LR warmup steps | 0 |
| `--gradient-clip` | Gradient clipping norm (0 = disabled) | 0.0 |
| `--eval-every` | Val AUC/ECE/IEB metrics every N epochs | 5 |
| `--max-samples` | Limit training samples (for smoke testing) | None (all) |
| `--quiet`, `-q` | Suppress progress output | False |
| `--distributed` | Enable distributed SPMD training | False |
| `--num-devices` | Number of devices (None = auto-detect) | None |
| `--resume-from` | Checkpoint path to resume from | None |
| `--use-wandb` | Enable W&B logging | False |
| `--wandb-project` | W&B project name | rtb-ipinyou |
| `--wandb-run-name` | W&B run name (auto-generated if None) | None |
| `--config-dir` | Hydra config directory (e.g., `configs`) | None |
| `--overrides` / `-O` | Hydra overrides, comma-separated | None |

**`evaluate`**: Compare all trained models

```bash
python scripts/train.py evaluate \
    --model-dir results/models \
    --data-dir data/ipinyou/prediction/features \
    --output results/comparison.json
```

**`calibration`**: Analyze model calibration

```bash
python scripts/train.py calibration \
    --model-dir results/models \
    --data-dir data/ipinyou/prediction/features \
    --task ctr
```

## Data Directory Structure

After running the pipeline:

```
data/ipinyou/
├── raw/ipinyou/           # Original data (read-only)
│   ├── training2nd/
│   │   ├── bid.*.txt.bz2
│   │   ├── imp.*.txt.bz2
│   │   ├── clk.*.txt.bz2
│   │   └── conv.*.txt.bz2
│   └── training3rd/
│       └── ...
│
└── prediction/            # Processed data
    ├── unified/           # Partitioned Parquet
    │   ├── season=2/
    │   │   ├── day=20130606/
    │   │   └── ...
    │   └── season=3/
    │       └── ...
    │
    └── features/          # Feature-engineered data
        ├── train.parquet
        ├── val.parquet
        ├── test.parquet
        ├── feature_metadata.json
        ├── vocab/
        │   └── usertag_vocab.json
        └── stats/
            ├── region_stats.parquet
            └── market_stats.parquet
```

## Output Files

### Preprocessing Outputs

| File | Description |
|------|-------------|
| `unified/season=*/day=*/` | Partitioned Parquet by season and day |

### Feature Outputs

| File | Description |
|------|-------------|
| `train.parquet` | Training set |
| `val.parquet` | Validation set |
| `test.parquet` | Test set |
| `feature_metadata.json` | Feature names, types, and counts |
| `vocab/usertag_vocab.json` | Usertag vocabulary mapping |
| `stats/region_stats.parquet` | Region-level statistics |
| `stats/market_stats.parquet` | Market price statistics |

### Model Outputs

| File | Description |
|------|-------------|
| `lgb_ctr.txt` | LightGBM CTR model (biased, won only) |
| `lgb_win.txt` | LightGBM Win model (all bids) |
| `lgb_ctr_all.txt` | LightGBM population CTR model (all bids) |
| `lgb_ctr_result.json` | LGB CTR training metrics |
| `lgb_win_result.json` | LGB Win training metrics |
| `lr_ctr.joblib` | Logistic Regression CTR model (won only) |
| `lr_ctr_result.json` | LR CTR training metrics |
| `lr_win.joblib` | Logistic Regression Win model (all bids) |
| `lr_win_result.json` | LR Win training metrics |
| `lr_ctr_all.joblib` | Logistic Regression population CTR model (all bids) |
| `lr_ctr_all_result.json` | LR CTR_all training metrics |
| `esmmwc_result.json` | ESMM-WC training results |
| `escm2wc_dr_result.json` | ESCM2-WC(DR) training results |

## Examples

### End-to-End Pipeline

```bash
#!/bin/bash
# Full pipeline from raw data to trained model

DATA_ROOT="data/ipinyou"
RESULTS="results"

# Step 1: Preprocess
python scripts/preprocess.py unify \
    --raw-dir $DATA_ROOT/raw/ipinyou \
    --output-dir $DATA_ROOT/prediction/unified \
    --seasons 2,3

# Validate
python scripts/preprocess.py validate \
    --data-dir $DATA_ROOT/prediction/unified

# Step 2: Feature engineering
python scripts/build_features.py build \
    --data-dir $DATA_ROOT/prediction/unified \
    --output-dir $DATA_ROOT/prediction/features \
    --split-method temporal

# Step 3: Train baselines
python scripts/train.py baseline \
    --data-dir $DATA_ROOT/prediction/features \
    --model-dir $RESULTS/models \
    --task ctr

python scripts/train.py baseline \
    --data-dir $DATA_ROOT/prediction/features \
    --model-dir $RESULTS/models \
    --task win

# Step 4: Train ESMM-WC / ESCM2-WC
python scripts/train.py esmmwc \
    --data-dir $DATA_ROOT/prediction/features \
    --model-dir $RESULTS/models

python scripts/train.py escm2wc \
    --data-dir $DATA_ROOT/prediction/features \
    --model-dir $RESULTS/models \
    --debiasing dr

# Step 5: Evaluate
python scripts/train.py evaluate \
    --model-dir $RESULTS/models \
    --data-dir $DATA_ROOT/prediction/features
```

### End-to-End Pipeline (Parallel)

```bash
#!/bin/bash
# Full pipeline with parallel processing (3-5x faster)

DATA_ROOT="data/ipinyou"
RESULTS="results"
WORKERS=8  # Adjust based on available CPUs

# Step 1: Preprocess (parallel file parsing)
python scripts/preprocess.py unify \
    --raw-dir $DATA_ROOT/raw/ipinyou \
    --output-dir $DATA_ROOT/prediction/unified \
    --seasons 2,3 \
    --workers $WORKERS

# Step 2: Feature engineering (parallel partitions)
python scripts/build_features.py build \
    --data-dir $DATA_ROOT/prediction/unified \
    --output-dir $DATA_ROOT/prediction/features \
    --split-method temporal \
    --workers $WORKERS

# Steps 3-5: Same as before (training is GPU-bound)
python scripts/train.py baseline \
    --data-dir $DATA_ROOT/prediction/features \
    --model-dir $RESULTS/models \
    --task ctr

# For ESMM-WC / ESCM2-WC
python scripts/train.py esmmwc \
    --data-dir $DATA_ROOT/prediction/features \
    --model-dir $RESULTS/models

python scripts/train.py escm2wc \
    --data-dir $DATA_ROOT/prediction/features \
    --model-dir $RESULTS/models \
    --debiasing dr
```

### Testing with Small Sample

```bash
# Process only 10000 rows per file for testing
python scripts/preprocess.py unify \
    --raw-dir data/ipinyou/raw/ipinyou \
    --output-dir data/ipinyou/prediction/unified_test \
    --seasons 2 \
    --max-rows 10000
```

## Troubleshooting

### Import Errors

```bash
# Add to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Memory Issues

For large datasets:
1. Process one season at a time
2. Use `--max-rows` for testing
3. Use sparse encoding for usertags

### Missing Dependencies

```bash
pip install typer pandas pyarrow lightgbm scikit-learn
pip install jax flax optax  # For ESMM-WC / ESCM2-WC
pip install ray  # For parallel processing (optional)
pip install hydra-core omegaconf  # For Hydra config management (optional)
```

## Ray Module Reference

The `src/ray_utils.py` module provides parallel processing utilities:

| Function | Description |
|----------|-------------|
| `init_ray(num_cpus=N)` | Initialize Ray cluster |
| `shutdown_ray()` | Shutdown Ray cluster |
| `parallel_files(files, parse_fn)` | Parse files in parallel |
| `parallel_apply(df, func)` | Apply function to DataFrame partitions |
| `parallel_apply_with_shared(df, func, shared_data)` | Same with shared data in object store |
| `batch_process(items, func)` | Process items in batches |
| `batch_encode(data, encode_fn, shared)` | Encode arrays in parallel batches |

### Usage in Custom Scripts

```python
from src.ray_utils import init_ray, parallel_apply, RAY_AVAILABLE

# Initialize with 8 CPUs
if RAY_AVAILABLE:
    init_ray(num_cpus=8)

# Process DataFrame in parallel
df_result = parallel_apply(df, my_transform_fn)

# Or use parallel feature engineering directly
from src.features.engineering import engineer_features_parallel
df_features = engineer_features_parallel(df, n_partitions=8)
```

### Parallel Functions by Module

**`src/data/parser.py`**:
- `parse_season_logs_parallel()` - Parse all season logs in parallel
- `parse_all_seasons_parallel()` - Parse multiple seasons in parallel

**`src/features/engineering.py`**:
- `engineer_features_parallel()` - Feature engineering with partition parallelism

**`src/features/usertag.py`**:
- `encode_multihot_parallel()` - Parallel multi-hot encoding
- `build_vocab_parallel()` - Parallel vocabulary building with map-reduce
