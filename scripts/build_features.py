#!/usr/bin/env python
"""Build features for RTB prediction models.

Apply feature engineering and create train/val/test splits.

Usage:
    # Build features and split data
    python scripts/build_features.py build \
        --data-dir data/ipinyou/prediction/unified \
        --output-dir data/ipinyou/prediction/features \
        --split-method temporal \
        --train-ratio 0.7 \
        --val-ratio 0.15

    # Build with parallel processing (Ray)
    python scripts/build_features.py build \
        --data-dir data/ipinyou/prediction/unified \
        --output-dir data/ipinyou/prediction/features \
        --workers 8

    # Build with sparse usertag encoding
    python scripts/build_features.py build \
        --data-dir data/ipinyou/prediction/unified \
        --output-dir data/ipinyou/prediction/features \
        --usertag-encoding sparse

    # Show feature information
    python scripts/build_features.py info \
        --data-dir data/ipinyou/prediction/features

    # Build usertag vocabulary
    python scripts/build_features.py vocab \
        --data-dir data/ipinyou/prediction/unified \
        --output-dir data/ipinyou/prediction/features/vocab \
        --top-n 100
"""

from pathlib import Path
from typing import List, Optional, Any
import sys
import json
import time

import pandas as pd
import pyarrow.parquet as pq
import typer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import format_duration

from src.data.unifier import load_from_parquet, compute_dataset_stats
from src.features.engineering import (
    engineer_features,
    get_feature_info,
    split_temporal,
    split_by_days,
    save_feature_splits,
    load_feature_splits,
    compute_region_stats,
    compute_market_stats,
    compute_normalization_stats,
    target_encode_kfold,
)
from src.features.usertag import (
    build_vocab,
    build_vocab_parallel,
    save_vocab,
    load_vocab,
    add_usertag_features,
    compute_tag_stats,
    compute_coverage,
    encode_multihot_parallel,
    encode_multihot_sparse,
    encode_hashing,
)


# =============================================================================
# Sparse Usertag Utilities
# =============================================================================

def save_usertag_sparse(matrix: "scipy.sparse.csr_matrix", path: Path) -> None:
    """Save sparse usertag matrix to .npz file."""
    from scipy.sparse import save_npz
    save_npz(str(path), matrix)


def load_usertag_sparse(path: Path) -> "scipy.sparse.csr_matrix":
    """Load sparse usertag matrix from .npz file."""
    from scipy.sparse import load_npz
    return load_npz(str(path))

app = typer.Typer(
    name="build_features",
    help="Build features for RTB prediction models",
    add_completion=False,
)


@app.command()
def build(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing unified Parquet files",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Output directory for feature files",
    ),
    split_method: str = typer.Option(
        "temporal",
        "--split-method",
        "-s",
        help="Split method: 'temporal' or 'by_days'",
    ),
    train_ratio: float = typer.Option(
        0.7,
        "--train-ratio",
        help="Training set ratio (for temporal split)",
    ),
    val_ratio: float = typer.Option(
        0.15,
        "--val-ratio",
        help="Validation set ratio (for temporal split)",
    ),
    train_days: Optional[str] = typer.Option(
        None,
        "--train-days",
        help="Training days (comma-separated, for by_days split)",
    ),
    val_days: Optional[str] = typer.Option(
        None,
        "--val-days",
        help="Validation days (comma-separated, for by_days split)",
    ),
    test_days: Optional[str] = typer.Option(
        None,
        "--test-days",
        help="Test days (comma-separated, for by_days split)",
    ),
    usertag_encoding: str = typer.Option(
        "summary",
        "--usertag-encoding",
        help="Usertag encoding: 'summary' (vocab only), 'sparse' (multi-hot .npz), 'hashing'. WARNING: usertag causes Win Tower leakage in shared embedding models — use only with tower-specific feature selection.",
    ),
    usertag_top_n: int = typer.Option(
        100,
        "--usertag-top-n",
        help="Top N usertags to include in vocabulary",
    ),
    usertag_min_count: int = typer.Option(
        10,
        "--usertag-min-count",
        help="Minimum occurrence count for a usertag",
    ),
    no_usertag: bool = typer.Option(
        True,
        "--no-usertag",
        help="Skip usertag feature encoding (default: True — usertag excluded to prevent Win Tower leakage in shared embedding)",
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers for usertag processing (uses Ray if available)",
    ),
    domain_buckets: int = typer.Option(
        10000,
        "--domain-buckets",
        help="Hash buckets for domain (high-cardinality encoding)",
    ),
    creative_buckets: int = typer.Option(
        5000,
        "--creative-buckets",
        help="Hash buckets for creative (high-cardinality encoding)",
    ),
    target_encoding: bool = typer.Option(
        False,
        "--target-encoding/--no-target-encoding",
        help="Apply target encoding (K-fold OOF) for click/win on categorical features",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress progress output",
    ),
    config_dir: Optional[str] = typer.Option(
        None,
        "--config-dir",
        help="Hydra config directory (e.g., configs)",
    ),
    overrides: Optional[str] = typer.Option(
        None,
        "--overrides",
        "-O",
        help="Hydra overrides, comma-separated (e.g., 'features.split_method=by_days')",
    ),
) -> None:
    """Build features and create train/val/test splits.

    Pipeline:
    1. Load unified data
    2. Apply feature engineering (time, slot, region, competition)
    3. Build usertag vocabulary and encode
    4. Split data temporally or by specific days
    5. Compute statistics from training set
    6. Save splits and metadata
    """
    # Load Hydra config if --config-dir or --overrides provided
    if config_dir is not None or overrides is not None:
        from src.config_utils import load_config, parse_overrides
        override_list = parse_overrides(overrides)
        cfg = load_config(config_dir=config_dir, overrides=override_list)
        # Apply config values where CLI args are at defaults
        if split_method == "temporal" and hasattr(cfg, "features"):
            split_method = cfg.features.get("split_method", split_method)
        if train_ratio == 0.7 and hasattr(cfg, "features"):
            train_ratio = cfg.features.get("train_ratio", train_ratio)
        if val_ratio == 0.15 and hasattr(cfg, "features"):
            val_ratio = cfg.features.get("val_ratio", val_ratio)
        if usertag_encoding == "summary" and hasattr(cfg, "features"):
            usertag_encoding = cfg.features.get("usertag_encoding", usertag_encoding)
        if usertag_top_n == 100 and hasattr(cfg, "features"):
            usertag_top_n = cfg.features.get("usertag_top_n", usertag_top_n)
        if usertag_min_count == 10 and hasattr(cfg, "features"):
            usertag_min_count = cfg.features.get("usertag_min_count", usertag_min_count)
        if hasattr(cfg, "features"):
            hash_cfg = cfg.features.get("hash_encoding", {})
            if domain_buckets == 10000:
                domain_buckets = hash_cfg.get("domain_buckets", domain_buckets)
            if creative_buckets == 5000:
                creative_buckets = hash_cfg.get("creative_buckets", creative_buckets)

    total_start = time.time()
    typer.echo(f"Loading data from: {data_dir}")

    # Columns needed for feature engineering — skip large string columns
    # (useragent, ip, url, urlid, keypageurl) to reduce I/O ~30-40%
    FEATURE_COLUMNS = [
        "bidid", "timestamp", "ipinyouid",
        "region", "city", "adexchange", "domain",
        "slotid", "slotwidth", "slotheight", "slotvisibility", "slotformat",
        "slotprice", "creative", "bidprice", "advertiser", "usertag",
        "win", "click", "conversion", "payprice",
        "season", "day",
    ]

    # Load unified data — direct pyarrow for progress + Arrow-backed types
    t0 = time.time()
    try:
        dataset = pq.ParquetDataset(data_dir)
        total_rows = sum(f.metadata.num_rows for f in dataset.fragments)
        typer.echo(f"  {total_rows:,} rows across {len(dataset.fragments)} files")
        typer.echo(f"  Reading {len(FEATURE_COLUMNS)} columns (skipping useragent, ip, url, urlid, keypageurl)...")

        table = pq.read_table(data_dir, columns=FEATURE_COLUMNS)
        typer.echo(f"  Converting to pandas (Arrow-backed dtypes)...")
        df = table.to_pandas(types_mapper=pd.ArrowDtype)
        del table

        mem_gb = df.memory_usage(deep=True).sum() / 1e9
        typer.echo(f"Loaded {len(df):,} rows, {len(df.columns)} columns ({mem_gb:.1f} GB)")
        typer.echo(f"[load data] completed in {format_duration(time.time() - t0)}")
    except Exception as e:
        typer.echo(f"Error loading data: {e}", err=True)
        raise typer.Exit(1)

    # Initialize Ray if workers specified
    use_parallel = False
    if workers is not None:
        try:
            from src.ray_utils import init_ray, RAY_AVAILABLE
            if RAY_AVAILABLE:
                init_ray(num_cpus=workers)
                use_parallel = True
                typer.echo(f"Parallel processing: {workers} workers (Ray)")
            else:
                typer.echo("Warning: Ray not available, using sequential processing", err=True)
        except ImportError:
            typer.echo("Warning: ray_utils not found, using sequential processing", err=True)

    # Apply feature engineering (always sequential — vectorized ops are memory-bandwidth
    # bound and faster than Ray partition parallelism which adds serialization overhead)
    typer.echo(f"\nEngineering features ({len(df):,} rows, vectorized)...")
    typer.echo(f"  Hash encoding: domain={domain_buckets} buckets, creative={creative_buckets} buckets")
    t0 = time.time()
    df = engineer_features(df, domain_buckets=domain_buckets, creative_buckets=creative_buckets)
    typer.echo(f"[feature engineering] completed in {format_duration(time.time() - t0)}")

    # Build usertag vocabulary and encode
    usertag_sparse_data = None  # Will hold {split_name: sparse_matrix} if sparse encoding
    if not no_usertag and "usertag" in df.columns:
        n_nonnull = df["usertag"].notna().sum()
        if n_nonnull == 0:
            typer.echo("\nWarning: All usertag values are null — skipping usertag features")
            vocab = None
        else:
            typer.echo(f"\nBuilding usertag vocabulary (top {usertag_top_n}, min_count={usertag_min_count}, {n_nonnull:,} non-null rows)...")
            t0 = time.time()
            if use_parallel:
                vocab = build_vocab_parallel(df["usertag"], top_n=usertag_top_n, min_count=usertag_min_count, n_partitions=workers)
            else:
                vocab = build_vocab(df["usertag"], top_n=usertag_top_n, min_count=usertag_min_count)
            typer.echo(f"[build vocab] completed in {format_duration(time.time() - t0)}")
            typer.echo(f"  Vocabulary size: {vocab.n_tags}")

            # Compute coverage
            coverage = compute_coverage(df["usertag"], vocab)
            typer.echo(f"  Coverage: {coverage:.2%}")

            typer.echo(f"  Encoding mode: {usertag_encoding}")

            # For sparse/hashing: encode after split (per-split .npz files)
            if usertag_encoding in ("sparse", "hashing"):
                typer.echo(f"  Sparse/hashing encoding will be saved as .npz after split")

            # Save vocabulary
            vocab_dir = output_dir / "vocab"
            vocab_dir.mkdir(parents=True, exist_ok=True)
            save_vocab(vocab, vocab_dir / "usertag_vocab.json")
            typer.echo(f"  Saved vocabulary to: {vocab_dir}")
    else:
        vocab = None

    # Split data
    typer.echo(f"\nSplitting data ({split_method})...")

    if split_method == "temporal":
        test_ratio = 1.0 - train_ratio - val_ratio
        train_df, val_df, test_df = split_temporal(
            df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    elif split_method == "by_days":
        if not all([train_days, val_days, test_days]):
            typer.echo("Error: --train-days, --val-days, --test-days required for by_days split", err=True)
            raise typer.Exit(1)

        train_df, val_df, test_df = split_by_days(
            df,
            train_days=train_days.split(","),
            val_days=val_days.split(","),
            test_days=test_days.split(","),
        )
    else:
        typer.echo(f"Error: Unknown split method: {split_method}", err=True)
        raise typer.Exit(1)

    typer.echo(f"  Train: {len(train_df):,}")
    typer.echo(f"  Val:   {len(val_df):,}")
    typer.echo(f"  Test:  {len(test_df):,}")

    # Target encoding (K-fold OOF for train, full train stats for val/test)
    if target_encoding:
        te_cats = ["region", "city", "advertiser", "domain_hash", "creative_hash"]
        typer.echo(f"\nApplying target encoding ({len(te_cats)} features × 2 targets)...")
        t0 = time.time()

        for target_col, suffix in [("click", "_te_click"), ("win", "_te_win")]:
            te_result = target_encode_kfold(train_df, val_df, te_cats, target_col)
            # Train: K-fold OOF encoded
            train_te = te_result.train_encoded.rename(
                columns={c: c.replace("_te", suffix) for c in te_result.te_features}
            )
            train_df = pd.concat([train_df, train_te], axis=1)
            # Val: full train stats
            val_te = te_result.val_encoded.rename(
                columns={c: c.replace("_te", suffix) for c in te_result.te_features}
            )
            val_df = pd.concat([val_df, val_te], axis=1)
            # Test: full train stats (same encodings as val)
            test_te = pd.DataFrame(index=test_df.index)
            for col, te_col in zip(te_cats, te_result.te_features):
                out_col = te_col.replace("_te", suffix)
                test_te[out_col] = (
                    pd.Series(test_df[col].values)
                    .map(te_result.encodings[col])
                    .fillna(te_result.global_mean)
                    .values
                )
            test_df = pd.concat([test_df, test_te], axis=1)
            typer.echo(f"  {target_col}: {len(te_cats)} features, global_mean={te_result.global_mean:.6f}")

        typer.echo(f"[target encoding] completed in {format_duration(time.time() - t0)}")

    # Compute statistics from training set (for region/market features)
    typer.echo("\nComputing training set statistics...")
    region_stats = compute_region_stats(train_df)
    typer.echo(f"  Region stats: {len(region_stats)} regions")

    # Compute market stats if slot_size_group exists
    if "slot_size_group" in train_df.columns:
        market_stats = compute_market_stats(train_df)
        typer.echo(f"  Market stats: {len(market_stats)} groups")
    else:
        market_stats = None

    # Save statistics
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    region_stats.to_parquet(stats_dir / "region_stats.parquet", index=False)
    if market_stats is not None and len(market_stats) > 0:
        market_stats.to_parquet(stats_dir / "market_stats.parquet", index=False)

    # Get feature info (explicit categorical set, not dtype-based)
    feature_info = get_feature_info(train_df)

    # Compute normalization stats from training set (for neural model preprocessing)
    typer.echo("\nComputing normalization statistics...")
    norm_stats = compute_normalization_stats(train_df, feature_info.numerical)
    typer.echo(f"  Z-score stats: {len(norm_stats.mean)} numerical features")

    # Save splits
    typer.echo(f"\nSaving to: {output_dir}")
    t0 = time.time()
    save_feature_splits(train_df, val_df, test_df, output_dir, feature_info, norm_stats)
    typer.echo(f"[save splits] completed in {format_duration(time.time() - t0)}")

    # Save sparse usertag encoding if requested
    if vocab is not None and usertag_encoding in ("sparse", "hashing"):
        import numpy as np

        typer.echo(f"\nEncoding usertags ({usertag_encoding})...")
        t0 = time.time()
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if usertag_encoding == "sparse":
                sparse_matrix = encode_multihot_sparse(split_df["usertag"], vocab)
            else:  # hashing
                dense_matrix = encode_hashing(split_df["usertag"], n_features=vocab.n_tags - 1)
                from scipy.sparse import csr_matrix
                sparse_matrix = csr_matrix(dense_matrix)

            out_path = output_dir / f"{split_name}_usertag.npz"
            save_usertag_sparse(sparse_matrix, out_path)
            typer.echo(f"  {split_name}_usertag.npz: shape={sparse_matrix.shape}, nnz={sparse_matrix.nnz:,}")

        typer.echo(f"[usertag encoding] completed in {format_duration(time.time() - t0)}")

    # Print summary
    typer.echo("\n" + "=" * 60)
    typer.echo(typer.style("✅ Feature building complete!", fg=typer.colors.GREEN, bold=True))
    typer.echo(f"Total features: {feature_info.total_count}")
    typer.echo(f"  Categorical: {len(feature_info.categorical)}")
    typer.echo(f"  Numerical: {len(feature_info.numerical)}")

    # Print dataset stats
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        stats = compute_dataset_stats(split_df)
        typer.echo(f"\n{name} set:")
        typer.echo(f"  Rows: {stats.n_bids:,} | Win: {stats.win_rate:.2%} | CTR: {stats.ctr:.4%}")

    typer.echo(f"\n[TOTAL] completed in {format_duration(time.time() - total_start)}")


@app.command()
def info(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing feature files",
    ),
) -> None:
    """Show information about feature dataset.

    Displays:
    - Split sizes
    - Feature counts by type
    - Feature groups
    - Sample statistics
    """
    typer.echo(f"Loading features from: {data_dir}")

    try:
        train_df, val_df, test_df, metadata = load_feature_splits(data_dir)
    except Exception as e:
        typer.echo(f"Error loading features: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("\n" + "=" * 60)
    typer.echo("Feature Dataset Information")
    typer.echo("=" * 60)

    typer.echo(f"\nSplit sizes:")
    typer.echo(f"  Train: {metadata['train_size']:,}")
    typer.echo(f"  Val:   {metadata['val_size']:,}")
    typer.echo(f"  Test:  {metadata['test_size']:,}")

    if "feature_info" in metadata:
        fi = metadata["feature_info"]
        typer.echo(f"\nFeature counts:")
        typer.echo(f"  Total:       {fi['total_count']}")
        typer.echo(f"  Categorical: {len(fi['categorical'])}")
        typer.echo(f"  Numerical:   {len(fi['numerical'])}")

        typer.echo(f"\nFeature groups:")
        for group, features in fi.get("feature_groups", {}).items():
            if features:
                typer.echo(f"  {group}: {len(features)} features")

        typer.echo(f"\nCategorical features:")
        for f in fi["categorical"][:10]:
            typer.echo(f"  - {f}")
        if len(fi["categorical"]) > 10:
            typer.echo(f"  ... and {len(fi['categorical']) - 10} more")

        typer.echo(f"\nNumerical features:")
        for f in fi["numerical"][:10]:
            typer.echo(f"  - {f}")
        if len(fi["numerical"]) > 10:
            typer.echo(f"  ... and {len(fi['numerical']) - 10} more")

    # Print label statistics
    typer.echo("\nLabel statistics (training set):")
    for col in ["win", "click", "conversion"]:
        if col in train_df.columns:
            rate = train_df[col].mean()
            count = train_df[col].sum()
            typer.echo(f"  {col}: {count:,} ({rate:.4%})")


@app.command()
def vocab(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing unified Parquet files",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Output directory for vocabulary",
    ),
    top_n: int = typer.Option(
        100,
        "--top-n",
        "-n",
        help="Top N most frequent tags to include",
    ),
    min_count: int = typer.Option(
        10,
        "--min-count",
        help="Minimum occurrence count for a tag",
    ),
) -> None:
    """Build usertag vocabulary from data.

    Creates vocabulary of top-N most frequent usertags.
    """
    typer.echo(f"Loading data from: {data_dir}")

    try:
        df = load_from_parquet(data_dir)
    except Exception as e:
        typer.echo(f"Error loading data: {e}", err=True)
        raise typer.Exit(1)

    if "usertag" not in df.columns:
        typer.echo("Error: No 'usertag' column in data", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nBuilding vocabulary (top {top_n}, min_count={min_count})...")
    vocab_obj = build_vocab(df["usertag"], top_n=top_n, min_count=min_count)

    typer.echo(f"  Vocabulary size: {vocab_obj.n_tags}")
    coverage = compute_coverage(df["usertag"], vocab_obj)
    typer.echo(f"  Coverage: {coverage:.2%}")

    # Show top tags
    typer.echo("\nTop 10 tags:")
    sorted_tags = sorted(vocab_obj.tag_counts.items(), key=lambda x: -x[1])[:10]
    for tag, count in sorted_tags:
        typer.echo(f"  Tag {tag}: {count:,}")

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_vocab(vocab_obj, output_dir / "usertag_vocab.json")

    typer.echo(f"\n✅ Saved to: {output_dir / 'usertag_vocab.json'}")


@app.command()
def stats(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing feature files",
    ),
    split: str = typer.Option(
        "train",
        "--split",
        "-s",
        help="Split to analyze: 'train', 'val', or 'test'",
    ),
) -> None:
    """Show detailed statistics for a split.

    Displays feature distributions and correlations with labels.
    """
    typer.echo(f"Loading {split} split from: {data_dir}")

    try:
        train_df, val_df, test_df, metadata = load_feature_splits(data_dir)
    except Exception as e:
        typer.echo(f"Error loading features: {e}", err=True)
        raise typer.Exit(1)

    if split == "train":
        df = train_df
    elif split == "val":
        df = val_df
    elif split == "test":
        df = test_df
    else:
        typer.echo(f"Error: Unknown split: {split}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\n{split.upper()} Split Statistics")
    typer.echo("=" * 60)
    typer.echo(f"Rows: {len(df):,}")

    # Label statistics
    typer.echo("\nLabel distribution:")
    for col in ["win", "click", "conversion"]:
        if col in df.columns:
            rate = df[col].mean()
            count = df[col].sum()
            typer.echo(f"  {col}: {count:,} / {len(df):,} ({rate:.4%})")

    # Numerical feature statistics
    typer.echo("\nNumerical feature statistics:")
    if "feature_info" in metadata:
        numerical = metadata["feature_info"]["numerical"][:5]
        for col in numerical:
            if col in df.columns:
                typer.echo(f"  {col}:")
                typer.echo(f"    mean={df[col].mean():.4f}, std={df[col].std():.4f}")
                typer.echo(f"    min={df[col].min():.4f}, max={df[col].max():.4f}")

    # Time distribution
    if "hour" in df.columns:
        typer.echo("\nHour distribution (top 5):")
        hour_dist = df["hour"].value_counts().head(5)
        for hour, count in hour_dist.items():
            typer.echo(f"  Hour {hour}: {count:,} ({count/len(df):.1%})")


@app.command("target-encode")
def target_encode_cmd(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        "-d",
        help="Directory containing existing feature splits (train/val/test.parquet)",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Output directory for TE-augmented feature splits",
    ),
) -> None:
    """Apply target encoding to existing feature splits.

    Loads train/val/test.parquet, applies K-fold OOF target encoding
    for click and win targets on categorical features, and saves
    augmented splits with updated metadata.
    """
    import numpy as np

    total_start = time.time()
    typer.echo(f"Loading existing splits from: {data_dir}")

    t0 = time.time()
    train_df, val_df, test_df, metadata = load_feature_splits(data_dir)
    typer.echo(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    typer.echo(f"[load splits] completed in {format_duration(time.time() - t0)}")

    te_cats = ["region", "city", "advertiser", "domain_hash", "creative_hash"]
    typer.echo(f"\nApplying target encoding ({len(te_cats)} features × 2 targets)...")
    t0 = time.time()

    for target_col, suffix in [("click", "_te_click"), ("win", "_te_win")]:
        te_result = target_encode_kfold(train_df, val_df, te_cats, target_col)
        # Train: K-fold OOF encoded
        train_te = te_result.train_encoded.rename(
            columns={c: c.replace("_te", suffix) for c in te_result.te_features}
        )
        train_df = pd.concat([train_df, train_te], axis=1)
        # Val: full train stats
        val_te = te_result.val_encoded.rename(
            columns={c: c.replace("_te", suffix) for c in te_result.te_features}
        )
        val_df = pd.concat([val_df, val_te], axis=1)
        # Test: full train stats (same encodings as val)
        test_te = pd.DataFrame(index=test_df.index)
        for col, te_col in zip(te_cats, te_result.te_features):
            out_col = te_col.replace("_te", suffix)
            test_te[out_col] = (
                pd.Series(test_df[col].values)
                .map(te_result.encodings[col])
                .fillna(te_result.global_mean)
                .values
            )
        test_df = pd.concat([test_df, test_te], axis=1)
        typer.echo(f"  {target_col}: {len(te_cats)} features, global_mean={te_result.global_mean:.6f}")

    typer.echo(f"[target encoding] completed in {format_duration(time.time() - t0)}")

    # Recompute feature info and normalization stats with new TE columns
    feature_info = get_feature_info(train_df)
    norm_stats = compute_normalization_stats(train_df, feature_info.numerical)
    typer.echo(f"\nFeatures: {feature_info.total_count} total ({len(feature_info.categorical)} cat, {len(feature_info.numerical)} num)")

    # Save augmented splits
    typer.echo(f"\nSaving to: {output_dir}")
    t0 = time.time()
    save_feature_splits(train_df, val_df, test_df, output_dir, feature_info, norm_stats)

    # Copy region/market stats from source
    src_stats = data_dir / "stats"
    dst_stats = output_dir / "stats"
    if src_stats.exists():
        dst_stats.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in src_stats.iterdir():
            shutil.copy2(f, dst_stats / f.name)
        typer.echo(f"  Copied stats/ from source")

    typer.echo(f"[save splits] completed in {format_duration(time.time() - t0)}")

    # Verify TE columns
    te_cols = [c for c in train_df.columns if "_te_click" in c or "_te_win" in c]
    typer.echo(f"\nTE columns added: {te_cols}")

    # Summary
    typer.echo("\n" + "=" * 60)
    typer.echo(typer.style("Feature building with TE complete!", fg=typer.colors.GREEN, bold=True))
    typer.echo(f"Total features: {feature_info.total_count}")
    typer.echo(f"  Categorical: {len(feature_info.categorical)}")
    typer.echo(f"  Numerical: {len(feature_info.numerical)}")

    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        stats = compute_dataset_stats(split_df)
        typer.echo(f"\n{name} set:")
        typer.echo(f"  Rows: {stats.n_bids:,} | Win: {stats.win_rate:.2%} | CTR: {stats.ctr:.4%}")

    typer.echo(f"\n[TOTAL] completed in {format_duration(time.time() - total_start)}")


if __name__ == "__main__":
    app()
