"""Feature engineering functions for RTB prediction models.

Features are organized into groups:
- Time features: hour, weekday, is_weekend, is_peak, hour_sin/cos
- Slot features: slot_area, slot_size_group, visibility
- Region features: region_freq, region_group
- Competition features: avg_market_price, std_market_price (from training data)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
import hashlib
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Classification
# =============================================================================

# Explicit categorical feature set — ID/ordinal features that should use
# embedding lookup (neural) or categorical split (LightGBM).
# Aligned with configs/features/default.yaml.
CATEGORICAL_FEATURES = frozenset({
    # ID features (integer-coded discrete entities)
    "region", "city", "adexchange", "advertiser",
    # Slot categorical (limited IAB standard values; continuous info in slot_area/aspect_ratio)
    "slotwidth", "slotheight", "slotvisibility", "slotformat", "slot_size_group",
    # Time categorical (ordinal → embedding; cyclical info in hour_sin/cos)
    "hour", "minute", "weekday",
    # Derived binary/categorical
    "is_weekend", "is_peak_hour", "region_group",
    # High-cardinality (hash-encoded)
    "domain_hash", "creative_hash",
})

# Columns to skip (raw IDs, labels, high-cardinality strings, metadata)
_SKIP_COLS = frozenset({
    "bidid", "timestamp", "ipinyouid", "useragent", "ip", "url", "urlid",
    "slotid", "creative", "keypageurl", "usertag", "day", "domain",
    "win", "click", "conversion", "payprice", "season",
})


# =============================================================================
# Result Types
# =============================================================================

class FeatureInfo(NamedTuple):
    """Metadata about engineered features."""
    categorical: List[str]
    numerical: List[str]
    total_count: int
    feature_groups: Dict[str, List[str]]


class NormalizationStats(NamedTuple):
    """Per-feature Z-score normalization statistics (from training set only)."""
    mean: Dict[str, float]
    std: Dict[str, float]


class TargetEncodingResult(NamedTuple):
    """K-fold target encoding result."""
    train_encoded: pd.DataFrame   # TE columns for train
    val_encoded: pd.DataFrame     # TE columns for val
    te_features: List[str]        # e.g. ['region_te', 'city_te', ...]
    encodings: Dict[str, Dict]    # col → {category: smoothed_mean}
    global_mean: float


def target_encode_kfold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cat_features: List[str],
    target_col: str,
    n_folds: int = 5,
    smoothing_m: float = 10.0,
    seed: int = 42,
) -> TargetEncodingResult:
    """K-fold target encoding with Bayesian smoothing (leakage-free).

    Train: K-fold out-of-fold encoding to prevent leakage.
    Val/Test: Full training set statistics applied.

    Smoothing formula: (count * cat_mean + m * global_mean) / (count + m)

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        cat_features: List of categorical column names to encode
        target_col: Target column name (e.g. 'win', 'click')
        n_folds: Number of folds for K-fold encoding
        smoothing_m: Bayesian smoothing parameter (higher = more regularization)
        seed: Random seed for KFold

    Returns:
        TargetEncodingResult with encoded DataFrames, feature names, and encodings
    """
    target = train_df[target_col].values
    global_mean = float(np.mean(target))

    te_features = [f"{col}_te" for col in cat_features]
    train_encoded = pd.DataFrame(index=train_df.index)
    val_encoded = pd.DataFrame(index=val_df.index)
    encodings: Dict[str, Dict] = {}

    # Pure numpy K-fold split (no sklearn dependency)
    n = len(train_df)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[:n % n_folds] += 1
    fold_indices = np.split(indices, np.cumsum(fold_sizes)[:-1])

    for col, te_col in zip(cat_features, te_features):
        # K-fold OOF encoding for train
        train_encoded[te_col] = global_mean
        for fold_i in range(n_folds):
            oof_idx = fold_indices[fold_i]
            train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != fold_i])
            fold_target = target[train_idx]
            fold_cats = train_df[col].values[train_idx]
            # Compute per-category stats on fold train (vectorized)
            fold_series = pd.Series(fold_target, index=fold_cats)
            fold_stats = fold_series.groupby(level=0).agg(['sum', 'count'])
            cat_sum = fold_stats['sum'].to_dict()
            cat_count = fold_stats['count'].to_dict()
            # Smoothed means
            smoothed = {
                k: (cat_sum[k] + smoothing_m * global_mean) / (cat_count[k] + smoothing_m)
                for k in cat_sum
            }
            # Apply to OOF indices (vectorized)
            oof_cats = train_df[col].values[oof_idx]
            oof_encoded = pd.Series(oof_cats).map(smoothed).fillna(global_mean).values
            train_encoded.iloc[oof_idx, train_encoded.columns.get_loc(te_col)] = oof_encoded

        # Full training set encoding for val (vectorized)
        full_series = pd.Series(target, index=train_df[col].values)
        full_stats = full_series.groupby(level=0).agg(['sum', 'count'])
        full_sum = full_stats['sum'].to_dict()
        full_count = full_stats['count'].to_dict()
        full_smoothed = {
            k: (full_sum[k] + smoothing_m * global_mean) / (full_count[k] + smoothing_m)
            for k in full_sum
        }
        encodings[col] = full_smoothed
        val_encoded[te_col] = pd.Series(val_df[col].values).map(full_smoothed).fillna(global_mean).values

    logger.info(
        "Target encoding: %d features, target='%s', global_mean=%.4f",
        len(cat_features), target_col, global_mean,
    )

    return TargetEncodingResult(
        train_encoded=train_encoded,
        val_encoded=val_encoded,
        te_features=te_features,
        encodings=encodings,
        global_mean=global_mean,
    )


# =============================================================================
# Time Features
# =============================================================================

def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add time-based features from timestamp.

    Features added:
    - hour (0-23)
    - minute (0-59)
    - weekday (0=Mon, 6=Sun)
    - is_weekend (0/1)
    - is_peak_hour (0/1) - morning/evening rush hours
    - hour_sin, hour_cos - cyclical encoding

    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column (format: YYYYMMDDHHMMSS...)

    Returns:
        DataFrame with added time features
    """
    ts = df[timestamp_col].astype(str)

    # Extract basic components
    df["hour"] = ts.str[8:10].astype("Int64")
    df["minute"] = ts.str[10:12].astype("Int64")

    # Parse date for weekday
    df["_date"] = pd.to_datetime(ts.str[:8], format="%Y%m%d", errors="coerce")
    df["weekday"] = df["_date"].dt.dayofweek.astype("Int64")
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Peak hours: 7-9 morning, 17-20 evening (vectorized)
    hour = df["hour"]
    df["is_peak_hour"] = (((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 20))).astype(int)

    # Cyclical encoding for hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Cleanup
    df = df.drop(columns=["_date"])

    return df


# =============================================================================
# Slot Features
# =============================================================================

_SLOT_SIZE_MAP = {
    (728, 90): "leaderboard", (970, 90): "leaderboard", (970, 250): "leaderboard",
    (300, 250): "medium_rectangle", (336, 280): "medium_rectangle",
    (300, 600): "skyscraper", (160, 600): "skyscraper", (120, 600): "skyscraper",
    (250, 250): "square", (200, 200): "square",
    (320, 50): "mobile", (320, 100): "mobile", (300, 50): "mobile",
    (468, 60): "banner",
}

# Pre-built lookup DataFrame for vectorized merge (avoids 130M+ Python tuple creation)
_SLOT_SIZE_LOOKUP = pd.DataFrame(
    [(w, h, group) for (w, h), group in _SLOT_SIZE_MAP.items()],
    columns=["slotwidth", "slotheight", "slot_size_group"],
)


def add_slot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ad slot derived features.

    Features added:
    - slot_area: width × height
    - slot_aspect_ratio: width / height
    - slot_size_group: standard IAB size categories

    Args:
        df: DataFrame with slotwidth, slotheight columns

    Returns:
        DataFrame with added slot features
    """
    # Area (log-transformed for better distribution)
    df["slot_area"] = df["slotwidth"] * df["slotheight"]
    df["slot_area_log"] = np.log1p(df["slot_area"])

    # Aspect ratio
    df["slot_aspect_ratio"] = (
        df["slotwidth"] / df["slotheight"].replace(0, 1)
    ).round(2)

    # Size group (categorical) — vectorized merge lookup (no Python tuple creation)
    n_before = len(df)
    df = df.merge(_SLOT_SIZE_LOOKUP, on=["slotwidth", "slotheight"], how="left")
    assert len(df) == n_before, "merge duplicated rows — check _SLOT_SIZE_LOOKUP for duplicate keys"
    df["slot_size_group"] = df["slot_size_group"].fillna("other")
    null_mask = df["slotwidth"].isna() | df["slotheight"].isna()
    df.loc[null_mask, "slot_size_group"] = "unknown"

    return df


# =============================================================================
# Region Features
# =============================================================================

def add_region_features(
    df: pd.DataFrame,
    region_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add region-based features.

    Features added:
    - region_freq: frequency encoding (count-based)
    - region_ctr: historical CTR by region (if stats provided)
    - region_group: high/medium/low volume groups

    Args:
        df: DataFrame with region column
        region_stats: Pre-computed region statistics (optional)

    Returns:
        DataFrame with added region features
    """
    # Frequency encoding
    region_counts = df["region"].value_counts()
    df["region_freq"] = df["region"].map(region_counts).fillna(0).astype(int)

    # Group by frequency (quantile-based)
    try:
        df["region_group"] = pd.qcut(
            df["region_freq"],
            q=4,
            labels=["low", "medium", "high", "very_high"],
            duplicates="drop",
        )
    except ValueError:
        # Fallback if not enough unique values
        df["region_group"] = "medium"

    # Add historical CTR if stats provided
    if region_stats is not None and "region" in region_stats.columns:
        n_before = len(df)
        df = df.merge(
            region_stats[["region", "region_ctr"]],
            on="region",
            how="left",
        )
        assert len(df) == n_before, "region merge duplicated rows — check region_stats for duplicate keys"
        df["region_ctr"] = df["region_ctr"].fillna(region_stats["region_ctr"].mean())

    return df


def compute_region_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute region-level statistics from training data.

    Args:
        df: Training DataFrame with region, win, click columns

    Returns:
        DataFrame with region statistics
    """
    stats = (
        df.groupby("region")
        .agg(
            n_bids=("bidid", "count"),
            n_wins=("win", "sum"),
            n_clicks=("click", "sum"),
        )
        .reset_index()
        .assign(
            region_win_rate=lambda x: x["n_wins"] / x["n_bids"],
            region_ctr=lambda x: x["n_clicks"] / x["n_wins"].clip(lower=1),
        )
    )
    return stats


# =============================================================================
# Competition Features
# =============================================================================

def add_competition_features(
    df: pd.DataFrame,
    market_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add competition/market-related features.

    Features added:
    - bid_floor_ratio: bidprice / slotprice (bid aggressiveness)
    - market_price_avg: average market price by slot/exchange (if stats provided)
    - market_price_std: std of market price (if stats provided)

    Args:
        df: DataFrame with bidprice, slotprice columns
        market_stats: Pre-computed market statistics (optional)

    Returns:
        DataFrame with added competition features
    """
    # Bid aggressiveness: how much above floor price
    df["bid_floor_ratio"] = (
        df["bidprice"] / df["slotprice"].replace(0, 1)
    ).round(3)

    # Clip extreme values
    df["bid_floor_ratio"] = df["bid_floor_ratio"].clip(0, 100)

    # Add market stats if provided
    if market_stats is not None:
        merge_keys = [c for c in ["adexchange", "slot_size_group"] if c in market_stats.columns]
        if merge_keys:
            n_before = len(df)
            df = df.merge(
                market_stats[merge_keys + ["market_price_avg", "market_price_std"]],
                on=merge_keys,
                how="left",
            )
            assert len(df) == n_before, "market merge duplicated rows — check market_stats for duplicate keys"
            # Fill with global mean
            df["market_price_avg"] = df["market_price_avg"].fillna(
                market_stats["market_price_avg"].mean()
            )
            df["market_price_std"] = df["market_price_std"].fillna(
                market_stats["market_price_std"].mean()
            )

    return df


def compute_market_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute market price statistics from won bids.

    Only uses won bids (payprice available) to compute statistics.

    Args:
        df: Training DataFrame with payprice, adexchange, slot_size_group

    Returns:
        DataFrame with market statistics
    """
    # Filter to won bids only
    df_won = df[df["win"] == 1].copy()

    if len(df_won) == 0:
        return pd.DataFrame()

    stats = (
        df_won.groupby(["adexchange", "slot_size_group"])
        .agg(
            market_price_avg=("payprice", "mean"),
            market_price_std=("payprice", "std"),
            n_wins=("bidid", "count"),
        )
        .reset_index()
    )

    # Fill NaN std with 0
    stats["market_price_std"] = stats["market_price_std"].fillna(0)

    return stats


def compute_normalization_stats(
    df: pd.DataFrame,
    numerical_features: List[str],
) -> NormalizationStats:
    """Compute Z-score normalization statistics from training data.

    Must be computed from the TRAINING set only to prevent data leakage.

    Args:
        df: Training DataFrame
        numerical_features: List of numerical feature column names

    Returns:
        NormalizationStats with per-feature mean and std
    """
    mean_dict = {}
    std_dict = {}

    for col in numerical_features:
        if col not in df.columns:
            continue
        col_data = df[col].dropna()
        mean_dict[col] = float(col_data.mean())
        std_dict[col] = max(float(col_data.std(ddof=0)), 1e-8)

    return NormalizationStats(mean=mean_dict, std=std_dict)


# =============================================================================
# High-Cardinality Features (Hash + Frequency Encoding)
# =============================================================================

def hash_encode(
    series: pd.Series,
    n_buckets: int,
    seed: int = 0,
) -> pd.Series:
    """Deterministic hash encoding for high-cardinality categorical features.

    Maps unique values to [1, n_buckets] range via MD5 hash.
    Bucket 0 is reserved for NaN/unknown.
    Uses unique-value mapping for efficiency (hash 108K uniques, map 129M rows).

    Args:
        series: Categorical series (string or numeric)
        n_buckets: Number of hash buckets (output range: [0, n_buckets])
        seed: Hash seed for reproducibility

    Returns:
        Integer-encoded series in [0, n_buckets] range
    """
    uniques = series.dropna().unique()
    hash_map = {
        v: int(hashlib.md5(f"{v}_{seed}".encode()).hexdigest(), 16) % n_buckets + 1
        for v in uniques
    }
    return series.map(hash_map).fillna(0).astype(np.int32)


def add_high_card_features(
    df: pd.DataFrame,
    domain_buckets: int = 10000,
    creative_buckets: int = 5000,
    hash_seed: int = 42,
) -> pd.DataFrame:
    """Add hash-encoded and frequency-encoded features for high-cardinality columns.

    Features added:
    - domain_hash: hash(domain) % n_buckets (categorical, for embedding)
    - creative_hash: hash(creative) % n_buckets (categorical, for embedding)
    - domain_freq: frequency encoding (numerical, count of each domain)
    - domain_freq_log: log(1 + domain_freq) (numerical, power law correction)
    - creative_freq: frequency encoding (numerical, count of each creative)
    - creative_freq_log: log(1 + creative_freq) (numerical, power law correction)

    Args:
        df: DataFrame with domain and/or creative columns
        domain_buckets: Number of hash buckets for domain (default: 10000)
        creative_buckets: Number of hash buckets for creative (default: 5000)
        hash_seed: Hash seed for reproducibility

    Returns:
        DataFrame with added high-cardinality features
    """
    if "domain" in df.columns:
        df["domain_hash"] = hash_encode(df["domain"], domain_buckets, hash_seed)
        domain_counts = df["domain"].value_counts()
        df["domain_freq"] = df["domain"].map(domain_counts).fillna(0).astype(np.int64)
        df["domain_freq_log"] = np.log1p(df["domain_freq"]).astype(np.float32)

    if "creative" in df.columns:
        df["creative_hash"] = hash_encode(df["creative"], creative_buckets, hash_seed)
        creative_counts = df["creative"].value_counts()
        df["creative_freq"] = df["creative"].map(creative_counts).fillna(0).astype(np.int64)
        df["creative_freq_log"] = np.log1p(df["creative_freq"]).astype(np.float32)

    return df


# =============================================================================
# Main Feature Engineering Pipeline
# =============================================================================

def engineer_features(
    df: pd.DataFrame,
    region_stats: Optional[pd.DataFrame] = None,
    market_stats: Optional[pd.DataFrame] = None,
    add_time: bool = True,
    add_slot: bool = True,
    add_region: bool = True,
    add_competition: bool = True,
    add_high_card: bool = True,
    domain_buckets: int = 10000,
    creative_buckets: int = 5000,
) -> pd.DataFrame:
    """Apply all feature engineering transformations.

    Args:
        df: Input DataFrame
        region_stats: Pre-computed region statistics (from training)
        market_stats: Pre-computed market statistics (from training)
        add_time: Add time features
        add_slot: Add slot features
        add_region: Add region features
        add_competition: Add competition features
        add_high_card: Add high-cardinality hash/frequency features (domain, creative)
        domain_buckets: Number of hash buckets for domain (default: 10000)
        creative_buckets: Number of hash buckets for creative (default: 5000)

    Returns:
        DataFrame with all engineered features (mutates input in-place)
    """
    if add_time and "timestamp" in df.columns:
        df = add_time_features(df)

    if add_slot and all(c in df.columns for c in ["slotwidth", "slotheight"]):
        df = add_slot_features(df)

    if add_region and "region" in df.columns:
        df = add_region_features(df, region_stats)

    if add_competition and "bidprice" in df.columns:
        df = add_competition_features(df, market_stats)

    if add_high_card:
        df = add_high_card_features(df, domain_buckets, creative_buckets)

    return df


def get_feature_info(
    df: pd.DataFrame,
    categorical_override: Optional[frozenset] = None,
) -> FeatureInfo:
    """Get metadata about features in DataFrame.

    Uses explicit CATEGORICAL_FEATURES set (not dtype-based detection)
    to correctly classify integer-typed ID/ordinal features as categorical.

    Args:
        df: DataFrame with engineered features
        categorical_override: Override for categorical feature set.
            If None, uses module-level CATEGORICAL_FEATURES.

    Returns:
        FeatureInfo with categorical/numerical lists and groups
    """
    cat_set = categorical_override if categorical_override is not None else CATEGORICAL_FEATURES

    # Define feature groups
    time_features = ["hour", "minute", "weekday", "is_weekend", "is_peak_hour", "hour_sin", "hour_cos"]
    slot_features = ["slot_area", "slot_area_log", "slot_aspect_ratio", "slot_size_group", "slotvisibility", "slotformat"]
    region_features = ["region", "city", "region_freq", "region_group", "region_ctr"]
    competition_features = ["bidprice", "slotprice", "bid_floor_ratio", "market_price_avg", "market_price_std"]
    id_features = ["adexchange", "advertiser"]
    high_card_features = ["domain_hash", "creative_hash", "domain_freq", "domain_freq_log", "creative_freq", "creative_freq_log"]

    def filter_existing(features):
        return [f for f in features if f in df.columns]

    feature_groups = {
        "time": filter_existing(time_features),
        "slot": filter_existing(slot_features),
        "region": filter_existing(region_features),
        "competition": filter_existing(competition_features),
        "id": filter_existing(id_features),
        "high_cardinality": filter_existing(high_card_features),
    }

    categorical = []
    numerical = []

    for col in df.columns:
        if col in _SKIP_COLS:
            continue

        if col in cat_set:
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(df[col].dtype):
            numerical.append(col)

    return FeatureInfo(
        categorical=categorical,
        numerical=numerical,
        total_count=len(categorical) + len(numerical),
        feature_groups=feature_groups,
    )


# =============================================================================
# Data Splitting
# =============================================================================

def _cast_to_large_string(df: pd.DataFrame) -> pd.DataFrame:
    """Cast string[pyarrow] columns to large_string[pyarrow] in-place.

    Avoids Arrow 32-bit offset overflow (>2GB) during take/reindex operations.
    """
    import pyarrow as pa

    for col in df.columns:
        dtype = df[col].dtype
        if hasattr(dtype, "pyarrow_dtype") and dtype.pyarrow_dtype == pa.string():
            df[col] = df[col].astype(pd.ArrowDtype(pa.large_string()))
    return df


def split_temporal(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    time_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data temporally (chronologically).

    Important for time-series to prevent data leakage.

    Uses np.argsort + iloc instead of df.sort_values().reset_index() to avoid
    Arrow string offset overflow on large DataFrames (>2GB per string column).
    Casts string[pyarrow] → large_string[pyarrow] (64-bit offsets) before iloc
    to prevent Arrow offset overflow during take() on columns exceeding 2GB.

    Args:
        df: DataFrame with timestamp column
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        time_col: Timestamp column name

    Returns:
        Tuple of (train, val, test) DataFrames
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Use argsort on the timestamp column alone to avoid Arrow string
    # concatenation overflow when sorting the full DataFrame
    sort_idx = np.argsort(df[time_col].values, kind="stable")
    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Cast string → large_string (64-bit offsets) to avoid Arrow overflow on take
    _cast_to_large_string(df)

    return (
        df.iloc[sort_idx[:train_end]].reset_index(drop=True),
        df.iloc[sort_idx[train_end:val_end]].reset_index(drop=True),
        df.iloc[sort_idx[val_end:]].reset_index(drop=True),
    )


def split_by_days(
    df: pd.DataFrame,
    train_days: List[str],
    val_days: List[str],
    test_days: List[str],
    day_col: str = "day",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by specific days.

    Args:
        df: DataFrame with day column
        train_days: List of training days (YYYYMMDD)
        val_days: List of validation days
        test_days: List of test days
        day_col: Day column name

    Returns:
        Tuple of (train, val, test) DataFrames
    """
    return (
        df[df[day_col].isin(train_days)],
        df[df[day_col].isin(val_days)],
        df[df[day_col].isin(test_days)],
    )


# =============================================================================
# Save/Load Features
# =============================================================================

def save_feature_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    feature_info: Optional[FeatureInfo] = None,
    normalization_stats: Optional[NormalizationStats] = None,
) -> None:
    """Save train/val/test splits to Parquet.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Output directory
        feature_info: Feature metadata to save
        normalization_stats: Z-score stats (mean/std) for numerical features
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert Arrow-backed dictionary / pandas category columns to plain types
    # to avoid "dictionary<values=..., indices=...>[pyarrow]" errors on read
    def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        import pyarrow as pa

        cols_to_fix = {}
        for col in df.columns:
            dtype = df[col].dtype
            # pandas CategoricalDtype
            if hasattr(dtype, "name") and dtype.name == "category":
                cols_to_fix[col] = str
            # Arrow-backed dictionary dtype (e.g., from Hive partitioning)
            elif isinstance(dtype, pd.ArrowDtype) and pa.types.is_dictionary(dtype.pyarrow_dtype):
                value_type = dtype.pyarrow_dtype.value_type
                if pa.types.is_integer(value_type):
                    cols_to_fix[col] = value_type.to_pandas_dtype()
                else:
                    cols_to_fix[col] = str
        if cols_to_fix:
            df = df.copy()
            for col, target in cols_to_fix.items():
                df[col] = df[col].astype(target)
        return df

    # Save splits
    _normalize_dtypes(train_df).to_parquet(output_dir / "train.parquet", index=False)
    _normalize_dtypes(val_df).to_parquet(output_dir / "val.parquet", index=False)
    _normalize_dtypes(test_df).to_parquet(output_dir / "test.parquet", index=False)

    # Save metadata
    metadata = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "columns": list(train_df.columns),
    }

    if feature_info:
        metadata["feature_info"] = {
            "categorical": feature_info.categorical,
            "numerical": feature_info.numerical,
            "total_count": feature_info.total_count,
            "feature_groups": feature_info.feature_groups,
        }

    if normalization_stats:
        metadata["normalization_stats"] = {
            "mean": normalization_stats.mean,
            "std": normalization_stats.std,
        }

    with open(output_dir / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _to_numpy_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ArrowDtype / pandas nullable columns to standard numpy dtypes.

    LightGBM requires int, float, or bool — not ArrowDtype or object.
    Handles NAs in integer columns by filling with 0.
    Skips string/object arrow columns.
    """
    for col in df.columns:
        dtype = df[col].dtype
        if not hasattr(dtype, "numpy_dtype"):
            continue
        np_dt = dtype.numpy_dtype
        if np_dt.kind in ("i", "u"):  # integer → fillna then cast
            df[col] = df[col].fillna(0).astype(np_dt)
        elif np_dt.kind == "f":  # float → NaN preserved
            df[col] = df[col].astype(np_dt)
    return df


def load_feature_splits(
    data_dir: Path,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Load train/val/test splits from Parquet.

    Args:
        data_dir: Directory containing split files
        columns: If provided, load only these columns from parquet
                 (saves memory by skipping string/unused columns)

    Returns:
        Tuple of (train_df, val_df, test_df, metadata)
    """
    import json

    data_dir = Path(data_dir)

    train_df = _to_numpy_dtypes(pd.read_parquet(data_dir / "train.parquet", columns=columns))
    val_df = _to_numpy_dtypes(pd.read_parquet(data_dir / "val.parquet", columns=columns))
    test_df = _to_numpy_dtypes(pd.read_parquet(data_dir / "test.parquet", columns=columns))

    # Encode object columns (e.g. slot_size_group, region_group) as integer codes
    # with consistent mapping across splits (union of unique values, no concat)
    object_cols = [c for c in train_df.columns if train_df[c].dtype == "object"]
    for col in object_cols:
        uniques = sorted(set(train_df[col].unique()) | set(val_df[col].unique()) | set(test_df[col].unique()))
        categories = pd.CategoricalDtype(categories=uniques)
        train_df[col] = train_df[col].astype(categories).cat.codes.astype("int32")
        val_df[col] = val_df[col].astype(categories).cat.codes.astype("int32")
        test_df[col] = test_df[col].astype(categories).cat.codes.astype("int32")

    with open(data_dir / "feature_metadata.json", "r") as f:
        metadata = json.load(f)

    return train_df, val_df, test_df, metadata
