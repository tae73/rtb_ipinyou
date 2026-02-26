"""Usertag encoding for RTB prediction models.

iPinYou usertags are comma-separated integer IDs representing user interests.
This module provides encoding strategies for these multi-valued categorical features.

Encoding strategies:
- Top-N multi-hot: Binary indicators for top N most frequent tags
- Hashing: Feature hashing for dimensionality reduction
- Embedding lookup: For neural network models

Parallel Processing:
- encode_multihot_parallel(): Batch parallelism with Ray
- build_vocab_parallel(): Parallel vocabulary building with map-reduce
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple, Set
import logging
import numpy as np
import pandas as pd
from collections import Counter

logger = logging.getLogger(__name__)


# =============================================================================
# Result Types
# =============================================================================

class UsertagVocab(NamedTuple):
    """Vocabulary for usertag encoding."""
    tag_to_idx: Dict[int, int]
    idx_to_tag: Dict[int, int]
    tag_counts: Dict[int, int]
    n_tags: int
    unk_idx: int  # Index for unknown tags


class UsertagEncodingResult(NamedTuple):
    """Result of usertag encoding."""
    encoded: np.ndarray  # Shape: (n_samples, n_tags)
    vocab: UsertagVocab
    coverage: float  # Fraction of tags covered by vocab


# =============================================================================
# Parsing Functions
# =============================================================================

def parse_usertag(usertag_str: str) -> List[int]:
    """Parse usertag string into list of integer tag IDs.

    Args:
        usertag_str: Comma-separated tag IDs (e.g., "10006,10024,10031")

    Returns:
        List of integer tag IDs (empty list if invalid)
    """
    if pd.isna(usertag_str) or usertag_str in ("null", "NULL", "", "nan"):
        return []

    try:
        return [int(t.strip()) for t in str(usertag_str).split(",") if t.strip()]
    except (ValueError, AttributeError):
        return []


def parse_usertag_series(series: pd.Series) -> pd.Series:
    """Parse usertag column into lists of tag IDs.

    Args:
        series: Pandas Series with usertag strings

    Returns:
        Series of lists of tag IDs
    """
    return series.apply(parse_usertag)


class ParseResult(NamedTuple):
    """Result of parallel usertag parsing."""
    tag_lists: pd.Series   # Series of List[int]
    n_tags: pd.Series      # len of each list
    has_tags: pd.Series    # 0/1 indicator


def parse_usertag_series_parallel(
    series: pd.Series,
    n_partitions: Optional[int] = None,
) -> ParseResult:
    """Parse usertag column into lists of tag IDs in parallel using Ray.

    Splits the Series into partitions, applies parse_usertag per chunk
    via Ray remote, then concatenates results.

    Falls back to sequential if Ray is unavailable or series is small.

    Args:
        series: Pandas Series with usertag strings
        n_partitions: Number of partitions (default: num_cpus)

    Returns:
        ParseResult with tag_lists, n_tags, has_tags
    """
    try:
        from src.ray_utils import RAY_AVAILABLE, init_ray, get_num_cpus
    except ImportError:
        RAY_AVAILABLE = False

    if not RAY_AVAILABLE or len(series) < 100_000:
        logger.info(f"Parsing {len(series):,} usertags sequentially")
        tag_lists = series.apply(parse_usertag)
        n_tags = tag_lists.apply(len)
        has_tags = (n_tags > 0).astype(int)
        return ParseResult(tag_lists=tag_lists, n_tags=n_tags, has_tags=has_tags)

    import ray

    if not ray.is_initialized():
        init_ray()

    if n_partitions is None:
        n_partitions = get_num_cpus()

    logger.info(
        f"Parsing {len(series):,} usertags in parallel "
        f"({n_partitions} partitions)"
    )

    partitions = np.array_split(series, n_partitions)

    @ray.remote
    def _parse_partition(partition: pd.Series) -> pd.Series:
        return partition.apply(parse_usertag)

    futures = [_parse_partition.remote(p) for p in partitions if len(p) > 0]
    results = ray.get(futures)
    tag_lists = pd.concat(results)

    n_tags = tag_lists.apply(len)
    has_tags = (n_tags > 0).astype(int)

    return ParseResult(tag_lists=tag_lists, n_tags=n_tags, has_tags=has_tags)


# =============================================================================
# Vocabulary Building
# =============================================================================

def build_vocab(
    usertags: pd.Series,
    top_n: int = 100,
    min_count: int = 10,
) -> UsertagVocab:
    """Build vocabulary from usertag column.

    Selects top N most frequent tags that appear at least min_count times.

    Args:
        usertags: Series of usertag strings
        top_n: Maximum number of tags to include
        min_count: Minimum occurrence count for a tag

    Returns:
        UsertagVocab with mappings
    """
    # Parse and count all tags
    all_tags: Counter = Counter()
    for tags_str in usertags:
        tags = parse_usertag(tags_str)
        all_tags.update(tags)

    # Filter by min_count and select top N
    filtered_tags = [
        (tag, count)
        for tag, count in all_tags.most_common()
        if count >= min_count
    ][:top_n]

    # Create mappings (reserve index 0 for unknown)
    tag_to_idx = {tag: idx + 1 for idx, (tag, _) in enumerate(filtered_tags)}
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    tag_counts = {tag: count for tag, count in filtered_tags}

    return UsertagVocab(
        tag_to_idx=tag_to_idx,
        idx_to_tag=idx_to_tag,
        tag_counts=tag_counts,
        n_tags=len(tag_to_idx) + 1,  # +1 for unknown
        unk_idx=0,
    )


def save_vocab(vocab: UsertagVocab, path: Path) -> None:
    """Save vocabulary to JSON file.

    Args:
        vocab: UsertagVocab to save
        path: Output path
    """
    import json

    data = {
        "tag_to_idx": {str(k): v for k, v in vocab.tag_to_idx.items()},
        "tag_counts": {str(k): v for k, v in vocab.tag_counts.items()},
        "n_tags": vocab.n_tags,
        "unk_idx": vocab.unk_idx,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_vocab(path: Path) -> UsertagVocab:
    """Load vocabulary from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        UsertagVocab
    """
    import json

    with open(path, "r") as f:
        data = json.load(f)

    tag_to_idx = {int(k): v for k, v in data["tag_to_idx"].items()}
    idx_to_tag = {v: k for k, v in tag_to_idx.items()}
    tag_counts = {int(k): v for k, v in data["tag_counts"].items()}

    return UsertagVocab(
        tag_to_idx=tag_to_idx,
        idx_to_tag=idx_to_tag,
        tag_counts=tag_counts,
        n_tags=data["n_tags"],
        unk_idx=data["unk_idx"],
    )


# =============================================================================
# Encoding Functions
# =============================================================================

def encode_multihot(
    usertags: pd.Series,
    vocab: UsertagVocab,
) -> np.ndarray:
    """Encode usertags as multi-hot vectors.

    Args:
        usertags: Series of usertag strings
        vocab: Vocabulary for encoding

    Returns:
        Array of shape (n_samples, vocab.n_tags) with binary indicators
    """
    n_samples = len(usertags)
    encoded = np.zeros((n_samples, vocab.n_tags), dtype=np.float32)

    for i, tags_str in enumerate(usertags):
        tags = parse_usertag(tags_str)
        for tag in tags:
            idx = vocab.tag_to_idx.get(tag, vocab.unk_idx)
            encoded[i, idx] = 1.0

    return encoded


def encode_multihot_sparse(
    usertags: pd.Series,
    vocab: UsertagVocab,
) -> "scipy.sparse.csr_matrix":
    """Encode usertags as sparse multi-hot matrix.

    More memory efficient for large datasets with high-dimensional encoding.

    Args:
        usertags: Series of usertag strings
        vocab: Vocabulary for encoding

    Returns:
        Sparse CSR matrix of shape (n_samples, vocab.n_tags)
    """
    from scipy.sparse import lil_matrix, csr_matrix

    n_samples = len(usertags)
    encoded = lil_matrix((n_samples, vocab.n_tags), dtype=np.float32)

    for i, tags_str in enumerate(usertags):
        tags = parse_usertag(tags_str)
        for tag in tags:
            idx = vocab.tag_to_idx.get(tag, vocab.unk_idx)
            encoded[i, idx] = 1.0

    return csr_matrix(encoded)


def encode_count(
    usertags: pd.Series,
    vocab: UsertagVocab,
    normalize: bool = True,
) -> np.ndarray:
    """Encode usertags as count vectors (allows repeated tags).

    Args:
        usertags: Series of usertag strings
        vocab: Vocabulary for encoding
        normalize: Normalize to sum to 1

    Returns:
        Array of shape (n_samples, vocab.n_tags) with counts
    """
    n_samples = len(usertags)
    encoded = np.zeros((n_samples, vocab.n_tags), dtype=np.float32)

    for i, tags_str in enumerate(usertags):
        tags = parse_usertag(tags_str)
        for tag in tags:
            idx = vocab.tag_to_idx.get(tag, vocab.unk_idx)
            encoded[i, idx] += 1.0

        if normalize and len(tags) > 0:
            encoded[i] /= len(tags)

    return encoded


def encode_hashing(
    usertags: pd.Series,
    n_features: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Encode usertags using feature hashing.

    Hashing trick for dimensionality reduction without vocabulary.

    Args:
        usertags: Series of usertag strings
        n_features: Number of hash buckets
        seed: Random seed for hashing

    Returns:
        Array of shape (n_samples, n_features) with hashed features
    """
    np.random.seed(seed)
    n_samples = len(usertags)
    encoded = np.zeros((n_samples, n_features), dtype=np.float32)

    for i, tags_str in enumerate(usertags):
        tags = parse_usertag(tags_str)
        for tag in tags:
            # Hash to bucket
            h = hash(str(tag) + str(seed)) % n_features
            # Use sign hashing for better properties
            sign = 1 if (hash(str(tag) + "sign" + str(seed)) % 2) == 0 else -1
            encoded[i, h] += sign

    return encoded


# =============================================================================
# Feature Column Generation
# =============================================================================

def add_usertag_features(
    df: pd.DataFrame,
    vocab: UsertagVocab,
    usertag_col: str = "usertag",
    prefix: str = "tag_",
    encoding: str = "multihot",
) -> pd.DataFrame:
    """Add usertag features as columns to DataFrame.

    Args:
        df: DataFrame with usertag column
        vocab: Vocabulary for encoding
        usertag_col: Name of usertag column
        prefix: Prefix for feature column names
        encoding: Encoding type ('multihot', 'count', 'hashing')

    Returns:
        DataFrame with added usertag feature columns
    """
    if encoding == "multihot":
        encoded = encode_multihot(df[usertag_col], vocab)
    elif encoding == "count":
        encoded = encode_count(df[usertag_col], vocab)
    elif encoding == "hashing":
        encoded = encode_hashing(df[usertag_col], n_features=vocab.n_tags - 1)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")

    # Create column names
    col_names = [f"{prefix}{i}" for i in range(encoded.shape[1])]

    # Add to DataFrame
    tag_df = pd.DataFrame(encoded, columns=col_names, index=df.index)

    return pd.concat([df, tag_df], axis=1)



# =============================================================================
# Analysis Functions
# =============================================================================

def compute_tag_stats(usertags: pd.Series) -> pd.DataFrame:
    """Compute statistics about usertag distribution.

    Args:
        usertags: Series of usertag strings

    Returns:
        DataFrame with tag statistics
    """
    all_tags: Counter = Counter()
    n_tags_per_sample = []

    for tags_str in usertags:
        tags = parse_usertag(tags_str)
        all_tags.update(tags)
        n_tags_per_sample.append(len(tags))

    # Top tags
    top_tags = pd.DataFrame(
        all_tags.most_common(100),
        columns=["tag_id", "count"],
    )
    top_tags["pct"] = top_tags["count"] / len(usertags) * 100

    return top_tags


def compute_tag_stats_parallel(
    usertags: pd.Series,
    top_n: int = 100,
    n_partitions: Optional[int] = None,
) -> pd.DataFrame:
    """Compute tag statistics in parallel using Ray map-reduce.

    Each partition counts tags locally via Counter, then results are merged.
    Falls back to sequential if Ray is unavailable.

    Args:
        usertags: Series of usertag strings
        top_n: Number of top tags to return
        n_partitions: Number of partitions (default: num_cpus)

    Returns:
        DataFrame with tag_id, count, pct columns
    """
    try:
        from src.ray_utils import RAY_AVAILABLE, map_reduce, init_ray
    except ImportError:
        RAY_AVAILABLE = False

    if not RAY_AVAILABLE or len(usertags) < 100_000:
        return compute_tag_stats(usertags)

    logger.info(f"Computing tag stats in parallel from {len(usertags):,} samples")

    n_total = len(usertags)

    def map_fn(partition: pd.DataFrame) -> Dict:
        counts: Counter = Counter()
        for tags_str in partition["usertag"]:
            tags = parse_usertag(tags_str)
            counts.update(tags)
        return dict(counts)

    def reduce_fn(results: List[Dict]) -> Dict:
        merged: Counter = Counter()
        for counts in results:
            merged.update(counts)
        return dict(merged)

    df_wrap = pd.DataFrame({"usertag": usertags})
    all_tags = map_reduce(df_wrap, map_fn, reduce_fn, n_partitions)

    top_tags = pd.DataFrame(
        sorted(all_tags.items(), key=lambda x: -x[1])[:top_n],
        columns=["tag_id", "count"],
    )
    top_tags["pct"] = top_tags["count"] / n_total * 100

    return top_tags


def compute_coverage(
    usertags: pd.Series,
    vocab: UsertagVocab,
) -> float:
    """Compute vocabulary coverage over dataset.

    Args:
        usertags: Series of usertag strings
        vocab: Vocabulary

    Returns:
        Fraction of tag occurrences covered by vocabulary
    """
    total_tags = 0
    covered_tags = 0

    for tags_str in usertags:
        tags = parse_usertag(tags_str)
        total_tags += len(tags)
        covered_tags += sum(1 for t in tags if t in vocab.tag_to_idx)

    return covered_tags / total_tags if total_tags > 0 else 0.0


# =============================================================================
# Parallel Encoding Functions
# =============================================================================

def encode_multihot_parallel(
    usertags: pd.Series,
    vocab: UsertagVocab,
    batch_size: int = 10000,
    n_workers: Optional[int] = None,
) -> np.ndarray:
    """Encode usertags as multi-hot vectors in parallel.

    Uses Ray for batch-level parallelism. The vocabulary is placed in
    the object store to avoid serialization overhead.

    Falls back to sequential encoding if Ray is unavailable.

    Args:
        usertags: Series of usertag strings
        vocab: Vocabulary for encoding
        batch_size: Number of samples per batch
        n_workers: Number of parallel workers (default: auto)

    Returns:
        Array of shape (n_samples, vocab.n_tags) with binary indicators
    """
    n_samples = len(usertags)

    # Check Ray availability
    try:
        from src.ray_utils import RAY_AVAILABLE, batch_encode, init_ray
        import ray
    except ImportError:
        RAY_AVAILABLE = False

    if not RAY_AVAILABLE or n_samples <= batch_size:
        logger.info(f"Encoding {n_samples:,} usertags sequentially")
        return encode_multihot(usertags, vocab)

    logger.info(f"Encoding {n_samples:,} usertags in parallel (batch_size={batch_size:,})")

    if not ray.is_initialized():
        init_ray()

    # Convert Series to list for batching
    usertag_list = usertags.tolist()

    # Put vocab in object store
    vocab_ref = ray.put(vocab)

    # Create batches
    batches = [
        usertag_list[i:i + batch_size]
        for i in range(0, n_samples, batch_size)
    ]

    @ray.remote
    def _encode_batch(batch: List[str], vocab_ref) -> np.ndarray:
        vocab = ray.get(vocab_ref)
        n_batch = len(batch)
        encoded = np.zeros((n_batch, vocab.n_tags), dtype=np.float32)

        for i, tags_str in enumerate(batch):
            tags = parse_usertag(tags_str)
            for tag in tags:
                idx = vocab.tag_to_idx.get(tag, vocab.unk_idx)
                encoded[i, idx] = 1.0

        return encoded

    # Process batches in parallel
    futures = [_encode_batch.remote(batch, vocab_ref) for batch in batches]
    results = ray.get(futures)

    return np.vstack(results)


def build_vocab_parallel(
    usertags: pd.Series,
    top_n: int = 100,
    min_count: int = 10,
    n_partitions: Optional[int] = None,
) -> UsertagVocab:
    """Build vocabulary from usertag column in parallel using map-reduce.

    Uses Ray for partition-level parallelism. Each partition counts tags
    locally, then results are merged.

    Falls back to sequential building if Ray is unavailable.

    Args:
        usertags: Series of usertag strings
        top_n: Maximum number of tags to include
        min_count: Minimum occurrence count for a tag
        n_partitions: Number of partitions (default: auto)

    Returns:
        UsertagVocab with mappings
    """
    # Check Ray availability
    try:
        from src.ray_utils import RAY_AVAILABLE, map_reduce, init_ray
    except ImportError:
        RAY_AVAILABLE = False

    if not RAY_AVAILABLE:
        return build_vocab(usertags, top_n, min_count)

    logger.info(f"Building vocabulary in parallel from {len(usertags):,} samples")

    # Map function: count tags in partition
    def map_fn(partition: pd.DataFrame) -> Dict:
        counts: Counter = Counter()
        for tags_str in partition["usertag"]:
            tags = parse_usertag(tags_str)
            counts.update(tags)
        return dict(counts)

    # Reduce function: merge all counts
    def reduce_fn(results: List[Dict]) -> Dict:
        merged: Counter = Counter()
        for counts in results:
            merged.update(counts)
        return dict(merged)

    # Create DataFrame for map_reduce (expects DataFrame input)
    df = pd.DataFrame({"usertag": usertags})

    # Run map-reduce
    all_tags = map_reduce(df, map_fn, reduce_fn, n_partitions)

    # Filter by min_count and select top N
    filtered_tags = [
        (tag, count)
        for tag, count in sorted(all_tags.items(), key=lambda x: -x[1])
        if count >= min_count
    ][:top_n]

    # Create mappings (reserve index 0 for unknown)
    tag_to_idx = {tag: idx + 1 for idx, (tag, _) in enumerate(filtered_tags)}
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    tag_counts = {tag: count for tag, count in filtered_tags}

    logger.info(f"Built vocabulary with {len(tag_to_idx)} tags")

    return UsertagVocab(
        tag_to_idx=tag_to_idx,
        idx_to_tag=idx_to_tag,
        tag_counts=tag_counts,
        n_tags=len(tag_to_idx) + 1,  # +1 for unknown
        unk_idx=0,
    )
