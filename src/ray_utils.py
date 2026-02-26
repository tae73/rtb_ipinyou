"""Ray distributed processing utilities for RTB data pipeline.

This module provides:
- Ray cluster initialization and management
- Parallel file processing for log parsing
- Batch processing utilities (usertag encoding, map-reduce)

All functions gracefully fall back to sequential processing if Ray is unavailable.

Example:
    >>> from src.ray_utils import init_ray, parallel_files
    >>> init_ray(num_cpus=8)
    >>> dfs = parallel_files(files, parse_fn)
"""

from typing import NamedTuple, Optional, List, Callable, Any, Dict, TypeVar
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar("T")

# Check Ray availability
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray not available. Install with: pip install ray")


# =============================================================================
# Configuration Types
# =============================================================================

class RayConfig(NamedTuple):
    """Configuration for Ray cluster."""
    num_cpus: Optional[int] = None  # None = use all available
    num_gpus: int = 0
    memory: Optional[int] = None  # Bytes
    object_store_memory: Optional[int] = None
    local_mode: bool = False  # Debug mode (single-threaded)
    ignore_reinit_error: bool = True


class ParallelResult(NamedTuple):
    """Generic result container for parallel operations."""
    results: List[Any]
    n_tasks: int
    n_success: int
    n_failed: int
    errors: List[str]


# =============================================================================
# Ray Initialization
# =============================================================================

def init_ray(
    config: Optional[RayConfig] = None,
    num_cpus: Optional[int] = None,
    **kwargs,
) -> bool:
    """Initialize Ray cluster.

    Args:
        config: RayConfig object
        num_cpus: Number of CPUs (convenience parameter)
        **kwargs: Additional Ray init arguments

    Returns:
        True if initialized successfully
    """
    if not RAY_AVAILABLE:
        logger.error("Ray is not installed")
        return False

    if config is None:
        config = RayConfig(num_cpus=num_cpus)

    # Check if already initialized
    if ray.is_initialized():
        if config.ignore_reinit_error:
            logger.info("Ray already initialized")
            return True
        else:
            raise RuntimeError("Ray already initialized")

    init_kwargs = {
        "num_cpus": config.num_cpus,
        "num_gpus": config.num_gpus,
        "local_mode": config.local_mode,
        "ignore_reinit_error": config.ignore_reinit_error,
    }

    if config.memory is not None:
        init_kwargs["_memory"] = config.memory

    if config.object_store_memory is not None:
        init_kwargs["object_store_memory"] = config.object_store_memory

    init_kwargs.update(kwargs)

    ray.init(**init_kwargs)

    resources = ray.cluster_resources()
    logger.info(
        f"Ray initialized: {resources.get('CPU', 0):.0f} CPUs, "
        f"{resources.get('GPU', 0):.0f} GPUs"
    )
    return True


def shutdown_ray() -> None:
    """Shutdown Ray cluster."""
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()
        logger.info("Ray shutdown complete")


def get_ray_resources() -> Dict:
    """Get current Ray cluster resources."""
    if not RAY_AVAILABLE or not ray.is_initialized():
        return {}
    return dict(ray.cluster_resources())


def get_num_cpus() -> int:
    """Get number of available CPUs.

    Returns:
        Number of CPUs from Ray if initialized, else os.cpu_count()
    """
    if RAY_AVAILABLE and ray.is_initialized():
        return int(ray.cluster_resources().get("CPU", 4))

    import os
    return os.cpu_count() or 4


# =============================================================================
# Parallel File Processing
# =============================================================================

def parallel_files(
    files: List[Path],
    parse_fn: Callable[[Path], pd.DataFrame],
    n_workers: Optional[int] = None,
    verbose: bool = True,
) -> List[pd.DataFrame]:
    """Parse multiple files in parallel.

    Args:
        files: List of file paths to process
        parse_fn: Function to parse each file (Path -> DataFrame)
        n_workers: Number of parallel workers (default: num_cpus)
        verbose: Print progress

    Returns:
        List of DataFrames from each file
    """
    if len(files) == 0:
        return []

    if len(files) == 1 or not RAY_AVAILABLE:
        # Sequential fallback
        if verbose:
            logger.info(f"Processing {len(files)} files sequentially")
        return [parse_fn(f) for f in files]

    if not ray.is_initialized():
        init_ray()

    if n_workers is None:
        n_workers = get_num_cpus()

    if verbose:
        logger.info(f"Processing {len(files)} files in parallel ({n_workers} workers)")

    @ray.remote
    def _parse_file(filepath: Path) -> pd.DataFrame:
        return parse_fn(filepath)

    # Submit all tasks
    futures = [_parse_file.remote(f) for f in files]

    # Gather results
    results = ray.get(futures)

    return results


def parallel_files_concat(
    files: List[Path],
    parse_fn: Callable[[Path], pd.DataFrame],
    n_workers: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Parse multiple files in parallel and concatenate.

    Args:
        files: List of file paths to process
        parse_fn: Function to parse each file (Path -> DataFrame)
        n_workers: Number of parallel workers
        verbose: Print progress

    Returns:
        Concatenated DataFrame from all files
    """
    dfs = parallel_files(files, parse_fn, n_workers, verbose)

    if len(dfs) == 0:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Batch Processing Utilities
# =============================================================================

def batch_process(
    items: List[T],
    process_fn: Callable[[T], Any],
    batch_size: int = 100,
) -> List[Any]:
    """Process items in batches using Ray.

    Args:
        items: List of items to process
        process_fn: Function to apply to each item
        batch_size: Number of items per batch

    Returns:
        List of results
    """
    if not RAY_AVAILABLE:
        return [process_fn(item) for item in items]

    if not ray.is_initialized():
        init_ray()

    @ray.remote
    def _process_batch(batch: List[T]) -> List[Any]:
        return [process_fn(item) for item in batch]

    # Create batches
    batches = [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]

    futures = [_process_batch.remote(batch) for batch in batches]
    batch_results = ray.get(futures)

    # Flatten results
    return [item for batch in batch_results for item in batch]


def batch_encode(
    data: List[Any],
    encode_fn: Callable[[List[Any]], np.ndarray],
    batch_size: int = 10000,
    shared_data: Optional[Any] = None,
) -> np.ndarray:
    """Encode data in parallel batches.

    Useful for encoding large arrays where each batch produces a numpy array.

    Args:
        data: List of items to encode
        encode_fn: Function that encodes a batch: (batch, shared) -> np.ndarray
        batch_size: Number of items per batch
        shared_data: Shared data (e.g., vocabulary) passed to encode_fn

    Returns:
        Vertically stacked numpy array
    """
    if not RAY_AVAILABLE or len(data) <= batch_size:
        # Sequential fallback
        return encode_fn(data, shared_data)

    if not ray.is_initialized():
        init_ray()

    # Put shared data in object store
    shared_ref = ray.put(shared_data) if shared_data is not None else None

    # Create batches
    batches = [
        data[i:i + batch_size]
        for i in range(0, len(data), batch_size)
    ]

    @ray.remote
    def _encode_batch(batch: List[Any], shared_ref) -> np.ndarray:
        shared = ray.get(shared_ref) if shared_ref is not None else None
        return encode_fn(batch, shared)

    futures = [_encode_batch.remote(b, shared_ref) for b in batches]
    batch_results = ray.get(futures)

    return np.vstack(batch_results)


# =============================================================================
# Map-Reduce Pattern
# =============================================================================

def map_reduce(
    data: pd.DataFrame,
    map_fn: Callable[[pd.DataFrame], Dict],
    reduce_fn: Callable[[List[Dict]], Dict],
    n_partitions: Optional[int] = None,
) -> Dict:
    """Map-reduce pattern for DataFrame processing.

    Args:
        data: Input DataFrame
        map_fn: Function applied to each partition, returns dict
        reduce_fn: Function to combine results from all partitions
        n_partitions: Number of partitions

    Returns:
        Combined result dict
    """
    if not RAY_AVAILABLE:
        return reduce_fn([map_fn(data)])

    if not ray.is_initialized():
        init_ray()

    if n_partitions is None:
        n_partitions = get_num_cpus()

    partitions = np.array_split(data, n_partitions)

    @ray.remote
    def _map(partition: pd.DataFrame) -> Dict:
        return map_fn(partition)

    futures = [_map.remote(p) for p in partitions if len(p) > 0]
    map_results = ray.get(futures)

    return reduce_fn(map_results)
