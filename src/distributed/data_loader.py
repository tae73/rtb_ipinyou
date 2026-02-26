"""Data loading for RTB training.

Provides vectorized numpy batch iteration (NumpyBatchIterator) that replaces
grain's per-sample __getitem__ + Python-level tree stacking with direct
numpy fancy indexing — yielding ~100x speedup on the data pipeline.

Key components:
    RTBDataSource — numpy array container for pre-materialized features
    materialize_to_source — DataFrame → RTBDataSource (1-time conversion)
    NumpyBatchIterator — vectorized batch iterator (replaces grain DataLoader)
    create_train_loader — epoch-wise NumpyBatchIterator (shuffle, deterministic)
    create_eval_loader — eval NumpyBatchIterator (no shuffle)
    batch_to_jax — numpy batch → JAX arrays with optional SPMD sharding
"""

from typing import Dict, Iterator, List, Optional

import gc

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

try:
    import grain.python as grain
    _GRAIN_AVAILABLE = True
except ImportError:
    _GRAIN_AVAILABLE = False


class RTBDataSource:
    """Numpy array container for RTB features.

    Pre-materialized numpy arrays. Used by NumpyBatchIterator for
    vectorized fancy indexing (no per-sample __getitem__ overhead).

    Also supports grain RandomAccessDataSource interface (__len__ + __getitem__)
    for backward compatibility if grain is available.
    """

    def __init__(
        self,
        cat_arrays: Dict[str, np.ndarray],
        num_arrays: Dict[str, np.ndarray],
        win: np.ndarray,
        click: np.ndarray,
    ):
        self.cat_arrays = cat_arrays  # {feature: int32 [N,]}
        self.num_arrays = num_arrays  # {feature: float32 [N,], Z-score applied}
        self.win = win                # float32 [N,]
        self.click = click            # float32 [N,]

    def __len__(self) -> int:
        return len(self.win)

    def __getitem__(self, idx) -> dict:
        x = {}
        for col, arr in self.cat_arrays.items():
            x[col] = arr[idx]
        for col, arr in self.num_arrays.items():
            x[col] = arr[idx]
        return {"x": x, "win": self.win[idx], "click": self.click[idx]}


class NumpyBatchIterator:
    """Vectorized numpy batch iterator.

    Replaces grain DataLoader's per-sample __getitem__ + Batch stacking
    with direct numpy fancy indexing: {col: arr[indices] for col, arr in ...}.

    This eliminates the ~950ms/step Python-level tree stacking overhead,
    reducing batch creation to <1ms via vectorized numpy operations.
    """

    def __init__(
        self,
        source: RTBDataSource,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_remainder: bool = True,
    ):
        self.source = source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_remainder = drop_remainder
        self._n = len(source)

    def __iter__(self) -> Iterator[dict]:
        rng = np.random.RandomState(self.seed)
        indices = rng.permutation(self._n) if self.shuffle else np.arange(self._n)

        bs = self.batch_size
        for start in range(0, self._n, bs):
            end = start + bs
            if end > self._n:
                if self.drop_remainder:
                    break
                end = self._n
            idx = indices[start:end]
            yield {
                "x": {col: arr[idx] for col, arr in self.source.cat_arrays.items()}
                   | {col: arr[idx] for col, arr in self.source.num_arrays.items()},
                "win": self.source.win[idx],
                "click": self.source.click[idx],
            }

    def __len__(self) -> int:
        if self.drop_remainder:
            return self._n // self.batch_size
        return (self._n + self.batch_size - 1) // self.batch_size


def materialize_to_source(
    df: pd.DataFrame,
    cat_features: List[str],
    num_features: List[str],
    norm_mean: Dict[str, float],
    norm_std: Dict[str, float],
) -> RTBDataSource:
    """One-time DataFrame → contiguous numpy conversion.

    Z-score normalization applied here so the pipeline doesn't repeat it per batch.

    Args:
        df: Feature DataFrame with cat/num columns + win/click labels.
        cat_features: Categorical feature column names.
        num_features: Numerical feature column names.
        norm_mean: Z-score means from training set.
        norm_std: Z-score stds from training set.

    Returns:
        RTBDataSource ready for NumpyBatchIterator.
    """
    cat_arrays = {
        col: df[col].values.astype(np.int32)
        for col in cat_features
    }

    num_arrays = {}
    for col in num_features:
        vals = df[col].values.astype(np.float32)
        if col in norm_mean:
            vals = (vals - norm_mean[col]) / norm_std[col]
        num_arrays[col] = vals

    win = df["win"].values.astype(np.float32)
    click = df["click"].values.astype(np.float32)

    return RTBDataSource(cat_arrays, num_arrays, win, click)


def create_train_loader(
    source: RTBDataSource,
    batch_size: int,
    seed: int,
    epoch: int,
    num_devices: int = 1,
    worker_count: int = 0,
    worker_buffer_size: int = 2,
) -> NumpyBatchIterator:
    """Create epoch-wise NumpyBatchIterator for training.

    Args:
        source: Pre-materialized RTBDataSource.
        batch_size: Global batch size (per_device x num_devices).
        seed: Base seed (epoch seed = seed + epoch for deterministic shuffle).
        epoch: Current epoch number.
        num_devices: Kept for API compatibility (unused by NumpyBatchIterator).
        worker_count: Kept for API compatibility (unused).
        worker_buffer_size: Kept for API compatibility (unused).

    Returns:
        NumpyBatchIterator that yields batched numpy dicts.
    """
    return NumpyBatchIterator(
        source=source,
        batch_size=batch_size,
        shuffle=True,
        seed=seed + epoch,
        drop_remainder=True,
    )


def create_eval_loader(
    source: RTBDataSource,
    batch_size: int,
    worker_count: int = 0,
) -> NumpyBatchIterator:
    """Create NumpyBatchIterator for validation/test (no shuffle).

    Args:
        source: Pre-materialized RTBDataSource.
        batch_size: Batch size.
        worker_count: Kept for API compatibility (unused).

    Returns:
        NumpyBatchIterator that yields batched numpy dicts (sequential order).
    """
    return NumpyBatchIterator(
        source=source,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        drop_remainder=False,
    )


def batch_to_jax(
    batch: dict,
    data_sharding=None,
) -> dict:
    """Convert numpy batch to JAX arrays with optional SPMD sharding.

    Uses jnp.asarray() for potential zero-copy transfer from contiguous numpy.

    Output format matches existing prepare_batch() contract:
        {"x": Dict[str, Array], "win": Array, "click": Array}

    Args:
        batch: Numpy dict from NumpyBatchIterator.
        data_sharding: NamedSharding for SPMD (None = single device).

    Returns:
        JAX batch dict compatible with model loss functions.
    """
    x = {}
    for col, arr in batch["x"].items():
        a = jnp.asarray(arr)
        x[col] = jax.device_put(a, data_sharding) if data_sharding else a

    win = jnp.asarray(batch["win"], dtype=jnp.float32)
    click = jnp.asarray(batch["click"], dtype=jnp.float32)

    if data_sharding:
        win = jax.device_put(win, data_sharding)
        click = jax.device_put(click, data_sharding)

    return {"x": x, "win": win, "click": click}
