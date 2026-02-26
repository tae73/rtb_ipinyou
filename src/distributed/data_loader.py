"""grain-based data loading for RTB training.

Replaces per-batch DataFrame.iloc[] → jnp.array() with pre-materialized numpy arrays
and grain DataLoader for deterministic shuffle + batching.

Key components:
    RTBDataSource — grain-compatible random access data source
    materialize_to_source — DataFrame → RTBDataSource (1-time conversion)
    create_train_loader — epoch-wise grain DataLoader (shuffle, shard, batch)
    create_eval_loader — eval DataLoader (no shuffle)
    batch_to_jax — grain numpy batch → JAX arrays with optional SPMD sharding
"""

from typing import Dict, List, Optional

import gc

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import grain.python as grain


class RTBDataSource(grain.RandomAccessDataSource):
    """grain-compatible random access data source for RTB features.

    Pre-materialized numpy arrays served via __len__ + __getitem__.
    __getitem__ returns a single sample dict; grain.Batch handles stacking.
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
            x[col] = arr[idx]  # int32 scalar
        for col, arr in self.num_arrays.items():
            x[col] = arr[idx]  # float32 scalar
        return {"x": x, "win": self.win[idx], "click": self.click[idx]}


def materialize_to_source(
    df: pd.DataFrame,
    cat_features: List[str],
    num_features: List[str],
    norm_mean: Dict[str, float],
    norm_std: Dict[str, float],
) -> RTBDataSource:
    """One-time DataFrame → contiguous numpy conversion.

    Z-score normalization applied here so grain pipeline doesn't repeat it per batch.

    Args:
        df: Feature DataFrame with cat/num columns + win/click labels.
        cat_features: Categorical feature column names.
        num_features: Numerical feature column names.
        norm_mean: Z-score means from training set.
        norm_std: Z-score stds from training set.

    Returns:
        RTBDataSource ready for grain DataLoader.
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
) -> grain.DataLoader:
    """Create epoch-wise grain DataLoader for training.

    Args:
        source: Pre-materialized RTBDataSource.
        batch_size: Global batch size (per_device x num_devices).
        seed: Base seed (epoch seed = seed + epoch for deterministic shuffle).
        epoch: Current epoch number.
        num_devices: SPMD device count (for ShardOptions).
        worker_count: 0 = no multiprocessing (optimal for in-memory numpy data).
        worker_buffer_size: Prefetch buffer size per worker.

    Returns:
        grain.DataLoader that yields batched numpy dicts.
    """
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=1,
        shard_options=grain.ShardOptions(
            shard_index=jax.process_index(),
            shard_count=jax.process_count(),
            drop_remainder=True,
        ),
        shuffle=True,
        seed=seed + epoch,
    )
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=True)],
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
    )


def create_eval_loader(
    source: RTBDataSource,
    batch_size: int,
    worker_count: int = 0,
) -> grain.DataLoader:
    """Create grain DataLoader for validation/test (no shuffle).

    Args:
        source: Pre-materialized RTBDataSource.
        batch_size: Batch size.
        worker_count: 0 = no multiprocessing.

    Returns:
        grain.DataLoader that yields batched numpy dicts (sequential order).
    """
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=1,
        shard_options=grain.NoSharding(),
        shuffle=False,
        seed=0,
    )
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=False)],
        worker_count=worker_count,
    )


def batch_to_jax(
    batch: dict,
    data_sharding=None,
) -> dict:
    """Convert grain numpy batch to JAX arrays with optional SPMD sharding.

    grain.Batch stacks individual samples into numpy arrays.
    This function converts them to jnp.array and optionally places
    on SPMD mesh via jax.device_put.

    Output format matches existing prepare_batch() contract:
        {"x": Dict[str, Array], "win": Array, "click": Array}

    Args:
        batch: Numpy dict from grain DataLoader.
        data_sharding: NamedSharding for SPMD (None = single device).

    Returns:
        JAX batch dict compatible with model loss functions.
    """
    x = {}
    for col, arr in batch["x"].items():
        a = jnp.array(arr)
        x[col] = jax.device_put(a, data_sharding) if data_sharding else a

    win = jnp.array(batch["win"], dtype=jnp.float32)
    click = jnp.array(batch["click"], dtype=jnp.float32)

    if data_sharding:
        win = jax.device_put(win, data_sharding)
        click = jax.device_put(click, data_sharding)

    return {"x": x, "win": win, "click": click}
