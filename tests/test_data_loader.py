"""Pins the CLAUDE.md batch-format INVARIANT for the data loader pipeline.

The upcoming streaming-loader refactor MUST preserve the batch dict contract:

    {"x": Dict[str, Array], "win": Array, "click": Array}   [+ optional ext_propensity]

where ``x`` is the FLAT union of all categorical + numerical features (no
cat/num nesting), and ``win``/``click`` are float32 of shape ``(batch_size,)``.

These tests exercise the full path:
    materialize_to_source -> NumpyBatchIterator -> batch_to_jax(data_sharding=None)

All synthetic, CPU-forced (via conftest), fixed-seed, <30s.
"""

import numpy as np
import pytest

from src.distributed.data_loader import (
    NumpyBatchIterator,
    batch_to_jax,
    materialize_to_source,
)

BATCH_SIZE = 16  # 64 rows / 16 -> 4 full batches, no remainder.


@pytest.fixture
def source(tiny_df, cat_features, num_features, norm_mean, norm_std):
    """RTBDataSource materialized from the tiny synthetic frame (Z-scored nums)."""
    return materialize_to_source(
        tiny_df,
        cat_features=cat_features,
        num_features=num_features,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )


# =============================================================================
# (a) batch-format invariant: keys, dtypes (float32), per-key shapes [B]
# =============================================================================


def test_batch_format_invariant_numpy_and_jax(
    source, cat_features, num_features
):
    """numpy batch and batch_to_jax(None) both satisfy the dict invariant.

    Top-level keys == {"x","win","click"}; win/click float32 shape (B,);
    cat keys int32, num keys float32, all shape (B,) under x.
    """
    it = NumpyBatchIterator(
        source, batch_size=BATCH_SIZE, shuffle=True, seed=0, drop_remainder=True
    )
    batch = next(iter(it))

    # --- numpy stage --------------------------------------------------------
    assert set(batch.keys()) == {"x", "win", "click"}
    assert isinstance(batch["x"], dict)

    assert batch["win"].shape == (BATCH_SIZE,)
    assert batch["click"].shape == (BATCH_SIZE,)
    assert batch["win"].dtype == np.float32
    assert batch["click"].dtype == np.float32

    for col in cat_features:
        assert batch["x"][col].shape == (BATCH_SIZE,)
        assert batch["x"][col].dtype == np.int32
    for col in num_features:
        assert batch["x"][col].shape == (BATCH_SIZE,)
        assert batch["x"][col].dtype == np.float32

    # --- jax stage (single device, data_sharding=None) ----------------------
    jb = batch_to_jax(batch, data_sharding=None)

    assert set(jb.keys()) == {"x", "win", "click"}
    assert set(jb["x"].keys()) == set(cat_features) | set(num_features)

    assert jb["win"].shape == (BATCH_SIZE,)
    assert jb["click"].shape == (BATCH_SIZE,)
    assert str(jb["win"].dtype) == "float32"
    assert str(jb["click"].dtype) == "float32"

    # x dtypes preserved through jnp.asarray: cat int32, num float32.
    for col in cat_features:
        assert jb["x"][col].shape == (BATCH_SIZE,)
        assert str(jb["x"][col].dtype) == "int32"
    for col in num_features:
        assert jb["x"][col].shape == (BATCH_SIZE,)
        assert str(jb["x"][col].dtype) == "float32"


# =============================================================================
# (b) every categorical/numerical feature appears as a flat key in x
# =============================================================================


def test_x_is_flat_union_of_all_features(source, cat_features, num_features):
    """x is the flat (un-nested) union of all 17 cat + 13 num feature keys."""
    expected = set(cat_features) | set(num_features)
    assert len(expected) == len(cat_features) + len(num_features)  # no overlap

    it = NumpyBatchIterator(
        source, batch_size=BATCH_SIZE, shuffle=False, seed=0, drop_remainder=True
    )
    batch = next(iter(it))

    assert set(batch["x"].keys()) == expected
    assert len(batch["x"]) == len(cat_features) + len(num_features)

    # No nested "cat"/"num" grouping leaked into x.
    for col in expected:
        assert not isinstance(batch["x"][col], dict)

    # Survives the jax conversion unchanged.
    jb = batch_to_jax(batch, data_sharding=None)
    assert set(jb["x"].keys()) == expected


# =============================================================================
# (c) iterating covers all rows exactly once (count identity)
# =============================================================================


def test_iteration_covers_all_rows_count_identity(tiny_df, source):
    """Eval-style loader (no shuffle, keep tail) visits every row exactly once.

    Uses a batch size that does NOT divide N to also exercise the tail batch.
    """
    n = len(tiny_df)
    bs = 10  # 64 = 6*10 + 4 -> 7 batches incl. a 4-row tail.

    it = NumpyBatchIterator(
        source, batch_size=bs, shuffle=False, seed=0, drop_remainder=False
    )

    seen_rows = 0
    region_concat = []
    for batch in it:
        b = batch["win"].shape[0]
        seen_rows += b
        # every per-key array in this batch shares the same length
        assert batch["click"].shape[0] == b
        for arr in batch["x"].values():
            assert arr.shape[0] == b
        region_concat.append(batch["x"]["region"])

    # count identity: total emitted == N
    assert seen_rows == n

    # Sequential (no shuffle) iteration reconstructs the original column order.
    recovered = np.concatenate(region_concat)
    assert recovered.shape[0] == n
    np.testing.assert_array_equal(
        recovered, tiny_df["region"].values.astype(np.int32)
    )

    # drop_remainder=True instead drops the 4-row tail -> 6 full batches only.
    it_drop = NumpyBatchIterator(
        source, batch_size=bs, shuffle=False, seed=0, drop_remainder=True
    )
    dropped_rows = sum(batch["win"].shape[0] for batch in it_drop)
    assert dropped_rows == (n // bs) * bs == 60
    assert len(it_drop) == n // bs == 6


# =============================================================================
# (d) Z-score normalization is applied to numerical columns
# =============================================================================


def test_zscore_normalization_applied(
    tiny_df, cat_features, num_features, norm_mean, norm_std
):
    """A numerical column comes out standardized as (raw - mean) / std.

    materialize_to_source applies Z-score ONCE (not per batch). We verify the
    exact transform against the known norm_mean/norm_std, and confirm the
    standardized values differ from the raw float32 values.
    """
    src = materialize_to_source(
        tiny_df,
        cat_features=cat_features,
        num_features=num_features,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )

    col = "slotprice"
    raw = tiny_df[col].values.astype(np.float32)
    expected = (raw - np.float32(norm_mean[col])) / np.float32(norm_std[col])

    got = src.num_arrays[col]
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    # Sanity: normalization actually changed the values (mean is far from 0,
    # std far from 1 for the raw distribution).
    assert not np.allclose(got, raw)

    # The standardized column's empirical mean/std are O(1), not the raw scale.
    assert abs(float(got.mean())) < 5.0
    assert float(got.std()) < 5.0

    # The same standardized values flow through the iterator + jax conversion.
    it = NumpyBatchIterator(
        src, batch_size=BATCH_SIZE, shuffle=False, seed=0, drop_remainder=True
    )
    batch = next(iter(it))
    np.testing.assert_allclose(
        batch["x"][col], expected[:BATCH_SIZE], rtol=1e-5, atol=1e-5
    )

    jb = batch_to_jax(batch, data_sharding=None)
    np.testing.assert_allclose(
        np.asarray(jb["x"][col]), expected[:BATCH_SIZE], rtol=1e-5, atol=1e-5
    )


# =============================================================================
# Determinism guard: same seed -> identical shuffle ordering (reusable iterator)
# =============================================================================


def test_iterator_is_deterministic_and_reusable(source):
    """A fresh RNG per __iter__ makes the shuffled order reproducible."""
    it = NumpyBatchIterator(
        source, batch_size=BATCH_SIZE, shuffle=True, seed=7, drop_remainder=True
    )
    first = np.concatenate([b["x"]["region"] for b in it])
    second = np.concatenate([b["x"]["region"] for b in it])  # re-iterate
    np.testing.assert_array_equal(first, second)
