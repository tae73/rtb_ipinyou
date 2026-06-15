"""Shared test scaffolding for the rtb_ipinyou test suite.

Provides deterministic, tiny, in-memory fixtures so downstream tests never
touch the real data parquet. All randomness is seeded.

Schema (feature_metadata.json):
  17 categorical: region, city, adexchange, slotwidth, slotheight,
                  slotvisibility, slotformat, advertiser, hour, minute,
                  weekday, is_weekend, is_peak_hour, slot_size_group,
                  region_group, domain_hash, creative_hash
  13 numerical:   slotprice, bidprice, hour_sin, hour_cos, slot_area,
                  slot_area_log, slot_aspect_ratio, region_freq,
                  bid_floor_ratio, domain_freq, domain_freq_log,
                  creative_freq, creative_freq_log
  labels:         win, click

JAX/flax imports are forced onto CPU (CUDA_VISIBLE_DEVICES='') at collection
time so the model factory fixtures run without GPU contention.
"""

import os

# Force CPU for any jax-importing test that uses the model fixtures, regardless
# of how pytest was invoked. Must run before jax is imported anywhere.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Schema constants
# =============================================================================

CAT_FEATURES: List[str] = [
    "region",
    "city",
    "adexchange",
    "slotwidth",
    "slotheight",
    "slotvisibility",
    "slotformat",
    "advertiser",
    "hour",
    "minute",
    "weekday",
    "is_weekend",
    "is_peak_hour",
    "slot_size_group",
    "region_group",
    "domain_hash",
    "creative_hash",
]

NUM_FEATURES: List[str] = [
    "slotprice",
    "bidprice",
    "hour_sin",
    "hour_cos",
    "slot_area",
    "slot_area_log",
    "slot_aspect_ratio",
    "region_freq",
    "bid_floor_ratio",
    "domain_freq",
    "domain_freq_log",
    "creative_freq",
    "creative_freq_log",
]

# Small, deterministic vocab sizes for each categorical column. Codes in
# tiny_df are drawn in [0, vocab_size) so they are valid embedding indices.
# (The model also clips indices to vocab internally, but we keep them valid.)
_CAT_VOCAB: Dict[str, int] = {
    "region": 35,
    "city": 40,
    "adexchange": 5,
    "slotwidth": 12,
    "slotheight": 10,
    "slotvisibility": 4,
    "slotformat": 3,
    "advertiser": 8,
    "hour": 24,
    "minute": 60,
    "weekday": 7,
    "is_weekend": 2,
    "is_peak_hour": 2,
    "slot_size_group": 6,
    "region_group": 5,
    "domain_hash": 50,
    "creative_hash": 50,
}

# Plausible (mean, std) ranges for each numerical column, loosely based on the
# real normalization_stats but downscaled so the synthetic frame is cheap.
_NUM_DIST: Dict[str, tuple] = {
    "slotprice": (46.8, 41.0),
    "bidprice": (272.3, 29.7),
    "hour_sin": (-0.2, 0.6),
    "hour_cos": (0.0, 0.77),
    "slot_area": (77474.0, 13446.0),
    "slot_area_log": (11.24, 0.22),
    "slot_aspect_ratio": (5.14, 4.56),
    "region_freq": (8.24e6, 7.41e6),
    "bid_floor_ratio": (25.96, 31.74),
    "domain_freq": (2.55e6, 3.32e6),
    "domain_freq_log": (10.76, 5.84),
    "creative_freq": (6.56e6, 5.58e6),
    "creative_freq_log": (15.18, 1.19),
}

N_ROWS = 64
SEED = 0


# =============================================================================
# Feature list fixtures
# =============================================================================


@pytest.fixture(scope="session")
def cat_features() -> List[str]:
    """The 17 categorical feature column names (ordered)."""
    return list(CAT_FEATURES)


@pytest.fixture(scope="session")
def num_features() -> List[str]:
    """The 13 numerical feature column names (ordered)."""
    return list(NUM_FEATURES)


@pytest.fixture(scope="session")
def feature_dims() -> Dict[str, int]:
    """feature_dims dict: column name -> vocab size.

    Convention (EmbeddingLayer): vocab_size > 1 => categorical (nnx.Embed);
    vocab_size <= 1 => numerical/dense (we use 1). Consistent with the codes
    present in ``tiny_df``.
    """
    dims: Dict[str, int] = {c: _CAT_VOCAB[c] for c in CAT_FEATURES}
    dims.update({n: 1 for n in NUM_FEATURES})
    return dims


# =============================================================================
# Normalization stats (numericals)
# =============================================================================


@pytest.fixture(scope="session")
def norm_mean() -> Dict[str, float]:
    """Per-numerical Z-score mean (matches tiny_df generating distribution)."""
    return {n: float(_NUM_DIST[n][0]) for n in NUM_FEATURES}


@pytest.fixture(scope="session")
def norm_std() -> Dict[str, float]:
    """Per-numerical Z-score std (all strictly positive)."""
    return {n: float(_NUM_DIST[n][1]) for n in NUM_FEATURES}


# =============================================================================
# Tiny synthetic DataFrame
# =============================================================================


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """A ~64-row deterministic DataFrame over the full 30-feature schema.

    - Categorical columns: small integer codes in ``[0, vocab_size)`` (int64).
    - Numerical columns: plausible float64 draws (downscaled real stats).
    - ``win``: ~40% positives (int8 0/1).
    - ``click``: a tiny positive count, only among winners (int8 0/1).

    Fully reproducible (fixed seed). Returns a fresh frame each call so tests
    can mutate without cross-contamination.
    """
    rng = np.random.default_rng(SEED)

    data: Dict[str, np.ndarray] = {}

    # Categorical codes: valid embedding indices in [0, vocab).
    for col in CAT_FEATURES:
        vocab = _CAT_VOCAB[col]
        data[col] = rng.integers(low=0, high=vocab, size=N_ROWS).astype(np.int64)

    # Numerical columns: Gaussian draws per the declared distribution.
    for col in NUM_FEATURES:
        mean, std = _NUM_DIST[col]
        data[col] = rng.normal(loc=mean, scale=std, size=N_ROWS).astype(np.float64)

    # win ~ 40% positive.
    win = (rng.random(N_ROWS) < 0.40).astype(np.int8)

    # click: positive only among winners; aim for a few positives overall.
    # Bernoulli(0.25) among winners -> with ~26 winners gives a handful of clicks,
    # guaranteed >= 1 by forcing the first winner (if any) to click.
    click = np.zeros(N_ROWS, dtype=np.int8)
    winner_idx = np.flatnonzero(win == 1)
    if winner_idx.size > 0:
        click_draw = rng.random(winner_idx.size) < 0.25
        click[winner_idx[click_draw]] = 1
        # Guarantee at least one positive click for stable downstream tests.
        if click.sum() == 0:
            click[winner_idx[0]] = 1

    data["win"] = win
    data["click"] = click

    # Column order: cats, nums, then labels.
    ordered = CAT_FEATURES + NUM_FEATURES + ["win", "click"]
    return pd.DataFrame(data)[ordered]


# =============================================================================
# Model factory fixtures
# =============================================================================


@pytest.fixture
def make_esmm(feature_dims):
    """Factory building a small, ready ESMMWC model.

    Returns a callable ``make(**overrides) -> ESMMWC`` so tests can tweak
    config fields. Defaults: embed_dim=8, hidden_dims=(16, 8),
    win_hidden_dims=(8, 4), rngs=nnx.Rngs(0).
    """
    from flax import nnx

    from src.models.esmm_wc import ESMMWC, ESMMWCConfig

    def _make(**overrides):
        cfg_kwargs = dict(
            feature_dims=feature_dims,
            embed_dim=8,
            hidden_dims=(16, 8),
            win_hidden_dims=(8, 4),
        )
        cfg_kwargs.update(overrides)
        config = ESMMWCConfig(**cfg_kwargs)
        return ESMMWC(config, rngs=nnx.Rngs(0))

    return _make


@pytest.fixture
def make_escm2(feature_dims):
    """Factory building a small, ready ESCM2WC model.

    Returns a callable ``make(**overrides) -> ESCM2WC``. Defaults:
    embed_dim=8, hidden_dims=(16, 8), win_hidden_dims=(8, 4),
    loss_type='dr', rngs=nnx.Rngs(0).
    """
    from flax import nnx

    from src.models.escm2_wc import ESCM2WC, ESCM2WCConfig

    def _make(**overrides):
        cfg_kwargs = dict(
            feature_dims=feature_dims,
            embed_dim=8,
            hidden_dims=(16, 8),
            win_hidden_dims=(8, 4),
            loss_type="dr",
        )
        cfg_kwargs.update(overrides)
        config = ESCM2WCConfig(**cfg_kwargs)
        return ESCM2WC(config, rngs=nnx.Rngs(0))

    return _make
