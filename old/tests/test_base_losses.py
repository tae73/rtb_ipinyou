"""Tests for loss utilities in src.models.base.

Focus: the optional ``pos_weight`` (and ``focal_gamma``) parameters added to
``binary_cross_entropy`` to counter winners-CTR under-prediction at a ~0.1%
base rate.

Invariants asserted:
  (a) pos_weight=None reproduces the *exact* old unweighted BCE value;
  (b) pos_weight > 1 strictly increases the loss contribution of POSITIVES,
      while leaving the NEGATIVE term untouched;
  (c) focal_gamma=None is a no-op; focal_gamma > 0 down-weights easy examples;
  (d) the config field ``ctr_pos_weight`` actually threads into the ESMM-WC
      winners-CTR BCE term (so the retrain can use it).

CPU is forced via conftest (CUDA_VISIBLE_DEVICES='').
"""

import jax.numpy as jnp
import numpy as np
import pytest

from src.models.base import binary_cross_entropy


# =============================================================================
# (a) None reproduces the old unweighted value
# =============================================================================


def test_pos_weight_none_matches_old_bce():
    """pos_weight=None must reproduce the legacy formula exactly."""
    y_pred = jnp.array([0.01, 0.2, 0.5, 0.8, 0.99])
    y_true = jnp.array([0.0, 1.0, 0.0, 1.0, 1.0])

    got = binary_cross_entropy(y_pred, y_true)

    # Legacy reference (pre-change formula), recomputed independently.
    eps = 1e-7
    p = np.clip(np.asarray(y_pred), eps, 1.0 - eps)
    t = np.asarray(y_true)
    ref = -(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))

    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-6, atol=1e-7)


def test_focal_none_is_noop():
    """focal_gamma=None leaves BCE unchanged."""
    y_pred = jnp.array([0.05, 0.4, 0.95])
    y_true = jnp.array([1.0, 0.0, 1.0])

    base = binary_cross_entropy(y_pred, y_true)
    with_none = binary_cross_entropy(y_pred, y_true, focal_gamma=None)

    np.testing.assert_allclose(np.asarray(base), np.asarray(with_none))


# =============================================================================
# (b) pos_weight > 1 increases the POSITIVE contribution, not the negative one
# =============================================================================


def test_pos_weight_increases_positive_loss():
    """pos_weight>1 strictly scales up the loss on positive samples."""
    y_pred = jnp.array([0.1, 0.1])  # under-predicting the positive
    y_true = jnp.array([1.0, 1.0])

    w = 5.0
    base = binary_cross_entropy(y_pred, y_true)
    weighted = binary_cross_entropy(y_pred, y_true, pos_weight=w)

    # Weighted BCE on a pure-positive batch is exactly w * base.
    np.testing.assert_allclose(np.asarray(weighted), w * np.asarray(base), rtol=1e-6)
    assert float(weighted.sum()) > float(base.sum())


def test_pos_weight_leaves_negatives_unchanged():
    """pos_weight only touches the y=1 term; negatives are untouched."""
    y_pred = jnp.array([0.3, 0.7])
    y_true = jnp.array([0.0, 0.0])  # all negatives

    base = binary_cross_entropy(y_pred, y_true)
    weighted = binary_cross_entropy(y_pred, y_true, pos_weight=10.0)

    np.testing.assert_allclose(np.asarray(base), np.asarray(weighted), rtol=1e-6)


def test_pos_weight_mixed_batch_only_scales_positive_term():
    """On a mixed batch, only the positive rows differ by the weight factor."""
    y_pred = jnp.array([0.2, 0.6])
    y_true = jnp.array([1.0, 0.0])  # one positive, one negative

    w = 3.0
    base = np.asarray(binary_cross_entropy(y_pred, y_true))
    weighted = np.asarray(binary_cross_entropy(y_pred, y_true, pos_weight=w))

    # Positive row scaled by w; negative row identical.
    np.testing.assert_allclose(weighted[0], w * base[0], rtol=1e-6)
    np.testing.assert_allclose(weighted[1], base[1], rtol=1e-6)


# =============================================================================
# (c) focal_gamma down-weights easy (well-classified) examples
# =============================================================================


def test_focal_downweights_easy_positive():
    """A confidently-correct positive contributes less under focal modulation."""
    y_pred = jnp.array([0.99])  # easy positive (well classified)
    y_true = jnp.array([1.0])

    base = float(binary_cross_entropy(y_pred, y_true)[0])
    focal = float(binary_cross_entropy(y_pred, y_true, focal_gamma=2.0)[0])

    assert focal < base
    assert focal > 0.0


# =============================================================================
# (d) ctr_pos_weight config field threads into the ESMM-WC winners-CTR loss
# =============================================================================


def _esmm_ctr_loss(model, batch, ctr_pos_weight):
    """Recompute just the ESMM-WC components with a given ctr_pos_weight."""
    from src.models.esmm_wc import create_esmm_wc_loss_fn

    cfg = model.config._replace(ctr_pos_weight=ctr_pos_weight)
    loss_fn = create_esmm_wc_loss_fn(cfg, return_components=True, jit_safe=True)
    _total, comps = loss_fn(model, batch, training=False)
    return float(comps.ctr)


def test_esmm_ctr_pos_weight_threads_through(make_esmm, feature_dims):
    """ctr_pos_weight>1 raises the ESMM-WC winners-CTR loss vs None (default)."""
    import jax

    model = make_esmm()

    # Build a batch with winners that DID click (positive winners-CTR signal).
    key = jax.random.PRNGKey(1)
    keys = jax.random.split(key, len(feature_dims) + 2)
    B = 16
    x = {}
    for i, (name, vocab) in enumerate(feature_dims.items()):
        if vocab > 1:
            x[name] = jax.random.randint(keys[i], (B,), 0, vocab)
        else:
            x[name] = jax.random.normal(keys[i], (B,)).astype(jnp.float32)
    win = jnp.ones((B,), dtype=jnp.float32)  # all won -> ctr term is dense
    click = (jax.random.uniform(keys[-1], (B,)) < 0.5).astype(jnp.float32)
    batch = {"x": x, "win": win, "click": click}

    loss_none = _esmm_ctr_loss(model, batch, ctr_pos_weight=None)
    loss_w1 = _esmm_ctr_loss(model, batch, ctr_pos_weight=1.0)
    loss_w5 = _esmm_ctr_loss(model, batch, ctr_pos_weight=5.0)

    # pos_weight=1.0 is numerically the same as unweighted.
    assert loss_w1 == pytest.approx(loss_none, rel=1e-5)
    # pos_weight=5 up-weights positives -> larger CTR loss (batch has positives).
    assert click.sum() > 0
    assert loss_w5 > loss_none
