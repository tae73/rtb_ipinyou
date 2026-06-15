"""Tests for ESMM-WC and ESCM²-WC models.

Covers, for BOTH ESMM and ESCM2WC:
  (a) forward pass on a 16-row batch -> documented output fields, shape [B];
  (b) ESMM constraint p_click_bid == p_win * p_ctr (allclose);
  (c) one train_step over ~5 steps reduces loss on a tiny separable synthetic
      task (loss finite and decreasing) — marked @pytest.mark.smoke;
  (d) ESCM2WC runs in BOTH loss_type 'dr' and 'ipw'.

All randomness is seeded; runs on CPU (conftest forces CUDA_VISIBLE_DEVICES='').
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from src.models.esmm_wc import (
    ESMMWCOutput,
    create_esmm_wc_train_step,
)
from src.models.escm2_wc import (
    ESCM2WCOutput,
    create_escm2wc_train_step,
)

B = 16


# =============================================================================
# Local helper fixtures
# =============================================================================


def _make_batch(feature_dims, *, separable: bool, seed: int = 0):
    """Build a {"x", "win", "click"} batch of size B.

    If ``separable`` is True, the labels are a deterministic function of one
    categorical feature ("region": even index -> 1) so that a small model can
    drive the loss down within a handful of steps. Otherwise labels are random.
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, len(feature_dims) + 2)

    x = {}
    for i, (name, vocab) in enumerate(feature_dims.items()):
        if vocab > 1:  # categorical -> integer indices
            x[name] = jax.random.randint(keys[i], (B,), 0, vocab)
        else:  # numerical -> float32
            x[name] = jax.random.normal(keys[i], (B,)).astype(jnp.float32)

    if separable:
        # Deterministic, learnable signal: region parity.
        signal = (x["region"] % 2 == 0).astype(jnp.float32)
        win = signal
        click = signal
    else:
        win = (jax.random.uniform(keys[-2], (B,)) < 0.5).astype(jnp.float32)
        click = (jax.random.uniform(keys[-1], (B,)) < 0.3).astype(jnp.float32)

    return {"x": x, "win": win, "click": click}


def _run_training(model, train_step, batch, n_steps: int = 5):
    """Run ``n_steps`` of training; return the list of per-step total losses."""
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    losses = []
    for _ in range(n_steps):
        loss, _components = train_step(model, optimizer, batch)
        losses.append(float(loss))
    return losses


# =============================================================================
# (a) Forward pass + output fields + shapes
# =============================================================================


def test_esmm_forward_shapes(make_esmm, feature_dims):
    """ESMM forward pass on a 16-row batch returns p_win/p_ctr/p_click_bid [B]."""
    model = make_esmm()
    batch = _make_batch(feature_dims, separable=False)
    out = model(batch["x"], training=False)

    assert isinstance(out, ESMMWCOutput)
    for field in ("p_win", "p_ctr", "p_click_bid"):
        arr = getattr(out, field)
        assert arr.shape == (B,), f"{field} shape {arr.shape} != ({B},)"
        assert jnp.all(jnp.isfinite(arr))
    # Probabilities in [0, 1].
    assert jnp.all((out.p_win >= 0) & (out.p_win <= 1))
    assert jnp.all((out.p_ctr >= 0) & (out.p_ctr <= 1))


def test_escm2_forward_shapes(make_escm2, feature_dims):
    """ESCM2WC forward returns p_win/p_ctr/p_click_bid/y_impute, each [B]."""
    model = make_escm2()
    batch = _make_batch(feature_dims, separable=False)
    out = model(batch["x"], training=False)

    assert isinstance(out, ESCM2WCOutput)
    for field in ("p_win", "p_ctr", "p_click_bid", "y_impute"):
        arr = getattr(out, field)
        assert arr.shape == (B,), f"{field} shape {arr.shape} != ({B},)"
        assert jnp.all(jnp.isfinite(arr))
    assert jnp.all((out.p_win >= 0) & (out.p_win <= 1))
    assert jnp.all((out.p_ctr >= 0) & (out.p_ctr <= 1))


# =============================================================================
# (b) ESMM constraint: p_click_bid == p_win * p_ctr
# =============================================================================


def test_esmm_constraint(make_esmm, feature_dims):
    """ESMM constraint holds: p_click_bid == p_win * p_ctr."""
    model = make_esmm()
    batch = _make_batch(feature_dims, separable=False)
    out = model(batch["x"], training=False)
    assert jnp.allclose(out.p_click_bid, out.p_win * out.p_ctr, atol=1e-5)


def test_escm2_constraint(make_escm2, feature_dims):
    """ESCM2WC ESMM constraint holds: p_click_bid == p_win * p_ctr."""
    model = make_escm2()
    batch = _make_batch(feature_dims, separable=False)
    out = model(batch["x"], training=False)
    assert jnp.allclose(out.p_click_bid, out.p_win * out.p_ctr, atol=1e-5)


# =============================================================================
# (d) ESCM2WC runs in BOTH loss_type 'dr' and 'ipw'
# =============================================================================


@pytest.mark.parametrize("loss_type", ["dr", "ipw"])
def test_escm2_loss_types_run(make_escm2, feature_dims, loss_type):
    """ESCM2WC train_step runs for both loss_type 'dr' and 'ipw' (finite loss)."""
    model = make_escm2(loss_type=loss_type)
    train_step = create_escm2wc_train_step(model.config)
    batch = _make_batch(feature_dims, separable=False)

    loss, components = train_step(model, nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param), batch)
    assert jnp.isfinite(loss)
    # Documented components present and finite.
    for field in ("total", "win", "ctr", "joint", "cfr", "impute"):
        val = getattr(components, field)
        assert jnp.all(jnp.isfinite(jnp.asarray(val))), f"{field} not finite"


# =============================================================================
# (c) Multi-step training reduces loss on a tiny separable task (smoke)
# =============================================================================


@pytest.mark.smoke
def test_esmm_training_decreases(make_esmm, feature_dims):
    """ESMM: ~5 train_steps yield finite, decreasing loss on a separable task."""
    model = make_esmm()
    train_step = create_esmm_wc_train_step(model.config)
    batch = _make_batch(feature_dims, separable=True, seed=1)

    losses = _run_training(model, train_step, batch, n_steps=5)
    assert all(jnp.isfinite(jnp.asarray(loss)) for loss in losses)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


@pytest.mark.smoke
@pytest.mark.parametrize("loss_type", ["dr", "ipw"])
def test_escm2_training_decreases(make_escm2, feature_dims, loss_type):
    """ESCM2WC (dr & ipw): ~5 train_steps yield finite, decreasing loss."""
    model = make_escm2(loss_type=loss_type)
    train_step = create_escm2wc_train_step(model.config)
    batch = _make_batch(feature_dims, separable=True, seed=2)

    losses = _run_training(model, train_step, batch, n_steps=5)
    assert all(jnp.isfinite(jnp.asarray(loss)) for loss in losses)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
