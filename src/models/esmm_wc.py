"""
ESMM-WC: Entire Space Multi-Task Model for Win→Click (2-tower)

Ablation baseline for ESCM²-WC. Applies ESMM constraint to Bid→Win→Click funnel
WITHOUT DR/IPW debiasing. Uses only the ESMM joint constraint for implicit debiasing.

Architecture:
- Win Tower: P(Win|X, bid) — trained on all 65M bids
- CTR Tower: P(Click|X, Win) — trained on won impressions only
- Joint: P(Click_bid) = P(Win) × P(Click|Win) — ESMM constraint (all bids)

Reference: Ma et al., "Entire Space Multi-Task Model" (SIGIR 2018)
Adapted for Bid→Win→Click funnel (instead of Impression→Click→Conversion).
"""

from typing import NamedTuple, Sequence, Optional, Dict, List
import jax
import jax.numpy as jnp
from flax import nnx

from .base import (
    MLP,
    EmbeddingLayer,
    FeatureInteraction,
    binary_cross_entropy,
)


# =============================================================================
# Configurations
# =============================================================================


class ESMMWCConfig(NamedTuple):
    """ESMM-WC Configuration"""
    feature_dims: Dict[str, int]  # Feature name -> vocab size
    embed_dim: int = 32
    hidden_dims: Sequence[int] = (128, 64)
    win_hidden_dims: Sequence[int] = (64, 32)
    dropout: float = 0.3
    # Loss weights
    win_weight: float = 1.0
    ctr_weight: float = 1.0
    joint_weight: float = 1.0
    # Architecture options
    use_fm_interaction: bool = True
    use_layer_norm: bool = False
    multi_hot_features: Optional[Sequence[str]] = None


class ESMMWCOutput(NamedTuple):
    """ESMM-WC Output"""
    p_win: jax.Array      # P(Win|X, bid)
    p_ctr: jax.Array      # P(Click|X, Win)
    p_click_bid: jax.Array  # P(Win) × P(Click|Win) — ESMM constraint


class ESMMWCLossComponents(NamedTuple):
    """ESMM-WC Loss Components"""
    total: float
    win: float    # Win BCE (all bids)
    ctr: float    # CTR BCE (won samples only)
    joint: float  # Joint BCE (all bids, ESMM constraint)


# =============================================================================
# Model
# =============================================================================


class ESMMWC(nnx.Module):
    """Entire Space Multi-Task Model for Win→Click (2-tower)

    ESMM constraint: P(Click_bid) = P(Win) × P(Click|Win)
    The joint BCE on all bids provides supervision signal for CTR tower
    even on non-won samples (implicit debiasing via ESMM constraint).

    No DR/IPW — serves as ablation baseline for ESCM²-WC.
    """

    def __init__(self, config: ESMMWCConfig, *, rngs: nnx.Rngs):
        self.config = config

        # Shared embedding
        multi_hot = list(config.multi_hot_features) if config.multi_hot_features else None
        self.embedding = EmbeddingLayer(
            feature_dims=config.feature_dims,
            embed_dim=config.embed_dim,
            multi_hot_features=multi_hot,
            rngs=rngs,
        )

        # Optional FM interaction
        self.use_fm_interaction = config.use_fm_interaction
        if config.use_fm_interaction:
            self.interaction = FeatureInteraction(
                n_features=len(config.feature_dims),
                embed_dim=config.embed_dim,
                rngs=rngs,
            )
            input_dim = len(config.feature_dims) * config.embed_dim + config.embed_dim
        else:
            self.interaction = None
            input_dim = len(config.feature_dims) * config.embed_dim

        # Win Tower: P(Win|X, bid)
        self.win_tower = MLP(
            hidden_dims=config.win_hidden_dims,
            output_dim=1,
            dropout=config.dropout,
            input_dim=input_dim,
            use_layer_norm=config.use_layer_norm,
            rngs=rngs,
        )

        # CTR Tower: P(Click|X, Win)
        self.ctr_tower = MLP(
            hidden_dims=config.hidden_dims,
            output_dim=1,
            dropout=config.dropout,
            input_dim=input_dim,
            use_layer_norm=config.use_layer_norm,
            rngs=rngs,
        )

    def __call__(
        self,
        x: Dict[str, jax.Array],
        training: bool = True,
    ) -> ESMMWCOutput:
        # Embed features
        embed = self.embedding(x)

        # Optional FM interaction
        if self.use_fm_interaction and self.interaction is not None:
            n_features = len(self.config.feature_dims)
            embed_reshaped = embed.reshape(
                *embed.shape[:-1], n_features, self.config.embed_dim
            )
            interaction_out = self.interaction(embed_reshaped)
            embed = jnp.concatenate([embed, interaction_out], axis=-1)

        # Win prediction
        win_logit = self.win_tower(embed, training=training)
        p_win = jax.nn.sigmoid(win_logit).squeeze(-1)

        # CTR prediction
        ctr_logit = self.ctr_tower(embed, training=training)
        p_ctr = jax.nn.sigmoid(ctr_logit).squeeze(-1)

        # ESMM constraint: P(Click_bid) = P(Win) × P(Click|Win)
        p_click_bid = p_win * p_ctr

        return ESMMWCOutput(
            p_win=p_win,
            p_ctr=p_ctr,
            p_click_bid=p_click_bid,
        )


# =============================================================================
# Loss and Training Functions
# =============================================================================


def create_esmm_wc_loss_fn(
    config: ESMMWCConfig,
    return_components: bool = False,
    jit_safe: bool = False,
):
    """Create loss function for ESMM-WC.

    Losses:
    1. Win BCE: BCE(p_win, win) on ALL bids
    2. CTR BCE: win × BCE(p_ctr, click) on WON samples only
    3. Joint BCE: BCE(p_click_bid, click) on ALL bids — ESMM constraint

    Args:
        config: ESMM-WC configuration.
        return_components: If True, return (total_loss, ESMMWCLossComponents).
        jit_safe: If True, skip float() cast in components (avoids
            ConcretizationTypeError inside JIT).
    """

    def loss_fn(model: ESMMWC, batch: dict):
        output = model(batch["x"], training=True)

        # Win loss (all bids)
        win_loss = jnp.mean(
            binary_cross_entropy(output.p_win, batch["win"])
        )

        # CTR loss (won samples only — masked)
        ctr_bce = binary_cross_entropy(output.p_ctr, batch["click"])
        n_won = jnp.clip(batch["win"].sum(), 1.0, None)
        ctr_loss = jnp.sum(batch["win"] * ctr_bce) / n_won

        # Joint loss (all bids — ESMM constraint)
        joint_loss = jnp.mean(
            binary_cross_entropy(output.p_click_bid, batch["click"])
        )

        # Total
        total_loss = (
            config.win_weight * win_loss
            + config.ctr_weight * ctr_loss
            + config.joint_weight * joint_loss
        )

        if return_components:
            _f = (lambda x: x) if jit_safe else float
            components = ESMMWCLossComponents(
                total=_f(total_loss),
                win=_f(win_loss),
                ctr=_f(ctr_loss),
                joint=_f(joint_loss),
            )
            return total_loss, components

        return total_loss

    return loss_fn


def create_esmm_wc_train_step(config: ESMMWCConfig):
    """Create JIT-compiled training step for ESMM-WC.

    Returns (loss, ESMMWCLossComponents) per step. Components use JAX arrays
    (jit_safe=True) — caller converts to Python float outside JIT.
    """
    loss_fn = create_esmm_wc_loss_fn(config, return_components=True, jit_safe=True)

    @nnx.jit
    def train_step(
        model: ESMMWC,
        optimizer: nnx.Optimizer,
        batch: dict,
    ):
        (loss, components), grads = nnx.value_and_grad(
            lambda m: loss_fn(m, batch), has_aux=True
        )(model)
        optimizer.update(model, grads)
        return loss, components

    return train_step


def create_esmm_wc_eval_step():
    """Create JIT-compiled evaluation step for ESMM-WC."""

    @nnx.jit
    def eval_step(model: ESMMWC, batch: dict) -> ESMMWCOutput:
        return model(batch["x"], training=False)

    return eval_step
